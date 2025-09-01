import os
import yfinance as yf
import streamlit as st
from agno.agent import Agent
from agno.models.google import Gemini
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from textblob import TextBlob
import requests
from datetime import datetime, timedelta
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('investment_strategist_app1.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set environment variable for Google API
os.environ["GOOGLE_API_KEY"] = "your_actual_api_key_here"  # Set this in .env file

# Enhanced stock data fetching with risk assessment
def fetch_sector_stocks(sector_name):
    """Fetch dynamically the top stocks for a sector (S&P 500 constituents) with risk assessment."""
    logger.info(f"fetch_sector_stocks called for sector={sector_name}")
    
    # Map user-friendly sector aliases to GICS sectors used on Wikipedia
    sector_aliases = {
        "technology": ["Information Technology"],
        "healthcare": ["Health Care"],
        "financials": ["Financials"],
        "energy": ["Energy"],
        "consumer": ["Consumer Discretionary", "Consumer Staples"],
        "industrials": ["Industrials"],
        "materials": ["Materials"],
        "utilities": ["Utilities"],
        "real estate": ["Real Estate"],
        "communication": ["Communication Services"],
    }
    
    sector_lower = sector_name.lower().strip()
    if sector_lower not in sector_aliases:
        logger.error(f"Unsupported sector requested: {sector_name}")
        return {"error": f"Sector {sector_name} not supported"}
    
    symbols = []
    
    # Attempt 1: Wikipedia with requests + User-Agent, parsed by pandas
    try:
        logger.info("Attempting to fetch S&P 500 constituents from Wikipedia via requests...")
        resp = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"},
            timeout=15
        )
        resp.raise_for_status()
        # Try multiple parsers
        tables = None
        for flavor in ["lxml", "html5lib"]:
            try:
                logger.info(f"Parsing HTML with flavor={flavor}")
                tables = pd.read_html(resp.text, flavor=flavor)
                break
            except Exception as e:
                logger.warning(f"pandas.read_html with flavor={flavor} failed: {str(e)}")
        if tables is None:
            raise RuntimeError("Failed to parse Wikipedia HTML with available parsers")
        sp500 = None
        for df in tables:
            cols_lower = [str(c).lower().strip() for c in df.columns]
            if "symbol" in cols_lower and "gics sector" in cols_lower:
                sp500 = df
                break
        if sp500 is None:
            raise RuntimeError("Could not locate S&P 500 table on Wikipedia page")
        sp500.columns = [c.strip() for c in sp500.columns]
        target_gics = sector_aliases[sector_lower]
        filtered = sp500[sp500["GICS Sector"].isin(target_gics)]
        symbols = filtered["Symbol"].dropna().astype(str).str.replace(".", "-", regex=False).unique().tolist()
        logger.info(f"Fetched {len(symbols)} symbols from Wikipedia for sector={sector_name}")
    except Exception as e:
        logger.error(f"Wikipedia fetch failed: {str(e)}")
        logger.error(traceback.format_exc())
        
    # Attempt 2 (fallback): Public CSV of S&P 500 constituents (DataHub)
    if not symbols:
        try:
            logger.info("Falling back to DataHub S&P 500 constituents CSV...")
            csv_url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
            df = pd.read_csv(csv_url)
            if not set(["Symbol", "Sector"]).issubset(df.columns):
                raise RuntimeError("Unexpected CSV schema from DataHub")
            # Map Sectors to GICS-like categories where possible
            # DataHub uses 'Sector' similar to GICS sector names
            target_gics = sector_aliases[sector_lower]
            filtered = df[df["Sector"].isin(target_gics)]
            symbols = filtered["Symbol"].dropna().astype(str).str.replace(".", "-", regex=False).unique().tolist()
            logger.info(f"Fetched {len(symbols)} symbols from DataHub for sector={sector_name}")
        except Exception as e:
            logger.error(f"DataHub CSV fallback failed: {str(e)}")
            logger.error(traceback.format_exc())
    
    if not symbols:
        return {"error": "Failed to fetch sector constituents after multiple attempts (Wikipedia, DataHub). See logs."}
    
    # Limit for performance; you can adjust this higher if you want
    symbols = symbols[:30]
    
    stock_data = {}
    for symbol in symbols:
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="6mo")
            if hist.empty:
                logger.warning(f"No price history for {symbol}, skipping")
                continue
            
            returns = hist['Close'].pct_change().dropna()
            if returns.empty:
                logger.warning(f"No returns sequence for {symbol}, skipping")
                continue
            volatility = returns.std() * np.sqrt(252)
            total_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100
            sharpe_ratio = (returns.mean() * 252) / volatility if volatility > 0 else 0
            
            info = stock.info
            stock_data[symbol] = {
                "name": info.get("longName", symbol),
                "sector": info.get("sector", sector_name),
                "market_cap": info.get("marketCap", 0),
                "current_price": float(hist['Close'].iloc[-1]),
                "total_return_6mo": float(total_return),
                "volatility": float(volatility * 100),
                "sharpe_ratio": float(sharpe_ratio),
                "pe_ratio": info.get("trailingPE", 0) or 0,
                "beta": info.get("beta", 1.0) or 1.0,
                "risk_score": float(calculate_risk_score(volatility, total_return, info.get("beta", 1.0) or 1.0)),
                "summary": (info.get("longBusinessSummary", "N/A") or "N/A")[:200] + "..."
            }
        except Exception as e:
            logger.error(f"Error processing symbol {symbol}: {str(e)}")
            logger.error(traceback.format_exc())
            continue
    
    logger.info(f"Prepared metrics for {len(stock_data)} symbols in sector={sector_name}")
    return stock_data

def calculate_risk_score(volatility, return_pct, beta):
    """Calculate risk score (higher = more risky)"""
    # Normalize values and create composite risk score
    vol_score = min(volatility * 100, 100)  # Cap at 100
    return_score = abs(return_pct)  # Absolute return (both positive and negative are risky)
    beta_score = min(abs(beta - 1) * 50, 100)  # Beta deviation from 1
    
    # Weighted risk score (volatility 40%, return volatility 40%, beta 20%)
    risk_score = (vol_score * 0.4) + (return_score * 0.4) + (beta_score * 0.2)
    return min(risk_score, 100)

def get_market_sentiment(symbol):
    """Analyze market sentiment for a stock"""
    try:
        stock = yf.Ticker(symbol)
        news = stock.news[:10]  # Get latest 10 news articles
        
        if not news:
            return {"sentiment": "neutral", "score": 0, "news_count": 0}
        
        # Analyze sentiment of news titles
        sentiments = []
        for article in news:
            title = article.get("title", "")
            blob = TextBlob(title)
            sentiments.append(blob.sentiment.polarity)
        
        avg_sentiment = np.mean(sentiments)
        sentiment_score = (avg_sentiment + 1) * 50  # Convert to 0-100 scale
        
        # Categorize sentiment
        if avg_sentiment > 0.1:
            sentiment = "positive"
        elif avg_sentiment < -0.1:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return {
            "sentiment": sentiment,
            "score": sentiment_score,
            "news_count": len(news),
            "recent_news": news[:3]  # Top 3 recent news
        }
        
    except Exception as e:
        return {"sentiment": "neutral", "score": 0, "news_count": 0, "error": str(e)}

def get_fundamental_analysis(symbol):
    """Get comprehensive fundamental analysis"""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        
        # Get financial ratios
        fundamentals = {
            "symbol": symbol,
            "name": info.get("longName", symbol),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", 0),
            "enterprise_value": info.get("enterpriseValue", 0),
            "pe_ratio": info.get("trailingPE", 0),
            "forward_pe": info.get("forwardPE", 0),
            "peg_ratio": info.get("pegRatio", 0),
            "price_to_book": info.get("priceToBook", 0),
            "price_to_sales": info.get("priceToSalesTrailing12Months", 0),
            "debt_to_equity": info.get("debtToEquity", 0),
            "current_ratio": info.get("currentRatio", 0),
            "quick_ratio": info.get("quickRatio", 0),
            "return_on_equity": info.get("returnOnEquity", 0),
            "return_on_assets": info.get("returnOnAssets", 0),
            "profit_margins": info.get("profitMargins", 0),
            "operating_margins": info.get("operatingMargins", 0),
            "ebitda_margins": info.get("ebitdaMargins", 0),
            "revenue_growth": info.get("revenueGrowth", 0),
            "earnings_growth": info.get("earningsGrowth", 0),
            "dividend_yield": info.get("dividendYield", 0) * 100 if info.get("dividendYield") else 0,
            "beta": info.get("beta", 1.0),
            "52_week_high": info.get("fiftyTwoWeekHigh", 0),
            "52_week_low": info.get("fiftyTwoWeekLow", 0),
            "summary": info.get("longBusinessSummary", "N/A")
        }
        
        return fundamentals
        
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}

# Enhanced AI Agents
sector_analyzer = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    description="Analyzes sectors and identifies high-risk, high-return opportunities",
    instructions=[
        "Analyze sector performance and identify high-risk, high-return stocks",
        "Consider volatility, beta, and growth potential",
        "Rank stocks by risk-reward ratio"
    ],
    show_tool_calls=True,
    markdown=True
)

risk_assessor = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    description="Assesses risk profiles and provides risk analysis",
    instructions=[
        "Analyze stock risk metrics including volatility, beta, and fundamentals",
        "Provide risk assessment and recommendations",
        "Consider market sentiment and news impact"
    ],
    markdown=True
)

investment_strategist = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    description="Provides comprehensive investment recommendations",
    instructions=[
        "Analyze all data including fundamentals, sentiment, and risk metrics",
        "Provide top 5 stock recommendations with detailed analysis",
        "Include risk warnings and investment strategy"
    ],
    markdown=True
)

def analyze_sector_opportunities(sector_name, risk_preference="high"):
    """Analyze sector for investment opportunities"""
    
    # Fetch sector data
    sector_data = fetch_sector_stocks(sector_name)
    
    if "error" in sector_data:
        return f"Error: {sector_data['error']}"
    
    # Filter by risk preference
    if risk_preference == "high":
        # Sort by risk score (descending) and return (descending)
        sorted_stocks = sorted(sector_data.items(), 
                              key=lambda x: (x[1]['risk_score'], x[1]['total_return_6mo']), 
                              reverse=True)
    elif risk_preference == "medium":
        # Sort by balanced risk-reward
        sorted_stocks = sorted(sector_data.items(), 
                              key=lambda x: x[1]['sharpe_ratio'], 
                              reverse=True)
    else:  # low risk
        # Sort by low volatility and stable returns
        sorted_stocks = sorted(
            sector_data.items(),
            key=lambda x: (x[1]['volatility'], -abs(x[1]['total_return_6mo']))
        )
    
    # Get top 5 stocks
    top_stocks = sorted_stocks[:5]
    
    # Enhance with sentiment and fundamentals
    enhanced_data = {}
    for symbol, data in top_stocks:
        sentiment_data = get_market_sentiment(symbol)
        fundamental_data = get_fundamental_analysis(symbol)
        
        enhanced_data[symbol] = {
            **data,
            "sentiment": sentiment_data,
            "fundamentals": fundamental_data
        }
    
    return enhanced_data

def _fmt_pct(value: float, decimals: int = 1) -> str:
    try:
        return f"{float(value):.{decimals}f}%"
    except Exception:
        return "N/A"


def _fmt_num(value: float, decimals: int = 2) -> str:
    try:
        return f"{float(value):.{decimals}f}"
    except Exception:
        return "N/A"


def _fmt_cap(market_cap: float) -> str:
    try:
        cap = float(market_cap)
        if cap >= 1e12:
            return f"${cap/1e12:.2f}T"
        if cap >= 1e9:
            return f"${cap/1e9:.2f}B"
        if cap >= 1e6:
            return f"${cap/1e6:.2f}M"
        return f"${cap:,.0f}"
    except Exception:
        return "N/A"


def _inference_timeline(risk: str) -> str:
    r = (risk or "").lower()
    if r == "high":
        return "6–18 months"
    if r == "medium":
        return "12–24 months"
    return "24–60 months"


def _rationale_points(d: dict) -> list:
    pts = []
    try:
        if d.get("sharpe_ratio", 0) > 1:
            pts.append("Attractive risk-adjusted performance (Sharpe > 1)")
        if d.get("total_return_6mo", 0) > 0:
            pts.append("Positive 6-month momentum")
        if d.get("volatility", 0) > 60:
            pts.append("High volatility — suitable for aggressive risk tolerance")
        if d.get("pe_ratio", 0) and d.get("pe_ratio", 0) < 15:
            pts.append("Reasonable valuation (low P/E) vs sector peers")
        if d.get("beta", 1.0) > 1.3:
            pts.append("High beta — amplified upside/downside vs market")
        sent = (d.get("sentiment", {}) or {}).get("sentiment", "").lower()
        if sent == "positive":
            pts.append("Favorable recent news & sentiment")
        elif sent == "negative":
            pts.append("Monitor negative news flow closely")
    except Exception:
        pass
    if not pts:
        pts.append("Balanced risk/reward profile based on current metrics")
    return pts


def build_decision_report(sector_name: str, risk_preference: str, enhanced_data: dict) -> str:
    """Create a decision-ready markdown report from computed metrics and signals."""
    sector_title = sector_name.strip().title()
    risk_title = (risk_preference or "").strip().title() or "High"

    items = list(enhanced_data.items())
    # Default ordering: higher risk_score then higher 6m return
    try:
        items = sorted(
            items,
            key=lambda kv: (kv[1].get("risk_score", 0), kv[1].get("total_return_6mo", 0)),
            reverse=True,
        )
    except Exception:
        pass

    # Build comparison table (Top 5)
    header = (
        "| Rank | Symbol | Company | Price | 6M Return | Volatility | Sharpe | Beta | P/E | Risk Score | Sentiment |\n"
        "|---:|:---:|:--|---:|---:|---:|---:|---:|---:|---:|:--|\n"
    )
    rows = []
    for idx, (sym, d) in enumerate(items[:5], start=1):
        rows.append(
            "| {rank} | {sym} | {name} | ${price} | {ret} | {vol} | {sharpe} | {beta} | {pe} | {risk} | {sent} |".format(
                rank=idx,
                sym=sym,
                name=(d.get("name") or sym).replace("|", "/"),
                price=_fmt_num(d.get("current_price", 0)),
                ret=_fmt_pct(d.get("total_return_6mo", 0)),
                vol=_fmt_pct(d.get("volatility", 0)),
                sharpe=_fmt_num(d.get("sharpe_ratio", 0), 2),
                beta=_fmt_num(d.get("beta", 1.0), 2),
                pe=_fmt_num(d.get("pe_ratio", 0), 1),
                risk=_fmt_num(d.get("risk_score", 0), 1),
                sent=(d.get("sentiment", {}) or {}).get("sentiment", "N/A").title(),
            )
        )
    comparison_table = header + "\n".join(rows) if rows else "_No qualifying stocks found._"

    # Build At‑a‑Glance list
    at_glance_lines = []
    for idx, (sym, d) in enumerate(items[:5], start=1):
        at_glance_lines.append(
            "- **{idx}. {sym}** — ${price} • {ret} 6M • {vol} vol • Sharpe {sharpe} • Risk {risk} • {sent} sentiment".format(
                idx=idx,
                sym=sym,
                price=_fmt_num(d.get("current_price", 0)),
                ret=_fmt_pct(d.get("total_return_6mo", 0)),
                vol=_fmt_pct(d.get("volatility", 0)),
                sharpe=_fmt_num(d.get("sharpe_ratio", 0), 2),
                risk=_fmt_num(d.get("risk_score", 0), 1),
                sent=(d.get("sentiment", {}) or {}).get("sentiment", "N/A").title(),
            )
        )
    at_glance_block = "\n".join(at_glance_lines) if at_glance_lines else "_No qualifying stocks found._"

    # Per-stock sections
    per_stock_sections = []
    for idx, (sym, d) in enumerate(items[:5], start=1):
        summary_text = (d.get("summary") or "").strip()
        if len(summary_text) > 600:
            summary_text = summary_text[:600].rsplit(" ", 1)[0] + " …"
        rationales = _rationale_points(d)
        sent = d.get("sentiment", {}) or {}

        per_stock_sections.append(
            (
                f"**{idx}. {d.get('name', sym)} ({sym})**\n\n"
                f"- **Company Overview**: {summary_text if summary_text else 'N/A'}\n"
                f"- **Risk Metrics**: Volatility { _fmt_pct(d.get('volatility', 0)) }, "
                f"Sharpe { _fmt_num(d.get('sharpe_ratio', 0), 2) }, "
                f"Beta { _fmt_num(d.get('beta', 1.0), 2) }, "
                f"Risk Score { _fmt_num(d.get('risk_score', 0), 1) }\n"
                f"- **Fundamentals**: Price ${ _fmt_num(d.get('current_price', 0)) }, "
                f"P/E { _fmt_num(d.get('pe_ratio', 0), 1) }, Market Cap { _fmt_cap(d.get('market_cap', 0)) }\n"
                f"- **Performance**: 6-Month Return { _fmt_pct(d.get('total_return_6mo', 0)) }\n"
                f"- **Sentiment**: { (sent.get('sentiment') or 'N/A').title() } "
                f"(Score { _fmt_num(sent.get('score', 0), 0) }, News { sent.get('news_count', 0) })\n"
                f"- **Investment Timeline**: {_inference_timeline(risk_preference)}\n"
                f"- **Rationale**: " + "; ".join(rationales) + "\n"
            )
        )

    per_stock_block = "\n\n".join(per_stock_sections) if per_stock_sections else ""

    report = f"""
## {risk_title}-Risk {sector_title} Investment Opportunities: A Decision-Ready Analysis

**Executive Summary**

This report presents {risk_title.lower()}-risk opportunities in the {sector_title} sector. It consolidates fundamentals, risk metrics, recent performance, and market sentiment to surface the top candidates. Diversification and strict risk controls are recommended given the risk profile.

### 1. Sector Overview and Market Conditions

- Regulatory environment and policy shifts can materially impact pricing and margins
- Innovation cycles (R&D pipelines, product launches) drive dispersion across names
- Macro sensitivity via rates, inflation, and growth impacts multiples and flows
- Position sizing and stop-loss discipline are critical due to elevated volatility

### 2. Top 5 Comparison (Decision Table)

{comparison_table}

### 3. At‑a‑Glance (Top 5)

{at_glance_block}

### 4. Detailed Recommendations

{per_stock_block}

### 5. Risk Guidance

- Expect larger drawdowns; use staged entries and predefined exits
- Avoid concentration: limit single-position exposure (e.g., 5–10% max)
- Reassess thesis on material news (earnings, guidance, regulatory updates)

_Disclaimer: Educational information, not investment advice._
"""
    return report.strip()

def generate_investment_report(sector_name, risk_preference, enhanced_data):
    """Generate comprehensive investment report"""
    try:
        return build_decision_report(sector_name, risk_preference, enhanced_data)
    except Exception as e:
        logger.error(f"Error building decision report: {str(e)}")
        logger.error(traceback.format_exc())
        # Fallback to LLM if formatting fails
        analysis_prompt = f"""
        Analyze the following {risk_preference}-risk investment opportunities in the {sector_name} sector:
        {enhanced_data}
        Provide a concise, decision-ready summary.
        """
        report = investment_strategist.run(analysis_prompt)
        return report.content

# Streamlit Interface
st.set_page_config(page_title="AI Investment Strategist", page_icon="📈", layout="wide")

# Title and header
st.markdown("""
    <h1 style="text-align: center; color: #4CAF50;">📈 AI Stocks Portfolio Strategist</h1>
    <h3 style="text-align: center; color: #6c757d;">Discover high-risk, high-return opportunities with AI-powered analysis</h3>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("""
    <h2 style="color: #343a40;">Investment Preferences</h2>
    <p style="color: #6c757d;">Tell us what you're looking for and we'll find the best opportunities.</p>
""", unsafe_allow_html=True)

# User input
investment_query = st.sidebar.text_area(
    "Describe Your Investment Goals", 
    placeholder="e.g., I want high-risk mid-cap stocks in technology sector, or high-risk stocks with high returns in healthcare"
)

sector_selection = st.sidebar.selectbox(
    "Select Sector (or let AI detect from your query)",
    ["Auto-detect from query", "Technology", "Healthcare", "Financials", "Energy", "Consumer", 
     "Industrials", "Materials", "Utilities", "Real Estate", "Communication"]
)

risk_preference = st.sidebar.selectbox(
    "Risk Preference",
    ["high", "medium", "low"]
)

api_key = st.sidebar.text_input("Enter your API Key (optional)", type="password")

# Generate Report Button
if st.sidebar.button("🚀 Generate Investment Report"):
    if not investment_query:
        st.sidebar.warning("Please describe your investment goals.")
    else:
        with st.spinner("🔍 Analyzing market opportunities..."):
            
            # Auto-detect sector from query if not manually selected
            if sector_selection == "Auto-detect from query":
                query_lower = investment_query.lower()
                if any(word in query_lower for word in ["tech", "technology", "software", "ai", "semiconductor"]):
                    detected_sector = "Technology"
                elif any(word in query_lower for word in ["health", "healthcare", "medical", "pharma", "biotech"]):
                    detected_sector = "Healthcare"
                elif any(word in query_lower for word in ["finance", "banking", "financial", "fintech"]):
                    detected_sector = "Financials"
                elif any(word in query_lower for word in ["energy", "oil", "gas", "renewable"]):
                    detected_sector = "Energy"
                elif any(word in query_lower for word in ["consumer", "retail", "ecommerce"]):
                    detected_sector = "Consumer"
                else:
                    detected_sector = "Technology"  # Default
            else:
                detected_sector = sector_selection
            
            st.info(f"🎯 Analyzing {detected_sector} sector for {risk_preference}-risk opportunities...")
            
            # Analyze sector opportunities
            enhanced_data = analyze_sector_opportunities(detected_sector, risk_preference)
            
            if isinstance(enhanced_data, str) and enhanced_data.startswith("Error"):
                st.error(enhanced_data)
            else:
                # Generate comprehensive report
                report = generate_investment_report(detected_sector, risk_preference, enhanced_data)
                
                # Display report
                st.subheader("📊 Investment Analysis Report")
                st.markdown(report)

                # Performance Chart
                st.subheader("📊 Performance (Top 5, 6 Months)")
                
                # Select top symbols for chart
                top_symbols = list(enhanced_data.keys())[:5]
                
                # Performance Chart
                
                try:
                    stock_data = yf.download(top_symbols, period="6mo")["Close"]
                    fig = go.Figure()
                    for symbol in top_symbols:
                        if symbol in stock_data.columns:
                            fig.add_trace(go.Scatter(
                                x=stock_data.index,
                                y=stock_data[symbol],
                                mode='lines',
                                name=symbol
                            ))
                    fig.update_layout(
                        title=f"Top 5 {detected_sector} Stocks Performance (6-Month)",
                        xaxis_title="Date",
                        yaxis_title="Price (USD)",
                        template="plotly_dark",
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not generate performance chart: {str(e)}")
                
                # Risk Analysis
                st.subheader("⚠️ Risk Analysis")
                
                risk_analysis = risk_assessor.run(f"""
                Analyze the risk profile of these stocks:
                {enhanced_data}
                
                Provide:
                1. Overall portfolio risk assessment
                2. Individual stock risk factors
                3. Risk mitigation strategies
                4. Recommended position sizing
                """)
                
                st.markdown(risk_analysis.content)
                
                st.success("✅ Analysis complete! Review the report and risk analysis before making investment decisions.")

# Add some helpful information
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 💡 How it works:
1. **Describe your goals** - Tell us what you're looking for
2. **AI analyzes** - We scan the market for opportunities
3. **Comprehensive report** - Get detailed analysis and recommendations
4. **Risk assessment** - Understand the risks involved
5. **Top 5 picks** - See the best opportunities with comparisons

### 🎯 Example queries:
- "I want high-risk mid-cap stocks in technology"
- "Show me high-risk stocks with high returns in healthcare"
- "Find me aggressive growth stocks in financial sector"
""")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d;">
    <p>⚠️ <strong>Disclaimer:</strong> This is for educational purposes only. Always do your own research and consult financial advisors before investing.</p>
</div>
""", unsafe_allow_html=True)
