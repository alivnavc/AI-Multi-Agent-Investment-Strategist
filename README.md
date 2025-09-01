# AI Investment Strategist

A comprehensive AI-powered investment analysis tool that provides real-time stock recommendations, risk assessment, and market insights.

## Overview

The AI Investment Strategist is a sophisticated financial analysis platform that combines:
- **Real-time Market Data**: Live stock prices, volatility metrics, and performance indicators
- **AI-Powered Analysis**: Gemini 2.0 Flash for intelligent investment recommendations
- **Risk Assessment**: Comprehensive risk scoring and portfolio analysis
- **Sector Intelligence**: Dynamic S&P 500 constituent analysis by sector
- **Sentiment Analysis**: News sentiment scoring using TextBlob NLP
- **Interactive Visualizations**: Plotly charts for performance tracking

## Features

### Core Functionality
- **Dynamic Stock Fetching**: Real-time S&P 500 constituent data from Wikipedia and DataHub
- **Risk Scoring Algorithm**: Multi-factor risk assessment (volatility, returns, beta)
- **Sector Analysis**: Automatic sector detection and specialized analysis
- **Fundamental Analysis**: P/E ratios, market cap, debt-to-equity, and more
- **Performance Tracking**: 6-month historical data with interactive charts
- **News Sentiment**: Real-time sentiment analysis of market news

### AI Capabilities
- **Natural Language Queries**: Understand investment goals in plain English
- **Intelligent Sector Detection**: Auto-detect sectors from user queries
- **Risk Preference Matching**: High, medium, and low-risk investment strategies
- **Comprehensive Reporting**: Decision-ready investment analysis reports
- **Portfolio Recommendations**: Top 5 stock picks with detailed rationale

### Technical Features
- **Streamlit Interface**: Modern, responsive web application
- **Real-time Data**: Live market data via yfinance API
- **Error Handling**: Robust error handling with comprehensive logging
- **Cross-platform**: Works on Windows, macOS, and Linux

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Google API key for Gemini AI

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/alivnavc/AI-Multi-Agent-Investment-Strategist.git
   cd AI-Multi-Agent-Investment-Strategist
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Create .env file
   echo "GOOGLE_API_KEY=your_google_api_key_here" > .env
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

## Usage

### Basic Investment Analysis

1. **Describe Your Goals**: Enter your investment objectives in natural language
   - Example: "I want high-risk mid-cap stocks in technology sector"
   - Example: "Show me conservative healthcare investments"

2. **Select Preferences**: Choose sector and risk level
   - Auto-detect from query (recommended)
   - Manual sector selection
   - Risk preference: High, Medium, or Low

3. **Generate Report**: Click "Generate Investment Report" for analysis

4. **Review Results**: Comprehensive analysis including:
   - Top 5 stock recommendations
   - Risk metrics and fundamentals
   - Performance charts
   - Risk assessment and guidance

### Advanced Features

- **Sector Intelligence**: Automatic sector detection from natural language
- **Risk Profiling**: Multi-factor risk scoring algorithm
- **Performance Tracking**: Historical data analysis and visualization
- **Sentiment Analysis**: Market news sentiment scoring
- **Fundamental Metrics**: Comprehensive financial ratio analysis

## API Keys Required

### Google Gemini AI
- **Purpose**: AI-powered investment analysis and recommendations
- **Setup**: Get API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
- **Usage**: Set as environment variable `GOOGLE_API_KEY`


## Project Structure

```
ai-investment-strategist/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                      # This file
├── LICENSE                        # License information
├── .env                          # Environment variables (create this)
├── .gitignore                    # Git ignore file
└── logs/                         # Application logs
    └── investment_strategist_app1.log
```

### Development Guidelines

- Follow PEP 8 Python style guidelines
- Add comprehensive error handling
- Include logging for debugging
- Write clear documentation
- Test with different market conditions



## Roadmap

### Planned Features
- **Real-time Portfolio Tracking**: Live portfolio monitoring
- **Advanced Risk Models**: Machine learning risk prediction
- **Multi-asset Support**: Bonds, ETFs, and alternative investments
- **Backtesting Engine**: Historical strategy performance testing
- **API Endpoints**: RESTful API for integration
- **Mobile Application**: Cross-platform mobile app

### Performance Improvements
- **Caching Layer**: Redis integration for data caching
- **Async Processing**: Improved concurrency for data fetching
- **Database Integration**: Persistent storage for analysis results

## Security

### Data Privacy
- No personal financial data is stored
- All analysis is performed locally
- API keys are stored securely in environment variables

### Market Data
- Data sourced from public APIs (Yahoo Finance, Wikipedia)
- No proprietary market data used
- Real-time data accuracy depends on source APIs


## Disclaimer

**Important**: This software is for educational and informational purposes only. It does not constitute investment advice, financial advice, or any other type of advice. 

- **Not Financial Advice**: Always consult qualified financial advisors
- **Market Risk**: All investments carry risk of loss
- **Data Accuracy**: Market data accuracy depends on source APIs
- **Past Performance**: Historical performance does not guarantee future results
- **Due Diligence**: Always perform your own research before investing


## Acknowledgments

- **Yahoo Finance**: Market data and financial information
- **Wikipedia**: S&P 500 constituent data
- **DataHub**: Alternative data sources
- **Google AI**: Gemini AI capabilities
- **Open Source Community**: Libraries and frameworks used

---

**Built with passion for financial technology and AI innovation** 
