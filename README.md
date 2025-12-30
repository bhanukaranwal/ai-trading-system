# AI-Driven Multi-Asset Quantitative Trading System

Production-grade algorithmic trading platform leveraging vectorized backtesting, deep learning, and advanced portfolio optimization for equities, bonds, and commodities.

## Features

- **Ultra-Fast Backtesting**: Vectorbt-powered vectorized simulations across multiple assets
- **Advanced AI Models**: XGBoost, LSTM, Transformer architectures with PyTorch
- **Multi-Asset Support**: Equities, Treasury ETFs, Commodity Futures
- **Portfolio Optimization**: Mean-variance, risk parity, max Sharpe via PyPortfolioOpt
- **Risk Management**: VaR, CVaR, Kelly criterion, volatility targeting, drawdown controls
- **Live Execution**: Interactive Brokers and Alpaca integration
- **Real-Time Dashboard**: Streamlit-based monitoring and visualization

## Installation

```bash
git clone https://github.com/yourusername/ai_trading_system.git
cd ai_trading_system
pip install -r requirements.txt
Configuration
Copy config.yaml and add your API credentials

Set environment variables or edit config directly:

POLYGON_API_KEY

ALPACA_API_KEY, ALPACA_SECRET_KEY

IBKR_PORT, IBKR_CLIENT_ID

Quick Start
Download Data
