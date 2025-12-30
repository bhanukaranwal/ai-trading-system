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
bash
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
bash
python scripts/download_data.py
Train Models
bash
python scripts/train_models.py
Run Backtest
bash
python scripts/run_backtest.py
Launch Dashboard
bash
streamlit run src/dashboard.py
Live Trading
bash
python scripts/run_live.py
Architecture
Data Layer: Polygon.io primary, Yahoo Finance fallback, Parquet storage

Feature Engineering: Technical indicators, volatility, momentum, macro proxies

AI Signals: Ensemble of XGBoost, LSTM, Transformer models

Portfolio: Multi-asset optimizer with rebalancing logic

Execution: Async order routing to IBKR/Alpaca

Monitoring: Real-time P&L, risk metrics, position tracking

Repository Structure

ai_trading_system/
├── data/              # Historical and live data storage
├── src/               # Core trading system modules
├── scripts/           # Executable workflows
├── models/            # Saved ML model weights
└── config.yaml        # System configuration
Risk Disclaimer
This software is for educational and research purposes. Live trading involves substantial risk of loss. Test thoroughly in paper trading before deploying capital.
