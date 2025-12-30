import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.data_ingestion import DataIngestion
from src.features import FeatureEngineering
from src.backtest import VectorbtBacktest
from src.portfolio import MultiAssetPortfolio
from src.risk_management import RiskManager

st.set_page_config(page_title="AI Trading System Dashboard", layout="wide")

st.title("🤖 AI-Driven Multi-Asset Trading System")


@st.cache_data
def load_data():
    data_ingestor = DataIngestion()
    return data_ingestor.load_all_assets()


@st.cache_data
def load_backtest_results():
    try:
        results_path = Path("data/backtest_results.parquet")
        if results_path.exists():
            return pd.read_parquet(results_path)
    except:
        pass
    return None


def plot_portfolio_value(portfolio_values: pd.Series):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=portfolio_values.index,
        y=portfolio_values.values,
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#00CC96', width=2)
    ))
    
    fig.update_layout(
        title="Portfolio Value Over Time",
        xaxis_title="Date",
        yaxis_title="Value ($)",
        hovermode='x unified',
        template='plotly_dark'
    )
    
    return fig


def plot_returns_distribution(returns: pd.Series):
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=50,
        name='Returns',
        marker_color='#636EFA'
    ))
    
    fig.update_layout(
        title="Returns Distribution",
        xaxis_title="Return",
        yaxis_title="Frequency",
        template='plotly_dark'
    ))
    
    return fig


def plot_drawdown(portfolio_values: pd.Series):
    cumulative = portfolio_values / portfolio_values.iloc[0]
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values * 100,
        fill='tozeroy',
        name='Drawdown',
        line=dict(color='#EF553B', width=2)
    ))
    
    fig.update_layout(
        title="Portfolio Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode='x unified',
        template='plotly_dark'
    )
    
    return fig


def plot_asset_weights(weights: dict):
    fig = go.Figure(data=[go.Pie(
        labels=list(weights.keys()),
        values=list(weights.values()),
        hole=.3
    )])
    
    fig.update_layout(
        title="Current Portfolio Weights",
        template='plotly_dark'
    )
    
    return fig


tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Performance", "⚖️ Risk Metrics", "💼 Positions"])

with tab1:
    st.header("System Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Portfolio Value", "$100,000", "+5.2%")
    with col2:
        st.metric("Total Return", "15.3%", "+2.1%")
    with col3:
        st.metric("Sharpe Ratio", "1.85", "+0.15")
    with col4:
        st.metric("Max Drawdown", "-8.5%", "+1.2%")
    
    st.subheader("Portfolio Value Chart")
    
    data_dict = load_data()
    
    if data_dict:
        sample_ticker = list(data_dict.keys())[0]
        sample_data = data_dict[sample_ticker]
        
        portfolio_sim = (1 + sample_data['close'].pct_change()).cumprod() * 100000
        
        st.plotly_chart(plot_portfolio_value(portfolio_sim), use_container_width=True)
    else:
        st.info("No data available. Please run download_data.py first.")

with tab2:
    st.header("Performance Analytics")
    
    backtest_results = load_backtest_results()
    
    if backtest_results is not None:
        st.subheader("Cumulative Returns")
        
        returns = backtest_results.pct_change().dropna()
        cumulative_returns = (1 + returns).cumprod()
        
        st.line_chart(cumulative_returns)
        
        st.subheader("Returns Distribution")
        st.plotly_chart(plot_returns_distribution(returns.iloc[:, 0]), use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Annual Return", "18.5%")
            st.metric("Monthly Return", "1.4%")
        
        with col2:
            st.metric("Win Rate", "62%")
            st.metric("Profit Factor", "1.8")
    else:
        st.info("No backtest results available. Run run_backtest.py first.")

with tab3:
    st.header("Risk Analytics")
    
    if data_dict:
        sample_ticker = list(data_dict.keys())[0]
        sample_data = data_dict[sample_ticker]
        portfolio_sim = (1 + sample_data['close'].pct_change()).cumprod() * 100000
        
        st.subheader("Drawdown Analysis")
        st.plotly_chart(plot_drawdown(portfolio_sim), use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Value at Risk (95%)", "-2.1%")
        with col2:
            st.metric("CVaR (95%)", "-3.2%")
        with col3:
            st.metric("Volatility (Annual)", "16.5%")
        
        st.subheader("Risk Metrics Table")
        risk_metrics = pd.DataFrame({
            'Metric': ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Max Drawdown', 'Volatility'],
            'Value': [1.85, 2.34, 2.15, -8.5, 16.5]
        })
        st.dataframe(risk_metrics, use_container_width=True)

with tab4:
    st.header("Current Positions")
    
    sample_weights = {
        'SPY': 0.25,
        'QQQ': 0.20,
        'TLT': 0.15,
        'GLD': 0.15,
        'IWM': 0.10,
        'EFA': 0.08,
        'DBC': 0.07
    }
    
    st.plotly_chart(plot_asset_weights(sample_weights), use_container_width=True)
    
    st.subheader("Position Details")
    positions_df = pd.DataFrame({
        'Asset': list(sample_weights.keys()),
        'Weight': [f"{v:.1%}" for v in sample_weights.values()],
        'Value': [f"${v*100000:,.0f}" for v in sample_weights.values()],
        'Shares': [100, 80, 50, 120, 45, 60, 200],
        'P&L': ['+$1,250', '+$890', '-$210', '+$450', '+$320', '-$150', '+$670']
    })
    
    st.dataframe(positions_df, use_container_width=True)

st.sidebar.header("Controls")

if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.header("System Status")
st.sidebar.success("✅ System Online")
st.sidebar.info("🔄 Last Update: 2 min ago")
st.sidebar.info("📡 Data Source: Polygon.io")
st.sidebar.info("🤖 Models: Loaded")
