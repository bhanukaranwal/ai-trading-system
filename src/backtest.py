import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Dict, List
from loguru import logger
import yaml


class VectorbtBacktest:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.backtest_params = self.config['backtest']
        self.portfolio = None
        
    def prepare_price_data(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        price_df = pd.DataFrame()
        
        for ticker, df in data_dict.items():
            price_df[ticker] = df['close']
        
        price_df = price_df.dropna()
        
        return price_df
    
    def prepare_signals_data(self, signals_dict: Dict[str, pd.DataFrame], 
                            signal_col: str = 'signal') -> pd.DataFrame:
        signals_df = pd.DataFrame()
        
        for ticker, df in signals_dict.items():
            signals_df[ticker] = df[signal_col]
        
        signals_df = signals_df.fillna(0)
        
        return signals_df
    
    def run_backtest(self, prices: pd.DataFrame, signals: pd.DataFrame, 
                    initial_cash: float = None, commission: float = None) -> vbt.Portfolio:
        
        if initial_cash is None:
            initial_cash = self.backtest_params['initial_cash']
        if commission is None:
            commission = self.backtest_params['commission']
        
        entries = signals > 0
        exits = signals < 0
        
        logger.info(f"Running backtest on {len(prices.columns)} assets")
        logger.info(f"Date range: {prices.index[0]} to {prices.index[-1]}")
        logger.info(f"Initial cash: ${initial_cash:,.2f}")
        
        self.portfolio = vbt.Portfolio.from_signals(
            close=prices,
            entries=entries,
            exits=exits,
            init_cash=initial_cash,
            fees=commission,
            freq='1D'
        )
        
        return self.portfolio
    
    def get_performance_metrics(self) -> Dict:
        if self.portfolio is None:
            raise ValueError("No backtest results available")
        
        metrics = {
            'total_return': self.portfolio.total_return(),
            'annual_return': self.portfolio.annualized_return(),
            'sharpe_ratio': self.portfolio.sharpe_ratio(),
            'sortino_ratio': self.portfolio.sortino_ratio(),
            'max_drawdown': self.portfolio.max_drawdown(),
            'calmar_ratio': self.portfolio.calmar_ratio(),
            'win_rate': self.portfolio.trades.win_rate(),
            'total_trades': self.portfolio.trades.count(),
            'profit_factor': self.portfolio.trades.profit_factor(),
        }
        
        logger.info("Backtest Performance:")
        logger.info(f"Total Return: {metrics['total_return']:.2%}")
        logger.info(f"Annual Return: {metrics['annual_return']:.2%}")
        logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        logger.info(f"Win Rate: {metrics['win_rate']:.2%}")
        
        return metrics
    
    def get_portfolio_value(self) -> pd.Series:
        if self.portfolio is None:
            raise ValueError("No backtest results available")
        return self.portfolio.value()
    
    def get_returns(self) -> pd.Series:
        if self.portfolio is None:
            raise ValueError("No backtest results available")
        return self.portfolio.returns()
    
    def get_positions(self) -> pd.DataFrame:
        if self.portfolio is None:
            raise ValueError("No backtest results available")
        return self.portfolio.positions()
    
    def plot_portfolio_value(self):
        if self.portfolio is None:
            raise ValueError("No backtest results available")
        
        fig = self.portfolio.plot()
        return fig
    
    def run_multi_strategy_backtest(self, prices: pd.DataFrame, 
                                   strategies_signals: Dict[str, pd.DataFrame]) -> Dict:
        
        results = {}
        
        for strategy_name, signals in strategies_signals.items():
            logger.info(f"Running backtest for strategy: {strategy_name}")
            
            portfolio = self.run_backtest(prices, signals)
            metrics = self.get_performance_metrics()
            
            results[strategy_name] = {
                'portfolio': portfolio,
                'metrics': metrics
            }
        
        return results
    
    def compare_strategies(self, results: Dict) -> pd.DataFrame:
        comparison_df = pd.DataFrame()
        
        for strategy_name, result in results.items():
            metrics = result['metrics']
            comparison_df[strategy_name] = pd.Series(metrics)
        
        return comparison_df.T
    
    def generate_trade_log(self) -> pd.DataFrame:
        if self.portfolio is None:
            raise ValueError("No backtest results available")
        
        trades = self.portfolio.trades.records_readable
        
        return trades
    
    def calculate_rolling_metrics(self, window: int = 252) -> pd.DataFrame:
        if self.portfolio is None:
            raise ValueError("No backtest results available")
        
        returns = self.get_returns()
        
        rolling_metrics = pd.DataFrame({
            'rolling_return': returns.rolling(window).sum(),
            'rolling_sharpe': returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252),
            'rolling_volatility': returns.rolling(window).std() * np.sqrt(252)
        })
        
        return rolling_metrics
