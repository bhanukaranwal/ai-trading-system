import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime
from loguru import logger
import yaml
from src.optimization import PortfolioOptimizer
from src.risk_management import RiskManager


class MultiAssetPortfolio:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.optimizer = PortfolioOptimizer(config_path)
        self.risk_manager = RiskManager(config_path)
        
        self.weights = {}
        self.positions = {}
        self.cash = self.config['backtest']['initial_cash']
        self.portfolio_value = self.cash
        self.history = []
        
    def initialize_portfolio(self, initial_cash: float):
        self.cash = initial_cash
        self.portfolio_value = initial_cash
        self.positions = {}
        self.weights = {}
        
        logger.info(f"Portfolio initialized with ${initial_cash:,.2f}")
    
    def calculate_current_weights(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        if not self.positions:
            return {}
        
        position_values = {}
        total_value = self.cash
        
        for symbol, qty in self.positions.items():
            value = qty * current_prices.get(symbol, 0)
            position_values[symbol] = value
            total_value += value
        
        if total_value == 0:
            return {}
        
        weights = {symbol: value / total_value for symbol, value in position_values.items()}
        
        return weights
    
    def optimize_and_rebalance(self, prices: pd.DataFrame, method: str = None) -> Dict[str, float]:
        
        target_weights = self.optimizer.optimize_portfolio(prices, method)
        
        returns_dict = {}
        for col in prices.columns:
            returns_dict[col] = prices[col].pct_change().dropna()
        
        target_weights = self.risk_manager.apply_risk_overlay(target_weights, returns_dict)
        
        self.weights = target_weights
        
        logger.info("Target portfolio weights:")
        for asset, weight in target_weights.items():
            logger.info(f"{asset}: {weight:.2%}")
        
        return target_weights
    
    def execute_rebalance(self, current_prices: Dict[str, float], target_weights: Dict[str, float]):
        
        total_value = self.portfolio_value
        
        new_positions = {}
        
        for symbol, weight in target_weights.items():
            if weight > 0.001:
                target_value = total_value * weight
                qty = target_value / current_prices[symbol]
                new_positions[symbol] = qty
        
        for symbol in list(self.positions.keys()):
            if symbol not in new_positions:
                sell_value = self.positions[symbol] * current_prices[symbol]
                self.cash += sell_value
                logger.info(f"Closed position in {symbol}: {self.positions[symbol]:.2f} shares")
        
        for symbol, target_qty in new_positions.items():
            current_qty = self.positions.get(symbol, 0)
            qty_diff = target_qty - current_qty
            
            if abs(qty_diff) > 0.01:
                trade_value = qty_diff * current_prices[symbol]
                self.cash -= trade_value
                
                if qty_diff > 0:
                    logger.info(f"Bought {qty_diff:.2f} shares of {symbol}")
                else:
                    logger.info(f"Sold {abs(qty_diff):.2f} shares of {symbol}")
        
        self.positions = new_positions
        
        self.update_portfolio_value(current_prices)
    
    def update_portfolio_value(self, current_prices: Dict[str, float]):
        position_value = sum(qty * current_prices.get(symbol, 0) 
                           for symbol, qty in self.positions.items())
        
        self.portfolio_value = self.cash + position_value
        
        self.history.append({
            'timestamp': datetime.now(),
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'position_value': position_value
        })
        
        return self.portfolio_value
    
    def get_portfolio_summary(self, current_prices: Dict[str, float]) -> Dict:
        
        position_details = []
        total_position_value = 0
        
        for symbol, qty in self.positions.items():
            price = current_prices.get(symbol, 0)
            value = qty * price
            total_position_value += value
            
            position_details.append({
                'symbol': symbol,
                'quantity': qty,
                'price': price,
                'value': value,
                'weight': value / self.portfolio_value if self.portfolio_value > 0 else 0
            })
        
        summary = {
            'total_value': self.portfolio_value,
            'cash': self.cash,
            'invested': total_position_value,
            'positions': position_details,
            'num_positions': len(self.positions)
        }
        
        return summary
    
    def get_performance_history(self) -> pd.DataFrame:
        if not self.history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.history)
        df['returns'] = df['portfolio_value'].pct_change()
        df['cumulative_returns'] = (1 + df['returns']).cumprod() - 1
        
        return df
    
    def calculate_portfolio_metrics(self) -> Dict:
        df = self.get_performance_history()
        
        if df.empty or len(df) < 2:
            return {}
        
        returns = df['returns'].dropna()
        
        metrics = self.risk_manager.calculate_portfolio_metrics(returns)
        metrics['total_value'] = self.portfolio_value
        metrics['total_return_pct'] = (self.portfolio_value / self.config['backtest']['initial_cash'] - 1) * 100
        
        return metrics
    
    def should_rebalance(self, current_date: datetime, last_rebalance: datetime = None) -> bool:
        if last_rebalance is None:
            return True
        
        freq = self.config['portfolio']['rebalance_frequency']
        
        if freq == 'daily':
            return True
        elif freq == 'weekly':
            return (current_date - last_rebalance).days >= 7
        elif freq == 'monthly':
            return (current_date - last_rebalance).days >= 30
        elif freq == 'quarterly':
            return (current_date - last_rebalance).days >= 90
        
        return False
