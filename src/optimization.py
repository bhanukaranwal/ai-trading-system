import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt import objective_functions
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from typing import Dict, Tuple
from loguru import logger
import yaml


class PortfolioOptimizer:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.portfolio_config = self.config['portfolio']
        self.weights = None
        self.performance = None
        
    def calculate_expected_returns(self, prices: pd.DataFrame, method: str = 'mean_historical_return') -> pd.Series:
        if method == 'mean_historical_return':
            return expected_returns.mean_historical_return(prices)
        elif method == 'ema_historical_return':
            return expected_returns.ema_historical_return(prices)
        elif method == 'capm_return':
            return expected_returns.capm_return(prices)
        else:
            return expected_returns.mean_historical_return(prices)
    
    def calculate_risk_matrix(self, prices: pd.DataFrame, method: str = 'sample_cov') -> pd.DataFrame:
        if method == 'sample_cov':
            return risk_models.sample_cov(prices)
        elif method == 'semicovariance':
            return risk_models.semicovariance(prices)
        elif method == 'exp_cov':
            return risk_models.exp_cov(prices)
        else:
            return risk_models.sample_cov(prices)
    
    def optimize_max_sharpe(self, prices: pd.DataFrame) -> Dict:
        mu = self.calculate_expected_returns(prices)
        S = self.calculate_risk_matrix(prices)
        
        ef = EfficientFrontier(mu, S)
        ef.add_objective(objective_functions.L2_reg, gamma=0.1)
        
        ef.min_weight = self.portfolio_config['min_weight']
        ef.max_weight = self.portfolio_config['max_weight']
        
        self.weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        
        self.performance = ef.portfolio_performance(verbose=True)
        
        logger.info(f"Expected annual return: {self.performance[0]:.2%}")
        logger.info(f"Annual volatility: {self.performance[1]:.2%}")
        logger.info(f"Sharpe Ratio: {self.performance[2]:.2f}")
        
        return cleaned_weights
    
    def optimize_min_volatility(self, prices: pd.DataFrame) -> Dict:
        mu = self.calculate_expected_returns(prices)
        S = self.calculate_risk_matrix(prices)
        
        ef = EfficientFrontier(mu, S)
        
        ef.min_weight = self.portfolio_config['min_weight']
        ef.max_weight = self.portfolio_config['max_weight']
        
        self.weights = ef.min_volatility()
        cleaned_weights = ef.clean_weights()
        
        self.performance = ef.portfolio_performance(verbose=True)
        
        return cleaned_weights
    
    def optimize_efficient_risk(self, prices: pd.DataFrame, target_volatility: float = None) -> Dict:
        if target_volatility is None:
            target_volatility = self.portfolio_config['target_volatility']
        
        mu = self.calculate_expected_returns(prices)
        S = self.calculate_risk_matrix(prices)
        
        ef = EfficientFrontier(mu, S)
        
        ef.min_weight = self.portfolio_config['min_weight']
        ef.max_weight = self.portfolio_config['max_weight']
        
        self.weights = ef.efficient_risk(target_volatility)
        cleaned_weights = ef.clean_weights()
        
        self.performance = ef.portfolio_performance(verbose=True)
        
        return cleaned_weights
    
    def optimize_efficient_return(self, prices: pd.DataFrame, target_return: float) -> Dict:
        mu = self.calculate_expected_returns(prices)
        S = self.calculate_risk_matrix(prices)
        
        ef = EfficientFrontier(mu, S)
        
        ef.min_weight = self.portfolio_config['min_weight']
        ef.max_weight = self.portfolio_config['max_weight']
        
        self.weights = ef.efficient_return(target_return)
        cleaned_weights = ef.clean_weights()
        
        self.performance = ef.portfolio_performance(verbose=True)
        
        return cleaned_weights
    
    def risk_parity_weights(self, prices: pd.DataFrame) -> Dict:
        S = self.calculate_risk_matrix(prices)
        
        n_assets = len(prices.columns)
        weights = np.ones(n_assets) / n_assets
        
        for _ in range(100):
            portfolio_vol = np.sqrt(weights @ S @ weights)
            marginal_contrib = S @ weights
            risk_contrib = weights * marginal_contrib / portfolio_vol
            
            target_risk = portfolio_vol / n_assets
            weights = weights * target_risk / risk_contrib
            weights = weights / weights.sum()
        
        weights_dict = dict(zip(prices.columns, weights))
        
        logger.info("Risk Parity Weights:")
        for asset, weight in weights_dict.items():
            logger.info(f"{asset}: {weight:.4f}")
        
        return weights_dict
    
    def optimize_portfolio(self, prices: pd.DataFrame, method: str = None) -> Dict:
        if method is None:
            method = self.portfolio_config['optimization_method']
        
        logger.info(f"Optimizing portfolio using method: {method}")
        
        if method == 'max_sharpe':
            return self.optimize_max_sharpe(prices)
        elif method == 'min_volatility':
            return self.optimize_min_volatility(prices)
        elif method == 'efficient_risk':
            return self.optimize_efficient_risk(prices)
        elif method == 'risk_parity':
            return self.risk_parity_weights(prices)
        else:
            return self.optimize_max_sharpe(prices)
    
    def allocate_discrete(self, weights: Dict, latest_prices: pd.Series, 
                         total_portfolio_value: float) -> Tuple[Dict, float]:
        
        da = DiscreteAllocation(weights, latest_prices, total_portfolio_value)
        allocation, leftover = da.greedy_portfolio()
        
        logger.info("Discrete Allocation:")
        for asset, shares in allocation.items():
            logger.info(f"{asset}: {shares} shares")
        logger.info(f"Leftover cash: ${leftover:.2f}")
        
        return allocation, leftover
    
    def rebalance_weights(self, current_weights: Dict, target_weights: Dict, 
                         threshold: float = 0.05) -> Dict:
        
        trades = {}
        
        for asset in target_weights:
            current = current_weights.get(asset, 0)
            target = target_weights[asset]
            diff = target - current
            
            if abs(diff) > threshold:
                trades[asset] = diff
        
        logger.info(f"Rebalancing: {len(trades)} assets need adjustment")
        
        return trades
