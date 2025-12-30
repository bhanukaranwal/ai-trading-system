import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple
from loguru import logger
import yaml


class RiskManager:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.risk_config = self.config['risk']
        
    def calculate_var(self, returns: pd.Series, confidence: float = None) -> float:
        if confidence is None:
            confidence = self.risk_config['var_confidence']
        
        var = np.percentile(returns.dropna(), (1 - confidence) * 100)
        
        logger.info(f"VaR at {confidence:.0%} confidence: {var:.4f}")
        
        return var
    
    def calculate_cvar(self, returns: pd.Series, confidence: float = None) -> float:
        if confidence is None:
            confidence = self.risk_config['cvar_confidence']
        
        var = self.calculate_var(returns, confidence)
        cvar = returns[returns <= var].mean()
        
        logger.info(f"CVaR at {confidence:.0%} confidence: {cvar:.4f}")
        
        return cvar
    
    def calculate_max_drawdown(self, returns: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        
        max_dd = drawdown.min()
        end_date = drawdown.idxmin()
        
        peak_date = cumulative[:end_date].idxmax()
        
        logger.info(f"Max Drawdown: {max_dd:.2%}")
        logger.info(f"Peak: {peak_date}, Trough: {end_date}")
        
        return max_dd, peak_date, end_date
    
    def calculate_volatility(self, returns: pd.Series, annualize: bool = True) -> float:
        vol = returns.std()
        
        if annualize:
            vol = vol * np.sqrt(252)
        
        return vol
    
    def kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        if avg_loss == 0:
            return 0
        
        win_loss_ratio = avg_win / abs(avg_loss)
        kelly = win_rate - (1 - win_rate) / win_loss_ratio
        
        kelly_fractional = kelly * self.risk_config['kelly_fraction']
        
        logger.info(f"Kelly Criterion: {kelly:.4f}, Fractional Kelly: {kelly_fractional:.4f}")
        
        return max(0, min(kelly_fractional, 1))
    
    def position_size_kelly(self, returns: pd.Series) -> float:
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        if len(positive_returns) == 0 or len(negative_returns) == 0:
            return 0.1
        
        win_rate = len(positive_returns) / len(returns)
        avg_win = positive_returns.mean()
        avg_loss = negative_returns.mean()
        
        kelly = self.kelly_criterion(win_rate, avg_win, abs(avg_loss))
        
        return kelly
    
    def position_size_volatility_target(self, returns: pd.Series, target_vol: float = None) -> float:
        if target_vol is None:
            target_vol = self.risk_config.get('target_volatility', 0.15)
        
        current_vol = self.calculate_volatility(returns, annualize=True)
        
        if current_vol == 0:
            return 0
        
        position_size = target_vol / current_vol
        
        position_size = min(position_size, self.risk_config['leverage'])
        
        logger.info(f"Volatility-targeted position size: {position_size:.4f}")
        
        return position_size
    
    def calculate_position_sizes(self, returns_dict: Dict[str, pd.Series], 
                                method: str = None) -> Dict[str, float]:
        
        if method is None:
            method = self.risk_config['position_sizing']
        
        position_sizes = {}
        
        for asset, returns in returns_dict.items():
            if method == 'kelly_fractional':
                size = self.position_size_kelly(returns)
            elif method == 'volatility_targeting':
                size = self.position_size_volatility_target(returns)
            else:
                size = 1.0 / len(returns_dict)
            
            position_sizes[asset] = size
        
        total = sum(position_sizes.values())
        if total > 0:
            position_sizes = {k: v/total for k, v in position_sizes.items()}
        
        return position_sizes
    
    def check_risk_limits(self, portfolio_value: float, current_drawdown: float) -> bool:
        max_dd_limit = self.risk_config['max_drawdown']
        
        if abs(current_drawdown) > max_dd_limit:
            logger.warning(f"Drawdown {current_drawdown:.2%} exceeds limit {max_dd_limit:.2%}")
            return False
        
        return True
    
    def apply_risk_overlay(self, weights: Dict[str, float], returns_dict: Dict[str, pd.Series]) -> Dict[str, float]:
        
        adjusted_weights = weights.copy()
        
        for asset, weight in weights.items():
            if asset in returns_dict:
                returns = returns_dict[asset]
                var = self.calculate_var(returns)
                
                if var < -0.05:
                    adjustment_factor = 0.5
                    adjusted_weights[asset] = weight * adjustment_factor
                    logger.info(f"Reduced {asset} weight due to high VaR")
        
        total = sum(adjusted_weights.values())
        if total > 0:
            adjusted_weights = {k: v/total for k, v in adjusted_weights.items()}
        
        return adjusted_weights
    
    def calculate_portfolio_metrics(self, returns: pd.Series) -> Dict:
        metrics = {
            'total_return': (1 + returns).prod() - 1,
            'annual_return': returns.mean() * 252,
            'annual_volatility': self.calculate_volatility(returns),
            'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
            'sortino_ratio': (returns.mean() * 252) / (returns[returns < 0].std() * np.sqrt(252)),
            'max_drawdown': self.calculate_max_drawdown(returns)[0],
            'var_95': self.calculate_var(returns, 0.95),
            'cvar_95': self.calculate_cvar(returns, 0.95),
            'skewness': stats.skew(returns.dropna()),
            'kurtosis': stats.kurtosis(returns.dropna())
        }
        
        return metrics
