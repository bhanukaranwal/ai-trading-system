import pandas as pd
import numpy as np
from typing import Dict, List
import talib as ta
from loguru import logger
import yaml


class FeatureEngineering:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.technical_indicators = self.config['features']['technical_indicators']
        self.returns_windows = self.config['features']['returns_windows']
        self.volatility_windows = self.config['features']['volatility_windows']
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        if 'SMA_20' in self.technical_indicators:
            df['sma_20'] = ta.SMA(close, timeperiod=20)
        if 'SMA_50' in self.technical_indicators:
            df['sma_50'] = ta.SMA(close, timeperiod=50)
        if 'RSI_14' in self.technical_indicators:
            df['rsi_14'] = ta.RSI(close, timeperiod=14)
        if 'MACD' in self.technical_indicators:
            macd, macdsignal, macdhist = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            df['macd'] = macd
            df['macd_signal'] = macdsignal
            df['macd_hist'] = macdhist
        if 'BB_UPPER' in self.technical_indicators or 'BB_LOWER' in self.technical_indicators:
            upper, middle, lower = ta.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            df['bb_upper'] = upper
            df['bb_middle'] = middle
            df['bb_lower'] = lower
        if 'ATR_14' in self.technical_indicators:
            df['atr_14'] = ta.ATR(high, low, close, timeperiod=14)
        if 'ADX_14' in self.technical_indicators:
            df['adx_14'] = ta.ADX(high, low, close, timeperiod=14)
        
        df['ema_12'] = ta.EMA(close, timeperiod=12)
        df['ema_26'] = ta.EMA(close, timeperiod=26)
        df['stoch_k'], df['stoch_d'] = ta.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
        df['cci'] = ta.CCI(high, low, close, timeperiod=20)
        df['willr'] = ta.WILLR(high, low, close, timeperiod=14)
        df['obv'] = ta.OBV(close, volume)
        
        return df
    
    def add_returns_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        for window in self.returns_windows:
            df[f'return_{window}d'] = df['close'].pct_change(window)
            df[f'log_return_{window}d'] = np.log(df['close'] / df['close'].shift(window))
        
        df['return_1d'] = df['close'].pct_change(1)
        
        return df
    
    def add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        returns = df['close'].pct_change()
        
        for window in self.volatility_windows:
            df[f'volatility_{window}d'] = returns.rolling(window).std() * np.sqrt(252)
            df[f'realized_vol_{window}d'] = returns.rolling(window).std()
        
        df['parkinson_vol'] = np.sqrt((1 / (4 * np.log(2))) * (np.log(df['high'] / df['low'])) ** 2)
        
        return df
    
    def add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        df['momentum_21'] = df['close'] / df['close'].shift(21) - 1
        df['momentum_63'] = df['close'] / df['close'].shift(63) - 1
        
        df['rate_of_change'] = ta.ROC(df['close'].values, timeperiod=10)
        
        return df
    
    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df['volume_sma_20'] = ta.SMA(df['volume'].values, timeperiod=20)
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        df['volume_change'] = df['volume'].pct_change()
        
        return df
    
    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df['high_low_spread'] = (df['high'] - df['low']) / df['close']
        df['close_open_spread'] = (df['close'] - df['open']) / df['open']
        df['daily_range'] = df['high'] - df['low']
        
        return df
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating technical indicators")
        df = self.add_technical_indicators(df)
        
        logger.info("Creating returns features")
        df = self.add_returns_features(df)
        
        logger.info("Creating volatility features")
        df = self.add_volatility_features(df)
        
        logger.info("Creating momentum features")
        df = self.add_momentum_features(df)
        
        logger.info("Creating volume features")
        df = self.add_volume_features(df)
        
        logger.info("Creating price features")
        df = self.add_price_features(df)
        
        df = df.replace([np.inf, -np.inf], np.nan)
        
        return df
    
    def create_features_for_assets(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        feature_dict = {}
        for ticker, df in data_dict.items():
            logger.info(f"Creating features for {ticker}")
            feature_dict[ticker] = self.create_all_features(df)
        
        return feature_dict
    
    def prepare_ml_dataset(self, df: pd.DataFrame, target_col: str = 'return_1d', lookback: int = 60) -> tuple:
        df = df.dropna()
        
        feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume', target_col]]
        
        X = df[feature_cols].values
        y = df[target_col].shift(-1).values
        
        mask = ~np.isnan(y)
        X = X[mask]
        y = y[mask]
        
        return X, y, feature_cols
