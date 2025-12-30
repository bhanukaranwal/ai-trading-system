import numpy as np
import pandas as pd
from typing import Dict, List
from pathlib import Path
from loguru import logger
import yaml
from src.models.baseline_xgboost import XGBoostModel
from src.models.lstm_model import LSTMTrainer
from src.models.transformer_model import TransformerTrainer


class SignalGenerator:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.xgb_model = XGBoostModel(config_path)
        self.lstm_model = LSTMTrainer(config_path)
        self.transformer_model = TransformerTrainer(config_path)
        
        self.models_loaded = False
    
    def load_models(self, xgb_path: str = "models/xgboost_model.pkl",
                   lstm_path: str = "models/lstm_model.pth",
                   transformer_path: str = "models/transformer_model.pth",
                   input_size: int = None):
        
        if Path(xgb_path).exists():
            self.xgb_model.load_model(xgb_path)
            logger.success("XGBoost model loaded")
        else:
            logger.warning(f"XGBoost model not found at {xgb_path}")
        
        if Path(lstm_path).exists() and input_size is not None:
            self.lstm_model.load_model(lstm_path, input_size)
            logger.success("LSTM model loaded")
        else:
            logger.warning(f"LSTM model not found or input_size not provided")
        
        if Path(transformer_path).exists() and input_size is not None:
            self.transformer_model.load_model(transformer_path, input_size)
            logger.success("Transformer model loaded")
        else:
            logger.warning(f"Transformer model not found or input_size not provided")
        
        self.models_loaded = True
    
    def generate_xgb_signals(self, X: np.ndarray) -> np.ndarray:
        try:
            predictions = self.xgb_model.predict(X)
            signals = np.sign(predictions)
            return signals
        except Exception as e:
            logger.error(f"XGBoost signal generation failed: {e}")
            return np.zeros(len(X))
    
    def generate_lstm_signals(self, X: np.ndarray) -> np.ndarray:
        try:
            predictions = self.lstm_model.predict(X)
            seq_length = self.config['models']['lstm']['sequence_length']
            full_predictions = np.zeros(len(X))
            full_predictions[seq_length:] = predictions
            signals = np.sign(full_predictions)
            return signals
        except Exception as e:
            logger.error(f"LSTM signal generation failed: {e}")
            return np.zeros(len(X))
    
    def generate_transformer_signals(self, X: np.ndarray) -> np.ndarray:
        try:
            predictions = self.transformer_model.predict(X)
            seq_length = self.config['models']['transformer']['sequence_length']
            full_predictions = np.zeros(len(X))
            full_predictions[seq_length:] = predictions
            signals = np.sign(full_predictions)
            return signals
        except Exception as e:
            logger.error(f"Transformer signal generation failed: {e}")
            return np.zeros(len(X))
    
    def generate_ensemble_signals(self, X: np.ndarray, weights: Dict[str, float] = None) -> np.ndarray:
        if weights is None:
            weights = {'xgboost': 0.4, 'lstm': 0.3, 'transformer': 0.3}
        
        xgb_signals = self.generate_xgb_signals(X)
        lstm_signals = self.generate_lstm_signals(X)
        transformer_signals = self.generate_transformer_signals(X)
        
        ensemble_signals = (
            weights['xgboost'] * xgb_signals +
            weights['lstm'] * lstm_signals +
            weights['transformer'] * transformer_signals
        )
        
        final_signals = np.sign(ensemble_signals)
        
        return final_signals
    
    def generate_signals_with_confidence(self, X: np.ndarray) -> pd.DataFrame:
        xgb_pred = self.xgb_model.predict(X)
        
        try:
            lstm_pred = self.lstm_model.predict(X)
            seq_length = self.config['models']['lstm']['sequence_length']
            full_lstm_pred = np.zeros(len(X))
            full_lstm_pred[seq_length:] = lstm_pred
            lstm_pred = full_lstm_pred
        except:
            lstm_pred = np.zeros(len(X))
        
        try:
            transformer_pred = self.transformer_model.predict(X)
            seq_length = self.config['models']['transformer']['sequence_length']
            full_transformer_pred = np.zeros(len(X))
            full_transformer_pred[seq_length:] = transformer_pred
            transformer_pred = full_transformer_pred
        except:
            transformer_pred = np.zeros(len(X))
        
        ensemble_pred = (xgb_pred + lstm_pred + transformer_pred) / 3.0
        
        signals_df = pd.DataFrame({
            'xgb_prediction': xgb_pred,
            'lstm_prediction': lstm_pred,
            'transformer_prediction': transformer_pred,
            'ensemble_prediction': ensemble_pred,
            'signal': np.sign(ensemble_pred),
            'confidence': np.abs(ensemble_pred)
        })
        
        return signals_df
    
    def generate_multi_asset_signals(self, feature_dict: Dict[str, pd.DataFrame], 
                                    feature_cols: List[str]) -> Dict[str, pd.DataFrame]:
        signals_dict = {}
        
        for ticker, df in feature_dict.items():
            logger.info(f"Generating signals for {ticker}")
            
            df_clean = df[feature_cols].dropna()
            X = df_clean.values
            
            signals_df = self.generate_signals_with_confidence(X)
            signals_df.index = df_clean.index
            
            signals_dict[ticker] = signals_df
        
        return signals_dict
