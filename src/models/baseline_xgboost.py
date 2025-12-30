import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, Tuple
import joblib
from pathlib import Path
from loguru import logger
import yaml


class XGBoostModel:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.params = self.config['models']['xgboost']
        self.model = None
        self.feature_names = None
        
    def build_model(self):
        self.model = xgb.XGBRegressor(
            n_estimators=self.params['n_estimators'],
            max_depth=self.params['max_depth'],
            learning_rate=self.params['learning_rate'],
            subsample=self.params['subsample'],
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )
        return self.model
    
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: list = None, test_size: float = 0.2):
        self.feature_names = feature_names
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        if self.model is None:
            self.build_model()
        
        logger.info(f"Training XGBoost with {len(X_train)} samples")
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        
        logger.info(f"Train RMSE: {train_rmse:.6f}, Test RMSE: {test_rmse:.6f}")
        logger.info(f"Train MAE: {train_mae:.6f}, Test MAE: {test_mae:.6f}")
        
        return {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        if self.model is None or self.feature_names is None:
            raise ValueError("Model not trained yet")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath: str = "models/xgboost_model.pkl"):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'params': self.params
        }, filepath)
        logger.success(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str = "models/xgboost_model.pkl"):
        data = joblib.load(filepath)
        self.model = data['model']
        self.feature_names = data['feature_names']
        self.params = data['params']
        logger.success(f"Model loaded from {filepath}")
