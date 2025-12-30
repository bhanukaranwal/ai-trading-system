import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data_ingestion import DataIngestion
from src.features import FeatureEngineering
from src.models.baseline_xgboost import XGBoostModel
from src.models.lstm_model import LSTMTrainer
from src.models.transformer_model import TransformerTrainer
from loguru import logger


def main():
    logger.info("Starting model training pipeline")
    
    data_ingestor = DataIngestion()
    data_dict = data_ingestor.load_all_assets()
    
    if not data_dict:
        logger.error("No data found. Please run download_data.py first")
        return
    
    feature_eng = FeatureEngineering()
    feature_dict = feature_eng.create_features_for_assets(data_dict)
    
    sample_ticker = list(feature_dict.keys())[0]
    sample_df = feature_dict[sample_ticker]
    
    X, y, feature_cols = feature_eng.prepare_ml_dataset(sample_df)
    
    logger.info(f"Dataset prepared: X shape {X.shape}, y shape {y.shape}")
    
    logger.info("Training XGBoost model")
    xgb_model = XGBoostModel()
    xgb_metrics = xgb_model.train(X, y, feature_cols)
    xgb_model.save_model("models/xgboost_model.pkl")
    
    logger.info("Training LSTM model")
    lstm_model = LSTMTrainer()
    lstm_metrics = lstm_model.train(X, y)
    lstm_model.save_model("models/lstm_model.pth")
    
    logger.info("Training Transformer model")
    transformer_model = TransformerTrainer()
    transformer_metrics = transformer_model.train(X, y)
    transformer_model.save_model("models/transformer_model.pth")
    
    logger.success("All models trained and saved successfully")


if __name__ == "__main__":
    main()
