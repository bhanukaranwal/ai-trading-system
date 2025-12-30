import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data_ingestion import DataIngestion
from src.features import FeatureEngineering
from src.signals import SignalGenerator
from src.backtest import VectorbtBacktest
from src.optimization import PortfolioOptimizer
from loguru import logger
import pandas as pd


def main():
    logger.info("Starting comprehensive backtest")
    
    data_ingestor = DataIngestion()
    data_dict = data_ingestor.load_all_assets()
    
    if not data_dict:
        logger.error("No data found. Please run download_data.py first")
        return
    
    feature_eng = FeatureEngineering()
    feature_dict = feature_eng.create_features_for_assets(data_dict)
    
    signal_gen = SignalGenerator()
    
    try:
        sample_df = feature_dict[list(feature_dict.keys())[0]]
        X_sample, _, feature_cols = feature_eng.prepare_ml_dataset(sample_df)
        input_size = X_sample.shape[1]
        
        signal_gen.load_models(input_size=input_size)
    except Exception as e:
        logger.warning(f"Could not load models: {e}. Using simple momentum signals.")
    
    signals_dict = {}
    for ticker, df in feature_dict.items():
        momentum_signal = df['close'].pct_change(10).apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
        signals_dict[ticker] = pd.DataFrame({'signal': momentum_signal})
    
    backtester = VectorbtBacktest()
    
    prices = backtester.prepare_price_data(data_dict)
    signals = backtester.prepare_signals_data(signals_dict)
    
    portfolio = backtester.run_backtest(prices, signals)
    
    metrics = backtester.get_performance_metrics()
    
    portfolio_value = backtester.get_portfolio_value()
    portfolio_value.to_frame('portfolio_value').to_parquet('data/backtest_results.parquet')
    
    logger.success("Backtest completed successfully")
    
    logger.info("\nPerformance Summary:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value}")


if __name__ == "__main__":
    main()
