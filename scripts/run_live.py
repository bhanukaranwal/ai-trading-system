import sys
from pathlib import Path
import asyncio

sys.path.append(str(Path(__file__).parent.parent))

from src.data_ingestion import DataIngestion
from src.features import FeatureEngineering
from src.signals import SignalGenerator
from src.execution import ExecutionManager
from src.portfolio import MultiAssetPortfolio
from loguru import logger
import yaml


async def main():
    logger.info("Starting live trading system")
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    data_ingestor = DataIngestion()
    feature_eng = FeatureEngineering()
    signal_gen = SignalGenerator()
    portfolio = MultiAssetPortfolio()
    execution_mgr = ExecutionManager()
    
    all_tickers = (
        config['assets']['equities'] +
        config['assets']['bonds'] +
        config['assets']['commodities']
    )
    
    try:
        sample_data = data_ingestor.load_from_parquet(all_tickers[0])
        sample_features = feature_eng.create_all_features(sample_data)
        X_sample, _, feature_cols = feature_eng.prepare_ml_dataset(sample_features)
        input_size = X_sample.shape[1]
        
        signal_gen.load_models(input_size=input_size)
    except Exception as e:
        logger.warning(f"Could not load models: {e}")
    
    account_info = execution_mgr.executor.get_account()
    initial_cash = account_info['portfolio_value']
    
    portfolio.initialize_portfolio(initial_cash)
    
    logger.info("Live trading system initialized. Press Ctrl+C to stop.")
    
    try:
        while True:
            logger.info("Fetching latest market data")
            
            live_quotes = await data_ingestor.fetch_live_quotes(all_tickers)
            
            logger.info(f"Received quotes for {len(live_quotes)} assets")
            
            current_value = portfolio.update_portfolio_value(live_quotes)
            logger.info(f"Current portfolio value: ${current_value:,.2f}")
            
            await asyncio.sleep(300)
            
    except KeyboardInterrupt:
        logger.info("Shutting down live trading system")


if __name__ == "__main__":
    asyncio.run(main())
