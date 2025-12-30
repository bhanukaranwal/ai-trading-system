import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data_ingestion import DataIngestion
from loguru import logger


def main():
    logger.info("Starting data download process")
    
    data_ingestor = DataIngestion()
    
    data_dict = data_ingestor.download_all_assets(force_yahoo=False)
    
    logger.success(f"Downloaded data for {len(data_dict)} assets")
    
    for ticker, df in data_dict.items():
        logger.info(f"{ticker}: {len(df)} rows, Date range: {df.index[0]} to {df.index[-1]}")


if __name__ == "__main__":
    main()
