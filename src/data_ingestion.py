import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import yfinance as yf
from polygon import RESTClient
import asyncio
import aiohttp
from loguru import logger
import yaml


class DataIngestion:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.polygon_key = self.config['api']['polygon_key']
        self.polygon_client = RESTClient(self.polygon_key) if self.polygon_key != 'YOUR_POLYGON_API_KEY' else None
        self.data_dir = Path(self.config['data']['data_dir'])
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def fetch_polygon_data(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        if not self.polygon_client:
            logger.warning(f"Polygon API key not configured, falling back to Yahoo Finance for {ticker}")
            return None
            
        try:
            logger.info(f"Fetching {ticker} from Polygon.io")
            aggs = []
            for a in self.polygon_client.list_aggs(
                ticker=ticker,
                multiplier=1,
                timespan="day",
                from_=start_date,
                to=end_date,
                limit=50000
            ):
                aggs.append(a)
            
            if not aggs:
                return None
                
            df = pd.DataFrame([{
                'timestamp': a.timestamp,
                'open': a.open,
                'high': a.high,
                'low': a.low,
                'close': a.close,
                'volume': a.volume
            } for a in aggs])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df.index.name = 'date'
            return df
            
        except Exception as e:
            logger.error(f"Polygon fetch failed for {ticker}: {e}")
            return None
    
    def fetch_yahoo_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        logger.info(f"Fetching {ticker} from Yahoo Finance")
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            df.columns = [col.lower() if isinstance(col, str) else col[0].lower() for col in df.columns]
            return df
        except Exception as e:
            logger.error(f"Yahoo Finance fetch failed for {ticker}: {e}")
            return pd.DataFrame()
    
    def download_asset(self, ticker: str, start_date: str, end_date: str, force_yahoo: bool = False) -> pd.DataFrame:
        if not force_yahoo:
            df = self.fetch_polygon_data(ticker, start_date, end_date)
            if df is not None and not df.empty:
                return df
        
        return self.fetch_yahoo_data(ticker, start_date, end_date)
    
    def download_all_assets(self, force_yahoo: bool = False) -> Dict[str, pd.DataFrame]:
        start_date = self.config['data']['start_date']
        end_date = self.config['data']['end_date']
        
        all_tickers = (
            self.config['assets']['equities'] +
            self.config['assets']['bonds'] +
            self.config['assets']['commodities']
        )
        
        data_dict = {}
        for ticker in all_tickers:
            df = self.download_asset(ticker, start_date, end_date, force_yahoo)
            if not df.empty:
                data_dict[ticker] = df
                self.save_to_parquet(df, ticker)
                logger.success(f"Downloaded and saved {ticker}: {len(df)} rows")
            else:
                logger.warning(f"No data retrieved for {ticker}")
        
        return data_dict
    
    def save_to_parquet(self, df: pd.DataFrame, ticker: str):
        filepath = self.data_dir / f"{ticker}.parquet"
        df.to_parquet(filepath, compression='snappy')
        logger.info(f"Saved {ticker} to {filepath}")
    
    def load_from_parquet(self, ticker: str) -> pd.DataFrame:
        filepath = self.data_dir / f"{ticker}.parquet"
        if filepath.exists():
            return pd.read_parquet(filepath)
        else:
            logger.warning(f"Parquet file not found for {ticker}")
            return pd.DataFrame()
    
    def load_all_assets(self) -> Dict[str, pd.DataFrame]:
        all_tickers = (
            self.config['assets']['equities'] +
            self.config['assets']['bonds'] +
            self.config['assets']['commodities']
        )
        
        data_dict = {}
        for ticker in all_tickers:
            df = self.load_from_parquet(ticker)
            if not df.empty:
                data_dict[ticker] = df
        
        return data_dict
    
    async def fetch_live_quote_async(self, ticker: str) -> Dict:
        try:
            data = yf.Ticker(ticker)
            info = data.info
            return {
                'ticker': ticker,
                'price': info.get('regularMarketPrice', info.get('previousClose', 0)),
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Live quote fetch failed for {ticker}: {e}")
            return {'ticker': ticker, 'price': None, 'timestamp': datetime.now()}
    
    async def fetch_live_quotes(self, tickers: List[str]) -> Dict[str, float]:
        tasks = [self.fetch_live_quote_async(ticker) for ticker in tickers]
        results = await asyncio.gather(*tasks)
        return {r['ticker']: r['price'] for r in results if r['price'] is not None}
