import ccxt
import pandas as pd
import logging
from datetime import datetime, timedelta
import time
from ..config.config import (
    EXCHANGE_ID, SYMBOL, TIMEFRAMES, BINANCE_API_KEY, BINANCE_SECRET_KEY
)

logger = logging.getLogger(__name__)

class ExchangeData:
    def __init__(self):
        """Initialize the exchange connection"""
        self.exchange = getattr(ccxt, EXCHANGE_ID)({
            'apiKey': BINANCE_API_KEY,
            'secret': BINANCE_SECRET_KEY,
            'enableRateLimit': True
        })
        self.symbol = SYMBOL
        self.timeframes = TIMEFRAMES
        
    def fetch_ohlcv_data(self):
        """Fetch OHLCV data for all timeframes"""
        try:
            data = {}
            for timeframe in self.timeframes:
                ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe, limit=100)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                data[timeframe] = df
            return data
        except Exception as e:
            logger.error(f"Error fetching OHLCV data: {e}")
            return None
            
    def fetch_historical_data_for_training(self, days=60):
        """Fetch historical data for ML model training"""
        try:
            logger.info(f"Fetching {days} days of historical data for ML training")
            
            # Calculate timestamps
            end_time = int(time.time() * 1000)
            start_time = end_time - (days * 24 * 60 * 60 * 1000)
            
            # Fetch historical data
            historical_data = self.exchange.fetch_ohlcv(
                self.symbol, 
                self.timeframes[0],  # Use 1h timeframe for training
                since=start_time,
                limit=1000
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(historical_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            logger.info(f"Fetched {len(df)} historical candles for ML training")
            return df
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return None
            
    def get_current_price(self):
        """Get the current price of the trading pair"""
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            return ticker['last']
        except Exception as e:
            logger.error(f"Error fetching current price: {e}")
            return None 