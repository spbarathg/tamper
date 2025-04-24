import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime, timedelta
import time
from dotenv import load_dotenv
import os
import telegram
import asyncio
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import pickle
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SolanaTradingBot")

# Load environment variables
load_dotenv()

class SolanaTradingBot:
    def __init__(self):
        """Initialize the trading bot with configuration validation"""
        # Validate configuration
        if not os.getenv('BINANCE_API_KEY') or not os.getenv('BINANCE_SECRET_KEY'):
            raise ValueError("Binance API credentials not configured")
            
        if not os.getenv('TELEGRAM_BOT_TOKEN') or not os.getenv('TELEGRAM_CHAT_ID'):
            raise ValueError("Telegram configuration not found")
            
        # Initialize exchange
        try:
            self.exchange = ccxt.binance({
                'apiKey': os.getenv('BINANCE_API_KEY'),
                'secret': os.getenv('BINANCE_SECRET_KEY'),
                'enableRateLimit': True
            })
        except Exception as e:
            raise ValueError(f"Failed to initialize exchange: {e}")
        
        # Initialize and validate Telegram bot
        try:
            self.telegram_bot = telegram.Bot(token=os.getenv('TELEGRAM_BOT_TOKEN'))
            self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
            
            # Validate chat ID format and type
            try:
                chat_id_int = int(self.chat_id)
                if chat_id_int < 0:  # Group chat IDs are negative
                    logger.info("Using group chat ID for Telegram notifications")
                else:
                    logger.info("Using personal chat ID for Telegram notifications")
            except ValueError:
                raise ValueError("Telegram chat ID must be a valid integer")
                
            # Test Telegram configuration
            asyncio.get_event_loop().run_until_complete(
                self.telegram_bot.send_message(
                    chat_id=self.chat_id,
                    text="🔄 Trading bot initialization test message"
                )
            )
            logger.info("Telegram configuration validated successfully")
            
        except Exception as e:
            raise ValueError(f"Failed to initialize Telegram bot: {e}")
        
        # Trading parameters
        self.trading_pairs = [
            'SOL/USDT',  # Solana
            'BTC/USDT',  # Bitcoin
            'ETH/USDT',  # Ethereum
            'BNB/USDT',  # Binance Coin
            'ADA/USDT',  # Cardano
            'DOT/USDT',  # Polkadot
            'AVAX/USDT', # Avalanche
            'MATIC/USDT' # Polygon
        ]
        self.active_pairs = {}  # Dictionary to track active positions
        self.timeframe = '1h'
        
        # Advanced parameters
        self.trend_strength_threshold = 0.7  # For trend confirmation
        self.volume_threshold = 1.5  # Volume must be 1.5x average to confirm signals
        self.signal_history = {}  # Store recent signals for each pair
        self.max_history_size = 10
        
        # ML model parameters
        self.ml_models = {}
        self.scalers = {}    # Dictionary to store scalers for each pair
        self.ml_prediction_threshold = 0.65  # Confidence threshold for ML predictions
        self.ml_weight = 2.5  # Weight for ML predictions in final decision
        
        # Initialize models for each pair
        for pair in self.trading_pairs:
            self.active_pairs[pair] = None  # Initialize with no position
            self.signal_history[pair] = []
            model_path = f"{pair.replace('/', '_')}_ml_model.pkl"
            scaler_path = f"{pair.replace('/', '_')}_scaler.pkl"
            self._initialize_ml_model(pair, model_path, scaler_path)
        
        logger.info("Multi-Pair Trading Bot initialized successfully")

    def _initialize_ml_model(self, pair, model_path, scaler_path):
        """Initialize or load the machine learning model for a specific pair"""
        try:
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                logger.info(f"Loading existing ML model and scaler for {pair}")
                self.ml_models[pair] = joblib.load(model_path)
                self.scalers[pair] = joblib.load(scaler_path)
            else:
                logger.info(f"No existing ML model found for {pair}. Will train a new one.")
                self._train_ml_model(pair, model_path, scaler_path)
        except Exception as e:
            logger.error(f"Error initializing ML model for {pair}: {e}")
            self.ml_models[pair] = None
            self.scalers[pair] = None

    def calculate_indicators(self, df):
        """Calculate technical indicators with improved error handling"""
        try:
            if df is None or df.empty:
                logger.error("Empty or None DataFrame provided to calculate_indicators")
                return None
                
            # Make a copy to avoid modifying the original
            df = df.copy()
            
            # RSI with multiple timeframes
            df['rsi'] = ta.rsi(df['close'], length=14)
            df['rsi_slow'] = ta.rsi(df['close'], length=21)
            
            # MACD - handle potential tuple return and None values
            try:
                # Check if close prices contain None values
                if df['close'].isna().any():
                    logger.warning("NaN values found in close prices, filling with forward fill")
                    df['close'] = df['close'].fillna(method='ffill')
                
                macd = ta.macd(df['close'])
                if isinstance(macd, tuple):
                    # If macd returns a tuple, extract the components
                    df['MACD_12_26_9'] = macd[0]
                    df['MACDs_12_26_9'] = macd[1]
                    df['MACDh_12_26_9'] = macd[2]
                else:
                    # If macd returns a DataFrame
                    df['MACD_12_26_9'] = macd['MACD_12_26_9']
                    df['MACDs_12_26_9'] = macd['MACDs_12_26_9']
                    df['MACDh_12_26_9'] = macd['MACDh_12_26_9']
                
                # Fill any NaN values in MACD columns
                df['MACD_12_26_9'] = df['MACD_12_26_9'].fillna(0)
                df['MACDs_12_26_9'] = df['MACDs_12_26_9'].fillna(0)
                df['MACDh_12_26_9'] = df['MACDh_12_26_9'].fillna(0)
            except Exception as e:
                logger.warning(f"Error calculating MACD: {e}")
                # Set default values
                df['MACD_12_26_9'] = 0
                df['MACDs_12_26_9'] = 0
                df['MACDh_12_26_9'] = 0
            
            # Bollinger Bands - handle potential tuple return
            try:
                # Calculate manually to avoid pandas_ta issues
                sma_20 = df['close'].rolling(window=20).mean()
                std_20 = df['close'].rolling(window=20).std()
                df['BBL_20_2.0'] = sma_20 - (std_20 * 2)
                df['BBM_20_2.0'] = sma_20
                df['BBU_20_2.0'] = sma_20 + (std_20 * 2)
                
                # Fill NaN values
                df['BBL_20_2.0'] = df['BBL_20_2.0'].fillna(df['close'])
                df['BBM_20_2.0'] = df['BBM_20_2.0'].fillna(df['close'])
                df['BBU_20_2.0'] = df['BBU_20_2.0'].fillna(df['close'])
            except Exception as e:
                logger.warning(f"Error calculating Bollinger Bands: {e}")
                # Set default values
                df['BBL_20_2.0'] = df['close']
                df['BBM_20_2.0'] = df['close']
                df['BBU_20_2.0'] = df['close']
            
            # Additional Bollinger Bands with different settings
            try:
                # Calculate manually to avoid pandas_ta issues
                sma_50 = df['close'].rolling(window=50).mean()
                std_50 = df['close'].rolling(window=50).std()
                df['BBL_50_2.0'] = sma_50 - (std_50 * 2)
                df['BBM_50_2.0'] = sma_50
                df['BBU_50_2.0'] = sma_50 + (std_50 * 2)
                
                # Fill NaN values
                df['BBL_50_2.0'] = df['BBL_50_2.0'].fillna(df['close'])
                df['BBM_50_2.0'] = df['BBM_50_2.0'].fillna(df['close'])
                df['BBU_50_2.0'] = df['BBU_50_2.0'].fillna(df['close'])
            except Exception as e:
                logger.warning(f"Error calculating Bollinger Bands 50: {e}")
                # Set default values
                df['BBL_50_2.0'] = df['close']
                df['BBM_50_2.0'] = df['close']
                df['BBU_50_2.0'] = df['close']
            
            # Moving Averages
            df['sma_20'] = ta.sma(df['close'], length=20)
            df['sma_50'] = ta.sma(df['close'], length=50)
            df['sma_200'] = ta.sma(df['close'], length=200)
            df['ema_9'] = ta.ema(df['close'], length=9)
            df['ema_21'] = ta.ema(df['close'], length=21)
            df['ema_50'] = ta.ema(df['close'], length=50)
            df['ema_200'] = ta.ema(df['close'], length=200)
            
            # Fill NaN values in moving averages
            for col in ['sma_20', 'sma_50', 'sma_200', 'ema_9', 'ema_21', 'ema_50', 'ema_200']:
                df[col] = df[col].fillna(df['close'])
            
            # Volume indicators
            df['volume_sma'] = ta.sma(df['volume'], length=20)
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['volume_ratio'] = df['volume_ratio'].fillna(1.0)  # Default to 1.0 if NaN
            
            # Trend indicators with fixed ADX calculation
            try:
                adx_indicator = ta.adx(df['high'], df['low'], df['close'], length=14)
                if isinstance(adx_indicator, tuple):
                    df['adx'] = adx_indicator[0]
                    df['dmi_plus'] = adx_indicator[1]
                    df['dmi_minus'] = adx_indicator[2]
                else:
                    # Check if the keys exist in the DataFrame
                    if 'ADX_14' in adx_indicator.columns:
                        df['adx'] = adx_indicator['ADX_14']
                        df['dmi_plus'] = adx_indicator['DMP_14']
                        df['dmi_minus'] = adx_indicator['DMN_14']
                    else:
                        # Try alternative column names
                        df['adx'] = adx_indicator['ADX']
                        df['dmi_plus'] = adx_indicator['DMP']
                        df['dmi_minus'] = adx_indicator['DMN']
                
                # Fill NaN values
                df['adx'] = df['adx'].fillna(0)
                df['dmi_plus'] = df['dmi_plus'].fillna(0)
                df['dmi_minus'] = df['dmi_minus'].fillna(0)
            except Exception as e:
                logger.warning(f"Error calculating ADX: {e}")
                # Set default values
                df['adx'] = 0
                df['dmi_plus'] = 0
                df['dmi_minus'] = 0
            
            # Stochastic RSI - handle potential tuple return
            try:
                stoch_rsi = ta.stochrsi(df['close'])
                if isinstance(stoch_rsi, tuple):
                    df['STOCHRSIk_14_14_3_3'] = stoch_rsi[0]
                    df['STOCHRSId_14_14_3_3'] = stoch_rsi[1]
                else:
                    # Check if the keys exist in the DataFrame
                    if 'STOCHRSIk_14_14_3_3' in stoch_rsi.columns:
                        df['STOCHRSIk_14_14_3_3'] = stoch_rsi['STOCHRSIk_14_14_3_3']
                        df['STOCHRSId_14_14_3_3'] = stoch_rsi['STOCHRSId_14_14_3_3']
                    else:
                        # Try alternative column names
                        df['STOCHRSIk_14_14_3_3'] = stoch_rsi['STOCHRSIk']
                        df['STOCHRSId_14_14_3_3'] = stoch_rsi['STOCHRSId']
                
                # Fill NaN values
                df['STOCHRSIk_14_14_3_3'] = df['STOCHRSIk_14_14_3_3'].fillna(50)
                df['STOCHRSId_14_14_3_3'] = df['STOCHRSId_14_14_3_3'].fillna(50)
            except Exception as e:
                logger.warning(f"Error calculating Stochastic RSI: {e}")
                # Set default values
                df['STOCHRSIk_14_14_3_3'] = 50
                df['STOCHRSId_14_14_3_3'] = 50
            
            # Ichimoku Cloud - handle potential tuple return
            try:
                # Calculate manually to avoid pandas_ta issues
                high_9 = df['high'].rolling(window=9).max()
                low_9 = df['low'].rolling(window=9).min()
                df['ISA_9'] = (high_9 + low_9) / 2
                
                high_26 = df['high'].rolling(window=26).max()
                low_26 = df['low'].rolling(window=26).min()
                df['ISB_26'] = (high_26 + low_26) / 2
                
                df['ITS_9'] = df['ISA_9'].shift(26)
                df['IKS_26'] = df['ISB_26'].shift(26)
                df['ICS_26'] = (df['high'] + df['low']) / 2
                
                # Fill NaN values
                for col in ['ISA_9', 'ISB_26', 'ITS_9', 'IKS_26', 'ICS_26']:
                    df[col] = df[col].fillna(df['close'])
            except Exception as e:
                logger.warning(f"Error calculating Ichimoku Cloud: {e}")
                # Set default values
                df['ISA_9'] = df['close']
                df['ISB_26'] = df['close']
                df['ITS_9'] = df['close']
                df['IKS_26'] = df['close']
                df['ICS_26'] = df['close']
            
            # Fibonacci Retracement levels - handle potential None values
            try:
                recent_high = df['high'].rolling(window=20).max().iloc[-1]
                recent_low = df['low'].rolling(window=20).min().iloc[-1]
                
                if pd.isna(recent_high) or pd.isna(recent_low):
                    logger.warning("NaN values in high/low for Fibonacci calculation")
                    recent_high = df['high'].max()
                    recent_low = df['low'].min()
                
                price_range = recent_high - recent_low
                
                df['fib_0.236'] = recent_low + price_range * 0.236
                df['fib_0.382'] = recent_low + price_range * 0.382
                df['fib_0.5'] = recent_low + price_range * 0.5
                df['fib_0.618'] = recent_low + price_range * 0.618
                df['fib_0.786'] = recent_low + price_range * 0.786
                
                # Fill NaN values
                for col in ['fib_0.236', 'fib_0.382', 'fib_0.5', 'fib_0.618', 'fib_0.786']:
                    df[col] = df[col].fillna(df['close'])
            except Exception as e:
                logger.error(f"Error calculating Fibonacci levels: {e}")
                # Set default values to avoid None errors
                df['fib_0.236'] = df['close'].mean() * 0.95
                df['fib_0.382'] = df['close'].mean() * 0.97
                df['fib_0.5'] = df['close'].mean()
                df['fib_0.618'] = df['close'].mean() * 1.03
                df['fib_0.786'] = df['close'].mean() * 1.05
            
            # ATR for volatility - handle potential tuple return
            try:
                # Calculate manually to avoid pandas_ta issues
                high_low = df['high'] - df['low']
                high_close = (df['high'] - df['close'].shift()).abs()
                low_close = (df['low'] - df['close'].shift()).abs()
                ranges = pd.concat([high_low, high_close, low_close], axis=1)
                true_range = ranges.max(axis=1)
                df['atr'] = true_range.rolling(window=14).mean()
                
                # Fill NaN values
                df['atr'] = df['atr'].fillna(df['close'].rolling(window=14).std())
            except Exception as e:
                logger.warning(f"Error calculating ATR: {e}")
                # Set default values
                df['atr'] = df['close'].rolling(window=14).std()
                df['atr'] = df['atr'].fillna(df['close'].std())
            
            # Price changes
            df['price_change'] = df['close'].pct_change()
            df['price_change_1h'] = df['close'].pct_change(periods=1)
            df['price_change_4h'] = df['close'].pct_change(periods=4)
            df['price_change_1d'] = df['close'].pct_change(periods=24)
            
            # Fill NaN values in price changes
            for col in ['price_change', 'price_change_1h', 'price_change_4h', 'price_change_1d']:
                df[col] = df[col].fillna(0)
            
            # Volatility
            df['volatility'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean()
            df['volatility'] = df['volatility'].fillna(0)
            
            # Fill NaN values with 0 to avoid None errors
            df = df.fillna(0)
            
            return df
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return None

    def determine_market_context(self, df_1h, df_4h, df_1d):
        """Determine overall market context and trend with improved error handling"""
        try:
            if df_1h is None or df_4h is None or df_1d is None:
                return "unknown", 0
                
            if 'sma_50' not in df_1h.columns:
                logger.warning("Required indicators not found in dataframe")
                return "unknown", 0
            
            # Determine trend on different timeframes
            trend_1h = self._get_trend(df_1h)
            trend_4h = self._get_trend(df_4h)
            trend_1d = self._get_trend(df_1d)
            
            # Calculate trend strength (0-1)
            trend_strength = self._calculate_trend_strength(df_1h, df_4h, df_1d)
            
            # Determine overall market context
            if trend_1d == "bullish" and trend_4h == "bullish" and trend_1h == "bullish":
                return "strong_bullish", trend_strength
            elif trend_1d == "bearish" and trend_4h == "bearish" and trend_1h == "bearish":
                return "strong_bearish", trend_strength
            elif trend_1d == "bullish" and (trend_4h == "bullish" or trend_1h == "bullish"):
                return "bullish", trend_strength
            elif trend_1d == "bearish" and (trend_4h == "bearish" or trend_1h == "bearish"):
                return "bearish", trend_strength
            else:
                return "sideways", trend_strength
        except Exception as e:
            logger.error(f"Error in market context analysis: {e}")
            return "unknown", 0

    def analyze_signals(self, df_1h, df_4h, df_1d):
        """Analyze trading signals with improved error handling"""
        try:
            if df_1h is None or df_4h is None or df_1d is None:
                logger.warning("Missing data for signal analysis")
                return None
                
            signals = {
                'long': False,
                'short': False,
                'strength': 0,
                'stop_loss': None,
                'take_profit': None,
                'confidence': 0,
                'entry_window_low': None,
                'entry_window_high': None,
                'entry_window_hours': 1,
                'reason': ""
            }
            
            # Get the latest data points
            try:
                current_1h = df_1h.iloc[-1]
                current_4h = df_4h.iloc[-1]
                current_1d = df_1d.iloc[-1]
            except Exception as e:
                logger.error(f"Error accessing latest data points: {e}")
                return None
            
            # 1. Trend Analysis (40% weight)
            trend_score = 0
            max_trend_score = 3
            trend_reasons = []
            
            # EMA alignment check
            try:
                if (current_1h['ema_50'] > current_1h['ema_200'] and 
                    current_4h['ema_50'] > current_4h['ema_200'] and 
                    current_1d['ema_50'] > current_1d['ema_200']):
                    trend_score += 1
                    trend_reasons.append("EMA alignment bullish across timeframes")
            except Exception as e:
                logger.warning(f"Error in EMA alignment check: {e}")
            
            # ADX strength check
            try:
                if current_1h['adx'] > 25 and current_4h['adx'] > 25:
                    trend_score += 1
                    trend_reasons.append("Strong trend (ADX > 25)")
            except Exception as e:
                logger.warning(f"Error in ADX strength check: {e}")
            
            # Price above EMAs
            try:
                if (current_1h['close'] > current_1h['ema_50'] and 
                    current_4h['close'] > current_4h['ema_50']):
                    trend_score += 1
                    trend_reasons.append("Price above EMAs")
            except Exception as e:
                logger.warning(f"Error in price above EMAs check: {e}")
            
            # 2. Momentum Analysis (30% weight)
            momentum_score = 0
            max_momentum_score = 3
            momentum_reasons = []
            
            # RSI conditions
            try:
                if (30 < current_1h['rsi'] < 70 and 
                    30 < current_4h['rsi'] < 70):
                    momentum_score += 1
                    momentum_reasons.append("RSI in healthy range")
            except Exception as e:
                logger.warning(f"Error in RSI conditions check: {e}")
            
            # MACD conditions
            try:
                if (current_1h['MACD_12_26_9'] > current_1h['MACDs_12_26_9'] and 
                    current_4h['MACD_12_26_9'] > current_4h['MACDs_12_26_9']):
                    momentum_score += 1
                    momentum_reasons.append("MACD bullish crossover")
            except Exception as e:
                logger.warning(f"Error in MACD conditions check: {e}")
            
            # 3. Volatility and Volume Analysis (30% weight)
            volatility_score = 0
            max_volatility_score = 3
            volatility_reasons = []
            
            # Bollinger Bands squeeze
            try:
                bb_width_1h = (current_1h['BBU_20_2.0'] - current_1h['BBL_20_2.0']) / current_1h['BBM_20_2.0']
                bb_width_4h = (current_4h['BBU_20_2.0'] - current_4h['BBL_20_2.0']) / current_4h['BBM_20_2.0']
                
                if bb_width_1h < 0.1 and bb_width_4h < 0.1:
                    volatility_score += 1
                    volatility_reasons.append("Bollinger Band squeeze (potential breakout)")
            except Exception as e:
                logger.warning(f"Error in Bollinger Bands squeeze check: {e}")
            
            # Volume confirmation
            try:
                if (current_1h['volume_ratio'] > 1.5 and 
                    current_4h['volume_ratio'] > 1.2):
                    volatility_score += 1
                    volatility_reasons.append("Above average volume")
            except Exception as e:
                logger.warning(f"Error in volume confirmation check: {e}")
            
            # ATR for volatility check
            try:
                # Use a simple comparison instead of rolling
                if current_1h['atr'] > df_1h['atr'].mean():
                    volatility_score += 1
                    volatility_reasons.append("Increasing volatility")
            except Exception as e:
                logger.warning(f"Error in ATR volatility check: {e}")
            
            # Calculate final scores
            trend_weight = 0.4
            momentum_weight = 0.3
            volatility_weight = 0.3
            
            final_score = (
                (trend_score / max_trend_score) * trend_weight +
                (momentum_score / max_momentum_score) * momentum_weight +
                (volatility_score / max_volatility_score) * volatility_weight
            )
            
            # Generate trading signals
            if final_score >= 0.7:  # High confidence threshold
                signals['long'] = True
                signals['confidence'] = final_score
                
                # Calculate stop loss and take profit
                try:
                    atr_multiplier = 2.0
                    current_price = current_1h['close']
                    signals['stop_loss'] = current_price - (current_1h['atr'] * atr_multiplier)
                    signals['take_profit'] = current_price + (current_1h['atr'] * atr_multiplier * 2)
                    
                    # Calculate entry window
                    recent_low = df_1h['low'].iloc[-5:].min()
                    recent_high = df_1h['high'].iloc[-5:].max()
                    
                    signals['entry_window_low'] = recent_low
                    signals['entry_window_high'] = current_price
                except Exception as e:
                    logger.warning(f"Error calculating stop loss and take profit: {e}")
                    # Set default values
                    current_price = current_1h['close']
                    signals['stop_loss'] = current_price * 0.95
                    signals['take_profit'] = current_price * 1.1
                    signals['entry_window_low'] = current_price * 0.98
                    signals['entry_window_high'] = current_price
                
                # Combine reasons
                all_reasons = trend_reasons + momentum_reasons + volatility_reasons
                signals['reason'] = ", ".join(all_reasons)
                
                # Adjust entry window hours based on volatility
                try:
                    volatility_factor = current_1h['atr'] / current_1h['close']
                    if volatility_factor > 0.03:
                        signals['entry_window_hours'] = 0.5
                    elif volatility_factor > 0.02:
                        signals['entry_window_hours'] = 1
                    else:
                        signals['entry_window_hours'] = 2
                except Exception as e:
                    logger.warning(f"Error adjusting entry window hours: {e}")
                    signals['entry_window_hours'] = 1
            
            return signals
            
        except Exception as e:
            logger.error(f"Error analyzing signals: {e}")
            return None

    async def run(self):
        """Main bot loop with improved error handling and health checks"""
        logger.info("Starting Multi-Pair Trading Bot")
        try:
            await self.send_telegram_message("🚀 Multi-Pair Trading Bot started!")
        except Exception as e:
            logger.error(f"Error sending startup message: {e}")
            
        last_training_time = datetime.now()
        
        while True:
            try:
                # Health check
                if not self.exchange or not self.telegram_bot:
                    logger.error("Critical services unavailable")
                    await asyncio.sleep(60)
                    continue
                
                # Check if it's time to retrain the model
                current_time = datetime.now()
                if (current_time - last_training_time).days >= 7:
                    logger.info("Scheduling model retraining for all pairs")
                    for pair in self.trading_pairs:
                        model_path = f"{pair.replace('/', '_')}_ml_model.pkl"
                        scaler_path = f"{pair.replace('/', '_')}_scaler.pkl"
                        self._train_ml_model(pair, model_path, scaler_path)
                    last_training_time = current_time
                
                # Analyze each trading pair
                for pair in self.trading_pairs:
                    try:
                        # Skip pairs we already have a position in
                        if self.active_pairs[pair] is not None:
                            continue
                        
                        # Fetch and analyze data
                        df_1h, df_4h, df_1d = self.fetch_ohlcv_data(pair)
                        if df_1h is not None and df_4h is not None and df_1d is not None:
                            # Calculate indicators for all timeframes
                            df_1h = self.calculate_indicators(df_1h)
                            df_4h = self.calculate_indicators(df_4h)
                            df_1d = self.calculate_indicators(df_1d)
                            
                            if df_1h is None or df_4h is None or df_1d is None:
                                logger.warning(f"Failed to calculate indicators for {pair}")
                                continue
                            
                            # Analyze signals
                            signals = self.analyze_signals(df_1h, df_4h, df_1d)
                            if signals:
                                decision, message = self.generate_trading_decision(signals)
                                
                                if decision:
                                    # Update position
                                    self.active_pairs[pair] = 'long' if decision == 'BUY' else 'short'
                                    
                                    # Send notification with detailed analysis
                                    await self.send_telegram_message(
                                        f"🚨 {pair} Conversion Alert 🚨\n{message}\n"
                                        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                                    )
                                    
                                    # Log the signal
                                    logger.info(f"Trading signal for {pair}: {decision} with confidence {signals['confidence']:.2f}")
                    
                    except Exception as e:
                        logger.error(f"Error processing pair {pair}: {e}")
                        continue
                
                # Wait for 5 minutes before next analysis
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                try:
                    await self.send_telegram_message(f"⚠️ Bot Error: {str(e)}")
                except:
                    pass
                await asyncio.sleep(60)

    def fetch_ohlcv_data(self, pair):
        """Fetch OHLCV data with improved error handling and rate limiting"""
        try:
            # Add exponential backoff for rate limiting
            max_retries = 3
            retry_delay = 5
            
            # Fetch data for multiple timeframes
            df_1h = None
            df_4h = None
            df_1d = None
            
            # Fetch 1h data
            for attempt in range(max_retries):
                try:
                    ohlcv_1h = self.exchange.fetch_ohlcv(
                        symbol=pair,
                        timeframe='1h',
                        limit=100
                    )
                    
                    if not ohlcv_1h:
                        logger.warning(f"No 1h data returned for {pair}")
                        break
                        
                    # Convert to DataFrame
                    df_1h = pd.DataFrame(
                        ohlcv_1h,
                        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    )
                    df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'], unit='ms')
                    df_1h.set_index('timestamp', inplace=True)
                    break
                    
                except ccxt.RateLimitExceeded:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        logger.warning(f"Rate limit exceeded for {pair} 1h data, waiting {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Max retries reached for {pair} 1h data due to rate limiting")
                        break
                        
                except ccxt.NetworkError as e:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        logger.warning(f"Network error for {pair} 1h data: {str(e)}, retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Max retries reached for {pair} 1h data due to network errors")
                        break
            
            # Fetch 4h data
            for attempt in range(max_retries):
                try:
                    ohlcv_4h = self.exchange.fetch_ohlcv(
                        symbol=pair,
                        timeframe='4h',
                        limit=50
                    )
                    
                    if not ohlcv_4h:
                        logger.warning(f"No 4h data returned for {pair}")
                        break
                        
                    # Convert to DataFrame
                    df_4h = pd.DataFrame(
                        ohlcv_4h,
                        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    )
                    df_4h['timestamp'] = pd.to_datetime(df_4h['timestamp'], unit='ms')
                    df_4h.set_index('timestamp', inplace=True)
                    break
                    
                except ccxt.RateLimitExceeded:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        logger.warning(f"Rate limit exceeded for {pair} 4h data, waiting {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Max retries reached for {pair} 4h data due to rate limiting")
                        break
                        
                except ccxt.NetworkError as e:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        logger.warning(f"Network error for {pair} 4h data: {str(e)}, retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Max retries reached for {pair} 4h data due to network errors")
                        break
            
            # Fetch 1d data
            for attempt in range(max_retries):
                try:
                    ohlcv_1d = self.exchange.fetch_ohlcv(
                        symbol=pair,
                        timeframe='1d',
                        limit=30
                    )
                    
                    if not ohlcv_1d:
                        logger.warning(f"No 1d data returned for {pair}")
                        break
                        
                    # Convert to DataFrame
                    df_1d = pd.DataFrame(
                        ohlcv_1d,
                        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    )
                    df_1d['timestamp'] = pd.to_datetime(df_1d['timestamp'], unit='ms')
                    df_1d.set_index('timestamp', inplace=True)
                    break
                    
                except ccxt.RateLimitExceeded:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        logger.warning(f"Rate limit exceeded for {pair} 1d data, waiting {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Max retries reached for {pair} 1d data due to rate limiting")
                        break
                        
                except ccxt.NetworkError as e:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        logger.warning(f"Network error for {pair} 1d data: {str(e)}, retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Max retries reached for {pair} 1d data due to network errors")
                        break
            
            # Return all dataframes
            return df_1h, df_4h, df_1d
                        
        except Exception as e:
            logger.error(f"Error fetching OHLCV data for {pair}: {str(e)}")
            return None, None, None

    async def send_telegram_message(self, message: str, max_retries: int = 3, retry_delay: int = 5):
        """
        Send a message to Telegram with retry logic
        
        Args:
            message (str): The message to send
            max_retries (int): Maximum number of retry attempts
            retry_delay (int): Delay between retries in seconds
        """
        if not self.telegram_bot or not self.chat_id:
            logger.error("Telegram configuration is missing")
            return
            
        for attempt in range(max_retries):
            try:
                await self.telegram_bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                    parse_mode='Markdown'
                )
                logger.info(f"Telegram message sent successfully: {message[:100]}...")
                return
            except telegram.error.TimedOut:
                if attempt < max_retries - 1:
                    logger.warning(f"Telegram request timed out. Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error("Failed to send Telegram message after maximum retries")
            except telegram.error.NetworkError as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Network error: {e}. Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"Network error persists after maximum retries: {e}")
            except telegram.error.BadRequest as e:
                logger.error(f"Bad request error (check chat_id and message format): {e}")
                break  # Don't retry for bad requests
            except Exception as e:
                logger.error(f"Unexpected error sending Telegram message: {e}")
                break  # Don't retry for unexpected errors
                
    def send_trading_signal(self, pair: str, signal_type: str, confidence: float, entry_price: float, 
                           stop_loss: float, take_profit: float, reasons: List[str]):
        """
        Format and send a trading signal message via Telegram
        
        Args:
            pair (str): Trading pair (e.g., 'SOL/USDT')
            signal_type (str): Type of signal ('buy' or 'sell')
            confidence (float): Signal confidence score (0-1)
            entry_price (float): Suggested entry price
            stop_loss (float): Suggested stop loss price
            take_profit (float): Suggested take profit price
            reasons (List[str]): List of reasons for the signal
        """
        # Format the confidence score as a percentage
        confidence_pct = round(confidence * 100, 2)
        
        # Calculate potential profit and loss percentages
        risk_pct = abs(round((stop_loss - entry_price) / entry_price * 100, 2))
        reward_pct = abs(round((take_profit - entry_price) / entry_price * 100, 2))
        
        # Create the message with emojis and formatting
        message = f"""
🚨 *New Trading Signal*
{'🟢 BUY' if signal_type.lower() == 'buy' else '🔴 SELL'} *{pair}*

*Confidence:* {'⭐' * int(confidence_pct/20)}  ({confidence_pct}%)
*Entry Window:* 30-60 minutes

💰 *Entry Zone:* {entry_price:.4f}
🛑 *Stop Loss:* {stop_loss:.4f} ({risk_pct}%)
🎯 *Take Profit:* {take_profit:.4f} ({reward_pct}%)

📊 *Risk/Reward Ratio:* 1:{round(reward_pct/risk_pct, 2)}

*Signal Reasons:*
"""
        # Add each reason with a bullet point
        for reason in reasons:
            message += f"• {reason}\n"
            
        message += "\n⚠️ *Always manage your risk and do your own research*"
        
        # Send the message asynchronously
        asyncio.get_event_loop().run_until_complete(self.send_telegram_message(message))
        
    def format_error_message(self, error_type: str, details: str):
        """
        Format an error message for Telegram notification
        
        Args:
            error_type (str): Type of error
            details (str): Error details
        """
        message = f"""
⚠️ *Trading Bot Alert*
*Type:* {error_type}
*Details:* {details}
*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        # Send the message asynchronously
        asyncio.get_event_loop().run_until_complete(self.send_telegram_message(message))

    def _train_ml_model(self, pair, model_path, scaler_path):
        """Train the machine learning model for a specific pair"""
        try:
            logger.info(f"Starting ML model training for {pair}")
            
            # Fetch historical data
            historical_data = self.fetch_historical_data_for_training(pair, days=60)  # 60 days of data
            if historical_data is None or len(historical_data) < 100:
                logger.error(f"Not enough historical data for {pair} ML training")
                return
            
            # Calculate indicators
            df = self.calculate_indicators(historical_data)
            if df is None:
                logger.error(f"Failed to calculate indicators for {pair} ML training")
                return
            
            # Prepare features
            X, y, feature_columns = self.prepare_ml_features(df)
            
            # Split data into training and validation sets
            train_size = int(len(X) * 0.8)
            X_train, X_val = X[:train_size], X[train_size:]
            y_train, y_val = y[:train_size], y[train_size:]
            
            # Scale features
            self.scalers[pair] = StandardScaler()
            X_train_scaled = self.scalers[pair].fit_transform(X_train)
            X_val_scaled = self.scalers[pair].transform(X_val)
            
            # Train model
            self.ml_models[pair] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            
            self.ml_models[pair].fit(X_train_scaled, y_train)
            
            # Evaluate model
            train_accuracy = self.ml_models[pair].score(X_train_scaled, y_train)
            val_accuracy = self.ml_models[pair].score(X_val_scaled, y_val)
            
            logger.info(f"ML model trained for {pair}. Train accuracy: {train_accuracy:.4f}, Validation accuracy: {val_accuracy:.4f}")
            
            # Save model and scaler
            joblib.dump(self.ml_models[pair], model_path)
            joblib.dump(self.scalers[pair], scaler_path)
            
            logger.info(f"ML model and scaler saved for {pair}")
            
        except Exception as e:
            logger.error(f"Error training ML model for {pair}: {e}")
            self.ml_models[pair] = None
            self.scalers[pair] = None

    def fetch_historical_data_for_training(self, pair, days=30):
        """Fetch historical data for training the ML model for a specific pair"""
        try:
            logger.info(f"Fetching {days} days of historical data for {pair} ML training")
            
            # Calculate timestamps
            end_time = int(time.time() * 1000)
            start_time = end_time - (days * 24 * 60 * 60 * 1000)
            
            # Fetch historical data
            historical_data = self.exchange.fetch_ohlcv(
                pair, 
                self.timeframe, 
                since=start_time,
                limit=1000
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(historical_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Fetched {len(df)} historical candles for {pair} ML training")
            return df
        except Exception as e:
            logger.error(f"Error fetching historical data for {pair}: {e}")
            return None

    def _get_trend(self, df):
        """
        Determine the trend direction based on multiple indicators
        
        Args:
            df (pd.DataFrame): DataFrame with price data and indicators
            
        Returns:
            str: 'bullish', 'bearish', or 'sideways'
        """
        try:
            # Get the latest data point
            current = df.iloc[-1]
            
            # Check EMA alignment
            ema_bullish = current['ema_9'] > current['ema_21'] > current['sma_50']
            ema_bearish = current['ema_9'] < current['ema_21'] < current['sma_50']
            
            # Check MACD
            macd_bullish = current['MACD_12_26_9'] > current['MACDs_12_26_9']
            macd_bearish = current['MACD_12_26_9'] < current['MACDs_12_26_9']
            
            # Check ADX trend strength
            strong_trend = current['adx'] > 25
            
            # Determine trend
            if ema_bullish and macd_bullish and strong_trend:
                return "bullish"
            elif ema_bearish and macd_bearish and strong_trend:
                return "bearish"
            else:
                return "sideways"
                
        except Exception as e:
            logger.error(f"Error determining trend: {e}")
            return "sideways"
            
    def _calculate_trend_strength(self, df_1h, df_4h, df_1d):
        """
        Calculate the overall trend strength across timeframes
        
        Args:
            df_1h (pd.DataFrame): 1-hour timeframe data
            df_4h (pd.DataFrame): 4-hour timeframe data
            df_1d (pd.DataFrame): 1-day timeframe data
            
        Returns:
            float: Trend strength score between 0 and 1
        """
        try:
            # Get latest data points
            current_1h = df_1h.iloc[-1]
            current_4h = df_4h.iloc[-1]
            current_1d = df_1d.iloc[-1]
            
            # Calculate ADX-based strength (0-1)
            adx_strength_1h = min(current_1h['adx'] / 100, 1)
            adx_strength_4h = min(current_4h['adx'] / 100, 1)
            adx_strength_1d = min(current_1d['adx'] / 100, 1)
            
            # Calculate price alignment strength (0-1)
            price_above_ema_1h = current_1h['close'] > current_1h['ema_50']
            price_above_ema_4h = current_4h['close'] > current_4h['ema_50']
            price_above_ema_1d = current_1d['close'] > current_1d['ema_50']
            
            alignment_score = sum([price_above_ema_1h, price_above_ema_4h, price_above_ema_1d]) / 3
            
            # Calculate volume strength (0-1)
            volume_strength_1h = min(current_1h['volume_ratio'] / 2, 1)
            volume_strength_4h = min(current_4h['volume_ratio'] / 2, 1)
            volume_strength_1d = min(current_1d['volume_ratio'] / 2, 1)
            
            # Combine scores with weights
            adx_weight = 0.4
            alignment_weight = 0.3
            volume_weight = 0.3
            
            trend_strength = (
                (adx_strength_1h * 0.5 + adx_strength_4h * 0.3 + adx_strength_1d * 0.2) * adx_weight +
                alignment_score * alignment_weight +
                (volume_strength_1h * 0.5 + volume_strength_4h * 0.3 + volume_strength_1d * 0.2) * volume_weight
            )
            
            return min(max(trend_strength, 0), 1)  # Ensure result is between 0 and 1
            
        except Exception as e:
            logger.error(f"Error calculating trend strength: {e}")
            return 0

    def generate_trading_decision(self, signals):
        """
        Generate a trading decision based on the analyzed signals
        
        Args:
            signals (dict): Dictionary containing signal analysis results
            
        Returns:
            tuple: (decision, message) where decision is 'BUY', 'SELL', or None
        """
        try:
            if not signals:
                return None, "No valid signals generated"
                
            # Check if we have a strong enough signal
            if signals['confidence'] < 0.7:
                return None, f"Signal confidence too low: {signals['confidence']:.2f}"
                
            # Determine the trading decision
            if signals['long']:
                decision = 'BUY'
                message = (
                    f"Strong buy signal detected!\n"
                    f"Confidence: {signals['confidence']:.2%}\n"
                    f"Entry Window: {signals['entry_window_hours']} hours\n"
                    f"Entry Zone: {signals['entry_window_low']:.4f} - {signals['entry_window_high']:.4f}\n"
                    f"Stop Loss: {signals['stop_loss']:.4f}\n"
                    f"Take Profit: {signals['take_profit']:.4f}\n"
                    f"Reasons: {signals['reason']}"
                )
            elif signals['short']:
                decision = 'SELL'
                message = (
                    f"Strong sell signal detected!\n"
                    f"Confidence: {signals['confidence']:.2%}\n"
                    f"Entry Window: {signals['entry_window_hours']} hours\n"
                    f"Entry Zone: {signals['entry_window_high']:.4f} - {signals['entry_window_low']:.4f}\n"
                    f"Stop Loss: {signals['stop_loss']:.4f}\n"
                    f"Take Profit: {signals['take_profit']:.4f}\n"
                    f"Reasons: {signals['reason']}"
                )
            else:
                return None, "No clear trading direction"
                
            return decision, message
            
        except Exception as e:
            logger.error(f"Error generating trading decision: {e}")
            return None, f"Error in decision generation: {str(e)}"

    def prepare_ml_features(self, df):
        """
        Prepare features for ML model training
        
        Args:
            df (pd.DataFrame): DataFrame with price data and indicators
            
        Returns:
            tuple: (X, y, feature_columns) where X is the feature matrix, y is the target vector,
                  and feature_columns is the list of feature names
        """
        try:
            # Define features to use
            feature_columns = [
                'rsi', 'rsi_slow', 'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9',
                'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBL_50_2.0', 'BBM_50_2.0', 'BBU_50_2.0',
                'sma_20', 'sma_50', 'sma_200', 'ema_9', 'ema_21',
                'volume_ratio', 'adx', 'dmi_plus', 'dmi_minus',
                'STOCHRSIk_14_14_3_3', 'STOCHRSId_14_14_3_3',
                'ISA_9', 'ISB_26', 'ITS_9', 'IKS_26', 'ICS_26',
                'fib_0.236', 'fib_0.382', 'fib_0.5', 'fib_0.618', 'fib_0.786',
                'atr', 'volatility'
            ]
            
            # Create target variable (1 if price goes up in next period, 0 if down)
            df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
            
            # Drop rows with NaN values
            df = df.dropna()
            
            # Prepare feature matrix X and target vector y
            X = df[feature_columns]
            y = df['target']
            
            return X, y, feature_columns
            
        except Exception as e:
            logger.error(f"Error preparing ML features: {e}")
            return None, None, None

if __name__ == "__main__":
    try:
        bot = SolanaTradingBot()
        asyncio.run(bot.run())
    except Exception as e:
        logger.critical(f"Fatal error: {e}") 