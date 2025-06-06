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
from typing import List, Dict, Any
import signal
import sys
from utils.telegram_handler import TelegramCommandHandler

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
        self.ml_prediction_threshold = float(os.getenv('ML_PREDICTION_THRESHOLD', 0.5))
        self.signal_confidence_threshold = float(os.getenv('SIGNAL_CONFIDENCE_THRESHOLD', 0.5))
        self.ml_weight = 2.5  # Weight for ML predictions in final decision
        
        # Initialize models for each pair
        for pair in self.trading_pairs:
            self.active_pairs[pair] = None  # Initialize with no position
            self.signal_history[pair] = []
            model_path = f"{pair.replace('/', '_')}_ml_model.pkl"
            scaler_path = f"{pair.replace('/', '_')}_scaler.pkl"
            self._initialize_ml_model(pair, model_path, scaler_path)
        
        logger.info("Multi-Pair Trading Bot initialized successfully")
        
        self._is_running = False

        # Trading interval (in seconds) - default 5 minutes, can override with env var
        self.trading_interval_seconds = int(os.getenv('TRADING_INTERVAL_SECONDS', 300))
        # Retrain interval (in seconds) - default 6 hours
        self.retrain_interval_seconds = int(os.getenv('RETRAIN_INTERVAL_SECONDS', 21600))
        self.last_retrain_time = time.time()

        # Initialize Telegram command handler
        self.telegram_handler = TelegramCommandHandler(
            bot_token=os.getenv('TELEGRAM_BOT_TOKEN'),
            chat_id=os.getenv('TELEGRAM_CHAT_ID'),
            trading_bot=self
        )
        
        # Start the command handler
        self.telegram_handler.start()

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
            if df is None or len(df) < 200:  # Need at least 200 candles for all indicators
                logger.warning("Not enough data for indicator calculation")
                return None
                
            # Calculate EMAs
            df['ema_9'] = ta.ema(df['close'], length=9)
            df['ema_21'] = ta.ema(df['close'], length=21)
            df['ema_50'] = ta.ema(df['close'], length=50)
            df['ema_200'] = ta.ema(df['close'], length=200)
            
            # Calculate MACD with error handling
            try:
                macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
                if macd is not None:
                    df['macd'] = macd['MACD_12_26_9']
                    df['macd_signal'] = macd['MACDs_12_26_9']
                    df['macd_hist'] = macd['MACDh_12_26_9']
                else:
                    logger.warning("MACD calculation returned None")
                    df['macd'] = None
                    df['macd_signal'] = None
                    df['macd_hist'] = None
            except Exception as e:
                logger.warning(f"Error calculating MACD: {e}")
                df['macd'] = None
                df['macd_signal'] = None
                df['macd_hist'] = None
            
            # Calculate RSI
            df['rsi'] = ta.rsi(df['close'], length=14)
            
            # Calculate Bollinger Bands
            bb = ta.bbands(df['close'], length=20, std=2)
            df['bb_upper'] = bb['BBU_20_2.0']
            df['bb_middle'] = bb['BBM_20_2.0']
            df['bb_lower'] = bb['BBL_20_2.0']
            
            # Calculate ATR
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            
            # Calculate Stochastic RSI
            df['stoch_rsi'] = ta.stochrsi(df['close'], length=14)
            
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
                if (current_1h['macd'] > current_1h['macd_signal'] and 
                    current_4h['macd'] > current_4h['macd_signal']):
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
                bb_width_1h = (current_1h['bb_upper'] - current_1h['bb_lower']) / current_1h['bb_middle']
                bb_width_4h = (current_4h['bb_upper'] - current_4h['bb_lower']) / current_4h['bb_middle']
                
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
                    recent_low = df_1h['bb_lower'].iloc[-5:].min()
                    recent_high = df_1h['bb_upper'].iloc[-5:].max()
                    
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

    async def cleanup(self):
        """Cleanup resources before shutdown"""
        try:
            # Stop the Telegram command handler
            self.telegram_handler.stop()
            
            # Close exchange connection
            if hasattr(self, 'exchange'):
                await self.exchange.close()
                
            logger.info("Cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def run(self):
        """Main bot execution loop."""
        try:
            logger.info("Starting Solana trading bot...")
            self._is_running = True
            
            # Initialize exchange connection
            if not await self.initialize_exchange():
                logger.error("Failed to initialize exchange connection. Exiting...")
                return
                
            while self._is_running:
                try:
                    # Retrain ML models periodically
                    now = time.time()
                    if now - self.last_retrain_time > self.retrain_interval_seconds:
                        logger.info("Retraining ML models with latest data...")
                        for pair in self.trading_pairs:
                            model_path = f"{pair.replace('/', '_')}_ml_model.pkl"
                            scaler_path = f"{pair.replace('/', '_')}_scaler.pkl"
                            self._train_ml_model(pair, model_path, scaler_path)
                        self.last_retrain_time = now
                    # Ensure exchange connection is active
                    if not await self.ensure_exchange_connection():
                        logger.error("Lost exchange connection. Attempting to reconnect...")
                        await asyncio.sleep(5)
                        continue
                        
                    # Fetch market data
                    market_data = await self.fetch_market_data()
                    if not market_data:
                        logger.warning("Failed to fetch market data. Retrying...")
                        await asyncio.sleep(5)
                        continue
                        
                    # Process market data and execute trades
                    await self.process_market_data(market_data)
                    
                    # Sleep for the configured interval
                    await asyncio.sleep(self.trading_interval_seconds)
                    
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    await asyncio.sleep(5)  # Sleep before retrying
                    
        except Exception as e:
            logger.error(f"Critical error in bot execution: {e}")
        finally:
            await self.cleanup()

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
                # Create a new event loop for this operation if needed
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
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
        confidence_pct = round(confidence * 100, 2)
        risk_pct = abs(round((stop_loss - entry_price) / entry_price * 100, 2))
        reward_pct = abs(round((take_profit - entry_price) / entry_price * 100, 2))
        # Use the latest event time for the signal
        event_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if signal_type == 'WATCHLIST':
            message = f"\n👀 *Watchlist Signal*\n*{pair}*\n\n*Confidence:* {confidence_pct}% (Low)\n*Entry Window:* 30-60 minutes\n\n💰 *Entry Zone:* {entry_price:.4f}\n🛑 *Stop Loss:* {stop_loss:.4f} ({risk_pct}%)\n🎯 *Take Profit:* {take_profit:.4f} ({reward_pct}%)\n\n*Signal Reasons:*\n"
            for reason in reasons:
                message += f"• {reason}\n"
            message += f"\n_Time: {event_time}_\n⚠️ *Experimental signal. Use caution.*"
        else:
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
            for reason in reasons:
                message += f"• {reason}\n"
            message += f"\n_Time: {event_time}_\n⚠️ *Always manage your risk and do your own research*"
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
            
            # Add exponential backoff for rate limiting
            max_retries = 3
            retry_delay = 5
            
            for attempt in range(max_retries):
                try:
                    # Fetch historical data with rate limit handling
                    historical_data = self.exchange.fetch_ohlcv(
                        pair, 
                        self.timeframe, 
                        since=start_time,
                        limit=1000
                    )
                    
                    if not historical_data:
                        logger.warning(f"No historical data returned for {pair}")
                        if attempt < max_retries - 1:
                            wait_time = retry_delay * (2 ** attempt)
                            logger.warning(f"Retrying in {wait_time} seconds...")
                            time.sleep(wait_time)
                            continue
                        else:
                            logger.error(f"Failed to fetch historical data for {pair} after {max_retries} attempts")
                            return None
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(historical_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    
                    # Validate data
                    if len(df) < 100:  # Need at least 100 candles for training
                        logger.warning(f"Not enough historical data for {pair}: {len(df)} candles")
                        return None
                    
                    logger.info(f"Fetched {len(df)} historical candles for {pair} ML training")
                    return df
                    
                except ccxt.RateLimitExceeded:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        logger.warning(f"Rate limit exceeded for {pair}, waiting {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Max retries reached for {pair} due to rate limiting")
                        return None
                    
                except ccxt.NetworkError as e:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        logger.warning(f"Network error for {pair}: {str(e)}, retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Max retries reached for {pair} due to network errors")
                        return None
                    
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
            macd_bullish = current['macd'] > current['macd_signal']
            macd_bearish = current['macd'] < current['macd_signal']
            
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
            tuple: (decision, message) where decision is 'BUY', 'SELL', 'WATCHLIST', or None
        """
        try:
            if not signals:
                return None, "No valid signals generated"
            # Use the new, lower confidence threshold
            if signals['confidence'] < self.signal_confidence_threshold:
                return 'WATCHLIST', f"Watchlist signal (low confidence: {signals['confidence']:.2f})"
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

    async def initialize_exchange(self):
        """Initialize exchange connection with retry logic."""
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                self.exchange = ccxt.binance({
                    'apiKey': self.config['exchange']['api_key'],
                    'secret': self.config['exchange']['api_secret'],
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'future',
                        'adjustForTimeDifference': True
                    }
                })
                
                # Test connection
                await self.exchange.load_markets()
                logger.info("Exchange connection established successfully")
                return True
                
            except ccxt.NetworkError as e:
                logger.error(f"Network error connecting to exchange (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    logger.critical("Failed to connect to exchange after maximum retries")
                    return False
                    
            except ccxt.ExchangeError as e:
                logger.error(f"Exchange error: {e}")
                return False
                
            except Exception as e:
                logger.error(f"Unexpected error connecting to exchange: {e}")
                return False
                
    async def ensure_exchange_connection(self):
        """Ensure exchange connection is active, reconnect if necessary."""
        if not hasattr(self, 'exchange') or not self.exchange:
            return await self.initialize_exchange()
            
        try:
            # Test connection with a simple API call
            await self.exchange.fetch_balance()
            return True
        except Exception as e:
            logger.warning(f"Exchange connection lost, attempting to reconnect: {e}")
            return await self.initialize_exchange()

    async def get_status(self) -> Dict[str, Any]:
        """Get current bot status"""
        return {
            'status': 'running' if self._is_running else 'stopped',
            'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'position': self.active_pairs.get('SOL/USDT', 'No position'),
            'entry_price': self.active_pairs.get('SOL/USDT', {}).get('entry_price', 0),
            'pnl': self.active_pairs.get('SOL/USDT', {}).get('pnl', 0),
            'active_signals': len(self.signal_history.get('SOL/USDT', []))
        }
        
    async def get_current_price_analysis(self) -> Dict[str, Any]:
        """Get current price and analysis"""
        try:
            # Fetch current data
            df_1h = self.fetch_ohlcv_data('SOL/USDT')
            if df_1h is None:
                raise Exception("Failed to fetch price data")
                
            # Calculate indicators
            df_1h = self.calculate_indicators(df_1h)
            
            # Get current price
            current_price = df_1h['close'].iloc[-1]
            
            # Calculate 24h change
            price_24h_ago = df_1h['close'].iloc[-24]
            change_24h = ((current_price - price_24h_ago) / price_24h_ago) * 100
            
            # Get volume
            volume_24h = df_1h['volume'].iloc[-24:].sum()
            
            # Get technical indicators
            rsi = df_1h['rsi_14'].iloc[-1]
            macd = "Bullish" if df_1h['macd'].iloc[-1] > df_1h['macd_signal'].iloc[-1] else "Bearish"
            trend = "Bullish" if df_1h['sma_50'].iloc[-1] > df_1h['sma_200'].iloc[-1] else "Bearish"
            
            # Calculate support and resistance
            support = df_1h['low'].iloc[-20:].min()
            resistance = df_1h['high'].iloc[-20:].max()
            
            return {
                'price': current_price,
                'change_24h': change_24h,
                'volume_24h': volume_24h,
                'rsi': rsi,
                'macd': macd,
                'trend': trend,
                'support': support,
                'resistance': resistance
            }
        except Exception as e:
            logger.error(f"Error getting price analysis: {e}")
            raise
            
    async def get_recent_signals(self) -> List[Dict[str, Any]]:
        """Get recent trading signals"""
        signals = []
        for signal in self.signal_history.get('SOL/USDT', [])[-5:]:  # Last 5 signals
            signals.append({
                'type': signal['type'],
                'time': signal['time'].strftime('%Y-%m-%d %H:%M:%S'),
                'price': signal['price'],
                'confidence': signal['confidence'],
                'status': signal['status']
            })
        return signals
        
    async def get_settings(self) -> Dict[str, Any]:
        """Get current bot settings"""
        return {
            'trading_interval': self.trading_interval_seconds,
            'ml_threshold': self.ml_prediction_threshold,
            'signal_threshold': self.signal_confidence_threshold,
            'retrain_interval': self.retrain_interval_seconds,
            'risk_reward_ratio': 1.5,  # Fixed for now
            'max_position_size': 100  # Fixed for now
        }
        
    async def get_market_analysis(self) -> str:
        """Get current market analysis"""
        try:
            df_1h, df_4h, df_1d = self.fetch_ohlcv_data('SOL/USDT')
            if df_1h is None or df_4h is None or df_1d is None:
                raise Exception("Failed to fetch market data")
                
            # Calculate market context
            context, strength = self.determine_market_context(df_1h, df_4h, df_1d)
            
            # Get current price
            current_price = df_1h['close'].iloc[-1]
            
            # Format analysis
            analysis = (
                f"📊 Market Analysis\n\n"
                f"Current Price: ${current_price:.2f}\n"
                f"Market Context: {context}\n"
                f"Trend Strength: {strength:.1%}\n\n"
                f"Timeframe Analysis:\n"
                f"1h: {self._get_trend(df_1h)}\n"
                f"4h: {self._get_trend(df_4h)}\n"
                f"1d: {self._get_trend(df_1d)}\n\n"
                f"Key Levels:\n"
                f"Support: ${df_1h['low'].iloc[-20:].min():.2f}\n"
                f"Resistance: ${df_1h['high'].iloc[-20:].max():.2f}"
            )
            
            return analysis
        except Exception as e:
            logger.error(f"Error getting market analysis: {e}")
            return "❌ Error fetching market analysis. Please try again later."
            
    async def get_latest_signal(self) -> str:
        """Get the latest trading signal"""
        try:
            signals = self.signal_history.get('SOL/USDT', [])
            if not signals:
                return "No recent signals available."
                
            latest = signals[-1]
            signal_message = (
                f"📈 Latest Signal\n\n"
                f"Type: {latest['type']}\n"
                f"Time: {latest['time'].strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Price: ${latest['price']:.2f}\n"
                f"Confidence: {latest['confidence']}%\n"
                f"Status: {latest['status']}\n\n"
                f"Reasons:\n"
            )
            
            for reason in latest.get('reasons', []):
                signal_message += f"• {reason}\n"
                
            return signal_message
        except Exception as e:
            logger.error(f"Error getting latest signal: {e}")
            return "❌ Error fetching latest signal. Please try again later."

if __name__ == "__main__":
    try:
        # Set up signal handlers for graceful shutdown
        def signal_handler(sig, frame):
            logger.info("Shutting down gracefully...")
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Create and run the bot
        bot = SolanaTradingBot()
        
        # Get or create event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        # Run the bot
        loop.run_until_complete(bot.run())
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        sys.exit(1) 