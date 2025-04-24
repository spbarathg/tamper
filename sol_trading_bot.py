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
        
        # Initialize Telegram bot
        try:
            self.telegram_bot = telegram.Bot(token=os.getenv('TELEGRAM_BOT_TOKEN'))
            self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
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
            self.active_pairs[pair] = None  # No position
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
            # RSI with multiple timeframes
            df['rsi'] = ta.rsi(df['close'], length=14)
            df['rsi_slow'] = ta.rsi(df['close'], length=21)
            
            # MACD with multiple settings
            macd = ta.macd(df['close'])
            df = pd.concat([df, macd], axis=1)
            
            # Bollinger Bands with multiple settings
            bollinger = ta.bbands(df['close'])
            df = pd.concat([df, bollinger], axis=1)
            
            # Additional Bollinger Bands with different settings
            bollinger_50 = ta.bbands(df['close'], length=50)
            df = pd.concat([df, bollinger_50], axis=1)
            
            # Moving Averages
            df['sma_20'] = ta.sma(df['close'], length=20)
            df['sma_50'] = ta.sma(df['close'], length=50)
            df['sma_200'] = ta.sma(df['close'], length=200)
            df['ema_9'] = ta.ema(df['close'], length=9)
            df['ema_21'] = ta.ema(df['close'], length=21)
            
            # Volume indicators
            df['volume_sma'] = ta.sma(df['volume'], length=20)
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Trend indicators with fixed ADX calculation
            adx_indicator = ta.adx(df['high'], df['low'], df['close'], length=14)
            df['adx'] = adx_indicator['ADX_14']
            df['dmi_plus'] = adx_indicator['DMP_14']
            df['dmi_minus'] = adx_indicator['DMN_14']
            
            # Stochastic RSI
            stoch_rsi = ta.stochrsi(df['close'])
            df = pd.concat([df, stoch_rsi], axis=1)
            
            # Ichimoku Cloud
            ichimoku = ta.ichimoku(df['high'], df['low'], df['close'])
            df = pd.concat([df, ichimoku], axis=1)
            
            # Fibonacci Retracement levels
            recent_high = df['high'].rolling(window=20).max().iloc[-1]
            recent_low = df['low'].rolling(window=20).min().iloc[-1]
            price_range = recent_high - recent_low
            
            df['fib_0.236'] = recent_low + price_range * 0.236
            df['fib_0.382'] = recent_low + price_range * 0.382
            df['fib_0.5'] = recent_low + price_range * 0.5
            df['fib_0.618'] = recent_low + price_range * 0.618
            df['fib_0.786'] = recent_low + price_range * 0.786
            
            # ATR for volatility
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            
            # Price changes
            df['price_change'] = df['close'].pct_change()
            df['price_change_1h'] = df['close'].pct_change(periods=1)
            df['price_change_4h'] = df['close'].pct_change(periods=4)
            df['price_change_1d'] = df['close'].pct_change(periods=24)
            
            # Volatility
            df['volatility'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean()
            
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
            current_1h = df_1h.iloc[-1]
            current_4h = df_4h.iloc[-1]
            current_1d = df_1d.iloc[-1]
            
            # 1. Trend Analysis (40% weight)
            trend_score = 0
            max_trend_score = 3
            trend_reasons = []
            
            # EMA alignment check
            if (current_1h['ema_50'] > current_1h['ema_200'] and 
                current_4h['ema_50'] > current_4h['ema_200'] and 
                current_1d['ema_50'] > current_1d['ema_200']):
                trend_score += 1
                trend_reasons.append("EMA alignment bullish across timeframes")
            
            # ADX strength check
            if current_1h['adx'] > 25 and current_4h['adx'] > 25:
                trend_score += 1
                trend_reasons.append("Strong trend (ADX > 25)")
            
            # Price above EMAs
            if (current_1h['close'] > current_1h['ema_50'] and 
                current_4h['close'] > current_4h['ema_50']):
                trend_score += 1
                trend_reasons.append("Price above EMAs")
            
            # 2. Momentum Analysis (30% weight)
            momentum_score = 0
            max_momentum_score = 3
            momentum_reasons = []
            
            # RSI conditions
            if (30 < current_1h['rsi'] < 70 and 
                30 < current_4h['rsi'] < 70):
                momentum_score += 1
                momentum_reasons.append("RSI in healthy range")
            
            # MACD conditions
            if (current_1h['MACD_12_26_9'] > current_1h['MACDs_12_26_9'] and 
                current_4h['MACD_12_26_9'] > current_4h['MACDs_12_26_9']):
                momentum_score += 1
                momentum_reasons.append("MACD bullish crossover")
            
            # 3. Volatility and Volume Analysis (30% weight)
            volatility_score = 0
            max_volatility_score = 3
            volatility_reasons = []
            
            # Bollinger Bands squeeze
            bb_width_1h = (current_1h['BBU_20_2.0'] - current_1h['BBL_20_2.0']) / current_1h['BBM_20_2.0']
            bb_width_4h = (current_4h['BBU_20_2.0'] - current_4h['BBL_20_2.0']) / current_4h['BBM_20_2.0']
            
            if bb_width_1h < 0.1 and bb_width_4h < 0.1:
                volatility_score += 1
                volatility_reasons.append("Bollinger Band squeeze (potential breakout)")
            
            # Volume confirmation
            if (current_1h['volume_ratio'] > 1.5 and 
                current_4h['volume_ratio'] > 1.2):
                volatility_score += 1
                volatility_reasons.append("Above average volume")
            
            # ATR for volatility check
            if current_1h['atr'] > current_1h['atr'].rolling(20).mean():
                volatility_score += 1
                volatility_reasons.append("Increasing volatility")
            
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
                atr_multiplier = 2.0
                current_price = current_1h['close']
                signals['stop_loss'] = current_price - (current_1h['atr'] * atr_multiplier)
                signals['take_profit'] = current_price + (current_1h['atr'] * atr_multiplier * 2)
                
                # Calculate entry window
                recent_low = df_1h['low'].iloc[-5:].min()
                recent_high = df_1h['high'].iloc[-5:].max()
                
                signals['entry_window_low'] = recent_low
                signals['entry_window_high'] = current_price
                
                # Combine reasons
                all_reasons = trend_reasons + momentum_reasons + volatility_reasons
                signals['reason'] = ", ".join(all_reasons)
                
                # Adjust entry window hours based on volatility
                volatility_factor = current_1h['atr'] / current_1h['close']
                if volatility_factor > 0.03:
                    signals['entry_window_hours'] = 0.5
                elif volatility_factor > 0.02:
                    signals['entry_window_hours'] = 1
                else:
                    signals['entry_window_hours'] = 2
            
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
        """Fetch OHLCV data from the exchange for a specific pair"""
        try:
            # Fetch multiple timeframes for better context
            ohlcv_1h = self.exchange.fetch_ohlcv(pair, '1h', limit=100)
            ohlcv_4h = self.exchange.fetch_ohlcv(pair, '4h', limit=50)
            ohlcv_1d = self.exchange.fetch_ohlcv(pair, '1d', limit=30)
            
            # Convert to DataFrames
            df_1h = pd.DataFrame(ohlcv_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_4h = pd.DataFrame(ohlcv_4h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_1d = pd.DataFrame(ohlcv_1d, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert timestamps
            for df in [df_1h, df_4h, df_1d]:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
            
            return df_1h, df_4h, df_1d
        except Exception as e:
            logger.error(f"Error fetching data for {pair}: {e}")
            return None, None, None

    async def send_telegram_message(self, message):
        """Send notification via Telegram with improved error handling"""
        try:
            if not self.chat_id:
                logger.error("No Telegram chat ID configured")
                return
                
            await self.telegram_bot.send_message(
                chat_id=self.chat_id, 
                text=message,
                parse_mode='HTML'
            )
            logger.info(f"Telegram message sent: {message[:50]}...")
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")

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

if __name__ == "__main__":
    try:
        bot = SolanaTradingBot()
        asyncio.run(bot.run())
    except Exception as e:
        logger.critical(f"Fatal error: {e}") 