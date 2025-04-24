import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Exchange settings
EXCHANGE_ID = 'binance'
SYMBOL = 'SOL/USDT'
TIMEFRAMES = ['1h', '4h', '1d']

# API credentials
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')

# Telegram settings
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Trading parameters
TREND_STRENGTH_THRESHOLD = 0.7
VOLUME_THRESHOLD = 1.5
SIGNAL_HISTORY_SIZE = 10

# ML model parameters
ML_MODEL_PATH = "models/sol_ml_model.pkl"
ML_SCALER_PATH = "models/sol_scaler.pkl"
ML_PREDICTION_THRESHOLD = 0.65
ML_WEIGHT = 2.5
ML_TRAINING_DAYS = 60
ML_RETRAINING_DAYS = 7

# Technical analysis parameters
RSI_PERIODS = [14, 21]
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2.0
SMA_PERIODS = [20, 50, 200]
EMA_PERIODS = [9, 21]
ADX_PERIOD = 14
ATR_PERIOD = 14
STOCH_RSI_PERIOD = 14

# Signal weights
SIGNAL_WEIGHTS = {
    'ML': 2.5,
    'RSI_DIV': 3.0,
    'MACD': 2.0,
    'MACD_HIST': 1.5,
    'BB': 1.5,
    'BB_SQUEEZE': 2.0,
    'MA': 2.0,
    'EMA': 1.5,
    'STOCH_RSI': 1.5,
    'ICHIMOKU': 2.0,
    'FIB': 1.5,
    'RSI': 1.0
}

# Risk management
MIN_SIGNAL_SCORE = 3.0
RISK_REWARD_RATIO = 1.5
ATR_STOP_LOSS_MULTIPLIER = 2.0
ATR_TAKE_PROFIT_MULTIPLIER = 3.0

# Logging
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = 'trading_bot.log' 