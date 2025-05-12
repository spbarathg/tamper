# SOL/USDT Trading Bot

A sophisticated Python-based trading bot that analyzes the SOL/USDT pair using advanced technical analysis and machine learning to generate high-probability trading signals. The bot features an interactive Telegram interface for real-time monitoring and control.

## 🌟 Features

### Trading Analysis
- Multi-timeframe analysis (1h, 4h, 1d) for better market context
- Machine learning model trained on historical data
- Comprehensive technical indicators:
  - RSI (Relative Strength Index) with divergence detection
  - MACD (Moving Average Convergence Divergence) with histogram analysis
  - Bollinger Bands with squeeze detection
  - Multiple Moving Averages (20, 50, 200 SMAs and 9, 21 EMAs)
  - Stochastic RSI
  - Ichimoku Cloud
  - Fibonacci Retracement levels
  - ADX for trend strength
  - ATR for volatility measurement
- Market context awareness (trend strength and direction)
- Weighted signal scoring system
- Risk management with dynamic stop-loss and take-profit levels

### Interactive Features
- Real-time Telegram notifications with detailed analysis
- Interactive command system for monitoring and control
- Natural language processing for easy interaction
- Available commands:
  - `/start` - Welcome message and basic info
  - `/help` - List of all available commands
  - `/status` - Check bot status and current positions
  - `/price` - Get current SOL price and analysis
  - `/signals` - View recent trading signals
  - `/settings` - View current bot settings
- Natural language support for:
  - Market condition queries
  - Trading signal information
  - Technical analysis details
  - Bot performance metrics

### System Features
- Comprehensive logging system
- Error handling and automatic reconnection
- Automatic model retraining every 7 days
- Rate limiting and API error handling
- Graceful shutdown handling

## 📋 Prerequisites

- Python 3.8 or higher
- Binance account with API access
- Telegram bot token (create one using [@BotFather](https://t.me/botfather))
- Basic understanding of cryptocurrency trading

## 🚀 Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file:
```bash
cp .env.example .env
```

4. Edit the `.env` file with your credentials:
```
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
```

## 💻 Usage

### Starting the Bot

1. Start the bot:
```bash
python sol_trading_bot.py
```

Or, to run in the background (recommended for servers):
```bash
nohup python sol_trading_bot.py > bot_output.log 2>&1 &
```

### Bot Operations

The bot will:
- Train or load a machine learning model for each trading pair
- Monitor price movements across multiple timeframes
- Calculate comprehensive technical indicators
- Generate ML predictions and technical analysis signals
- Analyze market context and trend strength
- Generate weighted trading signals
- Send Telegram notifications with detailed analysis
- Retrain the ML model automatically every 6 hours
- Wait a configurable interval between each analysis

### Interactive Commands

#### Basic Commands
- `/start` - Get started with the bot
- `/help` - View all available commands
- `/status` - Check current bot status and positions

#### Analysis Commands
- `/price` - Get current SOL price and technical analysis
- `/signals` - View recent trading signals
- `/settings` - View current bot configuration

#### Natural Language Queries
You can ask the bot questions like:
- "How's the market looking?"
- "Show me the latest signal"
- "What's the current trend?"
- "Tell me about the bot's performance"

## ⚙️ Configuration

### Environment Variables

| Variable                    | Default   | Description                                              |
|-----------------------------|-----------|----------------------------------------------------------|
| TRADING_INTERVAL_SECONDS    | 300       | How often to check for signals (in seconds)              |
| ML_PREDICTION_THRESHOLD     | 0.5       | Minimum ML confidence for a signal (0-1)                 |
| SIGNAL_CONFIDENCE_THRESHOLD | 0.5       | Minimum overall confidence for a trading signal (0-1)    |
| RETRAIN_INTERVAL_SECONDS    | 21600     | How often to retrain the ML model (in seconds)           |
| BINANCE_API_KEY             | (none)    | Your Binance API key                                     |
| BINANCE_SECRET_KEY          | (none)    | Your Binance API secret                                  |
| TELEGRAM_BOT_TOKEN          | (none)    | Your Telegram bot token                                  |
| TELEGRAM_CHAT_ID            | (none)    | Your Telegram chat ID                                    |

### Example Configurations

#### More Frequent Signals
```
TRADING_INTERVAL_SECONDS=60
ML_PREDICTION_THRESHOLD=0.4
SIGNAL_CONFIDENCE_THRESHOLD=0.4
RETRAIN_INTERVAL_SECONDS=3600
```

#### Conservative Trading
```
ML_PREDICTION_THRESHOLD=0.7
SIGNAL_CONFIDENCE_THRESHOLD=0.7
RETRAIN_INTERVAL_SECONDS=43200
```

## 📊 Signal Types

### Trading Signals
- **High-Confidence Signal:** Actionable trade opportunity with strong technical and ML confirmation
- **Watchlist Signal:** Lower-confidence opportunity marked with 👀 in Telegram
  - Use for monitoring and learning
  - Requires additional confirmation before trading
  - Good for understanding market patterns

## 📈 Monitoring & Logs

### Log Files
- `trading_bot.log` - Main log file
- `bot_output.log` - Console output when running in background

### Monitoring Commands
```bash
# View live logs
tail -f trading_bot.log

# View background process logs
tail -f bot_output.log
```

### Telegram Notifications
- All signals and errors are sent to your configured Telegram chat
- Interactive responses to commands and queries
- Real-time market analysis and updates

## 🔍 Advanced Trading Strategy

### Machine Learning Model
- Trained on latest available data
- Uses Random Forest Classifier
- Features include:
  - Technical indicators
  - Price changes
  - Volatility metrics
- Predicts price direction
- Configurable confidence threshold
- Automatic retraining

### Market Context Analysis
- Determines overall market context
- Calculates trend strength
- Considers multiple timeframes
- Only trades in trend direction when strong

### Technical Analysis
- **RSI:** Oversold/overbought conditions
- **MACD:** Crossovers and histogram analysis
- **Bollinger Bands:** Price touching/crossing bands
- **Moving Averages:** Golden/Death crosses
- **Stochastic RSI:** Additional confirmation
- **Ichimoku Cloud:** Trend direction
- **Fibonacci Retracement:** Support/resistance
- **ADX:** Trend strength
- **ATR:** Volatility measurement

### Signal Generation
- Weighted indicator scoring
- ML predictions weighted heavily
- Market context filtering
- Minimum score requirement
- 1:1.5 risk-reward ratio
- Dynamic stop-loss/take-profit

## ⚠️ Risk Management

The bot implements several risk management features:
- Trend-aligned trading only
- Dynamic stop-loss levels (2x ATR)
- 1:1.5 risk-reward ratio
- Multiple signal confirmation
- Comprehensive logging
- ML confidence threshold
- Watchlist signals for learning

## ⚠️ Disclaimer

This bot is for educational purposes only. Cryptocurrency trading carries significant risks. Always:
- Do your own research
- Never trade with money you cannot afford to lose
- Understand that past performance is not indicative of future results
- Monitor the bot's performance regularly
- Keep your API keys secure

## 📝 License

MIT License 