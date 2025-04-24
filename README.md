# SOL/USDT Trading Bot

A sophisticated Python-based trading bot that analyzes the SOL/USDT pair using advanced technical analysis and machine learning to generate high-probability trading signals.

## Features

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
- Telegram notifications with detailed analysis
- Comprehensive logging system
- Error handling and automatic reconnection
- Automatic model retraining every 7 days

## Prerequisites

- Python 3.8 or higher
- Binance account with API access
- Telegram bot token (create one using [@BotFather](https://t.me/botfather))

## Installation

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
- Add your Binance API key and secret
- Add your Telegram bot token
- Add your Telegram chat ID

## Usage

1. Start the bot:
```bash
python sol_trading_bot.py
```

2. The bot will:
- Train a machine learning model on historical data (if no model exists)
- Monitor SOL/USDT price movements across multiple timeframes
- Calculate comprehensive technical indicators
- Generate ML predictions and technical analysis signals
- Analyze market context and trend strength
- Generate weighted trading signals
- Send Telegram notifications with detailed analysis including stop-loss and take-profit levels
- Retrain the ML model every 7 days to adapt to changing market conditions
- Wait 5 minutes between each analysis

## Advanced Trading Strategy

The bot uses a sophisticated multi-factor approach combining machine learning with technical analysis to generate high-probability trading signals:

### Machine Learning Model
- Trained on 60 days of historical data
- Uses Random Forest Classifier for prediction
- Features include technical indicators, price changes, and volatility metrics
- Predicts whether the price will increase or decrease in the next period
- Confidence threshold of 0.65 required for ML signals
- Automatically retrained every 7 days to adapt to changing market conditions

### Market Context Analysis
- Determines overall market context (strong bullish, bullish, sideways, bearish, strong bearish)
- Calculates trend strength based on ADX, moving average alignment, and volume confirmation
- Only takes trades in the direction of the overall trend when trend is strong

### Technical Indicators
- **RSI**: Oversold (< 30) and overbought (> 70) conditions, plus divergence detection
- **MACD**: Crossovers of the MACD line and signal line, plus histogram analysis
- **Bollinger Bands**: Price touching or crossing the bands, plus squeeze detection for breakout setups
- **Moving Averages**: Golden/Death crosses of 20 and 50 SMAs, plus EMA crossovers
- **Stochastic RSI**: For additional confirmation of overbought/oversold conditions
- **Ichimoku Cloud**: For trend direction and support/resistance levels
- **Fibonacci Retracement**: For identifying key support and resistance levels
- **ADX**: For measuring trend strength
- **ATR**: For volatility-based position sizing and stop-loss placement

### Signal Generation
- Each indicator is weighted based on its historical reliability
- ML predictions get a high weight (2.5) in the final decision
- Signals are filtered based on market context
- A trading decision requires a minimum weighted score of 3.0
- Risk-reward ratio of 1:1.5 is maintained for all trades
- Dynamic stop-loss and take-profit levels based on ATR

## Risk Management

The bot implements several risk management features:
- Only takes trades in the direction of the overall trend when trend is strong
- Uses ATR to set dynamic stop-loss levels (2x ATR)
- Maintains a 1:1.5 risk-reward ratio for all trades
- Requires multiple confirming signals before taking a trade
- Logs all trading decisions for later analysis
- ML model confidence threshold to filter out low-probability signals

## Disclaimer

This bot is for educational purposes only. Cryptocurrency trading carries significant risks. Always do your own research and never trade with money you cannot afford to lose. Past performance is not indicative of future results.

## License

MIT License 