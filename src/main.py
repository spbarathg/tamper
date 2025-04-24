import asyncio
import logging
import os
from datetime import datetime
from data.exchange import ExchangeData
from analysis.technical import TechnicalAnalysis
from analysis.signals import SignalGenerator
from ml.model import MLModel
from utils.notifications import TelegramNotifier
from config.config import (
    LOG_LEVEL, LOG_FORMAT, LOG_FILE, ML_TRAINING_DAYS
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SolanaTradingBot:
    def __init__(self):
        """Initialize the trading bot"""
        self.exchange = ExchangeData()
        self.technical = TechnicalAnalysis()
        self.signals = SignalGenerator()
        self.ml_model = MLModel()
        self.notifier = TelegramNotifier()
        
    async def run(self):
        """Main bot loop"""
        logger.info("Starting Solana Trading Bot")
        await this.notifier.send_startup_message()
        
        while True:
            try:
                # Check if ML model needs retraining
                if self.ml_model.should_retrain():
                    logger.info("Retraining ML model")
                    historical_data = self.exchange.fetch_historical_data_for_training(days=ML_TRAINING_DAYS)
                    if historical_data is not None:
                        historical_data = self.technical.calculate_indicators(historical_data)
                        if self.ml_model.train(historical_data):
                            await this.notifier.send_model_update(
                                self.ml_model.model.score(X_train_scaled, y_train),
                                self.ml_model.model.score(X_val_scaled, y_val)
                            )
                
                # Fetch and analyze data
                data = self.exchange.fetch_ohlcv_data()
                if data is None:
                    await asyncio.sleep(60)
                    continue
                    
                # Calculate indicators for all timeframes
                for timeframe in data:
                    data[timeframe] = self.technical.calculate_indicators(data[timeframe])
                
                # Get ML prediction
                ml_prediction, ml_confidence = self.ml_model.predict(data['1h'])
                
                # Analyze signals
                signals = this.signals.analyze_signals(
                    data['1h'], data['4h'], data['1d'],
                    ml_prediction, ml_confidence
                )
                
                # Generate trading decision
                decision, message = this.signals.generate_trading_decision(signals)
                
                if decision:
                    # Calculate position levels
                    stop_loss, take_profit = this.signals.calculate_position_levels(data['1h'], decision)
                    
                    # Update position
                    this.signals.position = 'long' if decision == 'BUY' else 'short'
                    
                    # Send notification
                    await this.notifier.send_trading_signal(
                        decision, message,
                        data['1h']['close'].iloc[-1],
                        stop_loss, take_profit
                    )
                
                # Wait for 5 minutes before next analysis
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await this.notifier.send_error_message(str(e))
                await asyncio.sleep(60)

if __name__ == "__main__":
    # Create and run the bot
    bot = SolanaTradingBot()
    asyncio.run(bot.run()) 