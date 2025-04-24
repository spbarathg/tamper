import telegram
import logging
import asyncio
from datetime import datetime
from ..config.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)

class TelegramNotifier:
    def __init__(self):
        """Initialize the Telegram bot"""
        self.bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        self.chat_id = TELEGRAM_CHAT_ID
        
    async def send_message(self, message):
        """Send a message via Telegram"""
        try:
            await self.bot.send_message(chat_id=self.chat_id, text=message)
            logger.info(f"Telegram message sent: {message[:50]}...")
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
            
    async def send_trading_signal(self, decision, message, current_price, stop_loss, take_profit):
        """Send a trading signal notification"""
        try:
            signal_message = (
                f"🚨 Trading Signal Alert 🚨\n"
                f"Action: {decision}\n"
                f"Reason: {message}\n"
                f"Current Price: {current_price:.2f} USDT\n"
                f"Stop Loss: {stop_loss:.2f} USDT\n"
                f"Take Profit: {take_profit:.2f} USDT\n"
                f"Risk-Reward: 1:1.5\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            await self.send_message(signal_message)
        except Exception as e:
            logger.error(f"Error sending trading signal: {e}")
            
    async def send_error_message(self, error_message):
        """Send an error notification"""
        try:
            error_notification = (
                f"⚠️ Error Alert ⚠️\n"
                f"Error: {error_message}\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            await self.send_message(error_notification)
        except Exception as e:
            logger.error(f"Error sending error notification: {e}")
            
    async def send_model_update(self, train_accuracy, val_accuracy):
        """Send a model update notification"""
        try:
            update_message = (
                f"🔄 Model Update 🔄\n"
                f"Training Accuracy: {train_accuracy:.4f}\n"
                f"Validation Accuracy: {val_accuracy:.4f}\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            await self.send_message(update_message)
        except Exception as e:
            logger.error(f"Error sending model update: {e}")
            
    async def send_startup_message(self):
        """Send a startup notification"""
        try:
            startup_message = (
                f"🚀 Solana Trading Bot Started 🚀\n"
                f"Monitoring SOL/USDT pair\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            await this.send_message(startup_message)
        except Exception as e:
            logger.error(f"Error sending startup message: {e}") 