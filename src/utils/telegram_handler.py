import telegram
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import logging
from datetime import datetime
import asyncio
from typing import Dict, Any

logger = logging.getLogger(__name__)

class TelegramCommandHandler:
    def __init__(self, bot_token: str, chat_id: str, trading_bot: Any):
        """Initialize the Telegram command handler"""
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.trading_bot = trading_bot
        self.updater = Updater(token=bot_token, use_context=True)
        self.dispatcher = self.updater.dispatcher
        
        # Register command handlers
        self.dispatcher.add_handler(CommandHandler("start", self.start_command))
        self.dispatcher.add_handler(CommandHandler("help", self.help_command))
        self.dispatcher.add_handler(CommandHandler("status", self.status_command))
        self.dispatcher.add_handler(CommandHandler("price", self.price_command))
        self.dispatcher.add_handler(CommandHandler("signals", self.signals_command))
        self.dispatcher.add_handler(CommandHandler("settings", self.settings_command))
        
        # Register message handler for non-command messages
        self.dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, self.handle_message))
        
    def start(self):
        """Start the Telegram bot"""
        self.updater.start_polling()
        logger.info("Telegram command handler started")
        
    def stop(self):
        """Stop the Telegram bot"""
        self.updater.stop()
        logger.info("Telegram command handler stopped")
        
    async def start_command(self, update, context):
        """Handle the /start command"""
        welcome_message = (
            "👋 Welcome to the Solana Trading Bot!\n\n"
            "I can help you monitor and analyze SOL/USDT trading opportunities.\n\n"
            "Available commands:\n"
            "/help - Show all available commands\n"
            "/status - Check bot status and current positions\n"
            "/price - Get current SOL price and analysis\n"
            "/signals - View recent trading signals\n"
            "/settings - View current bot settings"
        )
        await update.message.reply_text(welcome_message)
        
    async def help_command(self, update, context):
        """Handle the /help command"""
        help_message = (
            "📚 Available Commands:\n\n"
            "/start - Start the bot and show welcome message\n"
            "/help - Show this help message\n"
            "/status - Check bot status and current positions\n"
            "/price - Get current SOL price and analysis\n"
            "/signals - View recent trading signals\n"
            "/settings - View current bot settings\n\n"
            "You can also ask me questions about:\n"
            "- Current market conditions\n"
            "- Trading signals\n"
            "- Technical analysis\n"
            "- Bot performance"
        )
        await update.message.reply_text(help_message)
        
    async def status_command(self, update, context):
        """Handle the /status command"""
        try:
            # Get current status from trading bot
            status = await self.trading_bot.get_status()
            status_message = (
                f"🤖 Bot Status\n\n"
                f"Status: {status['status']}\n"
                f"Last Update: {status['last_update']}\n"
                f"Current Position: {status['position']}\n"
                f"Entry Price: {status['entry_price']}\n"
                f"Current PnL: {status['pnl']}%\n"
                f"Active Signals: {status['active_signals']}"
            )
            await update.message.reply_text(status_message)
        except Exception as e:
            logger.error(f"Error in status command: {e}")
            await update.message.reply_text("❌ Error fetching bot status. Please try again later.")
            
    async def price_command(self, update, context):
        """Handle the /price command"""
        try:
            # Get current price and analysis from trading bot
            price_data = await self.trading_bot.get_current_price_analysis()
            price_message = (
                f"💰 SOL/USDT Price Analysis\n\n"
                f"Current Price: ${price_data['price']:.2f}\n"
                f"24h Change: {price_data['change_24h']}%\n"
                f"Volume 24h: ${price_data['volume_24h']:,.0f}\n\n"
                f"Technical Analysis:\n"
                f"RSI: {price_data['rsi']:.1f}\n"
                f"MACD: {price_data['macd']}\n"
                f"Trend: {price_data['trend']}\n"
                f"Support: ${price_data['support']:.2f}\n"
                f"Resistance: ${price_data['resistance']:.2f}"
            )
            await update.message.reply_text(price_message)
        except Exception as e:
            logger.error(f"Error in price command: {e}")
            await update.message.reply_text("❌ Error fetching price data. Please try again later.")
            
    async def signals_command(self, update, context):
        """Handle the /signals command"""
        try:
            # Get recent signals from trading bot
            signals = await self.trading_bot.get_recent_signals()
            signals_message = "📊 Recent Trading Signals\n\n"
            
            for signal in signals:
                signals_message += (
                    f"Type: {signal['type']}\n"
                    f"Time: {signal['time']}\n"
                    f"Price: ${signal['price']:.2f}\n"
                    f"Confidence: {signal['confidence']}%\n"
                    f"Status: {signal['status']}\n\n"
                )
                
            await update.message.reply_text(signals_message)
        except Exception as e:
            logger.error(f"Error in signals command: {e}")
            await update.message.reply_text("❌ Error fetching signals. Please try again later.")
            
    async def settings_command(self, update, context):
        """Handle the /settings command"""
        try:
            # Get current settings from trading bot
            settings = await self.trading_bot.get_settings()
            settings_message = (
                f"⚙️ Bot Settings\n\n"
                f"Trading Interval: {settings['trading_interval']} seconds\n"
                f"ML Prediction Threshold: {settings['ml_threshold']}\n"
                f"Signal Confidence Threshold: {settings['signal_threshold']}\n"
                f"Retrain Interval: {settings['retrain_interval']} seconds\n"
                f"Risk/Reward Ratio: 1:{settings['risk_reward_ratio']}\n"
                f"Max Position Size: {settings['max_position_size']}%"
            )
            await update.message.reply_text(settings_message)
        except Exception as e:
            logger.error(f"Error in settings command: {e}")
            await update.message.reply_text("❌ Error fetching settings. Please try again later.")
            
    async def handle_message(self, update, context):
        """Handle non-command messages"""
        try:
            message = update.message.text.lower()
            
            # Simple keyword-based responses
            if "hello" in message or "hi" in message:
                await update.message.reply_text("👋 Hello! How can I help you today?")
            elif "how are you" in message:
                await update.message.reply_text("I'm running smoothly! How can I assist you?")
            elif "thank" in message:
                await update.message.reply_text("You're welcome! Let me know if you need anything else.")
            elif "market" in message:
                # Get market analysis
                analysis = await self.trading_bot.get_market_analysis()
                await update.message.reply_text(analysis)
            elif "signal" in message:
                # Get latest signal
                signal = await self.trading_bot.get_latest_signal()
                await update.message.reply_text(signal)
            elif "help" in message:
                await self.help_command(update, context)
            else:
                await update.message.reply_text(
                    "I'm not sure I understand. Try using /help to see available commands."
                )
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            await update.message.reply_text("❌ Sorry, I encountered an error. Please try again.") 