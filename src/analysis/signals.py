import logging
from ..config.config import (
    TREND_STRENGTH_THRESHOLD, VOLUME_THRESHOLD, SIGNAL_WEIGHTS,
    MIN_SIGNAL_SCORE, ATR_STOP_LOSS_MULTIPLIER, ATR_TAKE_PROFIT_MULTIPLIER,
    RISK_REWARD_RATIO
)

logger = logging.getLogger(__name__)

class SignalGenerator:
    def __init__(self):
        """Initialize the signal generator"""
        self.position = None  # 'long' or 'short'
        
    def determine_market_context(self, df_1h, df_4h, df_1d):
        """Determine overall market context and trend"""
        if df_1h is None or df_4h is None or df_1d is None:
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
            
    def _get_trend(self, df):
        """Determine trend direction for a given timeframe"""
        if df['close'].iloc[-1] > df['sma_50'].iloc[-1] and df['sma_20'].iloc[-1] > df['sma_50'].iloc[-1]:
            return "bullish"
        elif df['close'].iloc[-1] < df['sma_50'].iloc[-1] and df['sma_20'].iloc[-1] < df['sma_50'].iloc[-1]:
            return "bearish"
        else:
            return "sideways"
            
    def _calculate_trend_strength(self, df_1h, df_4h, df_1d):
        """Calculate trend strength based on multiple factors"""
        # ADX strength (higher ADX = stronger trend)
        adx_strength = min(df_1h['adx'].iloc[-1] / 100, 1.0)
        
        # Moving average alignment
        ma_alignment_1h = 1.0 if (df_1h['close'].iloc[-1] > df_1h['sma_20'].iloc[-1] > df_1h['sma_50'].iloc[-1] > df_1h['sma_200'].iloc[-1] or 
                                 df_1h['close'].iloc[-1] < df_1h['sma_20'].iloc[-1] < df_1h['sma_50'].iloc[-1] < df_1h['sma_200'].iloc[-1]) else 0.5
        
        # Volume confirmation
        volume_confirmation = 1.0 if df_1h['volume_ratio'].iloc[-1] > VOLUME_THRESHOLD else 0.5
        
        # Combine factors (weighted average)
        trend_strength = (adx_strength * 0.4 + ma_alignment_1h * 0.3 + volume_confirmation * 0.3)
        
        return trend_strength
        
    def analyze_signals(self, df_1h, df_4h, df_1d, ml_prediction=None, ml_confidence=0.0):
        """Analyze trading signals based on technical indicators and ML predictions"""
        if df_1h is None or df_4h is None or df_1d is None:
            return []
            
        current_price = df_1h['close'].iloc[-1]
        signals = []
        
        # Get market context
        market_context, trend_strength = self.determine_market_context(df_1h, df_4h, df_1d)
        
        # Add ML prediction as a signal if confidence is high enough
        if ml_prediction is not None and ml_confidence > 0.65:
            if ml_prediction == 1:
                signals.append(('ML', 'BUY', f'ML predicts price increase (confidence: {ml_confidence:.2f})'))
            else:
                signals.append(('ML', 'SELL', f'ML predicts price decrease (confidence: {ml_confidence:.2f})'))
        
        # RSI signals with divergence detection
        rsi = df_1h['rsi_14'].iloc[-1]
        rsi_prev = df_1h['rsi_14'].iloc[-2]
        price_prev = df_1h['close'].iloc[-2]
        
        # Check for RSI divergence
        if current_price > price_prev and rsi < rsi_prev:
            signals.append(('RSI_DIV', 'SELL', f'Bearish RSI divergence: Price up, RSI down'))
        elif current_price < price_prev and rsi > rsi_prev:
            signals.append(('RSI_DIV', 'BUY', f'Bullish RSI divergence: Price down, RSI up'))
        
        # Standard RSI signals
        if rsi < 30:
            signals.append(('RSI', 'BUY', f'RSI oversold at {rsi:.2f}'))
        elif rsi > 70:
            signals.append(('RSI', 'SELL', f'RSI overbought at {rsi:.2f}'))
            
        # MACD signals with histogram analysis
        if df_1h['MACD_12_26_9'].iloc[-1] > df_1h['MACDs_12_26_9'].iloc[-1] and \
           df_1h['MACD_12_26_9'].iloc[-2] <= df_1h['MACDs_12_26_9'].iloc[-2]:
            signals.append(('MACD', 'BUY', 'MACD crossed above signal line'))
        elif df_1h['MACD_12_26_9'].iloc[-1] < df_1h['MACDs_12_26_9'].iloc[-1] and \
             df_1h['MACD_12_26_9'].iloc[-2] >= df_1h['MACDs_12_26_9'].iloc[-2]:
            signals.append(('MACD', 'SELL', 'MACD crossed below signal line'))
            
        # MACD histogram increasing/decreasing
        if df_1h['MACDh_12_26_9'].iloc[-1] > df_1h['MACDh_12_26_9'].iloc[-2] > 0:
            signals.append(('MACD_HIST', 'BUY', 'MACD histogram increasing (bullish momentum)'))
        elif df_1h['MACDh_12_26_9'].iloc[-1] < df_1h['MACDh_12_26_9'].iloc[-2] < 0:
            signals.append(('MACD_HIST', 'SELL', 'MACD histogram decreasing (bearish momentum)'))
            
        # Bollinger Bands signals with squeeze detection
        if current_price < df_1h['BBL_20_2.0'].iloc[-1]:
            signals.append(('BB', 'BUY', 'Price below lower Bollinger Band'))
        elif current_price > df_1h['BBU_20_2.0'].iloc[-1]:
            signals.append(('BB', 'SELL', 'Price above upper Bollinger Band'))
            
        # Bollinger Band squeeze (potential breakout setup)
        bb_width = (df_1h['BBU_20_2.0'].iloc[-1] - df_1h['BBL_20_2.0'].iloc[-1]) / df_1h['BBM_20_2.0'].iloc[-1]
        bb_width_prev = (df_1h['BBU_20_2.0'].iloc[-2] - df_1h['BBL_20_2.0'].iloc[-2]) / df_1h['BBM_20_2.0'].iloc[-2]
        
        if bb_width < bb_width_prev and bb_width < 0.1:
            if current_price > df_1h['BBM_20_2.0'].iloc[-1]:
                signals.append(('BB_SQUEEZE', 'BUY', 'Bollinger Band squeeze with price above middle band'))
            else:
                signals.append(('BB_SQUEEZE', 'SELL', 'Bollinger Band squeeze with price below middle band'))
            
        # Moving Average signals with golden/death cross
        if df_1h['sma_20'].iloc[-1] > df_1h['sma_50'].iloc[-1] and \
           df_1h['sma_20'].iloc[-2] <= df_1h['sma_50'].iloc[-2]:
            signals.append(('MA', 'BUY', 'Golden Cross: 20 SMA crossed above 50 SMA'))
        elif df_1h['sma_20'].iloc[-1] < df_1h['sma_50'].iloc[-1] and \
             df_1h['sma_20'].iloc[-2] >= df_1h['sma_50'].iloc[-2]:
            signals.append(('MA', 'SELL', 'Death Cross: 20 SMA crossed below 50 SMA'))
            
        # EMA signals
        if df_1h['ema_9'].iloc[-1] > df_1h['ema_21'].iloc[-1] and \
           df_1h['ema_9'].iloc[-2] <= df_1h['ema_21'].iloc[-2]:
            signals.append(('EMA', 'BUY', 'EMA 9 crossed above EMA 21'))
        elif df_1h['ema_9'].iloc[-1] < df_1h['ema_21'].iloc[-1] and \
             df_1h['ema_9'].iloc[-2] >= df_1h['ema_21'].iloc[-2]:
            signals.append(('EMA', 'SELL', 'EMA 9 crossed below EMA 21'))
            
        # Stochastic RSI signals
        if df_1h['STOCHRSId_14'].iloc[-1] < 20 and df_1h['STOCHRSIk_14'].iloc[-1] < 20:
            signals.append(('STOCH_RSI', 'BUY', 'Stochastic RSI oversold'))
        elif df_1h['STOCHRSId_14'].iloc[-1] > 80 and df_1h['STOCHRSIk_14'].iloc[-1] > 80:
            signals.append(('STOCH_RSI', 'SELL', 'Stochastic RSI overbought'))
            
        # Ichimoku Cloud signals
        if current_price > df_1h['ITS_9'].iloc[-1] and current_price > df_1h['IKS_26'].iloc[-1]:
            signals.append(('ICHIMOKU', 'BUY', 'Price above Ichimoku Cloud (bullish)'))
        elif current_price < df_1h['ITS_9'].iloc[-1] and current_price < df_1h['IKS_26'].iloc[-1]:
            signals.append(('ICHIMOKU', 'SELL', 'Price below Ichimoku Cloud (bearish)'))
            
        # Add market context to signals
        for i, signal in enumerate(signals):
            signals[i] = (signal[0], signal[1], signal[2], market_context, trend_strength)
            
        return signals
        
    def generate_trading_decision(self, signals):
        """Generate final trading decision based on all signals"""
        if not signals:
            return None, "No clear trading signals"
            
        # Filter signals by market context
        market_context = signals[0][3]  # Get market context from first signal
        trend_strength = signals[0][4]  # Get trend strength from first signal
        
        # Only take trades in the direction of the overall trend if trend is strong
        if trend_strength > TREND_STRENGTH_THRESHOLD:
            if market_context in ["strong_bullish", "bullish"]:
                signals = [s for s in signals if s[1] == 'BUY']
            elif market_context in ["strong_bearish", "bearish"]:
                signals = [s for s in signals if s[1] == 'SELL']
        
        buy_signals = [s for s in signals if s[1] == 'BUY']
        sell_signals = [s for s in signals if s[1] == 'SELL']
        
        # Calculate weighted scores
        weighted_buy = sum(SIGNAL_WEIGHTS.get(signal[0], 1.0) for signal in buy_signals)
        weighted_sell = sum(SIGNAL_WEIGHTS.get(signal[0], 1.0) for signal in sell_signals)
        
        # Decision logic with weighted scoring
        if weighted_buy > weighted_sell and weighted_buy >= MIN_SIGNAL_SCORE and self.position != 'long':
            return 'BUY', f"Strong buy signals (score: {weighted_buy:.1f}): {', '.join([s[2] for s in buy_signals])}"
        elif weighted_sell > weighted_buy and weighted_sell >= MIN_SIGNAL_SCORE and self.position != 'short':
            return 'SELL', f"Strong sell signals (score: {weighted_sell:.1f}): {', '.join([s[2] for s in sell_signals])}"
            
        return None, "Insufficient signals for trading decision"
        
    def calculate_position_levels(self, df, decision):
        """Calculate stop loss and take profit levels"""
        if decision is None:
            return None, None
            
        current_price = df['close'].iloc[-1]
        atr = df['atr'].iloc[-1]
        
        if decision == 'BUY':
            stop_loss = current_price - (atr * ATR_STOP_LOSS_MULTIPLIER)
            take_profit = current_price + (atr * ATR_TAKE_PROFIT_MULTIPLIER)
        else:
            stop_loss = current_price + (atr * ATR_STOP_LOSS_MULTIPLIER)
            take_profit = current_price - (atr * ATR_TAKE_PROFIT_MULTIPLIER)
            
        return stop_loss, take_profit 