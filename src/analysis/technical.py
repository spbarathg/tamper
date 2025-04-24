import pandas as pd
import pandas_ta as ta
import numpy as np
import logging
from ..config.config import (
    RSI_PERIODS, MACD_FAST, MACD_SLOW, MACD_SIGNAL, BB_PERIOD, BB_STD,
    SMA_PERIODS, EMA_PERIODS, ADX_PERIOD, ATR_PERIOD, STOCH_RSI_PERIOD
)

logger = logging.getLogger(__name__)

class TechnicalAnalysis:
    @staticmethod
    def calculate_indicators(df):
        """Calculate all technical indicators for a given DataFrame"""
        try:
            # RSI with multiple timeframes
            for period in RSI_PERIODS:
                df[f'rsi_{period}'] = ta.rsi(df['close'], length=period)
            
            # MACD
            macd = ta.macd(df['close'], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
            df = pd.concat([df, macd], axis=1)
            
            # Bollinger Bands
            bollinger = ta.bbands(df['close'], length=BB_PERIOD, std=BB_STD)
            df = pd.concat([df, bollinger], axis=1)
            
            # Moving Averages
            for period in SMA_PERIODS:
                df[f'sma_{period}'] = ta.sma(df['close'], length=period)
            
            for period in EMA_PERIODS:
                df[f'ema_{period}'] = ta.ema(df['close'], length=period)
            
            # Volume indicators
            df['volume_sma'] = ta.sma(df['volume'], length=20)
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Trend indicators
            df['adx'] = ta.adx(df['high'], df['low'], df['close'], length=ADX_PERIOD)
            df['dmi_plus'] = ta.dmi(df['high'], df['low'], df['close'], length=ADX_PERIOD)['DMP_14']
            df['dmi_minus'] = ta.dmi(df['high'], df['low'], df['close'], length=ADX_PERIOD)['DMN_14']
            
            # Stochastic RSI
            stoch_rsi = ta.stochrsi(df['close'], length=STOCH_RSI_PERIOD)
            df = pd.concat([df, stoch_rsi], axis=1)
            
            # Ichimoku Cloud
            ichimoku = ta.ichimoku(df['high'], df['low'], df['close'])
            df = pd.concat([df, ichimoku], axis=1)
            
            # ATR for volatility
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=ATR_PERIOD)
            
            # Price changes
            df['price_change'] = df['close'].pct_change()
            df['price_change_1h'] = df['close'].pct_change(periods=1)
            df['price_change_4h'] = df['close'].pct_change(periods=4)
            df['price_change_1d'] = df['close'].pct_change(periods=24)
            
            # Volatility
            df['volatility'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean()
            
            return df
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return None
            
    @staticmethod
    def calculate_fibonacci_levels(df, window=20):
        """Calculate Fibonacci retracement levels"""
        try:
            recent_high = df['high'].rolling(window=window).max().iloc[-1]
            recent_low = df['low'].rolling(window=window).min().iloc[-1]
            price_range = recent_high - recent_low
            
            fib_levels = {
                'fib_0.236': recent_low + price_range * 0.236,
                'fib_0.382': recent_low + price_range * 0.382,
                'fib_0.5': recent_low + price_range * 0.5,
                'fib_0.618': recent_low + price_range * 0.618,
                'fib_0.786': recent_low + price_range * 0.786
            }
            
            return fib_levels
        except Exception as e:
            logger.error(f"Error calculating Fibonacci levels: {e}")
            return None
            
    @staticmethod
    def detect_divergence(df, window=14):
        """Detect RSI divergence"""
        try:
            price_highs = df['close'].rolling(window=window, center=True).max()
            price_lows = df['close'].rolling(window=window, center=True).min()
            rsi_highs = df['rsi_14'].rolling(window=window, center=True).max()
            rsi_lows = df['rsi_14'].rolling(window=window, center=True).min()
            
            bullish_div = (price_lows.iloc[-1] < price_lows.iloc[-window]) and (rsi_lows.iloc[-1] > rsi_lows.iloc[-window])
            bearish_div = (price_highs.iloc[-1] > price_highs.iloc[-window]) and (rsi_highs.iloc[-1] < rsi_highs.iloc[-window])
            
            return bullish_div, bearish_div
        except Exception as e:
            logger.error(f"Error detecting divergence: {e}")
            return False, False
            
    @staticmethod
    def detect_squeeze(df):
        """Detect Bollinger Band squeeze"""
        try:
            bb_width = (df['BBU_20_2.0'] - df['BBL_20_2.0']) / df['BBM_20_2.0']
            bb_width_prev = bb_width.shift(1)
            
            squeeze = (bb_width < bb_width_prev) & (bb_width < 0.1)
            return squeeze.iloc[-1]
        except Exception as e:
            logger.error(f"Error detecting squeeze: {e}")
            return False 