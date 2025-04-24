import numpy as np
import pandas as pd
import talib
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import plotly.graph_objects as go
from datetime import datetime, timedelta

class AdvancedAnalysis:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        
    def fetch_multi_source_data(self, symbol, timeframe='1h', lookback='30d'):
        """
        Fetch data from multiple sources for cross-validation
        """
        # Binance data (from CCXT)
        binance_data = self.fetch_binance_data(symbol, timeframe, lookback)
        
        # Yahoo Finance data
        yf_data = yf.download(symbol, period=lookback, interval=timeframe)
        
        # Combine and validate data
        combined_data = self.validate_and_combine_data(binance_data, yf_data)
        return combined_data

    def calculate_advanced_indicators(self, df):
        """
        Calculate advanced technical indicators
        """
        # Trend Indicators
        df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        df['DI_plus'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
        df['DI_minus'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Momentum Indicators
        df['RSI'] = talib.RSI(df['close'], timeperiod=14)
        df['MFI'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
        df['ROC'] = talib.ROC(df['close'], timeperiod=10)
        
        # Volatility Indicators
        df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['NATR'] = talib.NATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Volume Indicators
        df['OBV'] = talib.OBV(df['close'], df['volume'])
        df['AD'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
        
        # Pattern Recognition
        df['CDL_HAMMER'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
        df['CDL_ENGULFING'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
        df['CDL_MORNING_STAR'] = talib.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close'])
        
        return df

    def build_lstm_model(self, sequence_length=60):
        """
        Build LSTM model for price prediction
        """
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(sequence_length, 5)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def prepare_lstm_data(self, df, sequence_length=60):
        """
        Prepare data for LSTM model
        """
        features = ['close', 'volume', 'RSI', 'MACD', 'BB_width']
        data = df[features].values
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(scaled_data[i, 0])
            
        return np.array(X), np.array(y)

    def detect_market_regime(self, df):
        """
        Detect current market regime using multiple indicators
        """
        # Trend strength
        trend_strength = df['ADX'].iloc[-1]
        
        # Volatility regime
        volatility = df['ATR'].iloc[-1] / df['close'].iloc[-1] * 100
        
        # Volume profile
        volume_trend = df['OBV'].pct_change().iloc[-1]
        
        if trend_strength > 25:
            if df['DI_plus'].iloc[-1] > df['DI_minus'].iloc[-1]:
                regime = 'STRONG_UPTREND'
            else:
                regime = 'STRONG_DOWNTREND'
        else:
            if volatility > 2:
                regime = 'HIGH_VOLATILITY'
            else:
                regime = 'RANGING'
                
        return regime

    def generate_trading_signals(self, df):
        """
        Generate trading signals using multiple confirmations
        """
        signals = []
        
        # Trend confirmation
        trend_confirmed = (df['ADX'].iloc[-1] > 25 and 
                         abs(df['DI_plus'].iloc[-1] - df['DI_minus'].iloc[-1]) > 5)
        
        # Momentum confirmation
        momentum_confirmed = (df['RSI'].iloc[-1] < 30 or df['RSI'].iloc[-1] > 70)
        
        # Volume confirmation
        volume_confirmed = df['volume'].iloc[-1] > df['volume'].rolling(20).mean().iloc[-1]
        
        # Pattern confirmation
        pattern_confirmed = (df['CDL_HAMMER'].iloc[-1] != 0 or 
                           df['CDL_ENGULFING'].iloc[-1] != 0 or 
                           df['CDL_MORNING_STAR'].iloc[-1] != 0)
        
        if trend_confirmed and momentum_confirmed and volume_confirmed:
            if df['RSI'].iloc[-1] < 30 and df['DI_plus'].iloc[-1] > df['DI_minus'].iloc[-1]:
                signals.append(('STRONG_BUY', 'Multiple confirmations for bullish reversal'))
            elif df['RSI'].iloc[-1] > 70 and df['DI_plus'].iloc[-1] < df['DI_minus'].iloc[-1]:
                signals.append(('STRONG_SELL', 'Multiple confirmations for bearish reversal'))
                
        return signals

    def calculate_risk_metrics(self, df):
        """
        Calculate advanced risk metrics
        """
        # Volatility
        daily_returns = df['close'].pct_change()
        volatility = daily_returns.std() * np.sqrt(252)
        
        # Value at Risk (VaR)
        var_95 = np.percentile(daily_returns, 5)
        
        # Maximum Drawdown
        cumulative_returns = (1 + daily_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        return {
            'volatility': volatility,
            'var_95': var_95,
            'max_drawdown': max_drawdown
        }

    def plot_analysis(self, df, signals=None):
        """
        Create interactive plot with technical analysis
        """
        fig = go.Figure()
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='OHLC'
        ))
        
        # Add indicators
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['RSI'],
            name='RSI',
            yaxis='y2'
        ))
        
        # Add signals if provided
        if signals:
            for signal in signals:
                if signal[0] == 'STRONG_BUY':
                    fig.add_annotation(
                        x=df.index[-1],
                        y=df['low'].iloc[-1],
                        text='BUY',
                        showarrow=True,
                        arrowhead=1,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor='green'
                    )
                elif signal[0] == 'STRONG_SELL':
                    fig.add_annotation(
                        x=df.index[-1],
                        y=df['high'].iloc[-1],
                        text='SELL',
                        showarrow=True,
                        arrowhead=1,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor='red'
                    )
        
        fig.update_layout(
            title='Advanced Technical Analysis',
            yaxis_title='Price',
            yaxis2=dict(
                title='RSI',
                overlaying='y',
                side='right'
            )
        )
        
        return fig 