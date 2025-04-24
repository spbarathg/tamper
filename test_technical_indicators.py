import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_sample_data(days=60):
    """Generate sample price data for testing"""
    dates = pd.date_range(end=datetime.now(), periods=days*24, freq='H')
    np.random.seed(42)
    
    # Generate random walk price data
    price = 100
    prices = []
    for _ in range(len(dates)):
        price *= (1 + np.random.normal(0, 0.002))
        prices.append(price)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
        'close': prices,
        'volume': [abs(np.random.normal(1000000, 200000)) for _ in prices]
    })
    df.set_index('timestamp', inplace=True)
    return df

def calculate_macd(close_prices, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    exp1 = close_prices.ewm(span=fast, adjust=False).mean()
    exp2 = close_prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_rsi(close_prices, periods=14):
    """Calculate RSI indicator"""
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def test_macd_calculation(df):
    """Test MACD calculation"""
    try:
        logger.info("Testing MACD calculation...")
        
        # Calculate MACD
        macd_line, signal_line, histogram = calculate_macd(df['close'])
        
        # Add to DataFrame
        df['MACD_12_26_9'] = macd_line
        df['MACDs_12_26_9'] = signal_line
        df['MACDh_12_26_9'] = histogram
        
        # Fill NaN values
        df['MACD_12_26_9'] = df['MACD_12_26_9'].fillna(0)
        df['MACDs_12_26_9'] = df['MACDs_12_26_9'].fillna(0)
        df['MACDh_12_26_9'] = df['MACDh_12_26_9'].fillna(0)
        
        logger.info("MACD calculation successful")
        return True
    except Exception as e:
        logger.error(f"Error in MACD calculation: {e}")
        return False

def test_rsi_calculation(df):
    """Test RSI calculation"""
    try:
        logger.info("Testing RSI calculation...")
        
        # Calculate RSI
        df['rsi_14'] = calculate_rsi(df['close'])
        df['rsi_14'] = df['rsi_14'].fillna(50)
        
        logger.info("RSI calculation successful")
        return True
    except Exception as e:
        logger.error(f"Error in RSI calculation: {e}")
        return False

def test_bollinger_bands(df):
    """Test Bollinger Bands calculation"""
    try:
        logger.info("Testing Bollinger Bands calculation...")
        
        # Calculate manually
        sma_20 = df['close'].rolling(window=20).mean()
        std_20 = df['close'].rolling(window=20).std()
        
        df['BBL_20_2.0'] = sma_20 - (std_20 * 2)
        df['BBM_20_2.0'] = sma_20
        df['BBU_20_2.0'] = sma_20 + (std_20 * 2)
        
        # Fill NaN values
        df['BBL_20_2.0'] = df['BBL_20_2.0'].fillna(df['close'])
        df['BBM_20_2.0'] = df['BBM_20_2.0'].fillna(df['close'])
        df['BBU_20_2.0'] = df['BBU_20_2.0'].fillna(df['close'])
        
        logger.info("Bollinger Bands calculation successful")
        return True
    except Exception as e:
        logger.error(f"Error in Bollinger Bands calculation: {e}")
        return False

def test_moving_averages(df):
    """Test Moving Averages calculation"""
    try:
        logger.info("Testing Moving Averages calculation...")
        
        # Calculate SMAs
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        
        # Calculate EMAs
        df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        # Fill NaN values
        for col in ['sma_20', 'sma_50', 'sma_200', 'ema_9', 'ema_21', 'ema_50', 'ema_200']:
            df[col] = df[col].fillna(df['close'])
        
        logger.info("Moving Averages calculation successful")
        return True
    except Exception as e:
        logger.error(f"Error in Moving Averages calculation: {e}")
        return False

def test_volume_indicators(df):
    """Test Volume Indicators calculation"""
    try:
        logger.info("Testing Volume Indicators calculation...")
        
        # Calculate volume SMA and ratio
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['volume_ratio'] = df['volume_ratio'].fillna(1)
        
        logger.info("Volume Indicators calculation successful")
        return True
    except Exception as e:
        logger.error(f"Error in Volume Indicators calculation: {e}")
        return False

def main():
    """Main test function"""
    logger.info("Starting technical indicators test...")
    
    # Generate sample data
    df = generate_sample_data()
    logger.info(f"Generated {len(df)} data points")
    
    # Run tests
    tests = [
        test_macd_calculation,
        test_rsi_calculation,
        test_bollinger_bands,
        test_moving_averages,
        test_volume_indicators
    ]
    
    results = {}
    for test in tests:
        results[test.__name__] = test(df)
    
    # Print results
    logger.info("\nTest Results:")
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    # Save results to CSV for inspection
    df.to_csv('test_results.csv')
    logger.info("Results saved to test_results.csv")

if __name__ == "__main__":
    main() 