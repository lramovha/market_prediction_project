# src/feature_engineering.py
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import mplfinance as mpf

# Moving Averages for Trend Analysis
def add_moving_averages(data: pd.DataFrame) -> pd.DataFrame:
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    return data

# Price Action Features
def add_price_action(data: pd.DataFrame) -> pd.DataFrame:
    data['Higher_High'] = (data['High'] > data['High'].shift(1)) & (data['High'] > data['High'].shift(-1))
    data['Lower_Low'] = (data['Low'] < data['Low'].shift(1)) & (data['Low'] < data['Low'].shift(-1))
    data['Higher_High'] = data['Higher_High'].astype(int)
    data['Lower_Low'] = data['Lower_Low'].astype(int)
    return data

# Supply and Demand Zones
def add_supply_demand_zones(data: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    data['Supply_Zone'] = data['High'].rolling(lookback).max()
    data['Demand_Zone'] = data['Low'].rolling(lookback).min()
    return data

# Support and Resistance Levels
def add_support_resistance(data: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    # Identifying local max and min
    data['Local_Max'] = data.iloc[argrelextrema(data['High'].values, np.greater_equal, order=window)[0]]['High']
    data['Local_Min'] = data.iloc[argrelextrema(data['Low'].values, np.less_equal, order=window)[0]]['Low']
    
    # Handling missing values (forward fill)
    data['Local_Max'].fillna(method='ffill', inplace=True)
    data['Local_Min'].fillna(method='ffill', inplace=True)
    
    return data
def calculate_bollinger_bands(data, window=20, num_std_dev=2):
    """Calculate Bollinger Bands."""
    data['MA20'] = data['Close'].rolling(window=window).mean()
    data['Upper_Band'] = data['MA20'] + (data['Close'].rolling(window=window).std() * num_std_dev)
    data['Lower_Band'] = data['MA20'] - (data['Close'].rolling(window=window).std() * num_std_dev)
    return data

def identify_breakouts(data):
    """Identify breakout signals based on Bollinger Bands."""
    data['Breakout_Up'] = np.where(data['Close'] > data['Upper_Band'], 1, 0)
    data['Breakout_Down'] = np.where(data['Close'] < data['Lower_Band'], -1, 0)
    return data

def identify_trend(data, short_window=20, long_window=50):
    """Identify trend direction using moving averages."""
    data['SMA_Short'] = data['Close'].rolling(window=short_window).mean()
    data['SMA_Long'] = data['Close'].rolling(window=long_window).mean()
    data['Trend'] = np.where(data['SMA_Short'] > data['SMA_Long'], 1, -1)  # 1 for uptrend, -1 for downtrend
    return data

# Main function to apply all strategy-based features
def preprocess_data(filename: str) -> pd.DataFrame:
    # Read the CSV file assuming the first row contains the correct headers
    data = pd.read_csv(f"data/{filename}.csv", header=0)
    
    # Debug: Print the first few rows and columns to check the data
    print("Columns in CSV:", data.columns)
    print("First few rows of data:", data.head())

    # Manually rename columns, we use 'Close' instead of 'Adj Close'
    data.columns = ['Date', 'Price', 'Close', 'High', 'Low', 'Open', 'Volume']

    # Ensure 'Date' is parsed as datetime
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

    # Set 'Date' as the index
    data.set_index('Date', inplace=True)

    # Drop rows with NaN in essential columns
    data.dropna(subset=['Price', 'Close', 'High', 'Low', 'Open'], inplace=True)

    # Convert columns to numeric
    data['Price'] = pd.to_numeric(data['Price'], errors='coerce')
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    data['High'] = pd.to_numeric(data['High'], errors='coerce')
    data['Low'] = pd.to_numeric(data['Low'], errors='coerce')
    data['Open'] = pd.to_numeric(data['Open'], errors='coerce')
    data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce')

    # Apply feature engineering functions
    data = add_moving_averages(data)
    data = add_price_action(data)
    data = add_supply_demand_zones(data)
    data = add_support_resistance(data)

    # Drop any rows with NaN values after transformations
    data.dropna(inplace=True)

    # Debug: Print the final transformed data
    print("Transformed data:", data.head())
    mpf.plot(data, type='candle', volume=True, style='yahoo')

    return data
