# # src/feature_engineering.py
# import pandas as pd
# import numpy as np
# from scipy.signal import argrelextrema
# import mplfinance as mpf

# # Moving Averages for Trend Analysis
# def add_moving_averages(data: pd.DataFrame) -> pd.DataFrame:
#     data['SMA_20'] = data['Close'].rolling(window=20).mean()
#     data['SMA_50'] = data['Close'].rolling(window=50).mean()
#     data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
#     return data

# # Price Action Features
# def add_price_action(data: pd.DataFrame) -> pd.DataFrame:
#     data['Higher_High'] = (data['High'] > data['High'].shift(1)) & (data['High'] > data['High'].shift(-1))
#     data['Lower_Low'] = (data['Low'] < data['Low'].shift(1)) & (data['Low'] < data['Low'].shift(-1))
#     data['Higher_High'] = data['Higher_High'].astype(int)
#     data['Lower_Low'] = data['Lower_Low'].astype(int)
#     return data

# # Supply and Demand Zones
# def add_supply_demand_zones(data: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
#     data['Supply_Zone'] = data['High'].rolling(lookback).max()
#     data['Demand_Zone'] = data['Low'].rolling(lookback).min()
#     return data

# # Support and Resistance Levels
# def add_support_resistance(data: pd.DataFrame, window: int = 5) -> pd.DataFrame:
#     # Identifying local max and min
#     data['Local_Max'] = data.iloc[argrelextrema(data['High'].values, np.greater_equal, order=window)[0]]['High']
#     data['Local_Min'] = data.iloc[argrelextrema(data['Low'].values, np.less_equal, order=window)[0]]['Low']
    
#     # Handling missing values (forward fill)
#     data['Local_Max'].fillna(method='ffill', inplace=True)
#     data['Local_Min'].fillna(method='ffill', inplace=True)
    
#     return data
# def calculate_bollinger_bands(data, window=20, num_std_dev=2):
#     """Calculate Bollinger Bands."""
#     data['MA20'] = data['Close'].rolling(window=window).mean()
#     data['Upper_Band'] = data['MA20'] + (data['Close'].rolling(window=window).std() * num_std_dev)
#     data['Lower_Band'] = data['MA20'] - (data['Close'].rolling(window=window).std() * num_std_dev)
#     return data

# def identify_breakouts(data):
#     """Identify breakout signals based on Bollinger Bands."""
#     data['Breakout_Up'] = np.where(data['Close'] > data['Upper_Band'], 1, 0)
#     data['Breakout_Down'] = np.where(data['Close'] < data['Lower_Band'], -1, 0)
#     return data

# def identify_trend(data, short_window=20, long_window=50):
#     """Identify trend direction using moving averages."""
#     data['SMA_Short'] = data['Close'].rolling(window=short_window).mean()
#     data['SMA_Long'] = data['Close'].rolling(window=long_window).mean()
#     data['Trend'] = np.where(data['SMA_Short'] > data['SMA_Long'], 1, -1)  # 1 for uptrend, -1 for downtrend
#     return data

# # Main function to apply all strategy-based features
# def preprocess_data(filename: str) -> pd.DataFrame:
#     # Read the CSV file assuming the first row contains the correct headers
#     data = pd.read_csv(f"data/{filename}.csv", header=0)
    
#     # Debug: Print the first few rows and columns to check the data
#     print("Columns in CSV:", data.columns)
#     print("First few rows of data:", data.head())

#     # Manually rename columns, we use 'Close' instead of 'Adj Close'
#     data.columns = ['Date', 'Price', 'Close', 'High', 'Low', 'Open', 'Volume']

#     # Ensure 'Date' is parsed as datetime
#     data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

#     # Set 'Date' as the index
#     data.set_index('Date', inplace=True)

#     # Drop rows with NaN in essential columns
#     data.dropna(subset=['Price', 'Close', 'High', 'Low', 'Open'], inplace=True)

#     # Convert columns to numeric
#     data['Price'] = pd.to_numeric(data['Price'], errors='coerce')
#     data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
#     data['High'] = pd.to_numeric(data['High'], errors='coerce')
#     data['Low'] = pd.to_numeric(data['Low'], errors='coerce')
#     data['Open'] = pd.to_numeric(data['Open'], errors='coerce')
#     data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce')

#     # Apply feature engineering functions
#     data = add_moving_averages(data)
#     data = add_price_action(data)
#     data = add_supply_demand_zones(data)
#     data = add_support_resistance(data)

#     # Drop any rows with NaN values after transformations
#     data.dropna(inplace=True)

#     # Debug: Print the final transformed data
#     print("Transformed data:", data.head())
#     mpf.plot(data, type='candle', volume=True, style='yahoo')

#     return data


import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import mplfinance as mpf

# Price Action Features
def add_price_action(data: pd.DataFrame) -> pd.DataFrame:
    # Ensure numeric values in High and Low columns
    data['High'] = pd.to_numeric(data['High'], errors='coerce')
    data['Low'] = pd.to_numeric(data['Low'], errors='coerce')
    
    # Drop rows with NaN values in High or Low columns
    data.dropna(subset=['High', 'Low'], inplace=True)

    # Calculate Higher Highs and Lower Lows
    data['Higher_High'] = ((data['High'] > data['High'].shift(1)) & (data['High'] > data['High'].shift(-1))).astype(int)
    data['Lower_Low'] = ((data['Low'] < data['Low'].shift(1)) & (data['Low'] < data['Low'].shift(-1))).astype(int)
    
    return data

# Add Feature Engineering Functions
def calculate_technical_indicators(data):
    # Calculate Simple Moving Averages (SMA)
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    
    # Calculate Exponential Moving Averages (EMA)
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    
    # Calculate Higher Highs and Lower Lows
    data['Higher_High'] = np.where(data['High'] > data['High'].shift(1), 1, 0)
    data['Lower_Low'] = np.where(data['Low'] < data['Low'].shift(1), 1, 0)
    
    # Calculate Supply and Demand Zones (basic example using High/Low averages)
    data['Supply_Zone'] = data['High'].rolling(window=10).mean()
    data['Demand_Zone'] = data['Low'].rolling(window=10).mean()
    
    # Identify Local Maxima and Minima
    data['Local_Max'] = (data['High'] > data['High'].shift(1)) & (data['High'] > data['High'].shift(-1))
    data['Local_Min'] = (data['Low'] < data['Low'].shift(1)) & (data['Low'] < data['Low'].shift(-1))
    data['Local_Max'] = data['Local_Max'].astype(int)
    data['Local_Min'] = data['Local_Min'].astype(int)
    
    return data

def add_price_action_confluence(data: pd.DataFrame) -> pd.DataFrame:
    # Ensure 'Higher_High' and 'Lower_Low' exist
    if 'Higher_High' not in data.columns or 'Lower_Low' not in data.columns:
        data = add_price_action(data)  # Call to ensure these columns are added

    # Add 'Price_Action_Confluence' based on conditions
    data['Price_Action_Confluence'] = ((data['Higher_High'] == 1) & (data['Lower_Low'] == 0)).astype(int)
    
    return data

def add_price_change(data):
    """Calculate the daily price change and add it as a feature."""
    data['Price_Change'] = data['Close'].pct_change()  # Percentage change in closing price
    data['Price_Change'].fillna(0, inplace=True)  # Fill NaN values with 0 (or another strategy)
    return data


# Support and Resistance Levels
def add_support_resistance(data: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    # Identifying local max and min
    data['Local_Max'] = data.iloc[argrelextrema(data['High'].values, np.greater_equal, order=window)[0]]['High']
    data['Local_Min'] = data.iloc[argrelextrema(data['Low'].values, np.less_equal, order=window)[0]]['Low']

    # Forward fill missing values to maintain structure
    data['Local_Max'].fillna(method='ffill', inplace=True)
    data['Local_Min'].fillna(method='ffill', inplace=True)

    # Add resistance and support signals
    data['Resistance_Broken'] = (data['Close'] > data['Local_Max']).astype(int)
    data['Support_Broken'] = (data['Close'] < data['Local_Min']).astype(int)
    return data

# Trend Analysis Using Price and Support/Resistance
def add_trend_analysis(data: pd.DataFrame) -> pd.DataFrame:
    # Trend Direction based on support and resistance
    data['Trend_Up'] = ((data['Resistance_Broken'] == 1) & (data['Close'] > data['Open'])).astype(int)
    data['Trend_Down'] = ((data['Support_Broken'] == 1) & (data['Close'] < data['Open'])).astype(int)

    # Combined trend signal
    data['Trend'] = np.where(data['Trend_Up'] == 1, 1, np.where(data['Trend_Down'] == 1, -1, 0))
    return data

# Signal Calculation for Buy/Sell
def add_signals(data: pd.DataFrame) -> pd.DataFrame:
    # Ensure 'Price_Action_Confluence' exists
    if 'Price_Action_Confluence' not in data.columns:
        raise KeyError("'Price_Action_Confluence' is missing. Ensure it is created before calling add_signals.")

    # Add Buy and Sell signals
    data['Buy_Signal'] = ((data['Trend'] == 1) & (data['Price_Action_Confluence'] == 1)).astype(int)
    data['Sell_Signal'] = ((data['Trend'] == -1) & (data['Price_Action_Confluence'] == 1)).astype(int)
    
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
    data = add_price_change(data)
    data = add_price_action(data)
    data = add_price_action_confluence(data)  # Create 'Price_Action_Confluence'
    data = add_support_resistance(data)
    data = add_trend_analysis(data)
    data = add_signals(data)
    data = calculate_technical_indicators(data)

    # Drop any rows with NaN values after transformations
    data.dropna(inplace=True)

    # Debug: Print the final transformed data
    print("Transformed data:", data.head())
    mpf.plot(data, type='candle', volume=True, style='yahoo')

    return data
