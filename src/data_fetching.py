# src/data_fetching.py
import yfinance as yf
import pandas as pd

def fetch_data(ticker: str, period='1y', interval='1d') -> pd.DataFrame:
    # Download data from Yahoo Finance
    data = yf.download(ticker, period=period, interval=interval)
    data.dropna(inplace=True)  # Remove any rows with missing values
    return data

def save_data(data: pd.DataFrame, filename: str):
    # Save to CSV
    data.to_csv(f"data/{filename}.csv")

if __name__ == "__main__":
    ticker = 'AAPL'  # Target stock
    data = fetch_data(ticker)
    save_data(data, f"{ticker}_data")
