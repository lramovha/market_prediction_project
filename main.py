# main.py
from src.data_fetching import fetch_data, save_data
from src.feature_engineering import preprocess_data
from src.model_training import prepare_data, train_model
from src.prediction import predict_and_plot

def analyze_stock(ticker):
    """Function to process stock market data."""
    print(f"Fetching stock data for:", ticker)
    data = fetch_data(ticker)
    save_data(data, f"{ticker}_data")

    # Step 2: Apply feature engineering (based on strategies)
    print("Applying feature engineering for stock market...")
    data = preprocess_data(f"{ticker}_data")

    # Step 3: Train model on engineered features
    print("Training the model for stock market...")
    X_train, X_test, y_train, y_test = prepare_data(data)
    model = train_model(X_train, y_train)
    print("Model trained successfully for stock market!")

    # Step 4: Predict and plot the results
    print("Making predictions and plotting results for stock market...")
    predict_and_plot(f"{ticker}_data")
    print("Prediction and plotting completed for stock market.")

def analyze_forex(ticker):
    """Function to process forex market data."""
    print(f"Fetching forex data for:", ticker)
    data = fetch_data(ticker)
    save_data(data, f"{ticker}_data")

    # Step 2: Apply feature engineering (based on strategies)
    print("Applying feature engineering for forex market...")
    data = preprocess_data(f"{ticker}_data")

    # Step 3: Train model on engineered features
    print("Training the model for forex market...")
    X_train, X_test, y_train, y_test = prepare_data(data)
    model = train_model(X_train, y_train)
    print("Model trained successfully for forex market!")

    # Step 4: Predict and plot the results
    print("Making predictions and plotting results for forex market...")
    predict_and_plot(f"{ticker}_data")
    print("Prediction and plotting completed for forex market.")

def main():
    # Analyze stock data (e.g., AAPL for stock market)
    stock_ticker = 'AAPL'  # Example stock
    analyze_stock(stock_ticker)

    # Analyze forex data (e.g., EURUSD for forex market)
    forex_ticker = 'EURUSD=X'  # Example forex pair (EUR/USD)
    analyze_forex(forex_ticker)

if __name__ == "__main__":
    main()

