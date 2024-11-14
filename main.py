# main.py
from src.data_fetching import fetch_data, save_data
from src.feature_engineering import preprocess_data
from src.model_training import prepare_data, train_model
from src.prediction import predict_and_plot

def main():
    # Step 1: Fetch and save data
    ticker = 'AAPL'  # Set the ticker for a single stock, e.g., Apple
    print("Fetching data for:", ticker)
    data = fetch_data(ticker)
    save_data(data, f"{ticker}_data")

    # Step 2: Apply feature engineering (based on strategies)
    print("Applying feature engineering...")
    data = preprocess_data(f"{ticker}_data")

    # Step 3: Train model on engineered features
    print("Training the model...")
    X_train, X_test, y_train, y_test = prepare_data(data)
    model = train_model(X_train, y_train)
    print("Model trained successfully!")

    # Step 4: Predict and plot the results
    print("Making predictions and plotting results...")
    predict_and_plot(f"{ticker}_data")
    print("Prediction and plotting completed.")

if __name__ == "__main__":
    main()
