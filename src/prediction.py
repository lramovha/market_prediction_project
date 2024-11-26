# src/prediction.py
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import mplfinance as mpf
from src.feature_engineering import preprocess_data
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


def load_model(model_path='models/random_forest_model.pkl'):
    """Load the trained machine learning model."""
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"The model file at {model_path} was not found.")

def predict_and_plot(filename: str, model=None):
    """Predict stock prices using the trained model and plot actual vs predicted."""
    # Step 1: Preprocess data (ensure same processing as training)
    try:
        data = preprocess_data(filename)  # Ensure same preprocessing steps as training
    except Exception as e:
        raise ValueError(f"Error preprocessing data: {e}")
    
    if model is None:
        model = load_model()  # Load model if not passed as argument

    # Step 2: Prepare feature set for prediction (match training feature set)
    feature_columns = [
        'SMA_20', 'SMA_50', 'EMA_20', 'Higher_High', 'Lower_Low', 'Supply_Zone', 'Demand_Zone', 'Local_Max', 'Local_Min'
    ]
    
    # Ensure the features exist in the data
    missing_features = [feature for feature in feature_columns if feature not in data.columns]
    if missing_features:
        raise ValueError(f"Missing required features in the data: {', '.join(missing_features)}")
    
    features = data[feature_columns]
    
    # Step 3: Predict stock price movements using the model
    try:
        data['Prediction'] = model.predict(features)
    except Exception as e:
        raise ValueError(f"Error during prediction: {e}")
    
    # Step 4: Calculate predicted close price as a cumulative change from actual close
    data['Predicted_Close'] = data['Close'].shift(1) * (1 + data['Prediction'] * 0.01)  # Assume 1% movement for simplicity
    data['Predicted_Close'].fillna(method='bfill', inplace=True)  # Backfill any NaN values in the prediction
    
    # Step 5: Determine if predictions were correct
    data['Actual_Movement'] = np.where(data['Close'] > data['Close'].shift(1), 1, -1)  # 1 if price went up, -1 if down
    data['Prediction_Correct'] = data['Prediction'] == data['Actual_Movement']
    
    # Step 6: Evaluate the model performance
    evaluate_model(data)
    
    # Step 7: Visualize results with color-coded points
    plot_results(data)

    # Step 8: Print number of correct and incorrect predictions
    correct_count = data['Prediction_Correct'].sum()
    incorrect_count = len(data) - correct_count
    print(f"Correct Predictions: {correct_count}")
    print(f"Incorrect Predictions: {incorrect_count}")

# Keep evaluate_model and plot_results unchanged from your original script



def evaluate_model(data):
    """Evaluate the model performance using R^2 and Mean Squared Error."""
    y_true = data['Close']
    y_pred = data['Predicted_Close']
    
    # Check for NaN and infinity in predictions
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        print("Warning: Predicted values contain NaN or infinity. Replacing invalid values.")
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Calculate R-squared and Mean Squared Error
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"Model Evaluation:")
    print(f"R^2: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")

  

def plot_results(data):
    """Plot the actual vs predicted close prices with color-coded points and candlestick chart."""
    # Plot Actual vs Predicted Close Prices
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['Close'], label='Actual Close Price', color='blue')
    
    # Plot predicted price movement (price change)
    ax.plot(data.index, data['Predicted_Close'], label='Predicted Close Price', linestyle='--', color='red')
    
    # Color-coded points: green for correct predictions, red for incorrect
    correct_predictions = data[data['Prediction_Correct']]
    incorrect_predictions = data[~data['Prediction_Correct']]
    ax.scatter(correct_predictions.index, correct_predictions['Predicted_Close'], color='green', label='Correct Prediction', marker='o')
    ax.scatter(incorrect_predictions.index, incorrect_predictions['Predicted_Close'], color='red', label='Incorrect Prediction', marker='x')
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title("Predicted vs Actual Close Price with Prediction Points")
    ax.legend(loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Candlestick chart with mplfinance
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in data.columns for col in required_columns):
        raise ValueError("Missing required columns for candlestick chart.")
    candlestick_data = data.set_index(data.index)[required_columns]
    mpf.plot(candlestick_data, type='candle', volume=True, style='yahoo')



if __name__ == "__main__":
    try:
        predict_and_plot("AAPL_data")
    except Exception as e:
        print(f"Error: {e}")
