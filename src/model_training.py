# src/model_training.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
from src.feature_engineering import preprocess_data
from sklearn.preprocessing import StandardScaler

# Prepare Data for Training
def prepare_data(data: pd.DataFrame) -> tuple:
    data = data.dropna()
    X = data[['SMA_20', 'SMA_50', 'EMA_20', 'Higher_High', 'Lower_Low', 'Supply_Zone', 'Demand_Zone', 'Local_Max', 'Local_Min']]
    y = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)  # Binary label for next-day up/down
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter Tuning with GridSearchCV
def tune_hyperparameters(X_train, y_train):
    # Define parameter grid for RandomForestClassifier
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    
    # Set up the grid search with cross-validation
    grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5, n_jobs=1, verbose=2)
    
    grid_search.fit(X_train, y_train)
    
    print("Best Hyperparameters:", grid_search.best_params_)
    return grid_search.best_estimator_

# Train the Model (with or without hyperparameter tuning)
def train_model(X_train, y_train, use_tuning=True):
    if use_tuning:
        model = tune_hyperparameters(X_train, y_train)
    else:
        # Default model without tuning
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
    
    # Ensure 'models/' directory exists before saving
    import os
    os.makedirs('models', exist_ok=True)
    
    # Save the trained model
    joblib.dump(model, 'models/random_forest_model.pkl')
    return model

# Evaluate Model Performance and Plot Actual vs Predicted
def evaluate_model(model, X_test, y_test, data):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    # Print confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Plot actual vs predicted stock prices
    plot_actual_vs_predicted(data, y_pred)
    
    # Plot feature importances
    plot_feature_importances(model)

# Plot Actual vs Predicted Stock Prices
def plot_actual_vs_predicted(data, y_pred):
    # Get actual stock prices from the data
    actual_prices = data['Close'][-len(y_pred):]  # Slice to match the length of predictions
    predicted_prices = actual_prices.copy()
    
    # Assume prediction 1 means price goes up and 0 means price goes down
    predicted_prices.iloc[1:] = actual_prices.iloc[:-1]  # Shift prices based on prediction
    
    # Create a plot for actual vs predicted prices
    plt.figure(figsize=(10, 6))
    plt.plot(data.index[-len(y_pred):], actual_prices, label='Actual Price', color='blue')
    plt.plot(data.index[-len(y_pred):], predicted_prices, label='Predicted Price', linestyle='--', color='red')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Actual vs Predicted Stock Prices')
    plt.legend(loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Plot Feature Importances
def plot_feature_importances(model):
    feature_importances = model.feature_importances_
    features = ['SMA_20', 'SMA_50', 'EMA_20', 'Higher_High', 'Lower_Low', 'Supply_Zone', 'Demand_Zone', 'Local_Max', 'Local_Min']
    
    # Create bar plot for feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(features, feature_importances, color='skyblue')
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance from Random Forest')
    plt.tight_layout()
    plt.show()

# Main function
if __name__ == "__main__":
    data = preprocess_data("AAPL_data")
    X_train, X_test, y_train, y_test = prepare_data(data)
    
    # Train model and evaluate performance
    model = train_model(X_train, y_train, use_tuning=True)  # Set use_tuning to False for default model
    evaluate_model(model, X_test, y_test, data)
