import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import matplotlib.pyplot as plt


# Step 1: Fetch historical Bitcoin price data
def fetch_data():
    # Download Bitcoin data from Yahoo Finance, explicitly setting auto_adjust
    data = yf.download('BTC-USD', start='2020-01-01', end='2025-05-01', interval='1d', auto_adjust=False)
    return data['Close'].values.reshape(-1, 1)


# Step 2: Prepare data for LSTM
def prepare_data(data, time_steps=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i - time_steps:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Split into train and test sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return X_train, X_test, y_train, y_test, scaler


# Step 3: Build and train LSTM model
def build_model(time_steps):
    model = Sequential()
    # Use Input layer to define input shape
    model.add(Input(shape=(time_steps, 1)))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# Step 4: Make predictions and visualize
def predict_and_plot(model, X_test, y_test, scaler, data):
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform([y_test])

    plt.figure(figsize=(10, 6))
    plt.plot(y_test.T, label='Actual Price')
    plt.plot(predictions, label='Predicted Price')
    plt.title('Bitcoin Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.savefig('prediction_plot.png')
    plt.close()


# Main execution
if __name__ == "__main__":
    # Note: If you see a urllib3 NotOpenSSLWarning, it is due to LibreSSL in your environment.
    # Consider updating your SSL library to OpenSSL 1.1.1+ or pinning urllib3 to a compatible version.

    # Fetch and prepare data
    data = fetch_data()
    time_steps = 60
    X_train, X_test, y_train, y_test, scaler = prepare_data(data, time_steps)

    # Build and train model
    model = build_model(time_steps)
    model.fit(X_train, y_train, epochs=25, batch_size=32, verbose=1)

    # Predict and visualize
    predict_and_plot(model, X_test, y_test, scaler, data)

    # Save model in native Keras format
    model.save('crypto_price_model.keras')
