# Cryptocurrency Price Prediction with LSTM
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Step 1: Load Data
# Download Bitcoin (BTC-USD) historical data
data = yf.download('BTC-USD', start='2018-01-01', end='2024-01-01')

# Display the first few rows
print(data.head())

# Step 2: Preprocess Data
# Use 'Close' prices only
close_prices = data['Close'].values.reshape(-1, 1)

# Normalize data between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# Create training datasets
training_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[0:training_size, :]
test_data = scaled_data[training_size:, :]

# Function to create sequences
def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape input for LSTM: [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Step 3: Build LSTM Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 4: Train the Model
model.fit(X_train, y_train, batch_size=32, epochs=50)

# Step 5: Model Evaluation
# Predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Reverse scaling
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Step 6: Plotting Results
# Shift predictions for plotting
look_back = time_step
trainPredictPlot = np.empty_like(scaled_data)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

testPredictPlot = np.empty_like(scaled_data)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(scaled_data)-1, :] = test_predict

# Plot baseline and predictions
plt.figure(figsize=(14,6))
plt.title('Cryptocurrency Price Prediction (BTC-USD)')
plt.xlabel('Days')
plt.ylabel('Close Price USD ($)')
plt.plot(scaler.inverse_transform(scaled_data), label='Actual Price')
plt.plot(trainPredictPlot, label='Training Prediction')
plt.plot(testPredictPlot, label='Testing Prediction')
plt.legend()
plt.show()

# Step 7: Save Model
model.save('model/crypto_price_model.h5')
print("✅ Model saved as 'crypto_price_model.h5'")
