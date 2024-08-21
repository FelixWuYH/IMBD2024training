import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt

# Load the CSV file
file_path = 'XXX2.csv'
data = pd.read_csv(file_path)

# Ensure 'datetime' column is in datetime format and set it as the index
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

# Select the required columns for analysis
df = data[['amount']]

# Fill missing values
df.fillna(0, inplace=True)

# Normalize the data using StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Split the data into training and testing sets
train_size = int(len(scaled_data) * 0.8)
train, test = scaled_data[:train_size], scaled_data[train_size:]

# Function to create sequences
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length, 0]  # Predicting amount
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 32
X_train, y_train = create_sequences(train, seq_length)
X_test, y_test = create_sequences(test, seq_length)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Make predictions
predicted_sales = model.predict(X_test)
predicted_sales = scaler.inverse_transform(np.concatenate((predicted_sales, np.zeros((predicted_sales.shape[0], X_test.shape[2]-1))), axis=1))[:, 0]

# Inverse transform actual values
actual_sales = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], X_test.shape[2]-1))), axis=1))[:, 0]

# Visualize the predictions
plt.figure(figsize=(14, 8))
plt.plot(actual_sales, color='blue', label='Actual Sales')
plt.plot(predicted_sales, color='red', label='Predicted Sales')
plt.title('LSTM Sales Prediction')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.legend()
plt.show()