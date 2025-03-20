import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd

# Load dataset
file_path = "dataset.csv"  # Update the path if necessary
df = pd.read_csv(file_path)

# Convert date column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Sort by date
df = df.sort_values(by='date')

# Select features and target
features = ['windspeedscalar', 'temperature_1p5', 'temperature_2', 'relativehumidity', 'stationpressure', 'solarradiation', 'windspeed_20f']
target = 'naturaltemperature_5'

# Normalize data
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[features + [target]])

# Convert DataFrame to NumPy array
data_array = np.array(df_scaled)

# Function to create sequences
def create_sequences(data, target_index, seq_length=24):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length, :-1])  # Features
        y.append(data[i + seq_length, target_index])  # Target
    return np.array(X), np.array(y)

# Create sequences
seq_length = 24
X, y = create_sequences(data_array, target_index=-1, seq_length=seq_length)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build LSTM model
lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(50, return_sequences=False),
    Dense(25, activation='relu'),
    Dense(1)
])

# Compile model
lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train model
lstm_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate model
lstm_loss, lstm_mae = lstm_model.evaluate(X_test, y_test)

# Print results
print(f"LSTM Model Loss: {lstm_loss:.4f}, MAE: {lstm_mae:.4f}")
