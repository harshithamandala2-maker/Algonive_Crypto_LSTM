# -------------------------------------------------------------
# Cryptocurrency Price Prediction using LSTM
# Algonive Data Science Internship – Task 2
# -------------------------------------------------------------

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import datetime

# Streamlit page configuration
st.set_page_config(page_title="Crypto Price Prediction", layout="wide")

# App title and description
st.title("💰 Cryptocurrency Price Prediction using LSTM")
st.write("""
This web app predicts **future cryptocurrency prices** using a 
Long Short-Term Memory (LSTM) neural network model.
""")

# ------------------ Sidebar Inputs ------------------
st.sidebar.header("⚙️ Settings")
crypto_symbol = st.sidebar.selectbox(
    "Select Cryptocurrency:",
    ["BTC-USD", "ETH-USD", "DOGE-USD", "BNB-USD", "SOL-USD"]
)
start_date = st.sidebar.date_input("Start Date", datetime.date(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())
time_steps = st.sidebar.slider("Sequence Length (Days)", 30, 120, 60)
epochs = st.sidebar.slider("Training Epochs", 5, 50, 10)

# ------------------ Fetch Data ------------------
st.subheader(f"📈 Historical Data for {crypto_symbol}")
data = yf.download(crypto_symbol, start=start_date, end=end_date)
st.dataframe(data.tail())

# Plot price trend
st.line_chart(data["Close"], use_container_width=True)

# ------------------ Preprocess Data ------------------
st.subheader("🔧 Data Preprocessing")

dataset = data['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Create sequences
def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(time_step, len(dataset)):
        X.append(dataset[i - time_step:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size - time_steps:]
X_train, y_train = create_dataset(train_data, time_steps)
X_test, y_test = create_dataset(test_data, time_steps)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

st.write("✅ Data successfully preprocessed and ready for model training.")

# ------------------ LSTM Model ------------------
st.subheader("🧠 Building and Training LSTM Model")

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

with st.spinner("Training the LSTM model... ⏳"):
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
st.success("✅ Model training completed!")

# ------------------ Predictions ------------------
st.subheader("📊 Model Predictions")

predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot actual vs predicted prices
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(y_test_actual, label='Actual Price', color='blue')
ax.plot(predictions, label='Predicted Price', color='red')
ax.set_title(f"{crypto_symbol} Price Prediction")
ax.set_xlabel("Days")
ax.set_ylabel("Price (USD)")
ax.legend()
st.pyplot(fig)

# ------------------ Future Forecast ------------------
st.subheader("🔮 Predict Next Day Price")

last_60_days = scaled_data[-time_steps:]
X_input = np.array(last_60_days).reshape(1, time_steps, 1)
next_day_scaled = model.predict(X_input)
next_day_price = scaler.inverse_transform(next_day_scaled)[0][0]

st.metric(
    label=f"Predicted Next Day {crypto_symbol} Price (USD)",
    value=f"${next_day_price:,.2f}"
)

st.caption("⚠️ Note: Predictions are based on historical data trends and do not guarantee future performance.")
