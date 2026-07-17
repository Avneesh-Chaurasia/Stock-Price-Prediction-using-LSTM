import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ------------------------------------------------
# Page Configuration
# ------------------------------------------------

st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Price Prediction using LSTM")
st.write("Predict the next day's stock closing price using a trained LSTM model.")

# ------------------------------------------------
# Load Model & Scaler
# ------------------------------------------------

@st.cache_resource
def load_lstm_model():
    return load_model("stock_lstm_model.keras")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

model = load_lstm_model()
scaler = load_scaler()

# ------------------------------------------------
# Load Dataset
# ------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("yahoo_stock.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

df = load_data()

# ------------------------------------------------
# Dataset Preview
# ------------------------------------------------

st.subheader("Dataset")

st.dataframe(df.tail())

# ------------------------------------------------
# Historical Price Chart
# ------------------------------------------------

st.subheader("Historical Closing Price")

fig, ax = plt.subplots(figsize=(12,5))

ax.plot(df["Date"], df["Close"])

ax.set_xlabel("Date")
ax.set_ylabel("Closing Price")

st.pyplot(fig)

# ------------------------------------------------
# Prediction
# ------------------------------------------------

st.subheader("Prediction")

# User selects a forecasting date
min_date = df["Date"].iloc[60].date()
max_date = df["Date"].iloc[-1].date()

forecast_date = st.date_input(
    "Select Forecast Date",
    value=max_date,
    min_value=min_date,
    max_value=max_date
)

# Find the selected date in the dataset
selected_idx = df[df["Date"].dt.date == forecast_date].index

if len(selected_idx) == 0:
    st.error("Selected date not found in dataset.")
else:
    idx = selected_idx[0]

    # Need at least 60 previous days
    if idx < 60:
        st.error("Not enough historical data before this date.")
    else:

        close_data = df["Close"].values.reshape(-1,1)

        scaled = scaler.transform(close_data)

        # Previous 60 days before selected date
        last_60 = scaled[idx-60:idx]

        X_test = np.array([last_60])

        prediction = model.predict(X_test, verbose=0)

        prediction = scaler.inverse_transform(prediction)

        actual_price = df.loc[idx, "Close"]

        st.metric(
            "Predicted Closing Price",
            f"${prediction[0][0]:.2f}"
        )

        st.metric(
            "Actual Closing Price",
            f"${actual_price:.2f}"
        )

        st.metric(
            "Difference",
            f"${abs(actual_price - prediction[0][0]):.2f}"
        )

# ------------------------------------------------
# Actual vs Predicted
# ------------------------------------------------

sequence = 60

X = []
y = []

for i in range(sequence, len(scaled)):
    X.append(scaled[i-sequence:i])
    y.append(scaled[i])

X = np.array(X)
y = np.array(y)

pred = model.predict(X)

pred = scaler.inverse_transform(pred)

actual = scaler.inverse_transform(y)

# ------------------------------------------------
# Metrics
# ------------------------------------------------

mae = mean_absolute_error(actual, pred)
rmse = np.sqrt(mean_squared_error(actual, pred))

col1, col2 = st.columns(2)

col1.metric("MAE", f"{mae:.2f}")

col2.metric("RMSE", f"{rmse:.2f}")

# ------------------------------------------------
# Actual vs Predicted Graph
# ------------------------------------------------

st.subheader("Actual vs Predicted")

fig2, ax2 = plt.subplots(figsize=(14,6))

ax2.plot(actual, label="Actual")

ax2.plot(pred, label="Predicted")

ax2.legend()

st.pyplot(fig2)

# ------------------------------------------------
# Last Few Predictions
# ------------------------------------------------

result = pd.DataFrame({
    "Actual": actual.flatten(),
    "Predicted": pred.flatten()
})

st.subheader("Prediction Comparison")

st.dataframe(result.tail(20))