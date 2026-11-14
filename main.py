"""Student demo: compare an LSTM model with historical closing prices."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

MODEL_PATH = Path(__file__).resolve().parent / "Stock Predictions Model.keras"
START = "2012-01-01"
END = "2022-12-31"
WINDOW = 100


@st.cache_resource
def get_model():
    return load_model(MODEL_PATH, compile=False)


@st.cache_data(ttl=3600)
def download_prices(symbol):
    # Explicit adjustment keeps the selected price convention visible.
    return yf.download(symbol, start=START, end=END, auto_adjust=True, progress=False)


def show_moving_averages(close, windows):
    st.subheader("Price vs " + " vs ".join(f"MA{w}" for w in windows))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(close.index, close.to_numpy().ravel(), label="Closing price")
    for window in windows:
        ax.plot(close.index, close.rolling(window).mean().to_numpy().ravel(),
                label=f"{window}-day moving average")
    ax.set(xlabel="Date", ylabel="Adjusted closing price")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)


def main():
    st.set_page_config(page_title="Stock Price Prediction", layout="wide")
    st.title("Stock Market Predictor")
    st.caption("Student learning demo • Historical comparisons, not future forecasts")
    st.info("Uses historical data from 2012–2022. This is not investment advice.")
    stock = st.text_input("Enter stock symbol", "GOOG").strip().upper()
    if not stock or len(stock.split()) != 1 or "," in stock:
        st.warning("Enter one stock symbol, such as GOOG.")
        return
    try:
        data = download_prices(stock)
    except Exception:
        st.error("Could not download prices. Check the symbol and internet connection.")
        return
    if data.empty or "Close" not in data:
        st.warning("No prices were returned. Try another symbol or retry later.")
        return
    close = data["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(name="Close")
    if close.shape[1] != 1:
        st.warning("Please enter a single stock symbol.")
        return
    close = close.dropna().astype(float)
    split = int(len(close) * 0.8)
    if split < WINDOW or split >= len(close):
        st.warning("Not enough historical prices for a 100-day input window.")
        return
    st.subheader("Stock data")
    st.dataframe(data)
    for windows in ((50,), (50, 100), (100, 200)):
        show_moving_averages(close, windows)

    # Fit only to the earlier training portion, never the held-out test prices.
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(close.iloc[:split])
    test_values = scaler.transform(close.iloc[split - WINDOW:])
    x = np.array([test_values[i - WINDOW:i] for i in range(WINDOW, len(test_values))])
    if not MODEL_PATH.exists():
        st.error(f"Put {MODEL_PATH.name} beside main.py, then restart the app.")
        return
    try:
        predictions = get_model().predict(x, verbose=0)
        predicted_prices = scaler.inverse_transform(predictions).ravel()
    except Exception:
        st.error("Could not load or run the model. Check the TensorFlow/Keras environment.")
        return
    actual_prices = close.iloc[split:].to_numpy().ravel()
    dates = close.index[split:]
    st.subheader("Original price vs predicted price")
    st.caption("Each point uses the previous 100 observed prices. No future dates are generated.")
    if stock != "GOOG":
        st.warning("The training notebook uses GOOG. Other symbols reuse that model without retraining.")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, actual_prices, label="Original price", color="green")
    ax.plot(dates, predicted_prices, label="Predicted price", color="red")
    ax.set(xlabel="Date", ylabel="Adjusted closing price")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("The original training scaler was not supplied. This demo reconstructs a scaler "
               "from downloaded training prices; exact training-time preprocessing is not verified.")


if __name__ == "__main__":
    main()
