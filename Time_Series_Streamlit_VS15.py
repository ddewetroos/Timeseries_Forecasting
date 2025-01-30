import streamlit as st
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# App Title
st.title("📈 Stock Price Forecasting App")

# Auto-run trigger from URL parameters
query_params = st.experimental_get_query_params()
auto_run = query_params.get("run", ["false"])[0] == "true"

# Sidebar
st.sidebar.header("🔍 Stock Selection")
ticker = st.sidebar.text_input("Enter Stock Ticker:", "AAPL").upper()

st.sidebar.header("📊 Forecasting Model")
model_option = st.sidebar.selectbox("Choose a Forecasting Model:", ("ARIMA", "SARIMA", "LSTM"))

if st.sidebar.button("Fetch and Forecast") or auto_run:  # Automatically run if triggered from URL
    try:
        # Fetch Stock Data
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=730)
        data = yf.download(ticker, start=start_date, end=end_date)

        if data.empty:
            st.error("⚠ No data fetched. Check ticker symbol.")
        else:
            st.success(f"✅ Data fetched successfully for {ticker}")
            st.line_chart(data["Close"], use_container_width=True)
            ts_data = data['Close'].dropna()

            # Forecasting
            forecast_df = None
            if model_option == "ARIMA":
                model = ARIMA(ts_data, order=(5,1,0))
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=180)
            elif model_option == "SARIMA":
                model = SARIMAX(ts_data, order=(1,1,1), seasonal_order=(1,1,1,12))
                model_fit = model.fit(disp=False)
                forecast = model_fit.forecast(steps=180)
            elif model_option == "LSTM":
                ts_data_values = ts_data.values.reshape(-1, 1)
                scaler = MinMaxScaler(feature_range=(0,1))
                ts_data_scaled = scaler.fit_transform(ts_data_values)

                def create_sequences(data, seq_length=60):
                    X, y = [], []
                    for i in range(len(data) - seq_length):
                        X.append(data[i:i + seq_length])
                        y.append(data[i + seq_length])
                    return np.array(X), np.array(y)

                sequence_length = 60
                X, y = create_sequences(ts_data_scaled)
                split = int(len(X) * 0.8)
                X_train, X_test = X[:split], X[split:]
                y_train, y_test = y[:split], y[split:]

                model = Sequential([
                    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
                    Dropout(0.2),
                    LSTM(units=50, return_sequences=False),
                    Dropout(0.2),
                    Dense(units=1)
                ])
                model.compile(optimizer='adam', loss='mean_squared_error')
                model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

                last_sequence = ts_data_scaled[-sequence_length:]
                forecast = []
                for _ in range(180):
                    pred = model.predict(last_sequence.reshape(1, sequence_length, 1), verbose=0)
                    forecast.append(pred[0,0])
                    last_sequence = np.append(last_sequence[1:], pred, axis=0)
                forecast = scaler.inverse_transform(np.array(forecast).reshape(-1,1)).flatten()

            forecast_dates = pd.date_range(start=ts_data.index[-1], periods=181, freq='B')[1:]
            forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecasted Price': forecast})

            # Display Forecast
            st.subheader("🔮 Forecasted Prices for Next 6 Months")
            st.dataframe(forecast_df)

            # Plot Forecast
            plt.figure(figsize=(12, 6))
            plt.plot(ts_data.index, ts_data.values, label="Historical Prices", color="blue")
            plt.plot(forecast_df['Date'], forecast_df['Forecasted Price'], label="Forecasted Prices", color="red")
            plt.legend()
            st.pyplot(plt)

            # Download Forecast
            csv = forecast_df.to_csv(index=False)
            st.download_button("📥 Download Forecast Data", csv, f"{ticker}_forecast.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Error: {e}")
