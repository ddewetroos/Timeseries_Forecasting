import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
import matplotlib.pyplot as plt
import datetime

# App Title
st.title("Stock Price Forecasting App")

# Sidebar for user input
st.sidebar.header("Stock Selection")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, MSFT):", "AAPL")

# Model Selection
st.sidebar.header("Forecasting Model")
model_option = st.sidebar.selectbox(
    "Choose a Forecasting Model:",
    ("ARIMA", "SARIMA", "LSTM", "Prophet")
)

# Main App
if st.sidebar.button("Fetch and Forecast"):
    try:
        # Fetch data
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=730)  # Last 2 years
        data = yf.download(ticker, start=start_date, end=end_date)

        if data.empty:
            st.error("No data fetched. Please check the ticker symbol.")
        else:
            st.success(f"Data fetched successfully for {ticker}")
            st.write(f"Showing data for {ticker} from {start_date.date()} to {end_date.date()}")
            st.dataframe(data.tail())

            # Plot historical data
            st.line_chart(data["Close"], use_container_width=True)

            # Prepare data for forecasting
            ts_data = data['Close'].dropna()

            if model_option == "ARIMA":
                st.info("Using ARIMA for Forecasting...")
                # ARIMA Model
                model = ARIMA(ts_data, order=(5, 1, 0))  # Default ARIMA order
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=180)  # Forecast next 6 months

            elif model_option == "SARIMA":
                st.info("Using SARIMA for Forecasting...")
                # SARIMA Model
                model = SARIMAX(ts_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
                model_fit = model.fit(disp=False)
                forecast = model_fit.forecast(steps=180)

            elif model_option == "LSTM":
                st.info("Using LSTM for Forecasting...")
                # LSTM Model Preparation
                ts_data_values = ts_data.values.reshape(-1, 1)  # Keep values separate from the index
                scaler = MinMaxScaler(feature_range=(0, 1))
                ts_data_scaled = scaler.fit_transform(ts_data_values)

                def create_sequences(data, sequence_length):
                    X, y = [], []
                    for i in range(len(data) - sequence_length):
                        X.append(data[i:i + sequence_length])
                        y.append(data[i + sequence_length])
                    return np.array(X), np.array(y)

                sequence_length = 60
                X, y = create_sequences(ts_data_scaled, sequence_length)
                split = int(len(X) * 0.8)
                X_train, X_test = X[:split], X[split:]
                y_train, y_test = y[:split], y[split:]

                # Build LSTM Model
                model = Sequential()
                model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
                model.add(Dropout(0.2))
                model.add(LSTM(units=50, return_sequences=False))
                model.add(Dropout(0.2))
                model.add(Dense(units=1))
                model.compile(optimizer='adam', loss='mean_squared_error')

                # Train the Model
                st.info("Training the LSTM model...")
                model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
                st.success("LSTM model trained successfully!")

                # Forecast Using LSTM
                last_sequence = ts_data_scaled[-sequence_length:]
                forecast = []
                for _ in range(180):
                    pred = model.predict(last_sequence.reshape(1, sequence_length, 1), verbose=0)
                    forecast.append(pred[0, 0])
                    last_sequence = np.append(last_sequence[1:], pred, axis=0)

                forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()

            elif model_option == "Prophet":
                st.info("Using Prophet for Forecasting...")
                try:
                    # Prepare data for Prophet
                    prophet_data = pd.DataFrame({
                        "ds": ts_data.index,
                        "y": ts_data.values
                    })

                    # Ensure correct data types
                    prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])
                    prophet_data['y'] = pd.to_numeric(prophet_data['y'], errors='coerce')

                    # Drop any rows with NaN values (just in case)
                    prophet_data = prophet_data.dropna()

                    # Fit Prophet model
                    prophet_model = Prophet(daily_seasonality=True)
                    prophet_model.fit(prophet_data)

                    # Create future dataframe
                    future = prophet_model.make_future_dataframe(periods=180)
                    forecast = prophet_model.predict(future)

                    # Extract forecasted values
                    forecast_dates = forecast['ds'][-180:]
                    forecast_values = forecast['yhat'][-180:].values

                    forecast_df = pd.DataFrame({
                        'Date': forecast_dates,
                        'Forecasted Price': forecast_values
                    })
                except Exception as prophet_error:
                    st.error(f"Error with Prophet model: {prophet_error}")

            # Prepare Forecast DataFrame (for ARIMA, SARIMA, and LSTM)
            if model_option != "Prophet":
                forecast_dates = pd.date_range(start=ts_data.index[-1], periods=181, freq='B')[1:]
                forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecasted Price': forecast})

            # Display Forecast
            st.write("Forecasted Prices for Next 6 Months:")
            st.dataframe(forecast_df)

            # Plot Historical and Forecasted Data
            plt.figure(figsize=(12, 6))
            plt.plot(ts_data.index, ts_data.values, label="Historical Prices", color="blue")
            if model_option == "Prophet":
                plt.plot(forecast_dates, forecast_values, label="Forecasted Prices", color="red")
            else:
                plt.plot(forecast_dates, forecast, label="Forecasted Prices", color="red")
            plt.title(f"Stock Price Forecast for {ticker} ({model_option})")
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend()
            st.pyplot(plt)

    except Exception as e:
        st.error(f"Error: {e}")
