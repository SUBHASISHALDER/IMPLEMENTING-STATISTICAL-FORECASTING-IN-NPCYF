import pandas as pd
import numpy as np
import streamlit as st
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

# Function to check stationarity
def check_stationarity(data):
    result = adfuller(data)
    return result[1]  # p-value

# Function to difference the data
def difference_data(data):
    return data.diff().dropna()

# Function to fit ARIMA model
def fit_arima_model(data, order):
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    return model_fit

# Streamlit application
def main():
    st.title("Time Series Forecasting with ARIMA")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.dataframe(df.head())

        # Select the column for forecasting
        column_name = st.selectbox("Select the column for forecasting", df.columns)

        # Check for stationarity
        st.subheader("Check for Stationarity")
        p_value = check_stationarity(df[column_name])
        st.write(f"ADF Statistic: {p_value}")
        if p_value < 0.05:
            st.success("The series is stationary.")
        else:
            st.warning("The series is non-stationary. Differencing will be applied.")

            # Apply differencing
            df[column_name] = difference_data(df[column_name])
            st.write("Differenced Data Preview:")
            st.line_chart(df[column_name])

        # Fit ARIMA model
        st.subheader("Fit ARIMA Model")
        p = st.number_input("Select p (AR order)", min_value=0, value=1)
        d = st.number_input("Select d (Differencing order)", min_value=0, value=1)
        q = st.number_input("Select q (MA order)", min_value=0, value=1)

        if st.button("Fit Model"):
            model_fit = fit_arima_model(df[column_name].dropna(), (p, d, q))
            st.success("Model fitted successfully!")

            # Forecasting
            forecast_steps = st.number_input("Number of steps to forecast", min_value=1, value=5)
            forecast = model_fit.forecast(steps=forecast_steps)
            st.write("Forecasted Values:")
            st.write(forecast)

            # Plotting
            plt.figure(figsize=(10, 5))
            plt.plot(df[column_name].index[-50:], df[column_name].values[-50:], label='Historical Data')
            plt.plot(range(len(df[column_name]), len(df[column_name]) + forecast_steps), forecast, label='Forecast', color='red')
            plt.title("Forecast vs Historical Data")
            plt.xlabel("Time")
            plt.ylabel(column_name)
            plt.legend()
            st.pyplot(plt)

if __name__ == "__main__":
    main()