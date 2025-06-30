import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="Demand Forecasting App", layout="centered")
st.title("📦 Short-Term Demand Forecasting")

# Upload CSV file
uploaded_file = st.file_uploader("Upload Historical Product Demand CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Clean and preprocess
    df['Date'] = pd.to_datetime(df['Date'].str.strip(), errors='coerce')
    df['Order_Demand'] = df['Order_Demand'].str.replace('[()]', '-', regex=True).str.strip()
    df['Order_Demand'] = pd.to_numeric(df['Order_Demand'], errors='coerce')
    df.dropna(subset=['Date', 'Order_Demand'], inplace=True)

    # Most common product-warehouse pair
    most_common_pair = df.groupby(['Product_Code', 'Warehouse']).size().idxmax()
    product, warehouse = most_common_pair

    st.success(f"Forecasting for Product: {product} | Warehouse: {warehouse}")

    df_selected = df[(df['Product_Code'] == product) & (df['Warehouse'] == warehouse)]
    daily_demand = df_selected.groupby('Date')['Order_Demand'].sum().reset_index()
    df_prophet = daily_demand.rename(columns={"Date": "ds", "Order_Demand": "y"})

    # Forecast
    periods = st.slider("Select forecast horizon (days):", 7, 60, 30)
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    # Rename forecast columns for clarity
    forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={
        'ds': 'Date',
        'yhat': 'Predicted Demand',
        'yhat_lower': 'Lower Confidence Interval',
        'yhat_upper': 'Upper Confidence Interval'
    })

    # Plot
    st.subheader("Forecast Plot")
    fig = model.plot(forecast)
    st.pyplot(fig)

    # Show table
    st.subheader("Forecast Table")
    st.dataframe(forecast_display.tail(periods))

    # Download forecast
    csv = forecast_display.to_csv(index=False).encode('utf-8')
    st.download_button("Download Forecast CSV", csv, "forecast.csv", "text/csv")
else:
    st.info("Please upload a CSV file with historical product demand.")
