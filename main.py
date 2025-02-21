import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from datastore import companies_by_index


# Streamlit app
st.title("Historical Stock Data Downloader")

# Dropdown to select index
selected_index = st.selectbox("Select Index", list(companies_by_index.keys()))

# Slider to select number of years
years = st.slider("Select number of years", min_value=1, max_value=10, value=6)

# Get the tickers for the selected index
tickers = list(companies_by_index[selected_index].values())

# Function to download historical data for a list of tickers
def download_historical_data(tickers, years):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)

    data = pd.DataFrame()

    for ticker in tickers:
        stock_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
        stock_data = stock_data[['Close']].round(2)  # Keep only 'Close' price and round to 2 decimal places
        stock_data.columns = [ticker]  # Rename the column to the ticker symbol
        data = pd.concat([data, stock_data], axis=1)

    return data

# Download historical data
if st.button("Download Historical Data"):
    with st.spinner("Downloading data..."):
        historical_data = download_historical_data(tickers, years)
        historical_data.to_csv(f"{selected_index}_historical_data.csv")
        st.success("Data downloaded successfully!")

# Display the data
if st.checkbox("Show historical data"):
    try:
        historical_data = pd.read_csv(f"{selected_index}_historical_data.csv", index_col=0)
        st.write(historical_data)
    except FileNotFoundError:
        st.error("No data available. Please download the data first.")

# Allow users to download the CSV file
try:
    with open(f"{selected_index}_historical_data.csv", "rb") as file:
        st.download_button(
            label="Download CSV",
            data=file,
            file_name=f"{selected_index}_historical_data.csv",
            mime="text/csv",
        )
except FileNotFoundError:
    st.error("No data available. Please download the data first.")