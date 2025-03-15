import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from datastore import companies_by_index


# st.title("Historical Stock Data Downloader")

# # Dropdown to select index
# selected_index = st.selectbox("Select Index", list(companies_by_index.keys()))

# # Get the tickers for the selected index
# tickers = list(companies_by_index[selected_index].values())

# # Dropdown to select a single stock for intraday data
# selected_stock = st.selectbox("Select a Stock for Intraday Data", tickers)

# # Slider to select number of years for historical data
# years = st.slider("Select number of years", min_value=1, max_value=6, value=6)

# # New inputs for intraday data: number of days and interval
# intraday_days = st.number_input("Select number of days for intraday data", min_value=1, max_value=30, value=25, step=1)
# intraday_interval = st.selectbox("Select intraday interval", options=["1m", "2m", "5m", "15m", "30m", "60m"], index=4)  # Default "15m"

# Function to clean data by replacing outliers
def clean_data(data):
    for ticker in data.columns:
        mean_value = data[ticker].mean()
        lower_bound = mean_value / 5
        upper_bound = mean_value * 5

        # Identify outliers
        outliers = (data[ticker] < lower_bound) | (data[ticker] > upper_bound)

        # Replace outliers with the previous value
        data[ticker] = data[ticker].where(~outliers, data[ticker].shift(1))

    return data


# Function to download historical data for a list of tickers
def download_historical_data(tickers, years):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)

    data = pd.DataFrame()

    for ticker in tickers:
        stock_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
        if not stock_data.empty:  # Only add data if it's not empty
            stock_data = stock_data[['Close']].round(2)  # Keep only 'Close' price and round to 2 decimal places
            stock_data.columns = [ticker]  # Rename the column to the ticker symbol
            data = pd.concat([data, stock_data], axis=1)

    # Drop columns with all NaN values
    data = data.dropna(axis=1, how='all')

    # Clean the data by replacing outliers
    data = clean_data(data)

    return data


# Updated function to download intraday data
def download_intraday_data(ticker, days, interval):
    """
    Download intraday data for a ticker using yfinance with the specified period and interval.
    After downloading, the function:
      - Selects only the 'Close' price and rounds it
      - Renames the column to the ticker symbol
      - Resets the index and renames it to 'Date'
    """
    period = f"{days}d"  # e.g., "30d" if days=30
    stock_data = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)

    if not stock_data.empty:
        # Select only the 'Close' price and round values to 2 decimal places
        stock_data = stock_data[['Close']].round(2)

        # Rename the 'Close' column to the ticker symbol
        stock_data.columns = [ticker]

        # Convert index to a column named 'Date'
        stock_data.reset_index(inplace=True)
        # Rename the index column to 'Date' if necessary
        if 'Datetime' in stock_data.columns:
            stock_data.rename(columns={'Datetime': 'Date'}, inplace=True)

        return stock_data

    return pd.DataFrame()


def finance_data_download():
    st.title("Historical Stock Data Downloader")

    # Dropdown to select index
    selected_index = st.selectbox("Select Index", list(companies_by_index.keys()))

    # Get the tickers for the selected index
    tickers = list(companies_by_index[selected_index].values())

    # Dropdown to select a single stock for intraday data
    selected_stock = st.selectbox("Select a Stock for Intraday Data", tickers)

    # Slider to select number of years for historical data
    years = st.slider("Select number of years", min_value=1, max_value=6, value=6)

    # New inputs for intraday data: number of days and interval
    intraday_days = st.number_input("Select number of days for intraday data", min_value=1, max_value=30, value=25,
                                    step=1)
    intraday_interval = st.selectbox("Select intraday interval", options=["1m", "2m", "5m", "15m", "30m", "60m"],
                                     index=4)  # Default "15m"

    # Download historical data
    if st.button("Download Historical Data"):
        with st.spinner("Downloading historical data..."):
            historical_data = download_historical_data(tickers, years)
            if not historical_data.empty:
                historical_data.to_csv(f"{selected_index}_historical_data.csv")
                st.success("Historical data downloaded and cleaned successfully!")
            else:
                st.error("No historical data available for the selected index and time period.")

    # Download intraday data
    if st.button("Download Intraday Data"):
        with st.spinner("Downloading intraday data..."):
            intraday_data = download_intraday_data(selected_stock, intraday_days, intraday_interval)
            if not intraday_data.empty:
                intraday_data.to_csv(f"{selected_stock}_intraday_data.csv")
                st.success("Intraday data downloaded successfully!")
            else:
                st.error("No intraday data available for the selected stock.")

    # Display historical data
    if st.checkbox("Show historical data"):
        try:
            historical_data = pd.read_csv(f"{selected_index}_historical_data.csv", index_col=0)
            st.write(historical_data)
        except FileNotFoundError:
            st.error("No historical data available. Please download the data first.")

    # Display intraday data
    if st.checkbox("Show intraday data"):
        try:
            intraday_data = pd.read_csv(f"{selected_stock}_intraday_data.csv", index_col=0)
            st.write(intraday_data)
        except FileNotFoundError:
            st.error("No intraday data available. Please download the data first.")

    # Allow users to download the historical CSV file
    try:
        with open(f"{selected_index}_historical_data.csv", "rb") as file:
            st.download_button(
                label="Download Historical CSV",
                data=file,
                file_name=f"{selected_index}_historical_data.csv",
                mime="text/csv",
            )
    except FileNotFoundError:
        st.error("No historical data available. Please download the data first.")

    # Allow users to download the intraday CSV file
    try:
        with open(f"{selected_stock}_intraday_data.csv", "rb") as file:
            st.download_button(
                label="Download Intraday CSV",
                data=file,
                file_name=f"{selected_stock}_intraday_data.csv",
                mime="text/csv",
            )
    except FileNotFoundError:
        st.error("No intraday data available. Please download the data first.")
