import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from datastore import companies_by_index

# Streamlit app
st.title("Historical Stock Data Downloader")

# Dropdown to select index
selected_index = st.selectbox("Select Index", list(companies_by_index.keys()))

# Get the tickers for the selected index
tickers = list(companies_by_index[selected_index].values())

# Dropdown to select a single stock for intraday data
selected_stock = st.selectbox("Select a Stock for Intraday Data", tickers)

# Slider to select number of years for historical data
years = st.slider("Select number of years", min_value=1, max_value=6, value=6)

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


# def download_intraday_data(ticker, period="30d", interval="15m"):
#     # Download only price data, auto-adjust removes dividend/split adjustments

#     data = pd.DataFrame()

#     stock_data = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)

#     if not stock_data.empty:
#         # Select only the 'Close' price to avoid extra metadata
#         stock_data = stock_data[['Close']].round(2)

#         # Rename the 'Close' column to the ticker symbol
#         stock_data.columns = [ticker]
#         data = pd.concat([data, stock_data], axis=1)
#         # Ensure the index is a datetime index

#         # Drop any rows with NaN values
#         # Reset index if you want a clean DataFrame without DateTime index
#         data.reset_index(inplace=True)

#     return data


def remove_weekends(data):
    """
    Remove weekend rows from a DataFrame whose index is datetime.
    Only keeps rows where the day of week is less than 5 (Monday=0, Friday=4).
    """
    return data[data.index.dayofweek < 5]

def remove_gaps(data):
    """
    Remove gaps by reindexing the data to a continuous datetime index.
    The new index uses a 15-minute frequency, matching the original interval.
    """
    if data.empty:
        return data
    
    # Calculate the new continuous index from the first to last timestamp
    new_index = pd.date_range(start=data.index[0], end=data.index[-1], freq='15T')
    
    # Reindex the DataFrame and forward-fill missing values (or use other method as needed)
    data_no_gaps = data.reindex(new_index).ffill()
    return data_no_gaps

# def download_intraday_data_original(ticker, period="30d", interval="15m"):
#     """
#     Download intraday data for a ticker using yfinance's history() method, then clean the data by:
#       - Selecting only the 'Close' price
#       - Renaming the column to the ticker symbol
#       - Converting the index to a column named 'Date'
#       - Checking if the values are doubled and issuing a warning
#     """
#     # Create a Ticker object
#     stock = yf.Ticker(ticker)

#     # Download price data
#     stock_data = stock.history(period=period, interval=interval, auto_adjust=True)

#     if not stock_data.empty:
#         # Select only the 'Close' price and round values to 2 decimal places
#         stock_data = stock_data[['Close']].round(2)
#         stock_data = stock_data.reset_index().rename(columns={'Datetime': 'Date'})

#         return stock_data

#     return pd.DataFrame()


def download_intraday_data(ticker, period="30d", interval="15m"):
    """
    Download intraday data for a ticker using yfinance, then clean the data by:
      - Selecting only the 'Close' price
      - Renaming the column to the ticker symbol
      - Removing weekends
      - Removing gaps (reindexing to a continuous 15-minute frequency)
      - Dropping any remaining NaN values
      - Converting the index to a column named 'Date'
    """
    # Download price data (auto_adjust removes dividend/split adjustments)
    stock_data = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)

    if not stock_data.empty:
        # Select only the 'Close' price and round values to 2 decimal places
        stock_data = stock_data[['Close']].round(2)
        
        # Rename the 'Close' column to the ticker symbol
        stock_data.columns = [ticker]
        
        # Ensure the index is a datetime index (usually already the case)
        # stock_data.index = pd.to_datetime(stock_data.index)
        
        # Remove weekends
        # stock_data = remove_weekends(stock_data)
        
        # Remove gaps: reindex the DataFrame to a continuous datetime index
        # stock_data = remove_gaps(stock_data)
        
        # Drop any rows with NaN values (if gaps couldn't be filled)
        # stock_data.dropna(inplace=True)
        
        # Convert index to a column named 'Date'
        stock_data.reset_index(inplace=True)
        # stock_data.rename(columns={'index': 'Date'}, inplace=True)        
        stock_data.rename(columns={'Datetime': 'Date'}, inplace=True)


        
        return stock_data

    return pd.DataFrame()

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
        intraday_data = download_intraday_data(selected_stock)
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