import streamlit as st
import yfinance as yf
import pandas as pd
from kalman import micro_stock_price_filtering_fourier_withlinear, micro_frequency_domain_filtering, remove_weekends, \
    remove_gaps
from datastore import micro_turbo_list, micro_fine_tune_list, companies_by_index
import time
import datetime


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
        stock_data.columns = [ticker]
        stock_data.reset_index(inplace=True)
        if 'Datetime' in stock_data.columns:
            stock_data.rename(columns={'Datetime': 'Date'}, inplace=True)

        return stock_data

    return pd.DataFrame()


companies_by_index_in_use = companies_by_index.copy()

# Dropdown to select index
selected_index = st.selectbox("Select Index", list(companies_by_index_in_use.keys()), key="index_page2")

# Get the tickers for the selected index
tickers = list(companies_by_index_in_use[selected_index].values())

# Dropdown to select a single stock for intraday data
selected_stock = st.selectbox("Select a Stock for Intraday Data", tickers, key="stock_page2")

with st.expander("Micro Adjust Settings "):
    # New inputs for intraday data: number of days and interval
    intraday_days = st.number_input("Select number of days for intraday data", min_value=1, max_value=30, value=10,
                                    step=1)
    intraday_interval = st.selectbox("Select intraday interval", options=["5m", "10m", "15m", "30m", "60m"],
                                     index=2)  # Default "15m"

# Initial data download
intraday_data = download_intraday_data(selected_stock, intraday_days, intraday_interval)

print(intraday_data.head())


# Step 1: Save each date's data into separate dataframes
def split_data_by_day(df):
    """Split the dataframe into separate dataframes for each day"""
    # Extract date from datetime
    df['Day'] = df['Date'].dt.date
    # Group by day and store in a dictionary
    daily_data = {day: group for day, group in df.groupby('Day')}
    return daily_data


# Step 2 (updated): Normalize time index AND normalize prices
def normalize_time_and_price(daily_data):
    """Normalize each day's data to have time-only index and prices starting at 1"""
    normalized_days = {}
    for day, df in daily_data.items():
        # Create a copy to avoid SettingWithCopyWarning
        df_copy = df.copy()

        # Set index to just the time component
        df_copy['Time'] = df_copy['Date'].dt.time
        df_copy.set_index('Time', inplace=True)

        # Normalize prices to start at 1
        first_price = df_copy[selected_stock].iloc[0]
        df_copy[selected_stock] = df_copy[selected_stock] / first_price

        normalized_days[day] = df_copy
    return normalized_days


# Step 3 (updated): Plot overlay with normalized prices
def plot_normalized_overlay(normalized_days):
    """Plot all days' normalized data on the same axes"""
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter, HourLocator
    import datetime

    fig, ax = plt.subplots(figsize=(12, 6))

    for day, df in normalized_days.items():
        # Convert time to datetime today (we'll just use the time component)
        times = [datetime.datetime.combine(datetime.date.today(), t) for t in df.index]
        ax.plot(times, df[selected_stock], label=day.strftime('%Y-%m-%d'))

    # Format x-axis
    ax.xaxis.set_major_locator(HourLocator(interval=2))  # Tick every 2 hours
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))  # Format as hours:minutes
    ax.set_xlabel('Time of Day')
    ax.set_ylabel('Normalized Price (Starting at 1.0)')
    ax.set_title(f'Normalized Intraday Price Patterns for {selected_stock}')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)

    # Add a horizontal line at 1.0 for reference
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    return fig


# Also add this import at the top of your file:
def day_pattern():
    # Add this to your existing code after the data download:
    if not intraday_data.empty:
        # Step 1: Split by day
        # Step 1: Split by day
        daily_data = split_data_by_day(intraday_data)

        # Step 2: Normalize time index and prices
        normalized_days = normalize_time_and_price(daily_data)

        # Step 3: Plot normalized overlay
        st.pyplot(plot_normalized_overlay(normalized_days))







