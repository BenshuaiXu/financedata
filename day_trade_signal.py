import streamlit as st
import yfinance as yf
import pandas as pd
from kalman import micro_stock_price_filtering_fourier_withlinear, micro_frequency_domain_filtering, remove_weekends, \
    remove_gaps
from datastore import micro_turbo_list, micro_fine_tune_list, companies_by_index
import time
from fibonacci import fibonacci_visualization


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


def get_latest_price(ticker):
    """
    Fetch the latest price for a given ticker.
    """
    latest_data = yf.download(ticker, period="1d", interval="1m", auto_adjust=True, progress=False)
    if not latest_data.empty:
        return float(latest_data['Close'].iloc[-1])  # Ensure it's a scalar float

        # return latest_data['Close'].iloc[-1]
    return None


def update_data(data, latest_price, ticker_symbol):
    """
    Update the historical data by appending the latest price and removing the oldest price.
    """
    latest_date = data.index[-1] + pd.Timedelta(minutes=30)
    new_row = pd.DataFrame({ticker_symbol: [latest_price]}, index=[latest_date])
    updated_data = pd.concat([data.iloc[1:], new_row])
    return updated_data


def run_analysis(data, ticker_symbol, micro_turbo, micro_fine_tune, micro_date_back):
    """
    Run the analysis on the given data.
    """
    data_no_gaps = remove_gaps(data)

    micro_date_back_to_data_points = micro_date_back * 16
    micro_top_frequencies = 60 + (5 - micro_turbo) * 10 - micro_fine_tune + 1
    micro_fourier_transformed_ticker_df = micro_frequency_domain_filtering(data_no_gaps,
                                                                           top_frequencies=micro_top_frequencies)
    micro_stock_price_filtering_fourier_withlinear(micro_fourier_transformed_ticker_df,
                                                   company_name=ticker_symbol,
                                                   backdate=micro_date_back_to_data_points)


def wave_trading():
    intraday_days = 20
    intraday_interval = "30m"

    companies_by_index_in_use = companies_by_index.copy()

    # Dropdown to select index
    selected_index = st.selectbox("Select Index", list(companies_by_index_in_use.keys()), key="index_page2")

    # Get the tickers for the selected index
    tickers = list(companies_by_index_in_use[selected_index].values())

    # Dropdown to select a single stock for intraday data
    selected_stock = st.selectbox("Select a Stock for Intraday Data", tickers, key="stock_page2")

    with st.expander("Micro Adjust Settings "):
        micro_turbo = st.slider(
            "Micro Turbo",  # Label for the slider
            min_value=1,  # Minimum value
            max_value=10,  # Maximum value
            value=micro_turbo_list[selected_stock],  # Default value
            step=1  # Step size
        )
        micro_fine_tune = st.slider(
            "Micro Fine Tune",  # Label for the slider
            min_value=1,  # Minimum value
            max_value=10,  # Maximum value
            value=micro_fine_tune_list[selected_stock],  # Default value
            step=1  # Step size
        )
        micro_date_back = st.slider(
            "Micro Time Frame",  # Label for the slider
            min_value=2,  # Minimum value
            max_value=35,  # Maximum value
            value=5,  # Default value
            step=1  # Step size
        )

    # Initial data download
    intraday_data = download_intraday_data(selected_stock, intraday_days, intraday_interval)

    if intraday_data is not None and "Date" in intraday_data.columns and selected_stock in intraday_data.columns:
        intraday_data = intraday_data[["Date", selected_stock]]
        intraday_data['Date'] = pd.to_datetime(intraday_data['Date'])
        intraday_data.set_index('Date', inplace=True)
    else:
        st.error("CSV must contain 'Date' and ticker symbol columns.")

    if intraday_data is not None:
        if intraday_data.empty:
            st.error("No data found for the given ticker and period.")
        else:
            # Remove weekends
            intraday_data = remove_weekends(intraday_data)

            # Create a placeholder to clear previous results
            analysis_placeholder = st.empty()

            # Initial analysis
            with analysis_placeholder:
                run_analysis(intraday_data, selected_stock, micro_turbo, micro_fine_tune, micro_date_back)

            # Loop to update data and re-run analysis every 30 minutes
            while True:
                time.sleep(1800)  # Wait for 30 minutes (1800 seconds)

                # Check market status
                ticker_info = yf.Ticker(selected_stock).info
                market_status = ticker_info.get('marketState', 'CLOSED')

                if market_status != "REGULAR":
                    st.warning(f"Market is closed for {selected_stock}. Stopping tracking.")
                    break

                latest_price = get_latest_price(selected_stock)
                if latest_price is not None:
                    st.write(intraday_data.head())
                    st.write(intraday_data.tail())

                    intraday_data = update_data(intraday_data, latest_price, selected_stock)
                    st.write(intraday_data.head())
                    st.write(intraday_data.tail())

                    with analysis_placeholder:
                        st.empty()  # Clear previous output
                        run_analysis(intraday_data, selected_stock, micro_turbo, micro_fine_tune, micro_date_back)

                else:
                    st.error("Failed to fetch the latest price.")

    fibonacci_visualization()

