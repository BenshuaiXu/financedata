import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_plotly_events import plotly_events
import plotly.graph_objects as go
from datastore import micro_turbo_list, micro_fine_tune_list, companies_by_index
from day_trade_signal import download_intraday_data

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return None



# Function to remove weekends
def remove_weekends(data):
    return data[data.index.dayofweek < 5]  # Keep only Monday-Friday


# Function to remove gaps in data
def remove_gaps(data):
    data_no_gaps = data.copy()
    data_no_gaps["TimeIndex"] = range(len(data_no_gaps))
    data_no_gaps["CorrectDateTime"] = data_no_gaps.index
    data_no_gaps = data_no_gaps.set_index("TimeIndex")
    return data_no_gaps


# Function to plot stock prices
def plot_price(data, ticker_symbol):
    # Compute the number of data points per unique date
    data['Date'] = pd.to_datetime(data.index).date
    date_counts = data['Date'].value_counts().sort_index()
    cumulative_counts = date_counts.cumsum().to_dict()

    # Create a mapping of index positions to corresponding dates
    index_to_date = {}
    counter = 0
    for date, count in date_counts.items():
        for i in range(count):
            index_to_date[counter] = date
            counter += 1

    # Ensure the last data point is included in the x-axis
    index_to_date[counter - 1] = list(date_counts.keys())[-1]

    step = max(1, len(data) // 10)  # Show only 10 labels max
    xticks = data.index[::step]
    xtick_labels = data["CorrectDateTime"].iloc[::step].dt.strftime('%Y-%m-%d %H:%M')

    fig = px.line(
        data,
        x=data.index,  # Use numeric index
        y=ticker_symbol,
        markers=True,
        title=f"Historical Price of {ticker_symbol}"
    )
    fig.update_traces(
        line=dict(color='orange', width=2),
        marker=dict(size=6, color='orangered', line=dict(width=1, color='black'))
    )
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=xticks,
            ticktext=xtick_labels,
            tickangle=-45,
            tickfont=dict(size=10)  # Reduce font size for better visibility
        ),
        xaxis_title="",  # Remove x-axis title

    )

    selected_points = plotly_events(fig, click_event=True)
    if selected_points:
        point = selected_points[0]
        if len(st.session_state.clicked_points) < 10 and point not in st.session_state.clicked_points:
            st.session_state.clicked_points.append((point["x"], point["y"]))

    if st.session_state.clicked_points:

        selected_df = pd.DataFrame(st.session_state.clicked_points, columns=["Index", "Price"])
        selected_df = selected_df.sort_values(by="Index")

        fig_selected = go.Figure()
        fig_selected.add_trace(go.Scatter(
            x=selected_df["Index"],
            y=selected_df["Price"],
            mode="lines+markers",
            line=dict(color='blue', width=2),
            marker=dict(size=8, color='blue', line=dict(width=1, color='black')),
            name="Selected Points",
            showlegend=False
        ))

        colors = ['red', 'green', 'purple', 'orange', 'cyan', 'magenta']

        for i in range(0, len(selected_df) - 2, 2):
            subset = selected_df.iloc[i:i + 3]
            if len(subset) < 3:
                break

            min_price = subset["Price"].min()
            max_price = subset["Price"].max()
            diff = max_price - min_price

            # Determine if it's an upward or downward trend
            if subset["Price"].iloc[0] < subset["Price"].iloc[-1]:  # Uptrend
                fib_levels = {
                    "0%": max_price,
                    "23.6%": max_price - (0.236 * diff),
                    "38.2%": max_price - (0.382 * diff),
                    "50%": max_price - (0.5 * diff),
                    "61.8%": max_price - (0.618 * diff),
                    "78.6%": max_price - (0.786 * diff),
                    "100%": min_price
                }
            else:  # Downtrend
                fib_levels = {
                    "0%": min_price,
                    "23.6%": min_price + (0.236 * diff),
                    "38.2%": min_price + (0.382 * diff),
                    "50%": min_price + (0.5 * diff),
                    "61.8%": min_price + (0.618 * diff),
                    "78.6%": min_price + (0.786 * diff),
                    "100%": max_price
                }

            color = colors[i % len(colors)]

            for level, price in fib_levels.items():
                fig_selected.add_trace(go.Scatter(
                    x=[subset["Index"].iloc[0], subset["Index"].iloc[-1]],
                    y=[price, price],
                    mode="lines",
                    line=dict(color=color, width=1, dash="dash"),
                    showlegend=False
                ))
                fig_selected.add_annotation(
                    x=subset["Index"].iloc[-1],
                    y=price,
                    text=f"{level}",
                    showarrow=True,
                    arrowhead=2,
                    ax=20,
                    ay=-10,
                    font=dict(color=color, size=10)
                )

        fig_selected.update_layout(
            title="Line Graph of Selected Points with Fibonacci Retracement",
            xaxis_title="Index",
            yaxis_title="Price",
            showlegend=False
        )
        st.plotly_chart(fig_selected, use_container_width=True)
        if st.button("Clear Points"):
            st.session_state.clicked_points = []

# Function to render the Streamlit page
def fibonacci_visualization():
    st.title("Fibonacci Retracement")

    if 'clicked_points' not in st.session_state:
        st.session_state.clicked_points = []

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    intraday_days = 20
    intraday_interval = "30m"

    companies_by_index_in_use = companies_by_index.copy()

    # Dropdown to select index
    selected_index = st.selectbox("Select Index", list(companies_by_index_in_use.keys()), key="index_page2")

    # Get the tickers for the selected index
    tickers = list(companies_by_index_in_use[selected_index].values())

    # Dropdown to select a single stock for intraday data
    selected_stock = st.selectbox("Select a Stock for Intraday Data", tickers, key="stock_page2")


        # Initial data download
    intraday_data = download_intraday_data(selected_stock, intraday_days, intraday_interval)

    if intraday_data is not None and "Date" in intraday_data.columns and selected_stock in intraday_data.columns:
        intraday_data = intraday_data[["Date", selected_stock]]
        intraday_data['Date'] = pd.to_datetime(intraday_data['Date'])
        intraday_data.set_index('Date', inplace=True)
    else:
        st.error("CSV must contain 'Date' and ticker symbol columns.")

    if uploaded_file is not None:
        data = load_data(uploaded_file)
        if data is not None and "Date" in data.columns:
            ticker_symbol = data.columns[2]  # Auto-detect the ticker symbol
            data = data[["Date", ticker_symbol]]
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)

            if data.empty:
                st.error("No data found for the given ticker and period.")
            else:
                data_no_gaps = remove_gaps(remove_weekends(data))
                plot_price(data_no_gaps, ticker_symbol)
        else:
            st.error("CSV must contain a 'Date' column and at least one stock price column.")
    else:
        if intraday_data.empty:
            st.error("No data found for the given ticker and period.")
        else:
            # Remove weekends
            intraday_data = remove_weekends(intraday_data)
            data_no_gaps = remove_gaps(intraday_data)
            plot_price(data_no_gaps, selected_stock)



        
