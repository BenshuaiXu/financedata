import streamlit as st
from data_download import finance_data_download
from day_trade_signal import wave_trading
from fibonacci import fibonacci_visualization
from day_pattern_ananlysis import day_pattern



def main():
    # Sidebar with navigation
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Welcome", "Data Download", "Wave Tracking", "Fibonacci", "Pattern"])

    if selection == "Welcome":
        show_welcome_page()
    elif selection == "Data Download":
        finance_data_download()
    elif selection == "Wave Tracking":
        wave_trading()
    elif selection == "Fibonacci":
        fibonacci_visualization()
    elif selection == "Pattern":
        day_pattern()



def show_welcome_page():
    st.title("Welcome to MillionData")
    st.write("""
        This is a simple app for data download and micro market wave tracking.
        Use the sidebar to switch between pages.
        """)


if __name__ == "__main__":
    main()
