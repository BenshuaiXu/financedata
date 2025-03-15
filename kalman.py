import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.dates import DateFormatter
from sklearn.linear_model import LinearRegression


class KalmanFilter:
    def __init__(self, initial_position=0, initial_velocity=0, initial_acceleration=0, initial_uncertainty=1, measurement_noise=5, process_noise=3):
        # Initial state variables
        self.position = initial_position
        self.velocity = initial_velocity
        self.acceleration = initial_acceleration

        # Uncertainties for each state component
        self.position_uncertainty = initial_uncertainty
        self.velocity_uncertainty = initial_uncertainty
        self.acceleration_uncertainty = initial_uncertainty

        # Process and measurement noise
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

    def predict(self, dt=1):
        """Predicts the next state based on the current state and time step dt."""
        # Update the position using current velocity and acceleration
        self.position += self.velocity * dt + 0.5 * self.acceleration * (dt ** 2)
        # Update the velocity using current acceleration
        self.velocity += self.acceleration * dt

        # Increase uncertainty to account for prediction step
        self.position_uncertainty += self.process_noise
        self.velocity_uncertainty += self.process_noise
        self.acceleration_uncertainty += self.process_noise

    def update(self, measurement_position):
        """Updates the state based on a new position measurement."""
        # Kalman gain for position update
        kalman_gain_position = self.position_uncertainty / (self.position_uncertainty + self.measurement_noise)

        # Update position based on measurement
        position_residual = measurement_position - self.position
        self.position += kalman_gain_position * position_residual

        # Update position uncertainty
        self.position_uncertainty = (1 - kalman_gain_position) * self.position_uncertainty

        # Approximate velocity from position changes (finite differences)
        new_velocity = position_residual  # This assumes `dt=1` for simplicity

        # Kalman gain for velocity update
        kalman_gain_velocity = self.velocity_uncertainty / (self.velocity_uncertainty + self.measurement_noise)
        self.velocity += kalman_gain_velocity * (new_velocity - self.velocity)
        self.velocity_uncertainty = (1 - kalman_gain_velocity) * self.velocity_uncertainty

        # Approximate acceleration from velocity changes
        new_acceleration = new_velocity - self.velocity  # This also assumes `dt=1`

        # Kalman gain for acceleration update
        kalman_gain_acceleration = self.acceleration_uncertainty / (self.acceleration_uncertainty + self.measurement_noise)
        self.acceleration += kalman_gain_acceleration * (new_acceleration - self.acceleration)
        self.acceleration_uncertainty = (1 - kalman_gain_acceleration) * self.acceleration_uncertainty

    def get_state(self):
        """Returns the current state estimate: position, velocity, acceleration."""
        return self.position, self.velocity, self.acceleration


def micro_frequency_domain_filtering(ticker_df, top_frequencies=150):
    """
    This function performs frequency domain filtering on the stock price data
    by applying a Fourier transform, removing low-frequency components,
    and reconstructing the price series.

    :param ticker_df: DataFrame containing stock price data (timestamp as index, price as first column).
    :param top_frequencies: Number of top frequency components to retain.
    :return: Processed ticker_df with the filtered price data.
    """
    # Extract the price series (assumed to be in the first column)
    prices = ticker_df.iloc[:, 0].values
    ticker_df["original_price"] = prices

    # Add 20 padding values to the beginning and the end (each equal to the first and last prices)
    first_value = prices[0]
    last_value = prices[-1]
    padding_length = 20
    padded_prices = np.concatenate(([first_value] * padding_length, prices, [last_value] * padding_length))

    # Step 1: Transform to frequency domain using FFT
    fft_prices = np.fft.fft(padded_prices)

    # # Step 2: Plot original prices in time domain
    # fig, ax = plt.subplots(figsize=(12, 6))
    # ax.plot(ticker_df.index, prices, label='Original Price', color='orange')  # Set plot color to orange
    # ax.set_title('Original Stock Price in Time Domain')
    # ax.set_xlabel('Date')
    # ax.set_ylabel('Price')
    # ax.legend()
    # st.pyplot(fig)

    # Step 3: Plot Frequency Spectrum
    # Calculate the frequency values (using FFT frequencies)
    freqs = np.fft.fftfreq(len(padded_prices))

    # Calculate the magnitudes of the Fourier coefficients
    magnitudes = np.abs(fft_prices)

    # # Plot the frequency domain
    # fig, ax = plt.subplots(figsize=(12, 6))
    # ax.plot(freqs[:len(freqs) // 2], magnitudes[:len(magnitudes) // 2], label='Frequency Spectrum')
    # ax.set_title('Frequency Spectrum of Stock Price')
    # ax.set_xlabel('Frequency')
    # ax.set_ylabel('Magnitude')
    # ax.legend()
    # st.pyplot(fig)

    # Step 4: Remove low-frequency components by keeping top `top_frequencies` frequencies
    # Find the indices of the top magnitudes
    top_indices = np.argsort(magnitudes)[-top_frequencies:]

    # Create a mask to keep only the top frequencies
    mask = np.zeros_like(fft_prices, dtype=bool)
    mask[top_indices] = True

    # Apply the mask to keep only the desired frequencies
    filtered_fft_prices = fft_prices * mask

    # Step 5: Revert back to time domain using Inverse FFT
    filtered_prices = np.fft.ifft(filtered_fft_prices).real

    # Remove the padding from the filtered prices
    filtered_prices = filtered_prices[padding_length:-padding_length]

    # # Step 6: Plot the filtered price in time domain
    # fig, ax = plt.subplots(figsize=(12, 6))
    # ax.plot(ticker_df.index, filtered_prices, label='Filtered Price', color='orange')
    # ax.set_title('Filtered Stock Price (Top Frequencies) in Time Domain')
    # ax.set_xlabel('Date')
    # ax.set_ylabel('Price')
    # ax.legend()
    # st.pyplot(fig)

    # # Step 7: Plot both original and filtered prices for comparison
    # fig, ax = plt.subplots(figsize=(12, 6))
    # ax.plot(ticker_df.index, prices, label='Original Price', color='orange')
    # ax.plot(ticker_df.index, filtered_prices, label='Simulated Price', color='#000000')
    # ax.set_title('Original vs Simulated Stock Price')
    # # ax.set_xlabel('Date')
    # ax.set_ylabel('Price')
    # ax.legend()
    # ax.tick_params(axis='x', labelrotation=45)
    #
    # st.pyplot(fig)

    # Step 8: Save the filtered price into ticker_df and return it
    ticker_df['filtered_price'] = filtered_prices

    return ticker_df


def micro_stock_price_filtering_fourier_withlinear(ticker_df, company_name, measurement_noise=5, backdate=60):
    prices = ticker_df['filtered_price']
    # Initialize the Kalman Filter with the first price, velocity, and acceleration
    initial_position = prices[0]
    initial_velocity = (prices[1] - prices[0])
    initial_acceleration = (prices[2] - 2 * prices[1] + prices[0])  # Estimate initial acceleration
    kalman_filter = KalmanFilter(initial_position=initial_position, initial_velocity=initial_velocity,
                                 initial_acceleration=initial_acceleration, measurement_noise=measurement_noise)

    predicted_prices = [initial_position]
    velocities = [initial_velocity]
    accelerations = [initial_acceleration]
    log_return_velocities = [0]
    log_return_accelerations = [0]

    # Run the filter on stock data
    for i in range(2, len(prices)):
        current_price = prices[i]

        # Predict the next state
        kalman_filter.predict()

        # Update the filter with the current measurement
        kalman_filter.update(current_price)

        # Get the predicted position, velocity, and acceleration
        predicted_position, predicted_velocity, predicted_acceleration = kalman_filter.get_state()
        predicted_prices.append(predicted_position)
        velocities.append(predicted_velocity)
        accelerations.append(predicted_acceleration)

        # Calculate log return velocity and acceleration
        if predicted_position != 0:  # Avoid division by zero
            log_return_velocities.append(np.log((predicted_velocity + predicted_position) / predicted_position))
            log_return_accelerations.append(np.log((predicted_acceleration + predicted_position) / predicted_position))
        else:
            log_return_velocities.append(0)
            log_return_accelerations.append(0)

    # Add one more prediction for the next day based on the last observation
    kalman_filter.predict()
    predicted_position, predicted_velocity, predicted_acceleration = kalman_filter.get_state()
    predicted_prices.append(predicted_position)
    velocities.append(predicted_velocity)
    accelerations.append(predicted_acceleration)

    if predicted_position != 0:
        log_return_velocities.append(np.log((predicted_velocity + predicted_position) / predicted_position))
        log_return_accelerations.append(np.log((predicted_acceleration + predicted_position) / predicted_position))
    else:
        log_return_velocities.append(0)
        log_return_accelerations.append(0)

    # Convert predicted values into arrays for easier plotting
    predicted_prices = np.array(predicted_prices)
    velocities = np.array(velocities)
    accelerations = np.array(accelerations)
    log_return_velocities = np.array(log_return_velocities)
    log_return_accelerations = np.array(log_return_accelerations)

    # # Extend data for plotting
    # original_price = ticker_df.iloc[1:, 0]
    # filtered_price_adjusted = ticker_df['filtered_price'].iloc[1:]

    # if not pd.api.types.is_datetime64_any_dtype(ticker_df["DateTime"]):
    #     ticker_df["DateTime"] = pd.to_datetime(ticker_df["DateTime"], errors="coerce")

    # # Format datetime as string for x-axis labels
    # ticker_df["FormattedTime"] = ticker_df["DateTime"].dt.strftime('%Y-%m-%d %H:%M')

    # # Extract correct_dates (excluding the first row)
    # correct_dates = ticker_df["FormattedTime"].iloc[1:]
    # # Get the last value of correct_dates (a formatted datetime string)
    # last_date_str = correct_dates.iloc[-1]
    # # Convert the last date string to a datetime object
    # last_date = pd.to_datetime(last_date_str)
    # # Calculate the next day
    # next_day = last_date + pd.Timedelta(days=1)
    # # Format the next day as a string (if needed)
    # next_day_str = next_day.strftime('%Y-%m-%d %H:%M')

    # extended_correct_dates = ticker_df["FormattedTime"].iloc[1:]

    # extended_dates = filtered_price_adjusted.index.append(
    #     pd.to_datetime([filtered_price_adjusted.index[-1] + pd.Timedelta(days=1)]))
    # extended_original_price = original_price.append(
    #     pd.Series([np.nan], index=[extended_dates[-1]]))
    # extended_filtered_price = filtered_price_adjusted.append(
    #     pd.Series([filtered_price_adjusted.iloc[-1]], index=[extended_dates[-1]]))
    # extended_correct_date = correct_dates.append(
    #     pd.Series([next_day_str], index=[extended_dates[-1]]))

    # Extend data for plotting
    original_price = ticker_df.iloc[1:, 0]
    filtered_price_adjusted = ticker_df['filtered_price'].iloc[1:]

    if not pd.api.types.is_datetime64_any_dtype(ticker_df["DateTime"]):
        ticker_df["DateTime"] = pd.to_datetime(ticker_df["DateTime"], errors="coerce")

    # Format datetime as string for x-axis labels
    ticker_df["FormattedTime"] = ticker_df["DateTime"].dt.strftime('%Y-%m-%d %H:%M')

    # Extract correct_dates (excluding the first row)
    correct_dates = ticker_df["FormattedTime"].iloc[1:]

    # Get the last value of correct_dates (a formatted datetime string)
    last_date_str = correct_dates.iloc[-1]
    # Convert the last date string to a datetime object
    last_date = pd.to_datetime(last_date_str)
    # Calculate the next day
    next_day = last_date + pd.Timedelta(days=1)
    # Format the next day as a string (if needed)
    next_day_str = next_day.strftime('%Y-%m-%d %H:%M')

    extended_correct_dates = ticker_df["FormattedTime"].iloc[1:]

    extended_dates = pd.concat([
        filtered_price_adjusted.index.to_series(),
        pd.Series(pd.to_datetime(filtered_price_adjusted.index[-1]) + pd.Timedelta(days=1))
    ])

    extended_original_price = pd.concat([
        original_price,
        pd.Series([np.nan], index=[extended_dates.iloc[-1]])
    ])

    extended_filtered_price = pd.concat([
        filtered_price_adjusted,
        pd.Series([filtered_price_adjusted.iloc[-1]], index=[extended_dates.iloc[-1]])
    ])

    extended_correct_date = pd.concat([
        correct_dates,
        pd.Series([next_day_str], index=[extended_dates.iloc[-1]])
    ])


    predicted_prices_series = pd.Series(predicted_prices, index=extended_dates)
    velocities_series = pd.Series(velocities, index=extended_dates)
    accelerations_series = pd.Series(accelerations, index=extended_dates)
    log_return_velocities_series = pd.Series(log_return_velocities, index=extended_dates)
    log_return_accelerations_series = pd.Series(log_return_accelerations, index=extended_dates)

    new_df = pd.DataFrame({
        'close_price': extended_original_price,
        'filtered_price': extended_filtered_price,
        'predicted_price': predicted_prices_series,
        'velocity': velocities_series,
        'acceleration': accelerations_series,
        'log_return_velocity': log_return_velocities_series,
        'log_return_acceleration': log_return_accelerations_series,
        'correct_dates': extended_correct_date
    })

    # Step 1: Create a new DataFrame for plotting, excluding non-trading days
    plot_df = new_df[(new_df['filtered_price'].notna()) & (new_df.iloc[:, 0] != 0)].copy()

    # Step 2: Convert the index in the new DataFrame to numerical values
    plot_df['Numeric_Index'] = range(len(plot_df))

    # Step 3: Clean the data for linear regression
    plot_df_cleaned = plot_df.dropna(subset=['close_price'])  # Drop rows with NaN in 'close_price'
    plot_df_cleaned = plot_df_cleaned[np.isfinite(plot_df_cleaned['close_price'])]  # Ensure all values are finite

    # Step 4: Run linear regression on the cleaned data
    if len(plot_df_cleaned) > 1:
        X = plot_df_cleaned['Numeric_Index'].values.reshape(-1, 1)
        y = plot_df_cleaned['close_price'].values

        # Fit the linear regression model
        model = LinearRegression()
        model.fit(X, y)

        # Predict the regression line
        y_pred = model.predict(X)

        # Calculate the 20th and 80th percentiles along the regression line
        residuals = y - y_pred
        lower_percentile = np.percentile(residuals, 20)
        upper_percentile = np.percentile(residuals, 80)

    backdate = -backdate
    # Plot using the numeric index for x-axis
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot Close Price on the primary y-axis (LEFT)
    ax1.plot(plot_df['Numeric_Index'][backdate:], plot_df['close_price'].iloc[backdate:], color='#151515', linewidth=1,
             label='Close Price')  # Black

    # Plot the regression line and percentiles (if applicable)
    if len(plot_df_cleaned) > 1:
        # ax1.plot(plot_df_cleaned['Numeric_Index'], y_pred, color='blue', label='Regression Line')
        # ax1.fill_between(plot_df_cleaned['Numeric_Index'], y_pred + lower_percentile, y_pred + upper_percentile,
        #                  color='lightblue', alpha=0.3, label='20th-80th Percentile')
        ax1.plot(plot_df_cleaned['Numeric_Index'][backdate:], y_pred[backdate:], color='blue', label='Regression Line')
        ax1.fill_between(plot_df_cleaned['Numeric_Index'][backdate:],
                         y_pred[backdate:] + lower_percentile,
                         y_pred[backdate:] + upper_percentile,
                         color='lightblue', alpha=0.3, label='20th-80th Percentile')

        # Get the latest date and corresponding regression value
        latest_index = plot_df_cleaned['Numeric_Index'].iloc[-1]
        latest_date = plot_df.index[-1]
        latest_regression_value = y_pred[-1]

        # Calculate 10th and 90th percentile values
        latest_percentile_10_value = latest_regression_value + lower_percentile
        latest_percentile_90_value = latest_regression_value + upper_percentile

        # Annotations
        ax1.annotate(f"{latest_regression_value:.2f}",
                     xy=(latest_index, latest_regression_value),
                     xytext=(latest_index + 2, latest_regression_value + 3),
                     color='red', fontsize=12,
                     arrowprops=dict(arrowstyle="->", color='red'))

        ax1.annotate(f"{latest_percentile_90_value:.2f}",
                     xy=(latest_index, latest_percentile_90_value),
                     xytext=(latest_index + 2, latest_percentile_90_value + 3),
                     color='blue', fontsize=12,
                     arrowprops=dict(arrowstyle="->", color='blue'))

        ax1.annotate(f"{latest_percentile_10_value:.2f}",
                     xy=(latest_index, latest_percentile_10_value),
                     xytext=(latest_index + 2, latest_percentile_10_value + 3),
                     color='green', fontsize=12,
                     arrowprops=dict(arrowstyle="->", color='green'))

    ax1.set_ylabel('Closed Price', color='#151515')  # Label for the left y-axis
    ax1.tick_params(axis='y', labelcolor='#151515')  # Color the ticks to match the line

    # Plot Momentum (velocity) and Canary on the secondary y-axis (RIGHT)
    ax2 = ax1.twinx()

    # Create color lists for bars
    momentum_colors = ['#ff9000'] * (len(plot_df['Numeric_Index'][backdate:]) - 1) + ['#FF0000']  # Last bar red
    wave_colors = ['#008080'] * (len(plot_df['Numeric_Index'][backdate:]) - 1) + ['#34c759']  # Last bar red

    # Plot bars with the updated color lists
    ax2.bar(plot_df['Numeric_Index'][backdate:], plot_df['log_return_velocity'].iloc[backdate:], color=momentum_colors,
            label='Momentum', alpha=0.5)
    ax2.bar(plot_df['Numeric_Index'][backdate:], plot_df['log_return_acceleration'].iloc[backdate:], color=wave_colors,
            label='Market Wave', alpha=0.7, width=0.6)

    # Set labels and ticks for the secondary y-axis
    ax2.set_ylabel('Momentum & Simulated Market Wave', color='#ff9000')  # Label for the right y-axis
    ax2.tick_params(axis='y', labelcolor='#ff9000')  # Color the ticks to match the bars

    # Annotate the last value of 'close_price'
    last_two_index = plot_df['Numeric_Index'].iloc[-2]
    last_close_price = plot_df['close_price'].iloc[-2]
    ax1.text(last_two_index, last_close_price, f'{last_close_price:.1f}', color='#151515',
             fontsize=10, verticalalignment='bottom', horizontalalignment='right')

    # Add legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='lower left')

    # Add grid lines for the primary y-axis
    ax1.grid(axis='y', color='#808080', linestyle='--', linewidth=0.5)

    # Change the frame (spines) color to grey
    ax1.spines['top'].set_color('#808080')
    ax1.spines['bottom'].set_color('#808080')
    ax1.spines['left'].set_color('#808080')
    ax2.spines['right'].set_color('#808080')

    # Add a title
    plt.title(f'Simulated Market Wave and Momentum of {company_name}', color='#292929')

    # Set xticks with labels
    min_labels = 6
    step = max(1, len(plot_df['Numeric_Index'][backdate:]) // min_labels)
    xticks = plot_df['Numeric_Index'][backdate:][::step]
    xtick_labels = plot_df['correct_dates'].iloc[backdate:][::step]

    # if plot_df['Numeric_Index'].iloc[-1] not in xticks.values:
    #     xticks = pd.concat([xticks, pd.Series([plot_df['Numeric_Index'].iloc[-1]])], ignore_index=True)
    #     xtick_labels = pd.concat([pd.Series(xtick_labels), pd.Series([plot_df['correct_dates'].iloc[-1]])],
    #                              ignore_index=True)

    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xtick_labels, rotation=45, ha='right')  # Rotate 45 degrees and align to the right

    plt.tight_layout()

    # Show the combined plot in Streamlit
    st.pyplot(fig)


def remove_weekends(data):
    # Filter out weekends (Saturday = 5, Sunday = 6)
    data = data[data.index.dayofweek < 5]
    # print(data.head())
    return data


def remove_gaps(data):
    # Group data by day
    data_no_gaps = data.copy()
    # Keep the original datetime in a column for reference
    data_no_gaps["DateTime"] = data_no_gaps.index
    # grouped = data_no_gaps.groupby(data_no_gaps.index.date)
    # Create a new index without time gaps
    new_index = pd.date_range(start=data_no_gaps.index[0].date(), periods=len(data_no_gaps), freq='15T')
    # Reindex the data with the new continuous index
    data_no_gaps.index = new_index

    return data_no_gaps


def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return None