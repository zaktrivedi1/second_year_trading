import numpy as np
import matplotlib.pyplot as plt

yahoo = False

# Load the data
point_two_percent = 2
file_name = 'TRES.csv'


def clean_data(data):
    """Removes rows where any non-date column contains non-float values."""
    cleaned_data = []
    for row in data:
        try:
            # Check only the necessary columns (i.e., 1 to 6) for non-float values
            float_row = [float(value) if i > 0 else value for i,
                         value in enumerate(row)]
            cleaned_data.append(float_row)
        except ValueError:
            continue
    cleaned_data = np.array(cleaned_data)
    return cleaned_data


# Extract prices and remove dollar signs
if not yahoo:
    data = np.genfromtxt(file_name, delimiter=',', dtype=str, skip_header=1)
    dates = np.arange(len(data))
    opens = np.array([float(price.replace('$', '')) for price in data[:, 3]])
    highs = np.array([float(price.replace('$', '')) for price in data[:, 4]])
    lows = np.array([float(price.replace('$', '')) for price in data[:, 5]])
    closes = np.array([float(price.replace('$', '')) for price in data[:, 1]])
    # Reverse arrays
    opens = np.flipud(opens)
    highs = np.flipud(highs)
    lows = np.flipud(lows)
    closes = np.flipud(closes)
else:
    data = np.genfromtxt(file_name, delimiter=',', dtype=str, skip_header=1)
    data = clean_data(data)

    if data.size == 0:
        raise ValueError(
            "Cleaned data is empty. Please check the input data and cleaning process.")

    dates = np.arange(len(data))
    opens = data[:, 1].astype(float)
    highs = data[:, 2].astype(float)
    lows = data[:, 3].astype(float)
    closes = data[:, 4].astype(float)


# Function to calculate simple moving averages


def simple_moving_average(data, window):
    sma = np.convolve(data, np.ones(window)/window, mode='valid')
    sma = np.concatenate((np.full(window-1, np.nan), sma))
    return sma

# Function to calculate exponential moving averages


def exponential_moving_average(data, window):
    ema = np.zeros_like(data)
    alpha = 2 / (window + 1)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    return ema

# Function to calculate ATR


def calculate_atr(highs, lows, closes, window):
    tr = np.zeros_like(closes)
    for i in range(1, len(closes)):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]),
                    abs(lows[i] - closes[i-1]))
    atr = exponential_moving_average(tr, window)
    return atr


# Calculate simple moving averages
windows = [98, 350]
moving_averages = {window: simple_moving_average(
    closes, window) for window in windows}

for window in windows:
    print(f'{window}-day SMA at the most recent time:',
          moving_averages[window][-1])

# Calculate ATR
atr_window = 350
atr = calculate_atr(highs, lows, closes, atr_window)
print(f'{atr_window}-day ATR at the most recent time:', atr[-1])

lower_channel = np.min(closes[-20:])
upper_channel = np.max(closes[-20:])
channel_range = (upper_channel - lower_channel) / 2
print('Lower Channel is {0}, Upper Channel is {1} and the Range is {2:.3f}'.format(
    lower_channel, upper_channel, channel_range))

# Plotting
plt.figure(figsize=(12, 6))

# Plot prices
plt.plot(dates, closes, marker='o', color='grey',
         linestyle='-', label='Close Price', alpha=0.5)

# Plot moving averages
for window in windows:
    plt.plot(dates, moving_averages[window], label=f'{window}-day SMA')

# Plot ATR
plt.plot(dates, atr, color='m', linestyle='-', label=f'{atr_window}-day ATR')

plt.axhline(upper_channel, color='g', linestyle='--', label='Upper Channel')
plt.axhline(lower_channel, color='r', linestyle='--', label='Lower Channel')

plt.title(file_name + ' Simple Moving Averages and ATR')
plt.xlabel('Days')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
N = atr[-1] / closes[-1]
unit_size = point_two_percent / N
print('Unit size: {0:.3f}'.format(unit_size))
