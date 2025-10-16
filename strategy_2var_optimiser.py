import re
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import lru_cache


# ---------- Utilities ----------

MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none", "-", "--"}

def parse_price(token: str):
    """Parse a price token into float; returns np.nan on failure."""
    s = str(token).strip().lower()
    if s in MISSING_TOKENS:
        return np.nan
    # remove common clutter
    s = s.replace("$", "").replace(",", "")
    try:
        return float(s)
    except ValueError:
        # last resort: extract first numeric like -1234.56 from messy strings
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        return float(m.group(0)) if m else np.nan


def load_prices_csv(stock_file: str) -> np.ndarray:
    """Load prices from CSV with header, assuming columns: Date, Price."""
    data = np.genfromtxt(
        stock_file,
        delimiter=",",
        dtype=str,
        skip_header=1,
        usecols=(0, 1),
        autostrip=True,
        encoding="utf-8",
    )
    if data.size == 0:
        return np.asarray([], dtype=float)

    # Ensure 2D shape even if only 1 data row
    if data.ndim == 1:
        data = data.reshape(1, -1)

    raw_prices = [parse_price(p) for p in data[:, 1]]
    prices = np.asarray(raw_prices, dtype=float)

    # Drop rows with NaNs (keep only finite prices)
    mask = np.isfinite(prices)
    return prices[mask]


# ---------- EMA (cached) ----------

@lru_cache(maxsize=None)
def ema_cached(data_tuple, window: int):
    """EMA for a tuple of prices; returns numpy array."""
    data = np.asarray(data_tuple, dtype=float)
    if window <= 0:
        raise ValueError("EMA window must be > 0")
    if data.size == 0:
        return np.asarray([], dtype=float)

    alpha = 2.0 / (window + 1.0)
    ema_data = np.empty_like(data, dtype=float)
    ema_data[0] = data[0]
    for i in range(1, len(data)):
        ema_data[i] = alpha * data[i] + (1.0 - alpha) * ema_data[i - 1]
    return ema_data


# ---------- Core calc ----------

def calculate_profits(stock_file, long_ma_range, short_ma_range):
    """
    For one stock file, compute total P&L for each (short, long) MA pair using a simple
    long/short crossover.
    Returns (list_of_results, prices), where list items are (short_window, long_window, total_profit)
    """
    prices = load_prices_csv(stock_file)

    # not enough data to compute anything meaningful
    if prices.size < 3:
        return [], prices

    prices_tuple = tuple(prices.tolist())

    results = []
    for long_ma_window in long_ma_range:
        long_ema_data = ema_cached(prices_tuple, int(long_ma_window))
        for short_ma_window in short_ma_range:
            short_ema_data = ema_cached(prices_tuple, int(short_ma_window))

            # If window > len(prices), EMAs still computed (start from first), safe to proceed.
            position = 0  # 1 long, -1 short, 0 neutral (pre-first signal)
            total_profit = 0.0

            for i in range(1, len(prices)):
                # explicit comparisons avoid precedence pitfalls
                if (short_ema_data[i] > 0) and (long_ema_data[i] > 0):
                    # crossover signals
                    if (short_ema_data[i] > long_ema_data[i]) and (position != 1):
                        position = 1
                    elif (short_ema_data[i] < long_ema_data[i]) and (position != -1):
                        position = -1

                    # accrue simple price-delta P&L
                    if position == 1:
                        total_profit += prices[i] - prices[i - 1]
                    elif position == -1:
                        total_profit += prices[i - 1] - prices[i]

            results.append((int(short_ma_window), int(long_ma_window), float(total_profit)))

    return results, prices


def parallel_calculate_profits(args):
    return calculate_profits(*args)


# ---------- Analysis helpers ----------

def plot_results(results_dict):
    plt.figure(figsize=(12, 8))
    for stock_file, stock_results in results_dict.items():
        if not stock_results:
            continue
        short_ma_windows = [res[0] for res in stock_results]
        long_ma_windows = [res[1] for res in stock_results]
        profits = [res[2] for res in stock_results]
        plt.scatter(short_ma_windows, long_ma_windows, c=profits, cmap="viridis", label=stock_file)

    plt.xlabel("Short Moving Average Window (days)")
    plt.ylabel("Long Moving Average Window (days)")
    plt.colorbar(label="Total Profit/Loss")
    plt.title("Profit/Loss vs. Moving Average Windows")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def find_best_ma_combination(results_dict):
    best_combination = {}
    for stock_file, stock_results in results_dict.items():
        if not stock_results:
            continue
        best_result = max(stock_results, key=lambda x: x[2])
        best_combination[stock_file] = best_result  # (short, long, profit)
    return best_combination


def calculate_annual_percentage_return(best_combination, stock_files):
    total_percentage_return = 0.0
    count = 0

    for stock_file in stock_files:
        if stock_file not in best_combination:
            continue

        best_short_window, best_long_window, _ = best_combination[stock_file]
        _, prices = calculate_profits(stock_file, [best_long_window], [best_short_window])

        if prices.size < 3:
            continue

        short_ema_data = ema_cached(tuple(prices.tolist()), int(best_short_window))
        long_ema_data = ema_cached(tuple(prices.tolist()), int(best_long_window))

        position = 0
        entry_price = None
        cumulative_return = 0.0

        for i in range(1, len(prices)):
            if (short_ema_data[i] > 0) and (long_ema_data[i] > 0):
                # set/flip position and reference price on signal
                if (short_ema_data[i] > long_ema_data[i]) and (position != 1):
                    position = 1
                    entry_price = prices[i]
                elif (short_ema_data[i] < long_ema_data[i]) and (position != -1):
                    position = -1
                    entry_price = prices[i]

                if entry_price is not None and entry_price != 0:
                    if position == 1:
                        cumulative_return += (prices[i] - prices[i - 1]) / entry_price
                    elif position == -1:
                        cumulative_return += (prices[i - 1] - prices[i]) / entry_price

        days = len(prices)
        years = max(days / 252.0, 1e-9)
        total_return_pct = cumulative_return * 100.0
        annual_percentage_return = ((1.0 + total_return_pct / 100.0) ** (1.0 / years) - 1.0) * 100.0

        total_percentage_return += annual_percentage_return
        count += 1

    return (total_percentage_return / count) if count > 0 else 0.0


# ---------- Main ----------

if __name__ == "__main__":
    # Your files here (must exist in the working directory)
    stock_files = [
        "SHEL.csv", "SPY.csv", "QQQM.csv", "COIL.csv", "NGAS.csv", "HOIL.csv", "SGML.csv", "CU.csv",
        "AG.csv", "COT.csv", "SUG.csv", "GBPUSD.csv", "EURUSD.csv", "TRES.csv", "COIN.csv", "USDJPY.csv",
    ]

    # MA grids
    long_ma_range = range(200, 401, 1)   # 200..400
    short_ma_range = range(20, 151, 10)  # 20,30,...,150

    # args for parallel map
    args_list = [(stock_file, long_ma_range, short_ma_range) for stock_file in stock_files]

    # multiprocessing (needs __main__ guard on Windows)
    with Pool(cpu_count()) as pool:
        results_list = pool.map(parallel_calculate_profits, args_list)

    # collect only those with data
    results = {}
    for i, (res_list, prices) in enumerate(results_list):
        if len(res_list) > 0:
            results[stock_files[i]] = res_list
        else:
            print(f"[WARN] Skipping {stock_files[i]} (no valid price rows after cleaning).")

    if not results:
        raise SystemExit("No valid datasets found. Check your CSV formats/paths.")

    # Plot
    plot_results(results)

    # Best combos
    best_combination = find_best_ma_combination(results)
    print("Best moving average combinations for each stock:")
    for fname, (s, l, prof) in best_combination.items():
        print(f"  {fname}: short={s}, long={l}, profit={prof:.2f}")

    # Average annualized %
    avg_ann = calculate_annual_percentage_return(best_combination, list(best_combination.keys()))
    print(
        "The average annual percentage return from all the stock data using the optimal moving averages is: "
        f"{avg_ann:.2f}%"
    )
