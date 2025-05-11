# === Standard Imports ===

import os
import sys

# Add project root directory to Python path to enable module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))  # Add Code directory

# === External and Data Science Libraries ===

import numpy as np  # For numerical operations and array manipulations
import pandas as pd  # For tabular data processing and dataframe operations
import logging  # For configurable logging and debugging
import pickle  # For loading and saving Python objects

# === Project-Specific Imports ===

from taq.MyDirectories import MyDirectories  # Utility for directory management
from scipy.optimize import minimize_scalar  # Optimization utility (e.g., tuning kappa)
from Project.src.dynamic_volume import DynamicVolumeModel  # Dynamic volume predictor
from Project.src.volume_model import BayesianDirichletStatic  # Static volume profile estimator
from taq.TAQQuotesReader import TAQQuotesReader  # Reader for TAQ quote binary files
from taq.MyDirectories import MyDirectories  # Reimported (can be removed if already imported)

# === Visualization Tools ===

import matplotlib.pyplot as plt  # Plotting library
import seaborn as sns  # Statistical visualization library (used with matplotlib)

# === Global Constants ===

NUM_BUCKETS = 13       # Number of 30-minute intervals in a trading day
sigma = 0.05           # Volatility parameter for price simulation
eta = 0.142            # Temporary impact coefficient
lambda_ac = 1          # Risk aversion weight in guaranteed VWAP cost function
beta = 0.5             # Power-law exponent for impact decay term


# === Price Simulation Function ===

'''
Unit Test:
1. test_arrival_price.py
2. test_simulate_price_path.py
'''

def compute_arrival_price(stock_code: str, date: str) -> float:
    """
    Compute the arrival price for a given stock and date.
    
    The arrival price is defined as the average of the first five mid-quote prices 
    observed after 9:30 AM. I am using this a starting price for market price. 

    Args:
        stock_code (str): Ticker symbol of the stock (e.g., 'AAPL').
        date (str): Date string in format 'YYYY-MM-DD' or as required by your directory.

    Returns:
        float: The computed arrival price, or 0.0 if insufficient data is available.
    """

    # Define time boundaries in milliseconds since midnight
    START_TS = 9 * 60 * 60 * 1000 + 30 * 60 * 1000     # 9:30 AM
    END_TS = 16 * 60 * 60 * 1000                       # 4:00 PM

    # Construct the full path to the TAQ quotes binary file
    # This assumes files are named like '<stock>_quotes.binRQ' under a date subfolder
    quotes_path = MyDirectories.getQuotesDir() + f'/{date}/{stock_code}_quotes.binRQ'

    # Initialize the quote reader for the binary file
    quotes_reader = TAQQuotesReader(quotes_path)

    # Initialize accumulators for the sum of prices and the number of quotes collected
    arrival_mid_quote_price_sum = 0.0
    counter = 0

    # Loop through all available quotes in chronological order
    for i in range(quotes_reader.getN()):
        # Get the timestamp of the current quote (in milliseconds since midnight)
        timestamp = quotes_reader.getTimestamp(i)

        # If we already have 5 quotes, we can stop early
        if counter >= 5:
            break

        # If the quote is after 4:00 PM, we stop processing further quotes
        if timestamp >= END_TS:
            break

        # If the quote is before the market opens (9:30 AM), skip it
        if timestamp < START_TS:
            continue

        # Retrieve the mid-quote price (assumes it's already mid, or close enough)
        price = quotes_reader.getPrice(i)

        # Accumulate the price and increment the counter
        arrival_mid_quote_price_sum += price
        counter += 1

    # Return the average of collected quotes, or 0.0 if none were found
    return arrival_mid_quote_price_sum / counter if counter > 0 else 0.0


def simulate_price_path(start_price=100.0, sigma=0.005, buckets=12, seed=42):
    """
    Simulate a geometric Brownian motion-like price path over a number of buckets.

    Args:
        start_price (float): Initial price of the asset.
        sigma (float): Standard deviation of returns (volatility).
        buckets (int): Number of time intervals to simulate.
        seed (int): Seed for random number generator for reproducibility.

    Returns:
        np.ndarray: Simulated price path as an array of prices.
    """
    np.random.seed(seed)  # Ensure reproducibility
    noise = np.random.normal(loc=0, scale=sigma, size=buckets)  # Generate return shocks
    returns = 1 + noise  # Convert shocks to multiplicative returns
    price_path = [start_price]  # Initialize path with starting price

    # Iteratively simulate each step in the price path
    for r in returns:
        price_path.append(price_path[-1] * r)

    return np.array(price_path)



# === Market Volume Simulation Function ===
'''
Unit Test: test_simulate_market_volume.py
'''
def simulate_market_volume_log_normal(my_trade: float, sigma: float = 0.2) -> float:
    """
    Simulate market volume using a log-normal distribution centered on your planned trade size.

    Args:
        my_trade (float): Your trade volume for the current bucket.
        sigma (float): Volatility parameter for the log-normal distribution.

    Returns:
        float: Simulated total market volume for the current bucket.
    """
    multiplier = np.random.lognormal(mean=0, sigma=sigma)  # Random log-normal factor
    return my_trade * multiplier  # Scale your trade size to simulate real market volume


# === Optimal Trajectory - For minimum risk ===
'''
Unit Test: test_optimal_kappa_cost.py
'''
def get_optimal_kappa_and_cost_from_trajectory(
    remaining_shares: float,
    total_shares: float,
    price_path: np.ndarray
) -> tuple[float, float, np.ndarray]:
    """
    Given future prices and the remaining shares to be traded, this function
    searches over a range of `kappa` values to determine the optimal 
    execution trajectory that minimizes the deviation from the market VWAP,
    penalized by variation (risk).

    Args:
        remaining_shares (float): Number of shares left to trade.
        total_shares (float): Total shares for the full execution horizon.
        price_path (np.ndarray): The future (remaining) price path from time t to T.

    Returns:
        Tuple containing:
        - best_kappa (float): The kappa value yielding minimal deviation from VWAP.
        - variation (float): Variance of deviations (risk estimate).
        - best_schedule (np.ndarray): The optimal trade schedule using sinh weights.
    """

    T = len(price_path)  # Remaining number of trading intervals

    # === Define execution schedule generator ===
    def sinh_trajectory(kappa):
        """
        Generate a trade schedule using a sinh-shaped trajectory parameterized by kappa.
        For small |kappa|, default to uniform schedule.
        """
        time = np.arange(1, T + 1)

        # For very small kappa, approximate sinh(kappa*(1 - t/T)) ≈ 1 (i.e., uniform)
        if abs(kappa) < 1e-4:
            weights = np.ones(T)
        else:
            weights = np.sinh(kappa * (1 - time / T))

        weights /= weights.sum()  # Normalize to ensure the weights sum to 1
        return remaining_shares * weights  # Scale to match total shares to trade

    # === Define custom execution VWAP with impact ===
    def my_vwap(trade_schedule):
        """
        Compute execution VWAP based on the provided trade schedule and impact model.
        Incorporates temporary price impact using eta, sigma, and beta.
        """
        trade_frac = trade_schedule / (total_shares + 1e-8)  # Normalized trade sizes
        impact = eta * sigma * trade_frac ** beta  # Nonlinear impact model
        executed_price = price_path + impact  # Adjusted execution prices
        return np.sum(executed_price * trade_schedule) / np.sum(trade_schedule)

    # === Define simple market VWAP (no volume assumption) ===
    def market_vwap():
        """
        Compute market VWAP assuming equal volume in each time bucket.
        """
        return np.mean(price_path)

    # === Initialize best trajectory search ===
    best_kappa = None
    best_schedule = None
    best_deviation = float('inf')
    deviations = []  # Store deviation per kappa for variance computation

    # === Grid search over possible kappa values ===
    for kappa in np.linspace(-40, 40, 1000):
        schedule = sinh_trajectory(kappa)  # Generate trade schedule
        my = my_vwap(schedule)             # Compute our execution VWAP
        mv = market_vwap()                 # Compute benchmark market VWAP
        deviation = abs(my - mv)          # Objective: minimize absolute deviation
        deviations.append(deviation)

        # Update best schedule if current one is better
        if deviation < best_deviation:
            best_kappa = kappa
            best_schedule = schedule
            best_deviation = deviation

    # === Compute variation (risk term) across all deviations ===
    variation = np.var(deviations)

    return best_kappa, variation, best_schedule


# === Execution Price Impact ===
'''
Unit Test: test_execution_impact_with_decay.py
'''
def compute_execution_impact_with_decay(
    kappa: float,
    total_volume: float,
    current_time: float,
    total_time: float,
    prev_trading_rate: float
) -> tuple[float, float]:
    """
    Compute the total temporary market impact at the current time bucket, incorporating:
    - Impact from current trading volume based on a sinh-shaped schedule
    - Residual impact from the previous trading bucket (decayed)

    Args:
        kappa (float): Execution aggressiveness parameter controlling trade curvature.
        total_volume (float): Total remaining volume to execute over [current_time, total_time].
        current_time (float): Current time step (bucket index, e.g., t).
        total_time (float): Total number of time steps remaining (e.g., T).
        prev_trading_rate (float): Trading rate used in the previous bucket (v_{t-1}).

    Returns:
        Tuple:
            - total_impact (float): Combined impact (current + decayed prior).
            - trading_rate (float): Current trading rate v_t (used in this bucket and passed forward).
    """

    # === Compute Current Trading Rate v_t Based on Sinh-Shaped Schedule ===
    # sinh(kappa * T) is the normalization constant for the trajectory
    sinh_kT = np.sinh(kappa * total_time)

    # cosh term gives relative trade intensity at time t
    cosh_term = np.cosh(kappa * (total_time - current_time))

    # Compute trading rate using sinh schedule:
    # v_t = [kappa * cosh(kappa * (T - t)) / sinh(kappa * T)] * total_volume
    trading_rate = (kappa * cosh_term / sinh_kT) * total_volume

    # === Current Impact from Trading at Rate v_t ===
    # Uses nonlinear power-law model: ησ|v|^β
    current_impact = eta * sigma * (abs(trading_rate) ** beta)

    # === Residual Impact from Previous Bucket ===
    # Decays linearly with decay factor φ = 0.5 (i.e., 50% of previous impact lingers)
    residual_impact = eta * sigma * (abs(prev_trading_rate) ** beta) * 0.5

    # === Total Temporary Impact ===
    # Sum of current impact and residual decayed impact
    total_impact = current_impact + residual_impact

    return total_impact, trading_rate


# === Trade Stimulation  ===
'''
Unit Test: test_vwap_simulation.py
'''
def vwap_simulation(stock, total_shares, date, static_model, dynamic_model):
    """
    Simulates a guaranteed VWAP execution strategy using both static and dynamic volume models.
    It adjusts trade volumes per bucket based on forecasts and market behavior, and evaluates
    cost and performance versus market VWAP.

    Args:
        stock (str): Ticker symbol.
        total_shares (int): Total number of shares to execute.
        date (str): Trading date (used for data access).
        static_model: Pre-trained static volume model.
        dynamic_model: Pre-trained dynamic volume model.

    Returns:
        result (dict): Metrics and arrays from the simulation (VWAPs, volumes, prices).
        df_logs (pd.DataFrame): Detailed per-bucket log of execution stats.
    """

    # === Step 0: Initialization ===

    # Predict baseline volume distribution using the static model
    static_pred = static_model.predict(stock)

    # Allocate arrays to store agent execution and market behavior
    my_executed_volume = np.zeros(NUM_BUCKETS)      # Volume traded in each bucket
    executed_volume_till_now = 0                    # Cumulative traded volume so far
    market_traded_volume = np.zeros(NUM_BUCKETS)    # Market volume in each bucket
    market_volume_fractions = np.zeros(NUM_BUCKETS) # Market volume as a fraction of total

    # Initialize price series and impact-tracked prices
    arrival_price = compute_arrival_price(stock, date)  # Use open price as starting point
    price_path = simulate_price_path(start_price=arrival_price, sigma=sigma)  # Simulate price movement
    my_execution_price = np.zeros(NUM_BUCKETS)  # Agent’s execution prices (with impact)

    # Initialize impact tracking variables
    impact = 0.0
    prev_rate = 0.0
    risk = 0.0

    # Container to store logging info for each bucket
    bucket_logs = []

    # === Step 1: Iterate Over Time Buckets ===
    for t in range(NUM_BUCKETS):
        log = {}  # Temporary dictionary for this bucket's logs
        log['bucket'] = t

        logging.info(f"\n--- Bucket {t} ---")

        # Compute how many shares still need to be traded
        remaining_shares = max(0, total_shares - executed_volume_till_now)
        log['remaining_shares'] = remaining_shares

        # Get the static prediction for the current bucket
        static_bucket = static_pred[t]
        log['static_bucket'] = static_bucket

        # === Step 2: Determine Shares to Trade in this Bucket ===
        if t == 0:
            # First bucket: follow static model exactly
            vol_fraction = static_bucket
            my_current_trade = total_shares * vol_fraction
        elif t < NUM_BUCKETS - 1:
            # Intermediate buckets: use dynamic model prediction
            vol_fraction = dynamic_model.predict_single_bucket(
                market_volume_fractions, static_pred, t
            )
            my_current_trade = total_shares * vol_fraction
        else:
            # Last bucket: dump remaining volume to ensure completion
            vol_fraction = float(remaining_shares / total_shares)
            my_current_trade = remaining_shares

        # Convert trade volume to integer shares
        my_current_trade = int(my_current_trade)
        log['vol_fraction'] = vol_fraction
        log['my_current_trade'] = my_current_trade

        # Update running volume and record execution
        my_executed_volume[t] = my_current_trade
        executed_volume_till_now += my_current_trade
        log['executed_volume_till_now'] = executed_volume_till_now

        # === Step 3: Optimize Future Execution Using VWAP Deviation ===
        if t + 2 >= len(price_path):
            # Final or penultimate bucket: no uncertainty → use default kappa
            kappa, cost = 1e-6, 0
        else:
            # Optimize kappa to follow optimal execution path in future buckets
            kappa, cost, schedule = get_optimal_kappa_and_cost_from_trajectory(
                remaining_shares=remaining_shares - my_current_trade,
                total_shares=total_shares,
                price_path=price_path[t+1:]
            )
        risk += cost
        log['kappa'] = kappa
        log['cost'] = cost

        # === Step 4: Simulate Market Volume ===
        market_current_trade = simulate_market_volume_log_normal(my_current_trade)
        market_current_trade = int(market_current_trade)
        market_traded_volume[t] = market_current_trade
        market_volume_fractions[t] = float(market_current_trade / total_shares)

        log['market_current_trade'] = market_current_trade
        log['market_volume_fraction'] = market_volume_fractions[t]

        # === Step 5: Apply Impact and Compute Execution Price ===
        impact, curr_rate = compute_execution_impact_with_decay(
            kappa=kappa,
            total_volume=remaining_shares,
            current_time=t,
            total_time=NUM_BUCKETS,
            prev_trading_rate=prev_rate
        )
        prev_rate = curr_rate

        my_execution_price[t] = price_path[t] + impact
        log['impact'] = impact
        log['curr_rate'] = curr_rate
        log['market_price'] = price_path[t]
        log['my_execution_price'] = my_execution_price[t]

        # Save this bucket’s log
        bucket_logs.append(log)

        # Optionally log to file/console
        for k, v in log.items():
            logging.info(f"{k}: {v}")

    # === Step 6: Post-Simulation Metrics ===

    # Compute agent's volume-weighted average execution price
    my_vwap = (my_execution_price * my_executed_volume).sum() / my_executed_volume.sum()

    # Compute market VWAP from simulated price and volume
    market_vwap = (price_path * market_traded_volume).sum() / market_traded_volume.sum()

    # Guaranteed VWAP cost: deviation from market VWAP plus risk penalty
    guaranteed_cost = abs(my_vwap - market_vwap) + lambda_ac * risk

    # === Step 7: Final Output ===

    # Return core simulation results
    result = {
        "price_path": price_path,
        "my_execution_price": my_execution_price,
        "my_executed_volume": my_executed_volume,
        "market_traded_volume": market_traded_volume,
        "my_vwap": my_vwap,
        "market_vwap": market_vwap,
        "risk": risk,
        "guaranteed_cost": guaranteed_cost,
    }

    # Convert logs into a transposed DataFrame for analysis and plotting
    df_logs = pd.DataFrame(bucket_logs).set_index('bucket').T

    return result, df_logs


# === Save Results  ===
'''
Print Results
This function logs and saves simulation outputs — including VWAP metrics,
executed volumes, and price paths — for post-analysis.
'''
def print_simulation_results(results: dict, stimuaton_dir: str):
    # === Console Output ===
    print("\n VWAP Execution Simulation Summary\n" + "-"*40)
    print(f"Market VWAP         : {results['market_vwap']:.4f}")
    print(f"My VWAP             : {results['my_vwap']:.4f}")
    print(f"Execution Risk      : {results['risk']:.6f}")
    print(f"Guaranteed Cost     : {results['guaranteed_cost']:.6f}\n")

    print("My Execution Prices:")
    print(np.round(results['my_execution_price'], 4))

    print("\nMarket Mid-Price Path:")
    print(np.round(results['price_path'], 4))

    print("\nMy Executed Volumes (per bucket):")
    print(np.round(results['my_executed_volume'], 2))

    print("\nMarket Traded Volumes (per bucket):")
    print(np.round(results['market_traded_volume'], 2))

    print("-" * 40 + "\n")

    # === Save Results to File ===
    result_path = os.path.join(stimuaton_dir, "simulation_result_logs.txt")
    with open(result_path, 'w') as f:
        f.write("VWAP Execution Simulation Summary\n" + "-"*40 + "\n")
        f.write(f"Market VWAP         : {results['market_vwap']:.4f}\n")
        f.write(f"My VWAP             : {results['my_vwap']:.4f}\n")
        f.write(f"Execution Risk      : {results['risk']:.6f}\n")
        f.write(f"Guaranteed Cost     : {results['guaranteed_cost']:.6f}\n\n")

        f.write("My Execution Prices:\n")
        f.write(str(np.round(results['my_execution_price'], 4)) + "\n\n")

        f.write("Market Mid-Price Path:\n")
        f.write(str(np.round(results['price_path'], 4)) + "\n\n")

        f.write("My Executed Volumes (per bucket):\n")
        f.write(str(np.round(results['my_executed_volume'], 2)) + "\n\n")

        f.write("Market Traded Volumes (per bucket):\n")
        f.write(str(np.round(results['market_traded_volume'], 2)) + "\n")
        f.write("-" * 40 + "\n")

'''
Plot Simulation Results
Generates and saves plots that visually analyze execution behavior,
such as price impact, volume comparison, and cumulative volume trajectory.
'''
def plot_simulation_results(result: dict, df_logs: pd.DataFrame, stimuaton_dir: str):
    # === Create Directory for Saving Plots ===
    plot_dir = os.path.join(stimuaton_dir, "result_plots")
    os.makedirs(plot_dir, exist_ok=True)

    # === 1. Execution vs Market Price Plot ===
    plt.figure()
    plt.plot(result['price_path'], label="Market Price", marker='o')
    plt.plot(result['my_execution_price'], label="My Execution Price", marker='x')
    plt.title("Execution Price vs Market Price")
    plt.xlabel("Bucket")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "exec_vs_market_price.png"))
    plt.close()

    # === 2. Executed Volume vs Market Volume (Bar Plot) ===
    plt.figure()
    plt.bar(range(len(result['my_executed_volume'])), result['my_executed_volume'],
            alpha=0.6, label='My Volume')
    plt.bar(range(len(result['market_traded_volume'])), result['market_traded_volume'],
            alpha=0.4, label='Market Volume')
    plt.title("Executed Volume Per Bucket")
    plt.xlabel("Bucket")
    plt.ylabel("Volume")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "volume_comparison.png"))
    plt.close()

    # === 3. Cumulative Execution Plot ===
    cumulative_volume = np.cumsum(result['my_executed_volume'])  # Aggregate volume
    plt.figure()
    plt.plot(cumulative_volume, marker='o', label='Cumulative My Volume')
    plt.axhline(y=np.sum(result['my_executed_volume']), color='r', linestyle='--',
                label='Total Target')
    plt.title("Cumulative Executed Volume")
    plt.xlabel("Bucket")
    plt.ylabel("Cumulative Volume")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "cumulative_volume.png"))
    plt.close()

    # === Completion Message ===
    print(f"All plots saved in: {plot_dir}")


# === Stimulation Example  ===
if __name__ == '__main__':
    # === Load Pretrained Models ===

    # Load the static volume model (Bayesian Dirichlet-based)
    with open(os.path.join(MyDirectories.getDataDir(), 'bayesian_dirichlet_static_volume.pkl'), 'rb') as f:
        static_model = pickle.load(f)

    # Load the dynamic volume prediction model (Bayesian Ridge-based)
    with open(os.path.join(MyDirectories.getDataDir(), 'bayesian_ridge_dynamic_volume.pkl'), 'rb') as f:
        dynamic_model = pickle.load(f)

    # === Simulation Configuration ===

    stock = 'SPY'                  # Ticker symbol for simulation
    total_shares = 50000           # Total shares to be executed during the trading day
    date = '20070920'              # Trading date for TAQ data and price start

    # Define where to store simulation results and logs
    data_dir = MyDirectories.getDataDir()
    stimuaton_dir = os.path.join(data_dir, f"stimulation_{stock}_{total_shares}")
    os.makedirs(stimuaton_dir, exist_ok=True)  # Ensure directory exists

    # === Set Up Logging ===

    # Log detailed bucket-wise trading info to a text file
    bucket_log_file = os.path.join(stimuaton_dir, 'bucket_logs.txt')
    logging.basicConfig(filename=bucket_log_file, filemode='w', level=logging.INFO, format='%(message)s')

    # === Run VWAP Simulation ===

    # Execute the simulation using the static and dynamic models
    results, df_logs = vwap_simulation(stock, total_shares, date, static_model, dynamic_model)

    # Save the bucket-level log DataFrame (features and outcomes) to CSV
    df_path = os.path.join(stimuaton_dir, 'bucket_feature_matrix.csv')
    df_logs.to_csv(df_path)

    # Print summary results and save them to file
    print_simulation_results(results, stimuaton_dir)

    # Generate and save all plots related to price, volume, and cumulative execution
    plot_simulation_results(results, df_logs, stimuaton_dir)

