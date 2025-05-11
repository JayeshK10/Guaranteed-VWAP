# === Standard Imports ===

# For interacting with the operating system (e.g., paths)
import os
# For modifying the system path so we can import custom code
import sys

# Add the root directory (two levels up) to the system path so modules in `Code/` can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# === External and Custom Libraries ===

import numpy as np  # For numerical operations
from collections import defaultdict  # For convenient grouping of data
from taq.TAQTradesReader import TAQTradesReader  # Custom binary TAQ trade data reader
from taq.MyDirectories import MyDirectories  # Custom directory manager
import pickle  # To save and load Python objects

# === Trading Constants ===

# Define start and end of trading day in milliseconds since midnight
TRADING_START_MS = 9 * 60 * 60 * 1000 + 30 * 60 * 1000  # 9:30 AM
TRADING_END_MS = 16 * 60 * 60 * 1000                    # 4:00 PM

# Each time bucket represents 30 minutes
BUCKET_SIZE_MS = 30 * 60 * 1000

# Compute how many such buckets fit in a trading day (e.g., 13 buckets)
NUM_BUCKETS = (TRADING_END_MS - TRADING_START_MS) // BUCKET_SIZE_MS

'''
Test - volume_static_test.py
Implements a static Bayesian Dirichlet model for estimating typical intraday volume curves.
'''

# === Volume Modeling Class ===

class BayesianDirichletStatic:
    def __init__(self, prior_strength=10, lambda_prior=5.0):
        """
        Initialize the Bayesian Dirichlet Static model.

        prior_strength: baseline strength of the global prior (shared across buckets).
        lambda_prior: controls weight given to global prior vs. stock-specific history.
        """

        # Strength of the global Dirichlet prior; applied uniformly across all buckets
        self.prior_strength = prior_strength

        # Equivalent number of "prior days" we assume for global prior in per-stock blending
        self.lambda_prior = lambda_prior

        # Initialize the global Dirichlet prior as a uniform vector
        self.alpha_prior_global = np.ones(NUM_BUCKETS) * prior_strength

        # Dictionary to store learned Dirichlet parameters per stock
        self.stock_alphas = {}

    def parse_taq_day(self, date, stock):
        """
        Load TAQ trade data for a single day and compute normalized volume per time bucket.
        Returns: array of fractional volume per bucket or None if data is missing.
        """

        # Construct the expected file path for the TAQ trade binary file
        trades_path = os.path.join(MyDirectories.getTradesDir(), f"{date}/{stock}_trades.binRT")

        # If file does not exist, return None (graceful failure)
        if not os.path.exists(trades_path):
            return None

        # Initialize binary reader for this file
        reader = TAQTradesReader(trades_path)

        # Total number of trades on this day
        N = reader.getN()

        # Create a zero-initialized array for bucketed volume
        bucket_volume = np.zeros(NUM_BUCKETS)

        # Process each trade
        for i in range(N):
            # Get the timestamp of the i-th trade
            ts = reader.getTimestamp(i)

            # Only consider trades within regular trading hours
            if TRADING_START_MS <= ts < TRADING_END_MS:
                # Determine which 30-min bucket this trade belongs to
                bucket = int((ts - TRADING_START_MS) // BUCKET_SIZE_MS)

                # Add the trade size to the correct bucket
                bucket_volume[bucket] += reader.getSize(i)

        # Compute total daily volume
        total = bucket_volume.sum()

        # If there's valid volume, normalize; otherwise return None
        return bucket_volume / total if total > 0 else None

    def fit(self, stock_list, date_list):
        """
        Train the model using historical TAQ data. Builds both:
        - A global Dirichlet posterior (across all stocks).
        - Stock-specific posteriors by smoothing global vs. local info.
        """

        # Dictionary for storing per-stock list of daily curves
        stock_buckets = defaultdict(list)

        # Global pool of all volume curves
        global_bucket_matrix = []

        # Loop through each stock and date
        for stock in stock_list:
            for date in date_list:
                # Attempt to get normalized daily volume curve
                curve = self.parse_taq_day(date, stock)

                # Only keep valid data
                if curve is not None:
                    stock_buckets[stock].append(curve)
                    global_bucket_matrix.append(curve)

        # Combine global prior and all daily curves to form global posterior
        self.alpha_posterior_global = self.alpha_prior_global + np.sum(global_bucket_matrix, axis=0)

        # For each stock, compute blended posterior based on available history
        for stock, curves in stock_buckets.items():
            n = len(curves)  # Number of days for this stock
            S_stock = np.sum(curves, axis=0)  # Aggregate volume per bucket

            # Blending weight: larger n → more weight to stock-specific data
            w = self.lambda_prior / (n + self.lambda_prior)

            # Final Dirichlet alpha: convex combination of global and stock-based evidence
            blended_alpha = w * self.alpha_posterior_global + (1 - w) * S_stock

            # Store result in stock_alpha dictionary
            self.stock_alphas[stock] = blended_alpha

    def predict(self, stock):
        """
        Predict the typical volume profile for a given stock as bucket-wise fractions.
        If no stock-specific posterior is available, fall back to global posterior.
        """

        # Choose stock-specific or global posterior
        alpha = self.stock_alphas.get(stock, self.alpha_posterior_global)

        # Return expected volume distribution under Dirichlet posterior
        return alpha / alpha.sum()

# === Script Entry Point ===

if __name__ == '__main__':

    # Get relevant data directories using helper
    quotes_directory = MyDirectories.getQuotesDir()
    data_dir = MyDirectories.getDataDir()

    # List and sort all dates with available quote data
    list_of_dates = sorted(list(set(os.listdir(quotes_directory))))
    
    # Use first 80% of dates as training data
    split_index = int(0.8 * len(list_of_dates))
    train_dates = list_of_dates[:split_index]

    # Load list of stocks selected for modeling
    stock_selection_path = os.path.join(data_dir, 'top_1500_liquid_stocks.txt')
    with open(stock_selection_path, 'r') as f:
        list_of_stocks = [line.strip() for line in f]
    stock_list = list_of_stocks

    # Initialize the Bayesian Dirichlet model with specified prior
    static_model = BayesianDirichletStatic(prior_strength=10)

    # Train model using selected stocks and dates
    static_model.fit(stock_list, train_dates)

    # Save trained model to file for later use
    static_model_path = os.path.join(data_dir, 'bayesian_dirichlet_static_volume.pkl')
    with open(static_model_path, 'wb') as f:
        pickle.dump(static_model, f)
    print("Static model trained and saved to:", static_model_path)