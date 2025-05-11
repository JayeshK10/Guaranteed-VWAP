# === Standard & System Imports ===
import unittest  # Python unit testing framework
import numpy as np  # Numerical operations
from unittest.mock import patch, MagicMock  # For mocking objects and functions

# Extend sys.path to import modules from parent directories
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# === Project Imports ===
from Project.src.stimulation import vwap_simulation  # Function under test
from Project.src import stimulation  # Module to patch internal functions/constants

# === Global Constants for the Simulation ===
NUM_BUCKETS = 13
sigma = 0.05
eta = 0.142
lambda_ac = 1
beta = 0.5

# === Unit Test Class for vwap_simulation ===
class TestVWAPSimulation(unittest.TestCase):
    """
    Test the full VWAP simulation pipeline using mocked models and functions.
    This isolates internal dependencies to test logic and integration of the simulation engine.
    """

    def test_vwap_simulation_with_mock_models(self):
        """
        Simulates execution for one stock-date pair using controlled mock components.
        Validates expected structure, constraints, and presence of key outputs.
        """

        # === Setup Inputs ===
        stock = "AAPL"
        date = "2024-01-01"
        total_shares = 1000

        # === Mock Static Model ===
        # Return a flat static volume curve (uniform 1/NUM_BUCKETS)
        mock_static_model = MagicMock()
        mock_static_model.predict.return_value = np.ones(stimulation.NUM_BUCKETS) / stimulation.NUM_BUCKETS

        # === Mock Dynamic Model ===
        # Always predict uniform dynamic volume as well
        mock_dynamic_model = MagicMock()
        mock_dynamic_model.predict_single_bucket.return_value = 1 / stimulation.NUM_BUCKETS

        # === Patch Internal Functions ===
        # Fixed arrival price
        stimulation.compute_arrival_price = lambda stock, date: 100.0
        
        # Linearly increasing price path: [100, ..., 110), length = NUM_BUCKETS
        stimulation.simulate_price_path = lambda start_price, sigma: np.linspace(start_price, start_price + 10, stimulation.NUM_BUCKETS + 1)[:-1]
        
        # Simulated market volume equals planned trade (no stochasticity)
        stimulation.simulate_market_volume_log_normal = lambda my_trade: my_trade
        
        # Always return same kappa, cost, and a flat trading schedule
        stimulation.get_optimal_kappa_and_cost_from_trajectory = \
            lambda remaining_shares, total_shares, price_path: (0.1, 2.5, np.ones_like(price_path))

        # Constant execution impact and trading rate
        stimulation.compute_execution_impact_with_decay = \
            lambda kappa, total_volume, current_time, total_time, prev_trading_rate: (0.5, 25.0)

        # === Run Simulation ===
        result, df_logs = vwap_simulation(
            stock=stock,
            total_shares=total_shares,
            date=date,
            static_model=mock_static_model,
            dynamic_model=mock_dynamic_model
        )

        # === Assertions on Output ===

        # Check that the returned price path has the correct number of buckets
        self.assertEqual(result["price_path"].shape[0], stimulation.NUM_BUCKETS)

        # Check that the volume executed per bucket has the correct shape
        self.assertEqual(result["my_executed_volume"].shape[0], stimulation.NUM_BUCKETS)

        # Check that total volume matches target (within NUM_BUCKETS tolerance)
        self.assertAlmostEqual(result["my_executed_volume"].sum(), total_shares, delta=stimulation.NUM_BUCKETS)

        # Ensure guaranteed cost is computed and non-negative
        self.assertIn("guaranteed_cost", result)
        self.assertGreaterEqual(result["guaranteed_cost"], 0)

# === Run the Tests ===
if __name__ == "__main__":
    unittest.main()