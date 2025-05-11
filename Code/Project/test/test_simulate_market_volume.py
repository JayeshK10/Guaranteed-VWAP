# === Standard Imports ===
import unittest  # Python's built-in unit testing framework
import os
import sys

# Add parent directories to the system path so that we can import project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# === External & Project-Specific Imports ===
import numpy as np  # For numerical and random utilities
from Project.src.stimulation import simulate_market_volume_log_normal  # Function under test

# === Unit Test Class for simulate_market_volume_log_normal ===
class TestSimulateMarketVolume(unittest.TestCase):
    """
    Unit tests for the simulate_market_volume_log_normal function,
    which generates random market volume around a planned trade size,
    using a log-normal distribution.
    """

    def test_output_positive(self):
        """
        Ensure the function always returns strictly positive market volume,
        as log-normal distributions are defined only for positive real numbers.
        """
        for _ in range(100):  # Run multiple trials to test random variation
            volume = simulate_market_volume_log_normal(my_trade=100)
            self.assertGreater(volume, 0.0)

    def test_output_scales_with_sigma(self):
        """
        Check that increasing sigma leads to a change in output.
        Using the same seed ensures reproducibility for comparison.
        """
        np.random.seed(42)  # Set seed for deterministic behavior
        vol_low = simulate_market_volume_log_normal(my_trade=100, sigma=0.1)

        np.random.seed(42)  # Reset seed to replicate the same random draw
        vol_high = simulate_market_volume_log_normal(my_trade=100, sigma=0.5)

        # Expect volumes to diverge due to difference in sigma (volatility of noise)
        self.assertNotAlmostEqual(vol_low, vol_high, places=2)

    def test_output_repeatable_with_seed(self):
        """
        Confirm that using the same random seed leads to consistent results,
        ensuring reproducibility which is critical for unit testing and debugging.
        """
        np.random.seed(123)
        v1 = simulate_market_volume_log_normal(my_trade=50, sigma=0.3)

        np.random.seed(123)
        v2 = simulate_market_volume_log_normal(my_trade=50, sigma=0.3)

        # Output must be numerically identical (to 6 decimal places)
        self.assertAlmostEqual(v1, v2, places=6)

# === Run the Test Suite ===
if __name__ == "__main__":
    unittest.main()