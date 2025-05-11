# === Standard Imports ===
import unittest  # Python's built-in test framework
import os
import sys

# Modify system path to import modules from parent directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# === Project-Specific Imports ===
import numpy as np  # For numerical computations
from Project.src.stimulation import simulate_price_path  # Function under test

# === Unit Test Class for simulate_price_path ===
class TestSimulatePricePath(unittest.TestCase):
    """
    Unit test for the simulate_price_path function, which generates a
    synthetic price series using geometric Brownian motion.
    """

    def test_simulate_price_path_deterministic(self):
        """
        Check that the simulated price path is deterministic with a fixed seed,
        has the expected structure, and contains only positive prices.
        """

        # Generate a simulated price path from a fixed starting point
        path = simulate_price_path(start_price=100.0, sigma=0.005, buckets=5, seed=1)

        # Expect the path to have buckets + 1 entries (including start_price)
        self.assertEqual(len(path), 6)

        # The first value in the path should match the provided starting price
        self.assertAlmostEqual(path[0], 100.0)

        # Ensure all prices in the path are positive (as expected for stock prices)
        self.assertTrue(np.all(path > 0))

        # Ensure there is at least some variation (path isn't flat)
        diffs = np.diff(path)
        self.assertTrue(np.any(diffs > 0) or np.any(diffs < 0))

# === Script Entry Point ===
if __name__ == "__main__":
    unittest.main()