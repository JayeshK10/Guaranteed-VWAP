# === Standard Imports ===
import unittest  # Python unit testing framework
import os
import sys

# Add parent directories to the system path to enable module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# === Project Imports ===
import numpy as np  # For numerical computations and test arrays
from Project.src.stimulation import get_optimal_kappa_and_cost_from_trajectory  # Function under test

# === Global Impact Parameters (if used internally in tested function) ===
eta = 0.142    # Market impact coefficient
sigma = 0.005  # Volatility
beta = 0.5     # Power exponent in the impact model

# === Unit Test Class for get_optimal_kappa_and_cost_from_trajectory ===
class TestOptimalKappaAndCost(unittest.TestCase):
    """
    Unit tests for optimal kappa selection and cost minimization logic
    in the trajectory-based guaranteed VWAP setting.
    """

    def test_returns_correct_shapes_and_types(self):
        """
        Ensure the function returns correct data types and dimensions,
        and that the trading schedule sums to the correct share count.
        """
        price_path = np.linspace(100, 105, 10)  # Simulated price path
        remaining_shares = 500
        total_shares = 1000

        kappa, variation, schedule = get_optimal_kappa_and_cost_from_trajectory(
            remaining_shares,
            total_shares,
            price_path
        )

        self.assertIsInstance(kappa, float)  # Kappa should be float
        self.assertIsInstance(variation, float)  # Cost deviation should be float
        self.assertEqual(schedule.shape, price_path.shape)  # Same length as price path
        self.assertAlmostEqual(np.sum(schedule), remaining_shares, places=4)  # All shares must be scheduled

    def test_kappa_range_and_behavior(self):
        """
        Test that the selected kappa lies within the expected search range,
        and that the cost is strictly positive (indicating some non-trivial variation).
        """
        price_path = np.linspace(95, 105, 13)  # Slightly volatile path

        kappa, variation, schedule = get_optimal_kappa_and_cost_from_trajectory(
            remaining_shares=100,
            total_shares=200,
            price_path=price_path
        )

        # Verify kappa is in the expected sweep range (assumed [-40, 40])
        self.assertTrue(-40 <= kappa <= 40)
        self.assertGreater(variation, 0.0)  # Some cost should be incurred
        self.assertAlmostEqual(schedule.sum(), 100, places=4)  # Volume constraint satisfied

    def test_identical_price_path_returns_valid_schedule(self):
        """
        Ensure that even in the trivial case of a flat price path,
        the function returns a feasible and non-negative schedule.
        """
        price_path = np.ones(20) * 150  # Constant price, no incentive to front/back-load
        remaining_shares = 1000
        total_shares = 2000

        kappa, variation, schedule = get_optimal_kappa_and_cost_from_trajectory(
            remaining_shares, total_shares, price_path
        )

        self.assertAlmostEqual(schedule.sum(), 1000, places=4)  # Shares must match
        self.assertTrue(np.all(schedule >= 0))  # No negative allocations

# === Run Unit Tests ===
if __name__ == "__main__":
    unittest.main()