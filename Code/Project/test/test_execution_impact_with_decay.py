# === Standard Imports ===
import unittest  # Unit testing framework
import os
import sys

# Extend Python path to import modules from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# === Project-Specific Imports ===
import numpy as np
from Project.src.stimulation import compute_execution_impact_with_decay  # Function under test

# === Global Parameters for the Model ===
NUM_BUCKETS = 13  # Total number of trading buckets (e.g., 30-min intervals in a 6.5-hour day)
eta = 0.142       # Market impact coefficient
sigma = 0.05      # Volatility factor
beta = 0.5        # Exponent in impact decay term

# === Unit Test Class ===
class TestComputeExecutionImpactWithDecay(unittest.TestCase):
    """
    Unit tests for the compute_execution_impact_with_decay function.
    This function calculates impact cost and trading rate with a decaying memory of past trades.
    """

    def test_output_structure_and_types(self):
        """
        Verify that the function returns a float for both impact and trading rate,
        and that the impact is non-negative.
        """
        impact, rate = compute_execution_impact_with_decay(
            kappa=0.2,
            total_volume=1000,
            current_time=1,
            total_time=NUM_BUCKETS,
            prev_trading_rate=50
        )

        self.assertIsInstance(impact, float)
        self.assertIsInstance(rate, float)
        self.assertGreaterEqual(impact, 0.0)

    def test_zero_kappa_produces_uniform_rate(self):
        """
        When kappa is near-zero, the optimal strategy should distribute
        volume uniformly across all time buckets.
        """
        impact, rate = compute_execution_impact_with_decay(
            kappa=1e-6,  # Effectively no urgency
            total_volume=1000,
            current_time=5,
            total_time=NUM_BUCKETS,
            prev_trading_rate=20
        )
        expected_rate = 1000 / NUM_BUCKETS
        self.assertAlmostEqual(rate, expected_rate, places=1)

    def test_zero_volume_gives_zero_impact(self):
        """
        If there is no volume to trade, then both impact and trading rate should be zero.
        """
        impact, rate = compute_execution_impact_with_decay(
            kappa=0.1,
            total_volume=0,
            current_time=2,
            total_time=NUM_BUCKETS,
            prev_trading_rate=0
        )
        self.assertEqual(rate, 0)
        self.assertEqual(impact, 0)

    def test_decay_component_works(self):
        """
        Test that the residual impact from the previous trading rate is correctly computed.
        We assume total_volume=0 so that only the decay term contributes to impact.
        """
        prev_tr = 100
        impact, _ = compute_execution_impact_with_decay(
            kappa=0.1,
            total_volume=0,
            current_time=1,
            total_time=NUM_BUCKETS,
            prev_trading_rate=prev_tr
        )

        # Expected impact from previous trading rate's decay term
        expected_residual = eta * sigma * (prev_tr ** beta) * 0.5
        self.assertAlmostEqual(impact, expected_residual, places=4)

# === Run Tests ===
if __name__ == '__main__':
    unittest.main()