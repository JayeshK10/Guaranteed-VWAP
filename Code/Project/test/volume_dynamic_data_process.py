# === Standard Imports ===
import unittest  # Python's built-in unit test framework
import os
import sys

# Extend system path to allow importing from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# === External & Project Imports ===
import numpy as np  # For array manipulation and assertions
from unittest.mock import MagicMock  # To mock the static model
from Project.src.dynamic_volume import generate_training_data  # Function under test

# === Unit Test Class for generate_training_data ===
class TestGenerateTrainingData(unittest.TestCase):
    """
    Unit tests for the generate_training_data function, which extracts training samples
    from stock-date combinations using static and market volume curves.
    """

    def setUp(self):
        """
        Initialize test setup: stock list, training dates, and a mock static model
        with predictable behavior for both static and market curves.
        """
        self.stock_list = ["AAPL", "GOOG"]
        self.train_dates = ["20240101", "20240102"]
        self.num_buckets = 13  # Fixed number of 30-min intervals in a trading day

        # === Mock Static Model ===
        self.mock_static_model = MagicMock()

        # Static curve is uniform (each bucket = 1/13)
        self.mock_static_model.predict.return_value = np.array(
            [1.0 / self.num_buckets] * self.num_buckets
        )

        # Market curve returns a slightly noisy normalized version of the base curve
        def mock_parse_taq_day(date, stock):
            base = np.array([1.0 / self.num_buckets] * self.num_buckets)
            noise = np.random.normal(0, 0.002, size=self.num_buckets)
            noisy = np.clip(base + noise, 0.0, 1.0)
            return noisy / noisy.sum()  # Normalize so it sums to 1

        self.mock_static_model.parse_taq_day.side_effect = mock_parse_taq_day

    def test_output_shapes(self):
        """
        Validate that the function returns the correct shapes for features (X) and labels (y).
        Each stock-date combination should yield 13 samples.
        """
        X, y = generate_training_data(self.stock_list, self.mock_static_model, self.train_dates)

        expected_samples = len(self.stock_list) * len(self.train_dates) * self.num_buckets
        self.assertEqual(X.shape, (expected_samples, 4))  # 4 features per sample
        self.assertEqual(y.shape, (expected_samples,))   # One target per sample

    def test_feature_ranges(self):
        """
        Ensure that all features and targets are non-negative,
        and that target curves sum to ~1 across each day.
        """
        X, y = generate_training_data(self.stock_list, self.mock_static_model, self.train_dates)

        # All features and targets should be >= 0
        self.assertTrue(np.all(X >= 0.0))
        self.assertTrue(np.all(y >= 0.0))

        # Verify that each market curve sums to 1 across buckets (within tolerance)
        self.assertTrue(np.allclose(np.sum(y.reshape(-1, self.num_buckets), axis=1), 1.0, atol=1e-4))

    def test_handles_missing_market_curve(self):
        """
        Confirm that the function gracefully skips stock-date pairs that return None.
        Here we simulate a missing curve for GOOG on 20240102.
        """

        # Override parse_taq_day to return None for one specific stock-date pair
        def patched_parse_taq_day(date, stock):
            if stock == "GOOG" and date == "20240102":
                return None
            else:
                return np.array([1.0 / self.num_buckets] * self.num_buckets)

        self.mock_static_model.parse_taq_day.side_effect = patched_parse_taq_day

        X, y = generate_training_data(self.stock_list, self.mock_static_model, self.train_dates)

        # Expected: 2 dates * 13 buckets for AAPL + 1 date * 13 buckets for GOOG = 39 samples
        self.assertEqual(X.shape[0], 39)
        self.assertEqual(y.shape[0], 39)

# === Run Tests ===
if __name__ == "__main__":
    unittest.main()