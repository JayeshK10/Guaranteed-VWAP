# === Standard Imports ===
import unittest  # Python's built-in unit testing module
import os
import sys

# Extend system path to import project files from parent directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# === Project-Specific Imports ===
import numpy as np  # For numerical arrays and operations
from Project.src.volume_model import BayesianDirichletStatic  # Included for completeness (not directly tested here)
from Project.src.volume_validation import compute_metrics, compute_sharpness  # Functions under test

# === Unit Test Class for Evaluation Metrics ===
class TestModelEvaluation(unittest.TestCase):
    """
    Unit tests for compute_metrics and compute_sharpness functions.
    These functions evaluate model predictions against actual volume curves.
    """

    def setUp(self):
        """
        Initialize fixed arrays for y_true (actuals) and y_pred (predictions)
        to simulate a known prediction scenario for controlled metric testing.
        """
        self.y_true = np.array([
            [0.1, 0.2, 0.3, 0.4],
            [0.15, 0.25, 0.3, 0.3],
            [0.05, 0.15, 0.35, 0.45]
        ])
        self.y_pred = np.array([
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4]
        ])

    def test_metric_keys_and_types(self):
        """
        Ensure compute_metrics returns all expected keys and float values.
        """
        metrics = compute_metrics(self.y_true, self.y_pred)
        expected_keys = {"MSE", "MAE", "LogLikelihood", "BayesianR2", "MCSE"}

        self.assertEqual(set(metrics.keys()), expected_keys)

        # All metric values must be floats
        for value in metrics.values():
            self.assertIsInstance(value, float)

    def test_metric_values_reasonable(self):
        """
        Check that the computed metric values fall within acceptable and logical ranges.
        """
        metrics = compute_metrics(self.y_true, self.y_pred)

        self.assertGreaterEqual(metrics["MSE"], 0)  # MSE can't be negative
        self.assertGreaterEqual(metrics["MAE"], 0)  # MAE can't be negative
        self.assertLessEqual(metrics["BayesianR2"], 1)  # R² must be ≤ 1
        self.assertGreaterEqual(metrics["LogLikelihood"], -10)  # Should not be extremely negative

    def test_perfect_prediction_case(self):
        """
        If predictions are perfect, error metrics should be zero and R² should be 1.
        """
        perfect_metrics = compute_metrics(self.y_true, self.y_true)

        self.assertAlmostEqual(perfect_metrics["MSE"], 0.0, places=6)
        self.assertAlmostEqual(perfect_metrics["MAE"], 0.0, places=6)
        self.assertAlmostEqual(perfect_metrics["BayesianR2"], 1.0, places=6)

    def test_sharpness_computation(self):
        """
        Test the compute_sharpness function to ensure it returns a valid interval width.
        """
        sharpness = compute_sharpness(self.y_true)

        self.assertIsInstance(sharpness, float)
        self.assertGreater(sharpness, 0)        # Interval width must be > 0
        self.assertLessEqual(sharpness, 1)      # Max possible spread in volume fractions

# === Run Tests ===
if __name__ == "__main__":
    unittest.main()