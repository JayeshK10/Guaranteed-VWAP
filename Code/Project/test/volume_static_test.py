# === Standard Imports ===
import unittest  # Python's built-in test framework
import os
import sys

# Extend system path to allow importing project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# === Project-Specific Imports ===
import numpy as np  # For numerical operations
from Project.src.volume_model import BayesianDirichletStatic, NUM_BUCKETS  # Class under test

# === Unit Test Class for BayesianDirichletStatic Model ===
class TestBayesianDirichletStatic(unittest.TestCase):
    """
    Unit tests for the BayesianDirichletStatic class, which models
    per-stock and global intraday volume distributions using Dirichlet priors.
    """

    def test_initialization(self):
        """
        Verify that the constructor initializes the global prior correctly.
        """
        model = BayesianDirichletStatic(prior_strength=15, lambda_prior=7)

        self.assertEqual(model.prior_strength, 15)
        self.assertEqual(model.lambda_prior, 7)

        # Check that global alpha vector is initialized with uniform value
        self.assertTrue(np.allclose(model.alpha_prior_global, np.ones(NUM_BUCKETS) * 15))

    def test_fit_and_predict_with_mock_data(self):
        """
        Fit the model using mocked constant volume curves and test whether
        prediction yields a valid normalized distribution for each stock.
        """
        model = BayesianDirichletStatic(prior_strength=10, lambda_prior=5)

        # Two mock stocks, three dates
        stock_list = ['AAA', 'BBB']
        date_list = ['20240101', '20240102', '20240103']

        # Simulated curve: full volume in bucket 0 (extreme case)
        fake_curve = np.array([1 if i == 0 else 0 for i in range(NUM_BUCKETS)])

        # Override the data reader with a deterministic fake curve
        model.parse_taq_day = lambda date, stock: fake_curve.copy()

        # Fit model using mock data
        model.fit(stock_list, date_list)

        # Predict volume profile for each stock and validate shape and normalization
        for stock in stock_list:
            pred = model.predict(stock)

            self.assertEqual(len(pred), NUM_BUCKETS)  # One value per bucket
            self.assertAlmostEqual(pred.sum(), 1.0, places=5)  # Should be a probability distribution

    def test_predict_fallback_to_global(self):
        """
        If a stock has not been fit, prediction should fall back to the global posterior.
        """
        model = BayesianDirichletStatic()

        # Manually set global posterior alpha to a known increasing sequence
        model.alpha_posterior_global = np.arange(1, NUM_BUCKETS + 1)

        # No stock-specific alphas available
        model.stock_alphas = {}

        # Predict for a stock that hasn't been fit
        pred = model.predict("AAA")

        # Output must be normalized and match global posterior expectation
        self.assertEqual(len(pred), NUM_BUCKETS)
        self.assertAlmostEqual(pred.sum(), 1.0, places=5)

        # Validate correctness of normalization logic
        expected = model.alpha_posterior_global / model.alpha_posterior_global.sum()
        self.assertTrue(np.allclose(pred, expected))

# === Run the Test Suite ===
if __name__ == '__main__':
    unittest.main()