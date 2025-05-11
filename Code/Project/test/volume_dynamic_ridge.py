# === Standard Imports ===
import unittest  # Python's built-in testing framework
import os
import sys

# Extend path to enable importing from the root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# === Project-Specific Imports ===
import numpy as np  # For numerical operations and assertions
from Project.src.dynamic_volume import DynamicVolumeModel  # Class under test
from sklearn.linear_model import BayesianRidge  # Confirm model architecture

# === Unit Test Class for DynamicVolumeModel ===
class TestDynamicVolumeModel(unittest.TestCase):
    """
    Unit tests for the DynamicVolumeModel, which uses features from
    market and static volume curves to predict volume bucket-by-bucket.
    """

    def setUp(self):
        """
        Set up test model, synthetic static & market curves, and training data.
        Trains the model using extracted features and target market curve.
        """
        self.model = DynamicVolumeModel()

        # Static volume profile (e.g., from prior historical mean)
        self.static_curve = np.array([0.1] * 13)

        # Simulated actual market curve with small variability
        self.market_curve = np.array([0.05, 0.07, 0.1, 0.12, 0.08, 0.07,
                                      0.09, 0.11, 0.08, 0.06, 0.05, 0.06, 0.06])
        self.y_train = self.market_curve  # Labels for training

        # Generate feature matrix using internal extractor for each bucket
        self.X_train = np.array([
            self.model.extract_features(self.market_curve, self.static_curve, t)
            for t in range(len(self.static_curve))
        ])

        # Fit model on synthetic data for downstream prediction tests
        self.model.fit(self.X_train, self.y_train)

    def test_extract_features_edge_case_t0(self):
        """
        Ensure that the extractor handles t=0 correctly (no history available).
        All recent-based stats should be 0.
        """
        features = self.model.extract_features(self.market_curve, self.static_curve, t=0)
        self.assertEqual(len(features), 4)
        self.assertAlmostEqual(features[0], self.static_curve[0])  # static curve at t=0
        self.assertEqual(features[1], 0.0)  # cum_vol = 0.0
        self.assertEqual(features[2], 0.0)  # mean_recent = 0.0
        self.assertEqual(features[3], 0.0)  # std_recent = 0.0

    def test_extract_features_t3_windowed(self):
        """
        Validate mean and std calculations for recent 3-bucket window at t=3.
        """
        t = 3
        features = self.model.extract_features(self.market_curve, self.static_curve, t)
        self.assertEqual(len(features), 4)
        expected_recent = self.market_curve[t-3:t]
        self.assertAlmostEqual(features[2], np.mean(expected_recent))
        self.assertAlmostEqual(features[3], np.std(expected_recent))

    def test_predict_single_bucket_output(self):
        """
        Check that a single bucket prediction returns a float (scalar).
        """
        t = 5
        pred = self.model.predict_single_bucket(self.market_curve, self.static_curve, t)
        self.assertIsInstance(pred, float)

    def test_batch_prediction_shape(self):
        """
        Confirm that batch prediction returns the correct output shape
        matching the number of training samples.
        """
        preds = self.model.predict_batch(self.X_train)
        self.assertEqual(preds.shape, self.y_train.shape)

    def test_fit_predict_consistency(self):
        """
        After training, the model's predictions should closely match training labels.
        Validates that model fit is working by checking low MSE.
        """
        preds = self.model.predict_batch(self.X_train)
        mse = np.mean((preds - self.y_train) ** 2)
        self.assertLess(mse, 1e-3, f"MSE too high: {mse:.6f}")  # Should fit nearly perfectly

# === Run Tests ===
if __name__ == '__main__':
    unittest.main()