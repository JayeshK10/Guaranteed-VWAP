# === Standard Imports ===
import unittest  # Python's built-in testing framework
import os
import sys

# Extend the Python path so we can import from parent directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# === External & Project Imports ===
import numpy as np  # For numerical operations and validation
from unittest.mock import MagicMock, patch  # To mock dependencies during tests
from Project.src.stimulation import compute_arrival_price  # Function under test


# === Unit Test Class for compute_arrival_price ===
class TestArrivalPrice(unittest.TestCase):

    @patch("Project.src.stimulation.TAQQuotesReader")  # Mock the binary quote reader class
    @patch("Project.src.stimulation.MyDirectories")    # Mock the directory manager
    def test_compute_arrival_price_valid(self, mock_dirs, mock_reader_cls):
        """
        Test case where valid quotes are available in the first 5 minutes.
        We expect the function to return the average price of those quotes.
        """

        # Set mock return path for quotes
        mock_dirs.getQuotesDir.return_value = "/mock/path"

        # Create a mock instance of TAQQuotesReader
        mock_reader = MagicMock()
        mock_reader.getN.return_value = 10  # Pretend there are 10 quotes

        # Simulate 10 timestamps, each 1 second apart starting at market open
        start_ts = 9 * 60 * 60 * 1000 + 30 * 60 * 1000  # 9:30 AM
        mock_reader.getTimestamp.side_effect = [start_ts + i * 1000 for i in range(10)]

        # Simulate prices increasing from 100 to 109
        mock_reader.getPrice.side_effect = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]

        # Patch TAQQuotesReader to return our mocked reader
        mock_reader_cls.return_value = mock_reader

        # Call the function under test
        price = compute_arrival_price("AAPL", "2024-01-01")

        # We expect the mean of the first 5 prices (100 to 104)
        expected = np.mean([100, 101, 102, 103, 104])

        # Check if the returned price is close to expected
        self.assertAlmostEqual(price, expected, places=5)

    @patch("Project.src.stimulation.TAQQuotesReader")  # Mock quote reader
    @patch("Project.src.stimulation.MyDirectories")    # Mock directory logic
    def test_compute_arrival_price_no_valid_quotes(self, mock_dirs, mock_reader_cls):
        """
        Test case where all quote timestamps are before the valid trading window.
        We expect the function to return 0.0 to indicate no valid data.
        """

        # Return a mock quotes directory
        mock_dirs.getQuotesDir.return_value = "/mock/path"

        # Create mocked TAQ reader with 5 entries
        mock_reader = MagicMock()
        mock_reader.getN.return_value = 5

        # Simulate all timestamps before market open (i.e., invalid)
        start_ts = 9 * 60 * 60 * 1000 + 30 * 60 * 1000
        mock_reader.getTimestamp.side_effect = [start_ts - 10000] * 5  # Invalid quotes

        # Prices do not matter as timestamps are invalid
        mock_reader.getPrice.side_effect = [100] * 5

        # Return mock reader from patch
        mock_reader_cls.return_value = mock_reader

        # Expect 0.0 because no timestamps fall in the valid arrival window
        price = compute_arrival_price("MSFT", "2024-01-01")
        self.assertEqual(price, 0.0)


# === Entry Point for Running Tests ===
if __name__ == "__main__":
    unittest.main()  # Launch the test suite when script is run directly