# Import standard libraries for file and system path handling
import os
import sys

# Add parent directories to the Python path to allow importing custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))  # Add root-level Code directory

# Import necessary libraries for numerical computation, modeling, and evaluation
import numpy as np
from Project.src.volume_model import BayesianDirichletStatic  # Static volume model for baseline predictions
from sklearn.linear_model import BayesianRidge  # Bayesian Ridge Regression for modeling
from taq.MyDirectories import MyDirectories  # Helper class to manage directory paths
import pickle  # For saving and loading model files
from sklearn.metrics import mean_squared_error  # To evaluate model performance using RMSE

'''
Test - volume_dynamic_ridge.py
Implements a dynamic volume prediction model using Bayesian Ridge Regression.
'''
class DynamicVolumeModel:
    def __init__(self):
        # Initialize a single Bayesian Ridge regression model
        self.model = BayesianRidge()

    def extract_features(self, market_curve, static_curve, t):
        # Calculate cumulative market volume up to time t
        market_cum = np.cumsum(market_curve)
        
        # Use previous cumulative volume, if t > 0
        cum_vol = market_cum[t - 1] if t > 0 else 0.0
        
        # Extract recent trading volumes over the past 3 intervals
        recent = market_curve[max(0, t - 3):t]
        
        # Compute mean of recent volumes (0 if no history)
        mean_recent = np.mean(recent) if len(recent) > 0 else 0.0
        
        # Compute standard deviation of recent volumes (0 if insufficient history)
        std_recent = np.std(recent) if len(recent) > 1 else 0.0

        # Return a feature vector of static prediction, cumulative volume, mean, and std
        return np.array([static_curve[t], cum_vol, mean_recent, std_recent])

    def fit(self, X_train, y_train):
        # Fit the regression model to the training data
        self.model.fit(X_train, y_train)

    def predict_single_bucket(self, market_curve, static_curve, t):
        # Generate features for a single bucket and predict volume
        x = self.extract_features(market_curve, static_curve, t)
        return self.model.predict(x.reshape(1, -1))[0]
    
    def predict_batch(self, X):
        # Predict volumes for a batch of feature inputs
        return self.model.predict(X)

'''
Test - volume_dynamic_data_process.py
Generates training data by extracting volume features across time buckets.
'''
def generate_training_data(stock_list, static_model, train_dates):
    X_real, y_real = [], []  # Lists to store features and targets

    for stock in stock_list:
        # Get the static volume prediction curve (length 13 buckets)
        static_curve = static_model.predict(stock)

        for date in train_dates:
            # Extract actual market curve from TAQ data
            market_curve = static_model.parse_taq_day(date, stock)
            if market_curve is None:
                continue  # Skip if data is missing or corrupt

            # Precompute cumulative volume for efficiency
            market_cum = np.cumsum(market_curve)

            for t in range(len(static_curve)):
                # Compose feature vector for bucket t
                feat = [
                    static_curve[t],
                    market_cum[t - 1] if t > 0 else 0.0,
                    np.mean(market_curve[max(0, t - 3):t]) if t > 0 else 0.0,
                    np.std(market_curve[max(0, t - 3):t]) if t > 1 else 0.0,
                ]
                X_real.append(feat)  # Add feature vector to list
                y_real.append(market_curve[t])  # Add actual volume as target

    return np.array(X_real), np.array(y_real)  # Return features and targets as arrays


if __name__ == '__main__':

    # Load relevant directory paths using MyDirectories utility
    quotes_directory = MyDirectories.getQuotesDir()
    data_dir = MyDirectories.getDataDir()

    # Get the list of all available trading dates and sort them
    list_of_dates = sorted(list(set(os.listdir(quotes_directory))))
    
    # Split data into 80% training and 20% testing
    split_index = int(0.8 * len(list_of_dates))
    train_dates = list_of_dates[:split_index]   # Dates used for training
    test_dates = list_of_dates[split_index:]    # Unseen dates used for evaluation

    # Load list of top liquid stocks to include in training
    stock_selection_path = os.path.join(data_dir, 'top_1500_liquid_stocks.txt')
    with open(stock_selection_path, 'r') as f:
        list_of_stocks = [line.strip() for line in f]
    stock_list = list_of_stocks

    # Load the pre-trained static volume model from disk
    model_path = os.path.join(MyDirectories.getDataDir(), 'bayesian_dirichlet_static_volume.pkl')
    with open(model_path, 'rb') as f:
        static_model = pickle.load(f)

    # Generate training and test datasets using stock list and date splits
    X_train, y_train = generate_training_data(stock_list, static_model, train_dates)
    X_test, y_test = generate_training_data(stock_list, static_model, test_dates)

    # Initialize and train the dynamic volume prediction model
    dynamic_model = DynamicVolumeModel()
    dynamic_model.fit(X_train, y_train)
    
    # Predict volume curves for both training and testing data
    y_pred_train = dynamic_model.predict_batch(X_train)
    y_pred_test = dynamic_model.predict_batch(X_test)

    # Evaluate model performance using RMSE metric
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    # Output model evaluation metrics
    print(f"Train RMSE: {train_rmse:.6f}")
    print(f"Test RMSE:  {test_rmse:.6f}")

    # Save the trained dynamic model to disk for future use
    dynamic_model_path = os.path.join(data_dir, 'bayesian_ridge_dynamic_volume.pkl')
    with open(dynamic_model_path, 'wb') as f:
        pickle.dump(dynamic_model, f)
    print("Dynamic model trained and saved to:", dynamic_model_path)