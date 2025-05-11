# === Environment Setup ===

import os
import sys

# Add parent directories to the Python module path to allow importing custom code
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# === Standard and Custom Imports ===

import pickle  # For loading the saved model
import numpy as np  # For numerical computations
from sklearn.metrics import mean_squared_error, mean_absolute_error  # Evaluation metrics
import matplotlib.pyplot as plt  # For generating plots
from taq.MyDirectories import MyDirectories  # Directory management
from Project.src.volume_model import BayesianDirichletStatic  # Static volume model class

# === Constants for Time Buckets ===

TRADING_START_MS = 9 * 60 * 60 * 1000 + 30 * 60 * 1000  # 9:30 AM in milliseconds
TRADING_END_MS = 16 * 60 * 60 * 1000  # 4:00 PM in milliseconds
BUCKET_SIZE_MS = 30 * 60 * 1000  # 30-minute intervals
NUM_BUCKETS = (TRADING_END_MS - TRADING_START_MS) // BUCKET_SIZE_MS  # Total number of buckets in a trading day


'''
Test - volume_static_eval_test.py
Evaluate the fitted static volume model using prediction metrics and visual diagnostics.
'''

# === Metric Computation Functions ===

def compute_metrics(y_true, y_pred):
    # Mean Squared Error
    mse = np.mean((y_true - y_pred) ** 2)

    # Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))

    # Log-Likelihood under a multinomial assumption
    log_likelihood = np.sum(y_true * np.log(y_pred + 1e-8))  # Add epsilon for numerical stability

    # Bayesian R²: fraction of variance explained
    bayesian_r2 = 1 - np.var(y_true - y_pred) / np.var(y_true)

    # Monte Carlo Standard Error of the residuals
    mcse = np.std(y_pred - y_true) / np.sqrt(len(y_true))

    return {
        "MSE": mse,
        "MAE": mae,
        "LogLikelihood": log_likelihood,
        "BayesianR2": bayesian_r2,
        "MCSE": mcse
    }

def compute_sharpness(y_true):
    # Sharpness measures the spread of actual volume profiles (95% interval width)
    lower = np.percentile(y_true, 2.5, axis=0)
    upper = np.percentile(y_true, 97.5, axis=0)
    return np.mean(upper - lower)


# === Main Evaluation Script ===

if __name__ == '__main__':
    # Load the saved static volume model from disk
    model_path = os.path.join(MyDirectories.getDataDir(), 'bayesian_dirichlet_static_volume.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Load test dates (last 20% of available trading days)
    quotes_directory = MyDirectories.getQuotesDir()
    list_of_dates = sorted(list(set(os.listdir(quotes_directory))))
    split_index = int(0.8 * len(list_of_dates))
    test_dates = list_of_dates[split_index:]

    # Load the list of top 1500 liquid stocks
    stock_selection_path = os.path.join(MyDirectories.getDataDir(), 'top_1500_liquid_stocks.txt')
    with open(stock_selection_path, 'r') as f:
        stock_list = [line.strip() for line in f]

    # Initialize containers for evaluation
    y_true, y_pred = [], []  # Will store actual and predicted volume curves
    errors_per_bucket = [[] for _ in range(NUM_BUCKETS)]  # To analyze error per time bucket

    # Generate predictions for all stock-date pairs
    for stock in stock_list:
        for date in test_dates:
            actual_curve = model.parse_taq_day(date, stock)
            if actual_curve is not None:
                pred_curve = model.predict(stock)
                y_true.append(actual_curve)
                y_pred.append(pred_curve)

                # Accumulate squared errors bucket-wise
                for i in range(NUM_BUCKETS):
                    errors_per_bucket[i].append((actual_curve[i] - pred_curve[i])**2)

    # Convert prediction lists into NumPy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    residuals = y_true - y_pred  # Compute residuals

    # Compute evaluation metrics
    metrics = compute_metrics(y_true, y_pred)
    sharpness = compute_sharpness(y_true)

    # Save metrics to a text file
    data_dir = MyDirectories.getDataDir()
    metrics_path = os.path.join(data_dir, "model_fit_metrics.txt")
    os.makedirs(data_dir, exist_ok=True)
    with open(metrics_path, "w") as f:
        f.write("=== Model Fit / Predictive Accuracy ===\n")
        for name, value in metrics.items():
            f.write(f"{name}: {value:.6f}\n")
        f.write(f"Sharpness (Mean Interval Width): {sharpness:.6f}\n")

    # Prepare directory for saving plots
    plot_dir = os.path.join(data_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # === 1. Observed vs Predicted Mean Volume Profile ===
    plt.figure()
    plt.plot(np.mean(y_pred, axis=0), label='Predicted', marker='o')
    plt.plot(np.mean(y_true, axis=0), label='Actual', marker='x')
    plt.title("Observed vs Predicted")
    plt.xlabel("Bucket")
    plt.ylabel("Fraction of Volume")
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "observed_vs_predicted.png"))
    plt.close()

    # === 2. Posterior Predictive Plot ===
    plt.figure()
    for i in range(min(20, len(y_true))):  # Plot first 20 actual curves
        plt.plot(y_true[i], color='gray', alpha=0.3)
    plt.plot(np.mean(y_pred, axis=0), color='red', label='Predicted Mean', linewidth=2)
    plt.title("Posterior Predictive Plot")
    plt.xlabel("Bucket")
    plt.ylabel("Fraction")
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "posterior_predictive_plot.png"))
    plt.close()

    # === 3. Calibration Plot (Q-Q Plot) ===
    plt.figure()
    empirical_cdf = np.sort(np.mean(y_true, axis=0))  # Sort mean actuals
    model_cdf = np.sort(np.mean(y_pred, axis=0))  # Sort mean predictions
    plt.plot(empirical_cdf, model_cdf, marker='o')
    plt.plot([0, 1], [0, 1], linestyle='--', color='black')  # Perfect calibration line
    plt.xlabel("Empirical Quantiles")
    plt.ylabel("Model Quantiles")
    plt.title("Calibration Plot (Q-Q)")
    plt.savefig(os.path.join(plot_dir, "calibration_plot.png"))
    plt.close()

    # === 4. Residual Plot ===
    plt.figure()
    plt.plot(np.mean(residuals, axis=0), marker='o')  # Mean residual per bucket
    plt.axhline(0, color='black', linestyle='--')
    plt.title("Residual Plot (Mean Across Stocks)")
    plt.xlabel("Bucket")
    plt.ylabel("Residual")
    plt.savefig(os.path.join(plot_dir, "residual_plot.png"))
    plt.close()

    # === 5. Interval Plot (95% Actual Range + Model Mean) ===
    lower = np.percentile(y_true, 2.5, axis=0)  # Lower bound of actual volume
    upper = np.percentile(y_true, 97.5, axis=0)  # Upper bound
    mean_pred = np.mean(y_pred, axis=0)  # Predicted mean

    plt.figure()
    plt.plot(mean_pred, label='Predicted Mean', marker='o')
    plt.fill_between(range(NUM_BUCKETS), lower, upper, color='gray', alpha=0.3,
                     label='95% Interval (Actual)')
    plt.title("Interval Plot")
    plt.xlabel("Bucket")
    plt.ylabel("Fraction")
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "interval_plot.png"))
    plt.close()