# Guaranteed VWAP Simulation and Pricing Framework

This project implements a complete pipeline for simulating and evaluating Guaranteed VWAP (Volume Weighted Average Price) execution. It combines static and dynamic volume modeling, market impact simulation, adaptive scheduling, and risk-adjusted cost evaluation using TAQ data.

---

## 🔍 Highlights

- Static volume model using Bayesian Dirichlet prior  
- Dynamic volume forecasting using Bayesian Ridge regression  
- Price path and market volume simulation with impact and noise  
- VWAP deviation + risk penalty cost objective  
- Optimal trade trajectory selection via sinh function  
- Unit-tested components and modular architecture  


---

## 📄 Report

Please refer to `Guaranteed_VWAP.pdf` for a complete explanation of the methodology, results, and interpretation of the findings.


---

## 📁 Folder Structure

<pre lang="md">
Question1/
│
├── Code/                             → All code for reading TAQ data, model development, and simulation
│   │
│   ├── taq/                          → TAQ data readers and utilities
│   │   ├── TAQQuotesReader.py             → Class to read and parse TAQ quote data
│   │   ├── TAQTradesReader.py             → Class to read and parse TAQ trade data
│   │   ├── MyDirectories.py               → Custom path configuration for TAQ datasets
│   │   ├── Test_TAQQuotesReader.py        → Unit tests for quote reader
│   │   ├── Test_TAQTradesReader.py        → Unit tests for trade reader
│   │   └── __init__.py                    → Package initializer
│   │
│   └── Project/
│       ├── test/                     → Unit and integration test scripts
│       │   ├── test_vwap_simulation.py          → Tests the full simulation pipeline
│       │   ├── test_optimal_kappa_cost.py       → Tests cost minimization under varying kappa
│       │   ├── test_simulate_price_path.py      → Tests price impact and noise simulation
│       │   ├── test_arrival_price.py            → Tests arrival price and terminal price calc
│       │   ├── test_execution_impact_with_decay.py → Tests decaying temporary impact model
│       │   ├── test_simulate_market_volume.py   → Tests log-normal volume simulation
│       │   ├── volume_static_test.py            → Tests static volume model predictions
│       │   ├── volume_dynamic_ridge.py          → Test harness for dynamic Ridge model
│       │   ├── volume_dynamic_data_process.py   → Feature engineering for dynamic model
│       │   ├── volume_static_eval_test.py       → Static model evaluation script
│       │   └── __init__.py                      → Package initializer
│       │
│       └── src/                      → Core simulation and modeling source code
│           ├── volume_model.py              → Unified interface for volume model wrappers
│           ├── dynamic_volume.py            → Dynamic volume predictor (Bayesian Ridge)
│           ├── volume_validation.py         → Volume prediction metrics and diagnostics
│           ├── stimulation.py               → Main simulation loop (VWAP trade execution)
│           └── __init__.py                  → Package initializer
│
├── Processed_Data/                  → Model outputs, simulation logs, and visualizations
│   ├── model_fit_metrics.txt                → Evaluation metrics for volume models (e.g., RMSE)
│   ├── top_1500_liquid_stocks.txt          → List of top-liquid stocks used for simulation
│   ├── bayesian_ridge_dynamic_volume.pkl   → Trained dynamic volume model (Bayesian Ridge)
│   ├── bayesian_dirichlet_static_volume.pkl→ Trained static volume model (Bayesian Dirichlet)
│   ├── .DS_Store                            → (System file — can be ignored)
│   │
│   ├── plots/                               → Visualization outputs (model fits, distributions, etc.)
│   │
│   └── stimulation_SPY_50000/               → Full simulation run for SPY (50,000 shares)
│       ├── simulation_result_logs.txt       → Summary log of simulation (costs, VWAPs, deviations)
│       ├── bucket_feature_matrix.csv        → Feature matrix used for dynamic prediction
│       ├── bucket_logs.txt                  → Logs of trades and volumes for each 30-min bucket
│       └── result_plots/                    → Detailed plots of trade schedule, impact, price paths
│
├── unit_test.txt                    → Describes each test script and its purpose in the GVWAP pipeline
├── packages_used.txt                → Lists Python dependencies used across test and core simulation scripts
</pre>

---

## 📄 Supporting Docs

- `unit_test.txt` — Explains each test module and its target functionality  

---

## 🛠 Prerequisites

Python 3.10+ environment with:

- numpy  
- pandas  
- scikit-learn  
- scipy  
- statsmodels  
- matplotlib  
- seaborn  
- tqdm  
- joblib  
- typing-extensions

### Setup

```bash
conda create -n gvwapsim python=3.10
conda activate gvwapsim
conda install numpy pandas scikit-learn scipy statsmodels matplotlib seaborn tqdm joblib