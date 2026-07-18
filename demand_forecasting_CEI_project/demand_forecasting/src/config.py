"""
Central configuration: paths, constants, and model hyperparameters.
Keeping these in one place makes the pipeline easy to tune from a single file.
"""
import os

# --- Paths -------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

MODEL_PATH = os.path.join(OUTPUT_DIR, "lgbm_model.pkl")
SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")
VAL_METRICS_PATH = os.path.join(OUTPUT_DIR, "validation_metrics.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Time-series settings ----------------------------------------------
# Validation window: last N days of the training set, held out with a
# time-based split (never randomly shuffled).
VALIDATION_DAYS = 90

# Forecast horizon length (matches the Kaggle test period: 3 months / 90 days)
FORECAST_HORIZON_DAYS = 90

# Lag / rolling windows (in days) used as features.
LAG_DAYS = [7, 14, 28, 90, 365]
ROLLING_WINDOWS = [7, 14, 30, 90]

TARGET_COL = "sales"
DATE_COL = "date"
GROUP_COLS = ["store", "item"]

# --- Model hyperparameters ----------------------------------------------
LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "num_leaves": 63,
    "learning_rate": 0.03,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_data_in_leaf": 50,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbose": -1,
    "seed": 42,
}

NUM_BOOST_ROUND = 2000
EARLY_STOPPING_ROUNDS = 100
