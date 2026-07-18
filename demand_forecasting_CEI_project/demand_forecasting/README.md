# Store-Item Demand Forecasting

A complete, production-style pipeline for the Kaggle **Store Item Demand Forecasting Challenge**
(https://www.kaggle.com/competitions/demand-forecasting-kernels-only).

**Problem**: forecast daily unit sales for 10 stores × 50 items (500 series) for a 3-month
horizon (Jan–Mar 2018), using 5 years of daily history (2013–2017). The pipeline captures
series-specific dynamics (lags, rolling stats per store-item) and shared structure
(calendar/seasonality features, global gradient-boosted model across all series) so it scales
to hundreds of series without training 500 separate models.

## Project layout

```
demand_forecasting/
├── data/                   # put train.csv / test.csv here (see "Get the data" below)
├── src/
│   ├── config.py            # paths, constants, feature/model settings
│   ├── generate_sample_data.py  # makes small synthetic data so you can test without Kaggle
│   ├── data_prep.py          # loading, cleaning, train/val split
│   ├── features.py           # calendar + lag + rolling + encoding features
│   ├── train.py               # trains the LightGBM model, saves it + validation metrics
│   ├── predict.py             # recursive multi-horizon forecast -> submission.csv
│   └── evaluate.py            # SMAPE scoring + diagnostic plots
├── outputs/                 # models, submission.csv, plots land here
├── requirements.txt
└── README.md
```

## 1. Setup (VS Code, local laptop)

```bash
# 1. Unzip the project and open the folder in VS Code
cd demand_forecasting

# 2. Create & activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

In VS Code: `Ctrl+Shift+P` → "Python: Select Interpreter" → choose `.venv`.

## 2. Get the data

Two options:

**Option A — real Kaggle data (recommended for your actual submission)**
1. Go to https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data
2. Download `train.csv` and `test.csv`
3. Place both files in `data/`

**Option B — quick test with synthetic data (no Kaggle account needed)**
```bash
python src/generate_sample_data.py
```
This creates a small synthetic `train.csv`/`test.csv` with the same schema (date, store, item,
sales / id, date, store, item) so you can run the whole pipeline end-to-end immediately and
confirm everything works before plugging in the real data.

## 3. Run the pipeline

```bash
# Step 1: Explore the data (prints summary stats + saves plots to outputs/)
python src/data_prep.py

# Step 2: Train the model (time-based validation on the last 3 months of train)
python src/train.py

# Step 3: Evaluate on the validation set (SMAPE + plots)
python src/evaluate.py

# Step 4: Generate the 3-month forecast for the real test period -> outputs/submission.csv
python src/predict.py
```

Each script can also just be run top-to-bottom inside VS Code's interactive window
(the "Run Cell"/"Run File in Interactive Window" button) if you prefer notebook-style execution.

## 4. What the model does

- **Calendar/seasonality features**: year, month, day, day-of-week, week-of-year, quarter,
  weekend flag, and cyclical (sin/cos) encodings of month & day-of-week to capture yearly and
  weekly seasonality smoothly.
- **Autocorrelation / series-specific dynamics**: lag features (7, 14, 28, 90, 365 days) and
  rolling mean/std (7, 14, 30, 90-day windows) computed **per store-item series**.
- **Shared structure across series**: store-level and item-level mean-encodings, plus a single
  global LightGBM model trained on all 500 series at once (with `store`/`item` as categorical
  features) so the model learns shared temporal patterns while still discriminating between
  individual series — this is what makes the approach scale to a large number of series instead
  of fitting one ARIMA/Prophet model per series.
- **Multi-horizon forecasting**: since lag features depend on prior days' sales, and the test
  period sales are unknown, `predict.py` forecasts **recursively**, day by day, feeding each
  day's predictions back in as lags for the next day.
- **Validation**: time-based split (train on all but the last 90 days, validate on the last 90
  days) — never a random split, since that would leak future information into the past for a
  time series problem.
- **Metric**: SMAPE (Symmetric Mean Absolute Percentage Error), the metric used by the actual
  Kaggle leaderboard for this competition.

## 5. Swapping in a stronger/alternate model

`src/train.py` isolates the model in one function (`train_model`), so you can swap LightGBM for
XGBoost, CatBoost, or an ensemble without touching the feature pipeline. The `config.py` file
has an `LGBM_PARAMS` dict you can tune directly (num_leaves, learning_rate, etc.).

## 6. Submitting to Kaggle

`outputs/submission.csv` is already formatted as required (`id,sales`). Upload it directly at
https://www.kaggle.com/competitions/demand-forecasting-kernels-only/submit.
