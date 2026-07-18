"""
Generates small synthetic train.csv / test.csv files with the exact same
schema as the real Kaggle "Store Item Demand Forecasting Challenge" data,
so the full pipeline can be run and sanity-checked before plugging in the
real dataset (which you download separately from Kaggle).

Real data:  10 stores x 50 items x 5 years daily (2013-01-01 -> 2017-12-31)
This synthetic version uses fewer stores/items and 2 years of history so it
runs in seconds, but keeps the same column names and structure.
"""
import numpy as np
import pandas as pd
import os
from config import TRAIN_PATH, TEST_PATH, DATA_DIR

N_STORES = 4
N_ITEMS = 8
TRAIN_START = "2013-01-01"
TRAIN_END = "2017-12-31"
TEST_START = "2018-01-01"
TEST_END = "2018-03-31"


def make_series(dates, store, item, rng):
    n = len(dates)
    t = np.arange(n)

    base = 10 + 2.0 * store + 0.5 * item
    trend = 0.0015 * t
    yearly_season = 6 * np.sin(2 * np.pi * t / 365.25 + item)
    weekly_season = 3 * np.sin(2 * np.pi * dates.dayofweek.values / 7 + store)
    noise = rng.normal(0, 2.0, size=n)

    sales = base + trend + yearly_season + weekly_season + noise
    sales = np.clip(np.round(sales), 0, None)
    return sales


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    rng = np.random.default_rng(42)

    train_dates = pd.date_range(TRAIN_START, TRAIN_END, freq="D")
    test_dates = pd.date_range(TEST_START, TEST_END, freq="D")

    train_rows = []
    for store in range(1, N_STORES + 1):
        for item in range(1, N_ITEMS + 1):
            sales = make_series(train_dates, store, item, rng)
            df = pd.DataFrame({
                "date": train_dates.strftime("%Y-%m-%d"),
                "store": store,
                "item": item,
                "sales": sales.astype(int),
            })
            train_rows.append(df)
    train_df = pd.concat(train_rows, ignore_index=True)
    train_df.to_csv(TRAIN_PATH, index=False)

    test_rows = []
    row_id = 0
    for store in range(1, N_STORES + 1):
        for item in range(1, N_ITEMS + 1):
            n = len(test_dates)
            df = pd.DataFrame({
                "id": range(row_id, row_id + n),
                "date": test_dates.strftime("%Y-%m-%d"),
                "store": store,
                "item": item,
            })
            row_id += n
            test_rows.append(df)
    test_df = pd.concat(test_rows, ignore_index=True)
    test_df.to_csv(TEST_PATH, index=False)

    print(f"Synthetic train.csv written: {train_df.shape} -> {TRAIN_PATH}")
    print(f"Synthetic test.csv written:  {test_df.shape} -> {TEST_PATH}")
    print(f"Stores: {N_STORES}, Items: {N_ITEMS}, "
          f"Train range: {TRAIN_START}..{TRAIN_END}, Test range: {TEST_START}..{TEST_END}")


if __name__ == "__main__":
    main()
