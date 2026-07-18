"""
Generates the multi-horizon (3-month) forecast for the Kaggle test period.

Because lag/rolling features depend on prior days' sales, and those days are
unknown in the test period, this forecasts RECURSIVELY: one day at a time,
feeding each day's predictions back in as history for the next day's lag
features. This is standard practice for lag-feature-based ML forecasting
models (as opposed to classical models like ARIMA/Prophet fit per series).
"""
import joblib
import numpy as np
import pandas as pd

from config import MODEL_PATH, SUBMISSION_PATH, DATE_COL, TARGET_COL, LAG_DAYS, ROLLING_WINDOWS
from data_prep import load_train, load_test
from features import build_full_feature_frame

# Keep this much history (in days) per series when growing the recursive frame,
# to bound memory/compute while still covering the largest lag/rolling window used.
HISTORY_BUFFER_DAYS = max(max(LAG_DAYS), max(ROLLING_WINDOWS)) + 30


def recursive_forecast(history_df: pd.DataFrame, test_df: pd.DataFrame, model, encodings, feature_cols):
    extended = history_df[["store", "item", DATE_COL, TARGET_COL]].copy()
    extended = extended.sort_values(["store", "item", DATE_COL])

    test_dates = sorted(test_df[DATE_COL].unique())
    all_preds = []

    for i, current_date in enumerate(test_dates):
        today_rows = test_df.loc[test_df[DATE_COL] == current_date, ["id", "store", "item", DATE_COL]].copy()
        today_rows[TARGET_COL] = np.nan

        combined = pd.concat(
            [extended, today_rows[["store", "item", DATE_COL, TARGET_COL]]],
            ignore_index=True,
        )
        combined = combined.sort_values(["store", "item", DATE_COL])

        feat = build_full_feature_frame(combined, encodings)
        today_feat = feat[feat[DATE_COL] == current_date].copy()
        # Preserve row order matching today_rows["id"]
        today_feat = today_feat.merge(today_rows[["id", "store", "item"]], on=["store", "item"], how="inner")

        X_today = today_feat[feature_cols]
        preds = model.predict(X_today, num_iteration=getattr(model, "best_iteration", None))
        preds = np.clip(preds, 0, None)

        today_rows = today_rows.reset_index(drop=True)
        today_rows[TARGET_COL] = preds
        all_preds.append(today_rows[["id", TARGET_COL]])

        new_hist = today_rows[["store", "item", DATE_COL, TARGET_COL]]
        extended = pd.concat([extended, new_hist], ignore_index=True)

        # Trim old history we no longer need, to keep the recursive step fast
        cutoff = pd.Timestamp(current_date) - pd.Timedelta(days=HISTORY_BUFFER_DAYS)
        extended = extended[extended[DATE_COL] >= cutoff]

        if (i + 1) % 10 == 0 or (i + 1) == len(test_dates):
            print(f"  Forecasted {i + 1}/{len(test_dates)} days...")

    result = pd.concat(all_preds, ignore_index=True)
    return result


def main():
    print("Loading trained model...")
    bundle = joblib.load(MODEL_PATH)
    model, encodings, feature_cols = bundle["model"], bundle["encodings"], bundle["feature_cols"]

    print("Loading full training history and test period...")
    train_df = load_train()
    test_df = load_test()

    print(f"Forecasting {test_df[DATE_COL].nunique()} days across "
          f"{test_df[['store', 'item']].drop_duplicates().shape[0]} store-item series "
          f"(recursive, day-by-day)...")
    submission = recursive_forecast(train_df, test_df, model, encodings, feature_cols)

    submission = submission.sort_values("id").reset_index(drop=True)
    submission[TARGET_COL] = submission[TARGET_COL].round(2)
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"\nSaved submission to {SUBMISSION_PATH}")
    print(submission.head())


if __name__ == "__main__":
    main()
