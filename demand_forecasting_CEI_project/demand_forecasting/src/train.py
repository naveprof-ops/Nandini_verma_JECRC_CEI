"""
Trains a single global LightGBM model across all store-item series.

Validation is a TIME-BASED split (last VALIDATION_DAYS of the training range),
never a random split -- a random split would leak future information into
the training set for a time-series problem.
"""
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from config import (
    VALIDATION_DAYS, LGBM_PARAMS, NUM_BOOST_ROUND, EARLY_STOPPING_ROUNDS,
    MODEL_PATH, VAL_METRICS_PATH, DATE_COL, TARGET_COL,
)
from data_prep import load_train
from features import build_full_feature_frame, compute_group_encodings, get_feature_columns


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred))
    diff = np.abs(y_true - y_pred)
    ratio = np.where(denom == 0, 0.0, diff / denom)
    return 200.0 * np.mean(ratio)


def time_based_split(df: pd.DataFrame, validation_days: int = VALIDATION_DAYS):
    cutoff = df[DATE_COL].max() - pd.Timedelta(days=validation_days)
    train_part = df[df[DATE_COL] <= cutoff].copy()
    val_part = df[df[DATE_COL] > cutoff].copy()
    return train_part, val_part


def train_model(X_train, y_train, X_val, y_val, categorical_features):
    train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
    val_set = lgb.Dataset(X_val, label=y_val, categorical_feature=categorical_features, reference=train_set)

    model = lgb.train(
        LGBM_PARAMS,
        train_set,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True),
            lgb.log_evaluation(period=100),
        ],
    )
    return model


def main():
    print("Loading training data...")
    raw_train = load_train()

    print(f"Splitting: last {VALIDATION_DAYS} days held out for validation (time-based split)...")
    train_part, val_part = time_based_split(raw_train)
    print(f"  Train rows: {len(train_part)}  ({train_part[DATE_COL].min()} -> {train_part[DATE_COL].max()})")
    print(f"  Val rows:   {len(val_part)}  ({val_part[DATE_COL].min()} -> {val_part[DATE_COL].max()})")

    print("Fitting store/item mean-encodings on the training portion only...")
    encodings = compute_group_encodings(train_part)

    print("Building features (calendar + lag + rolling + encodings)...")
    # Build features on the FULL frame (train+val together) so validation rows have correct
    # lag/rolling history from the days immediately preceding them, then split back apart.
    full_feat = build_full_feature_frame(raw_train, encodings)
    feature_cols = get_feature_columns()

    train_feat = full_feat[full_feat[DATE_COL] <= train_part[DATE_COL].max()].copy()
    val_feat = full_feat[full_feat[DATE_COL] > train_part[DATE_COL].max()].copy()

    # Drop rows where lag features are NaN (only happens in the very first `max(lag)` days
    # of the whole series' history -- unavoidable, and a negligible fraction of 5 years of data)
    train_feat = train_feat.dropna(subset=feature_cols)

    X_train, y_train = train_feat[feature_cols], train_feat[TARGET_COL]
    X_val, y_val = val_feat[feature_cols], val_feat[TARGET_COL]

    print(f"Training on {len(X_train)} rows, validating on {len(X_val)} rows...")
    model = train_model(X_train, y_train, X_val, y_val, categorical_features=["store", "item"])

    val_preds = np.clip(model.predict(X_val, num_iteration=model.best_iteration), 0, None)
    val_smape = smape(y_val.values, val_preds)
    val_mae = float(np.mean(np.abs(y_val.values - val_preds)))
    print(f"\nValidation SMAPE: {val_smape:.3f}")
    print(f"Validation MAE:   {val_mae:.3f}")

    joblib.dump({"model": model, "encodings": encodings, "feature_cols": feature_cols}, MODEL_PATH)
    with open(VAL_METRICS_PATH, "w") as f:
        json.dump({"val_smape": val_smape, "val_mae": val_mae,
                    "best_iteration": model.best_iteration}, f, indent=2)

    print(f"\nSaved model + encodings to {MODEL_PATH}")
    print(f"Saved validation metrics to {VAL_METRICS_PATH}")

    print("\nTop 15 feature importances:")
    importances = pd.Series(model.feature_importance(importance_type="gain"), index=feature_cols)
    print(importances.sort_values(ascending=False).head(15))


if __name__ == "__main__":
    main()
