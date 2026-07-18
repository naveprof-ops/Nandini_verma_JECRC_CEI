"""
Feature engineering shared by both training (features.py used on the full
historical frame) and recursive forecasting (predict.py calls the same
functions one day at a time on a growing history frame).

Feature families:
  1. Calendar / seasonality features (shared structure across all series).
  2. Lag + rolling-window features per store-item series (series-specific
     dynamics / autocorrelation).
  3. Store / item / store-item mean encodings, fit on train only (shared
     structure, prevents leakage from validation/test).

All lag & rolling features use `.shift(1)` before rolling so that the value
for a given day never uses that same day's own sales (no leakage).
"""
import numpy as np
import pandas as pd

from config import LAG_DAYS, ROLLING_WINDOWS, TARGET_COL, DATE_COL, GROUP_COLS


def build_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    d = df[DATE_COL]
    df["year"] = d.dt.year
    df["month"] = d.dt.month
    df["day"] = d.dt.day
    df["dayofweek"] = d.dt.dayofweek
    df["weekofyear"] = d.dt.isocalendar().week.astype(int)
    df["quarter"] = d.dt.quarter
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_month_start"] = d.dt.is_month_start.astype(int)
    df["is_month_end"] = d.dt.is_month_end.astype(int)

    # Cyclical encodings so the model sees Dec->Jan and Sun->Mon as "close"
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    return df


def build_lag_roll_features(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    df = df.sort_values(GROUP_COLS + [DATE_COL]).copy()
    grp = df.groupby(GROUP_COLS)[target_col]

    for lag in LAG_DAYS:
        df[f"lag_{lag}"] = grp.shift(lag)

    shifted = grp.shift(1)  # never include the current day itself
    for window in ROLLING_WINDOWS:
        df[f"rollmean_{window}"] = (
            shifted.groupby([df["store"], df["item"]]).rolling(window).mean().reset_index(drop=True)
        )
        df[f"rollstd_{window}"] = (
            shifted.groupby([df["store"], df["item"]]).rolling(window).std().reset_index(drop=True)
        )
    return df


def compute_group_encodings(train_df: pd.DataFrame, target_col: str = TARGET_COL) -> dict:
    """Fit mean-encodings on TRAIN ONLY to avoid leakage."""
    encodings = {
        "store_mean": train_df.groupby("store")[target_col].mean(),
        "item_mean": train_df.groupby("item")[target_col].mean(),
        "store_item_mean": train_df.groupby(["store", "item"])[target_col].mean(),
        "global_mean": train_df[target_col].mean(),
    }
    return encodings


def apply_group_encodings(df: pd.DataFrame, encodings: dict) -> pd.DataFrame:
    df = df.copy()
    df["store_mean_enc"] = df["store"].map(encodings["store_mean"]).fillna(encodings["global_mean"])
    df["item_mean_enc"] = df["item"].map(encodings["item_mean"]).fillna(encodings["global_mean"])
    si_map = encodings["store_item_mean"]
    df["store_item_mean_enc"] = (
        df.set_index(["store", "item"]).index.map(si_map).to_numpy()
    )
    df["store_item_mean_enc"] = pd.Series(df["store_item_mean_enc"]).fillna(encodings["global_mean"]).values
    return df


def get_feature_columns() -> list:
    calendar_feats = [
        "year", "month", "day", "dayofweek", "weekofyear", "quarter",
        "is_weekend", "is_month_start", "is_month_end",
        "month_sin", "month_cos", "dow_sin", "dow_cos",
    ]
    lag_feats = [f"lag_{lag}" for lag in LAG_DAYS]
    roll_feats = [f"rollmean_{w}" for w in ROLLING_WINDOWS] + [f"rollstd_{w}" for w in ROLLING_WINDOWS]
    enc_feats = ["store_mean_enc", "item_mean_enc", "store_item_mean_enc"]
    cat_feats = ["store", "item"]
    return cat_feats + calendar_feats + lag_feats + roll_feats + enc_feats


def build_full_feature_frame(df: pd.DataFrame, encodings: dict, target_col: str = TARGET_COL) -> pd.DataFrame:
    """Convenience wrapper: calendar + lag/roll + encodings, in one call."""
    df = build_calendar_features(df)
    df = build_lag_roll_features(df, target_col=target_col)
    df = apply_group_encodings(df, encodings)
    return df
