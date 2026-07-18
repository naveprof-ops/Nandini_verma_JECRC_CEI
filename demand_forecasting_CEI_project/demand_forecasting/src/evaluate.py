"""
Evaluates the trained model on the held-out validation window and produces
diagnostic plots (actual vs predicted) for a few example series.
"""
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import MODEL_PATH, OUTPUT_DIR, VALIDATION_DAYS, DATE_COL, TARGET_COL
from data_prep import load_train
from features import build_full_feature_frame
from train import time_based_split, smape


def main():
    bundle = joblib.load(MODEL_PATH)
    model, encodings, feature_cols = bundle["model"], bundle["encodings"], bundle["feature_cols"]

    raw_train = load_train()
    train_part, val_part = time_based_split(raw_train, VALIDATION_DAYS)

    full_feat = build_full_feature_frame(raw_train, encodings)
    val_feat = full_feat[full_feat[DATE_COL] > train_part[DATE_COL].max()].copy()

    X_val = val_feat[feature_cols]
    y_val = val_feat[TARGET_COL].values
    preds = np.clip(model.predict(X_val, num_iteration=getattr(model, "best_iteration", None)), 0, None)
    val_feat["prediction"] = preds

    overall_smape = smape(y_val, preds)
    print(f"Overall validation SMAPE: {overall_smape:.3f}")

    # Per-series SMAPE, to check whether the model struggles on specific stores/items
    per_series = (
        val_feat.groupby(["store", "item"])
        .apply(lambda g: smape(g[TARGET_COL].values, g["prediction"].values))
        .reset_index(name="smape")
        .sort_values("smape", ascending=False)
    )
    print("\nWorst 5 series by SMAPE:")
    print(per_series.head())
    print("\nBest 5 series by SMAPE:")
    print(per_series.tail())

    # Plot a few example series: actual vs predicted over the validation window
    examples = per_series.iloc[[0, len(per_series) // 2, -1]][["store", "item"]].values
    fig, axes = plt.subplots(len(examples), 1, figsize=(12, 4 * len(examples)))
    if len(examples) == 1:
        axes = [axes]
    for ax, (store, item) in zip(axes, examples):
        sub = val_feat[(val_feat["store"] == store) & (val_feat["item"] == item)].sort_values(DATE_COL)
        ax.plot(sub[DATE_COL], sub[TARGET_COL], label="actual", marker="o", markersize=3)
        ax.plot(sub[DATE_COL], sub["prediction"], label="predicted", marker="x", markersize=3)
        ax.set_title(f"Store {store}, Item {item}")
        ax.legend()
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "validation_actual_vs_predicted.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"\nSaved diagnostic plot to {out_path}")

    per_series.to_csv(os.path.join(OUTPUT_DIR, "per_series_smape.csv"), index=False)


if __name__ == "__main__":
    main()
