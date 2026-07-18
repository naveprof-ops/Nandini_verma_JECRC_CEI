"""
Data loading, basic cleaning, and a quick EDA summary.
Run standalone (`python src/data_prep.py`) to sanity-check the raw data and
produce a couple of diagnostic plots in outputs/.
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import TRAIN_PATH, TEST_PATH, OUTPUT_DIR, DATE_COL, TARGET_COL, GROUP_COLS


def load_train(path: str = TRAIN_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}. Either download train.csv/test.csv from Kaggle into "
            f"data/, or run `python src/generate_sample_data.py` first."
        )
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    df = df.sort_values(GROUP_COLS + [DATE_COL]).reset_index(drop=True)
    return df


def load_test(path: str = TEST_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}. Either download train.csv/test.csv from Kaggle into "
            f"data/, or run `python src/generate_sample_data.py` first."
        )
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    return df


def basic_checks(df: pd.DataFrame) -> None:
    print("Shape:", df.shape)
    print("\nDate range:", df[DATE_COL].min(), "->", df[DATE_COL].max())
    print("\nStores:", sorted(df["store"].unique()))
    print("Items:", sorted(df["item"].unique()))
    print("\nMissing values:\n", df.isna().sum())
    print("\nSales summary:\n", df[TARGET_COL].describe())


def plot_overview(df: pd.DataFrame) -> None:
    # Total daily sales across all series -- shows overall trend + yearly seasonality
    daily_total = df.groupby(DATE_COL)[TARGET_COL].sum()
    plt.figure(figsize=(12, 4))
    plt.plot(daily_total.index, daily_total.values)
    plt.title("Total daily sales across all store-item series")
    plt.xlabel("Date")
    plt.ylabel("Total sales")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eda_total_daily_sales.png"), dpi=120)
    plt.close()

    # One example series to show weekly + yearly seasonality clearly
    sample = df[(df["store"] == df["store"].iloc[0]) & (df["item"] == df["item"].iloc[0])]
    plt.figure(figsize=(12, 4))
    plt.plot(sample[DATE_COL], sample[TARGET_COL])
    plt.title(f"Example series: store={sample['store'].iloc[0]}, item={sample['item'].iloc[0]}")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eda_example_series.png"), dpi=120)
    plt.close()

    print(f"\nSaved EDA plots to {OUTPUT_DIR}/eda_total_daily_sales.png and eda_example_series.png")


if __name__ == "__main__":
    train_df = load_train()
    basic_checks(train_df)
    plot_overview(train_df)
