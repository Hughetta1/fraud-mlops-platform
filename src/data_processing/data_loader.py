# Kaggle Credit Card Fraud dataset loader.
# Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# 284,807 transactions, 492 frauds (0.17%), 30 features (Time + V1-V28 + Amount + Class).
# V1-V28 are PCA-transformed principal components; Time and Amount are raw.

from pathlib import Path
from typing import Optional

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DATA_PATH = _PROJECT_ROOT / "data" / "raw" / "creditcard.csv"


def load_real_data(filepath=None):
    path = Path(filepath) if filepath else _DEFAULT_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Real dataset not found at {path}. "
            f"Download from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
        )
    return pd.read_csv(path)


def is_real_data_available(filepath=None):
    path = Path(filepath) if filepath else _DEFAULT_DATA_PATH
    return path.exists()


def get_dataset_statistics(df):
    stats = {
        "total_transactions": len(df),
        "fraud_transactions": int(df["Class"].sum()),
        "normal_transactions": int((df["Class"] == 0).sum()),
        "fraud_rate": float(df["Class"].mean()),
        "avg_transaction_amount": float(df["Amount"].mean()),
        "median_transaction_amount": float(df["Amount"].median()),
        "max_transaction_amount": float(df["Amount"].max()),
        "min_transaction_amount": float(df["Amount"].min()),
        "dataset_shape": df.shape,
        "time_range_hours": float(df["Time"].max() - df["Time"].min()) / 3600,
        "num_v_features": len([c for c in df.columns if c.startswith("V") and c[1:].isdigit()]),
    }
    stats["avg_normal_amount"] = float(df[df["Class"] == 0]["Amount"].mean())
    stats["avg_fraud_amount"] = float(df[df["Class"] == 1]["Amount"].mean())
    return stats


def validate_dataset(df):
    missing = [c for c in ["Time", "Amount", "Class"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    v_cols = [c for c in df.columns if c.startswith("V") and c[1:].isdigit()]
    if len(v_cols) == 0:
        raise ValueError("No V-feature columns found")
    if df["Class"].nunique() != 2:
        raise ValueError(f"Target 'Class' must be binary")
    if (df["Amount"] < 0).any():
        raise ValueError("Negative transaction amounts found")
    fraud_rate = df["Class"].mean()
    if fraud_rate == 0 or fraud_rate == 1:
        raise ValueError(f"Fraud rate is {fraud_rate:.4f} — only one class present")
    nulls = df.isnull().sum().sum()
    if nulls > 0:
        print(f"Warning: {nulls} missing values detected")
    return True
