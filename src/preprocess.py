"""
preprocess.py
-------------
Privacy-aware data preprocessing for ShopSmart loyalty churn prediction.

Design principle: Only aggregated, non-personal features are used.
Individual transaction history, login timestamps, and browsing data
are explicitly excluded in line with GDPR data minimisation (Art. 5(1)(c)).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ── Privacy-safe feature whitelist ────────────────────────────────────────────
PRIVACY_SAFE_FEATURES = [
    "Tenure",
    "CityTier",
    "WarehouseToHome",
    "NumberOfDeviceRegistered",
    "SatisfactionScore",
    "NumberOfAddress",
    "Complain",
    "OrderAmountHikeFromlastYear",
    "CouponUsed",
    "OrderCount",
    "DaySinceLastOrder",
]

CATEGORICAL_FEATURES = [
    "PreferredLoginDevice",
    "PreferredPaymentMode",
    "Gender",
    "PreferedOrderCat",
    "MaritalStatus",
]

# Features explicitly excluded for privacy reasons
EXCLUDED_FEATURES = [
    "CustomerID",       # Direct identifier
    "CashbackAmount",   # Replaced by bucketed version
]


def load_data(filepath: str) -> pd.DataFrame:
    """Load the E-Commerce dataset (Excel format)."""
    df = pd.read_excel(filepath, sheet_name="E Comm")
    df.columns = df.columns.str.strip()
    print(f"Loaded: {df.shape[0]:,} members | {df.shape[1]} columns")
    return df


def privacy_audit(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Apply privacy audit: drop excluded columns and warn about any
    columns not in the approved whitelist.
    """
    df = df.copy()

    # Drop direct identifiers and columns being replaced
    df.drop(columns=EXCLUDED_FEATURES, errors="ignore", inplace=True)

    if verbose:
        approved = set(PRIVACY_SAFE_FEATURES + CATEGORICAL_FEATURES + ["Churn"])
        unexpected = set(df.columns) - approved
        if unexpected:
            print(f"⚠️  Columns not in privacy whitelist (review before use): {unexpected}")
        else:
            print("✅ Privacy audit passed — all columns are in approved whitelist.")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values."""
    df = df.copy()

    for col in df.select_dtypes(include=np.number).columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    for col in df.select_dtypes(include="object").columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    RFM-inspired feature engineering using only privacy-safe aggregated signals.

    R (Recency)   → DaySinceLastOrder, is_inactive flag
    F (Frequency) → OrderCount, CouponUsed
    M (Monetary)  → CashbackAmount bucketed into tiers
    """
    df = df.copy()

    # Recency flag: not ordered in 5+ days
    if "DaySinceLastOrder" in df.columns:
        df["is_inactive"] = (df["DaySinceLastOrder"] > 5).astype(int)

    # Engagement score: devices registered + coupons used
    if "NumberOfDeviceRegistered" in df.columns and "CouponUsed" in df.columns:
        df["engagement_score"] = df["NumberOfDeviceRegistered"] + df["CouponUsed"].fillna(0)

    # Tenure buckets
    if "Tenure" in df.columns:
        df["tenure_group"] = pd.cut(
            df["Tenure"],
            bins=[0, 3, 6, 12, 24, 100],
            labels=["0-3m", "3-6m", "6-12m", "12-24m", "24m+"],
        )

    # Cashback bucket (privacy-preserving monetary proxy)
    if "CashbackAmount" in df.columns:
        df["cashback_bucket"] = pd.cut(
            df["CashbackAmount"],
            bins=[0, 100, 200, 300, 9999],
            labels=["Low", "Medium", "High", "VeryHigh"],
            include_lowest=True,
        )
        df.drop(columns=["CashbackAmount"], inplace=True)

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns."""
    cat_cols = (
        df.select_dtypes(include="object").columns.tolist()
        + df.select_dtypes(include="category").columns.tolist()
    )
    cat_cols = [c for c in cat_cols if c != "Churn"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df


def scale_numeric(X_train, X_test):
    """Standard-scale numeric columns. Fit on train only."""
    num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    return X_train, X_test, scaler


def run_pipeline(filepath: str) -> tuple:
    """Full preprocessing pipeline with privacy audit built in."""
    df = load_data(filepath)
    df = privacy_audit(df)
    df = clean_data(df)
    df = engineer_features(df)
    df = encode_features(df)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_test, scaler = scale_numeric(X_train, X_test)

    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Churn rate (train): {y_train.mean():.2%}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    run_pipeline("../data/E Commerce.xlsx")
