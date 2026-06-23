"""
test_preprocess.py
------------------
Unit tests for ShopSmart privacy-aware preprocessing pipeline.
"""

import pytest
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocess import clean_data, engineer_features, encode_features, privacy_audit


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "CustomerID": ["C001", "C002", "C003"],
        "Churn": [1, 0, 0],
        "Tenure": [1, 12, 36],
        "CityTier": [1, 2, 3],
        "WarehouseToHome": [10, 20, 15],
        "NumberOfDeviceRegistered": [2, 3, 1],
        "SatisfactionScore": [2, 4, 5],
        "NumberOfAddress": [1, 2, 3],
        "Complain": [1, 0, 0],
        "OrderAmountHikeFromlastYear": [10, 20, 15],
        "CouponUsed": [0, 3, 1],
        "OrderCount": [2, 8, 5],
        "DaySinceLastOrder": [7, 2, 1],
        "CashbackAmount": [80, 150, 250],
        "PreferredLoginDevice": ["Mobile Phone", "Computer", "Mobile Phone"],
        "PreferredPaymentMode": ["Debit Card", "UPI", "Credit Card"],
        "Gender": ["Male", "Female", "Male"],
        "PreferedOrderCat": ["Mobile", "Fashion", "Grocery"],
        "MaritalStatus": ["Single", "Married", "Single"],
    })


class TestPrivacyAudit:
    def test_drops_customer_id(self, sample_df):
        result = privacy_audit(sample_df, verbose=False)
        assert "CustomerID" not in result.columns

    def test_cashback_not_dropped_here(self, sample_df):
        # CashbackAmount is dropped in engineer_features, not privacy_audit
        result = privacy_audit(sample_df, verbose=False)
        assert "CashbackAmount" in result.columns


class TestCleanData:
    def test_no_nulls_after_clean(self, sample_df):
        df = privacy_audit(sample_df, verbose=False)
        df.loc[0, "SatisfactionScore"] = np.nan
        result = clean_data(df)
        assert result.isnull().sum().sum() == 0


class TestEngineerFeatures:
    def test_is_inactive_created(self, sample_df):
        df = privacy_audit(sample_df, verbose=False)
        df = clean_data(df)
        result = engineer_features(df)
        assert "is_inactive" in result.columns

    def test_engagement_score_created(self, sample_df):
        df = privacy_audit(sample_df, verbose=False)
        df = clean_data(df)
        result = engineer_features(df)
        assert "engagement_score" in result.columns

    def test_cashback_replaced_by_bucket(self, sample_df):
        df = privacy_audit(sample_df, verbose=False)
        df = clean_data(df)
        result = engineer_features(df)
        assert "CashbackAmount" not in result.columns
        cashback_cols = [c for c in result.columns if "cashback_bucket" in c]
        assert len(cashback_cols) > 0

    def test_is_inactive_correct_logic(self, sample_df):
        df = privacy_audit(sample_df, verbose=False)
        df = clean_data(df)
        result = engineer_features(df)
        # Row 0 has DaySinceLastOrder=7 → should be inactive
        assert result["is_inactive"].iloc[0] == 1
        # Row 2 has DaySinceLastOrder=1 → should not be inactive
        assert result["is_inactive"].iloc[2] == 0


class TestEncodeFeatures:
    def test_no_object_columns_remain(self, sample_df):
        df = privacy_audit(sample_df, verbose=False)
        df = clean_data(df)
        df = engineer_features(df)
        result = encode_features(df)
        obj_cols = result.select_dtypes(include="object").columns.tolist()
        assert len(obj_cols) == 0, f"Object cols remain: {obj_cols}"
