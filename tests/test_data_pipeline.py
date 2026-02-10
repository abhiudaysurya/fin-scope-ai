"""
FinScope AI - Unit Tests for Data Pipeline
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from data.generate_data import generate_synthetic_data
from data.feature_engineering import (
    FeaturePipeline,
    engineer_features,
    handle_missing_values,
    handle_outliers,
    split_data,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)


class TestDataGeneration:
    def test_generates_correct_shape(self):
        df = generate_synthetic_data(n_samples=100, random_seed=42)
        assert df.shape[0] == 100
        assert "default" in df.columns

    def test_target_is_binary(self):
        df = generate_synthetic_data(n_samples=500, random_seed=42)
        assert set(df["default"].dropna().unique()).issubset({0, 1})

    def test_has_missing_values(self):
        df = generate_synthetic_data(n_samples=1000, random_seed=42)
        assert df.isnull().sum().sum() > 0

    def test_reproducibility(self):
        df1 = generate_synthetic_data(n_samples=100, random_seed=42)
        df2 = generate_synthetic_data(n_samples=100, random_seed=42)
        pd.testing.assert_frame_equal(df1, df2)


class TestFeatureEngineering:
    @pytest.fixture
    def sample_df(self):
        return generate_synthetic_data(n_samples=500, random_seed=42)

    def test_engineer_features_adds_columns(self, sample_df):
        result = engineer_features(sample_df)
        assert "debt_to_income_ratio" in result.columns
        assert "credit_utilization_rate" in result.columns
        assert "transaction_variance_ratio" in result.columns
        assert "missed_payment_frequency" in result.columns

    def test_handle_missing_values(self, sample_df):
        result = handle_missing_values(sample_df)
        assert result.select_dtypes(include=[np.number]).isnull().sum().sum() == 0

    def test_handle_outliers(self, sample_df):
        result = handle_outliers(sample_df)
        assert result.shape == sample_df.shape

    def test_split_data_stratified(self, sample_df):
        train, val, test = split_data(sample_df)
        total = len(train) + len(val) + len(test)
        assert total == len(sample_df)
        # Check stratification (rates should be close)
        assert abs(train["default"].mean() - val["default"].mean()) < 0.05


class TestFeaturePipeline:
    @pytest.fixture
    def pipeline_and_data(self):
        df = generate_synthetic_data(n_samples=500, random_seed=42)
        train, val, _ = split_data(df)
        pipeline = FeaturePipeline()
        X_train, y_train = pipeline.fit_transform(train)
        return pipeline, val, X_train, y_train

    def test_fit_transform_shapes(self, pipeline_and_data):
        pipeline, _, X_train, y_train = pipeline_and_data
        assert X_train.ndim == 2
        assert y_train.ndim == 1
        assert X_train.shape[0] == y_train.shape[0]

    def test_transform_consistency(self, pipeline_and_data):
        pipeline, val, X_train, _ = pipeline_and_data
        X_val = pipeline.transform(val)
        assert X_val.shape[1] == X_train.shape[1]

    def test_save_and_load(self, pipeline_and_data, tmp_path):
        pipeline, val, _, _ = pipeline_and_data
        pipeline.save(str(tmp_path))
        loaded = FeaturePipeline().load(str(tmp_path))
        X1 = pipeline.transform(val)
        X2 = loaded.transform(val)
        np.testing.assert_array_almost_equal(X1, X2)
