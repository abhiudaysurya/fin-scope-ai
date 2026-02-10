"""
FinScope AI - Unit Tests for ML Models
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from data.generate_data import generate_synthetic_data
from data.feature_engineering import FeaturePipeline, split_data
from models.logistic_model import LogisticRegressionModel
from models.xgboost_model import XGBoostModel
from models.dnn_model import DNNModel


@pytest.fixture(scope="module")
def prepared_data():
    """Prepare data once for all model tests."""
    df = generate_synthetic_data(n_samples=800, random_seed=42)
    train, val, test = split_data(df)
    pipeline = FeaturePipeline()
    X_train, y_train = pipeline.fit_transform(train)
    X_val = pipeline.transform(val)
    y_val = val["default"].values.astype(np.float32)
    X_test = pipeline.transform(test)
    y_test = test["default"].values.astype(np.float32)
    return X_train, y_train, X_val, y_val, X_test, y_test, pipeline


class TestLogisticRegression:
    def test_train_and_predict(self, prepared_data):
        X_train, y_train, X_val, y_val, X_test, y_test, _ = prepared_data
        model = LogisticRegressionModel(C=1.0)
        metrics = model.train(X_train, y_train, X_val, y_val)
        assert "val_auc" in metrics
        assert 0 <= metrics["val_auc"] <= 1

        proba = model.predict_proba(X_test)
        assert proba.shape == (len(X_test),)
        assert np.all((proba >= 0) & (proba <= 1))

    def test_save_load(self, prepared_data, tmp_path):
        X_train, y_train, X_val, y_val, X_test, _, _ = prepared_data
        model = LogisticRegressionModel()
        model.train(X_train, y_train, X_val, y_val)
        model.save(str(tmp_path))

        loaded = LogisticRegressionModel()
        loaded.load(str(tmp_path))
        np.testing.assert_array_almost_equal(
            model.predict_proba(X_test),
            loaded.predict_proba(X_test),
        )


class TestXGBoost:
    def test_train_and_predict(self, prepared_data):
        X_train, y_train, X_val, y_val, X_test, y_test, _ = prepared_data
        model = XGBoostModel(params={"n_estimators": 50, "max_depth": 3})
        metrics = model.train(X_train, y_train, X_val, y_val)
        assert metrics["val_auc"] > 0.5  # Better than random

        proba = model.predict_proba(X_test)
        assert np.all((proba >= 0) & (proba <= 1))

    def test_feature_importance(self, prepared_data):
        X_train, y_train, X_val, y_val, _, _, _ = prepared_data
        model = XGBoostModel(params={"n_estimators": 50})
        model.train(X_train, y_train, X_val, y_val)
        importance = model.get_feature_importance()
        assert len(importance) > 0


class TestDNN:
    def test_train_and_predict(self, prepared_data):
        X_train, y_train, X_val, y_val, X_test, _, _ = prepared_data
        model = DNNModel(
            hidden_dims=(32, 16),
            epochs=5,
            patience=3,
            batch_size=128,
        )
        metrics = model.train(X_train, y_train, X_val, y_val)
        assert "val_auc" in metrics

        proba = model.predict_proba(X_test)
        assert proba.shape == (len(X_test),)
        assert np.all((proba >= 0) & (proba <= 1))

    def test_save_load(self, prepared_data, tmp_path):
        X_train, y_train, X_val, y_val, X_test, _, _ = prepared_data
        model = DNNModel(hidden_dims=(32, 16), epochs=5, patience=3)
        model.train(X_train, y_train, X_val, y_val)
        model.save(str(tmp_path))

        loaded = DNNModel()
        loaded.load(str(tmp_path))
        original = model.predict_proba(X_test)
        restored = loaded.predict_proba(X_test)
        np.testing.assert_array_almost_equal(original, restored, decimal=4)
