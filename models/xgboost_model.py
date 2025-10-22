"""
FinScope AI - XGBoost Model

Gradient-boosted tree model with hyperparameter tuning for credit risk.
"""

from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
from loguru import logger
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb

from models.base_model import BaseModel


class XGBoostModel(BaseModel):
    """XGBoost model for credit risk prediction."""

    DEFAULT_PARAMS = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "auc",
        "early_stopping_rounds": 30,
    }

    def __init__(self, params: Optional[Dict] = None):
        super().__init__(name="xgboost")
        model_params = {**self.DEFAULT_PARAMS, **(params or {})}
        early_stopping = model_params.pop("early_stopping_rounds", 30)
        self._early_stopping_rounds = early_stopping
        self.model = xgb.XGBClassifier(**model_params)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        tune_hyperparams: bool = False,
        **kwargs,
    ) -> Dict[str, float]:
        """Train XGBoost model with optional hyperparameter tuning."""
        logger.info(f"Training {self.name} | train={X_train.shape} | val={X_val.shape}")

        # Calculate scale_pos_weight for imbalanced data
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        if n_pos > 0:
            self.model.set_params(scale_pos_weight=n_neg / n_pos)

        if tune_hyperparams:
            self._tune_hyperparams(X_train, y_train)

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        self._is_trained = True

        # Evaluate
        train_proba = self.model.predict_proba(X_train)[:, 1]
        val_proba = self.model.predict_proba(X_val)[:, 1]

        metrics = {
            "train_auc": roc_auc_score(y_train, train_proba),
            "val_auc": roc_auc_score(y_val, val_proba),
            "train_logloss": log_loss(y_train, train_proba),
            "val_logloss": log_loss(y_val, val_proba),
            "best_iteration": self.model.best_iteration if hasattr(self.model, "best_iteration") else -1,
        }

        logger.info(
            f"{self.name} trained | train_auc={metrics['train_auc']:.4f} | "
            f"val_auc={metrics['val_auc']:.4f} | best_iter={metrics['best_iteration']}"
        )
        return metrics

    def _tune_hyperparams(self, X: np.ndarray, y: np.ndarray) -> None:
        """Randomized search over hyperparameter space."""
        logger.info("Running hyperparameter tuning for XGBoost...")

        param_distributions = {
            "max_depth": [3, 4, 5, 6, 7, 8],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "subsample": [0.6, 0.7, 0.8, 0.9],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
            "min_child_weight": [1, 3, 5, 7],
            "gamma": [0, 0.1, 0.2, 0.5],
            "reg_alpha": [0, 0.1, 0.5, 1.0],
            "reg_lambda": [0.5, 1.0, 2.0, 5.0],
        }

        search = RandomizedSearchCV(
            self.model,
            param_distributions,
            n_iter=30,
            scoring="roc_auc",
            cv=3,
            random_state=42,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X, y)

        logger.info(f"Best params: {search.best_params_} | Best AUC: {search.best_score_:.4f}")
        self.model = search.best_estimator_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return default probability for each sample."""
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call train() or load() first.")
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature importance scores."""
        if not self._is_trained:
            raise RuntimeError("Model not trained.")
        importance = self.model.feature_importances_
        return dict(enumerate(importance))

    def save(self, path: str) -> None:
        """Save model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        filepath = path / f"{self.name}.joblib"
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")

    def load(self, path: str) -> "XGBoostModel":
        """Load model from disk."""
        path = Path(path)
        filepath = path / f"{self.name}.joblib"
        self.model = joblib.load(filepath)
        self._is_trained = True
        logger.info(f"Model loaded from {filepath}")
        return self
