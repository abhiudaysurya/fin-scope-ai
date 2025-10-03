"""
FinScope AI - Logistic Regression Model (Baseline)

Regularized logistic regression serving as interpretable baseline.
"""

from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss

from models.base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    """Logistic Regression baseline model for credit risk prediction."""

    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        super().__init__(name="logistic_regression")
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver="lbfgs",
            class_weight="balanced",
            random_state=42,
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        **kwargs,
    ) -> Dict[str, float]:
        """Train logistic regression model."""
        logger.info(f"Training {self.name} | train={X_train.shape} | val={X_val.shape}")

        self.model.fit(X_train, y_train)
        self._is_trained = True

        # Evaluate
        train_proba = self.model.predict_proba(X_train)[:, 1]
        val_proba = self.model.predict_proba(X_val)[:, 1]

        metrics = {
            "train_auc": roc_auc_score(y_train, train_proba),
            "val_auc": roc_auc_score(y_val, val_proba),
            "train_logloss": log_loss(y_train, train_proba),
            "val_logloss": log_loss(y_val, val_proba),
        }

        logger.info(
            f"{self.name} trained | train_auc={metrics['train_auc']:.4f} | "
            f"val_auc={metrics['val_auc']:.4f}"
        )
        return metrics

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return default probability for each sample."""
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call train() or load() first.")
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str) -> None:
        """Save model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        filepath = path / f"{self.name}.joblib"
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")

    def load(self, path: str) -> "LogisticRegressionModel":
        """Load model from disk."""
        path = Path(path)
        filepath = path / f"{self.name}.joblib"
        self.model = joblib.load(filepath)
        self._is_trained = True
        logger.info(f"Model loaded from {filepath}")
        return self
