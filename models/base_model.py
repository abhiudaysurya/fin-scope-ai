"""
FinScope AI - Base Model Interface

Abstract base class defining the contract for all risk prediction models.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class BaseModel(ABC):
    """Abstract base class for all FinScope prediction models."""

    def __init__(self, name: str):
        self.name = name
        self.model: Any = None
        self._is_trained: bool = False

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, **kwargs) -> Dict[str, float]:
        """Train the model and return training metrics."""
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of default (class 1) for each sample."""
        ...

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions using threshold."""
        return (self.predict_proba(X) >= threshold).astype(int)

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model artifacts to disk."""
        ...

    @abstractmethod
    def load(self, path: str) -> "BaseModel":
        """Load model artifacts from disk."""
        ...

    @property
    def is_trained(self) -> bool:
        return self._is_trained
