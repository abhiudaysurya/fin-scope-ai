"""
FinScope AI - Model Registry

Manages model loading, caching, and selection for inference.
"""

from pathlib import Path
from typing import Dict, Optional

from loguru import logger

from data.feature_engineering import FeaturePipeline
from models.base_model import BaseModel
from models.dnn_model import DNNModel
from models.logistic_model import LogisticRegressionModel
from models.xgboost_model import XGBoostModel
from utils.config import get_settings
from utils.exceptions import ModelNotFoundError


class ModelRegistry:
    """
    Registry for managing multiple trained models.

    Handles loading models from disk, caching them in memory,
    and providing a unified interface for prediction.
    """

    MODEL_CLASSES = {
        "logistic_regression": LogisticRegressionModel,
        "xgboost": XGBoostModel,
        "dnn": DNNModel,
    }

    def __init__(self, artifacts_path: Optional[str] = None):
        self.artifacts_path = Path(artifacts_path or get_settings().MODEL_PATH)
        self._models: Dict[str, BaseModel] = {}
        self._pipeline: Optional[FeaturePipeline] = None

    def load_model(self, model_name: str) -> BaseModel:
        """
        Load a model from disk and cache it.

        Args:
            model_name: Name of the model ('logistic_regression', 'xgboost', 'dnn')

        Returns:
            Loaded model instance
        """
        if model_name in self._models:
            return self._models[model_name]

        if model_name not in self.MODEL_CLASSES:
            raise ModelNotFoundError(model_name)

        model_class = self.MODEL_CLASSES[model_name]
        model = model_class()

        model_path = self.artifacts_path
        if not model_path.exists():
            raise ModelNotFoundError(model_name)

        try:
            model.load(str(model_path))
            self._models[model_name] = model
            logger.info(f"Model '{model_name}' loaded and cached")
            return model
        except FileNotFoundError:
            raise ModelNotFoundError(model_name)

    def load_pipeline(self) -> FeaturePipeline:
        """Load the feature pipeline from disk."""
        if self._pipeline is not None:
            return self._pipeline

        self._pipeline = FeaturePipeline()
        self._pipeline.load(str(self.artifacts_path))
        return self._pipeline

    def get_model(self, model_name: Optional[str] = None) -> BaseModel:
        """
        Get a model by name, loading if necessary.

        Args:
            model_name: Model name (defaults to settings.DEFAULT_MODEL)

        Returns:
            Ready-to-use model instance
        """
        if model_name is None:
            model_name = get_settings().DEFAULT_MODEL
        return self.load_model(model_name)

    def list_available_models(self) -> Dict[str, bool]:
        """List all model types and whether they're loaded."""
        return {name: name in self._models for name in self.MODEL_CLASSES}

    def unload_model(self, model_name: str) -> None:
        """Remove a model from the cache."""
        if model_name in self._models:
            del self._models[model_name]
            logger.info(f"Model '{model_name}' unloaded from cache")
