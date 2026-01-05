"""
FinScope AI - Prediction Service

Core business logic for running credit risk predictions.
"""

import time
import uuid
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from data.feature_engineering import FeaturePipeline
from models.base_model import BaseModel
from models.model_registry import ModelRegistry
from utils.config import get_settings
from utils.exceptions import ModelInferenceError


def categorize_risk(probability: float) -> str:
    """Categorize risk based on default probability."""
    if probability < 0.2:
        return "LOW"
    elif probability < 0.5:
        return "MEDIUM"
    elif probability < 0.8:
        return "HIGH"
    else:
        return "CRITICAL"


class PredictionService:
    """
    Service for running credit risk predictions.

    Handles model loading, feature transformation, inference,
    and result formatting.
    """

    def __init__(self, registry: Optional[ModelRegistry] = None):
        self.registry = registry or ModelRegistry()
        self._pipeline: Optional[FeaturePipeline] = None
        self._model: Optional[BaseModel] = None

    def _ensure_loaded(self, model_name: Optional[str] = None) -> None:
        """Ensure model and pipeline are loaded."""
        if self._pipeline is None:
            self._pipeline = self.registry.load_pipeline()
        if self._model is None or (model_name and self._model.name != model_name):
            self._model = self.registry.get_model(model_name)

    def predict_single(
        self,
        features: Dict,
        model_name: Optional[str] = None,
    ) -> Dict:
        """
        Run prediction on a single financial record.

        Args:
            features: Dictionary of raw financial features
            model_name: Optional model to use

        Returns:
            Prediction result with probability, risk category, and metadata
        """
        try:
            self._ensure_loaded(model_name)
            start_time = time.time()

            # Transform features
            X = self._pipeline.transform_single(features)

            # Predict
            probability = float(self._model.predict_proba(X)[0])
            risk_category = categorize_risk(probability)

            inference_time_ms = (time.time() - start_time) * 1000

            result = {
                "request_id": str(uuid.uuid4()),
                "default_probability": round(probability, 6),
                "risk_category": risk_category,
                "model_name": self._model.name,
                "threshold": get_settings().MODEL_THRESHOLD,
                "is_default": probability >= get_settings().MODEL_THRESHOLD,
                "inference_time_ms": round(inference_time_ms, 2),
            }

            logger.info(
                f"Prediction | prob={probability:.4f} | risk={risk_category} | "
                f"model={self._model.name} | time={inference_time_ms:.1f}ms"
            )
            return result

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise ModelInferenceError(f"Prediction failed: {str(e)}")

    def predict_batch(
        self,
        records: List[Dict],
        model_name: Optional[str] = None,
    ) -> List[Dict]:
        """
        Run predictions on a batch of financial records.

        Args:
            records: List of feature dictionaries
            model_name: Optional model to use

        Returns:
            List of prediction results
        """
        try:
            self._ensure_loaded(model_name)
            start_time = time.time()

            # Transform all records
            df = pd.DataFrame(records)
            X = self._pipeline.transform(df)

            # Batch predict
            probabilities = self._model.predict_proba(X)

            inference_time_ms = (time.time() - start_time) * 1000

            results = []
            for i, prob in enumerate(probabilities):
                prob = float(prob)
                results.append({
                    "request_id": str(uuid.uuid4()),
                    "record_index": i,
                    "default_probability": round(prob, 6),
                    "risk_category": categorize_risk(prob),
                    "model_name": self._model.name,
                    "threshold": get_settings().MODEL_THRESHOLD,
                    "is_default": prob >= get_settings().MODEL_THRESHOLD,
                })

            logger.info(
                f"Batch prediction | records={len(records)} | "
                f"model={self._model.name} | time={inference_time_ms:.1f}ms"
            )

            return results

        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise ModelInferenceError(f"Batch prediction failed: {str(e)}")
