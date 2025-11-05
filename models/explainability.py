"""
FinScope AI - SHAP Explainability

Feature explainability using SHAP (SHapley Additive exPlanations).
Provides both global and local feature importance explanations.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import shap
from loguru import logger


class SHAPExplainer:
    """SHAP-based model explainability for credit risk models."""

    def __init__(self, model, feature_names: List[str], model_type: str = "tree"):
        """
        Initialize SHAP explainer.

        Args:
            model: Trained model (sklearn, xgboost, or callable)
            feature_names: List of feature column names
            model_type: One of 'tree', 'linear', 'kernel'
        """
        self.feature_names = feature_names
        self.model_type = model_type

        if model_type == "tree":
            self.explainer = shap.TreeExplainer(model)
        elif model_type == "linear":
            self.explainer = shap.LinearExplainer(model, feature_names=feature_names)
        elif model_type == "kernel":
            self.explainer = shap.KernelExplainer(model, np.zeros((1, len(feature_names))))
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        logger.info(f"SHAP explainer initialized | type={model_type} | features={len(feature_names)}")

    def compute_shap_values(self, X: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for given samples.

        Args:
            X: Feature array of shape (n_samples, n_features)

        Returns:
            SHAP values array
        """
        shap_values = self.explainer.shap_values(X)
        # For binary classification, shap_values may be a list [class_0, class_1]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use class 1 (default)
        return shap_values

    def global_importance(self, X: np.ndarray, top_k: int = 15) -> Dict[str, float]:
        """
        Compute global feature importance (mean |SHAP|).

        Args:
            X: Feature array
            top_k: Number of top features to return

        Returns:
            Dictionary mapping feature name to mean absolute SHAP value
        """
        shap_values = self.compute_shap_values(X)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        importance = dict(zip(self.feature_names, mean_abs_shap))
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_k])

        logger.info(f"Top {top_k} features: {list(importance.keys())[:5]}...")
        return importance

    def local_explanation(self, X_single: np.ndarray) -> Dict[str, float]:
        """
        Compute local SHAP explanation for a single prediction.

        Args:
            X_single: Feature array of shape (1, n_features)

        Returns:
            Dictionary mapping feature name to SHAP value
        """
        shap_values = self.compute_shap_values(X_single)
        if shap_values.ndim > 1:
            shap_values = shap_values[0]
        return dict(zip(self.feature_names, shap_values.tolist()))

    def plot_summary(
        self,
        X: np.ndarray,
        save_path: Optional[str] = None,
        max_display: int = 15,
    ) -> plt.Figure:
        """Plot SHAP summary (beeswarm) plot."""
        shap_values = self.compute_shap_values(X)

        fig = plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False,
        )
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"SHAP summary plot saved to {save_path}")

        return fig

    def plot_waterfall(
        self,
        X_single: np.ndarray,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot SHAP waterfall for a single prediction."""
        shap_values = self.compute_shap_values(X_single)
        if shap_values.ndim > 1:
            shap_values = shap_values[0]

        base_value = self.explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1] if len(base_value) > 1 else base_value[0]

        explanation = shap.Explanation(
            values=shap_values,
            base_values=base_value,
            data=X_single[0] if X_single.ndim > 1 else X_single,
            feature_names=self.feature_names,
        )

        fig = plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation, show=False)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"SHAP waterfall plot saved to {save_path}")

        return fig
