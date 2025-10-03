"""
FinScope AI - Model Evaluation Metrics

Comprehensive evaluation suite for credit risk models:
  - ROC-AUC
  - Precision-Recall
  - Confusion Matrix
  - Calibration Curve
"""

from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities for positive class
        threshold: Decision threshold

    Returns:
        Dictionary of computed metrics
    """
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "average_precision": average_precision_score(y_true, y_proba),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "threshold": threshold,
    }

    logger.info(
        f"Metrics | AUC={metrics['roc_auc']:.4f} | AP={metrics['average_precision']:.4f} | "
        f"F1={metrics['f1_score']:.4f} | Acc={metrics['accuracy']:.4f}"
    )
    return metrics


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"{model_name} (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"ROC curve saved to {save_path}")

    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="darkorange", lw=2, label=f"{model_name} (AP = {ap:.4f})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14)
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"PR curve saved to {save_path}")

    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    model_name: str = "Model",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot confusion matrix heatmap."""
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Default", "Default"],
        yticklabels=["No Default", "Default"],
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Confusion matrix saved to {save_path}")

    return fig


def plot_calibration_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    model_name: str = "Model",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot calibration (reliability) curve."""
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(prob_pred, prob_true, marker="o", lw=2, label=model_name)
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Perfectly calibrated")
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title("Calibration Curve", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Calibration curve saved to {save_path}")

    return fig


def plot_all_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Model",
    save_dir: Optional[str] = None,
) -> Dict[str, plt.Figure]:
    """Generate all evaluation plots."""
    from pathlib import Path

    figures = {}

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    figures["roc"] = plot_roc_curve(
        y_true, y_proba, model_name,
        save_path=str(save_dir / f"{model_name}_roc.png") if save_dir else None,
    )
    figures["pr"] = plot_precision_recall_curve(
        y_true, y_proba, model_name,
        save_path=str(save_dir / f"{model_name}_pr.png") if save_dir else None,
    )
    figures["cm"] = plot_confusion_matrix(
        y_true, y_proba, 0.5, model_name,
        save_path=str(save_dir / f"{model_name}_cm.png") if save_dir else None,
    )
    figures["calibration"] = plot_calibration_curve(
        y_true, y_proba, 10, model_name,
        save_path=str(save_dir / f"{model_name}_calibration.png") if save_dir else None,
    )

    return figures
