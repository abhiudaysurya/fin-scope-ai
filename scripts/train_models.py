"""
FinScope AI - Model Training Script

End-to-end training pipeline that can be run standalone:
  1. Generate or load data
  2. Feature engineering
  3. Train all models (Logistic, XGBoost, DNN)
  4. Evaluate and compare
  5. Save artifacts

Usage:
    python -m scripts.train_models
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from loguru import logger

from data.feature_engineering import (
    FeaturePipeline,
    engineer_features,
    handle_missing_values,
    handle_outliers,
    split_data,
)
from data.generate_data import generate_synthetic_data
from models.dnn_model import DNNModel
from models.evaluation import compute_metrics, plot_all_metrics
from models.logistic_model import LogisticRegressionModel
from models.xgboost_model import XGBoostModel


def main():
    """Run the full training pipeline."""
    ARTIFACTS_DIR = Path("models/artifacts")
    PLOTS_DIR = Path("models/plots")
    DATA_DIR = Path("data/raw")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Generate Data ──
    logger.info("=" * 60)
    logger.info("STEP 1: Generating synthetic financial data")
    logger.info("=" * 60)
    df = generate_synthetic_data(n_samples=10000, random_seed=42)
    df.to_csv(DATA_DIR / "financial_data.csv", index=False)
    logger.info(f"Dataset shape: {df.shape} | Default rate: {df['default'].mean():.3f}")

    # ── 2. Feature Engineering & Split ──
    logger.info("=" * 60)
    logger.info("STEP 2: Feature engineering and data splitting")
    logger.info("=" * 60)
    train_df, val_df, test_df = split_data(df)

    pipeline = FeaturePipeline()
    X_train, y_train = pipeline.fit_transform(train_df)
    X_val = pipeline.transform(val_df)
    y_val = val_df["default"].values.astype(np.float32)
    X_test = pipeline.transform(test_df)
    y_test = test_df["default"].values.astype(np.float32)

    pipeline.save(str(ARTIFACTS_DIR))
    logger.info(f"Features: {len(pipeline.feature_columns)}")

    # ── 3. Train Models ──
    all_metrics = {}

    # 3a. Logistic Regression
    logger.info("=" * 60)
    logger.info("STEP 3a: Training Logistic Regression (baseline)")
    logger.info("=" * 60)
    lr_model = LogisticRegressionModel(C=0.5)
    lr_metrics = lr_model.train(X_train, y_train, X_val, y_val)
    lr_model.save(str(ARTIFACTS_DIR))

    test_proba_lr = lr_model.predict_proba(X_test)
    lr_test_metrics = compute_metrics(y_test, test_proba_lr)
    all_metrics["logistic_regression"] = {**lr_metrics, "test": lr_test_metrics}
    plot_all_metrics(y_test, test_proba_lr, "Logistic Regression", str(PLOTS_DIR))

    # 3b. XGBoost
    logger.info("=" * 60)
    logger.info("STEP 3b: Training XGBoost")
    logger.info("=" * 60)
    xgb_model = XGBoostModel()
    xgb_metrics = xgb_model.train(X_train, y_train, X_val, y_val, tune_hyperparams=False)
    xgb_model.save(str(ARTIFACTS_DIR))

    test_proba_xgb = xgb_model.predict_proba(X_test)
    xgb_test_metrics = compute_metrics(y_test, test_proba_xgb)
    all_metrics["xgboost"] = {**xgb_metrics, "test": xgb_test_metrics}
    plot_all_metrics(y_test, test_proba_xgb, "XGBoost", str(PLOTS_DIR))

    # 3c. DNN
    logger.info("=" * 60)
    logger.info("STEP 3c: Training Deep Neural Network")
    logger.info("=" * 60)
    dnn_model = DNNModel(
        hidden_dims=(128, 64, 32),
        dropout=0.3,
        learning_rate=1e-3,
        batch_size=256,
        epochs=100,
        patience=10,
    )
    dnn_metrics = dnn_model.train(X_train, y_train, X_val, y_val)
    dnn_model.save(str(ARTIFACTS_DIR))

    test_proba_dnn = dnn_model.predict_proba(X_test)
    dnn_test_metrics = compute_metrics(y_test, test_proba_dnn)
    all_metrics["dnn"] = {**dnn_metrics, "test": dnn_test_metrics}
    plot_all_metrics(y_test, test_proba_dnn, "DNN", str(PLOTS_DIR))

    # ── 4. Summary ──
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE — Model Comparison")
    logger.info("=" * 60)

    comparison = []
    for name, metrics in all_metrics.items():
        comparison.append({
            "model": name,
            "val_auc": metrics.get("val_auc", 0),
            "test_auc": metrics["test"]["roc_auc"],
            "test_f1": metrics["test"]["f1_score"],
            "test_precision": metrics["test"]["precision"],
            "test_recall": metrics["test"]["recall"],
        })

    comparison_df = pd.DataFrame(comparison)
    logger.info(f"\n{comparison_df.to_string(index=False)}")

    # Save metrics
    # Convert nested metrics for JSON serialization
    serializable_metrics = {}
    for k, v in all_metrics.items():
        serializable_metrics[k] = {
            mk: float(mv) if isinstance(mv, (np.floating, float)) else mv
            for mk, mv in v.items()
            if not isinstance(mv, dict)
        }
        if "test" in v:
            serializable_metrics[k]["test"] = {
                mk: float(mv) for mk, mv in v["test"].items()
            }

    with open(ARTIFACTS_DIR / "training_metrics.json", "w") as f:
        json.dump(serializable_metrics, f, indent=2)

    # Save feature columns
    with open(ARTIFACTS_DIR / "feature_names.json", "w") as f:
        json.dump(pipeline.feature_columns, f, indent=2)

    logger.info(f"Artifacts saved to {ARTIFACTS_DIR}")
    logger.info(f"Plots saved to {PLOTS_DIR}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
