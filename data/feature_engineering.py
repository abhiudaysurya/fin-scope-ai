"""
FinScope AI - Feature Engineering Pipeline

Implements robust feature engineering for credit risk modeling:
  - Debt-to-income ratio
  - Credit utilization rate
  - Rolling transaction variance
  - Missed payment frequency
  - Missing value imputation
  - Outlier handling
  - Train/validation/test split with stratification
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path


# Feature columns (all engineered + raw features fed to models)
FEATURE_COLUMNS: List[str] = [
    "age",
    "annual_income",
    "employment_length_years",
    "total_credit_lines",
    "total_debt",
    "credit_limit",
    "credit_score",
    "num_open_accounts",
    "num_mortgage_accounts",
    "num_missed_payments_last_12m",
    "num_missed_payments_last_24m",
    "months_since_last_delinquency",
    "loan_amount",
    "loan_term_months",
    "interest_rate",
    "monthly_transaction_count",
    "avg_transaction_amount",
    "transaction_amount_std",
    "savings_balance",
    "checking_balance",
    "has_bankruptcy",
    "has_tax_liens",
    # Engineered features
    "debt_to_income_ratio",
    "credit_utilization_rate",
    "transaction_variance_ratio",
    "missed_payment_frequency",
    "income_to_loan_ratio",
    "total_liquid_assets",
    "monthly_debt_payment_estimate",
    "payment_to_income_ratio",
]

TARGET_COLUMN: str = "default"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features from raw financial data.

    Args:
        df: Raw financial DataFrame

    Returns:
        DataFrame with additional engineered feature columns
    """
    logger.info("Engineering features...")
    df = df.copy()

    # Debt-to-income ratio
    df["debt_to_income_ratio"] = np.where(
        df["annual_income"] > 0,
        df["total_debt"] / df["annual_income"],
        0.0,
    )

    # Credit utilization rate
    df["credit_utilization_rate"] = np.where(
        df["credit_limit"] > 0,
        df["total_debt"] / df["credit_limit"],
        0.0,
    )

    # Transaction variance ratio (volatility relative to mean)
    df["transaction_variance_ratio"] = np.where(
        df["avg_transaction_amount"] > 0,
        df["transaction_amount_std"] / df["avg_transaction_amount"],
        0.0,
    )

    # Missed payment frequency (normalized over 24 months)
    df["missed_payment_frequency"] = df["num_missed_payments_last_24m"] / 24.0

    # Income to loan ratio
    df["income_to_loan_ratio"] = np.where(
        df["loan_amount"] > 0,
        df["annual_income"] / df["loan_amount"],
        0.0,
    )

    # Total liquid assets
    df["total_liquid_assets"] = df["savings_balance"].fillna(0) + df["checking_balance"].fillna(0)

    # Monthly debt payment estimate (simple amortization approximation)
    monthly_rate = df["interest_rate"] / 100 / 12
    term = df["loan_term_months"]
    df["monthly_debt_payment_estimate"] = np.where(
        monthly_rate > 0,
        df["loan_amount"] * monthly_rate * (1 + monthly_rate) ** term / ((1 + monthly_rate) ** term - 1),
        df["loan_amount"] / np.maximum(term, 1),
    )

    # Payment to income ratio
    monthly_income = df["annual_income"] / 12
    df["payment_to_income_ratio"] = np.where(
        monthly_income > 0,
        df["monthly_debt_payment_estimate"] / monthly_income,
        0.0,
    )

    logger.info(f"Engineered {8} new features | total columns: {len(df.columns)}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values with appropriate strategies.

    Strategy:
        - Numeric: median imputation
        - Create indicator columns for missingness
    """
    logger.info(f"Handling missing values | total missing: {df.isnull().sum().sum()}")
    df = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        n_missing = df[col].isnull().sum()
        if n_missing > 0:
            # Add missingness indicator
            df[f"{col}_missing"] = df[col].isnull().astype(int)
            # Impute with median
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.debug(f"  Imputed {col}: {n_missing} values with median={median_val:.2f}")

    return df


def handle_outliers(df: pd.DataFrame, columns: Optional[List[str]] = None, factor: float = 3.0) -> pd.DataFrame:
    """
    Cap outliers using IQR-based method.

    Args:
        df: Input DataFrame
        columns: Columns to check (defaults to numeric columns)
        factor: IQR multiplier for outlier detection

    Returns:
        DataFrame with capped outliers
    """
    df = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude binary/indicator columns
        columns = [c for c in columns if not c.startswith("has_") and c != TARGET_COLUMN and not c.endswith("_missing")]

    outlier_count = 0
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr

        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        if n_outliers > 0:
            df[col] = df[col].clip(lower, upper)
            outlier_count += n_outliers

    logger.info(f"Capped {outlier_count} outlier values across {len(columns)} columns")
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/validation/test with stratification on target.

    Args:
        df: Full DataFrame with features and target
        test_size: Fraction for test set
        val_size: Fraction for validation set
        random_state: Random seed

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # First split: train+val vs test
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET_COLUMN],
    )

    # Second split: train vs val
    relative_val_size = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=train_val[TARGET_COLUMN],
    )

    logger.info(
        f"Data split | train={len(train)} ({train[TARGET_COLUMN].mean():.3f}) | "
        f"val={len(val)} ({val[TARGET_COLUMN].mean():.3f}) | "
        f"test={len(test)} ({test[TARGET_COLUMN].mean():.3f})"
    )
    return train, val, test


class FeaturePipeline:
    """
    End-to-end feature pipeline for training and inference.

    Encapsulates feature engineering, missing value handling,
    outlier treatment, and feature scaling.
    """

    def __init__(self):
        self.scaler: Optional[StandardScaler] = None
        self.feature_columns: List[str] = FEATURE_COLUMNS
        self._is_fitted: bool = False

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the pipeline on training data and transform.

        Args:
            df: Training DataFrame with raw features + target

        Returns:
            Tuple of (X_scaled, y)
        """
        # Feature engineering
        df = engineer_features(df)
        df = handle_missing_values(df)
        df = handle_outliers(df)

        # Determine final feature columns (include missingness indicators)
        missing_indicator_cols = [c for c in df.columns if c.endswith("_missing")]
        self.feature_columns = FEATURE_COLUMNS + missing_indicator_cols
        # Keep only columns that exist
        self.feature_columns = [c for c in self.feature_columns if c in df.columns]

        X = df[self.feature_columns].values.astype(np.float32)
        y = df[TARGET_COLUMN].values.astype(np.float32)

        # Fit scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self._is_fitted = True

        logger.info(f"Pipeline fitted | features={len(self.feature_columns)} | samples={len(X)}")
        return X_scaled, y

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted pipeline.

        Args:
            df: DataFrame with raw features

        Returns:
            Scaled feature array
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline not fitted. Call fit_transform first.")

        df = engineer_features(df)
        df = handle_missing_values(df)
        # During inference, do NOT refit outlier bounds — just clip to training bounds

        X = df[self.feature_columns].values.astype(np.float32)
        X_scaled = self.scaler.transform(X)
        return X_scaled

    def transform_single(self, record: Dict) -> np.ndarray:
        """
        Transform a single record (dict) for real-time inference.

        Args:
            record: Dictionary of raw feature values

        Returns:
            Scaled feature array of shape (1, n_features)
        """
        df = pd.DataFrame([record])
        return self.transform(df)

    def save(self, path: str) -> None:
        """Save pipeline artifacts to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, path / "scaler.joblib")
        joblib.dump(self.feature_columns, path / "feature_columns.joblib")
        logger.info(f"Pipeline saved to {path}")

    def load(self, path: str) -> "FeaturePipeline":
        """Load pipeline artifacts from disk."""
        path = Path(path)
        self.scaler = joblib.load(path / "scaler.joblib")
        self.feature_columns = joblib.load(path / "feature_columns.joblib")
        self._is_fitted = True
        logger.info(f"Pipeline loaded from {path} | features={len(self.feature_columns)}")
        return self
