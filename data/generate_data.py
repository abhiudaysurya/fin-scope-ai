"""
FinScope AI - Synthetic Data Generator

Generates realistic synthetic financial data for training and testing.
"""

import numpy as np
import pandas as pd
from loguru import logger


def generate_synthetic_data(n_samples: int = 10000, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic financial data for credit risk modeling.

    Features generated:
        - age: Applicant age (21-70)
        - annual_income: Annual income in USD
        - monthly_income: Derived from annual
        - employment_length_years: Years at current employer
        - total_credit_lines: Number of open credit lines
        - total_debt: Total outstanding debt
        - credit_limit: Total available credit
        - credit_score: FICO-like score (300-850)
        - num_open_accounts: Number of active accounts
        - num_mortgage_accounts: Number of mortgage accounts
        - num_missed_payments_last_12m: Missed payments in past 12 months
        - num_missed_payments_last_24m: Missed payments in past 24 months
        - months_since_last_delinquency: Months since last delinquency
        - loan_amount: Requested loan amount
        - loan_term_months: Loan term (12, 24, 36, 48, 60)
        - interest_rate: Assigned interest rate
        - monthly_transaction_count: Average monthly transactions
        - avg_transaction_amount: Average transaction value
        - transaction_amount_std: Std deviation of transaction amounts
        - savings_balance: Current savings balance
        - checking_balance: Current checking balance
        - has_bankruptcy: Whether applicant has bankruptcy history
        - has_tax_liens: Whether applicant has tax liens
        - default: Target variable (0 = no default, 1 = default)

    Args:
        n_samples: Number of samples to generate
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame with synthetic financial data
    """
    logger.info(f"Generating {n_samples} synthetic financial records (seed={random_seed})")
    rng = np.random.RandomState(random_seed)

    # Demographics
    age = rng.randint(21, 71, size=n_samples)
    employment_length = np.clip(rng.exponential(5, size=n_samples), 0, 40).round(1)

    # Income (correlated with age and employment)
    base_income = 25000 + age * 500 + employment_length * 1500
    annual_income = np.clip(
        base_income + rng.normal(0, 15000, size=n_samples), 15000, 500000
    ).round(2)
    monthly_income = (annual_income / 12).round(2)

    # Credit profile
    credit_score = np.clip(
        580 + (annual_income / 5000) + employment_length * 3 + rng.normal(0, 50, n_samples),
        300, 850,
    ).astype(int)

    total_credit_lines = np.clip(rng.poisson(6, size=n_samples), 1, 30)
    num_open_accounts = np.clip(
        (total_credit_lines * rng.uniform(0.3, 0.9, n_samples)).astype(int), 1, 25
    )
    num_mortgage_accounts = (rng.random(n_samples) < 0.35).astype(int)

    # Debt & Credit
    credit_limit = np.clip(
        annual_income * rng.uniform(0.3, 1.5, n_samples), 5000, 200000
    ).round(2)
    total_debt = np.clip(
        credit_limit * rng.beta(2, 5, n_samples), 0, credit_limit
    ).round(2)

    # Payment history (lower credit scores -> more missed payments)
    miss_prob = np.clip(1 - (credit_score - 300) / 550, 0.01, 0.6)
    num_missed_12m = rng.binomial(6, miss_prob).astype(int)
    num_missed_24m = num_missed_12m + rng.binomial(6, miss_prob * 0.7).astype(int)
    months_since_delinquency = np.where(
        num_missed_24m > 0,
        rng.randint(1, 25, n_samples),
        0,
    )

    # Loan details
    loan_amount = np.clip(
        annual_income * rng.uniform(0.1, 0.8, n_samples), 1000, 100000
    ).round(2)
    loan_term_months = rng.choice([12, 24, 36, 48, 60], size=n_samples)
    interest_rate = np.clip(
        18 - (credit_score - 300) / 50 + rng.normal(0, 1.5, n_samples), 3.0, 30.0
    ).round(2)

    # Transaction behavior
    monthly_tx_count = np.clip(rng.poisson(30, n_samples), 5, 200)
    avg_tx_amount = np.clip(
        monthly_income / monthly_tx_count * rng.uniform(0.5, 1.5, n_samples), 10, 5000
    ).round(2)
    tx_amount_std = (avg_tx_amount * rng.uniform(0.2, 0.8, n_samples)).round(2)

    # Balances
    savings_balance = np.clip(
        annual_income * rng.exponential(0.3, n_samples), 0, 500000
    ).round(2)
    checking_balance = np.clip(
        monthly_income * rng.uniform(0.1, 3.0, n_samples), 100, 100000
    ).round(2)

    # Negative events
    has_bankruptcy = (rng.random(n_samples) < 0.05).astype(int)
    has_tax_liens = (rng.random(n_samples) < 0.08).astype(int)

    # --- Target variable: default ---
    # Logistic model for default probability
    log_odds = (
        -3.5
        + 0.8 * (total_debt / np.maximum(annual_income, 1))       # debt-to-income
        + 0.6 * (total_debt / np.maximum(credit_limit, 1))        # credit utilization
        + 0.3 * num_missed_12m
        + 0.15 * num_missed_24m
        - 0.005 * credit_score
        + 1.2 * has_bankruptcy
        + 0.5 * has_tax_liens
        - 0.02 * employment_length
        + 0.4 * (tx_amount_std / np.maximum(avg_tx_amount, 1))   # transaction variance ratio
        - 0.001 * savings_balance / 1000
        + rng.normal(0, 0.3, n_samples)
    )
    default_prob = 1 / (1 + np.exp(-log_odds))
    default = (rng.random(n_samples) < default_prob).astype(int)

    # Introduce ~3% missing values in select columns
    df = pd.DataFrame({
        "age": age,
        "annual_income": annual_income,
        "monthly_income": monthly_income,
        "employment_length_years": employment_length,
        "total_credit_lines": total_credit_lines,
        "total_debt": total_debt,
        "credit_limit": credit_limit,
        "credit_score": credit_score,
        "num_open_accounts": num_open_accounts,
        "num_mortgage_accounts": num_mortgage_accounts,
        "num_missed_payments_last_12m": num_missed_12m,
        "num_missed_payments_last_24m": num_missed_24m,
        "months_since_last_delinquency": months_since_delinquency,
        "loan_amount": loan_amount,
        "loan_term_months": loan_term_months,
        "interest_rate": interest_rate,
        "monthly_transaction_count": monthly_tx_count,
        "avg_transaction_amount": avg_tx_amount,
        "transaction_amount_std": tx_amount_std,
        "savings_balance": savings_balance,
        "checking_balance": checking_balance,
        "has_bankruptcy": has_bankruptcy,
        "has_tax_liens": has_tax_liens,
        "default": default,
    })

    # Inject missing values
    cols_with_missing = [
        "employment_length_years", "credit_score", "months_since_last_delinquency",
        "savings_balance", "avg_transaction_amount",
    ]
    for col in cols_with_missing:
        mask = rng.random(n_samples) < 0.03
        df.loc[mask, col] = np.nan

    logger.info(
        f"Generated {n_samples} records | default_rate={default.mean():.3f} | "
        f"missing_values={df.isnull().sum().sum()}"
    )
    return df


if __name__ == "__main__":
    from pathlib import Path

    df = generate_synthetic_data(n_samples=10000)
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "financial_data.csv", index=False)
    print(f"Saved {len(df)} records to {out_dir / 'financial_data.csv'}")
    print(f"Default rate: {df['default'].mean():.3f}")
    print(f"Shape: {df.shape}")
