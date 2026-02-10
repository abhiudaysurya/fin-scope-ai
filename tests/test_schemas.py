"""
FinScope AI - API Schema Validation Tests
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from pydantic import ValidationError

from api.schemas import FinancialInput, BatchPredictRequest


class TestFinancialInputSchema:
    def test_valid_input(self):
        data = FinancialInput(
            age=35,
            annual_income=75000.0,
            loan_amount=25000.0,
            loan_term_months=36,
        )
        assert data.age == 35
        assert data.annual_income == 75000.0

    def test_invalid_age(self):
        with pytest.raises(ValidationError):
            FinancialInput(age=15, annual_income=50000, loan_amount=10000, loan_term_months=36)

    def test_invalid_loan_term(self):
        with pytest.raises(ValidationError):
            FinancialInput(age=30, annual_income=50000, loan_amount=10000, loan_term_months=37)

    def test_to_feature_dict_fills_defaults(self):
        data = FinancialInput(
            age=30,
            annual_income=60000.0,
            loan_amount=20000.0,
            loan_term_months=36,
        )
        features = data.to_feature_dict()
        assert features["credit_score"] == 650  # default
        assert features["monthly_income"] == 5000.0  # derived

    def test_full_input(self):
        data = FinancialInput(
            age=45,
            annual_income=120000.0,
            employment_length_years=15.0,
            total_credit_lines=8,
            total_debt=35000.0,
            credit_limit=80000.0,
            credit_score=750,
            num_open_accounts=5,
            num_mortgage_accounts=1,
            num_missed_payments_last_12m=0,
            num_missed_payments_last_24m=0,
            months_since_last_delinquency=0,
            loan_amount=50000.0,
            loan_term_months=60,
            interest_rate=6.5,
            monthly_transaction_count=50,
            avg_transaction_amount=300.0,
            transaction_amount_std=150.0,
            savings_balance=25000.0,
            checking_balance=8000.0,
            has_bankruptcy=0,
            has_tax_liens=0,
        )
        features = data.to_feature_dict()
        assert features["credit_score"] == 750
        assert features["has_bankruptcy"] == 0


class TestBatchPredictRequest:
    def test_valid_batch(self):
        records = [
            FinancialInput(age=30, annual_income=60000, loan_amount=20000, loan_term_months=36),
            FinancialInput(age=45, annual_income=90000, loan_amount=40000, loan_term_months=60),
        ]
        batch = BatchPredictRequest(records=records)
        assert len(batch.records) == 2

    def test_empty_batch_rejected(self):
        with pytest.raises(ValidationError):
            BatchPredictRequest(records=[])
