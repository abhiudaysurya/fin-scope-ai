"""
FinScope AI - Pydantic Schemas

Request/response models with validation for the API layer.
"""

from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


# ──────────────────────── Auth Schemas ────────────────────────


class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=100, examples=["john_doe"])
    email: str = Field(..., examples=["john@example.com"])
    password: str = Field(..., min_length=8, max_length=128)


class UserLogin(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class UserResponse(BaseModel):
    id: UUID
    username: str
    email: str
    is_active: bool
    is_admin: bool
    created_at: datetime

    model_config = {"from_attributes": True}


# ──────────────────────── Financial Input Schemas ────────────────────────


class FinancialInput(BaseModel):
    """Input schema for a single financial record prediction."""

    age: int = Field(..., ge=18, le=100, description="Applicant age")
    annual_income: float = Field(..., gt=0, description="Annual income in USD")
    employment_length_years: Optional[float] = Field(None, ge=0, le=50)
    total_credit_lines: Optional[int] = Field(None, ge=0, le=50)
    total_debt: Optional[float] = Field(None, ge=0)
    credit_limit: Optional[float] = Field(None, ge=0)
    credit_score: Optional[int] = Field(None, ge=300, le=850)
    num_open_accounts: Optional[int] = Field(None, ge=0, le=50)
    num_mortgage_accounts: Optional[int] = Field(None, ge=0, le=10)
    num_missed_payments_last_12m: int = Field(0, ge=0, le=12)
    num_missed_payments_last_24m: int = Field(0, ge=0, le=24)
    months_since_last_delinquency: int = Field(0, ge=0)
    loan_amount: float = Field(..., gt=0, description="Requested loan amount")
    loan_term_months: int = Field(..., description="Loan term in months")
    interest_rate: Optional[float] = Field(None, ge=0, le=50)
    monthly_transaction_count: Optional[int] = Field(None, ge=0)
    avg_transaction_amount: Optional[float] = Field(None, ge=0)
    transaction_amount_std: Optional[float] = Field(None, ge=0)
    savings_balance: Optional[float] = Field(None, ge=0)
    checking_balance: Optional[float] = Field(None, ge=0)
    has_bankruptcy: int = Field(0, ge=0, le=1)
    has_tax_liens: int = Field(0, ge=0, le=1)

    @field_validator("loan_term_months")
    @classmethod
    def validate_loan_term(cls, v):
        valid_terms = [6, 12, 18, 24, 36, 48, 60, 72, 84, 120]
        if v not in valid_terms:
            raise ValueError(f"loan_term_months must be one of {valid_terms}")
        return v

    def to_feature_dict(self) -> dict:
        """Convert to dict for model inference with defaults."""
        data = self.model_dump()
        # Fill defaults for optional fields
        data["monthly_income"] = data.get("monthly_income") or data["annual_income"] / 12
        for key, default in [
            ("employment_length_years", 0.0),
            ("total_credit_lines", 3),
            ("total_debt", 0.0),
            ("credit_limit", data["annual_income"] * 0.5),
            ("credit_score", 650),
            ("num_open_accounts", 2),
            ("num_mortgage_accounts", 0),
            ("interest_rate", 10.0),
            ("monthly_transaction_count", 20),
            ("avg_transaction_amount", data["annual_income"] / 12 / 20),
            ("transaction_amount_std", data["annual_income"] / 12 / 40),
            ("savings_balance", 0.0),
            ("checking_balance", data["annual_income"] / 12),
        ]:
            if data.get(key) is None:
                data[key] = default
        return data

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 35,
                    "annual_income": 75000.0,
                    "employment_length_years": 5.0,
                    "total_credit_lines": 6,
                    "total_debt": 15000.0,
                    "credit_limit": 40000.0,
                    "credit_score": 720,
                    "num_open_accounts": 4,
                    "num_mortgage_accounts": 1,
                    "num_missed_payments_last_12m": 0,
                    "num_missed_payments_last_24m": 1,
                    "months_since_last_delinquency": 18,
                    "loan_amount": 25000.0,
                    "loan_term_months": 36,
                    "interest_rate": 8.5,
                    "monthly_transaction_count": 45,
                    "avg_transaction_amount": 250.0,
                    "transaction_amount_std": 120.0,
                    "savings_balance": 12000.0,
                    "checking_balance": 5000.0,
                    "has_bankruptcy": 0,
                    "has_tax_liens": 0,
                }
            ]
        }
    }


class BatchPredictRequest(BaseModel):
    """Batch prediction request."""

    records: List[FinancialInput] = Field(..., min_length=1, max_length=1000)
    model_name: Optional[str] = Field(None, description="Model to use for prediction")


# ──────────────────────── Prediction Response Schemas ────────────────────────


class PredictionResult(BaseModel):
    """Single prediction result."""

    request_id: str
    default_probability: float = Field(..., ge=0.0, le=1.0)
    risk_category: str = Field(..., description="LOW, MEDIUM, HIGH, or CRITICAL")
    model_name: str
    threshold: float
    is_default: bool
    inference_time_ms: Optional[float] = None


class BatchPredictionResult(BaseModel):
    """Single record result within a batch."""

    request_id: str
    record_index: int
    default_probability: float = Field(..., ge=0.0, le=1.0)
    risk_category: str
    model_name: str
    threshold: float
    is_default: bool


class BatchPredictResponse(BaseModel):
    """Batch prediction response."""

    predictions: List[BatchPredictionResult]
    total_records: int
    model_name: str


# ──────────────────────── Health Check ────────────────────────


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    models_available: Dict[str, bool]
    database_connected: bool


# ──────────────────────── Error Response ────────────────────────


class ErrorResponse(BaseModel):
    error: str
    code: str
    detail: Optional[str] = None
