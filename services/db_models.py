"""
FinScope AI - Database Models (SQLAlchemy ORM)

Tables:
  - users: Authentication credentials
  - financial_records: User financial data
  - prediction_logs: Prediction audit trail
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    JSON,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from services.database import Base


class User(Base):
    """User accounts for API authentication."""

    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    financial_records = relationship("FinancialRecord", back_populates="user", lazy="selectin")
    prediction_logs = relationship("PredictionLog", back_populates="user", lazy="selectin")

    def __repr__(self):
        return f"<User(username={self.username})>"


class FinancialRecord(Base):
    """User financial data for risk assessment."""

    __tablename__ = "financial_records"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)

    # Demographics
    age = Column(Integer, nullable=False)
    employment_length_years = Column(Float)

    # Income
    annual_income = Column(Float, nullable=False)
    monthly_income = Column(Float)

    # Credit profile
    credit_score = Column(Integer)
    total_credit_lines = Column(Integer)
    num_open_accounts = Column(Integer)
    num_mortgage_accounts = Column(Integer)

    # Debt
    total_debt = Column(Float)
    credit_limit = Column(Float)

    # Payment history
    num_missed_payments_last_12m = Column(Integer, default=0)
    num_missed_payments_last_24m = Column(Integer, default=0)
    months_since_last_delinquency = Column(Integer, default=0)

    # Loan details
    loan_amount = Column(Float, nullable=False)
    loan_term_months = Column(Integer, nullable=False)
    interest_rate = Column(Float)

    # Transaction behavior
    monthly_transaction_count = Column(Integer)
    avg_transaction_amount = Column(Float)
    transaction_amount_std = Column(Float)

    # Balances
    savings_balance = Column(Float)
    checking_balance = Column(Float)

    # Negative events
    has_bankruptcy = Column(Integer, default=0)
    has_tax_liens = Column(Integer, default=0)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="financial_records")

    def to_feature_dict(self) -> dict:
        """Convert record to dictionary for model inference."""
        return {
            "age": self.age,
            "annual_income": self.annual_income,
            "monthly_income": self.monthly_income or (self.annual_income / 12),
            "employment_length_years": self.employment_length_years,
            "total_credit_lines": self.total_credit_lines,
            "total_debt": self.total_debt,
            "credit_limit": self.credit_limit,
            "credit_score": self.credit_score,
            "num_open_accounts": self.num_open_accounts,
            "num_mortgage_accounts": self.num_mortgage_accounts,
            "num_missed_payments_last_12m": self.num_missed_payments_last_12m,
            "num_missed_payments_last_24m": self.num_missed_payments_last_24m,
            "months_since_last_delinquency": self.months_since_last_delinquency,
            "loan_amount": self.loan_amount,
            "loan_term_months": self.loan_term_months,
            "interest_rate": self.interest_rate,
            "monthly_transaction_count": self.monthly_transaction_count,
            "avg_transaction_amount": self.avg_transaction_amount,
            "transaction_amount_std": self.transaction_amount_std,
            "savings_balance": self.savings_balance,
            "checking_balance": self.checking_balance,
            "has_bankruptcy": self.has_bankruptcy,
            "has_tax_liens": self.has_tax_liens,
        }

    def __repr__(self):
        return f"<FinancialRecord(user_id={self.user_id}, income={self.annual_income})>"


class PredictionLog(Base):
    """Audit trail for all predictions made by the system."""

    __tablename__ = "prediction_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True, index=True)
    financial_record_id = Column(UUID(as_uuid=True), ForeignKey("financial_records.id"), nullable=True)

    # Prediction details
    model_name = Column(String(50), nullable=False)
    model_version = Column(String(20), default="1.0.0")
    default_probability = Column(Float, nullable=False)
    risk_category = Column(String(20), nullable=False)  # LOW, MEDIUM, HIGH, CRITICAL
    threshold_used = Column(Float, default=0.5)

    # Input snapshot (for reproducibility)
    input_features = Column(JSON)

    # SHAP explanation (top features)
    feature_explanations = Column(JSON)

    # Metadata
    inference_time_ms = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    ip_address = Column(String(45))
    request_id = Column(String(36))

    # Relationships
    user = relationship("User", back_populates="prediction_logs")

    def __repr__(self):
        return f"<PredictionLog(prob={self.default_probability:.3f}, risk={self.risk_category})>"
