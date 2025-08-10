# FinScope AI

**Production-grade AI-driven credit risk and financial default prediction engine.**

FinScope AI evaluates user creditworthiness using structured financial datasets with multiple ML models, SHAP-based explainability, and a FastAPI microservice for real-time inference.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FinScope AI System                         │
├──────────────┬──────────────────────────────────────────────────────┤
│              │                                                      │
│   Client     │   POST /api/v1/predict                              │
│   (REST)     │   POST /api/v1/batch_predict                        │
│              │   GET  /api/v1/health                                │
│              │   POST /api/v1/auth/login                            │
│              │   POST /api/v1/auth/register                         │
│              │                                                      │
├──────────────┼──────────────────────────────────────────────────────┤
│              │                                                      │
│   FastAPI    │  ┌──────────┐  ┌────────────┐  ┌──────────────┐    │
│   Gateway    │  │  Auth    │  │ Prediction │  │   Health     │    │
│   (uvicorn)  │  │  Routes  │  │  Routes    │  │   Routes     │    │
│              │  └────┬─────┘  └─────┬──────┘  └──────────────┘    │
│              │       │              │                               │
│              │  ┌────▼─────┐  ┌─────▼──────────────────────┐      │
│              │  │  JWT     │  │  Prediction Service         │      │
│              │  │  Auth    │  │  ┌────────────────────────┐ │      │
│              │  │  Service │  │  │  Feature Pipeline      │ │      │
│              │  └──────────┘  │  │  (transform + scale)   │ │      │
│              │                │  └────────────────────────┘ │      │
│              │                │  ┌────────────────────────┐ │      │
│              │                │  │  Model Registry         │ │      │
│              │                │  │  ┌──────────────────┐  │ │      │
│              │                │  │  │ Logistic Reg.    │  │ │      │
│              │                │  │  │ XGBoost          │  │ │      │
│              │                │  │  │ DNN (PyTorch)    │  │ │      │
│              │                │  │  └──────────────────┘  │ │      │
│              │                │  └────────────────────────┘ │      │
│              │                └────────────────────────────┘      │
│              │                         │                           │
├──────────────┼─────────────────────────┼───────────────────────────┤
│              │                         │                           │
│  PostgreSQL  │  ┌──────────────────────▼──────────────────────┐   │
│  Database    │  │  users │ financial_records │ prediction_logs │   │
│              │  └─────────────────────────────────────────────┘   │
│              │                                                      │
└──────────────┴──────────────────────────────────────────────────────┘
```

### Project Structure

```
finscope-ai/
├── api/                    # FastAPI application layer
│   ├── main.py             # App factory, middleware, lifespan
│   ├── schemas.py          # Pydantic request/response models
│   ├── dependencies.py     # Auth & DB dependency injection
│   ├── routes_predict.py   # /predict, /batch_predict endpoints
│   ├── routes_auth.py      # /auth/register, /auth/login
│   └── routes_health.py    # /health endpoint
│
├── data/                   # Data pipeline
│   ├── generate_data.py    # Synthetic data generator
│   └── feature_engineering.py  # Feature engineering, scaling, splitting
│
├── models/                 # ML model implementations
│   ├── base_model.py       # Abstract base class
│   ├── logistic_model.py   # Logistic Regression (baseline)
│   ├── xgboost_model.py    # XGBoost with hyperparameter tuning
│   ├── dnn_model.py        # PyTorch Deep Neural Network
│   ├── evaluation.py       # ROC, PR, confusion matrix, calibration
│   ├── explainability.py   # SHAP-based feature explanations
│   ├── model_registry.py   # Model loading & caching
│   └── artifacts/          # Saved model files
│
├── services/               # Business logic & database
│   ├── database.py         # SQLAlchemy async engine & session
│   ├── db_models.py        # ORM models (User, FinancialRecord, PredictionLog)
│   ├── auth_service.py     # JWT authentication & password hashing
│   └── prediction_service.py  # Core prediction orchestration
│
├── utils/                  # Shared utilities
│   ├── config.py           # Pydantic settings (env-based)
│   ├── logging.py          # Loguru structured logging
│   └── exceptions.py       # Custom exception hierarchy
│
├── scripts/                # CLI scripts
│   └── train_models.py     # End-to-end training pipeline
│
├── notebooks/              # Jupyter notebooks
│   └── model_training.ipynb  # Interactive training & evaluation
│
├── tests/                  # Unit tests
│   ├── test_data_pipeline.py
│   ├── test_models.py
│   └── test_schemas.py
│
├── alembic/                # Database migrations
│   ├── env.py
│   └── versions/
│
├── Dockerfile              # Multi-stage production image
├── docker-compose.yml      # API + PostgreSQL orchestration
├── pyproject.toml          # Python project & dependency config
├── alembic.ini             # Migration config
├── .env.example            # Environment template
└── .gitignore
```

---

## Features

### Data Pipeline
- Synthetic financial data generation (10K+ records)
- Engineered features: debt-to-income ratio, credit utilization, transaction variance, missed payment frequency
- Robust missing value imputation (median + missingness indicators)
- IQR-based outlier capping
- Stratified train/validation/test split

### Machine Learning Models
| Model | Type | Description |
|-------|------|-------------|
| **Logistic Regression** | Baseline | L2-regularized with balanced class weights |
| **XGBoost** | Gradient Boosting | With hyperparameter tuning via RandomizedSearchCV |
| **DNN** | Deep Learning | 3-layer PyTorch network with BatchNorm, Dropout, early stopping |

All models output calibrated probability scores (0-1) for default risk.

### Evaluation Metrics
- ROC-AUC curve
- Precision-Recall curve
- Confusion Matrix
- Calibration Curve
- F1 Score, Accuracy, Precision, Recall

### Explainability
- SHAP TreeExplainer for XGBoost
- Global feature importance (beeswarm plot)
- Local explanations (waterfall plot per prediction)

### API Endpoints
| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/api/v1/predict` | JWT | Single record risk prediction |
| `POST` | `/api/v1/batch_predict` | JWT | Batch prediction (up to 1000 records) |
| `GET` | `/api/v1/health` | None | System health check |
| `POST` | `/api/v1/auth/register` | None | User registration |
| `POST` | `/api/v1/auth/login` | None | JWT token generation |

### Production Quality
- JWT authentication with bcrypt password hashing
- Structured logging (loguru) with rotation
- Pydantic input validation with detailed error messages
- Custom exception hierarchy
- Prediction audit trail in PostgreSQL
- Docker containerization with health checks
- Environment-based configuration

---

## Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose (for containerized deployment)
- PostgreSQL 14+ (or use Docker)

### 1. Clone & Setup

```bash
git clone https://github.com/your-username/finscope-ai.git
cd finscope-ai

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -e ".[dev]"

# Copy environment config
cp .env.example .env
# Edit .env with your settings
```

### 2. Train Models

```bash
# Generate synthetic data and train all models
python -m scripts.train_models
```

Or use the interactive notebook:
```bash
jupyter notebook notebooks/model_training.ipynb
```

### 3. Start the API

**Option A: Local (requires PostgreSQL running)**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Option B: Docker Compose (recommended)**
```bash
docker-compose up --build
```

### 4. Test the API

```bash
# Register a user
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username": "testuser", "email": "test@example.com", "password": "securepass123"}'

# Login and get token
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "testuser", "password": "securepass123"}'

# Make a prediction (use token from login response)
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR_TOKEN>" \
  -d '{
    "age": 35,
    "annual_income": 75000,
    "employment_length_years": 5,
    "total_debt": 15000,
    "credit_limit": 40000,
    "credit_score": 720,
    "num_missed_payments_last_12m": 0,
    "num_missed_payments_last_24m": 1,
    "months_since_last_delinquency": 18,
    "loan_amount": 25000,
    "loan_term_months": 36,
    "interest_rate": 8.5,
    "has_bankruptcy": 0,
    "has_tax_liens": 0
  }'

# Health check
curl http://localhost:8000/api/v1/health
```

### 5. Run Tests

```bash
pytest tests/ -v --tb=short
```

---

## API Response Examples

### POST /api/v1/predict
```json
{
  "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "default_probability": 0.1234,
  "risk_category": "LOW",
  "model_name": "xgboost",
  "threshold": 0.5,
  "is_default": false,
  "inference_time_ms": 2.45
}
```

### Risk Categories
| Probability Range | Category |
|-------------------|----------|
| 0.0 - 0.2 | **LOW** |
| 0.2 - 0.5 | **MEDIUM** |
| 0.5 - 0.8 | **HIGH** |
| 0.8 - 1.0 | **CRITICAL** |

---

## Database Schema

```sql
-- Users (authentication)
users (id UUID PK, username, email, hashed_password, is_active, is_admin, created_at)

-- Financial records (input data)
financial_records (id UUID PK, user_id FK, age, annual_income, credit_score, ..., created_at)

-- Prediction audit trail
prediction_logs (id UUID PK, user_id FK, model_name, default_probability, risk_category,
                 input_features JSON, feature_explanations JSON, inference_time_ms, created_at)
```

---

## Configuration

All settings are managed via environment variables (`.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://...` | Async database connection |
| `JWT_SECRET_KEY` | - | Secret for signing JWT tokens |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | `30` | Token expiry duration |
| `DEFAULT_MODEL` | `xgboost` | Model used for predictions |
| `MODEL_THRESHOLD` | `0.5` | Decision threshold for default |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `APP_ENV` | `development` | Environment (development/production) |

---

## Deployment

### Docker Compose (Recommended)

```bash
# Build and start all services
docker-compose up --build -d

# View logs
docker-compose logs -f api

# Stop
docker-compose down
```

### Database Migrations

```bash
# Generate migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.
