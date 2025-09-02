"""
FinScope AI - Application Configuration

Centralized configuration management using pydantic-settings.
All configuration is loaded from environment variables / .env file.
"""

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    APP_NAME: str = "FinScope AI"
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = "development"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://finscope:finscope_secret@localhost:5432/finscope_db"
    DATABASE_URL_SYNC: str = "postgresql://finscope:finscope_secret@localhost:5432/finscope_db"

    # JWT
    JWT_SECRET_KEY: str = "change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Model
    MODEL_PATH: str = "./models/artifacts"
    DEFAULT_MODEL: str = "xgboost"
    MODEL_THRESHOLD: float = 0.5

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/finscope.log"

    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]

    @property
    def model_artifacts_path(self) -> Path:
        return Path(self.MODEL_PATH)

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance."""
    return Settings()
