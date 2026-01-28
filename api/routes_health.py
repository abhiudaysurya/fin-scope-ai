"""
FinScope AI - Health Check Route

System health and readiness endpoint.
"""

from fastapi import APIRouter
from loguru import logger

from api.schemas import HealthResponse
from models.model_registry import ModelRegistry
from utils.config import get_settings

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System health check",
)
async def health_check():
    """
    Check system health including model availability and database connectivity.
    """
    settings = get_settings()

    # Check models
    registry = ModelRegistry()
    models_available = registry.list_available_models()

    # Check database connectivity
    db_connected = False
    try:
        from services.database import engine
        async with engine.connect() as conn:
            await conn.execute(
                __import__("sqlalchemy").text("SELECT 1")
            )
            db_connected = True
    except Exception as e:
        logger.warning(f"Database health check failed: {e}")

    return HealthResponse(
        status="healthy" if db_connected else "degraded",
        version=settings.APP_VERSION,
        environment=settings.APP_ENV,
        models_available=models_available,
        database_connected=db_connected,
    )
