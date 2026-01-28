"""
FinScope AI - FastAPI Application Entry Point

Production-grade FastAPI application with:
  - CORS middleware
  - Exception handling
  - Lifespan management (startup/shutdown)
  - Structured logging
  - API versioning
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from api.routes_auth import router as auth_router
from api.routes_health import router as health_router
from api.routes_predict import router as predict_router
from services.database import close_db, init_db
from utils.config import get_settings
from utils.exceptions import (
    AuthenticationError,
    DataValidationError,
    FinScopeError,
    ModelInferenceError,
    ModelNotFoundError,
)
from utils.logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown hooks."""
    # Startup
    setup_logging()
    settings = get_settings()
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION} ({settings.APP_ENV})")

    # Initialize database
    try:
        await init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.warning(f"Database initialization skipped: {e}")

    yield

    # Shutdown
    logger.info("Shutting down...")
    await close_db()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """Factory function to create the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "AI-driven credit risk and financial default prediction engine. "
            "Evaluates user creditworthiness using structured financial datasets "
            "with multiple ML models and SHAP-based explainability."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS Middleware ──
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Exception Handlers ──
    @app.exception_handler(DataValidationError)
    async def data_validation_handler(request: Request, exc: DataValidationError):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": exc.message, "code": exc.code},
        )

    @app.exception_handler(ModelNotFoundError)
    async def model_not_found_handler(request: Request, exc: ModelNotFoundError):
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"error": exc.message, "code": exc.code},
        )

    @app.exception_handler(ModelInferenceError)
    async def model_inference_handler(request: Request, exc: ModelInferenceError):
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": exc.message, "code": exc.code},
        )

    @app.exception_handler(AuthenticationError)
    async def auth_error_handler(request: Request, exc: AuthenticationError):
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"error": exc.message, "code": exc.code},
        )

    @app.exception_handler(FinScopeError)
    async def finscope_error_handler(request: Request, exc: FinScopeError):
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": exc.message, "code": exc.code},
        )

    # ── Routes ──
    app.include_router(health_router, prefix="/api/v1")
    app.include_router(auth_router, prefix="/api/v1")
    app.include_router(predict_router, prefix="/api/v1")

    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "app": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "docs": "/docs",
        }

    return app


# Application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
