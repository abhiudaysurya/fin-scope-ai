"""
FinScope AI - Logging Configuration

Structured logging with loguru for application-wide observability.
"""

import sys
from pathlib import Path

from loguru import logger

from utils.config import get_settings


def setup_logging() -> None:
    """Configure application logging with loguru."""
    settings = get_settings()

    # Remove default handler
    logger.remove()

    # Console handler with colored output
    logger.add(
        sys.stdout,
        level=settings.LOG_LEVEL,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File handler with rotation
    log_path = Path(settings.LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        str(log_path),
        level=settings.LOG_LEVEL,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation="10 MB",
        retention="30 days",
        compression="gz",
        serialize=False,
    )

    logger.info(f"Logging initialized | level={settings.LOG_LEVEL} | env={settings.APP_ENV}")


def get_logger(name: str = "finscope"):
    """Get a named logger instance."""
    return logger.bind(module=name)
