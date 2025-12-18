"""
FinScope AI - Authentication Service

JWT-based authentication with password hashing.
"""

from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

from jose import JWTError, jwt
from loguru import logger
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from services.db_models import User
from utils.config import get_settings
from utils.exceptions import AuthenticationError

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """Hash a plain-text password."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.

    Args:
        data: Payload data (must include 'sub' key)
        expires_delta: Token expiration duration

    Returns:
        Encoded JWT string
    """
    settings = get_settings()
    to_encode = data.copy()

    expire = datetime.utcnow() + (
        expires_delta or timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})

    return jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    """
    Decode and validate a JWT token.

    Args:
        token: JWT string

    Returns:
        Decoded payload

    Raises:
        AuthenticationError: If token is invalid or expired
    """
    settings = get_settings()
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        return payload
    except JWTError as e:
        raise AuthenticationError(f"Invalid token: {str(e)}")


async def authenticate_user(db: AsyncSession, username: str, password: str) -> User:
    """
    Authenticate a user by username and password.

    Args:
        db: Database session
        username: Username
        password: Plain-text password

    Returns:
        Authenticated User object

    Raises:
        AuthenticationError: If credentials are invalid
    """
    result = await db.execute(select(User).where(User.username == username))
    user = result.scalar_one_or_none()

    if user is None or not verify_password(password, user.hashed_password):
        raise AuthenticationError("Invalid username or password")

    if not user.is_active:
        raise AuthenticationError("User account is disabled")

    logger.info(f"User authenticated: {username}")
    return user


async def create_user(
    db: AsyncSession,
    username: str,
    email: str,
    password: str,
    is_admin: bool = False,
) -> User:
    """
    Create a new user.

    Args:
        db: Database session
        username: Username
        email: Email address
        password: Plain-text password
        is_admin: Whether user has admin privileges

    Returns:
        Created User object
    """
    user = User(
        username=username,
        email=email,
        hashed_password=hash_password(password),
        is_admin=is_admin,
    )
    db.add(user)
    await db.flush()
    logger.info(f"User created: {username} (admin={is_admin})")
    return user
