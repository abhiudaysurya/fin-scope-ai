"""
FinScope AI - Authentication Routes

API endpoints for user registration and JWT token management.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas import TokenResponse, UserCreate, UserLogin, UserResponse
from services.auth_service import authenticate_user, create_access_token, create_user
from services.database import get_db
from utils.config import get_settings
from utils.exceptions import AuthenticationError

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
)
async def register(user_data: UserCreate, db: AsyncSession = Depends(get_db)):
    """Create a new user account."""
    try:
        user = await create_user(
            db=db,
            username=user_data.username,
            email=user_data.email,
            password=user_data.password,
        )
        return UserResponse.model_validate(user)
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username or email already exists",
        )


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Login and obtain JWT token",
)
async def login(login_data: UserLogin, db: AsyncSession = Depends(get_db)):
    """Authenticate user and return JWT access token."""
    try:
        user = await authenticate_user(db, login_data.username, login_data.password)
        settings = get_settings()

        access_token = create_access_token(data={"sub": str(user.id)})

        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        )
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )
