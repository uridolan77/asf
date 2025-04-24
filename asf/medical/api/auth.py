"""
Unified authentication module for the Medical Research Synthesizer API.

This module provides a comprehensive JWT-based authentication system for the FastAPI implementation,
including user management, token generation, validation, and role-based access control.
"""
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Callable, Any

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, EmailStr, Field

from asf.medical.core.config import settings
from asf.medical.storage.database import get_db_session
from asf.medical.storage.repositories.user_repository import UserRepository
from asf.medical.storage.models import MedicalUser as DBUser
from asf.medical.storage.models.user import Role
from asf.medical.core.observability import log_error
from asf.medical.core.exceptions import DatabaseError

logger = logging.getLogger(__name__)

# OAuth2 password bearer scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/v1/auth/token")

# Password context for hashing and verifying passwords
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Demo Mode Authentication Bypass
DEMO_MODE_ENABLED = True  # Set to False in production
DEMO_USER_EMAIL = "demo_admin@example.com"
DEMO_USER_ROLE = "admin"


class Token(BaseModel):
    """
    Token model for authentication responses.
    """
    access_token: str
    token_type: str
    expires_at: datetime


class TokenData(BaseModel):
    """
    Token data model for decoded JWT tokens.
    """
    username: Optional[str] = None
    user_id: Optional[str] = None
    role: Optional[str] = None
    exp: Optional[datetime] = None


class UserCreate(BaseModel):
    """
    User creation request model.
    """
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None
    role: Optional[Role] = Role.USER


class UserUpdate(BaseModel):
    """
    User update request model.
    """
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    password: Optional[str] = Field(None, min_length=8)
    role: Optional[Role] = None
    is_active: Optional[bool] = None


class UserResponse(BaseModel):
    """
    User response model for API responses.
    """
    id: str
    username: str
    email: str
    full_name: Optional[str]
    role: str
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime]


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against a hash.

    Args:
        plain_password: The plain text password
        hashed_password: The hashed password to compare against

    Returns:
        bool: True if password matches, False otherwise
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    Hash a password.

    Args:
        password: The password to hash

    Returns:
        str: The hashed password
    """
    return pwd_context.hash(password)


async def authenticate_user(username: str, password: str, db: AsyncSession) -> Optional[DBUser]:
    """
    Authenticate a user by username and password.

    Args:
        username: The username to authenticate
        password: The password to verify
        db: Database session

    Returns:
        Optional[DBUser]: The user if authentication is successful, None otherwise
    """
    try:
        user_repo = UserRepository(db)
        user = await user_repo.get_by_username(username)

        if not user:
            return None

        if not verify_password(password, user.password):
            return None

        return user
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        logger.error(f"Database error: {str(e)}")
        raise DatabaseError(f"Database operation failed: {str(e)}")
        log_error(e, {"username": username})
        return None


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.

    Args:
        data: The data to encode in the token
        expires_delta: Token expiration time delta

    Returns:
        str: The encoded JWT token
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db_session)) -> DBUser:
    """
    Get the current authenticated user from the JWT token.

    Args:
        token: The JWT token
        db: Database session

    Returns:
        DBUser: The authenticated user

    Raises:
        HTTPException: If authentication fails
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: str = payload.get("sub")
        user_id: str = payload.get("id")

        if username is None or user_id is None:
            raise credentials_exception

        token_data = TokenData(username=username, user_id=user_id)
    except JWTError as e:
        logger.error(f"Error: {str(e)}")
        log_error(e, {"token": token[:10] + "..."})
        raise credentials_exception

    user_repo = UserRepository(db)
    user = await user_repo.get_by_id(token_data.user_id)

    if user is None:
        raise credentials_exception

    return user


async def get_current_active_user(current_user: DBUser = Depends(get_current_user)) -> DBUser:
    """
    Get the current active user.

    Args:
        current_user: The current authenticated user

    Returns:
        DBUser: The active user

    Raises:
        HTTPException: If the user is inactive
    """
    if not current_user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Inactive user")
    return current_user


async def get_current_active_user_demo_mode(current_user=Depends(get_current_active_user)):
    """
    Demo mode override for the get_current_active_user dependency.
    In demo mode, this will return a mock admin user without requiring authentication.
    
    Args:
        current_user: The actual authenticated user (not used in demo mode)
        
    Returns:
        A mock admin user for demonstration purposes
    """
    if DEMO_MODE_ENABLED:
        # In demo mode, return a mock admin user
        logger.warning("DEMO MODE ACTIVE: Authentication bypassed. Using demo admin account!")
        from ...storage.models import MedicalUser, Role
        # Create a fake admin user for demonstration
        return MedicalUser(
            id=0,
            email=DEMO_USER_EMAIL,
            hashed_password="demo_not_real_password",
            is_active=True,
            role=Role.ADMIN,
            full_name="Demo Admin"
        )
    # If demo mode is not enabled, use the normal authentication
    return current_user


async def get_admin_user_demo_mode(current_user=Depends(get_current_active_user_demo_mode)):
    """
    Demo mode override for the get_admin_user dependency.
    In demo mode, this will return a mock admin user without requiring authentication.
    
    Args:
        current_user: The user from get_current_active_user_demo_mode
        
    Returns:
        The same user if they're an admin, or raises an exception
    """
    if DEMO_MODE_ENABLED:
        # In demo mode, the current_user is already an admin
        return current_user
    
    # Normal admin check
    if current_user.role != "admin":
        logger.warning(f"Non-admin user {current_user.email} attempted to access admin endpoint")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required"
        )
    return current_user


def has_role(role: Role):
    """
    Dependency for role-based access control.

    Args:
        role: The required role

    Returns:
        Callable: A dependency function that checks the user's role
    """
    async def _has_role(current_user: DBUser = Depends(get_current_active_user)) -> DBUser:
        if current_user.role != role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role {role.value} required"
            )
        return current_user

    return _has_role


def has_any_role(roles: List[Role]):
    """
    Dependency for role-based access control with multiple allowed roles.

    Args:
        roles: List of allowed roles

    Returns:
        Callable: A dependency function that checks if the user has any of the allowed roles
    """
    async def _has_any_role(current_user: DBUser = Depends(get_current_active_user)) -> DBUser:
        if current_user.role not in roles:
            role_values = [role.value for role in roles]
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"One of roles {role_values} required"
            )
        return current_user

    return _has_any_role
