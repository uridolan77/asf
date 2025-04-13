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
from asf.medical.storage.models import User as DBUser
from asf.medical.core.observability import log_error
from asf.medical.core.exceptions import DatabaseError

logger = logging.getLogger(__name__)

# OAuth2 password bearer scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/v1/auth/token")

# Password context for hashing and verifying passwords
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


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
    role: Optional[str] = "user"


class UserUpdate(BaseModel):
    """
    User update request model.
    """
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    password: Optional[str] = Field(None, min_length=8)
    role: Optional[str] = None
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


def has_role(role: str):
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
                detail=f"Role {role} required"
            )
        return current_user

    return _has_role


def has_any_role(roles: List[str]):
    """
    Dependency for role-based access control with multiple allowed roles.
    
    Args:
        roles: List of allowed roles
        
    Returns:
        Callable: A dependency function that checks if the user has any of the allowed roles
    """
    async def _has_any_role(current_user: DBUser = Depends(get_current_active_user)) -> DBUser:
        if current_user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"One of roles {roles} required"
            )
        return current_user

    return _has_any_role
