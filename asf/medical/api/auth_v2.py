"""
Authentication module for the Medical Research Synthesizer API.

This module provides JWT-based authentication for the FastAPI implementation,
including user management, token generation, and validation.
"""

from datetime import datetime, timedelta
from typing import Dict, Optional, List, Union, Any

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from pydantic import BaseModel

from asf.medical.core.config import settings
from asf.medical.core.security import verify_password, get_password_hash, create_access_token
from asf.medical.storage.database import get_db, get_db_session
from asf.medical.storage.repositories import UserRepository
from asf.medical.storage.models import User as DBUser

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Models
class Token(BaseModel):
    """Token model for authentication responses."""
    access_token: str
    token_type: str
    role: str

class TokenData(BaseModel):
    """Token data model for JWT payload."""
    email: Optional[str] = None

class User(BaseModel):
    """User model for API responses."""
    id: int
    email: str
    role: str
    is_active: bool
    
    class Config:
        """Pydantic config."""
        orm_mode = True

# Repository instance
user_repository = UserRepository()

# Helper functions
def get_user(db: Union[Session, AsyncSession], email: str) -> Optional[DBUser]:
    """
    Get a user from the database.
    
    Args:
        db: Database session
        email: User email
        
    Returns:
        User or None if not found
    """
    if isinstance(db, AsyncSession):
        # This is an async session, we need to use the async method
        # But we can't use await here, so this function should only be used
        # with sync sessions or in an async context where we can await
        raise ValueError("Cannot use get_user with AsyncSession, use get_user_async instead")
    
    return user_repository.get_by_email(db, email)

async def get_user_async(db: AsyncSession, email: str) -> Optional[DBUser]:
    """
    Get a user from the database asynchronously.
    
    Args:
        db: Async database session
        email: User email
        
    Returns:
        User or None if not found
    """
    return await user_repository.get_by_email_async(db, email)

def authenticate_user(db: Session, email: str, password: str) -> Optional[DBUser]:
    """
    Authenticate a user.
    
    Args:
        db: Database session
        email: User email
        password: User password
        
    Returns:
        User if authentication is successful, None otherwise
    """
    user = get_user(db, email)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    if not user.is_active:
        return None
    return user

async def authenticate_user_async(db: AsyncSession, email: str, password: str) -> Optional[DBUser]:
    """
    Authenticate a user asynchronously.
    
    Args:
        db: Async database session
        email: User email
        password: User password
        
    Returns:
        User if authentication is successful, None otherwise
    """
    user = await get_user_async(db, email)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    if not user.is_active:
        return None
    return user

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db_session)
) -> DBUser:
    """
    Get the current user from a JWT token.
    
    Args:
        token: JWT token
        db: Database session
        
    Returns:
        User
        
    Raises:
        HTTPException: If the token is invalid or the user is not found
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token, 
            settings.SECRET_KEY.get_secret_value(), 
            algorithms=["HS256"]
        )
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except JWTError:
        raise credentials_exception
    
    user = await get_user_async(db, email=token_data.email)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(
    current_user: DBUser = Depends(get_current_user)
) -> DBUser:
    """
    Get the current active user.
    
    Args:
        current_user: Current user
        
    Returns:
        User
        
    Raises:
        HTTPException: If the user is inactive
    """
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

def register_user(
    db: Session, email: str, password: str, role: str = "user"
) -> Optional[DBUser]:
    """
    Register a new user.
    
    Args:
        db: Database session
        email: User email
        password: User password
        role: User role (default: "user")
        
    Returns:
        User if registration is successful, None if the email is already registered
    """
    # Check if user exists
    existing_user = get_user(db, email)
    if existing_user:
        return None
    
    # Create user
    hashed_password = get_password_hash(password)
    return user_repository.create_user(db, email, hashed_password, role)

async def register_user_async(
    db: AsyncSession, email: str, password: str, role: str = "user"
) -> Optional[DBUser]:
    """
    Register a new user asynchronously.
    
    Args:
        db: Async database session
        email: User email
        password: User password
        role: User role (default: "user")
        
    Returns:
        User if registration is successful, None if the email is already registered
    """
    # Check if user exists
    existing_user = await get_user_async(db, email)
    if existing_user:
        return None
    
    # Create user
    hashed_password = get_password_hash(password)
    return await user_repository.create_user_async(db, email, hashed_password, role)
