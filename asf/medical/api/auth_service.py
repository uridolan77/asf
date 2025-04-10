"""
Authentication service for the Medical Research Synthesizer API.

This module provides a service for user authentication and management.
"""

from datetime import datetime, timedelta
from typing import Dict, Optional, Union, Any

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from asf.medical.api.models.auth import TokenData, User, UserInDB
from asf.medical.core.config import settings
from asf.medical.core.security import verify_password, get_password_hash, create_access_token
from asf.medical.storage.database import get_db, get_db_session
from asf.medical.storage.repositories.user_repository import UserRepository
from asf.medical.storage.models import User as DBUser

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class AuthService:
    """
    Service for user authentication and management.
    """
    
    def __init__(self, user_repository: UserRepository):
        """
        Initialize the authentication service.
        
        Args:
            user_repository: User repository
        """
        self.user_repository = user_repository
    
    async def authenticate_user(
        self, db: AsyncSession, email: str, password: str
    ) -> Optional[DBUser]:
        """
        Authenticate a user.
        
        Args:
            db: Async database session
            email: User email
            password: User password
            
        Returns:
            User if authentication is successful, None otherwise
        """
        user = await self.user_repository.get_by_email_async(db, email)
        
        if not user:
            return None
        
        if not verify_password(password, user.hashed_password):
            return None
        
        if not user.is_active:
            return None
        
        return user
    
    async def get_current_user(
        self, db: AsyncSession, token: str = Depends(oauth2_scheme)
    ) -> DBUser:
        """
        Get the current user from a JWT token.
        
        Args:
            db: Async database session
            token: JWT token
            
        Returns:
            Current user
            
        Raises:
            HTTPException: If the token is invalid or the user is not found
        """
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
        try:
            # Decode the token
            payload = jwt.decode(
                token, 
                settings.SECRET_KEY.get_secret_value(), 
                algorithms=["HS256"]
            )
            
            # Extract the email from the token
            email: str = payload.get("sub")
            
            if email is None:
                raise credentials_exception
            
            token_data = TokenData(email=email)
        except JWTError:
            raise credentials_exception
        
        # Get the user from the database
        user = await self.user_repository.get_by_email_async(db, token_data.email)
        
        if user is None:
            raise credentials_exception
        
        return user
    
    async def get_current_active_user(
        self, current_user: DBUser = Depends(get_current_user)
    ) -> DBUser:
        """
        Get the current active user.
        
        Args:
            current_user: Current user
            
        Returns:
            Current active user
            
        Raises:
            HTTPException: If the user is inactive
        """
        if not current_user.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Inactive user"
            )
        
        return current_user
    
    async def register_user(
        self, db: AsyncSession, email: str, password: str, role: str = "user"
    ) -> Optional[DBUser]:
        """
        Register a new user.
        
        Args:
            db: Async database session
            email: User email
            password: User password
            role: User role (default: "user")
            
        Returns:
            User if registration is successful, None if the email is already registered
        """
        # Check if user exists
        existing_user = await self.user_repository.get_by_email_async(db, email)
        
        if existing_user:
            return None
        
        # Create user
        hashed_password = get_password_hash(password)
        return await self.user_repository.create_user_async(db, email, hashed_password, role)

# Create a dependency for the auth service
async def get_auth_service(
    db: AsyncSession = Depends(get_db_session),
) -> AuthService:
    """
    Get the authentication service.
    
    Args:
        db: Async database session
        
    Returns:
        Authentication service
    """
    user_repository = UserRepository()
    return AuthService(user_repository)

# Convenience dependencies
async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db_session),
    auth_service: AuthService = Depends(get_auth_service),
) -> DBUser:
    """
    Get the current user from a JWT token.
    
    Args:
        token: JWT token
        db: Async database session
        auth_service: Authentication service
        
    Returns:
        Current user
    """
    return await auth_service.get_current_user(db, token)

async def get_current_active_user(
    current_user: DBUser = Depends(get_current_user),
) -> DBUser:
    """
    Get the current active user.
    
    Args:
        current_user: Current user
        
    Returns:
        Current active user
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    return current_user

async def get_admin_user(
    current_user: DBUser = Depends(get_current_active_user),
) -> DBUser:
    """
    Get the current admin user.
    
    Args:
        current_user: Current active user
        
    Returns:
        Current admin user
        
    Raises:
        HTTPException: If the user is not an admin
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    return current_user
