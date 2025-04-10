"""
Unified authentication module for the Medical Research Synthesizer API.

This module provides a comprehensive JWT-based authentication system for the FastAPI implementation,
including user management, token generation, validation, and role-based access control.
"""

from datetime import datetime, timedelta
from typing import Dict, Optional, List, Union, Any, Callable

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr, Field

from asf.medical.core.config import settings
from asf.medical.core.security import verify_password, get_password_hash, create_access_token
from asf.medical.storage.database import get_db_session
from asf.medical.storage.repositories.user_repository import UserRepository
from asf.medical.storage.models import User as DBUser
from asf.medical.core.monitoring import log_error

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/v1/auth/token")

# Models
class Token(BaseModel):
    """Token model for authentication responses."""
    access_token: str
    token_type: str
    role: str
    expires_in: int

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

class UserCreate(BaseModel):
    """User creation model for API requests."""
    email: EmailStr = Field(..., description="User email")
    password: str = Field(..., description="User password", min_length=8)
    role: str = Field("user", description="User role (admin or user)")

class UserUpdate(BaseModel):
    """User update model for API requests."""
    email: Optional[EmailStr] = Field(None, description="User email")
    password: Optional[str] = Field(None, description="User password", min_length=8)
    is_active: Optional[bool] = Field(None, description="User active status")

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
        self, db: AsyncSession, token: str
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
        except JWTError as e:
            log_error(e, {"token": token[:10] + "..."})
            raise credentials_exception
        
        # Get the user from the database
        user = await self.user_repository.get_by_email_async(db, token_data.email)
        
        if user is None:
            raise credentials_exception
        
        return user
    
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
    
    async def update_user(
        self, db: AsyncSession, user_id: int, update_data: Dict[str, Any]
    ) -> Optional[DBUser]:
        """
        Update a user.
        
        Args:
            db: Async database session
            user_id: User ID
            update_data: User data to update
            
        Returns:
            Updated user if successful, None if the user is not found
        """
        # Check if user exists
        user = await self.user_repository.get_by_id_async(db, user_id)
        
        if not user:
            return None
        
        # Update password if provided
        if "password" in update_data and update_data["password"]:
            update_data["hashed_password"] = get_password_hash(update_data.pop("password"))
        
        # Update user
        return await self.user_repository.update_user_async(db, user_id, update_data)
    
    async def delete_user(
        self, db: AsyncSession, user_id: int
    ) -> bool:
        """
        Delete a user.
        
        Args:
            db: Async database session
            user_id: User ID
            
        Returns:
            True if successful, False if the user is not found
        """
        return await self.user_repository.delete_user_async(db, user_id)
    
    async def get_users(
        self, db: AsyncSession, skip: int = 0, limit: int = 100
    ) -> List[DBUser]:
        """
        Get all users.
        
        Args:
            db: Async database session
            skip: Number of users to skip
            limit: Maximum number of users to return
            
        Returns:
            List of users
        """
        return await self.user_repository.get_users_async(db, skip, limit)

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

def has_role(role: str) -> Callable:
    """
    Create a dependency that checks if the current user has a specific role.
    
    Args:
        role: Required role
        
    Returns:
        Dependency function
    """
    async def _has_role(current_user: DBUser = Depends(get_current_active_user)) -> DBUser:
        if current_user.role != role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{role}' required"
            )
        return current_user
    
    return _has_role

def has_any_role(roles: List[str]) -> Callable:
    """
    Create a dependency that checks if the current user has any of the specified roles.
    
    Args:
        roles: List of acceptable roles
        
    Returns:
        Dependency function
    """
    async def _has_any_role(current_user: DBUser = Depends(get_current_active_user)) -> DBUser:
        if current_user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"One of roles {roles} required"
            )
        return current_user
    
    return _has_any_role
