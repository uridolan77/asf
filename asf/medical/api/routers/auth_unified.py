"""
Authentication router for the Medical Research Synthesizer API.

This module provides endpoints for user authentication and management.
"""

import logging
from datetime import timedelta
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from asf.medical.api.auth_unified import (
    Token, User, UserCreate, UserUpdate, AuthService, 
    get_auth_service, get_current_active_user, get_admin_user
)
from asf.medical.core.config import settings
from asf.medical.core.security import create_access_token
from asf.medical.storage.database import get_db_session
from asf.medical.storage.models import User as DBUser

# Initialize router
router = APIRouter(prefix="/v1/auth", tags=["auth"])

# Set up logging
logger = logging.getLogger(__name__)

@router.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db_session),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Get an access token.
    
    This endpoint authenticates a user and returns a JWT access token.
    """
    user = await auth_service.authenticate_user(db, form_data.username, form_data.password)
    
    if not user:
        logger.warning(f"Failed login attempt for user: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Update last login time
    # await auth_service.update_user(db, user.id, {"last_login": datetime.utcnow()})
    
    # Create access token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        subject=user.email,
        expires_delta=access_token_expires
    )
    
    logger.info(f"User logged in: {user.email}")
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "role": user.role,
        "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    }

@router.post("/register", response_model=User, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db_session),
    auth_service: AuthService = Depends(get_auth_service),
    admin_user: DBUser = Depends(get_admin_user)
):
    """
    Register a new user.
    
    This endpoint registers a new user. Only admin users can register new users.
    """
    user = await auth_service.register_user(
        db=db,
        email=user_data.email,
        password=user_data.password,
        role=user_data.role
    )
    
    if not user:
        logger.warning(f"Failed to register user: {user_data.email} (email already exists)")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    logger.info(f"User registered: {user.email} (role: {user.role})")
    
    return user

@router.get("/me", response_model=User)
async def get_current_user_info(
    current_user: DBUser = Depends(get_current_active_user)
):
    """
    Get current user information.
    
    This endpoint returns information about the current authenticated user.
    """
    return current_user

@router.put("/me", response_model=User)
async def update_current_user_info(
    user_data: UserUpdate,
    db: AsyncSession = Depends(get_db_session),
    auth_service: AuthService = Depends(get_auth_service),
    current_user: DBUser = Depends(get_current_active_user)
):
    """
    Update current user information.
    
    This endpoint updates information for the current authenticated user.
    """
    # Convert Pydantic model to dict
    update_data = user_data.dict(exclude_unset=True)
    
    # Update user
    updated_user = await auth_service.update_user(
        db=db,
        user_id=current_user.id,
        update_data=update_data
    )
    
    if not updated_user:
        logger.error(f"Failed to update user: {current_user.email}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user"
        )
    
    logger.info(f"User updated: {updated_user.email}")
    
    return updated_user

@router.get("/users", response_model=List[User])
async def get_users(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db_session),
    auth_service: AuthService = Depends(get_auth_service),
    admin_user: DBUser = Depends(get_admin_user)
):
    """
    Get all users.
    
    This endpoint returns a list of all users. Only admin users can access this endpoint.
    """
    users = await auth_service.get_users(db, skip, limit)
    return users

@router.get("/users/{user_id}", response_model=User)
async def get_user(
    user_id: int,
    db: AsyncSession = Depends(get_db_session),
    auth_service: AuthService = Depends(get_auth_service),
    admin_user: DBUser = Depends(get_admin_user)
):
    """
    Get a user by ID.
    
    This endpoint returns a user by ID. Only admin users can access this endpoint.
    """
    user = await auth_service.user_repository.get_by_id_async(db, user_id)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return user

@router.put("/users/{user_id}", response_model=User)
async def update_user(
    user_id: int,
    user_data: UserUpdate,
    db: AsyncSession = Depends(get_db_session),
    auth_service: AuthService = Depends(get_auth_service),
    admin_user: DBUser = Depends(get_admin_user)
):
    """
    Update a user.
    
    This endpoint updates a user by ID. Only admin users can access this endpoint.
    """
    # Convert Pydantic model to dict
    update_data = user_data.dict(exclude_unset=True)
    
    # Update user
    updated_user = await auth_service.update_user(
        db=db,
        user_id=user_id,
        update_data=update_data
    )
    
    if not updated_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    logger.info(f"User updated by admin: {updated_user.email}")
    
    return updated_user

@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: int,
    db: AsyncSession = Depends(get_db_session),
    auth_service: AuthService = Depends(get_auth_service),
    admin_user: DBUser = Depends(get_admin_user)
):
    """
    Delete a user.
    
    This endpoint deletes a user by ID. Only admin users can access this endpoint.
    """
    success = await auth_service.delete_user(db, user_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    logger.info(f"User deleted by admin: ID {user_id}")
    
    return None
