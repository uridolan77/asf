"""Authentication router for the Medical Research Synthesizer API.

This module provides endpoints for user authentication and management.
"""

import logging
from datetime import timedelta
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from jose import JWTError, jwt
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth import (
    Token, User, UserCreate, UserUpdate, AuthService,
    get_auth_service, get_current_active_user, get_admin_user
)
from ...core.config import settings
from ...core.security import create_access_token, create_refresh_token
from ...storage.database import get_db_session
from ...storage.models import MedicalUser as DBUser

router = APIRouter(prefix="/v1/auth", tags=["auth"])

logger = logging.getLogger(__name__)

@router.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db_session),
    auth_service: AuthService = Depends(get_auth_service)
):
    """Authenticate user and generate access and refresh tokens.
    
    Args:
        form_data: OAuth2 password request form containing username and password
        db: Database session
        auth_service: Authentication service instance
        
    Returns:
        Token object containing access token, refresh token, token type, user role and expiration
        
    Raises:
        HTTPException: If authentication fails
    """
    user = await auth_service.authenticate_user(db, form_data.username, form_data.password)

    if not user:
        logger.warning(f"Failed login attempt for user: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )


    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        subject=user.email,
        expires_delta=access_token_expires
    )

    refresh_token_expires = timedelta(days=7)
    refresh_token = create_refresh_token(
        subject=user.email,
        expires_delta=refresh_token_expires
    )

    logger.info(f"User logged in: {user.email}")

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "role": user.role,
        "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    }

@router.post("/refresh", response_model=Token)
async def refresh_access_token(
    refresh_token: str,
    db: AsyncSession = Depends(get_db_session),
    auth_service: AuthService = Depends(get_auth_service)
):
    """Generate new access and refresh tokens using a valid refresh token.
    
    Args:
        refresh_token: The refresh token to validate
        db: Database session
        auth_service: Authentication service instance
        
    Returns:
        Token object containing new access token, refresh token, token type, user role and expiration
        
    Raises:
        HTTPException: If refresh token is invalid or expired
    """
    try:
        payload = jwt.decode(
            refresh_token,
            settings.SECRET_KEY.get_secret_value(),
            algorithms=["HS256"]
        )

        email: str = payload.get("sub")
        token_type: str = payload.get("type")

        if token_type != "refresh":
            logger.warning(f"Invalid token type for refresh: {token_type}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type",
                headers={"WWW-Authenticate": "Bearer"},
            )

        user = await auth_service.get_user_by_email(db, email)

        if not user or not user.is_active:
            logger.warning(f"User not found or inactive: {email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            subject=user.email,
            expires_delta=access_token_expires
        )

        refresh_token_expires = timedelta(days=7)
        new_refresh_token = create_refresh_token(
            subject=user.email,
            expires_delta=refresh_token_expires
        )

        logger.info(f"Tokens refreshed for user: {user.email}")

        return {
            "access_token": access_token,
            "refresh_token": new_refresh_token,
            "token_type": "bearer",
            "role": user.role,
            "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        }
    except JWTError as e:
        logger.warning(f"JWT error during token refresh: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )

@router.post("/register", response_model=User, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db_session),
    auth_service: AuthService = Depends(get_auth_service),
    admin_user: DBUser = Depends(get_admin_user)
):
    """Register a new user (admin only).
    
    Args:
        user_data: User creation data containing email, password and role
        db: Database session
        auth_service: Authentication service instance
        admin_user: Current authenticated admin user
        
    Returns:
        Newly created user object
        
    Raises:
        HTTPException: If user with the same email already exists
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
    """Get information about the current authenticated user.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User object with current user information
    """
    return current_user

@router.put("/me", response_model=User)
async def update_current_user_info(
    user_data: UserUpdate,
    db: AsyncSession = Depends(get_db_session),
    auth_service: AuthService = Depends(get_auth_service),
    current_user: DBUser = Depends(get_current_active_user)
):
    """Update the current authenticated user's information.
    
    Args:
        user_data: User update data
        db: Database session
        auth_service: Authentication service instance
        current_user: Current authenticated user
        
    Returns:
        Updated user object
        
    Raises:
        HTTPException: If update fails
    """
    update_data = user_data.dict(exclude_unset=True)

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
    """Get a list of users (admin only).
    
    Args:
        skip: Number of users to skip (pagination)
        limit: Maximum number of users to return (pagination)
        db: Database session
        auth_service: Authentication service instance
        admin_user: Current authenticated admin user
        
    Returns:
        List of user objects
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
    """Get a specific user by ID (admin only).
    
    Args:
        user_id: ID of the user to retrieve
        db: Database session
        auth_service: Authentication service instance
        admin_user: Current authenticated admin user
        
    Returns:
        User object
        
    Raises:
        HTTPException: If user not found
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
    """Update a specific user's information (admin only).
    
    Args:
        user_id: ID of the user to update
        user_data: User update data
        db: Database session
        auth_service: Authentication service instance
        admin_user: Current authenticated admin user
        
    Returns:
        Updated user object
        
    Raises:
        HTTPException: If user not found
    """
    update_data = user_data.dict(exclude_unset=True)

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
    """Delete a specific user (admin only).
    
    Args:
        user_id: ID of the user to delete
        db: Database session
        auth_service: Authentication service instance
        admin_user: Current authenticated admin user
        
    Returns:
        None
        
    Raises:
        HTTPException: If user not found
    """
    success = await auth_service.delete_user(db, user_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    logger.info(f"User deleted by admin: ID {user_id}")

    return None
