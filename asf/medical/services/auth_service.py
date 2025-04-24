"""
Authentication service for the Medical Research Synthesizer.

This module provides a service for user authentication and management.
It handles user registration, authentication, and token validation.
"""
from typing import Optional, Dict, List, Any
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from jose import JWTError, jwt
from fastapi import HTTPException, status
from ..storage.repositories.user_repository import UserRepository
from ..storage.models import User as DBUser
from ..core.security import verify_password, get_password_hash
from ..core.config import settings
from ..core.logging_config import get_logger
logger = get_logger(__name__)
class AuthService:
    """
    Service for user authentication and management.
    
    This service provides methods for user registration, authentication,
    and token validation. It uses JWT tokens for authentication.
    """
    def __init__(self, user_repository: UserRepository):
        """
        Initialize the authentication service.
        
        Args:
            user_repository: Repository for user operations
        """
        self.user_repository = user_repository
    async def authenticate_user(
        self, db: AsyncSession, email: str, password: str
    ) -> Optional[DBUser]:
        """
        Authenticate a user with email and password.
        Args:
            db: Database session
            email: User's email
            password: User's password
        Returns:
            User object if authentication is successful, None otherwise
        """
        user = await self.user_repository.get_by_email(db, email)
        if not user:
            return None
        if not verify_password(password, user.hashed_password):
            return None
        if not user.is_active:
            return None
        return user
    async def register_user(
        self, db: AsyncSession, email: str, password: str, role: str = "user"
    ) -> Optional[DBUser]:
        """
        Register a new user.
        Args:
            db: Database session
            email: User's email
            password: User's password
            role: User's role (default: "user")
        Returns:
            Created user object if registration is successful, None otherwise
        """
        # Check if user already exists
        existing_user = await self.user_repository.get_by_email(db, email)
        if existing_user:
            return None
        # Create new user
        hashed_password = get_password_hash(password)
        user_data = {
            "email": email,
            "hashed_password": hashed_password,
            "is_active": True,
            "role": role,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
        return await self.user_repository.create(db, user_data)
    async def get_user_by_email(self, db: AsyncSession, email: str) -> Optional[DBUser]:
        """
        Get a user by email.
        Args:
            db: Database session
            email: User's email
        Returns:
            User object if found, None otherwise
        """
        return await self.user_repository.get_by_email(db, email)
    async def get_user_by_id(self, db: AsyncSession, user_id: int) -> Optional[DBUser]:
        """
        Get a user by ID.
        Args:
            db: Database session
            user_id: User's ID
        Returns:
            User object if found, None otherwise
        """
        return await self.user_repository.get_by_id(db, user_id)
    async def update_user(
        self, db: AsyncSession, user_id: int, update_data: Dict[str, Any]
    ) -> Optional[DBUser]:
        """
        Update a user.
        Args:
            db: Database session
            user_id: User's ID
            update_data: Dictionary with fields to update
        Returns:
            Updated user object if successful, None otherwise
        """
        # Check if user exists
        user = await self.user_repository.get_by_id(db, user_id)
        if not user:
            return None
        # Handle password update
        if "password" in update_data:
            update_data["hashed_password"] = get_password_hash(update_data.pop("password"))
        # Update timestamp
        update_data["updated_at"] = datetime.utcnow()
        return await self.user_repository.update(db, user_id, update_data)
    async def delete_user(self, db: AsyncSession, user_id: int) -> bool:
        """
        Delete a user.
        Args:
            db: Database session
            user_id: User's ID
        Returns:
            True if deletion was successful, False otherwise
        """
        return await self.user_repository.delete(db, user_id)
    async def get_users(
        self, db: AsyncSession, skip: int = 0, limit: int = 100
    ) -> List[DBUser]:
        """
        Get a list of users with pagination.
        Args:
            db: Database session
            skip: Number of records to skip
            limit: Maximum number of records to return
        Returns:
            List of user objects
        """
        return await self.user_repository.get_all(db, skip=skip, limit=limit)
    async def get_current_user(self, db: AsyncSession, token: str) -> DBUser:
        """
        Get the current user from a JWT token.
        Args:
            db: Database session
            token: JWT token
        Returns:
            User object
        Raises:
            HTTPException: If token is invalid or user is not found
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
            token_type: str = payload.get("type", "access")
            if email is None:
                raise credentials_exception
            if token_type == "refresh":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Cannot use refresh token for authentication",
                    headers={"WWW-Authenticate": "Bearer"},
                )
        except JWTError as e:
            logger.error(f"JWT error: {str(e)}", extra={"token": token[:10] + "..."})
            raise credentials_exception
        user = await self.user_repository.get_by_email(db, email)
        if user is None:
            raise credentials_exception
        return user