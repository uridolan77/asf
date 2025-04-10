"""
User repository for the Medical Research Synthesizer.

This module provides a repository for user-related database operations.
"""

from typing import Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import Session
import datetime

from asf.medical.storage.models import User
from asf.medical.storage.repositories.base_repository import BaseRepository
from asf.medical.storage.database import is_async

class UserRepository(BaseRepository[User]):
    """
    Repository for user-related database operations.
    """
    
    def __init__(self):
        """Initialize the repository with the User model."""
        super().__init__(User)
    
    # Synchronous methods
    def get_by_email(self, db: Session, email: str) -> Optional[User]:
        """
        Get a user by email.
        
        Args:
            db: Database session
            email: User email
            
        Returns:
            The user or None if not found
        """
        if is_async:
            raise ValueError("Cannot use synchronous methods with async database")
        
        return db.query(User).filter(User.email == email).first()
    
    def create_user(self, db: Session, email: str, hashed_password: str, role: str = "user") -> User:
        """
        Create a new user.
        
        Args:
            db: Database session
            email: User email
            hashed_password: Hashed password
            role: User role (default: "user")
            
        Returns:
            The created user
        """
        if is_async:
            raise ValueError("Cannot use synchronous methods with async database")
        
        user = User(
            email=email,
            hashed_password=hashed_password,
            role=role,
            created_at=datetime.datetime.utcnow(),
            is_active=True
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    
    def update_last_login(self, db: Session, user_id: int) -> Optional[User]:
        """
        Update a user's last login timestamp.
        
        Args:
            db: Database session
            user_id: User ID
            
        Returns:
            The updated user or None if not found
        """
        if is_async:
            raise ValueError("Cannot use synchronous methods with async database")
        
        user = self.get(db, user_id)
        if user:
            user.last_login = datetime.datetime.utcnow()
            db.commit()
            db.refresh(user)
        return user
    
    def deactivate_user(self, db: Session, user_id: int) -> Optional[User]:
        """
        Deactivate a user.
        
        Args:
            db: Database session
            user_id: User ID
            
        Returns:
            The updated user or None if not found
        """
        if is_async:
            raise ValueError("Cannot use synchronous methods with async database")
        
        user = self.get(db, user_id)
        if user:
            user.is_active = False
            db.commit()
            db.refresh(user)
        return user
    
    # Asynchronous methods
    async def get_by_email_async(self, db: AsyncSession, email: str) -> Optional[User]:
        """
        Get a user by email asynchronously.
        
        Args:
            db: Async database session
            email: User email
            
        Returns:
            The user or None if not found
        """
        result = await db.execute(select(User).filter(User.email == email))
        return result.scalars().first()
    
    async def create_user_async(self, db: AsyncSession, email: str, hashed_password: str, role: str = "user") -> User:
        """
        Create a new user asynchronously.
        
        Args:
            db: Async database session
            email: User email
            hashed_password: Hashed password
            role: User role (default: "user")
            
        Returns:
            The created user
        """
        user = User(
            email=email,
            hashed_password=hashed_password,
            role=role,
            created_at=datetime.datetime.utcnow(),
            is_active=True
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
        return user
    
    async def update_last_login_async(self, db: AsyncSession, user_id: int) -> Optional[User]:
        """
        Update a user's last login timestamp asynchronously.
        
        Args:
            db: Async database session
            user_id: User ID
            
        Returns:
            The updated user or None if not found
        """
        user = await self.get_async(db, user_id)
        if user:
            user.last_login = datetime.datetime.utcnow()
            await db.commit()
            await db.refresh(user)
        return user
    
    async def deactivate_user_async(self, db: AsyncSession, user_id: int) -> Optional[User]:
        """
        Deactivate a user asynchronously.
        
        Args:
            db: Async database session
            user_id: User ID
            
        Returns:
            The updated user or None if not found
        """
        user = await self.get_async(db, user_id)
        if user:
            user.is_active = False
            await db.commit()
            await db.refresh(user)
        return user
