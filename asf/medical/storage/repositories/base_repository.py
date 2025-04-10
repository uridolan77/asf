"""
Base repository for the Medical Research Synthesizer.

This module provides a base repository class for database operations.
"""

from typing import Generic, TypeVar, Type, List, Optional, Any, Dict, Union
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update, delete
from sqlalchemy.orm import Session

from asf.medical.storage.database import Base, is_async

# Type variable for ORM models
T = TypeVar('T', bound=Base)

class BaseRepository(Generic[T]):
    """
    Base repository for database operations.
    
    This class provides common CRUD operations for database models.
    """
    
    def __init__(self, model: Type[T]):
        """
        Initialize the repository with a model class.
        
        Args:
            model: The SQLAlchemy model class
        """
        self.model = model
    
    # Synchronous methods
    def create(self, db: Session, obj_in: Dict[str, Any]) -> T:
        """
        Create a new record.
        
        Args:
            db: Database session
            obj_in: Dictionary with model attributes
            
        Returns:
            The created model instance
        """
        if is_async:
            raise ValueError("Cannot use synchronous methods with async database")
        
        db_obj = self.model(**obj_in)
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj
    
    def get(self, db: Session, id: Any) -> Optional[T]:
        """
        Get a record by ID.
        
        Args:
            db: Database session
            id: Record ID
            
        Returns:
            The model instance or None if not found
        """
        if is_async:
            raise ValueError("Cannot use synchronous methods with async database")
        
        return db.query(self.model).filter(self.model.id == id).first()
    
    def get_by(self, db: Session, **kwargs) -> Optional[T]:
        """
        Get a record by arbitrary attributes.
        
        Args:
            db: Database session
            **kwargs: Attribute-value pairs to filter by
            
        Returns:
            The model instance or None if not found
        """
        if is_async:
            raise ValueError("Cannot use synchronous methods with async database")
        
        query = db.query(self.model)
        for attr, value in kwargs.items():
            query = query.filter(getattr(self.model, attr) == value)
        return query.first()
    
    def get_all(self, db: Session, skip: int = 0, limit: int = 100) -> List[T]:
        """
        Get all records with pagination.
        
        Args:
            db: Database session
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of model instances
        """
        if is_async:
            raise ValueError("Cannot use synchronous methods with async database")
        
        return db.query(self.model).offset(skip).limit(limit).all()
    
    def update(self, db: Session, id: Any, obj_in: Dict[str, Any]) -> Optional[T]:
        """
        Update a record.
        
        Args:
            db: Database session
            id: Record ID
            obj_in: Dictionary with model attributes to update
            
        Returns:
            The updated model instance or None if not found
        """
        if is_async:
            raise ValueError("Cannot use synchronous methods with async database")
        
        db_obj = self.get(db, id)
        if db_obj:
            for key, value in obj_in.items():
                setattr(db_obj, key, value)
            db.commit()
            db.refresh(db_obj)
        return db_obj
    
    def delete(self, db: Session, id: Any) -> bool:
        """
        Delete a record.
        
        Args:
            db: Database session
            id: Record ID
            
        Returns:
            True if the record was deleted, False otherwise
        """
        if is_async:
            raise ValueError("Cannot use synchronous methods with async database")
        
        db_obj = self.get(db, id)
        if db_obj:
            db.delete(db_obj)
            db.commit()
            return True
        return False
    
    # Asynchronous methods
    async def create_async(self, db: AsyncSession, obj_in: Dict[str, Any]) -> T:
        """
        Create a new record asynchronously.
        
        Args:
            db: Async database session
            obj_in: Dictionary with model attributes
            
        Returns:
            The created model instance
        """
        db_obj = self.model(**obj_in)
        db.add(db_obj)
        await db.commit()
        await db.refresh(db_obj)
        return db_obj
    
    async def get_async(self, db: AsyncSession, id: Any) -> Optional[T]:
        """
        Get a record by ID asynchronously.
        
        Args:
            db: Async database session
            id: Record ID
            
        Returns:
            The model instance or None if not found
        """
        result = await db.execute(select(self.model).filter(self.model.id == id))
        return result.scalars().first()
    
    async def get_by_async(self, db: AsyncSession, **kwargs) -> Optional[T]:
        """
        Get a record by arbitrary attributes asynchronously.
        
        Args:
            db: Async database session
            **kwargs: Attribute-value pairs to filter by
            
        Returns:
            The model instance or None if not found
        """
        query = select(self.model)
        for attr, value in kwargs.items():
            query = query.filter(getattr(self.model, attr) == value)
        result = await db.execute(query)
        return result.scalars().first()
    
    async def get_all_async(self, db: AsyncSession, skip: int = 0, limit: int = 100) -> List[T]:
        """
        Get all records with pagination asynchronously.
        
        Args:
            db: Async database session
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of model instances
        """
        result = await db.execute(select(self.model).offset(skip).limit(limit))
        return result.scalars().all()
    
    async def update_async(self, db: AsyncSession, id: Any, obj_in: Dict[str, Any]) -> Optional[T]:
        """
        Update a record asynchronously.
        
        Args:
            db: Async database session
            id: Record ID
            obj_in: Dictionary with model attributes to update
            
        Returns:
            The updated model instance or None if not found
        """
        db_obj = await self.get_async(db, id)
        if db_obj:
            for key, value in obj_in.items():
                setattr(db_obj, key, value)
            await db.commit()
            await db.refresh(db_obj)
            return db_obj
        return None
    
    async def delete_async(self, db: AsyncSession, id: Any) -> bool:
        """
        Delete a record asynchronously.
        
        Args:
            db: Async database session
            id: Record ID
            
        Returns:
            True if the record was deleted, False otherwise
        """
        db_obj = await self.get_async(db, id)
        if db_obj:
            await db.delete(db_obj)
            await db.commit()
            return True
        return False
