"""
Enhanced Base Repository for the Medical Research Synthesizer.

This module provides an enhanced base repository class for database operations,
implementing proper dependency injection and removing global state.
"""

import logging
from typing import Dict, List, Optional, Any, Type, TypeVar, Generic, Union, Tuple
from sqlalchemy import select, insert, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from asf.medical.core.exceptions import DatabaseError, ResourceNotFoundError

logger = logging.getLogger(__name__)

T = TypeVar('T')

class EnhancedBaseRepository(Generic[T]):
    """
    Enhanced base repository for database operations.
    
    This class provides a base repository for database operations,
    implementing proper dependency injection and removing global state.
    """
    
    def __init__(self, model: Type[T], is_async: bool = True):
        """
        Initialize the enhanced base repository.
        
        Args:
            model: SQLAlchemy model class
            is_async: Whether to use async database operations
        """
        self.model = model
        self.is_async = is_async
        
        logger.debug(f"Initialized {self.__class__.__name__} for {model.__name__}")
    
    async def get_by_id(self, db: Union[AsyncSession, Session], id: Any) -> Optional[T]:
        try:
            if self.is_async:
                stmt = select(self.model).where(self.model.id == id)
                result = await db.execute(stmt)
                return result.scalars().first()
            else:
                return db.query(self.model).filter(self.model.id == id).first()
        except SQLAlchemyError as e:
            await await await db.rollback()
            logger.error(f\"Error getting {self.model.__name__} by ID: {str(e)}\")
            raise DatabaseError(f\"Error getting {self.model.__name__} by ID: {str(e)}\") DatabaseError(f"Failed to get {self.model.__name__} by ID: {str(e)}")
    
    async def get_by_id_or_404(self, db: Union[AsyncSession, Session], id: Any) -> T:
        record = await self.get_by_id(db, id)
        
        if record is None:
            raise ResourceNotFoundError(f"{self.model.__name__} with ID {id} not found")
        
        return record
    
    async def get_all(self, db: Union[AsyncSession, Session], skip: int = 0, limit: int = 100) -> List[T]:
        try:
            if self.is_async:
                stmt = select(self.model).offset(skip).limit(limit)
                result = await db.execute(stmt)
                return list(result.scalars().all())
            else:
                return db.query(self.model).offset(skip).limit(limit).all()
        except SQLAlchemyError as e:
            await await await db.rollback()
            logger.error(f\"Error getting all {self.model.__name__}: {str(e)}\")
            raise DatabaseError(f\"Error getting all {self.model.__name__}: {str(e)}\") DatabaseError(f"Failed to get all {self.model.__name__}: {str(e)}")
    
    async def create(self, db: Union[AsyncSession, Session], obj_in: Dict[str, Any]) -> T:
        try:
            if self.is_async:
                stmt = insert(self.model).values(**obj_in).returning(self.model)
                result = await db.execute(stmt)
                await db.commit()
                return result.scalars().first()
            else:
                db_obj = self.model(**obj_in)
                db.add(db_obj)
                await db.commit()
        await db.refresh(db_obj)
                return db_obj
        except SQLAlchemyError as e:
            if self.is_async:
                await await await await db.rollback()
            else:
                await await await db.rollback()
            logger.error(f"Error creating {self.model.__name__}: {str(e)}")
            raise DatabaseError(f"Failed to create {self.model.__name__}: {str(e)}")
    
    async def update(self, db: Union[AsyncSession, Session], id: Any, obj_in: Dict[str, Any]) -> T:
        try:
            db_obj = await self.get_by_id_or_404(db, id)
            
            if self.is_async:
                stmt = update(self.model).where(self.model.id == id).values(**obj_in).returning(self.model)
                result = await db.execute(stmt)
                await db.commit()
                return result.scalars().first()
            else:
                for key, value in obj_in.items():
                    setattr(db_obj, key, value)
                await db.commit()
        await db.refresh(db_obj)
                return db_obj
        except ResourceNotFoundError:
            raise
        except SQLAlchemyError as e:
            if self.is_async:
                await await await await db.rollback()
            else:
                await await await db.rollback()
            logger.error(f"Error updating {self.model.__name__}: {str(e)}")
            raise DatabaseError(f"Failed to update {self.model.__name__}: {str(e)}")
    
    async def delete(self, db: Union[AsyncSession, Session], id: Any) -> T:
        try:
            db_obj = await self.get_by_id_or_404(db, id)
            
            if self.is_async:
                stmt = delete(self.model).where(self.model.id == id).returning(self.model)
                result = await db.execute(stmt)
                await db.commit()
                return result.scalars().first()
            else:
                db.delete(db_obj)
                db.commit()
                return db_obj
        except ResourceNotFoundError:
            raise
        except SQLAlchemyError as e:
            if self.is_async:
                await await await await db.rollback()
            else:
                await await await db.rollback()
            logger.error(f"Error deleting {self.model.__name__}: {str(e)}")
            raise DatabaseError(f"Failed to delete {self.model.__name__}: {str(e)}")
    
    async def count(self, db: Union[AsyncSession, Session]) -> int:
        try:
            if self.is_async:
                stmt = select(self.model.id)
                result = await db.execute(stmt)
                return len(result.all())
            else:
                return db.query(self.model).count()
        except SQLAlchemyError as e:
            await await await db.rollback()
            logger.error(f\"Error counting {self.model.__name__}: {str(e)}\")
            raise DatabaseError(f\"Error counting {self.model.__name__}: {str(e)}\") DatabaseError(f"Failed to count {self.model.__name__}: {str(e)}")
    
    async def exists(self, db: Union[AsyncSession, Session], id: Any) -> bool:
        try:
            if self.is_async:
                stmt = select(self.model.id).where(self.model.id == id)
                result = await db.execute(stmt)
                return result.first() is not None
            else:
                return db.query(self.model.id).filter(self.model.id == id).first() is not None
        except SQLAlchemyError as e:
            await await await db.rollback()
            logger.error(f\"Error checking if {self.model.__name__} exists: {str(e)}\")
            raise DatabaseError(f\"Error checking if {self.model.__name__} exists: {str(e)}\") DatabaseError(f"Failed to check if {self.model.__name__} exists: {str(e)}")
    
    async def get_by_field(self, db: Union[AsyncSession, Session], field: str, value: Any) -> Optional[T]:
        try:
            if self.is_async:
                stmt = select(self.model).where(getattr(self.model, field) == value)
                result = await db.execute(stmt)
                return result.scalars().first()
            else:
                return db.query(self.model).filter(getattr(self.model, field) == value).first()
        except SQLAlchemyError as e:
            await await await db.rollback()
            logger.error(f\"Error getting {self.model.__name__} by field: {str(e)}\")
            raise DatabaseError(f\"Error getting {self.model.__name__} by field: {str(e)}\") DatabaseError(f"Failed to get {self.model.__name__} by field: {str(e)}")
    
    async def get_by_field_or_404(self, db: Union[AsyncSession, Session], field: str, value: Any) -> T:
        record = await self.get_by_field(db, field, value)
        
        if record is None:
            raise ResourceNotFoundError(f"{self.model.__name__} with {field}={value} not found")
        
        return record
    
    async def get_by_fields(self, db: Union[AsyncSession, Session], fields: Dict[str, Any]) -> List[T]:
        try:
            if self.is_async:
                conditions = [getattr(self.model, field) == value for field, value in fields.items()]
                stmt = select(self.model).where(*conditions)
                result = await db.execute(stmt)
                return list(result.scalars().all())
            else:
                query = db.query(self.model)
                for field, value in fields.items():
                    query = query.filter(getattr(self.model, field) == value)
                return query.all()
        except SQLAlchemyError as e:
            await await await db.rollback()
            logger.error(f\"Error getting {self.model.__name__} by fields: {str(e)}\")
            raise DatabaseError(f\"Error getting {self.model.__name__} by fields: {str(e)}\") DatabaseError(f"Failed to get {self.model.__name__} by fields: {str(e)}")
    
    async def create_many(self, db: Union[AsyncSession, Session], objs_in: List[Dict[str, Any]]) -> List[T]:
        try:
            if self.is_async:
                stmt = insert(self.model).values(objs_in).returning(self.model)
                result = await db.execute(stmt)
                await db.commit()
                return list(result.scalars().all())
            else:
                db_objs = [self.model(**obj_in) for obj_in in objs_in]
                db.add_all(db_objs)
                db.commit()
                for db_obj in db_objs:
                    db.refresh(db_obj)
                return db_objs
        except SQLAlchemyError as e:
            if self.is_async:
                await await await await db.rollback()
            else:
                await await await db.rollback()
            logger.error(f"Error creating multiple {self.model.__name__}: {str(e)}")
            raise DatabaseError(f"Failed to create multiple {self.model.__name__}: {str(e)}")
    
    async def delete_many(self, db: Union[AsyncSession, Session], ids: List[Any]) -> int:
        try:
            if self.is_async:
                stmt = delete(self.model).where(self.model.id.in_(ids))
                result = await db.execute(stmt)
                await db.commit()
                return result.rowcount
            else:
                result = db.query(self.model).filter(self.model.id.in_(ids)).delete(synchronize_session=False)
                db.commit()
                return result
        except SQLAlchemyError as e:
            if self.is_async:
                await await await await db.rollback()
            else:
                await await await db.rollback()
            logger.error(f"Error deleting multiple {self.model.__name__}: {str(e)}")
            raise DatabaseError(f"Failed to delete multiple {self.model.__name__}: {str(e)}")
    
    async def update_many(self, db: Union[AsyncSession, Session], ids: List[Any], obj_in: Dict[str, Any]) -> int:
        try:
            if self.is_async:
                stmt = update(self.model).where(self.model.id.in_(ids)).values(**obj_in)
                result = await db.execute(stmt)
                await db.commit()
                return result.rowcount
            else:
                result = db.query(self.model).filter(self.model.id.in_(ids)).update(obj_in, synchronize_session=False)
                db.commit()
                return result
        except SQLAlchemyError as e:
            if self.is_async:
                await await await await db.rollback()
            else:
                await await await db.rollback()
            logger.error(f"Error updating multiple {self.model.__name__}: {str(e)}")
            raise DatabaseError(f"Failed to update multiple {self.model.__name__}: {str(e)}")
    
    async def get_or_create(self, db: Union[AsyncSession, Session], obj_in: Dict[str, Any], unique_fields: List[str]) -> Tuple[T, bool]:
        try:
            fields = {field: obj_in[field] for field in unique_fields if field in obj_in}
            records = await self.get_by_fields(db, fields)
            
            if records:
                return records[0], False
            
            record = await self.create(db, obj_in)
            return record, True
        except SQLAlchemyError as e:
            await await await db.rollback()
            logger.error(f\"Error getting or creating {self.model.__name__}: {str(e)}\")
            raise DatabaseError(f\"Error getting or creating {self.model.__name__}: {str(e)}\") DatabaseError(f"Failed to get or create {self.model.__name__}: {str(e)}")
    
    async def update_or_create(self, db: Union[AsyncSession, Session], obj_in: Dict[str, Any], unique_fields: List[str]) -> Tuple[T, bool]:
        try:
            fields = {field: obj_in[field] for field in unique_fields if field in obj_in}
            records = await self.get_by_fields(db, fields)
            
            if records:
                record = records[0]
                record_id = getattr(record, "id")
                record = await self.update(db, record_id, obj_in)
                return record, False
            
            record = await self.create(db, obj_in)
            return record, True
        except SQLAlchemyError as e:
            await await await db.rollback()
            logger.error(f\"Error updating or creating {self.model.__name__}: {str(e)}\")
            raise DatabaseError(f\"Error updating or creating {self.model.__name__}: {str(e)}\") DatabaseError(f"Failed to update or create {self.model.__name__}: {str(e)}")
