"""
Base repository for database operations.
This module provides a generic base repository class for database operations
using SQLAlchemy's async API. It includes proper error handling, transaction
management, and retry logic for common database operations.
"""
from typing import Generic, TypeVar, Type, List, Optional, Any, Dict
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.sql import delete, update
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from sqlalchemy import func
from ...core.exceptions import (
    DatabaseError, NotFoundError, DuplicateError, RepositoryError
)
from ...core.logging_config import get_logger
T = TypeVar('T')
class AsyncRepository(Generic[T]):
    """
    Base repository for async database operations.
    This class provides a generic implementation of common database operations
    using SQLAlchemy's async API. It includes proper error handling and
    transaction management.
    """
    def __init__(self, model_class: Type[T]):
        """
        Initialize the repository with the model class.
        Args:
            model_class: The SQLAlchemy model class this repository will handle
        """
        self.model_class = model_class
        self.logger = get_logger(f"{__name__}.{model_class.__name__}Repository")
    async def get_by_id(self, db: AsyncSession, id: Any) -> Optional[T]:
        """
        Get an entity by its ID.
        Args:
            db: The database session
            id: The entity ID
        Returns:
            The entity or None if not found
        Raises:
            DatabaseError: If there's an error accessing the database
        """
        try:
            stmt = select(self.model_class).where(self.model_class.id == id)
            result = await db.execute(stmt)
            return result.scalars().first()
        except SQLAlchemyError as e:
            self.logger.error(
                f"Error retrieving {self.model_class.__name__} with id {id}",
                extra={"error": str(e), "id": id},
                exc_info=e
            )
            raise DatabaseError(
                f"Failed to retrieve {self.model_class.__name__}",
                details={"id": id, "error": str(e)}
            ) from e
    async def get_by_id_or_404(self, db: AsyncSession, id: Any) -> T:
        """
        Get an entity by its ID or raise a NotFoundError.
        Args:
            db: The database session
            id: The entity ID
        Returns:
            The entity
        Raises:
            NotFoundError: If the entity is not found
            DatabaseError: If there's an error accessing the database
        """
        entity = await self.get_by_id(db, id)
        if entity is None:
            raise NotFoundError(
                self.model_class.__name__,
                str(id),
                details={"id": id}
            )
        return entity
    async def get_all(
        self,
        db: AsyncSession,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[T]:
        """
        Get all entities with pagination and optional filtering.
        Args:
            db: The database session
            skip: Number of records to skip
            limit: Maximum number of records to return
            filters: Optional dictionary of filters to apply
        Returns:
            List of entities
        Raises:
            DatabaseError: If there's an error accessing the database
        """
        try:
            stmt = select(self.model_class)
            if filters:
                for key, value in filters.items():
                    if hasattr(self.model_class, key):
                        stmt = stmt.where(getattr(self.model_class, key) == value)
            stmt = stmt.offset(skip).limit(limit)
            result = await db.execute(stmt)
            return list(result.scalars().all())
        except SQLAlchemyError as e:
            self.logger.error(
                f"Error retrieving {self.model_class.__name__} list",
                extra={"error": str(e), "skip": skip, "limit": limit, "filters": filters},
                exc_info=e
            )
            raise DatabaseError(
                f"Failed to retrieve {self.model_class.__name__} list",
                details={"skip": skip, "limit": limit, "filters": filters, "error": str(e)}
            ) from e
    async def create(self, db: AsyncSession, obj_in: Dict[str, Any]) -> T:
        """
        Create a new entity.
        Args:
            db: The database session
            obj_in: Dictionary with entity attributes
        Returns:
            The created entity
        Raises:
            DuplicateError: If the entity already exists
            DatabaseError: If there's an error accessing the database
        """
        try:
            db_obj = self.model_class(**obj_in)
            db.add(db_obj)
            await db.commit()
            await db.refresh(db_obj)
            return db_obj
        except IntegrityError as e:
            await db.rollback()
            self.logger.error(
                f"Integrity error creating {self.model_class.__name__}",
                extra={"error": str(e), "data": obj_in},
                exc_info=e
            )
            if "unique constraint" in str(e).lower() or "duplicate key" in str(e).lower():
                raise DuplicateError(
                    self.model_class.__name__,
                    str(obj_in.get("id", "unknown")),
                    details={"data": obj_in, "error": str(e)}
                ) from e
            raise DatabaseError(
                f"Failed to create {self.model_class.__name__} due to integrity error",
                details={"data": obj_in, "error": str(e)}
            ) from e
        except SQLAlchemyError as e:
            await db.rollback()
            self.logger.error(
                f"Error creating {self.model_class.__name__}",
                extra={"error": str(e), "data": obj_in},
                exc_info=e
            )
            raise DatabaseError(
                f"Failed to create {self.model_class.__name__}",
                details={"data": obj_in, "error": str(e)}
            ) from e
    async def update(self, db: AsyncSession, id: Any, obj_in: Dict[str, Any]) -> T:
        """
        Update an existing entity.
        Args:
            db: The database session
            id: The entity ID
            obj_in: Dictionary with entity attributes to update
        Returns:
            The updated entity
        Raises:
            NotFoundError: If the entity is not found
            DatabaseError: If there's an error accessing the database
        """
        try:
            db_obj = await self.get_by_id_or_404(db, id)
            for key, value in obj_in.items():
                if hasattr(db_obj, key):
                    setattr(db_obj, key, value)
            await db.commit()
            await db.refresh(db_obj)
            return db_obj
        except NotFoundError:
            raise
        except SQLAlchemyError as e:
            await db.rollback()
            self.logger.error(
                f"Error updating {self.model_class.__name__} with id {id}",
                extra={"error": str(e), "id": id, "data": obj_in},
                exc_info=e
            )
            raise DatabaseError(
                f"Failed to update {self.model_class.__name__}",
                details={"id": id, "data": obj_in, "error": str(e)}
            ) from e
    async def delete(self, db: AsyncSession, id: Any) -> bool:
        """
        Delete an entity by ID.
        Args:
            db: The database session
            id: The entity ID
        Returns:
            True if the entity was deleted, False if it wasn't found
        Raises:
            DatabaseError: If there's an error accessing the database
        """
        try:
            # Check if the entity exists
            entity = await self.get_by_id(db, id)
            if entity is None:
                return False
            await db.delete(entity)
            await db.commit()
            return True
        except SQLAlchemyError as e:
            await db.rollback()
            self.logger.error(
                f"Error deleting {self.model_class.__name__} with id {id}",
                extra={"error": str(e), "id": id},
                exc_info=e
            )
            raise DatabaseError(
                f"Failed to delete {self.model_class.__name__}",
                details={"id": id, "error": str(e)}
            ) from e
    async def count(self, db: AsyncSession, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Count entities with optional filtering.
        Args:
            db: The database session
            filters: Optional dictionary of filters to apply
        Returns:
            The number of entities
        Raises:
            DatabaseError: If there's an error accessing the database
        """
        try:
            stmt = select(func.count()).select_from(self.model_class)
            if filters:
                for key, value in filters.items():
                    if hasattr(self.model_class, key):
                        stmt = stmt.where(getattr(self.model_class, key) == value)
            result = await db.execute(stmt)
            return result.scalar_one()
        except SQLAlchemyError as e:
            self.logger.error(
                f"Error counting {self.model_class.__name__} entities",
                extra={"error": str(e), "filters": filters},
                exc_info=e
            )
            raise DatabaseError(
                f"Failed to count {self.model_class.__name__} entities",
                details={"filters": filters, "error": str(e)}
            ) from e
    async def exists(self, db: AsyncSession, id: Any) -> bool:
        """
        Check if an entity with the given ID exists.
        Args:
            db: The database session
            id: The entity ID
        Returns:
            True if the entity exists, False otherwise
        Raises:
            DatabaseError: If there's an error accessing the database
        """
        try:
            stmt = select(func.count()).select_from(self.model_class).where(self.model_class.id == id)
            result = await db.execute(stmt)
            return result.scalar_one() > 0
        except SQLAlchemyError as e:
            self.logger.error(
                f"Error checking existence of {self.model_class.__name__} with id {id}",
                extra={"error": str(e), "id": id},
                exc_info=e
            )
            raise DatabaseError(
                f"Failed to check existence of {self.model_class.__name__}",
                details={"id": id, "error": str(e)}
            ) from e
    async def bulk_create(self, db: AsyncSession, objects: List[Dict[str, Any]]) -> List[T]:
        """
        Create multiple entities in a single transaction.
        Args:
            db: The database session
            objects: List of dictionaries with entity attributes
        Returns:
            List of created entities
        Raises:
            DatabaseError: If there's an error accessing the database
        """
        try:
            db_objects = [self.model_class(**obj) for obj in objects]
            db.add_all(db_objects)
            await db.commit()
            for obj in db_objects:
                await db.refresh(obj)
            return db_objects
        except SQLAlchemyError as e:
            await db.rollback()
            self.logger.error(
                f"Error bulk creating {self.model_class.__name__} entities",
                extra={"error": str(e), "count": len(objects)},
                exc_info=e
            )
            raise DatabaseError(
                f"Failed to bulk create {self.model_class.__name__} entities",
                details={"count": len(objects), "error": str(e)}
            ) from e
    async def bulk_update(self, db: AsyncSession, updates: List[Dict[str, Any]]) -> int:
        """
        Update multiple entities in a single transaction.
        Args:
            db: The database session
            updates: List of dictionaries with 'id' and update attributes
        Returns:
            Number of updated entities
        Raises:
            DatabaseError: If there's an error accessing the database
        """
        try:
            updated_count = 0
            for update_data in updates:
                id_value = update_data.pop('id', None)
                if id_value is None:
                    continue
                stmt = (
                    update(self.model_class)
                    .where(self.model_class.id == id_value)
                    .values(**update_data)
                )
                result = await db.execute(stmt)
                updated_count += result.rowcount
            await db.commit()
            return updated_count
        except SQLAlchemyError as e:
            await db.rollback()
            self.logger.error(
                f"Error bulk updating {self.model_class.__name__} entities",
                extra={"error": str(e), "count": len(updates)},
                exc_info=e
            )
            raise DatabaseError(
                f"Failed to bulk update {self.model_class.__name__} entities",
                details={"count": len(updates), "error": str(e)}
            ) from e
    async def bulk_delete(self, db: AsyncSession, ids: List[Any]) -> int:
        """
        Delete multiple entities by ID in a single transaction.
        Args:
            db: The database session
            ids: List of entity IDs to delete
        Returns:
            Number of deleted entities
        Raises:
            DatabaseError: If there's an error accessing the database
        """
        try:
            stmt = delete(self.model_class).where(self.model_class.id.in_(ids))
            result = await db.execute(stmt)
            await db.commit()
            return result.rowcount
        except SQLAlchemyError as e:
            await db.rollback()
            self.logger.error(
                f"Error bulk deleting {self.model_class.__name__} entities",
                extra={"error": str(e), "count": len(ids)},
                exc_info=e
            )
            raise DatabaseError(
                f"Failed to bulk delete {self.model_class.__name__} entities",
                details={"count": len(ids), "error": str(e)}
            ) from e