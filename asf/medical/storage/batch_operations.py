"""
Batch Database Operations for the Medical Research Synthesizer.
This module provides utilities for performing batch database operations
to improve performance when dealing with large datasets.
"""
import logging
import asyncio
from typing import List, Dict, Any, Type, TypeVar, Callable
from sqlalchemy import insert, update, delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import SQLAlchemyError
from database import is_async
from ..core.exceptions import DatabaseError
logger = logging.getLogger(__name__)
T = TypeVar('T')
class BatchOperations:
    """
    Batch database operations for the Medical Research Synthesizer.

    This class provides methods for performing batch database operations
    to improve performance when dealing with large datasets. It includes
    methods for batch insert, update, delete, upsert, and fetch operations,
    as well as a utility for executing operations in batches with concurrency
    control.

    All methods in this class are static and require an async database session.
    They are designed to work with SQLAlchemy models and async sessions.
    """
    @staticmethod
    async def batch_insert(
        db: AsyncSession,
        model: Type[T],
        items: List[Dict[str, Any]],
        batch_size: int = 100,
        return_defaults: bool = False
    ) -> List[T]:
        """
        Insert multiple items into the database in batches.

        This method divides the items into batches of the specified size and
        inserts them into the database using SQLAlchemy's insert statement.
        It can optionally return the inserted items with their default values.

        Args:
            db: The async database session
            model: The SQLAlchemy model class
            items: List of dictionaries with item attributes
            batch_size: Number of items to insert in each batch
            return_defaults: Whether to return the inserted items with defaults

        Returns:
            List of inserted items if return_defaults is True, otherwise empty list

        Raises:
            ValueError: If batch operations are not supported with the database
            DatabaseError: If there's an error inserting the items
        """
        if not is_async:
            raise ValueError("Batch operations are only supported with async database")
        if not items:
            return []
        batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]
        results = []
        for batch in batches:
            try:
                stmt = insert(model).values(batch)
                if return_defaults:
                    result = await db.execute(stmt.returning(*model.__table__.columns))
                    batch_results = result.fetchall()
                    results.extend(batch_results)
                else:
                    await db.execute(stmt)
            except SQLAlchemyError as e:
                logger.error(f"Error in batch insert: {str(e)}")
                await db.rollback()
                raise DatabaseError(f"Failed to insert batch: {str(e)}")
        await db.commit()
        return results
    @staticmethod
    async def batch_update(
        db: AsyncSession,
        model: Type[T],
        items: List[Dict[str, Any]],
        primary_key: str = "id",
        batch_size: int = 100
    ) -> int:
        """
        Update multiple items in the database in batches.

        This method divides the items into batches of the specified size and
        updates them in the database using SQLAlchemy's update statement.
        Each item must contain the primary key to identify the record to update.

        Args:
            db: The async database session
            model: The SQLAlchemy model class
            items: List of dictionaries with item attributes including primary key
            primary_key: Name of the primary key field (default: "id")
            batch_size: Number of items to update in each batch

        Returns:
            Number of updated items

        Raises:
            ValueError: If batch operations are not supported with the database
            DatabaseError: If there's an error updating the items
        """
        if not is_async:
            raise ValueError("Batch operations are only supported with async database")
        if not items:
            return 0
        batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]
        total_updated = 0
        for batch in batches:
            try:
                for item in batch:
                    if primary_key not in item:
                        logger.warning(f"Skipping update for item without primary key: {item}")
                        continue
                    stmt = update(model).where(getattr(model, primary_key) == item[primary_key]).values(item)
                    result = await db.execute(stmt)
                    total_updated += result.rowcount
            except SQLAlchemyError as e:
                logger.error(f"Error in batch update: {str(e)}")
                await db.rollback()
                raise DatabaseError(f"Failed to update batch: {str(e)}")
        await db.commit()
        return total_updated
    @staticmethod
    async def batch_delete(
        db: AsyncSession,
        model: Type[T],
        ids: List[Any],
        primary_key: str = "id",
        batch_size: int = 100
    ) -> int:
        """
        Delete multiple items from the database in batches.

        This method divides the IDs into batches of the specified size and
        deletes the corresponding items from the database using SQLAlchemy's
        delete statement.

        Args:
            db: The async database session
            model: The SQLAlchemy model class
            ids: List of primary key values for items to delete
            primary_key: Name of the primary key field (default: "id")
            batch_size: Number of items to delete in each batch

        Returns:
            Number of deleted items

        Raises:
            ValueError: If batch operations are not supported with the database
            DatabaseError: If there's an error deleting the items
        """
        if not is_async:
            raise ValueError("Batch operations are only supported with async database")
        if not ids:
            return 0
        batches = [ids[i:i+batch_size] for i in range(0, len(ids), batch_size)]
        total_deleted = 0
        for batch in batches:
            try:
                stmt = delete(model).where(getattr(model, primary_key).in_(batch))
                result = await db.execute(stmt)
                total_deleted += result.rowcount
            except SQLAlchemyError as e:
                logger.error(f"Error in batch delete: {str(e)}")
                await db.rollback()
                raise DatabaseError(f"Failed to delete batch: {str(e)}")
        await db.commit()
        return total_deleted
    @staticmethod
    async def batch_upsert(
        db: AsyncSession,
        model: Type[T],
        items: List[Dict[str, Any]],
        constraint_columns: List[str],
        batch_size: int = 100,
        return_defaults: bool = False
    ) -> List[T]:
        """
        Insert or update multiple items in the database in batches.

        This method divides the items into batches of the specified size and
        performs an upsert operation (insert or update) using PostgreSQL's
        ON CONFLICT DO UPDATE clause. It can optionally return the inserted
        or updated items with their default values.

        Args:
            db: The async database session
            model: The SQLAlchemy model class
            items: List of dictionaries with item attributes
            constraint_columns: List of column names that form the constraint
            batch_size: Number of items to upsert in each batch
            return_defaults: Whether to return the upserted items with defaults

        Returns:
            List of upserted items if return_defaults is True, otherwise empty list

        Raises:
            ValueError: If batch operations are not supported with the database
            DatabaseError: If there's an error upserting the items
        """
        if not is_async:
            raise ValueError("Batch operations are only supported with async database")
        if not items:
            return []
        batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]
        results = []
        for batch in batches:
            try:
                stmt = pg_insert(model).values(batch)
                update_dict = {c.name: getattr(stmt.excluded, c.name)
                              for c in model.__table__.columns
                              if c.name not in constraint_columns}
                stmt = stmt.on_conflict_do_update(
                    constraint=model.__table__.primary_key,
                    set_=update_dict
                )
                if return_defaults:
                    result = await db.execute(stmt.returning(*model.__table__.columns))
                    batch_results = result.fetchall()
                    results.extend(batch_results)
                else:
                    await db.execute(stmt)
            except SQLAlchemyError as e:
                logger.error(f"Error in batch upsert: {str(e)}")
                await db.rollback()
                raise DatabaseError(f"Failed to upsert batch: {str(e)}")
        await db.commit()
        return results
    @staticmethod
    async def batch_fetch(
        db: AsyncSession,
        model: Type[T],
        ids: List[Any],
        primary_key: str = "id",
        batch_size: int = 100
    ) -> List[T]:
        """
        Fetch multiple items from the database in batches.

        This method divides the IDs into batches of the specified size and
        fetches the corresponding items from the database using SQLAlchemy's
        select statement.

        Args:
            db: The async database session
            model: The SQLAlchemy model class
            ids: List of primary key values for items to fetch
            primary_key: Name of the primary key field (default: "id")
            batch_size: Number of items to fetch in each batch

        Returns:
            List of fetched items

        Raises:
            ValueError: If batch operations are not supported with the database
            DatabaseError: If there's an error fetching the items
        """
        if not is_async:
            raise ValueError("Batch operations are only supported with async database")
        if not ids:
            return []
        batches = [ids[i:i+batch_size] for i in range(0, len(ids), batch_size)]
        results = []
        for batch in batches:
            try:
                stmt = select(model).where(getattr(model, primary_key).in_(batch))
                result = await db.execute(stmt)
                batch_results = result.scalars().all()
                results.extend(batch_results)
            except SQLAlchemyError as e:
                await db.rollback()
                logger.error(f"Error in batch fetch: {str(e)}")
                raise DatabaseError(f"Failed to fetch batch: {str(e)}")
        return results
    @staticmethod
    async def execute_in_batches(
        func: Callable,
        items: List[Any],
        batch_size: int = 100,
        max_concurrency: int = 5,
        *args,
        **kwargs
    ) -> List[Any]:
        """
        Execute a function on batches of items with concurrency control.

        This method divides the items into batches of the specified size and
        executes the provided function on each batch concurrently, up to the
        specified maximum concurrency. It collects and returns the results.

        Args:
            func: The async function to execute on each batch
            items: List of items to process
            batch_size: Number of items to process in each batch
            max_concurrency: Maximum number of concurrent batch operations
            *args: Additional positional arguments to pass to the function
            **kwargs: Additional keyword arguments to pass to the function

        Returns:
            List of results from all batches

        Raises:
            Exception: Any exception raised by the function
        """
        if not items:
            return []
        batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]
        semaphore = asyncio.Semaphore(max_concurrency)
        async def process_batch(batch: List[Any]) -> List[Any]:
            async with semaphore:
                try:
                    return await func(batch, *args, **kwargs)
                except Exception as e:
                    logger.error(f"Error processing batch: {str(e)}")
                    raise DatabaseError(f"Error processing batch: {str(e)}")
        tasks = [process_batch(batch) for batch in batches]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error(f"Batch {i} failed: {str(result)}")
                raise result
        results = []
        for batch_result in batch_results:
            if isinstance(batch_result, list):
                results.extend(batch_result)
            else:
                results.append(batch_result)
        return results