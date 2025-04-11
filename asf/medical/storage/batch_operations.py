"""
Batch Database Operations for the Medical Research Synthesizer.

This module provides utilities for performing batch database operations
to improve performance when dealing with large datasets.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Type, TypeVar, Callable, Union, Tuple
from sqlalchemy import insert, update, delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import SQLAlchemyError

from asf.medical.storage.database import is_async
from asf.medical.core.exceptions import DatabaseError

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for SQLAlchemy models
T = TypeVar('T')

class BatchOperations:
    """
    Batch database operations for the Medical Research Synthesizer.
    
    This class provides methods for performing batch database operations
    to improve performance when dealing with large datasets.
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
        Insert multiple items in batches.
        
        Args:
            db: Database session
            model: SQLAlchemy model class
            items: List of items to insert
            batch_size: Number of items per batch (default: 100)
            return_defaults: Whether to return default values (default: False)
            
        Returns:
            List of inserted items
            
        Raises:
            DatabaseError: If the insert fails
        """
        if not is_async:
            raise ValueError("Batch operations are only supported with async database")
        
        if not items:
            return []
        
        # Split items into batches
        batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]
        
        # Insert batches
        results = []
        for batch in batches:
            try:
                # Create insert statement
                stmt = insert(model).values(batch)
                
                # Execute statement
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
        
        # Commit changes
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
        Update multiple items in batches.
        
        Args:
            db: Database session
            model: SQLAlchemy model class
            items: List of items to update
            primary_key: Name of the primary key column (default: "id")
            batch_size: Number of items per batch (default: 100)
            
        Returns:
            Number of updated items
            
        Raises:
            DatabaseError: If the update fails
        """
        if not is_async:
            raise ValueError("Batch operations are only supported with async database")
        
        if not items:
            return 0
        
        # Split items into batches
        batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]
        
        # Update batches
        total_updated = 0
        for batch in batches:
            try:
                # Update each item in the batch
                for item in batch:
                    if primary_key not in item:
                        logger.warning(f"Skipping update for item without primary key: {item}")
                        continue
                    
                    # Create update statement
                    stmt = update(model).where(getattr(model, primary_key) == item[primary_key]).values(item)
                    
                    # Execute statement
                    result = await db.execute(stmt)
                    total_updated += result.rowcount
            except SQLAlchemyError as e:
                logger.error(f"Error in batch update: {str(e)}")
                await db.rollback()
                raise DatabaseError(f"Failed to update batch: {str(e)}")
        
        # Commit changes
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
        Delete multiple items in batches.
        
        Args:
            db: Database session
            model: SQLAlchemy model class
            ids: List of primary key values to delete
            primary_key: Name of the primary key column (default: "id")
            batch_size: Number of items per batch (default: 100)
            
        Returns:
            Number of deleted items
            
        Raises:
            DatabaseError: If the delete fails
        """
        if not is_async:
            raise ValueError("Batch operations are only supported with async database")
        
        if not ids:
            return 0
        
        # Split IDs into batches
        batches = [ids[i:i+batch_size] for i in range(0, len(ids), batch_size)]
        
        # Delete batches
        total_deleted = 0
        for batch in batches:
            try:
                # Create delete statement
                stmt = delete(model).where(getattr(model, primary_key).in_(batch))
                
                # Execute statement
                result = await db.execute(stmt)
                total_deleted += result.rowcount
            except SQLAlchemyError as e:
                logger.error(f"Error in batch delete: {str(e)}")
                await db.rollback()
                raise DatabaseError(f"Failed to delete batch: {str(e)}")
        
        # Commit changes
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
        Upsert (insert or update) multiple items in batches.
        
        Args:
            db: Database session
            model: SQLAlchemy model class
            items: List of items to upsert
            constraint_columns: List of column names that form the constraint
            batch_size: Number of items per batch (default: 100)
            return_defaults: Whether to return default values (default: False)
            
        Returns:
            List of upserted items
            
        Raises:
            DatabaseError: If the upsert fails
        """
        if not is_async:
            raise ValueError("Batch operations are only supported with async database")
        
        if not items:
            return []
        
        # Split items into batches
        batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]
        
        # Upsert batches
        results = []
        for batch in batches:
            try:
                # Create upsert statement
                stmt = pg_insert(model).values(batch)
                
                # Add on conflict do update clause
                update_dict = {c.name: getattr(stmt.excluded, c.name) 
                              for c in model.__table__.columns 
                              if c.name not in constraint_columns}
                
                stmt = stmt.on_conflict_do_update(
                    constraint=model.__table__.primary_key,
                    set_=update_dict
                )
                
                # Execute statement
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
        
        # Commit changes
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
        Fetch multiple items in batches.
        
        Args:
            db: Database session
            model: SQLAlchemy model class
            ids: List of primary key values to fetch
            primary_key: Name of the primary key column (default: "id")
            batch_size: Number of items per batch (default: 100)
            
        Returns:
            List of fetched items
            
        Raises:
            DatabaseError: If the fetch fails
        """
        if not is_async:
            raise ValueError("Batch operations are only supported with async database")
        
        if not ids:
            return []
        
        # Split IDs into batches
        batches = [ids[i:i+batch_size] for i in range(0, len(ids), batch_size)]
        
        # Fetch batches
        results = []
        for batch in batches:
            try:
                # Create select statement
                stmt = select(model).where(getattr(model, primary_key).in_(batch))
                
                # Execute statement
                result = await db.execute(stmt)
                batch_results = result.scalars().all()
                results.extend(batch_results)
            except SQLAlchemyError as e:
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
        Execute a function in batches with concurrency control.
        
        Args:
            func: Function to execute
            items: List of items to process
            batch_size: Number of items per batch (default: 100)
            max_concurrency: Maximum number of concurrent batches (default: 5)
            *args: Additional positional arguments to pass to the function
            **kwargs: Additional keyword arguments to pass to the function
            
        Returns:
            List of results
            
        Raises:
            Exception: If the function execution fails
        """
        if not items:
            return []
        
        # Split items into batches
        batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]
        
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrency)
        
        # Define a worker function to process each batch
        async def process_batch(batch: List[Any]) -> List[Any]:
            async with semaphore:
                try:
                    return await func(batch, *args, **kwargs)
                except Exception as e:
                    logger.error(f"Error processing batch: {str(e)}")
                    raise
        
        # Process all batches concurrently
        tasks = [process_batch(batch) for batch in batches]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check for exceptions
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error(f"Batch {i} failed: {str(result)}")
                raise result
        
        # Combine results
        results = []
        for batch_result in batch_results:
            if isinstance(batch_result, list):
                results.extend(batch_result)
            else:
                results.append(batch_result)
        
        return results
