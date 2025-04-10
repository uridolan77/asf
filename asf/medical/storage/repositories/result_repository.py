"""
Result repository for the Medical Research Synthesizer.

This module provides a repository for result-related database operations.
"""

from typing import Optional, List, Dict, Any, Union
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import Session
import datetime
import uuid
import logging

# Set up logging
logger = logging.getLogger(__name__)

from asf.medical.storage.models import Result
from asf.medical.storage.repositories.base_repository import BaseRepository
from asf.medical.storage.database import is_async

class ResultRepository(BaseRepository[Result]):
    """
    Repository for result-related database operations.
    """

    def __init__(self):
        """Initialize the repository with the Result model."""
        super().__init__(Result)

    # Synchronous methods
    def create_result(self, db: Session, query_id: int, user_id: int, result_type: str, result_data: Dict[str, Any]) -> Result:
        """
        Create a new result.

        Args:
            db: Database session
            query_id: Query ID
            user_id: User ID
            result_type: Result type (search, analysis)
            result_data: Result data

        Returns:
            The created result
        """
        if is_async:
            raise ValueError("Cannot use synchronous methods with async database")

        result = Result(
            result_id=str(uuid.uuid4()),
            query_id=query_id,
            user_id=user_id,
            result_type=result_type,
            result_data=result_data,
            created_at=datetime.datetime.utcnow()
        )
        db.add(result)
        db.commit()
        db.refresh(result)
        return result

    def get_by_result_id(self, db: Session, result_id: str) -> Optional[Result]:
        """
        Get a result by result_id.

        Args:
            db: Database session
            result_id: Result ID (UUID)

        Returns:
            The result or None if not found
        """
        if is_async:
            raise ValueError("Cannot use synchronous methods with async database")

        return db.query(Result).filter(Result.result_id == result_id).first()

    def get_user_results(self, db: Session, user_id: int, skip: int = 0, limit: int = 100) -> List[Result]:
        """
        Get all results for a user.

        Args:
            db: Database session
            user_id: User ID
            skip: Number of results to skip
            limit: Maximum number of results to return

        Returns:
            List of results
        """
        if is_async:
            raise ValueError("Cannot use synchronous methods with async database")

        return db.query(Result).filter(Result.user_id == user_id).order_by(Result.created_at.desc()).offset(skip).limit(limit).all()

    def get_query_results(self, db: Session, query_id: int) -> List[Result]:
        """
        Get all results for a query.

        Args:
            db: Database session
            query_id: Query ID

        Returns:
            List of results
        """
        if is_async:
            raise ValueError("Cannot use synchronous methods with async database")

        return db.query(Result).filter(Result.query_id == query_id).order_by(Result.created_at.desc()).all()

    # Asynchronous methods
    async def create_async(self, db: Optional[AsyncSession] = None, obj_in: Dict[str, Any] = None) -> Result:
        """
        Create a new result asynchronously.

        Args:
            db: Async database session (optional)
            obj_in: Dictionary with model attributes

        Returns:
            The created result
        """
        # Get a database session if not provided
        if db is None:
            from asf.medical.storage.database import get_db_session
            async for session in get_db_session():
                db = session
                break

        # Generate a result ID if not provided
        if 'result_id' not in obj_in:
            obj_in['result_id'] = str(uuid.uuid4())

        # Set created_at if not provided
        if 'created_at' not in obj_in:
            obj_in['created_at'] = datetime.datetime.utcnow()

        # Create the result
        try:
            result = Result(**obj_in)
            db.add(result)
            await db.commit()
            await db.refresh(result)
            return result
        except Exception as e:
            await db.rollback()
            logger.error(f"Error creating result: {str(e)}")
            raise

    async def create_result_async(self, db: AsyncSession, query_id: int, user_id: int, result_type: str, result_data: Dict[str, Any]) -> Result:
        """
        Create a new result asynchronously.

        Args:
            db: Async database session
            query_id: Query ID
            user_id: User ID
            result_type: Result type (search, analysis)
            result_data: Result data

        Returns:
            The created result
        """
        # Create a dictionary with the model attributes
        obj_in = {
            'result_id': str(uuid.uuid4()),
            'query_id': query_id,
            'user_id': user_id,
            'result_type': result_type,
            'result_data': result_data,
            'created_at': datetime.datetime.utcnow()
        }

        # Call the generic create method
        return await self.create_async(db, obj_in)

    async def get_by_result_id_async(self, db: Optional[AsyncSession] = None, result_id: str = None) -> Optional[Result]:
        """
        Get a result by result_id asynchronously.

        Args:
            db: Async database session (optional)
            result_id: Result ID

        Returns:
            The result or None if not found
        """
        # Get a database session if not provided
        if db is None:
            from asf.medical.storage.database import get_db_session
            async for session in get_db_session():
                db = session
                break

        try:
            result = await db.execute(
                select(Result).filter(Result.result_id == result_id)
            )
            return result.scalars().first()
        except Exception as e:
            logger.error(f"Error getting result by ID: {str(e)}")
            return None

    async def get_user_results_async(self, db: AsyncSession, user_id: int, skip: int = 0, limit: int = 100) -> List[Result]:
        """
        Get all results for a user asynchronously.

        Args:
            db: Async database session
            user_id: User ID
            skip: Number of results to skip
            limit: Maximum number of results to return

        Returns:
            List of results
        """
        result = await db.execute(
            select(Result)
            .filter(Result.user_id == user_id)
            .order_by(Result.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()

    async def get_query_results_async(self, db: AsyncSession, query_id: int) -> List[Result]:
        """
        Get all results for a query asynchronously.

        Args:
            db: Async database session
            query_id: Query ID

        Returns:
            List of results
        """
        result = await db.execute(
            select(Result)
            .filter(Result.query_id == query_id)
            .order_by(Result.created_at.desc())
        )
        return result.scalars().all()
