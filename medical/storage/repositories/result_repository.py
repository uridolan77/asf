"""
Result repository for the Medical Research Synthesizer.
This module provides a repository for result-related database operations.
"""
from typing import Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
import logging
logger = logging.getLogger(__name__)
from ..models import Result
from .enhanced_base_repository import EnhancedBaseRepository
class ResultRepository(EnhancedBaseRepository[Result]):
    """
    Repository for result-related database operations.
    """
    def __init__(self):
        """Initialize the repository with the Result model.

        This constructor initializes the repository with the Result model
        for database operations.
        """
        super().__init__(Result)
    async def create_result_async(self, db: AsyncSession, query_id: int, user_id: int, result_type: str, result_data: Dict[str, Any]) -> Result:
        """Create a new result asynchronously.

        Args:
            db: The database session
            query_id: The ID of the query associated with this result
            user_id: The ID of the user who created this result
            result_type: The type of result (e.g., 'search', 'analysis')
            result_data: The result data as a dictionary

        Returns:
            The created Result object
        """
        result = Result(
            query_id=query_id,
            user_id=user_id,
            result_type=result_type,
            result_data=result_data
        )
        return await self.create_async(db, result)

    async def get_by_result_id_async(self, db: AsyncSession, result_id: str) -> Result:
        """Get a result by its ID asynchronously.

        Args:
            db: The database session
            result_id: The ID of the result to retrieve

        Returns:
            The Result object if found, None otherwise
        """
        return await self.get_by_id_async(db, result_id)