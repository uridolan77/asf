"""
Query repository for the Medical Research Synthesizer.
This module provides a repository for query-related database operations.
"""
from typing import List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from datetime import datetime, timezone
import logging
from ..models import Query
from ..repositories.enhanced_base_repository import EnhancedBaseRepository
from ...core.exceptions import DatabaseError
logger = logging.getLogger(__name__)
class QueryRepository(EnhancedBaseRepository[Query]):
    """
    Repository for query-related database operations.
    """
    def __init__(self):
        """Initialize the repository with the Query model.

        This constructor initializes the repository with the Query model
        for database operations.
        """
        super().__init__(Query)
    async def create_query_async(self, db: AsyncSession, user_id: int, query_text: str, query_type: str = "text", parameters: Dict[str, Any] = None) -> Query:
        """Create a new query asynchronously.

        Args:
            db: The database session
            user_id: The ID of the user who created this query
            query_text: The query text
            query_type: The type of query (default: 'text')
            parameters: Additional parameters for the query

        Returns:
            The created Query object

        Raises:
            DatabaseError: If an error occurs during query creation
        """
        try:
            query = Query(
                user_id=user_id,
                query_text=query_text,
                query_type=query_type,
                parameters=parameters,
                created_at=datetime.now(timezone.utc)
            )
            db.add(query)
            await db.commit()
            await db.refresh(query)
            return query
        except Exception as e:
            await db.rollback()
            logger.error(f"Error creating query: {str(e)}")
            raise DatabaseError(f"Failed to create query: {str(e)}")
    async def get_user_queries_async(self, db: AsyncSession, user_id: int, skip: int = 0, limit: int = 100) -> List[Query]:
        """Get queries for a specific user asynchronously.

        Args:
            db: The database session
            user_id: The ID of the user whose queries to retrieve
            skip: Number of queries to skip (for pagination)
            limit: Maximum number of queries to return

        Returns:
            List of Query objects

        Raises:
            DatabaseError: If an error occurs during query retrieval
        """
        try:
            stmt = select(Query).where(Query.user_id == user_id).offset(skip).limit(limit).order_by(Query.created_at.desc())
            result = await db.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            logger.error(f"Error getting user queries: {str(e)}")
            raise DatabaseError(f"Failed to get user queries: {str(e)}")