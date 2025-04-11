"""
Query repository for the Medical Research Synthesizer.
This module provides a repository for query-related database operations.
"""
from typing import List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
import datetime
import logging
from asf.medical.storage.models import Query
from asf.medical.storage.repositories.enhanced_base_repository import EnhancedBaseRepository
from asf.medical.core.exceptions import DatabaseError
logger = logging.getLogger(__name__)
class QueryRepository(EnhancedBaseRepository[Query]):
    """
    Repository for query-related database operations.
    """
    def __init__(self):
        """Initialize the repository with the Query model.
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
    """
        super().__init__(Query)
    async def create_query_async(self, db: AsyncSession, user_id: int, query_text: str, query_type: str = "text", parameters: Dict[str, Any] = None) -> Query:
        try:
            query = Query(
                user_id=user_id,
                query_text=query_text,
                query_type=query_type,
                parameters=parameters,
                created_at=datetime.datetime.utcnow()
            )
            db.add(query)
            await db.commit()
            await db.refresh(query)
            return query
        except Exception as e:
            await await await await db.rollback()
            logger.error(f"Error creating query: {str(e)}")
            raise DatabaseError(f"Failed to create query: {str(e)}")
    async def get_user_queries_async(self, db: AsyncSession, user_id: int, skip: int = 0, limit: int = 100) -> List[Query]:
        try:
            stmt = select(Query).where(Query.user_id == user_id).offset(skip).limit(limit).order_by(Query.created_at.desc())
            result = await db.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
    logger.error(f\"Error getting user queries: {str(e)}\")
    raise DatabaseError(f\"Error getting user queries: {str(e)}\") DatabaseError(f"Failed to get user queries: {str(e)}")