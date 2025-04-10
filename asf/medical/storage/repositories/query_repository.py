"""
Query repository for the Medical Research Synthesizer.

This module provides a repository for query-related database operations.
"""

from typing import Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import Session
import datetime

from asf.medical.storage.models import Query
from asf.medical.storage.repositories.base_repository import BaseRepository
from asf.medical.storage.database import is_async

class QueryRepository(BaseRepository[Query]):
    """
    Repository for query-related database operations.
    """
    
    def __init__(self):
        """Initialize the repository with the Query model."""
        super().__init__(Query)
    
    # Synchronous methods
    def create_query(self, db: Session, user_id: int, query_text: str, query_type: str = "text", parameters: Dict[str, Any] = None) -> Query:
        """
        Create a new query.
        
        Args:
            db: Database session
            user_id: User ID
            query_text: Query text
            query_type: Query type (default: "text")
            parameters: Query parameters (default: None)
            
        Returns:
            The created query
        """
        if is_async:
            raise ValueError("Cannot use synchronous methods with async database")
        
        query = Query(
            user_id=user_id,
            query_text=query_text,
            query_type=query_type,
            parameters=parameters,
            created_at=datetime.datetime.utcnow()
        )
        db.add(query)
        db.commit()
        db.refresh(query)
        return query
    
    def get_user_queries(self, db: Session, user_id: int, skip: int = 0, limit: int = 100) -> List[Query]:
        """
        Get all queries for a user.
        
        Args:
            db: Database session
            user_id: User ID
            skip: Number of queries to skip
            limit: Maximum number of queries to return
            
        Returns:
            List of queries
        """
        if is_async:
            raise ValueError("Cannot use synchronous methods with async database")
        
        return db.query(Query).filter(Query.user_id == user_id).order_by(Query.created_at.desc()).offset(skip).limit(limit).all()
    
    # Asynchronous methods
    async def create_query_async(self, db: AsyncSession, user_id: int, query_text: str, query_type: str = "text", parameters: Dict[str, Any] = None) -> Query:
        """
        Create a new query asynchronously.
        
        Args:
            db: Async database session
            user_id: User ID
            query_text: Query text
            query_type: Query type (default: "text")
            parameters: Query parameters (default: None)
            
        Returns:
            The created query
        """
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
    
    async def get_user_queries_async(self, db: AsyncSession, user_id: int, skip: int = 0, limit: int = 100) -> List[Query]:
        """
        Get all queries for a user asynchronously.
        
        Args:
            db: Async database session
            user_id: User ID
            skip: Number of queries to skip
            limit: Maximum number of queries to return
            
        Returns:
            List of queries
        """
        result = await db.execute(
            select(Query)
            .filter(Query.user_id == user_id)
            .order_by(Query.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
