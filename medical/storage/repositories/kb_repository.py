"""
Knowledge Base repository for the Medical Research Synthesizer.
This module provides a repository for knowledge base-related database operations.
"""
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession
import logging
logger = logging.getLogger(__name__)
from asf.medical.storage.models import KnowledgeBase
from asf.medical.storage.repositories.enhanced_base_repository import EnhancedBaseRepository
class KnowledgeBaseRepository(EnhancedBaseRepository[KnowledgeBase]):
    """
    Repository for knowledge base-related database operations.
    """
    def __init__(self):
        """Initialize the repository with the KnowledgeBase model.

        This constructor initializes the repository with the KnowledgeBase model
        for database operations.
        """
        super().__init__(KnowledgeBase)
    def create_knowledge_base(
        self, db: Session, name: str, query: str, file_path: str,
        update_schedule: str, initial_results: int, user_id: int
    ) -> KnowledgeBase:
        """Create a new knowledge base.

        Args:
            db: The database session
            name: The name of the knowledge base
            query: The query used to generate the knowledge base
            file_path: The file path where the knowledge base is stored
            update_schedule: The schedule for updating the knowledge base
            initial_results: The number of initial results
            user_id: The ID of the user who created this knowledge base

        Returns:
            The created KnowledgeBase object
        """
        kb = KnowledgeBase(
            name=name,
            query=query,
            file_path=file_path,
            update_schedule=update_schedule,
            initial_results=initial_results,
            user_id=user_id
        )
        return self.create(db, kb)

    async def create_knowledge_base_async(
        self, db: AsyncSession, name: str, query: str, file_path: str,
        update_schedule: str, initial_results: int, user_id: int
    ) -> KnowledgeBase:
        """Create a new knowledge base asynchronously.

        Args:
            db: The database session
            name: The name of the knowledge base
            query: The query used to generate the knowledge base
            file_path: The file path where the knowledge base is stored
            update_schedule: The schedule for updating the knowledge base
            initial_results: The number of initial results
            user_id: The ID of the user who created this knowledge base

        Returns:
            The created KnowledgeBase object
        """
        kb = KnowledgeBase(
            name=name,
            query=query,
            file_path=file_path,
            update_schedule=update_schedule,
            initial_results=initial_results,
            user_id=user_id
        )
        return await self.create_async(db, kb)