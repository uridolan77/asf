"""
Knowledge Base repository for the Medical Research Synthesizer.

This module provides a repository for knowledge base-related database operations.
"""

from typing import Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import Session
import datetime
import uuid

from asf.medical.storage.models import KnowledgeBase, KnowledgeBaseUpdate
from asf.medical.storage.repositories.base_repository import BaseRepository
from asf.medical.storage.database import is_async

class KnowledgeBaseRepository(BaseRepository[KnowledgeBase]):
    """
    Repository for knowledge base-related database operations.
    """
    
    def __init__(self):
        """Initialize the repository with the KnowledgeBase model."""
        super().__init__(KnowledgeBase)
    
    # Synchronous methods
    def create_knowledge_base(
        self, db: Session, name: str, query: str, file_path: str, 
        update_schedule: str, initial_results: int, user_id: int
    ) -> KnowledgeBase:
        """
        Create a new knowledge base.
        
        Args:
            db: Database session
            name: Knowledge base name
            query: Search query
            file_path: Path to the knowledge base file
            update_schedule: Update schedule (daily, weekly, monthly)
            initial_results: Number of initial results
            user_id: User ID
            
        Returns:
            The created knowledge base
        """
        if is_async:
            raise ValueError("Cannot use synchronous methods with async database")
        
        # Calculate next update time based on schedule
        next_update = self._calculate_next_update(update_schedule)
        
        kb = KnowledgeBase(
            kb_id=str(uuid.uuid4()),
            name=name,
            query=query,
            file_path=file_path,
            update_schedule=update_schedule,
            last_updated=datetime.datetime.utcnow(),
            next_update=next_update,
            initial_results=initial_results,
            user_id=user_id,
            created_at=datetime.datetime.utcnow()
        )
        db.add(kb)
        db.commit()
        db.refresh(kb)
        return kb
    
    def get_by_name(self, db: Session, name: str) -> Optional[KnowledgeBase]:
        """
        Get a knowledge base by name.
        
        Args:
            db: Database session
            name: Knowledge base name
            
        Returns:
            The knowledge base or None if not found
        """
        if is_async:
            raise ValueError("Cannot use synchronous methods with async database")
        
        return db.query(KnowledgeBase).filter(KnowledgeBase.name == name).first()
    
    def get_by_kb_id(self, db: Session, kb_id: str) -> Optional[KnowledgeBase]:
        """
        Get a knowledge base by kb_id.
        
        Args:
            db: Database session
            kb_id: Knowledge base ID (UUID)
            
        Returns:
            The knowledge base or None if not found
        """
        if is_async:
            raise ValueError("Cannot use synchronous methods with async database")
        
        return db.query(KnowledgeBase).filter(KnowledgeBase.kb_id == kb_id).first()
    
    def get_user_knowledge_bases(self, db: Session, user_id: int) -> List[KnowledgeBase]:
        """
        Get all knowledge bases for a user.
        
        Args:
            db: Database session
            user_id: User ID
            
        Returns:
            List of knowledge bases
        """
        if is_async:
            raise ValueError("Cannot use synchronous methods with async database")
        
        return db.query(KnowledgeBase).filter(KnowledgeBase.user_id == user_id).order_by(KnowledgeBase.created_at.desc()).all()
    
    def get_due_for_update(self, db: Session) -> List[KnowledgeBase]:
        """
        Get all knowledge bases that are due for an update.
        
        Args:
            db: Database session
            
        Returns:
            List of knowledge bases
        """
        if is_async:
            raise ValueError("Cannot use synchronous methods with async database")
        
        now = datetime.datetime.utcnow()
        return db.query(KnowledgeBase).filter(KnowledgeBase.next_update <= now).all()
    
    def update_last_updated(self, db: Session, kb_id: int) -> Optional[KnowledgeBase]:
        """
        Update a knowledge base's last_updated timestamp and calculate next update.
        
        Args:
            db: Database session
            kb_id: Knowledge base ID
            
        Returns:
            The updated knowledge base or None if not found
        """
        if is_async:
            raise ValueError("Cannot use synchronous methods with async database")
        
        kb = self.get(db, kb_id)
        if kb:
            kb.last_updated = datetime.datetime.utcnow()
            kb.next_update = self._calculate_next_update(kb.update_schedule)
            db.commit()
            db.refresh(kb)
        return kb
    
    def add_update_record(
        self, db: Session, kb_id: int, new_results: int, 
        total_results: int, status: str, error_message: str = None
    ) -> KnowledgeBaseUpdate:
        """
        Add an update record for a knowledge base.
        
        Args:
            db: Database session
            kb_id: Knowledge base ID
            new_results: Number of new results
            total_results: Total number of results
            status: Update status (success, failure)
            error_message: Error message (if status is failure)
            
        Returns:
            The created update record
        """
        if is_async:
            raise ValueError("Cannot use synchronous methods with async database")
        
        update = KnowledgeBaseUpdate(
            kb_id=kb_id,
            update_time=datetime.datetime.utcnow(),
            new_results=new_results,
            total_results=total_results,
            status=status,
            error_message=error_message
        )
        db.add(update)
        db.commit()
        db.refresh(update)
        return update
    
    # Asynchronous methods
    async def create_knowledge_base_async(
        self, db: AsyncSession, name: str, query: str, file_path: str, 
        update_schedule: str, initial_results: int, user_id: int
    ) -> KnowledgeBase:
        """
        Create a new knowledge base asynchronously.
        
        Args:
            db: Async database session
            name: Knowledge base name
            query: Search query
            file_path: Path to the knowledge base file
            update_schedule: Update schedule (daily, weekly, monthly)
            initial_results: Number of initial results
            user_id: User ID
            
        Returns:
            The created knowledge base
        """
        # Calculate next update time based on schedule
        next_update = self._calculate_next_update(update_schedule)
        
        kb = KnowledgeBase(
            kb_id=str(uuid.uuid4()),
            name=name,
            query=query,
            file_path=file_path,
            update_schedule=update_schedule,
            last_updated=datetime.datetime.utcnow(),
            next_update=next_update,
            initial_results=initial_results,
            user_id=user_id,
            created_at=datetime.datetime.utcnow()
        )
        db.add(kb)
        await db.commit()
        await db.refresh(kb)
        return kb
    
    async def get_by_name_async(self, db: AsyncSession, name: str) -> Optional[KnowledgeBase]:
        """
        Get a knowledge base by name asynchronously.
        
        Args:
            db: Async database session
            name: Knowledge base name
            
        Returns:
            The knowledge base or None if not found
        """
        result = await db.execute(select(KnowledgeBase).filter(KnowledgeBase.name == name))
        return result.scalars().first()
    
    async def get_by_kb_id_async(self, db: AsyncSession, kb_id: str) -> Optional[KnowledgeBase]:
        """
        Get a knowledge base by kb_id asynchronously.
        
        Args:
            db: Async database session
            kb_id: Knowledge base ID (UUID)
            
        Returns:
            The knowledge base or None if not found
        """
        result = await db.execute(select(KnowledgeBase).filter(KnowledgeBase.kb_id == kb_id))
        return result.scalars().first()
    
    async def get_user_knowledge_bases_async(self, db: AsyncSession, user_id: int) -> List[KnowledgeBase]:
        """
        Get all knowledge bases for a user asynchronously.
        
        Args:
            db: Async database session
            user_id: User ID
            
        Returns:
            List of knowledge bases
        """
        result = await db.execute(
            select(KnowledgeBase)
            .filter(KnowledgeBase.user_id == user_id)
            .order_by(KnowledgeBase.created_at.desc())
        )
        return result.scalars().all()
    
    async def get_due_for_update_async(self, db: AsyncSession) -> List[KnowledgeBase]:
        """
        Get all knowledge bases that are due for an update asynchronously.
        
        Args:
            db: Async database session
            
        Returns:
            List of knowledge bases
        """
        now = datetime.datetime.utcnow()
        result = await db.execute(select(KnowledgeBase).filter(KnowledgeBase.next_update <= now))
        return result.scalars().all()
    
    async def update_last_updated_async(self, db: AsyncSession, kb_id: int) -> Optional[KnowledgeBase]:
        """
        Update a knowledge base's last_updated timestamp and calculate next update asynchronously.
        
        Args:
            db: Async database session
            kb_id: Knowledge base ID
            
        Returns:
            The updated knowledge base or None if not found
        """
        kb = await self.get_async(db, kb_id)
        if kb:
            kb.last_updated = datetime.datetime.utcnow()
            kb.next_update = self._calculate_next_update(kb.update_schedule)
            await db.commit()
            await db.refresh(kb)
        return kb
    
    async def add_update_record_async(
        self, db: AsyncSession, kb_id: int, new_results: int, 
        total_results: int, status: str, error_message: str = None
    ) -> KnowledgeBaseUpdate:
        """
        Add an update record for a knowledge base asynchronously.
        
        Args:
            db: Async database session
            kb_id: Knowledge base ID
            new_results: Number of new results
            total_results: Total number of results
            status: Update status (success, failure)
            error_message: Error message (if status is failure)
            
        Returns:
            The created update record
        """
        update = KnowledgeBaseUpdate(
            kb_id=kb_id,
            update_time=datetime.datetime.utcnow(),
            new_results=new_results,
            total_results=total_results,
            status=status,
            error_message=error_message
        )
        db.add(update)
        await db.commit()
        await db.refresh(update)
        return update
    
    # Helper methods
    def _calculate_next_update(self, schedule: str) -> datetime.datetime:
        """
        Calculate the next update time based on the schedule.
        
        Args:
            schedule: Update schedule (daily, weekly, monthly)
            
        Returns:
            Next update time
        """
        now = datetime.datetime.utcnow()
        
        if schedule == "daily":
            return now + datetime.timedelta(days=1)
        elif schedule == "weekly":
            return now + datetime.timedelta(weeks=1)
        elif schedule == "monthly":
            # Approximate a month as 30 days
            return now + datetime.timedelta(days=30)
        else:
            # Default to weekly
            return now + datetime.timedelta(weeks=1)
