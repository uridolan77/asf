"""
User repository for the Medical Research Synthesizer.
This module provides a repository for user-related database operations.
"""
from typing import Optional, Dict, Any, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from asf.medical.storage.models.user import MedicalUser
import logging

logger = logging.getLogger(__name__)

class UserRepository:
    """
    Repository for user-related database operations.
    """
    async def get_by_email(self, db: AsyncSession, email: str) -> Optional[MedicalUser]:
        result = await db.execute(select(MedicalUser).where(MedicalUser.email == email))
        return result.scalars().first()

    async def get_by_id(self, db: AsyncSession, user_id: int) -> Optional[MedicalUser]:
        result = await db.execute(select(MedicalUser).where(MedicalUser.id == user_id))
        return result.scalars().first()

    async def create(self, db: AsyncSession, user_data: Dict[str, Any]) -> MedicalUser:
        user = MedicalUser(**user_data)
        db.add(user)
        await db.commit()
        await db.refresh(user)
        return user

    async def update(self, db: AsyncSession, user_id: int, update_data: Dict[str, Any]) -> Optional[MedicalUser]:
        user = await self.get_by_id(db, user_id)
        if not user:
            return None
        for key, value in update_data.items():
            setattr(user, key, value)
        await db.commit()
        await db.refresh(user)
        return user

    async def delete(self, db: AsyncSession, user_id: int) -> bool:
        user = await self.get_by_id(db, user_id)
        if not user:
            return False
        await db.delete(user)
        await db.commit()
        return True

    async def get_all(self, db: AsyncSession, skip: int = 0, limit: int = 100) -> List[MedicalUser]:
        result = await db.execute(select(MedicalUser).offset(skip).limit(limit))
        return result.scalars().all()