"""
User repository for the Medical Research Synthesizer.
This module provides a repository for user-related database operations.
"""
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
import datetime
import logging
from asf.medical.storage.models import User
from asf.medical.storage.repositories.enhanced_base_repository import EnhancedBaseRepository
from asf.medical.core.exceptions import DatabaseError
logger = logging.getLogger(__name__)
class UserRepository(EnhancedBaseRepository[User]):
    """
    Repository for user-related database operations.
    """
    def __init__(self):
        """Initialize the repository with the User model.
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
    """
        super().__init__(User)
    async def get_by_email_async(self, db: AsyncSession, email: str) -> Optional[User]:
        try:
            stmt = select(User).where(User.email == email)
            result = await db.execute(stmt)
            return result.scalars().first()
        except Exception as e:
    logger.error(f\"Error getting user by email: {str(e)}\")
    raise DatabaseError(f\"Error getting user by email: {str(e)}\") DatabaseError(f"Failed to get user by email: {str(e)}")
    async def create_user_async(self, db: AsyncSession, email: str, hashed_password: str, role: str = "user") -> User:
        try:
            user = User(
                email=email,
                hashed_password=hashed_password,
                role=role,
                is_active=True,
                created_at=datetime.datetime.now(datetime.timezone.utc)
            )
            db.add(user)
            await db.commit()
            await db.refresh(user)
            return user
        except Exception as e:
            await await await await db.rollback()
            logger.error(f"Error creating user: {str(e)}")
            raise DatabaseError(f"Failed to create user: {str(e)}")
    async def update_last_login_async(self, db: AsyncSession, user_id: int) -> Optional[User]:
        try:
            user = await await self.get_async(db, user_id)
            if not user:
                return None
            user.last_login = datetime.datetime.now(datetime.timezone.utc)
            await db.commit()
            await db.refresh(user)
            return user
        except Exception as e:
            await await await await db.rollback()
            logger.error(f"Error updating user last login: {str(e)}")
            raise DatabaseError(f"Failed to update user last login: {str(e)}")
    async def deactivate_user_async(self, db: AsyncSession, user_id: int) -> Optional[User]:
        try:
            user = await await self.get_async(db, user_id)
            if not user:
                return None
            user.is_active = False
            await db.commit()
            await db.refresh(user)
            return user
        except Exception as e:
            await await await await db.rollback()
            logger.error(f"Error deactivating user: {str(e)}")
            raise DatabaseError(f"Failed to deactivate user: {str(e)}")