from sqlalchemy import Column, Integer, String, Boolean, DateTime, Enum
import datetime
import enum
from sqlalchemy.orm import relationship

# Import MedicalBase from database instead of creating a new one
from ..database import MedicalBase

class Role(str, enum.Enum):
    """User role enumeration."""
    ADMIN = "admin"
    USER = "user"
    RESEARCHER = "researcher"

class MedicalUser(MedicalBase):
    """
    Medical user model. 
    
    This model is explicitly qualified as MedicalUser to avoid
    conflicts with the User model in the bollm module.
    """
    __tablename__ = "medical_users"
    __mapper_args__ = {"polymorphic_identity": "medical_user"}

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    role = Column(Enum(Role), default=Role.USER)
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc),
                        onupdate=lambda: datetime.datetime.now(datetime.timezone.utc))
    
    # Define relationships with other tables - using strings instead of direct references
    # to avoid circular import issues
    tasks = relationship("Task", back_populates="user", cascade="all, delete-orphan")
    knowledge_bases = relationship("KnowledgeBase", back_populates="user", cascade="all, delete-orphan")
