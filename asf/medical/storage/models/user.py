from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import datetime

Base = declarative_base()

class Role(Base):
    __tablename__ = "roles"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), unique=True, index=True)
    description = Column(String(255))
    
    # Relationship with users
    users = relationship("User", back_populates="role_obj")

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    
    # Change from a simple string to a foreign key relationship
    role_id = Column(Integer, ForeignKey("roles.id"))
    role = Column(String, default="user")  # Keep for backward compatibility
    role_obj = relationship("Role", back_populates="users")
    
    # Add relationship with knowledge bases
    knowledge_bases = relationship("KnowledgeBase", back_populates="user")
    
    # Add relationship with tasks
    tasks = relationship("Task", back_populates="user")
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
