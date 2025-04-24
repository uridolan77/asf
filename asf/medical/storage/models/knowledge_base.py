"""
Knowledge base model for the Medical Research Synthesizer.
This module defines SQLAlchemy models for knowledge bases.
"""
from datetime import datetime
from typing import Dict, Any, List, Optional
from sqlalchemy import Column, Integer, String, DateTime, JSON, Text, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from ..database import MedicalBase

class KnowledgeBase(MedicalBase):
    """Knowledge base model for storing medical knowledge."""
    __tablename__ = "knowledge_bases"
    
    id = Column(String(36), primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    query = Column(String(1000), nullable=False)
    update_schedule = Column(String(50), nullable=False, default="weekly")
    last_updated = Column(DateTime, nullable=True)
    next_update = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    user_id = Column(Integer, ForeignKey("medical_users.id"), nullable=True, index=True)  # Updated from "users.id"
    user = relationship("MedicalUser", back_populates="knowledge_bases")  # Updated from "User"
    data = Column(JSON, nullable=True)
    active = Column(Boolean, nullable=False, default=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the knowledge base to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the knowledge base
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "query": self.query,
            "update_schedule": self.update_schedule,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "next_update": self.next_update.isoformat() if self.next_update else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "user_id": self.user_id,
            "active": self.active
        }