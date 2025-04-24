"""
Query model for the Medical Research Synthesizer.
This module defines SQLAlchemy models for storing search queries.
"""
from datetime import datetime
from typing import Dict, Any, Optional
from sqlalchemy import Column, Integer, String, DateTime, JSON, Text, ForeignKey
from sqlalchemy.orm import relationship
from ..database import MedicalBase

class Query(MedicalBase):
    """Query model for storing search queries."""
    __tablename__ = "queries"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("medical_users.id"), nullable=True, index=True)  # Updated from "users.id"
    user = relationship("MedicalUser")  # Updated from "User"
    query_text = Column(Text, nullable=False)
    query_type = Column(String(50), nullable=False, default="text")  # text, pico, etc.
    parameters = Column(JSON, nullable=True)  # Additional parameters used for the search
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship with results
    results = relationship("Result", back_populates="query")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the query to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the query
        """
        return {
            "id": self.id,
            "user_id": self.user_id,
            "query_text": self.query_text,
            "query_type": self.query_type,
            "parameters": self.parameters,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }

class Result(MedicalBase):
    """Result model for storing search results."""
    __tablename__ = "results"
    
    id = Column(Integer, primary_key=True, index=True)
    result_id = Column(String(100), nullable=False, index=True, unique=True)
    user_id = Column(Integer, ForeignKey("medical_users.id"), nullable=True, index=True)  # Updated from "users.id"
    user = relationship("MedicalUser")  # Updated from "User"
    query_id = Column(Integer, ForeignKey("queries.id"), nullable=True, index=True)
    query = relationship("Query", back_populates="results")
    result_type = Column(String(50), nullable=False, default="search")  # search, analysis, etc.
    result_data = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the result to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the result
        """
        return {
            "id": self.id,
            "result_id": self.result_id,
            "user_id": self.user_id,
            "query_id": self.query_id,
            "result_type": self.result_type,
            "result_data": self.result_data,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }