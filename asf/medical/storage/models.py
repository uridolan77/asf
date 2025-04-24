"""
Database models for the Medical Research Synthesizer.

This module defines the SQLAlchemy ORM models for the database. These models
represent the database schema and provide a type-safe interface for interacting
with the database through the SQLAlchemy ORM.

Models:
- User: User model for authentication and authorization
- Query: Query model for storing search queries
- Result: Result model for storing search results and analyses
- KnowledgeBase: Knowledge base model for storing knowledge base metadata
- KnowledgeBaseUpdate: Model for tracking knowledge base updates
- Contradiction: Model for storing detected contradictions
"""
from sqlalchemy import Column
from ..core.exceptions import Integer, String, Float, DateTime, ForeignKey, Boolean, JSON, Text
from sqlalchemy.orm import relationship
import datetime
from database import Base
class User(Base):
    """
    User model for authentication and authorization.

    This model stores user information for authentication, authorization, and
    user management. It includes fields for email, password, role, and activity
    status, as well as relationships to other models.
    """
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="user", nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    last_login = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    queries = relationship("Query", back_populates="user")
    results = relationship("Result", back_populates="user")
    knowledge_bases = relationship("KnowledgeBase", back_populates="user")
    tasks = relationship("Task", back_populates="user")
class Query(Base):
    """
    Query model for storing search queries.

    This model stores search queries submitted by users, including the query text,
    query type, and any parameters. It has relationships to the User and Result models.
    """
    __tablename__ = "queries"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    query_text = Column(String, nullable=False)
    query_type = Column(String, default="text", nullable=False)  # text, pico, template
    parameters = Column(JSON, nullable=True)  # For PICO parameters
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    user = relationship("User", back_populates="queries")
    results = relationship("Result", back_populates="query")
class Result(Base):
    """
    Result model for storing search results and analyses.

    This model stores the results of search queries and analyses, including the
    result data in JSON format. It has relationships to the Query and User models.
    """
    __tablename__ = "results"
    id = Column(Integer, primary_key=True, index=True)
    result_id = Column(String, unique=True, index=True, nullable=False)  # UUID for API reference
    query_id = Column(Integer, ForeignKey("queries.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    result_type = Column(String, nullable=False)  # search, analysis
    result_data = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    query = relationship("Query", back_populates="results")
    user = relationship("User", back_populates="results")
class KnowledgeBase(Base):
    """
    KnowledgeBase model for storing knowledge base metadata.

    This model stores metadata about knowledge bases, including the name, query,
    file path, update schedule, and statistics. It has relationships to the User
    and KnowledgeBaseUpdate models.
    """
    __tablename__ = "knowledge_bases"
    id = Column(Integer, primary_key=True, index=True)
    kb_id = Column(String, unique=True, index=True, nullable=False)  # UUID for API reference
    name = Column(String, unique=True, index=True, nullable=False)
    query = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    update_schedule = Column(String, nullable=False)  # daily, weekly, monthly
    last_updated = Column(DateTime, nullable=True)
    next_update = Column(DateTime, nullable=True)
    initial_results = Column(Integer, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    user = relationship("User", back_populates="knowledge_bases")
    updates = relationship("KnowledgeBaseUpdate", back_populates="knowledge_base")
class KnowledgeBaseUpdate(Base):
    """
    KnowledgeBaseUpdate model for tracking knowledge base updates.

    This model tracks updates to knowledge bases, including the update time,
    number of new results, total results, status, and any error messages.
    It has a relationship to the KnowledgeBase model.
    """
    __tablename__ = "knowledge_base_updates"
    id = Column(Integer, primary_key=True, index=True)
    kb_id = Column(Integer, ForeignKey("knowledge_bases.id"), nullable=False)
    update_time = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    new_results = Column(Integer, nullable=False)
    total_results = Column(Integer, nullable=False)
    status = Column(String, nullable=False)  # success, failure
    error_message = Column(Text, nullable=True)
    knowledge_base = relationship("KnowledgeBase", back_populates="updates")
class Contradiction(Base):
    """
    Contradiction model for storing detected contradictions.

    This model stores information about detected contradictions between publications,
    including the contradiction score, confidence, detection method, topic, and
    explanation. It is related to the Result model through the result_id field.
    """
    __tablename__ = "contradictions"
    id = Column(Integer, primary_key=True, index=True)
    result_id = Column(Integer, ForeignKey("results.id"), nullable=False)
    publication1_pmid = Column(String, nullable=False)
    publication2_pmid = Column(String, nullable=False)
    contradiction_score = Column(Float, nullable=False)
    confidence = Column(String, nullable=False)  # high, medium, low
    detection_method = Column(String, nullable=False)  # keyword, biomedlm, tsmixer
    topic = Column(String, nullable=True)
    explanation = Column(JSON, nullable=True)  # SHAP explanation
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    __table_args__ = (
    )