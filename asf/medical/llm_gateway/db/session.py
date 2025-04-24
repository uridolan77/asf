"""
Database session for LLM Gateway.

This module provides a database session factory for the LLM Gateway.
"""

import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Set up logging
logger = logging.getLogger(__name__)

# Get database URL from environment variable
DATABASE_URL = os.environ.get("LLM_GATEWAY_DATABASE_URL", "sqlite:///./llm_gateway.db")

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    pool_pre_ping=True,
    echo=False
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

def get_db_session() -> Session:
    """
    Get a database session.
    
    Returns:
        SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        return db
    except Exception as e:
        db.close()
        logger.error(f"Error getting database session: {e}")
        raise

def init_db():
    """
    Initialize the database.
    
    This function creates all tables in the database.
    """
    try:
        # Import models to ensure they are registered with the Base
        from asf.medical.llm_gateway.models.provider import Provider, ProviderModel, ApiKey, ConnectionParameter, AuditLog
        
        # Create tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise
