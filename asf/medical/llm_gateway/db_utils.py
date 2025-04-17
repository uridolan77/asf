"""
Database utilities for the LLM Gateway.

This module provides functions to connect to the database and get a database session.
"""

import os
import logging
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base

logger = logging.getLogger(__name__)

# Get database URL from environment variable or use default
DATABASE_URL = os.environ.get("DATABASE_URL", "mysql+pymysql://root:Dt%g_9W3z0*!I@localhost/bo_admin")

# Create engine and session factory
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db() -> Generator[Session, None, None]:
    """
    Get a database session.

    Yields:
        SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_db_session() -> Session:
    """
    Get a database session (non-generator version).

    Returns:
        SQLAlchemy database session
    """
    return SessionLocal()
