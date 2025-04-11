"""
Database connection module for the Medical Research Synthesizer.

This module provides functions for connecting to the database and managing sessions.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from contextlib import contextmanager

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./medical_research_synthesizer.db")

if DATABASE_URL.startswith("postgresql+asyncpg"):
    engine = create_async_engine(
        DATABASE_URL,
        echo=False,
        future=True
    )
    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
        class_=AsyncSession
    )
    is_async = True
else:
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
        echo=False,
        future=True
    )
    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine
    )
    is_async = False

Base = declarative_base()

@contextmanager
def get_db():
    """
    Get a database session for synchronous operations.
    
    Yields:
        Session: A SQLAlchemy session
        
    Example:
        with get_db() as db:
            db.query(User).all()
    """
    if is_async:
        raise ValueError("Cannot use synchronous session with async database")
    
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

async def get_db_session():
    if not is_async:
        with get_db() as db:
            yield db
        return
    
    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

def create_tables():
    """Create all tables in the database.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
    Base.metadata.create_all(bind=engine)

def init_db():
    """Initialize the database.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
    create_tables()
