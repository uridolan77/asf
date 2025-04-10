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

# Get database URL from environment variable or use SQLite as fallback
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./medical_research_synthesizer.db")

# Create async engine if using PostgreSQL, otherwise use sync engine
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
    # Fallback to sync SQLite for development
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

# Create base class for declarative models
Base = declarative_base()

# Session management for sync database
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

# Session dependency for FastAPI
async def get_db_session():
    """
    Get a database session for asynchronous operations.
    
    Yields:
        AsyncSession: A SQLAlchemy async session
        
    Example:
        @app.get("/users")
        async def get_users(db: AsyncSession = Depends(get_db_session)):
            result = await db.execute(select(User))
            return result.scalars().all()
    """
    if not is_async:
        # For sync database, wrap in async context
        with get_db() as db:
            yield db
        return
    
    # For async database
    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

# Create all tables
def create_tables():
    """Create all tables in the database."""
    Base.metadata.create_all(bind=engine)

# Initialize database
def init_db():
    """Initialize the database."""
    create_tables()
