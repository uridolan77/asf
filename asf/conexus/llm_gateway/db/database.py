"""
Database module for the Conexus LLM Gateway.

This module provides database connection management and common
database operations for the LLM Gateway.
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from asf.conexus.llm_gateway.config.settings import get_settings

logger = logging.getLogger(__name__)

# Base class for all SQLAlchemy models
Base = declarative_base()

# Engine and session factory, initialized on startup
_engine = None
_async_session_factory = None


async def initialize_database() -> None:
    """
    Initialize database engine and session factory.
    
    This should be called once during application startup.
    """
    global _engine, _async_session_factory
    
    settings = get_settings()
    
    if not settings.database.enabled:
        logger.info("Database is disabled in settings")
        return
    
    # Convert SQLite URL to async format if needed
    db_url = settings.database.url
    if db_url.startswith('sqlite:'):
        # For SQLite, convert to async driver
        db_url = db_url.replace('sqlite:', 'sqlite+aiosqlite:')
    
    # Create async engine
    _engine = create_async_engine(
        db_url,
        pool_size=settings.database.pool_size,
        max_overflow=settings.database.max_overflow,
        pool_recycle=settings.database.pool_recycle,
        pool_pre_ping=True,
        echo=settings.mode.debug_mode,
    )
    
    # Create session factory
    _async_session_factory = sessionmaker(
        _engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    # Create all tables
    async with _engine.begin() as conn:
        if settings.mode.debug_mode:
            logger.debug("Creating database tables")
            await conn.run_sync(Base.metadata.create_all)
    
    logger.info(f"Database initialized with engine {_engine}")


def get_db_engine():
    """
    Get the SQLAlchemy engine.
    
    Returns:
        The SQLAlchemy engine
    
    Raises:
        RuntimeError: If database is not initialized
    """
    if _engine is None:
        raise RuntimeError("Database engine not initialized")
    
    return _engine


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get a database session.
    
    This is an async context manager that provides a database session.
    
    Yields:
        An async database session
        
    Raises:
        RuntimeError: If database is not initialized
    """
    if _async_session_factory is None:
        raise RuntimeError("Database not initialized")
    
    session = _async_session_factory()
    try:
        yield session
    finally:
        await session.close()


async def get_db_status() -> Dict[str, Any]:
    """
    Get database status information.
    
    Returns:
        A dictionary with status information
    """
    settings = get_settings()
    
    if not settings.database.enabled:
        return {
            "enabled": False,
            "connected": False,
            "message": "Database disabled in settings"
        }
    
    if _engine is None:
        return {
            "enabled": True,
            "connected": False,
            "message": "Database not initialized"
        }
    
    try:
        # Check if we can connect to the database
        start_time = time.time()
        
        # Execute a simple query to check connection
        async with _engine.begin() as conn:
            await conn.execute(sa.text("SELECT 1"))
        
        response_time = time.time() - start_time
        
        return {
            "enabled": True,
            "connected": True,
            "message": "Database connection successful",
            "response_time_ms": round(response_time * 1000, 2)
        }
    
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        return {
            "enabled": True,
            "connected": False,
            "message": f"Database connection error: {str(e)}"
        }


# --- Common database utility functions ---

async def execute_query(query: sa.sql.Select) -> List[Dict[str, Any]]:
    """
    Execute a SQLAlchemy select query and return the results.
    
    Args:
        query: SQLAlchemy select query
        
    Returns:
        List of result rows as dictionaries
        
    Raises:
        RuntimeError: If database is not initialized
    """
    if _engine is None:
        raise RuntimeError("Database not initialized")
    
    async with get_db_session() as session:
        result = await session.execute(query)
        # Convert results to dictionaries
        rows = result.mappings().all()
        return [dict(row) for row in rows]


async def execute_statement(statement: sa.sql.Executable) -> int:
    """
    Execute a SQLAlchemy statement and return the number of affected rows.
    
    Args:
        statement: SQLAlchemy statement
        
    Returns:
        Number of affected rows
        
    Raises:
        RuntimeError: If database is not initialized
    """
    if _engine is None:
        raise RuntimeError("Database not initialized")
    
    async with get_db_session() as session:
        result = await session.execute(statement)
        await session.commit()
        
        # Return rowcount if available
        return result.rowcount if hasattr(result, 'rowcount') else 0