"""
Database connection module for the Medical Research Synthesizer.

This module provides functions for connecting to the database and managing sessions.
It includes connection pooling, retry logic, and proper transaction management.

Features:
- Async and sync database support
- Connection pooling for efficient resource usage
- Retry logic for transient database errors
- Proper transaction management with commit/rollback
- Comprehensive error handling and logging
- Support for PostgreSQL and SQLite databases

The module configures the database connection based on settings from the application
configuration, including connection pooling parameters and SQL echo mode.
"""

import time
from functools import wraps
from typing import Callable, TypeVar, Any
from sqlalchemy.orm import registry

from sqlalchemy import create_engine, select, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from sqlalchemy.pool import NullPool
from contextlib import contextmanager

from ..core.config import settings
from ..core.exceptions import DatabaseError
from ..core.logging_config import get_logger

logger = get_logger(__name__)

# Retry decorator for database operations
T = TypeVar("T")

def with_retry(
    max_retries: int = 3,
    retry_delay: float = 0.1,
    backoff_factor: float = 2,
    retryable_errors: tuple = (OperationalError,),
) -> Callable:
    """
    Decorator to retry database operations on transient errors.

    Args:
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries in seconds
        backoff_factor: Factor to increase delay with each retry
        retryable_errors: Tuple of exception types to retry on

    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        """
        Decorator function that wraps the target function with retry logic.

        This function wraps the target async function with retry logic that will
        retry the operation on specified exceptions up to the maximum number of
        retries, with exponential backoff between attempts.

        Args:
            func: The async function to wrap with retry logic

        Returns:
            A wrapped function that includes retry logic
        """
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None
            current_delay = retry_delay

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable_errors as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Retryable error on attempt {attempt + 1}/{max_retries + 1}, "
                            f"retrying in {current_delay:.2f}s",
                            extra={"error": str(e), "attempt": attempt + 1},
                            exc_info=e
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        logger.error(
                            f"Operation failed after {max_retries + 1} attempts",
                            extra={"error": str(e), "max_retries": max_retries},
                            exc_info=e
                        )
                        raise DatabaseError(
                            f"Database operation failed after {max_retries + 1} attempts",
                            details={"error": str(e), "max_retries": max_retries}
                        ) from e

            # This should never be reached, but just in case
            if last_exception:
                raise last_exception

            # This should also never be reached
            raise RuntimeError("Unexpected error in retry logic")

        return wrapper

    return decorator

# Get database URL from settings
DATABASE_URL = settings.DATABASE_URL

# Configure connection pooling parameters
POOL_SIZE = getattr(settings, "DB_POOL_SIZE", 5)
MAX_OVERFLOW = getattr(settings, "DB_MAX_OVERFLOW", 10)
POOL_TIMEOUT = getattr(settings, "DB_POOL_TIMEOUT", 30)
POOL_RECYCLE = getattr(settings, "DB_POOL_RECYCLE", 1800)  # 30 minutes
ECHO_SQL = getattr(settings, "SQL_ECHO", False)

if DATABASE_URL.startswith("postgresql+asyncpg"):
    # Create async engine with connection pooling
    engine = create_async_engine(
        DATABASE_URL,
        echo=ECHO_SQL,
        future=True,
        pool_size=POOL_SIZE,
        max_overflow=MAX_OVERFLOW,
        pool_timeout=POOL_TIMEOUT,
        pool_recycle=POOL_RECYCLE,
        pool_pre_ping=True,  # Verify connections before using them
    )

    # Create async session factory
    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False  # Don't expire objects after commit
    )

    is_async = True

    logger.info(
        f"Configured async database connection pool",
        extra={
            "pool_size": POOL_SIZE,
            "max_overflow": MAX_OVERFLOW,
            "pool_timeout": POOL_TIMEOUT,
            "pool_recycle": POOL_RECYCLE,
            "database_type": "postgresql+asyncpg"
        }
    )
else:
    # For SQLite and other non-async databases
    connect_args = {}
    if DATABASE_URL.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    # Create sync engine
    engine = create_engine(
        DATABASE_URL,
        connect_args=connect_args,
        echo=ECHO_SQL,
        future=True,
        # For SQLite, we should skip pool settings entirely rather than setting to None
        **({"pool_size": POOL_SIZE,
           "max_overflow": MAX_OVERFLOW,
           "pool_timeout": POOL_TIMEOUT,
           "pool_recycle": POOL_RECYCLE,
           "pool_pre_ping": True} if not DATABASE_URL.startswith("sqlite") else {"poolclass": NullPool})
    )

    # Create sync session factory
    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
        expire_on_commit=False
    )

    is_async = False

    logger.info(
        f"Configured sync database connection pool",
        extra={
            "pool_size": POOL_SIZE if not DATABASE_URL.startswith("sqlite") else "N/A",
            "database_type": DATABASE_URL.split(":")[0]
        }
    )

# Create a registry with a custom naming convention for models
mapper_registry = registry()

# Renamed from Base to MedicalBase to avoid conflicts with other modules
# Force SQLAlchemy to use fully qualified class paths in registry
MedicalBase = declarative_base(
    metadata=mapper_registry.metadata,
    class_registry=dict()
)

# Set __module_path__ on the MedicalBase to ensure unique model registration
MedicalBase.__module__ = "asf.medical.storage.models"

@contextmanager
def get_db():
    """
    Get a database session for synchronous operations.

    This function provides a context manager for synchronous database operations.
    It handles session creation, committing, and error handling.

    Yields:
        Session: A SQLAlchemy session

    Example:
        with get_db() as db:
            db.query(User).all()

    Raises:
        ValueError: If used with an async database
        DatabaseError: If there's an error accessing the database
    """
    if is_async:
        raise ValueError("Cannot use synchronous session with async database")

    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(
            "Error in database session",
            extra={"error": str(e)},
            exc_info=e
        )
        if isinstance(e, SQLAlchemyError):
            raise DatabaseError(f"Database error: {str(e)}", details={"error": str(e)}) from e
        raise
    finally:
        db.close()

@with_retry()
async def get_db_session():
    """
    Get a database session for asynchronous operations.

    This function provides an async context manager for database operations.
    It handles session creation, committing, and error handling with retry logic.

    Yields:
        AsyncSession: An async SQLAlchemy session

    Example:
        async with get_db_session() as db:
            result = await db.execute(select(User))
            users = result.scalars().all()

    Raises:
        DatabaseError: If there's an error accessing the database after retries
    """
    if not is_async:
        # For non-async databases, use the sync session
        with get_db() as db:
            yield db
        return

    # For async databases, use the async session with retry logic
    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(
                "Error in async database session",
                extra={"error": str(e)},
                exc_info=e
            )
            if isinstance(e, SQLAlchemyError):
                raise DatabaseError(f"Database error: {str(e)}", details={"error": str(e)}) from e
            raise

def create_tables():
    """
    Create all tables in the database.

    This function creates all tables defined in SQLAlchemy models.
    It should be called during application startup.

    Raises:
        DatabaseError: If there's an error creating the tables
    """
    try:
        MedicalBase.metadata.create_all(bind=engine)  # Updated to use MedicalBase
        logger.info("Database tables created successfully")
    except SQLAlchemyError as e:
        logger.error(
            "Error creating database tables",
            extra={"error": str(e)},
            exc_info=e
        )
        raise DatabaseError(f"Failed to create database tables: {str(e)}", details={"error": str(e)}) from e

def init_db():
    """
    Initialize the database.

    This function initializes the database by creating all tables
    and performing any other necessary setup.

    Raises:
        DatabaseError: If there's an error initializing the database
    """
    create_tables()
    logger.info("Database initialized successfully")

async def check_database_connection():
    """
    Check if the database connection is working.

    This function attempts to connect to the database and perform a simple query
    to verify that the connection is working properly.

    Returns:
        bool: True if the connection is working, False otherwise
    """
    try:
        if is_async:
            async with SessionLocal() as session:
                # For PostgreSQL
                if DATABASE_URL.startswith("postgresql"):
                    result = await session.execute("SELECT 1")
                # For SQLite
                else:
                    result = await session.execute("SELECT 1")
                await result.fetchone()
        else:
            with get_db() as session:
                # For PostgreSQL
                if DATABASE_URL.startswith("postgresql"):
                    result = session.execute("SELECT 1")
                # For SQLite
                else:
                    result = session.execute("SELECT 1")
                result.fetchone()

        logger.info("Database connection check successful")
        return True
    except Exception as e:
        logger.error(
            "Database connection check failed",
            extra={"error": str(e)},
            exc_info=e
        )
        return False
