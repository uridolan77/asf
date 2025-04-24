"""
Initialize the database tables for user authentication.

This script creates the necessary database tables for storing users and roles.
"""

import os
import sys
import logging
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from passlib.context import CryptContext

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from asf.bollm.backend.models.base import Base
from asf.bollm.backend.models.user import User, Role
from asf.bollm.backend.models.association import users_providers

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password):
    return pwd_context.hash(password)

def init_db():
    """Initialize the database tables."""
    # Get database URL from environment variables or use defaults
    DB_USER = os.getenv('BO_DB_USER', 'root')
    DB_PASSWORD = os.getenv('BO_DB_PASSWORD', 'Dt%g_9W3z0*!I')
    DB_HOST = os.getenv('BO_DB_HOST', 'localhost')
    DB_PORT = os.getenv('BO_DB_PORT', '3306')
    DB_NAME = os.getenv('BO_DB_NAME', 'bo_admin')
    
    # Create database URL
    db_url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    
    # Print the database URL being used
    logger.info(f"Using database URL: {db_url}")
    
    try:
        # Create engine
        engine = create_engine(db_url)
        
        # Check if tables already exist
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()
        
        tables_to_create = [
            Role.__tablename__,
            User.__tablename__,
            'users_providers'  # Association table
        ]
        
        # Filter out tables that already exist
        tables_to_create = [table for table in tables_to_create if table not in existing_tables]
        
        if not tables_to_create:
            logger.info("All user authentication tables already exist. No new tables created.")
            return
        
        # Create only the tables that don't exist yet
        logger.info(f"Creating database tables for user authentication: {', '.join(tables_to_create)}")
        
        # Create tables
        Base.metadata.create_all(engine, tables=[
            Base.metadata.tables[table] for table in tables_to_create
        ])
        
        # Create session
        Session = sessionmaker(bind=engine)
        session = Session()
        
        try:
            # Check if tables were created
            for table in tables_to_create:
