"""
Check the provider configuration in the database.

This script connects to the database and checks if the openai_gpt4_default provider
can be properly retrieved.
"""

import os
import sys
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database connection settings
DB_USER = os.getenv('BO_DB_USER', 'root')
DB_PASSWORD = os.getenv('BO_DB_PASSWORD', 'Dt%g_9W3z0*!I')
DB_HOST = os.getenv('BO_DB_HOST', 'localhost')
DB_PORT = os.getenv('BO_DB_PORT', '3306')
DB_NAME = os.getenv('BO_DB_NAME', 'bo_admin')

# Create database URL
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

def check_provider():
    """Check if the openai_gpt4_default provider exists in the database."""
    try:
        # Create engine and session
        engine = create_engine(DATABASE_URL)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        db = SessionLocal()
        
        logger.info(f"Connected to database {DB_NAME} on {DB_HOST}")
        
        # Check if the provider exists
        query = text("SELECT * FROM providers WHERE provider_id = :provider_id")
        result = db.execute(query, {"provider_id": "openai_gpt4_default"}).fetchone()
        
        if result:
            logger.info(f"Provider 'openai_gpt4_default' found in the database: {result}")
            
            # Check provider models
            query = text("SELECT * FROM provider_models WHERE provider_id = :provider_id")
            models = db.execute(query, {"provider_id": "openai_gpt4_default"}).fetchall()
            logger.info(f"Provider models: {models}")
            
            # Check connection parameters
            query = text("SELECT * FROM connection_parameters WHERE provider_id = :provider_id")
            params = db.execute(query, {"provider_id": "openai_gpt4_default"}).fetchall()
            logger.info(f"Connection parameters: {params}")
            
            return True
        else:
            logger.warning("Provider 'openai_gpt4_default' not found in the database")
            return False
    except Exception as e:
        logger.error(f"Error checking provider: {e}")
        return False

if __name__ == "__main__":
    check_provider()
