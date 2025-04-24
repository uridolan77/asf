"""
Check the database connection used by the LLM Gateway.

This script checks the database connection settings used by the LLM Gateway
and verifies that the provider information can be properly retrieved.
"""

import os
import sys
import logging
import importlib.util
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_db_connection():
    """Check the database connection used by the LLM Gateway."""
    try:
        # Try to import the db_utils module
        db_utils_path = Path("asf/medical/llm_gateway/db_utils.py")
        if db_utils_path.exists():
            logger.info(f"Found db_utils module at {db_utils_path}")
            
            # Load the module
            spec = importlib.util.spec_from_file_location("db_utils", db_utils_path)
            db_utils = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(db_utils)
            
            # Get the DATABASE_URL
            DATABASE_URL = db_utils.DATABASE_URL
            logger.info(f"DATABASE_URL: {DATABASE_URL}")
            
            # Create engine and session
            engine = create_engine(DATABASE_URL)
            SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
            db = SessionLocal()
            
            logger.info("Successfully created database session")
            
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
                
                # Check configurations
                query = text("SELECT * FROM configurations WHERE config_key LIKE 'llm_gateway.%'")
                configs = db.execute(query).fetchall()
                logger.info(f"LLM Gateway configurations: {configs}")
                
                return True
            else:
                logger.warning("Provider 'openai_gpt4_default' not found in the database")
                return False
        else:
            logger.warning(f"db_utils module not found at {db_utils_path}")
            return False
    except Exception as e:
        logger.error(f"Error checking database connection: {e}")
        return False

if __name__ == "__main__":
    check_db_connection()
