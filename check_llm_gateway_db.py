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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_db_utils():
    """Check the database utilities used by the LLM Gateway."""
    try:
        # Try to import the db_utils module
        db_utils_path = Path("asf/medical/llm_gateway/db_utils.py")
        if db_utils_path.exists():
            logger.info(f"Found db_utils module at {db_utils_path}")
            
            # Load the module
            spec = importlib.util.spec_from_file_location("db_utils", db_utils_path)
            db_utils = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(db_utils)
            
            # Check the DATABASE_URL
            logger.info(f"DATABASE_URL: {db_utils.DATABASE_URL}")
            
            # Try to get a database session
            try:
                db = db_utils.get_db_session()
                logger.info("Successfully created database session")
                
                # Check if the provider exists
                try:
                    result = db.execute("SELECT * FROM providers WHERE provider_id = 'openai_gpt4_default'").fetchone()
                    if result:
                        logger.info(f"Provider 'openai_gpt4_default' found in the database: {result}")
                    else:
                        logger.warning("Provider 'openai_gpt4_default' not found in the database")
                except Exception as e:
                    logger.error(f"Error querying provider: {e}")
                
                # Close the session
                db.close()
                
            except Exception as e:
                logger.error(f"Error creating database session: {e}")
        else:
            logger.warning(f"db_utils module not found at {db_utils_path}")
    except Exception as e:
        logger.error(f"Error checking db_utils: {e}")

def check_database_module():
    """Check the database module used by the application."""
    try:
        # Try to import the database module
        database_path = Path("asf/medical/storage/database.py")
        if database_path.exists():
            logger.info(f"Found database module at {database_path}")
            
            # Load the module
            spec = importlib.util.spec_from_file_location("database", database_path)
            database = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(database)
            
            # Check the DATABASE_URL
            logger.info(f"DATABASE_URL: {database.DATABASE_URL}")
            
            # Try to get a database session
            try:
                with database.get_db() as db:
                    logger.info("Successfully created database session")
                    
                    # Check if the provider exists
                    try:
                        from sqlalchemy import text
                        result = db.execute(text("SELECT * FROM providers WHERE provider_id = 'openai_gpt4_default'")).fetchone()
                        if result:
                            logger.info(f"Provider 'openai_gpt4_default' found in the database: {result}")
                        else:
                            logger.warning("Provider 'openai_gpt4_default' not found in the database")
                    except Exception as e:
                        logger.error(f"Error querying provider: {e}")
            except Exception as e:
                logger.error(f"Error creating database session: {e}")
        else:
            logger.warning(f"database module not found at {database_path}")
    except Exception as e:
        logger.error(f"Error checking database module: {e}")

if __name__ == "__main__":
    check_db_utils()
    check_database_module()
