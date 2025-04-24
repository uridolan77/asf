"""
Check if providers are properly loaded from the database.

This script connects to the database and retrieves all providers.
"""

import logging
import sys
import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from asf.bollm.backend.repositories.provider_dao import ProviderDAO
from asf.bollm.backend.config.config import SQLALCHEMY_DATABASE_URL

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_providers_table_exists():
    """Check if the llm_providers table exists."""
    try:
        # Create engine and session
        engine = create_engine(SQLALCHEMY_DATABASE_URL)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        # Check if table exists
        with SessionLocal() as db:
            from sqlalchemy import inspect
            inspector = inspect(db.get_bind())
            if inspector.has_table("llm_providers"):
                logger.info("llm_providers table exists")
                return True
            else:
                logger.error("llm_providers table does not exist")
                return False
    except Exception as e:
        logger.error(f"Error checking if llm_providers table exists: {e}")
        return False

def get_providers():
    """Get all providers from the database."""
    try:
        # Create DAO
        provider_dao = ProviderDAO()
        
        # Get all providers
        providers = provider_dao.get_all_providers()
        
        # Print providers
        print(f"Providers in database: {len(providers)}")
        for provider in providers:
            print(f"Provider: {provider['display_name']} (ID: {provider['provider_id']}, Type: {provider['provider_type']})")
            
            # Get models for this provider
            models = provider_dao.get_models_by_provider_id(provider['provider_id'])
            print(f"  Models: {len(models)}")
            for model in models:
                print(f"  - {model['model_id']} ({model['display_name']})")
            
            print()
        
        return providers
    except Exception as e:
        logger.error(f"Error getting providers: {e}")
        return []

def main():
    """Main function."""
    # Check if table exists
    if not check_providers_table_exists():
        logger.error("Please run init_providers_db.py to create the llm_providers table")
        return
    
    # Get providers
    providers = get_providers()
    
    if not providers:
        logger.warning("No providers found in the database")
        logger.info("Please run init_providers_db.py to insert initial providers")

if __name__ == "__main__":
    main()
