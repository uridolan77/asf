"""
Debug the provider retrieval process in the LLM Gateway.

This script simulates how the LLM Gateway retrieves provider information
from the database to help identify why it can't find the openai_gpt4_default provider.
"""

import os
import sys
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

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

def debug_provider_retrieval():
    """Debug the provider retrieval process."""
    try:
        # Create engine and session
        engine = create_engine(DATABASE_URL)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        db = SessionLocal()
        
        logger.info(f"Connected to database {DB_NAME} on {DB_HOST}")
        
        # Import the necessary modules
        try:
            from asf.bo.backend.repositories.provider_repository import ProviderRepository
            from asf.bo.backend.utils.crypto import generate_key
            
            # Generate encryption key
            encryption_key = generate_key()
            
            # Initialize provider repository
            provider_repo = ProviderRepository(db, encryption_key)
            logger.info("Provider repository initialized successfully")
            
            # Get provider by ID
            provider_id = "openai_gpt4_default"
            provider = provider_repo.get_provider_by_id(provider_id)
            
            if provider:
                logger.info(f"Provider '{provider_id}' found: {provider.provider_id}, {provider.display_name}, {provider.provider_type}")
                
                # Get provider models
                models = provider_repo.get_models_by_provider_id(provider_id)
                logger.info(f"Provider models: {[model.model_id for model in models]}")
                
                # Get connection parameters
                params = provider_repo.get_connection_parameters_by_provider_id(provider_id)
                logger.info(f"Connection parameters: {[(param.param_name, param.param_value) for param in params]}")
                
                # Try to load the provider configuration
                try:
                    from asf.medical.llm_gateway.core.config_loader import ConfigLoader
                    
                    config_loader = ConfigLoader(db)
                    config = config_loader.load_from_db(provider_id)
                    
                    logger.info(f"Provider configuration loaded: {config}")
                    
                    # Check if the provider is in the configuration
                    if 'additional_config' in config and 'providers' in config['additional_config']:
                        providers = config['additional_config']['providers']
                        if provider_id in providers:
                            logger.info(f"Provider '{provider_id}' found in configuration")
                        else:
                            logger.warning(f"Provider '{provider_id}' not found in configuration providers: {list(providers.keys())}")
                    else:
                        logger.warning("No providers found in configuration")
                    
                except ImportError as e:
                    logger.error(f"Error importing ConfigLoader: {e}")
                except Exception as e:
                    logger.error(f"Error loading provider configuration: {e}")
            else:
                logger.warning(f"Provider '{provider_id}' not found in the database")
            
        except ImportError as e:
            logger.error(f"Error importing modules: {e}")
        except Exception as e:
            logger.error(f"Error retrieving provider: {e}")
        
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")

if __name__ == "__main__":
    debug_provider_retrieval()
