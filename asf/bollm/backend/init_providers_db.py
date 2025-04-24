"""
Initialize the LLM providers database.

This script creates the llm_providers table and populates it with initial data.
"""

import logging
import argparse
import json
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from asf.bollm.backend.config.config import SQLALCHEMY_DATABASE_URL
from asf.bollm.backend.repositories.provider_dao import ProviderDAO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_provider_table():
    """Create the llm_providers table if it doesn't exist."""
    try:
        # Create engine and session
        engine = create_engine(SQLALCHEMY_DATABASE_URL)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        # Create table
        with SessionLocal() as db:
            db.execute(text("""
            CREATE TABLE IF NOT EXISTS llm_providers (
                provider_id VARCHAR(255) PRIMARY KEY,
                provider_type VARCHAR(255) NOT NULL,
                display_name VARCHAR(255) NOT NULL,
                description TEXT,
                enabled BOOLEAN DEFAULT TRUE,
                connection_params TEXT,
                request_settings TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """))
            db.commit()
            
        logger.info("llm_providers table created successfully")
        return True
    except Exception as e:
        logger.error(f"Error creating llm_providers table: {e}")
        return False

def insert_initial_providers():
    """Insert initial providers into the database."""
    try:
        # Create DAO
        provider_dao = ProviderDAO()
        
        # Define initial providers
        initial_providers = [
            {
                "provider_id": "openai",
                "provider_type": "openai",
                "display_name": "OpenAI",
                "description": "OpenAI API provider for GPT models",
                "enabled": True,
                "connection_params": {
                    "api_key": "",  # Will be set by environment variable
                    "base_url": "https://api.openai.com/v1",
                    "auth_type": "api_key"
                },
                "request_settings": {
                    "timeout_seconds": 60,
                    "retry_attempts": 3,
                    "rate_limit_rpm": 60
                }
            },
            {
                "provider_id": "anthropic",
                "provider_type": "anthropic",
                "display_name": "Anthropic",
                "description": "Anthropic API provider for Claude models",
                "enabled": True,
                "connection_params": {
                    "api_key": "",  # Will be set by environment variable
                    "base_url": "https://api.anthropic.com",
                    "auth_type": "api_key"
                },
                "request_settings": {
                    "timeout_seconds": 60,
                    "retry_attempts": 3,
                    "rate_limit_rpm": 60
                }
            },
            {
                "provider_id": "biomed",
                "provider_type": "biomedlm",
                "display_name": "BioMedLM",
                "description": "Local BioMedLM model provider",
                "enabled": True,
                "connection_params": {
                    "model_path": "",  # Will be set by environment variable
                    "use_gpu": True
                },
                "request_settings": {
                    "timeout_seconds": 120,
                    "retry_attempts": 1,
                    "rate_limit_rpm": 10
                }
            }
        ]
        
        # Insert providers
        for provider_data in initial_providers:
            # Check if provider already exists
            existing_provider = provider_dao.get_provider_by_id(provider_data["provider_id"])
            
            if existing_provider:
                logger.info(f"Provider {provider_data['provider_id']} already exists, skipping")
                continue
            
            # Create provider
            provider = provider_dao.create_provider(provider_data)
            
            if provider:
                logger.info(f"Provider {provider_data['provider_id']} created successfully")
            else:
                logger.error(f"Failed to create provider {provider_data['provider_id']}")
        
        logger.info("Initial providers inserted successfully")
        return True
    except Exception as e:
        logger.error(f"Error inserting initial providers: {e}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Initialize the LLM providers database")
    parser.add_argument("--create-table", action="store_true", help="Create the llm_providers table")
    parser.add_argument("--insert-providers", action="store_true", help="Insert initial providers")
    parser.add_argument("--all", action="store_true", help="Perform all initialization steps")
    
    args = parser.parse_args()
    
    # Default to all if no arguments provided
    if not (args.create_table or args.insert_providers or args.all):
        args.all = True
    
    # Create table
    if args.create_table or args.all:
        if create_provider_table():
            logger.info("Table creation completed successfully")
        else:
            logger.error("Table creation failed")
    
    # Insert initial providers
    if args.insert_providers or args.all:
        if insert_initial_providers():
            logger.info("Provider insertion completed successfully")
        else:
            logger.error("Provider insertion failed")
    
    logger.info("Initialization completed")

if __name__ == "__main__":
    main()
