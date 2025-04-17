#!/usr/bin/env python
# check_db_api_key.py

"""
Script to check if the OpenAI API key exists in the database.
"""

import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to Python path if needed
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    logger.info(f"Added {project_root} to Python path")

# Import database utilities
from asf.medical.llm_gateway.db_utils import get_db_session
from asf.bo.backend.repositories.provider_repository import ProviderRepository
from asf.bo.backend.utils.crypto import generate_key

def check_db_api_key():
    """Check if the OpenAI API key exists in the database."""
    try:
        # Get database session
        db = get_db_session()
        logger.info("Database session created successfully")
        
        # Get encryption key (in production, this should be loaded from a secure source)
        encryption_key = generate_key()
        
        # Initialize provider repository
        provider_repo = ProviderRepository(db, encryption_key)
        logger.info("Provider repository initialized successfully")
        
        # Get API keys for OpenAI provider
        provider_id = "openai_gpt4_default"
        api_keys = provider_repo.get_api_keys_by_provider_id(provider_id)
        
        if not api_keys:
            logger.error(f"No API keys found for provider '{provider_id}'")
            return False
        
        logger.info(f"Found {len(api_keys)} API key(s) for provider '{provider_id}'")
        
        # Get the first API key
        api_key = api_keys[0]
        logger.info(f"API key ID: {api_key.key_id}, Encrypted: {api_key.is_encrypted}, Environment: {api_key.environment}")
        
        # Try to decrypt the API key
        key_value = provider_repo.get_decrypted_api_key(api_key.key_id)
        if key_value:
            # Mask the API key for logging
            if len(key_value) > 8:
                masked_key = f"{key_value[:5]}...{key_value[-4:]}"
            else:
                masked_key = "***MASKED***"
            logger.info(f"Successfully decrypted API key: {masked_key}")
            return True
        else:
            logger.error(f"Failed to decrypt API key with ID {api_key.key_id}")
            return False
    
    except Exception as e:
        logger.exception(f"Error checking API key in database: {e}")
        return False
    finally:
        # Close the database session
        if 'db' in locals():
            db.close()
            logger.info("Database session closed")

if __name__ == "__main__":
    logger.info("Checking OpenAI API key in database...")
    
    success = check_db_api_key()
    
    if success:
        logger.info("✅ API key check PASSED!")
        sys.exit(0)
    else:
        logger.error("❌ API key check FAILED!")
        sys.exit(1)
