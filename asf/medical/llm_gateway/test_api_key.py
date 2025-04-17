#!/usr/bin/env python
# test_api_key.py

"""
Simple test script to verify the OpenAI API key is correctly set up in the Secret Manager.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to Python path if needed
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.info(f"Added {project_root} to Python path")

# Import secret manager
from asf.medical.core.secrets import SecretManager

def test_api_key():
    """Test that the OpenAI API key is correctly set up in the Secret Manager."""
    try:
        # Initialize Secret Manager
        secret_manager = SecretManager()
        
        # Get the API key from the Secret Manager
        api_key = secret_manager.get_secret("llm", "openai_api_key")
        
        if api_key:
            # Mask the API key for logging
            if len(api_key) > 8:
                masked_key = f"{api_key[:5]}...{api_key[-4:]}"
            else:
                masked_key = "***MASKED***"
            logger.info(f"Successfully retrieved API key from Secret Manager: {masked_key}")
            return True
        else:
            logger.error("API key not found in Secret Manager")
            return False
    
    except Exception as e:
        logger.exception(f"Error testing API key: {e}")
        return False

if __name__ == "__main__":
    logger.info("Testing OpenAI API key in Secret Manager...")
    
    success = test_api_key()
    
    if success:
        logger.info("✅ API key test PASSED!")
        sys.exit(0)
    else:
        logger.error("❌ API key test FAILED!")
        sys.exit(1)
