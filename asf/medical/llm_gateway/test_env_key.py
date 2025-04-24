#!/usr/bin/env python
# test_env_key.py

"""
Simple test script to verify the OpenAI API key is correctly set as an environment variable.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_env_key():
    """Test that the OpenAI API key is correctly set as an environment variable."""
    try:
        # Get the API key from the environment
        api_key = os.environ.get("OPENAI_API_KEY")
        
        if api_key:
            # Mask the API key for logging
            if len(api_key) > 8:
                masked_key = f"{api_key[:5]}...{api_key[-4:]}"
            else:
                masked_key = "***MASKED***"
            logger.info(f"Successfully retrieved API key from environment: {masked_key}")
            return True
        else:
            logger.error("API key not found in environment")
            return False
    
    except Exception as e:
        logger.exception(f"Error testing API key: {e}")
        return False

if __name__ == "__main__":
    logger.info("Testing OpenAI API key in environment...")
    
    success = test_env_key()
    
    if success:
        logger.info("✅ API key test PASSED!")
        sys.exit(0)
    else:
        logger.error("❌ API key test FAILED!")
        sys.exit(1)
