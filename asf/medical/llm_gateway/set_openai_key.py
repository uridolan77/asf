#!/usr/bin/env python
# set_openai_key.py

"""
Script to set the OpenAI API key as an environment variable.
"""

import os
import sys
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def set_openai_key(api_key):
    """Set the OpenAI API key as an environment variable."""
    try:
        # Set the environment variable
        os.environ["OPENAI_API_KEY"] = api_key
        
        # Verify it was set correctly
        if os.environ.get("OPENAI_API_KEY") == api_key:
            # Mask the API key for logging
            if len(api_key) > 8:
                masked_key = f"{api_key[:5]}...{api_key[-4:]}"
            else:
                masked_key = "***MASKED***"
            logger.info(f"Successfully set OPENAI_API_KEY environment variable: {masked_key}")
            return True
        else:
            logger.error("Failed to set OPENAI_API_KEY environment variable")
            return False
    
    except Exception as e:
        logger.exception(f"Error setting API key: {e}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Set the OpenAI API key as an environment variable")
    parser.add_argument("api_key", help="OpenAI API key")
    args = parser.parse_args()
    
    success = set_openai_key(args.api_key)
    
    if success:
        logger.info("✅ API key set successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Failed to set API key!")
        sys.exit(1)

if __name__ == "__main__":
    main()
