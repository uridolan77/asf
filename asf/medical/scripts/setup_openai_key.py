"""
Set up OpenAI API key in the secret manager.

This script sets up the OpenAI API key in the secret manager.
Run this script once to store your API key securely.
"""

import logging
import sys
from pathlib import Path
import os

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from asf.medical.core.secrets import SecretManager

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_openai_api_key(api_key: str) -> None:
    """
    Store the OpenAI API key in the secret manager.
    
    Args:
        api_key: OpenAI API key
    """
    try:
        secret_manager = SecretManager()
        
        # Store the API key
        secret_manager.set_secret('llm', 'openai_api_key', api_key)
        
        logger.info("Successfully stored OpenAI API key in secret manager")
        logger.info(f"Secret location: {secret_manager.secrets_file}")
        
        # Verify we can retrieve it
        retrieved_key = secret_manager.get_secret('llm', 'openai_api_key')
        if retrieved_key == api_key:
            logger.info("Successfully verified API key retrieval")
        else:
            logger.error("Failed to verify API key retrieval")
    
    except Exception as e:
        logger.error(f"Failed to store OpenAI API key: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Check if API key is provided as argument
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
    else:
        # Get API key from environment variable or prompt
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            api_key = input("Enter your OpenAI API key: ").strip()
    
    if not api_key:
        logger.error("No API key provided")
        sys.exit(1)
    
    # Store the key in secret manager
    setup_openai_api_key(api_key)