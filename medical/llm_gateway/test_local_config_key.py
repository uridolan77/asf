#!/usr/bin/env python
# test_local_config_key.py

"""
Test script to verify the OpenAI API key is correctly loaded from the local configuration.
"""

import os
import sys
import logging
import yaml
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

def test_local_config_key():
    """Test that the OpenAI API key is correctly loaded from the local configuration."""
    try:
        # Get configuration directory
        config_dir = os.path.join(project_root, "bo", "backend", "config", "llm")
        local_config_path = os.path.join(config_dir, "llm_gateway_config.local.yaml")
        
        if not os.path.exists(local_config_path):
            logger.error(f"Local configuration file not found at {local_config_path}")
            return False
        
        logger.info(f"Loading local configuration from {local_config_path}")
        with open(local_config_path, 'r') as f:
            local_config = yaml.safe_load(f)
        
        # Extract API key from configuration
        providers = local_config.get("additional_config", {}).get("providers", {})
        openai_provider = providers.get("openai_gpt4_default", {})
        api_key = openai_provider.get("connection_params", {}).get("api_key")
        
        if api_key:
            # Mask the API key for logging
            if len(api_key) > 8:
                masked_key = f"{api_key[:5]}...{api_key[-4:]}"
            else:
                masked_key = "***MASKED***"
            logger.info(f"Successfully retrieved API key from local configuration: {masked_key}")
            return True
        else:
            logger.error("API key not found in local configuration")
            return False
    
    except Exception as e:
        logger.exception(f"Error testing API key: {e}")
        return False

if __name__ == "__main__":
    logger.info("Testing OpenAI API key in local configuration...")
    
    success = test_local_config_key()
    
    if success:
        logger.info("✅ API key test PASSED!")
        sys.exit(0)
    else:
        logger.error("❌ API key test FAILED!")
        sys.exit(1)
