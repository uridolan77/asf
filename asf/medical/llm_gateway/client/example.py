"""
Example usage of the LLM Gateway API client.

This script demonstrates how to use the LLM Gateway API client to interact with the LLM Gateway API.
"""

import os
import sys
import logging
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from asf.medical.llm_gateway.client.api_client import LLMGatewayClient

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    """
    Main function for the example script.
    """
    # Get API URL and key from environment variables
    api_url = os.environ.get("LLM_GATEWAY_API_URL", "http://localhost:8000")
    api_key = os.environ.get("LLM_GATEWAY_API_KEY")
    
    # Create API client
    client = LLMGatewayClient(api_url, api_key)
    
    # Example 1: Get all providers
    logger.info("Getting all providers...")
    providers = client.get_providers()
    logger.info(f"Found {len(providers)} providers")
    
    # Example 2: Create a new provider
    logger.info("Creating a new provider...")
    new_provider = {
        "provider_id": "anthropic",
        "display_name": "Anthropic",
        "provider_type": "anthropic",
        "description": "Anthropic Claude models",
        "enabled": True,
        "connection_params": {
            "api_base": "https://api.anthropic.com"
        },
        "models": [
            {
                "model_id": "claude-3-opus-20240229",
                "display_name": "Claude 3 Opus",
                "model_type": "chat",
                "context_window": 200000,
                "max_tokens": 4096
            }
        ],
        "api_key": {
            "key_value": "your-api-key",
            "is_encrypted": True
        }
    }
    try:
        created_provider = client.create_provider(new_provider)
        logger.info(f"Created provider: {created_provider['provider_id']}")
    except Exception as e:
        logger.error(f"Error creating provider: {e}")
    
    # Example 3: Get a provider by ID
    logger.info("Getting provider by ID...")
    try:
        provider = client.get_provider("anthropic")
        logger.info(f"Got provider: {provider['provider_id']}")
    except Exception as e:
        logger.error(f"Error getting provider: {e}")
    
    # Example 4: Update a provider
    logger.info("Updating provider...")
    update_data = {
        "display_name": "Anthropic AI",
        "enabled": True
    }
    try:
        updated_provider = client.update_provider("anthropic", update_data)
        logger.info(f"Updated provider: {updated_provider['provider_id']}")
    except Exception as e:
        logger.error(f"Error updating provider: {e}")
    
    # Example 5: Get all models for a provider
    logger.info("Getting models for provider...")
    try:
        models = client.get_models("anthropic")
        logger.info(f"Found {len(models)} models for provider 'anthropic'")
    except Exception as e:
        logger.error(f"Error getting models: {e}")
    
    # Example 6: Create a new model
    logger.info("Creating a new model...")
    new_model = {
        "model_id": "claude-3-haiku-20240307",
        "display_name": "Claude 3 Haiku",
        "model_type": "chat",
        "context_window": 200000,
        "max_tokens": 4096,
        "enabled": True
    }
    try:
        created_model = client.create_model("anthropic", new_model)
        logger.info(f"Created model: {created_model['model_id']}")
    except Exception as e:
        logger.error(f"Error creating model: {e}")
    
    # Example 7: Update a model
    logger.info("Updating model...")
    update_data = {
        "display_name": "Claude 3 Haiku (2024)",
        "context_window": 256000
    }
    try:
        updated_model = client.update_model("anthropic", "claude-3-haiku-20240307", update_data)
        logger.info(f"Updated model: {updated_model['model_id']}")
    except Exception as e:
        logger.error(f"Error updating model: {e}")
    
    # Example 8: Create a new API key
    logger.info("Creating a new API key...")
    new_api_key = {
        "key_value": "your-api-key",
        "is_encrypted": True,
        "environment": "development"
    }
    try:
        created_api_key = client.create_api_key("anthropic", new_api_key)
        logger.info(f"Created API key: {created_api_key['key_id']}")
    except Exception as e:
        logger.error(f"Error creating API key: {e}")
    
    # Example 9: Set a connection parameter
    logger.info("Setting a connection parameter...")
    param_data = {
        "param_name": "api_version",
        "param_value": "2023-06-01",
        "is_sensitive": False,
        "environment": "development"
    }
    try:
        set_param = client.set_connection_param("anthropic", param_data)
        logger.info(f"Set connection parameter: {set_param['param_name']}")
    except Exception as e:
        logger.error(f"Error setting connection parameter: {e}")
    
    # Example 10: Test a provider connection
    logger.info("Testing provider connection...")
    try:
        test_result = client.test_provider("anthropic")
        logger.info(f"Test result: {test_result['success']}")
    except Exception as e:
        logger.error(f"Error testing provider: {e}")
    
    # Example 11: Generate a response from an LLM
    logger.info("Generating a response from an LLM...")
    request_data = {
        "prompt": "Hello, world!",
        "model": "claude-3-opus-20240229",
        "provider_id": "anthropic",
        "max_tokens": 100,
        "temperature": 0.7
    }
    try:
        response = client.generate(request_data)
        logger.info(f"Generated response: {response['content']}")
    except Exception as e:
        logger.error(f"Error generating response: {e}")
    
    # Example 12: Delete a model
    logger.info("Deleting model...")
    try:
        delete_result = client.delete_model("anthropic", "claude-3-haiku-20240307")
        logger.info(f"Deleted model: {delete_result['model_id']}")
    except Exception as e:
        logger.error(f"Error deleting model: {e}")
    
    # Example 13: Delete a provider
    logger.info("Deleting provider...")
    try:
        delete_result = client.delete_provider("anthropic")
        logger.info(f"Deleted provider: {delete_result['provider_id']}")
    except Exception as e:
        logger.error(f"Error deleting provider: {e}")

if __name__ == "__main__":
    main()
