"""
Configuration management for the OpenAI client.

This module provides functions for retrieving API keys and other configuration values
from various sources (connection parameters, secrets, environment variables).
"""

import logging
import os
from typing import Dict, Any, Optional

from asf.medical.core.secrets import SecretManager
from asf.medical.llm_gateway.core.models import ProviderConfig

logger = logging.getLogger(__name__)

def get_api_key(provider_config: ProviderConfig, provider_id: str) -> str:
    """
    Get the OpenAI API key from various sources.
    
    Args:
        provider_config: Provider configuration containing connection parameters
        provider_id: Provider ID for logging
        
    Returns:
        str: The API key
        
    Raises:
        ValueError: If no API key can be found from any source
    """
    # Try multiple sources for the API key in this order:
    # 1. Direct API key in connection_params
    # 2. Secret reference in connection_params
    # 3. Environment variable

    # 1. First check for direct API key in connection_params
    api_key = provider_config.connection_params.get("api_key")
    if api_key:
        masked_key = _mask_api_key(api_key)
        logger.debug(f"Using API key from connection_params.api_key for provider '{provider_id}': {masked_key}")
        return api_key

    # 2. If not found, check for secret reference
    if provider_config.connection_params.get("api_key_secret"):
        secret_ref = provider_config.connection_params.get("api_key_secret")
        # Format expected: "category:name" e.g. "llm:openai_api_key"
        if ":" in secret_ref:
            category, name = secret_ref.split(":", 1)
            secret_manager = SecretManager()
            api_key = secret_manager.get_secret(category, name)
            if api_key:
                masked_key = _mask_api_key(api_key)
                logger.debug(f"Retrieved API key from secret manager: {category}:{name} for provider '{provider_id}': {masked_key}")
                return api_key
            else:
                logger.warning(f"Secret {category}:{name} not found for provider '{provider_id}'")

    # 3. If still not found, try environment variable
    api_key_env_var = provider_config.connection_params.get("api_key_env_var", "OPENAI_API_KEY")
    api_key = os.environ.get(api_key_env_var)
    if api_key:
        masked_key = _mask_api_key(api_key)
        logger.debug(f"Using API key from environment variable {api_key_env_var} for provider '{provider_id}': {masked_key}")
        return api_key

    # If all methods failed, use a fallback for testing only in development
    if os.environ.get("ENV", "development") == "development":
        # Fallback API key for development
        api_key = os.environ.get("OPENAI_FALLBACK_API_KEY")
        if api_key:
            masked_key = _mask_api_key(api_key)
            logger.warning(f"Using fallback API key for provider '{provider_id}' - this should only be used in development: {masked_key}")
            return api_key

    # If we get here, no API key was found
    raise ValueError(
        f"OpenAI API key not found for provider '{provider_id}'. "
        f"Checked connection_params.api_key, connection_params.api_key_secret ({provider_config.connection_params.get('api_key_secret')}), "
        f"and environment variable '{api_key_env_var}'."
    )

def get_organization_id(provider_config: ProviderConfig) -> Optional[str]:
    """
    Get the OpenAI organization ID from various sources.
    
    Args:
        provider_config: Provider configuration containing connection parameters
        
    Returns:
        Optional[str]: The organization ID or None if not found
    """
    # First check for direct org_id in connection_params
    org_id = provider_config.connection_params.get("org_id")
    if org_id:
        return org_id

    # If not found, try environment variable
    org_id_env_var = provider_config.connection_params.get("org_id_env_var", "OPENAI_ORG_ID")
    return os.environ.get(org_id_env_var)

def _mask_api_key(api_key: str) -> str:
    """
    Mask an API key for logging.
    
    Args:
        api_key: The API key to mask
        
    Returns:
        str: The masked API key
    """
    if not api_key or len(api_key) < 10:
        return "***MASKED***"
    return f"{api_key[:5]}...{api_key[-4:]}"