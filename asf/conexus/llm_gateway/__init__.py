"""
Conexus LLM Gateway - A domain-agnostic LLM service interface

This package provides a standardized gateway for accessing various LLM providers
with unified interfaces, observability, caching, and resilience features.
"""

__version__ = "1.0.0"
__author__ = "ASF Research"

# Convenience imports
from asf.conexus.llm_gateway.core.models import LLMRequest, LLMResponse, LLMConfig
from asf.conexus.llm_gateway.core.client import LLMClient

# Module initialization
def initialize():
    """Initialize the LLM Gateway with default settings."""
    from asf.conexus.llm_gateway.core.factory import get_gateway_factory
    from asf.conexus.llm_gateway.config.manager import ConfigManager
    
    # Initialize config
    config_manager = ConfigManager()
    config = config_manager.load_default_config()
    
    # Create and initialize gateway
    factory = get_gateway_factory()
    return factory.create_gateway(config)

# Simplified client creation
def create_client(config_path=None):
    """Create a configured LLM client for easy access to LLM services."""
    from asf.conexus.llm_gateway.core.factory import get_gateway_factory
    from asf.conexus.llm_gateway.config.manager import ConfigManager
    
    # Initialize config
    config_manager = ConfigManager()
    config = config_manager.load_config(config_path) if config_path else config_manager.load_default_config()
    
    # Create client
    factory = get_gateway_factory()
    return factory.create_client(config)

