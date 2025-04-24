"""
Provider factory for Domain-Agnostic LLM Gateway providers.

This module manages the creation and pooling of LLM provider client instances.
"""

import logging
from typing import Dict, Any, Optional, Type, Callable

from asf.conexus.llm_gateway.core.models import ProviderConfig, GatewayConfig
from asf.conexus.llm_gateway.providers.base import BaseProvider
from asf.conexus.llm_gateway.providers.connection_pool import LLMConnectionPool
# Import the resilient provider wrapper
from asf.conexus.llm_gateway.providers.resilient_provider import ResilientProviderWrapper
# Import the cached provider wrapper
from asf.conexus.llm_gateway.providers.cached_provider import create_cached_provider
# Import provider-specific factory functions
from asf.conexus.llm_gateway.providers.openai import create_resilient_openai_provider

logger = logging.getLogger(__name__)

# Global provider pools
_provider_pools: Dict[str, LLMConnectionPool] = {}

# Provider registry mapping provider_type to provider class
_provider_registry: Dict[str, Type[BaseProvider]] = {}

# Provider factory registry for specialized factory functions
_provider_factory_registry: Dict[str, Callable[[ProviderConfig], BaseProvider]] = {}


def register_provider(provider_type: str, provider_class: Type[BaseProvider]):
    """
    Register a provider class for a given provider type.

    Args:
        provider_type: The provider type identifier (e.g., "openai", "anthropic")
        provider_class: The provider class to register
    """
    _provider_registry[provider_type] = provider_class
    logger.info(f"Registered provider class {provider_class.__name__} for provider type '{provider_type}'")


def register_provider_factory(provider_type: str, factory_fn: Callable[[ProviderConfig], BaseProvider]):
    """
    Register a factory function for creating providers with a given provider type.
    
    This is useful for providers that require special initialization or resilience patterns.

    Args:
        provider_type: The provider type identifier (e.g., "openai", "anthropic")
        factory_fn: Function that creates a provider instance
    """
    _provider_factory_registry[provider_type] = factory_fn
    logger.info(f"Registered factory function for provider type '{provider_type}'")


# Register built-in provider factory functions
register_provider_factory("openai", create_resilient_openai_provider)


def get_or_create_provider_pool(
    provider_type: str,
    create_fn: Callable[[], Any],
    config: Dict[str, Any],
    metrics_service: Optional[Any] = None
) -> LLMConnectionPool:
    """
    Get an existing provider pool or create a new one.

    Args:
        provider_type: The provider type identifier
        create_fn: Factory function to create new provider client instances
        config: Connection pool configuration
        metrics_service: Optional metrics service

    Returns:
        LLMConnectionPool: Connection pool for the specified provider
    """
    pool_key = f"{provider_type}"
    
    if pool_key in _provider_pools:
        logger.debug(f"Reusing existing connection pool for provider '{provider_type}'")
        return _provider_pools[pool_key]
    
    # Extract pool configuration
    max_connections = config.get("max_connections", 10)
    min_connections = config.get("min_connections", 1)
    max_idle_time = config.get("max_idle_time_seconds", 300)
    health_check_interval = config.get("health_check_interval_seconds", 60)
    circuit_breaker_config = config.get("circuit_breaker", None)
    rate_limit_config = config.get("rate_limit", None)
    
    # Create new pool
    logger.info(f"Creating new connection pool for provider '{provider_type}'")
    pool = LLMConnectionPool(
        create_client_fn=create_fn,
        max_connections=max_connections,
        min_connections=min_connections,
        max_idle_time_seconds=max_idle_time,
        health_check_interval_seconds=health_check_interval,
        circuit_breaker_config=circuit_breaker_config,
        rate_limit_config=rate_limit_config,
        name=provider_type,
        metrics_service=metrics_service
    )
    
    _provider_pools[pool_key] = pool
    return pool


async def cleanup_provider_pools():
    """Clean up all provider pools."""
    logger.info(f"Cleaning up {len(_provider_pools)} provider connection pools")
    
    for pool_key, pool in list(_provider_pools.items()):
        try:
            await pool.cleanup()
            logger.info(f"Cleaned up connection pool: {pool_key}")
        except Exception as e:
            logger.error(f"Error cleaning up connection pool {pool_key}: {e}", exc_info=True)
    
    _provider_pools.clear()


def create_resilient_openai_provider_with_cache(provider_config: ProviderConfig) -> BaseProvider:
    """
    Create a resilient OpenAI provider with caching.

    This is a convenience function for creating a cached OpenAI provider
    with resilience patterns already applied.

    Args:
        provider_config: Provider configuration

    Returns:
        BaseProvider: Cached resilient OpenAI provider
    """
    # First create the resilient provider
    provider = create_resilient_openai_provider(provider_config)
    
    # Extract cache configuration from provider_config
    cache_config = provider_config.connection_params.get("cache", {})
    
    # If caching is disabled, return the provider as-is
    if not cache_config.get("enabled", True):
        logger.info(f"Caching disabled for provider '{provider_config.provider_id}'")
        return provider
    
    # Wrap with cache
    logger.info(f"Creating cached provider for '{provider_config.provider_id}'")
    return create_cached_provider(
        provider=provider,
        provider_id=provider_config.provider_id,
        cache_config=cache_config
    )


# Register built-in cached provider factory functions
register_provider_factory("openai_cached", create_resilient_openai_provider_with_cache)


class ProviderFactory:
    """
    Factory for creating and managing provider instances.
    
    This class handles the creation of provider instances based on the provider type
    and manages the lifecycle of provider connections.
    """
    
    def __init__(self, gateway_config: GatewayConfig, metrics_service: Optional[Any] = None):
        """
        Initialize the provider factory.
        
        Args:
            gateway_config: The gateway configuration
            metrics_service: Optional metrics service
        """
        self.gateway_config = gateway_config
        self.metrics_service = metrics_service
        self.default_provider = gateway_config.default_provider
        logger.info(f"Initializing ProviderFactory with default provider: {self.default_provider}")
    
    def get_provider_config(self, provider_id: str) -> Optional[ProviderConfig]:
        """
        Get the configuration for a specific provider.
        
        Args:
            provider_id: The provider identifier
            
        Returns:
            ProviderConfig or None if not found
        """
        if provider_id in self.gateway_config.providers:
            return self.gateway_config.providers[provider_id]
        return None
    
    def create_provider(self, provider_id: str) -> BaseProvider:
        """
        Create a provider instance.
        
        Args:
            provider_id: The provider identifier
            
        Returns:
            Provider instance
            
        Raises:
            ValueError: If provider_id is not found in configuration
            KeyError: If provider_type is not registered
        """
        # Get provider configuration
        provider_config = self.get_provider_config(provider_id)
        if not provider_config:
            raise ValueError(f"Provider '{provider_id}' not found in gateway configuration")
        
        # Get provider type and check if we have a specialized factory
        provider_type = provider_config.provider_type
        
        # Check if this provider should use caching
        use_cache = provider_config.connection_params.get("use_cache", False)
        if use_cache:
            # Modify provider_type to use the cached version if available
            cached_provider_type = f"{provider_type}_cached"
            if cached_provider_type in _provider_factory_registry:
                provider_type = cached_provider_type
            else:
                logger.warning(f"Caching requested for provider '{provider_id}' but no cached factory available for '{provider_type}'. Using non-cached version.")
        
        # Check if we have a specialized factory for this provider type
        if provider_type in _provider_factory_registry:
            # Use specialized factory
            factory_fn = _provider_factory_registry[provider_type]
            logger.info(f"Using specialized factory for provider '{provider_id}' of type '{provider_type}'")
            return factory_fn(provider_config)
        
        # Fall back to standard provider creation
        if provider_type not in _provider_registry:
            raise KeyError(f"No provider class registered for type '{provider_type}'")
        
        provider_class = _provider_registry[provider_type]
        logger.info(f"Creating provider instance for '{provider_id}' of type '{provider_type}'")
        
        # Create standard provider instance
        provider = provider_class(provider_config)
        
        # Wrap with resilient provider pattern if resilience is enabled
        if provider_config.connection_params.get("use_resilience", True):
            logger.info(f"Creating resilient wrapper for provider '{provider_id}'")
            provider = ResilientProviderWrapper(
                provider=provider,
                provider_id=provider_id
            )
        
        # Add caching if requested but no specialized factory exists
        if use_cache:
            logger.info(f"Adding caching wrapper for provider '{provider_id}'")
            provider = create_cached_provider(
                provider=provider,
                provider_id=provider_id,
                cache_config=provider_config.connection_params.get("cache", {})
            )
        
        return provider
        
    def get_provider_for_model(self, model_identifier: str) -> BaseProvider:
        """
        Get the appropriate provider for a given model.
        
        This method determines which provider should be used for the requested model,
        using explicit mappings or default provider as fallback.
        
        Args:
            model_identifier: The model identifier
            
        Returns:
            Provider instance
            
        Raises:
            ValueError: If no provider can be found for the model
        """
        # Check if there's an explicit mapping
        if self.gateway_config.model_provider_mapping:
            provider_id = self.gateway_config.model_provider_mapping.get(model_identifier)
            if provider_id:
                logger.debug(f"Using explicit provider mapping: {model_identifier} -> {provider_id}")
                return self.create_provider(provider_id)
        
        # Try to find a provider that supports this model
        for provider_id, config in self.gateway_config.providers.items():
            if model_identifier in config.models:
                logger.debug(f"Found provider '{provider_id}' supporting model '{model_identifier}'")
                return self.create_provider(provider_id)
        
        # Fall back to default provider
        logger.warning(f"No specific provider found for model '{model_identifier}', using default")
        return self.create_provider(self.default_provider)
    
    async def cleanup(self) -> None:
        """Clean up all provider resources."""
        await cleanup_provider_pools()