"""
Provider factory for LLM Gateway providers.

This module manages the creation and pooling of LLM provider client instances.
"""

import logging
from typing import Dict, Any, Optional, Type, Callable

from asf.medical.llm_gateway.core.models import ProviderConfig, GatewayConfig
from asf.medical.llm_gateway.providers.base import BaseProvider
from asf.medical.llm_gateway.providers.connection_pool import LLMConnectionPool
from asf.medical.llm_gateway.observability.metrics import MetricsService

logger = logging.getLogger(__name__)

# Global provider pools
_provider_pools: Dict[str, LLMConnectionPool] = {}

# Provider registry mapping provider_type to provider class
_provider_registry: Dict[str, Type[BaseProvider]] = {}


def register_provider(provider_type: str, provider_class: Type[BaseProvider]):
    """
    Register a provider class for a given provider type.

    Args:
        provider_type: The provider type identifier (e.g., "openai", "anthropic")
        provider_class: The provider class to register
    """
    _provider_registry[provider_type] = provider_class
    logger.info(f"Registered provider class {provider_class.__name__} for provider type '{provider_type}'")


def get_or_create_provider_pool(
    provider_type: str,
    create_fn: Callable[[], Any],
    config: Dict[str, Any],
    metrics_service: Optional[MetricsService] = None
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