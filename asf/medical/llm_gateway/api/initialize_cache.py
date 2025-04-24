"""
Cache initialization module for the LLM Gateway.

This module provides functions to initialize and configure the cache system.
It should be imported and used during application startup to ensure all cache
components are properly initialized.
"""

import os
import logging
import asyncio
from typing import Dict, Any, Optional

from asf.medical.llm_gateway.cache.cache_manager import (
    get_cache_manager, 
    initialize_cache_manager
)
from asf.medical.llm_gateway.cache.persistent_store import DiskCacheStore
from asf.medical.llm_gateway.cache.embedding_providers import (
    LocalModelEmbeddingProvider,
    OpenAIEmbeddingProvider
)

logger = logging.getLogger(__name__)

async def setup_cache(
    config: Dict[str, Any],
    initialize_now: bool = True
) -> None:
    """
    Set up and initialize the cache system based on the provided configuration.
    
    Args:
        config: Gateway configuration dictionary
        initialize_now: Whether to initialize the cache immediately
    """
    # Extract cache config
    enable_caching = config.get("caching_enabled", True)
    
    if not enable_caching:
        logger.info("Caching is disabled. Skipping cache initialization.")
        return
    
    similarity_threshold = config.get("cache_similarity_threshold", 0.92)
    max_entries = config.get("cache_max_entries", 10000)
    ttl_seconds = config.get("cache_default_ttl_seconds", 3600)
    
    # Set up persistence
    persistence_config = config.get("cache_persistence", {})
    persistence_type = persistence_config.get("type", "disk") 
    
    # Get cache directory path
    cache_dir = persistence_config.get("cache_dir") or os.environ.get("LLM_CACHE_DIR")
    if not cache_dir:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".llm_gateway", "cache")
    
    # Set up embedding provider
    embedding_provider = None
    embedding_config = config.get("cache_embeddings", {})
    embedding_type = embedding_config.get("type", "default")
    
    if embedding_type == "openai":
        api_key = embedding_config.get("api_key") or os.environ.get("OPENAI_API_KEY")
        model = embedding_config.get("model", "text-embedding-3-small")
        try:
            embedding_provider = OpenAIEmbeddingProvider(
                api_key=api_key,
                model=model,
                local_cache_dir=os.path.join(cache_dir, "embeddings")
            )
            logger.info(f"Using OpenAI embedding provider with model {model}")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI embedding provider: {str(e)}")
    elif embedding_type == "local":
        model_name = embedding_config.get("model_name", "all-MiniLM-L6-v2")
        try:
            embedding_provider = LocalModelEmbeddingProvider(
                model_name=model_name,
                local_cache_dir=os.path.join(cache_dir, "embeddings")
            )
            logger.info(f"Using local embedding provider with model {model_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize local embedding provider: {str(e)}")
    
    # Set up persistent store
    persistent_store = None
    if persistence_type == "disk":
        use_pickle = persistence_config.get("use_pickle", False)
        persistent_store = DiskCacheStore(
            cache_dir=cache_dir,
            use_pickle=use_pickle
        )
        logger.info(f"Using disk-based cache persistence at {cache_dir}")
    
    # Configure cache manager
    cache_manager = get_cache_manager(
        enable_caching=enable_caching,
        similarity_threshold=similarity_threshold,
        max_entries=max_entries,
        ttl_seconds=ttl_seconds,
        embedding_provider=embedding_provider,
        persistent_store=persistent_store,
        persistence_type=persistence_type,
        persistence_config={
            "cache_dir": cache_dir,
            "use_pickle": persistence_config.get("use_pickle", False)
        }
    )
    
    # Initialize cache if requested
    if initialize_now:
        logger.info("Initializing cache...")
        await initialize_cache_manager()
        logger.info("Cache initialization complete")

async def shutdown_cache() -> None:
    """
    Properly shut down the cache system.
    
    This should be called during application shutdown to ensure
    all cache data is properly persisted and resources are released.
    """
    cache_manager = get_cache_manager()
    await cache_manager.close()
    logger.info("Cache system shut down successfully")