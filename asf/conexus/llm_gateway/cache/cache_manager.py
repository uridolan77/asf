"""
Cache Manager for LLM Gateway

Provides centralized access to caching functionality for the LLM Gateway.
"""

import asyncio
import logging
import os
from typing import Any, Dict, Optional, List
from pathlib import Path

from asf.conexus.llm_gateway.core.models import LLMRequest, LLMResponse
from asf.conexus.llm_gateway.cache.semantic_cache import (
    SemanticCache, EmbeddingProvider, DefaultEmbeddingProvider, CacheEntry
)
from asf.conexus.llm_gateway.cache.persistent_store import (
    BaseCacheStore, DiskCacheStore
)
from asf.conexus.llm_gateway.cache.embedding_providers import (
    LocalModelEmbeddingProvider, OpenAIEmbeddingProvider
)

logger = logging.getLogger(__name__)

class CacheManager:
    """
    Manager for LLM Gateway caching functionality.
    
    This class provides centralized access to caching functionality,
    including semantic caching and configuration management.
    """
    
    def __init__(
        self,
        enable_caching: bool = True,
        similarity_threshold: float = 0.92,
        max_entries: int = 10000,
        ttl_seconds: int = 3600,
        embedding_provider: Optional[EmbeddingProvider] = None,
        persistent_store: Optional[BaseCacheStore] = None,
        persistence_type: str = "disk",
        persistence_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize cache manager.
        
        Args:
            enable_caching: Whether caching is enabled
            similarity_threshold: Minimum similarity score for cache hit (0-1)
            max_entries: Maximum number of entries in cache
            ttl_seconds: Time-to-live for cache entries in seconds
            embedding_provider: Provider for generating embeddings (optional)
            persistent_store: Custom persistent store implementation (optional)
            persistence_type: Type of persistence to use: "none", "disk"
            persistence_config: Configuration options for persistence
        """
        self.enable_caching = enable_caching
        self.cache_lock = asyncio.Lock()
        self._initialized = False
        
        # Create semantic cache if enabled
        if enable_caching:
            # Setup persistence if requested
            if persistent_store:
                self.persistent_store = persistent_store
            elif persistence_type == "disk":
                config = persistence_config or {}
                cache_dir = config.get("cache_dir") or os.environ.get("LLM_CACHE_DIR")
                
                # Default cache directory
                if not cache_dir:
                    home_dir = os.path.expanduser("~")
                    cache_dir = os.path.join(home_dir, ".llm_gateway", "cache")
                
                use_pickle = config.get("use_pickle", False)
                
                self.persistent_store = DiskCacheStore(
                    cache_dir=cache_dir,
                    use_pickle=use_pickle
                )
                logger.info(f"Using disk-based cache persistence at: {cache_dir}")
            else:
                self.persistent_store = None
                
            # Choose embedding provider if not provided
            if embedding_provider:
                self.embedding_provider = embedding_provider
            else:
                # Try to use a sophisticated embedding provider, falling back to default if unavailable
                try:
                    # First try OpenAI embeddings if API key is available
                    openai_api_key = os.environ.get("OPENAI_API_KEY")
                    if openai_api_key:
                        cache_dir = os.environ.get("LLM_CACHE_DIR")
                        if not cache_dir:
                            home_dir = os.path.expanduser("~")
                            cache_dir = os.path.join(home_dir, ".llm_gateway")
                        
                        embeddings_cache_dir = os.path.join(cache_dir, "embeddings")
                        self.embedding_provider = OpenAIEmbeddingProvider(
                            api_key=openai_api_key,
                            local_cache_dir=embeddings_cache_dir
                        )
                        logger.info("Using OpenAI embedding provider for semantic cache")
                    else:
                        # Fall back to local model if OpenAI API key not available
                        try:
                            cache_dir = os.environ.get("LLM_CACHE_DIR")
                            if not cache_dir:
                                home_dir = os.path.expanduser("~")
                                cache_dir = os.path.join(home_dir, ".llm_gateway")
                                
                            embeddings_cache_dir = os.path.join(cache_dir, "embeddings")
                            self.embedding_provider = LocalModelEmbeddingProvider(
                                model_name="all-MiniLM-L6-v2",
                                local_cache_dir=embeddings_cache_dir
                            )
                            logger.info("Using local embedding model for semantic cache")
                        except Exception as e:
                            logger.warning(f"Failed to initialize local embedding model: {e}. Using default.")
                            self.embedding_provider = DefaultEmbeddingProvider()
                except Exception as e:
                    logger.warning(f"Failed to initialize advanced embedding provider: {e}. Using default.")
                    self.embedding_provider = DefaultEmbeddingProvider()
            
            # Initialize semantic cache
            self.semantic_cache = SemanticCache(
                embedding_provider=self.embedding_provider,
                similarity_threshold=similarity_threshold,
                max_entries=max_entries,
                ttl_seconds=ttl_seconds,
                cache_lock=self.cache_lock,
                persistent_store=self.persistent_store
            )
            logger.info(
                f"Initialized cache manager with semantic caching "
                f"(threshold={similarity_threshold:.2f}, max_entries={max_entries}, "
                f"ttl={ttl_seconds}s, persistence={persistence_type})"
            )
        else:
            self.semantic_cache = None
            self.embedding_provider = None
            self.persistent_store = None
            logger.info("Initialized cache manager with caching disabled")
    
    async def initialize(self) -> None:
        """Initialize the cache manager and load data from persistent store if available."""
        if self._initialized:
            return
            
        if self.enable_caching and self.semantic_cache:
            try:
                # Initialize semantic cache (loads from persistent store)
                await self.semantic_cache.initialize()
                logger.info("Cache manager initialization complete")
            except Exception as e:
                logger.error(f"Error initializing cache manager: {e}")
        
        self._initialized = True
    
    async def get_response(self, request: LLMRequest) -> Optional[LLMResponse]:
        """
        Get cached response for request.
        
        Args:
            request: LLM request
            
        Returns:
            Cached response if available, None otherwise
        """
        if not self.enable_caching or not self.semantic_cache:
            return None
            
        try:
            return await self.semantic_cache.get(request)
        except Exception as e:
            logger.error(f"Error getting response from cache: {e}", exc_info=True)
            return None
    
    async def store_response(self, request: LLMRequest, response: LLMResponse) -> None:
        """
        Store response in cache.
        
        Args:
            request: LLM request
            response: LLM response
        """
        if not self.enable_caching or not self.semantic_cache:
            return
            
        try:
            await self.semantic_cache.store(request, response)
        except Exception as e:
            logger.error(f"Error storing response in cache: {e}", exc_info=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        if not self.enable_caching or not self.semantic_cache:
            return {"enabled": False}
            
        try:
            stats = self.semantic_cache.get_stats()
            stats["enabled"] = True
            # Add persistence info
            stats["persistence"] = {
                "type": self._get_persistence_type(),
                "location": self._get_persistence_location()
            }
            # Add embedding provider info
            stats["embedding_provider"] = self._get_embedding_provider_info()
            return stats
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}", exc_info=True)
            return {"enabled": True, "error": str(e)}
    
    def _get_persistence_type(self) -> str:
        """Get the type of persistence being used."""
        if not self.persistent_store:
            return "none"
        if isinstance(self.persistent_store, DiskCacheStore):
            return "disk"
        return f"custom ({self.persistent_store.__class__.__name__})"
    
    def _get_persistence_location(self) -> str:
        """Get the location of the persistent storage."""
        if not self.persistent_store:
            return "N/A"
        if isinstance(self.persistent_store, DiskCacheStore):
            return str(self.persistent_store.cache_dir)
        return "unknown"
    
    def _get_embedding_provider_info(self) -> Dict[str, Any]:
        """Get information about the embedding provider."""
        if not self.embedding_provider:
            return {"type": "none"}
            
        if isinstance(self.embedding_provider, DefaultEmbeddingProvider):
            return {
                "type": "default_hash",
                "vector_dimension": self.embedding_provider.vector_dim
            }
        if isinstance(self.embedding_provider, OpenAIEmbeddingProvider):
            return {
                "type": "openai",
                "model": self.embedding_provider.model,
                "dimensions": self.embedding_provider.dimensions or "default"
            }
        if isinstance(self.embedding_provider, LocalModelEmbeddingProvider):
            return {
                "type": "local_model",
                "model_name": self.embedding_provider.model_name,
                "dimensions": getattr(self.embedding_provider, 'embedding_dimension', 'unknown')
            }
        
        return {"type": self.embedding_provider.__class__.__name__}
    
    async def clear(self) -> None:
        """Clear the cache."""
        if not self.enable_caching or not self.semantic_cache:
            return
            
        try:
            await self.semantic_cache.clear()
        except Exception as e:
            logger.error(f"Error clearing cache: {e}", exc_info=True)
    
    async def close(self) -> None:
        """Close the cache manager and release resources."""
        if self.semantic_cache:
            try:
                await self.semantic_cache.close()
            except Exception as e:
                logger.error(f"Error closing semantic cache: {e}", exc_info=True)

# Global cache manager instance
_cache_manager = None

def get_cache_manager(
    enable_caching: bool = True,
    similarity_threshold: float = 0.92,
    max_entries: int = 10000,
    ttl_seconds: int = 3600,
    embedding_provider: Optional[EmbeddingProvider] = None,
    persistent_store: Optional[BaseCacheStore] = None,
    persistence_type: str = "disk",
    persistence_config: Optional[Dict[str, Any]] = None
) -> CacheManager:
    """
    Get the global cache manager instance.
    
    Args:
        enable_caching: Whether caching is enabled
        similarity_threshold: Minimum similarity score for cache hit (0-1)
        max_entries: Maximum number of entries in cache
        ttl_seconds: Time-to-live for cache entries in seconds
        embedding_provider: Provider for generating embeddings (optional)
        persistent_store: Custom persistent store implementation (optional)
        persistence_type: Type of persistence to use: "none", "disk", or "redis"
        persistence_config: Configuration options for persistence
        
    Returns:
        Global cache manager instance
    """
    global _cache_manager
    
    if _cache_manager is None:
        _cache_manager = CacheManager(
            enable_caching=enable_caching,
            similarity_threshold=similarity_threshold,
            max_entries=max_entries,
            ttl_seconds=ttl_seconds,
            embedding_provider=embedding_provider,
            persistent_store=persistent_store,
            persistence_type=persistence_type,
            persistence_config=persistence_config
        )
        
    return _cache_manager

async def initialize_cache_manager() -> None:
    """Initialize the global cache manager."""
    cache_manager = get_cache_manager()
    await cache_manager.initialize()