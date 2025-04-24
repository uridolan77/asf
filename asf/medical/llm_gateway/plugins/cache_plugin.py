"""
Caching Plugin for LLM Gateway.

This plugin provides caching functionality for LLM responses,
reducing costs and improving latency for common queries.
"""

import hashlib
import json
import logging
import time
from typing import Any, Dict, Optional, Set

from asf.medical.llm_gateway.core.plugins import BasePlugin, PluginCategory, PluginEventType
from asf.medical.llm_gateway.core.models import LLMRequest, LLMResponse

logger = logging.getLogger(__name__)

class CachingPlugin(BasePlugin):
    """
    Plugin for caching LLM responses based on request content.
    
    This plugin can dramatically reduce costs and latency for common
    queries by caching responses and serving them without calling the LLM.
    """
    
    name = "caching_plugin"
    display_name = "Caching Plugin"
    category: PluginCategory = "provider"  # Affects provider selection
    priority = 20  # Run after metrics but before most processing
    description = "Provides caching for LLM responses"
    version = "1.0.0"
    author = "ASF Medical Research"
    tags = {"cache", "optimization", "performance"}
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the caching plugin with configuration."""
        super().__init__(config)
        
        # Cache settings from config
        self.ttl_seconds = self.config.get("ttl_seconds", 3600)  # 1 hour default
        self.max_cache_size = self.config.get("max_cache_size", 1000)
        self.cache_enabled = self.config.get("enabled", True)
        
        # Excluded models (never cache these)
        self.excluded_models: Set[str] = set(self.config.get("excluded_models", []))
        
        # Simple in-memory cache: {hash: {response, timestamp}}
        self.cache: Dict[str, Dict[str, Any]] = {}
        
        # Stats
        self.hits = 0
        self.misses = 0
        
        logger.info(f"Caching plugin initialized (enabled: {self.cache_enabled})")
    
    async def initialize(self) -> None:
        """Initialize the plugin (placeholder for future enhancements)."""
        pass
    
    async def on_event(self, event_type: PluginEventType, payload: Any) -> Any:
        """
        Process gateway events for caching.
        
        Args:
            event_type: The type of event
            payload: Event data (request, response, etc.)
            
        Returns:
            Modified payload with cached response if found
        """
        if not self.cache_enabled:
            return payload
        
        # For request_start, check cache and intercept if we have a cached response
        if event_type == "request_start" and isinstance(payload, dict) and "request" in payload:
            request = payload["request"]
            model_id = getattr(request.config, "model_identifier", "")
            
            # Skip caching for excluded models
            if model_id in self.excluded_models:
                return payload
            
            # Skip caching for streaming requests
            if getattr(request.config, "stream", False):
                return payload
            
            cache_key = self._compute_cache_key(request)
            cached = self._get_cached_response(cache_key)
            
            if cached:
                logger.info(f"Cache hit for request {request.request_id} (model: {model_id})")
                self.hits += 1
                
                # Set the cached response in the payload
                payload["response"] = cached
                payload["from_cache"] = True
                
                # Signal that we should skip the normal provider call
                payload["skip_provider"] = True
            else:
                self.misses += 1
                
        # For response_end, store in cache
        elif event_type == "response_end" and isinstance(payload, dict) and "request" in payload and "response" in payload:
            request = payload["request"]
            response = payload["response"]
            
            # Don't cache error responses or results from streaming
            if response.error_details or getattr(request.config, "stream", False):
                return payload
                
            model_id = getattr(request.config, "model_identifier", "")
            
            # Skip caching for excluded models
            if model_id in self.excluded_models:
                return payload
            
            cache_key = self._compute_cache_key(request)
            self._cache_response(cache_key, response)
            
        return payload
    
    async def shutdown(self) -> None:
        """Clean up cache on shutdown."""
        self.cache.clear()
        logger.info(f"Cache plugin shutdown. Stats: {self.hits} hits, {self.misses} misses")
    
    def _compute_cache_key(self, request: LLMRequest) -> str:
        """
        Compute a cache key for a request.
        
        Args:
            request: The LLM request
            
        Returns:
            A hash string to use as the cache key
        """
        # Extract components that affect the response
        key_components = {
            "model": getattr(request.config, "model_identifier", ""),
            "prompt": request.prompt_content,
            "temperature": getattr(request.config, "temperature", 1.0),
            "max_tokens": getattr(request.config, "max_tokens", None),
            "top_p": getattr(request.config, "top_p", 1.0),
            "frequency_penalty": getattr(request.config, "frequency_penalty", 0.0),
            "presence_penalty": getattr(request.config, "presence_penalty", 0.0),
            "system_prompt": getattr(request.config, "system_prompt", ""),
        }
        
        # Convert to string and hash
        key_str = json.dumps(key_components, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[LLMResponse]:
        """
        Get a cached response if it exists and is not expired.
        
        Args:
            cache_key: The cache key
            
        Returns:
            Cached response or None if not found/expired
        """
        cached_item = self.cache.get(cache_key)
        if not cached_item:
            return None
        
        # Check if expired
        current_time = time.time()
        if current_time - cached_item["timestamp"] > self.ttl_seconds:
            # Expired - remove from cache
            del self.cache[cache_key]
            return None
        
        return cached_item["response"]
    
    def _cache_response(self, cache_key: str, response: LLMResponse) -> None:
        """
        Cache a response.
        
        Args:
            cache_key: The cache key
            response: The response to cache
        """
        # Simple LRU eviction if we hit the limit
        if len(self.cache) >= self.max_cache_size:
            # Remove oldest entry (simple approach for now)
            oldest_key = min(self.cache.items(), key=lambda x: x[1]["timestamp"])[0]
            del self.cache[oldest_key]
            
        self.cache[cache_key] = {
            "response": response,
            "timestamp": time.time()
        }
        
        logger.debug(f"Cached response with key {cache_key[:8]}..., cache size: {len(self.cache)}")