import json
import hashlib
from typing import Optional, Dict, Any
import logging
from aiocache import Cache
from pydantic import BaseModel

from agentor.llm_gateway.llm.base import LLMRequest, LLMResponse

logger = logging.getLogger(__name__)


class LLMCache:
    """Cache for LLM responses."""
    
    def __init__(self, ttl: int = 3600):
        """Initialize the LLM cache.
        
        Args:
            ttl: Time-to-live for cache entries in seconds (default: 1 hour)
        """
        self.cache = Cache(Cache.MEMORY)
        self.ttl = ttl
    
    def _generate_key(self, request: LLMRequest) -> str:
        """Generate a cache key for a request.
        
        Args:
            request: The LLM request
            
        Returns:
            A cache key
        """
        # Create a dictionary with the relevant fields
        key_dict = {
            "prompt": request.prompt,
            "model": request.model,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stop_sequences": request.stop_sequences
        }
        
        # Convert to a stable string representation
        key_str = json.dumps(key_dict, sort_keys=True)
        
        # Hash the string to create a fixed-length key
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    async def get(self, request: LLMRequest) -> Optional[LLMResponse]:
        """Get a cached response for a request.
        
        Args:
            request: The LLM request
            
        Returns:
            The cached response, or None if not found
        """
        key = self._generate_key(request)
        cached_data = await self.cache.get(key)
        
        if cached_data:
            logger.info(f"Cache hit for key: {key[:8]}...")
            return LLMResponse.parse_raw(cached_data)
        
        logger.info(f"Cache miss for key: {key[:8]}...")
        return None
    
    async def set(self, request: LLMRequest, response: LLMResponse):
        """Cache a response for a request.
        
        Args:
            request: The LLM request
            response: The LLM response
        """
        key = self._generate_key(request)
        await self.cache.set(key, response.json(), ttl=self.ttl)
        logger.info(f"Cached response for key: {key[:8]}...")


class CachedLLM(BaseModel):
    """A wrapper around an LLM that adds caching."""
    
    llm: Any  # BaseLLM
    cache: LLMCache
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response, using the cache if possible.
        
        Args:
            request: The LLM request
            
        Returns:
            The LLM response
        """
        # Check the cache first
        cached_response = await self.cache.get(request)
        if cached_response:
            return cached_response
        
        # Generate a new response
        response = await self.llm.generate(request)
        
        # Cache the response
        await self.cache.set(request, response)
        
        return response
