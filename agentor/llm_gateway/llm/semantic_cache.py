import numpy as np
import hashlib
import json
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
import asyncio

from agentor.llm_gateway.llm.base import LLMRequest, LLMResponse

logger = logging.getLogger(__name__)


class SemanticCache:
    """Cache for LLM responses with semantic deduplication."""
    
    def __init__(self, threshold: float = 0.92, ttl: int = 3600):
        """Initialize the semantic cache.
        
        Args:
            threshold: The similarity threshold for considering two prompts as similar
            ttl: Time-to-live for cache entries in seconds (default: 1 hour)
        """
        self.threshold = threshold
        self.ttl = ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.embeddings: Dict[str, List[float]] = {}
        self.lock = asyncio.Lock()
    
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
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate the cosine similarity between two vectors.
        
        Args:
            a: The first vector
            b: The second vector
            
        Returns:
            The cosine similarity
        """
        a_np = np.array(a)
        b_np = np.array(b)
        return np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np))
    
    async def get_embedding(self, text: str) -> List[float]:
        """Get an embedding for a text.
        
        This method should be implemented to use a real embedding model.
        For now, we'll use a simple hash-based approach for demonstration.
        
        Args:
            text: The text to embed
            
        Returns:
            The embedding
        """
        # This is a placeholder. In a real implementation, you would use
        # a proper embedding model like OpenAI's text-embedding-ada-002
        # or a local model like sentence-transformers.
        
        # For demonstration, we'll use a simple hash-based approach
        hash_value = hashlib.md5(text.encode()).hexdigest()
        
        # Convert the hash to a list of floats
        embedding = []
        for i in range(0, len(hash_value), 2):
            if i + 2 <= len(hash_value):
                value = int(hash_value[i:i+2], 16) / 255.0
                embedding.append(value)
        
        # Normalize the embedding
        embedding_np = np.array(embedding)
        norm = np.linalg.norm(embedding_np)
        if norm > 0:
            embedding_np = embedding_np / norm
        
        return embedding_np.tolist()
    
    async def get(self, request: LLMRequest) -> Optional[LLMResponse]:
        """Get a cached response for a request.
        
        Args:
            request: The LLM request
            
        Returns:
            The cached response, or None if not found
        """
        # First, try an exact match
        exact_key = self._generate_key(request)
        async with self.lock:
            if exact_key in self.cache:
                entry = self.cache[exact_key]
                if time.time() < entry["expiry"]:
                    logger.info(f"Exact cache hit for key: {exact_key[:8]}...")
                    return entry["response"]
                else:
                    # Remove expired entry
                    del self.cache[exact_key]
                    if exact_key in self.embeddings:
                        del self.embeddings[exact_key]
        
        # If no exact match, try semantic matching
        query_embedding = await self.get_embedding(request.prompt)
        
        async with self.lock:
            # Find the most similar cached prompt
            best_match = None
            best_similarity = 0.0
            
            for key, embedding in self.embeddings.items():
                if key in self.cache and time.time() < self.cache[key]["expiry"]:
                    similarity = self._cosine_similarity(query_embedding, embedding)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = key
            
            # If we found a match above the threshold, return it
            if best_match and best_similarity >= self.threshold:
                logger.info(f"Semantic cache hit for key: {best_match[:8]}... (similarity: {best_similarity:.4f})")
                return self.cache[best_match]["response"]
        
        logger.info(f"Cache miss for prompt: {request.prompt[:50]}...")
        return None
    
    async def set(self, request: LLMRequest, response: LLMResponse):
        """Cache a response for a request.
        
        Args:
            request: The LLM request
            response: The LLM response
        """
        key = self._generate_key(request)
        expiry = time.time() + self.ttl
        
        # Get the embedding for the prompt
        embedding = await self.get_embedding(request.prompt)
        
        async with self.lock:
            # Store the response and embedding
            self.cache[key] = {
                "response": response,
                "expiry": expiry
            }
            self.embeddings[key] = embedding
        
        logger.info(f"Cached response for key: {key[:8]}...")
    
    async def clear_expired(self):
        """Clear expired cache entries."""
        now = time.time()
        async with self.lock:
            # Find expired keys
            expired_keys = [
                key for key, entry in self.cache.items()
                if now >= entry["expiry"]
            ]
            
            # Remove expired entries
            for key in expired_keys:
                del self.cache[key]
                if key in self.embeddings:
                    del self.embeddings[key]
            
            if expired_keys:
                logger.info(f"Cleared {len(expired_keys)} expired cache entries")


class SemanticCachedLLM:
    """A wrapper around an LLM that adds semantic caching."""
    
    def __init__(self, llm, cache: SemanticCache):
        """Initialize the cached LLM.
        
        Args:
            llm: The LLM to wrap
            cache: The semantic cache to use
        """
        self.llm = llm
        self.cache = cache
    
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
