"""
Enhanced embedding providers for semantic caching in LLM Gateway.

This module provides various embedding providers to generate vector
representations of text for semantic similarity matching.
"""

import os
import json
import logging
import hashlib
from typing import Dict, List, Optional, Any

import aiohttp
import numpy as np

from asf.medical.llm_gateway.cache.semantic_cache import EmbeddingProvider

logger = logging.getLogger(__name__)

class LocalModelEmbeddingProvider(EmbeddingProvider):
    """
    Embedding provider using local models via sentence-transformers.
    
    Requires the sentence-transformers package to be installed:
    pip install sentence-transformers
    """
    
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        local_cache_dir: Optional[str] = None
    ):
        """
        Initialize local model embedding provider.
        
        Args:
            model_name: Name of the model to use
            device: Device to run the model on ('cpu' or 'cuda')
            local_cache_dir: Directory to cache embeddings (optional)
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.embedding_dimension = 0
        self.local_cache_dir = local_cache_dir
        self._cache = {}
        
        # Create local cache directory if specified
        if self.local_cache_dir:
            os.makedirs(self.local_cache_dir, exist_ok=True)
            
        try:
            # Lazy load the model when first needed
            self._load_model()
            logger.info(f"Initialized local model embedding provider with model {model_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize local model: {e}")
            
    def _load_model(self):
        """Load the embedding model if it's not already loaded."""
        if self.model is None:
            try:
                # Import here to avoid dependency issues
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name, device=self.device)
                self.embedding_dimension = self.model.get_sentence_embedding_dimension()
                logger.info(f"Loaded embedding model {self.model_name} with {self.embedding_dimension} dimensions")
            except ImportError:
                logger.error("sentence-transformers package not installed. Install with: pip install sentence-transformers")
                raise
    
    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for text."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _load_from_cache(self, text: str) -> Optional[List[float]]:
        """Load embedding from cache."""
        if not self.local_cache_dir:
            return self._cache.get(self._get_cache_key(text))
            
        # Try to load from disk cache
        cache_key = self._get_cache_key(text)
        cache_path = os.path.join(self.local_cache_dir, f"{cache_key}.json")
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load embedding from cache: {e}")
                
        return None
    
    def _save_to_cache(self, text: str, embedding: List[float]) -> None:
        """Save embedding to cache."""
        cache_key = self._get_cache_key(text)
        
        # Save to memory cache
        self._cache[cache_key] = embedding
        
        # Save to disk cache if enabled
        if self.local_cache_dir:
            cache_path = os.path.join(self.local_cache_dir, f"{cache_key}.json")
            try:
                with open(cache_path, 'w') as f:
                    json.dump(embedding, f)
            except Exception as e:
                logger.warning(f"Failed to save embedding to cache: {e}")
                
    async def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding vector for text.
        
        Args:
            text: The text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        if not text:
            return [0.0] * (self.embedding_dimension or 384)
            
        # Check cache first
        cached_embedding = self._load_from_cache(text)
        if cached_embedding:
            return cached_embedding
            
        try:
            # Make sure model is loaded
            self._load_model()
            
            # Generate embedding
            embedding = self.model.encode(text).tolist()
            
            # Cache the result
            self._save_to_cache(text, embedding)
            
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding with local model: {e}")
            # Fall back to a simple hash-based embedding
            hash_obj = hashlib.sha256(text.encode('utf-8'))
            hash_bytes = hash_obj.digest()
            dim = self.embedding_dimension or 384
            
            vector = []
            for i in range(dim):
                byte_val = hash_bytes[i % len(hash_bytes)]
                val = (byte_val / 127.5) - 1.0
                vector.append(val)
                
            # Normalize vector
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = [v / norm for v in vector]
                
            return vector


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    Embedding provider using OpenAI's embedding models.
    
    Requires an OpenAI API key and the openai package to be installed:
    pip install openai
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        dimensions: Optional[int] = None,
        local_cache_dir: Optional[str] = None
    ):
        """
        Initialize OpenAI embedding provider.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
            model: Name of the embedding model to use
            dimensions: Number of dimensions to use (optional)
            local_cache_dir: Directory to cache embeddings (optional)
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Provide as argument or set OPENAI_API_KEY environment variable.")
            
        self.model = model
        self.dimensions = dimensions
        self.local_cache_dir = local_cache_dir
        self._cache = {}
        
        # Create local cache directory if specified
        if self.local_cache_dir:
            os.makedirs(self.local_cache_dir, exist_ok=True)
            
        logger.info(f"Initialized OpenAI embedding provider with model {model}")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for text."""
        model_info = f"{self.model}_{self.dimensions or 'default'}"
        return hashlib.md5(f"{model_info}:{text}".encode('utf-8')).hexdigest()
    
    def _load_from_cache(self, text: str) -> Optional[List[float]]:
        """Load embedding from cache."""
        if not self.local_cache_dir:
            return self._cache.get(self._get_cache_key(text))
            
        # Try to load from disk cache
        cache_key = self._get_cache_key(text)
        cache_path = os.path.join(self.local_cache_dir, f"{cache_key}.json")
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load embedding from cache: {e}")
                
        return None
    
    def _save_to_cache(self, text: str, embedding: List[float]) -> None:
        """Save embedding to cache."""
        cache_key = self._get_cache_key(text)
        
        # Save to memory cache
        self._cache[cache_key] = embedding
        
        # Save to disk cache if enabled
        if self.local_cache_dir:
            cache_path = os.path.join(self.local_cache_dir, f"{cache_key}.json")
            try:
                with open(cache_path, 'w') as f:
                    json.dump(embedding, f)
            except Exception as e:
                logger.warning(f"Failed to save embedding to cache: {e}")
                
    async def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding vector for text using the OpenAI API.
        
        Args:
            text: The text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        if not text:
            # Return zero vector with appropriate dimensions
            # text-embedding-3-small is 1536 dimensions by default
            dimensions = self.dimensions or 1536
            return [0.0] * dimensions
            
        # Check cache first
        cached_embedding = self._load_from_cache(text)
        if cached_embedding:
            return cached_embedding
            
        # Prepare request parameters
        params = {
            "model": self.model,
            "input": text,
        }
        if self.dimensions:
            params["dimensions"] = self.dimensions
            
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            # Make API request
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.openai.com/v1/embeddings",
                    headers=headers,
                    json=params
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise ValueError(f"OpenAI API error: {response.status} - {error_text}")
                        
                    result = await response.json()
                    embedding = result["data"][0]["embedding"]
                    
                    # Cache the result
                    self._save_to_cache(text, embedding)
                    
                    return embedding
        except Exception as e:
            logger.error(f"Error generating embedding with OpenAI: {e}")
            # Fall back to a simple hash-based embedding
            hash_obj = hashlib.sha256(text.encode('utf-8'))
            hash_bytes = hash_obj.digest()
            dim = self.dimensions or 1536
            
            vector = []
            for i in range(dim):
                byte_val = hash_bytes[i % len(hash_bytes)]
                val = (byte_val / 127.5) - 1.0
                vector.append(val)
                
            # Normalize vector
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = [v / norm for v in vector]
                
            return vector