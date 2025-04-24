"""
Semantic Cache for LLM Gateway

This module implements a semantic caching system that stores LLM responses
based on the semantic similarity of prompts rather than exact matches.
This enables higher cache hit rates and greater cost savings compared to
traditional caching mechanisms.
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Set, Union

import numpy as np
from pydantic import BaseModel, Field

from asf.conexus.llm_gateway.core.models import (
    LLMRequest, LLMResponse, UsageStats, PerformanceMetrics
)

logger = logging.getLogger(__name__)

class EmbeddingProvider:
    """Interface for embedding providers."""
    
    async def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding vector for text.
        
        Args:
            text: The text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        raise NotImplementedError("Embedding provider must implement get_embedding")
    
class DefaultEmbeddingProvider(EmbeddingProvider):
    """
    Default embedding provider that uses a simple hashing approach.
    
    Note: This is a fallback implementation when no specialized embedding model
    is available. For production use, implement a proper embedding provider that
    uses a proper embedding model (e.g., OpenAI's text-embedding-ada-002).
    """
    
    def __init__(self, vector_dim: int = 384):
        """
        Initialize default embedding provider.
        
        Args:
            vector_dim: Dimension of the output embedding vector
        """
        self.vector_dim = vector_dim
        
    async def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding vector for text using simple hashing.
        
        Args:
            text: The text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        # Create a pseudo-embedding based on hash of chunks of the text
        # This is NOT a proper semantic embedding, but a placeholder
        if not text:
            return [0.0] * self.vector_dim
            
        # Normalize text
        text = text.strip().lower()
        
        # Get SHA-256 hash of text
        hash_obj = hashlib.sha256(text.encode('utf-8'))
        hash_bytes = hash_obj.digest()
        
        # Convert hash bytes to vector of floats (normalize to -1 to 1)
        vector = []
        for i in range(self.vector_dim):
            # Take modulo to handle case where hash_bytes is shorter than vector_dim
            byte_val = hash_bytes[i % len(hash_bytes)]
            val = (byte_val / 127.5) - 1.0
            vector.append(val)
            
        # Normalize vector to unit length
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = [v / norm for v in vector]
            
        return vector

class CacheEntry(BaseModel):
    """Entry in semantic cache."""
    
    request_id: str
    embedding: List[float]
    response: Dict[str, Any]  # Serialized LLMResponse
    prompt_hash: str
    model_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    access_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SemanticCache:
    """
    Cache for LLM responses based on semantic similarity.
    
    This cache stores responses along with embeddings of the prompts,
    allowing for retrieval based on semantic similarity rather than
    exact matches.
    """
    
    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        similarity_threshold: float = 0.92,
        max_entries: int = 10000,
        ttl_seconds: int = 3600,
        enable_exact_match: bool = True,
        cache_lock: Optional[asyncio.Lock] = None,
        persistent_store: Optional['BaseCacheStore'] = None
    ):
        """
        Initialize semantic cache.
        
        Args:
            embedding_provider: Provider for generating embeddings
            similarity_threshold: Minimum similarity score for cache hit (0-1)
            max_entries: Maximum number of entries to store in cache
            ttl_seconds: Time-to-live for cache entries in seconds
            enable_exact_match: Whether to also check for exact matches (faster)
            cache_lock: Lock for thread safety (created if not provided)
            persistent_store: Optional persistent storage backend
        """
        self.embedding_provider = embedding_provider or DefaultEmbeddingProvider()
        self.similarity_threshold = max(0.0, min(1.0, similarity_threshold))
        self.max_entries = max(1, max_entries)
        self.ttl_seconds = max(1, ttl_seconds)
        self.enable_exact_match = enable_exact_match
        self.cache_lock = cache_lock or asyncio.Lock()
        self.persistent_store = persistent_store
        
        # Cache storage
        self.entries_by_id: Dict[str, CacheEntry] = {}  # Primary storage
        self.exact_match_index: Dict[str, str] = {}  # Maps prompt_hash to entry_id
        self.model_index: Dict[str, Set[str]] = {}  # Maps model_id to set of entry_ids
        
        # Metrics
        self.hit_count = 0
        self.miss_count = 0
        self.semantic_hit_count = 0
        self.exact_hit_count = 0
        
        # Load cache from persistent store if available
        if self.persistent_store:
            logger.info("Persistent cache store provided, will load entries on initialization")
        
        logger.info(
            f"Initialized semantic cache with threshold={similarity_threshold}, "
            f"max_entries={max_entries}, ttl={ttl_seconds}s, "
            f"persistence={'enabled' if persistent_store else 'disabled'}"
        )
    
    async def initialize(self) -> None:
        """Initialize the cache, loading entries from persistent store if available."""
        if not self.persistent_store:
            return
            
        try:
            # Load metadata first
            metadata = await self.persistent_store.load_metadata()
            if metadata:
                # Update metrics from metadata
                self.hit_count = metadata.get("hit_count", 0)
                self.miss_count = metadata.get("miss_count", 0)
                self.semantic_hit_count = metadata.get("semantic_hit_count", 0)
                self.exact_hit_count = metadata.get("exact_hit_count", 0)
                
                logger.info(
                    f"Loaded cache metrics from persistent store: "
                    f"{self.hit_count} hits, {self.miss_count} misses"
                )
            
            # Load entries from persistent store
            entry_ids = await self.persistent_store.list_entries()
            if not entry_ids:
                logger.info("No entries found in persistent store")
                return
                
            logger.info(f"Loading {len(entry_ids)} entries from persistent store")
            loaded_count = 0
            
            for entry_id in entry_ids:
                try:
                    entry = await self.persistent_store.load_entry(entry_id)
                    if entry and not self._is_entry_expired(entry):
                        # Add to in-memory cache
                        self.entries_by_id[entry_id] = entry
                        
                        # Update exact match index
                        self.exact_match_index[entry.prompt_hash] = entry_id
                        
                        # Update model index
                        if entry.model_id not in self.model_index:
                            self.model_index[entry.model_id] = set()
                        self.model_index[entry.model_id].add(entry_id)
                        
                        loaded_count += 1
                except Exception as e:
                    logger.warning(f"Error loading cache entry {entry_id}: {str(e)}")
            
            logger.info(f"Successfully loaded {loaded_count} entries from persistent store")
        except Exception as e:
            logger.error(f"Error initializing cache from persistent store: {str(e)}")
            
    async def save_metrics(self) -> None:
        """Save cache metrics to persistent store."""
        if not self.persistent_store:
            return
            
        try:
            metadata = {
                "hit_count": self.hit_count,
                "miss_count": self.miss_count,
                "semantic_hit_count": self.semantic_hit_count,
                "exact_hit_count": self.exact_hit_count,
                "last_updated": datetime.utcnow().isoformat()
            }
            
            await self.persistent_store.save_metadata(metadata)
        except Exception as e:
            logger.warning(f"Error saving cache metrics: {str(e)}")
        
    def _get_prompt_hash(self, request: LLMRequest) -> str:
        """
        Get hash of prompt for exact matching.
        
        Args:
            request: LLM request
            
        Returns:
            Hash of prompt
        """
        # Convert prompt content to string if it's not already
        if isinstance(request.prompt_content, str):
            content = request.prompt_content
        elif isinstance(request.prompt_content, list):
            # Handle list of content items
            content = json.dumps(request.prompt_content, sort_keys=True)
        else:
            content = str(request.prompt_content)
            
        # Include system prompt in hash if present
        if request.config.system_prompt:
            content = f"{request.config.system_prompt}\n{content}"
            
        # Include conversation history in hash if present
        if request.initial_context and request.initial_context.conversation_history:
            history = json.dumps(
                [
                    {"role": turn.role, "content": str(turn.content)}
                    for turn in request.initial_context.conversation_history
                ],
                sort_keys=True
            )
            content = f"{history}\n{content}"
            
        # Include model parameters that affect output
        params = {
            "temperature": request.config.temperature,
            "top_p": request.config.top_p,
            "max_tokens": request.config.max_tokens,
        }
        param_str = json.dumps(params, sort_keys=True)
        content = f"{content}\n{param_str}"
            
        # Return SHA-256 hash
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def _extract_prompt_text(self, request: LLMRequest) -> str:
        """
        Extract text content from prompt for embedding.
        
        Args:
            request: LLM request
            
        Returns:
            Text content of prompt
        """
        # Start with prompt content
        if isinstance(request.prompt_content, str):
            text = request.prompt_content
        elif isinstance(request.prompt_content, list):
            # Extract text from content items list
            texts = []
            for item in request.prompt_content:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    texts.append(item["text"])
                elif hasattr(item, "text") and item.text:
                    texts.append(item.text)
            text = " ".join(texts)
        else:
            text = str(request.prompt_content)
            
        # Add system prompt if available
        if request.config.system_prompt:
            text = f"{request.config.system_prompt}\n{text}"
            
        # Add conversation history if available
        if request.initial_context and request.initial_context.conversation_history:
            history_texts = []
            for turn in request.initial_context.conversation_history:
                if isinstance(turn.content, str):
                    history_texts.append(turn.content)
                elif isinstance(turn.content, list):
                    for item in turn.content:
                        if isinstance(item, str):
                            history_texts.append(item)
                        elif isinstance(item, dict) and "text" in item:
                            history_texts.append(item["text"])
            
            if history_texts:
                text = f"{' '.join(history_texts)}\n{text}"
                
        return text
        
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity (between -1 and 1)
        """
        # Convert to numpy arrays for efficient computation
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        # Compute cosine similarity
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        # Avoid division by zero
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        
        return dot_product / (norm_v1 * norm_v2)
    
    def _find_most_similar_entry(
        self, 
        embedding: List[float], 
        model_id: str
    ) -> Tuple[Optional[CacheEntry], float]:
        """
        Find the most similar cache entry for a given embedding.
        
        Args:
            embedding: Embedding vector to match against
            model_id: Model identifier to filter entries
            
        Returns:
            Tuple of (most similar entry or None, similarity score)
        """
        best_entry = None
        best_score = -1.0
        
        # Get entry IDs for the model
        entry_ids = self.model_index.get(model_id, set())
        if not entry_ids:
            return None, 0.0
        
        # Find most similar entry
        for entry_id in entry_ids:
            entry = self.entries_by_id.get(entry_id)
            if not entry:
                continue
                
            score = self._cosine_similarity(embedding, entry.embedding)
            if score > best_score:
                best_score = score
                best_entry = entry
                
        return best_entry, best_score
    
    def _deserialize_response(self, serialized: Dict[str, Any]) -> LLMResponse:
        """
        Deserialize LLMResponse from dictionary.
        
        Args:
            serialized: Serialized response
            
        Returns:
            Deserialized LLMResponse
        """
        # Reconstruct LLMResponse from serialized data
        from asf.conexus.llm_gateway.core.models import LLMResponse
        
        # Create UsageStats if available
        usage = None
        if "usage" in serialized and serialized["usage"]:
            usage = UsageStats(
                prompt_tokens=serialized["usage"].get("prompt_tokens", 0),
                completion_tokens=serialized["usage"].get("completion_tokens", 0),
                total_tokens=serialized["usage"].get("total_tokens", 0)
            )
            
        # Create PerformanceMetrics if available
        perf_metrics = None
        if "performance_metrics" in serialized and serialized["performance_metrics"]:
            perf_metrics = PerformanceMetrics(
                total_duration_ms=serialized["performance_metrics"].get("total_duration_ms", 0),
                llm_latency_ms=serialized["performance_metrics"].get("llm_latency_ms", 0),
                gateway_overhead_ms=serialized["performance_metrics"].get("gateway_overhead_ms", 0)
            )
        
        # Create response
        response = LLMResponse(
            request_id=serialized["request_id"],
            generated_content=serialized["generated_content"],
            finish_reason=serialized.get("finish_reason"),
            usage=usage,
            performance_metrics=perf_metrics,
            # Add any other fields as needed
        )
        
        return response
    
    def _serialize_response(self, response: LLMResponse) -> Dict[str, Any]:
        """
        Serialize LLMResponse to dictionary for storage.
        
        Args:
            response: LLM response
            
        Returns:
            Serialized response as dictionary
        """
        serialized = {
            "request_id": response.request_id,
            "generated_content": response.generated_content,
            "finish_reason": response.finish_reason,
        }
        
        # Serialize usage if available
        if response.usage:
            serialized["usage"] = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens if hasattr(response.usage, 'total_tokens') else 
                                (response.usage.prompt_tokens + response.usage.completion_tokens),
            }
            
        # Serialize performance metrics if available
        if response.performance_metrics:
            serialized["performance_metrics"] = {
                "total_duration_ms": response.performance_metrics.total_duration_ms,
                "llm_latency_ms": response.performance_metrics.llm_latency_ms,
                "gateway_overhead_ms": response.performance_metrics.gateway_overhead_ms 
                                      if hasattr(response.performance_metrics, 'gateway_overhead_ms') else 0,
            }
            
        return serialized
    
    def _is_entry_expired(self, entry: CacheEntry) -> bool:
        """
        Check if a cache entry has expired.
        
        Args:
            entry: Cache entry to check
            
        Returns:
            True if entry has expired, False otherwise
        """
        now = datetime.utcnow()
        age_seconds = (now - entry.last_accessed).total_seconds()
        return age_seconds > self.ttl_seconds
    
    def _clean_expired_entries(self) -> None:
        """Clean expired entries from cache."""
        expired_ids = []
        
        # Find expired entries
        for entry_id, entry in self.entries_by_id.items():
            if self._is_entry_expired(entry):
                expired_ids.append(entry_id)
                
        # Remove expired entries
        for entry_id in expired_ids:
            self._remove_entry(entry_id)
            
    async def _remove_entry(self, entry_id: str) -> None:
        """
        Remove entry from cache.
        
        Args:
            entry_id: ID of entry to remove
        """
        if entry_id not in self.entries_by_id:
            return
            
        # Get entry
        entry = self.entries_by_id[entry_id]
        
        # Remove from exact match index
        prompt_hash = entry.prompt_hash
        if prompt_hash in self.exact_match_index and self.exact_match_index[prompt_hash] == entry_id:
            del self.exact_match_index[prompt_hash]
            
        # Remove from model index
        model_id = entry.model_id
        if model_id in self.model_index and entry_id in self.model_index[model_id]:
            self.model_index[model_id].remove(entry_id)
            if not self.model_index[model_id]:
                del self.model_index[model_id]
                
        # Remove from main storage
        del self.entries_by_id[entry_id]
        
        # Remove from persistent store if available
        if self.persistent_store:
            try:
                await self.persistent_store.delete_entry(entry_id)
            except Exception as e:
                logger.warning(f"Error removing entry {entry_id} from persistent store: {str(e)}")
        
    async def _make_room(self) -> None:
        """
        Make room in the cache by removing least recently used entries.
        Ensures that the total number of entries is below max_entries.
        """
        if len(self.entries_by_id) <= self.max_entries:
            return
            
        # Sort entries by last accessed time (oldest first)
        sorted_entries = sorted(
            self.entries_by_id.items(),
            key=lambda x: x[1].last_accessed
        )
        
        # Remove oldest entries until we're below max_entries
        entries_to_remove = len(sorted_entries) - self.max_entries
        for i in range(entries_to_remove):
            entry_id, _ = sorted_entries[i]
            await self._remove_entry(entry_id)
    
    async def get(
        self, 
        request: LLMRequest,
        check_exact_only: bool = False
    ) -> Optional[LLMResponse]:
        """
        Get response from cache if available.
        
        Args:
            request: LLM request
            check_exact_only: If True, only check for exact matches (faster)
            
        Returns:
            Cached response or None if not found
        """
        model_id = request.config.model_identifier
        start_time = time.time()
        
        async with self.cache_lock:
            # Step 1: Clean expired entries
            self._clean_expired_entries()
            
            # Step 2: Try exact match if enabled (faster path)
            if self.enable_exact_match:
                prompt_hash = self._get_prompt_hash(request)
                if prompt_hash in self.exact_match_index:
                    entry_id = self.exact_match_index[prompt_hash]
                    if entry_id in self.entries_by_id:
                        entry = self.entries_by_id[entry_id]
                        
                        # Check if models match
                        if entry.model_id != model_id:
                            self.miss_count += 1
                            await self.save_metrics()
                            return None
                            
                        # Check if expired
                        if self._is_entry_expired(entry):
                            await self._remove_entry(entry_id)
                            self.miss_count += 1
                            await self.save_metrics()
                            return None
                        
                        # Update stats
                        entry.last_accessed = datetime.utcnow()
                        entry.access_count += 1
                        self.hit_count += 1
                        self.exact_hit_count += 1
                        
                        # Update entry in persistent store if available
                        if self.persistent_store:
                            try:
                                await self.persistent_store.save_entry(entry_id, entry)
                            except Exception as e:
                                logger.warning(f"Error updating entry {entry_id} in persistent store: {str(e)}")
                        
                        # Save updated metrics
                        await self.save_metrics()
                        
                        # Deserialize and return response
                        logger.info(
                            f"Cache hit (exact match): {request.config.model_identifier}, "
                            f"request_id={request.initial_context.request_id}, "
                            f"duration={int((time.time() - start_time) * 1000)}ms"
                        )
                        return self._deserialize_response(entry.response)
            
            # Stop here if exact match only
            if check_exact_only:
                self.miss_count += 1
                await self.save_metrics()
                return None
                
            # Step 3: Try semantic match
            # Get embedding for prompt
            prompt_text = self._extract_prompt_text(request)
            embedding = await self.embedding_provider.get_embedding(prompt_text)
            
            # Find most similar entry
            similar_entry, similarity = self._find_most_similar_entry(embedding, model_id)
            
            # Check if similar enough
            if similar_entry and similarity >= self.similarity_threshold:
                # Update stats
                similar_entry.last_accessed = datetime.utcnow()
                similar_entry.access_count += 1
                self.hit_count += 1
                self.semantic_hit_count += 1
                
                # Update entry in persistent store if available
                entry_id = f"{similar_entry.model_id}:{similar_entry.request_id}"
                if self.persistent_store:
                    try:
                        await self.persistent_store.save_entry(entry_id, similar_entry)
                    except Exception as e:
                        logger.warning(
                            f"Error updating entry {entry_id} in persistent store: {str(e)}"
                        )
                
                # Save updated metrics
                await self.save_metrics()
                
                # Log hit
                logger.info(
                    f"Cache hit (semantic match): {request.config.model_identifier}, "
                    f"similarity={similarity:.4f}, "
                    f"request_id={request.initial_context.request_id}, "
                    f"duration={int((time.time() - start_time) * 1000)}ms"
                )
                
                # Deserialize and return response
                return self._deserialize_response(similar_entry.response)
            
            # No match found
            self.miss_count += 1
            await self.save_metrics()
            return None
    
    async def store(self, request: LLMRequest, response: LLMResponse) -> None:
        """
        Store response in cache.
        
        Args:
            request: Original LLM request
            response: LLM response to cache
        """
        if not response or not request:
            return
            
        # Skip caching if no content or error response
        if not response.generated_content:
            return
            
        model_id = request.config.model_identifier
        prompt_hash = self._get_prompt_hash(request)
        entry_id = f"{model_id}:{response.request_id}"
        
        async with self.cache_lock:
            # Make room if needed
            await self._make_room()
            
            # Get embedding for prompt
            prompt_text = self._extract_prompt_text(request)
            embedding = await self.embedding_provider.get_embedding(prompt_text)
            
            # Serialize response for storage
            serialized_response = self._serialize_response(response)
            
            # Get token counts
            prompt_tokens = 0
            completion_tokens = 0
            if response.usage:
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
            
            # Create cache entry
            entry = CacheEntry(
                request_id=response.request_id,
                embedding=embedding,
                response=serialized_response,
                prompt_hash=prompt_hash,
                model_id=model_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                metadata={
                    "temperature": request.config.temperature,
                    "max_tokens": request.config.max_tokens,
                    "cached_at": datetime.utcnow().isoformat(),
                }
            )
            
            # Store entry in memory
            self.entries_by_id[entry_id] = entry
            
            # Update exact match index
            self.exact_match_index[prompt_hash] = entry_id
            
            # Update model index
            if model_id not in self.model_index:
                self.model_index[model_id] = set()
            self.model_index[model_id].add(entry_id)
            
            # Store entry in persistent store if available
            if self.persistent_store:
                try:
                    await self.persistent_store.save_entry(entry_id, entry)
                except Exception as e:
                    logger.warning(f"Error storing entry {entry_id} in persistent store: {str(e)}")
            
            logger.info(
                f"Cached response for {model_id}, "
                f"request_id={response.request_id}, "
                f"tokens={prompt_tokens}+{completion_tokens}"
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        hit_rate = 0.0
        total_requests = self.hit_count + self.miss_count
        if total_requests > 0:
            hit_rate = self.hit_count / total_requests
            
        semantic_percentage = 0.0
        if self.hit_count > 0:
            semantic_percentage = self.semantic_hit_count / self.hit_count
            
        # Calculate estimated token savings
        token_savings = 0
        for entry in self.entries_by_id.values():
            # Each access after the first is a "saving"
            if entry.access_count > 1:
                token_savings += (entry.prompt_tokens + entry.completion_tokens) * (entry.access_count - 1)
                
        stats = {
            "total_entries": len(self.entries_by_id),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "semantic_hit_count": self.semantic_hit_count,
            "exact_hit_count": self.exact_hit_count,
            "hit_rate": hit_rate,
            "semantic_percentage": semantic_percentage,
            "max_entries": self.max_entries,
            "ttl_seconds": self.ttl_seconds,
            "similarity_threshold": self.similarity_threshold,
            "estimated_token_savings": token_savings,
            "persistence_enabled": self.persistent_store is not None
        }
        
        # Get model-specific stats
        model_stats = {}
        for model_id, entry_ids in self.model_index.items():
            model_entries = [self.entries_by_id[entry_id] for entry_id in entry_ids]
            model_token_savings = 0
            total_tokens = 0
            
            for entry in model_entries:
                entry_tokens = entry.prompt_tokens + entry.completion_tokens
                total_tokens += entry_tokens
                if entry.access_count > 1:
                    model_token_savings += entry_tokens * (entry.access_count - 1)
            
            model_stats[model_id] = {
                "entries": len(entry_ids),
                "total_tokens": total_tokens,
                "estimated_token_savings": model_token_savings
            }
        
        stats["model_stats"] = model_stats
        
        return stats
    
    async def clear(self) -> None:
        """Clear the cache."""
        async with self.cache_lock:
            self.entries_by_id = {}
            self.exact_match_index = {}
            self.model_index = {}
            
            # Clear persistent store if available
            if self.persistent_store:
                try:
                    await self.persistent_store.clear()
                except Exception as e:
                    logger.warning(f"Error clearing persistent store: {str(e)}")
            
            logger.info("Semantic cache cleared")
    
    async def close(self) -> None:
        """Close the cache and release resources."""
        # Save metrics before closing
        await self.save_metrics()
        
        # Close persistent store if available
        if self.persistent_store:
            try:
                await self.persistent_store.close()
            except Exception as e:
                logger.warning(f"Error closing persistent store: {str(e)}")
                
        logger.info("Semantic cache closed")