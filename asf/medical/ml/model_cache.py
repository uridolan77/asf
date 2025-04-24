"""Model Cache

This module provides caching for ML models to improve performance and manage resources.
It implements a least-recently-used (LRU) cache with memory monitoring to prevent
out-of-memory errors when loading large models.
"""

import time
import logging
import threading
from typing import Dict, Any, Optional, Callable, TypeVar, Generic
from dataclasses import dataclass

logger = logging.getLogger(__name__)
T = TypeVar('T')

@dataclass
class MemoryStats:
    """Memory statistics for system and GPU resources.
    
    Attributes:
        total: Total system memory in bytes
        available: Available system memory in bytes
        used: Used system memory in bytes
        percent: Percentage of system memory used (0.0-1.0)
        gpu_total: Total GPU memory in bytes
        gpu_available: Available GPU memory in bytes
        gpu_used: Used GPU memory in bytes
        gpu_percent: Percentage of GPU memory used (0.0-1.0)
    """
    total: int = 0  # Total memory in bytes
    available: int = 0  # Available memory in bytes
    used: int = 0  # Used memory in bytes
    percent: float = 0.0  # Percent of memory used
    gpu_total: int = 0  # Total GPU memory in bytes
    gpu_available: int = 0  # Available GPU memory in bytes
    gpu_used: int = 0  # Used GPU memory in bytes
    gpu_percent: float = 0.0  # Percent of GPU memory used

class ModelCache(Generic[T]):
    """Cache for ML models with automatic resource management.
    
    This class provides caching for ML models to improve performance by keeping
    models in memory and unloading them when they are not used for a while.
    It implements a least-recently-used (LRU) cache with memory monitoring to prevent
    out-of-memory errors when loading large models.
    
    Features:
        - Automatic unloading of expired models based on TTL
        - Memory usage monitoring and cleanup
        - GPU memory usage monitoring and cleanup
        - Thread-safe operations
    """

    def __init__(
        self,
        max_models: int = 5,
        ttl: int = 3600,
        check_interval: int = 300,
        memory_threshold: float = 0.9,
        gpu_memory_threshold: float = 0.9
    ):
        """
        Initialize the model cache.

        Args:
            max_models: Maximum number of models to keep in cache (default: 5)
            ttl: Time to live in seconds (default: 3600 = 1 hour)
            check_interval: Interval in seconds for checking expired models (default: 300 = 5 minutes)
            memory_threshold: Memory usage threshold (0.0-1.0) for triggering cleanup (default: 0.9 = 90%)
            gpu_memory_threshold: GPU memory usage threshold (0.0-1.0) for triggering cleanup (default: 0.9 = 90%)
        """
        self.max_models = max_models
        self.ttl = ttl
        self.check_interval = check_interval
        self.memory_threshold = memory_threshold
        self.gpu_memory_threshold = gpu_memory_threshold
        self.models = {}
        self.lock = threading.RLock()
        self.cleanup_thread = None
        self.running = False
        self.has_gpu = False  # Will be set by _get_memory_stats

        # Start cleanup thread
        if self.cleanup_thread is None or not self.cleanup_thread.is_alive():
            self.running = True
            self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
            self.cleanup_thread.start()
            logger.info("Model cache cleanup thread started")

    def _cleanup_loop(self):
        """
        Background thread that periodically checks for expired models and memory usage.

        This method runs in a separate thread and periodically checks for expired models
        and memory usage, removing models as needed to keep memory usage below thresholds.
        """

        while self.running:
            try:
                memory_stats = self._get_memory_stats()
                logger.debug(f"Memory usage: {memory_stats.percent:.1%}, GPU: {memory_stats.gpu_percent:.1%}")
                if memory_stats.percent > self.memory_threshold:
                    logger.warning(f"Memory usage above threshold: {memory_stats.percent:.1%} > {self.memory_threshold:.1%}")
                    self._cleanup_for_memory()
                if self.has_gpu and memory_stats.gpu_percent > self.gpu_memory_threshold:
                    logger.warning(f"GPU memory usage above threshold: {memory_stats.gpu_percent:.1%} > {self.gpu_memory_threshold:.1%}")
                    self._cleanup_for_memory()
                self._cleanup_expired_models()
            except Exception as e:
                logger.error(f"Error in model cache cleanup: {str(e)}")
            time.sleep(self.check_interval)

    def _cleanup_expired_models(self):
        """
        Remove models that have exceeded their time-to-live (TTL) from the cache.

        This method checks all models in the cache and removes those that have not been
        used for longer than the TTL period.
        """

        with self.lock:
            current_time = time.time()
            expired_models = []
            for model_id, model_info in self.models.items():
                last_used = model_info.get("last_used", 0)
                if current_time - last_used > self.ttl:
                    expired_models.append(model_id)
            for model_id in expired_models:
                self._remove_model(model_id)
            if expired_models:
                logger.info(f"Removed {len(expired_models)} expired models from cache")

    def _remove_model(self, model_id: str):
        """
        Remove a model from cache.
        Args:
            model_id: Model ID
        """
        if model_id in self.models:
            model_info = self.models[model_id]
            model = model_info.get("model")
            if hasattr(model, "unload") and callable(getattr(model, "unload")):
                try:
                    model.unload()
                    logger.info(f"Unloaded model: {model_id}")
                except Exception as e:
                    logger.error(f"Error unloading model {model_id}: {str(e)}")
            del self.models[model_id]
            logger.info(f"Removed model from cache: {model_id}")

    def _make_room(self):
        """
        Make room for a new model by removing the least recently used model.

        This method is called when the cache is full and a new model needs to be added.
        It removes the least recently used model to make room for the new one.
        """

        with self.lock:
            if len(self.models) >= self.max_models:
                lru_model_id = None
                lru_time = float('inf')
                for model_id, model_info in self.models.items():
                    last_used = model_info.get("last_used", 0)
                    if last_used < lru_time:
                        lru_time = last_used
                        lru_model_id = model_id
                if lru_model_id:
                    self._remove_model(lru_model_id)
                    logger.info(f"Removed least recently used model to make room: {lru_model_id}")

    def get(self, model_id: str) -> Optional[T]:
        """
        Get a model from cache.

        Args:
            model_id: Unique identifier for the model

        Returns:
            The cached model if found, None otherwise
        """
        with self.lock:
            if model_id in self.models:
                model_info = self.models[model_id]
                model_info["last_used"] = time.time()
                return model_info["model"]
            return None

    def put(self, model_id: str, model: T, metadata: Optional[Dict[str, Any]] = None):
        """
        Put a model in cache.

        Args:
            model_id: Unique identifier for the model
            model: The model instance to cache
            metadata: Optional metadata about the model (e.g., memory usage)
        """
        with self.lock:
            if len(self.models) >= self.max_models and model_id not in self.models:
                self._make_room()

            self.models[model_id] = {
                "model": model,
                "loaded_at": time.time(),
                "last_used": time.time(),
                "metadata": metadata or {}
            }
            logger.info(f"Added model to cache: {model_id}")

    def remove(self, model_id: str):
        """
        Remove a model from cache.

        Args:
            model_id: Unique identifier for the model to remove
        """
        with self.lock:
            if model_id in self.models:
                self._remove_model(model_id)

    def clear(self):
        """Clear all models from the cache.
        
        Args:
        """
        with self.lock:
            model_ids = list(self.models.keys())
            for model_id in model_ids:
                self._remove_model(model_id)
            logger.info("Cleared model cache")

    def get_or_create(
        self,
        model_id: str,
        factory: Callable[[], T],
        metadata: Optional[Dict[str, Any]] = None
    ) -> T:
        """
        Get a model from cache or create it if not found.

        Args:
            model_id: Unique identifier for the model
            factory: Function that creates the model if not found in cache
            metadata: Optional metadata about the model (e.g., memory usage)

        Returns:
            The cached or newly created model
        """
        model = self.get(model_id)
        if model is not None:
            logger.debug(f"Model cache hit: {model_id}")
            return model
        logger.debug(f"Model cache miss: {model_id}")
        model = factory()
        self.put(model_id, model, metadata)
        return model

    def _get_memory_stats(self) -> MemoryStats:
        """
        Get current memory statistics for system and GPU.

        Returns:
            MemoryStats object with current memory usage information
        """
        # Implementation would get actual memory stats
        # This is a placeholder that would be implemented with psutil and GPU libraries
        return MemoryStats()

    def _cleanup_for_memory(self):
        """
        Clean up models to free memory when usage is above threshold.

        This method removes models from the cache starting with the least recently used
        to reduce memory usage when it exceeds the configured threshold.
        """
        with self.lock:
            if not self.models:
                return

            # Remove least recently used models until we're below threshold or cache is empty
            while self.models:
                memory_stats = self._get_memory_stats()
                if memory_stats.percent < self.memory_threshold and \
                   (not self.has_gpu or memory_stats.gpu_percent < self.gpu_memory_threshold):
                    break

                # Find least recently used model
                lru_model_id = None
                lru_time = float('inf')
                for model_id, model_info in self.models.items():
                    last_used = model_info.get("last_used", 0)
                    if last_used < lru_time:
                        lru_time = last_used
                        lru_model_id = model_id

                if lru_model_id:
                    self._remove_model(lru_model_id)
                    logger.info(f"Removed model to free memory: {lru_model_id}")
                else:
                    break

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics including size, max size, TTL, and models
        """
        with self.lock:
            return {
                "size": len(self.models),
                "max_size": self.max_models,
                "ttl": self.ttl,
                "models": [
                    {
                        "id": model_id,
                        "loaded_at": model_info.get("loaded_at", 0),
                        "last_used": model_info.get("last_used", 0),
                        "metadata": model_info.get("metadata", {})
                    }
                    for model_id, model_info in self.models.items()
                ]
            }

# Create a singleton instance
model_cache = ModelCache()