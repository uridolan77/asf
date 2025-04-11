"""
Model Cache
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
    """Memory statistics."""
    total: int = 0  # Total memory in bytes
    available: int = 0  # Available memory in bytes
    used: int = 0  # Used memory in bytes
    percent: float = 0.0  # Percent of memory used
    gpu_total: int = 0  # Total GPU memory in bytes
    gpu_available: int = 0  # Available GPU memory in bytes
    gpu_used: int = 0  # Used GPU memory in bytes
    gpu_percent: float = 0.0  # Percent of GPU memory used
class ModelCache(Generic[T]):
    """
    Cache for ML models.
    This class provides caching for ML models to improve performance by keeping
    models in memory and unloading them when they are not used for a while.
        Initialize the model cache.
        Args:
            max_models: Maximum number of models to keep in cache (default: 5)
            ttl: Time to live in seconds (default: 3600 = 1 hour)
            check_interval: Interval in seconds for checking expired models (default: 300 = 5 minutes)
            memory_threshold: Memory usage threshold (0.0-1.0) for triggering cleanup (default: 0.9 = 90%)
            gpu_memory_threshold: GPU memory usage threshold (0.0-1.0) for triggering cleanup (default: 0.9 = 90%)
        if self.cleanup_thread is None or not self.cleanup_thread.is_alive():
            self.running = True
            self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
            self.cleanup_thread.start()
            logger.info("Model cache cleanup thread started")
    def _cleanup_loop(self):
        """Cleanup loop for removing expired models.
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
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
        """Remove expired models from cache.
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
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
        """Make room for a new model by removing the least recently used model.
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
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
            model_id: Model ID
        Returns:
            Model or None if not found
        Put a model in cache.
        Args:
            model_id: Model ID
            model: Model to cache
            metadata: Optional metadata
        Remove a model from cache.
        Args:
            model_id: Model ID
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
        Get memory statistics.
        Returns:
            MemoryStats: Memory statistics
        Clean up models to free memory.
        This method removes models from the cache to free memory when memory usage
        is above the threshold.
        Get cache statistics.
        Returns:
            Cache statistics
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description