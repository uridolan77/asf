"""
Model Cache

This module provides caching for ML models to improve performance.
"""

import os
import time
import logging
import threading
from typing import Dict, Any, Optional, Callable, Type, TypeVar, Generic, List

from asf.medical.core.config import settings

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for model classes
T = TypeVar('T')

class ModelCache(Generic[T]):
    """
    Cache for ML models.
    
    This class provides caching for ML models to improve performance by keeping
    models in memory and unloading them when they are not used for a while.
    """
    
    def __init__(
        self,
        max_models: int = 5,
        ttl: int = 3600,  # 1 hour
        check_interval: int = 300  # 5 minutes
    ):
        """
        Initialize the model cache.
        
        Args:
            max_models: Maximum number of models to keep in cache (default: 5)
            ttl: Time to live in seconds (default: 3600 = 1 hour)
            check_interval: Interval in seconds for checking expired models (default: 300 = 5 minutes)
        """
        self.max_models = int(os.environ.get("MAX_CACHED_MODELS", max_models))
        self.ttl = int(os.environ.get("MODEL_CACHE_TTL", ttl))
        self.check_interval = int(os.environ.get("MODEL_CACHE_CHECK_INTERVAL", check_interval))
        
        self.models: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.RLock()
        self.cleanup_thread = None
        self.running = False
        
        # Start cleanup thread
        self._start_cleanup_thread()
    
    def _start_cleanup_thread(self):
        """Start the cleanup thread."""
        if self.cleanup_thread is None or not self.cleanup_thread.is_alive():
            self.running = True
            self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
            self.cleanup_thread.start()
            logger.info("Model cache cleanup thread started")
    
    def _cleanup_loop(self):
        """Cleanup loop for removing expired models."""
        while self.running:
            try:
                self._cleanup_expired_models()
            except Exception as e:
                logger.error(f"Error in model cache cleanup: {str(e)}")
            
            # Sleep for check interval
            time.sleep(self.check_interval)
    
    def _cleanup_expired_models(self):
        """Remove expired models from cache."""
        with self.lock:
            current_time = time.time()
            expired_models = []
            
            # Find expired models
            for model_id, model_info in self.models.items():
                last_used = model_info.get("last_used", 0)
                if current_time - last_used > self.ttl:
                    expired_models.append(model_id)
            
            # Remove expired models
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
            
            # Call unload method if available
            if hasattr(model, "unload") and callable(getattr(model, "unload")):
                try:
                    model.unload()
                    logger.info(f"Unloaded model: {model_id}")
                except Exception as e:
                    logger.error(f"Error unloading model {model_id}: {str(e)}")
            
            # Remove from cache
            del self.models[model_id]
            logger.info(f"Removed model from cache: {model_id}")
    
    def _make_room(self):
        """Make room for a new model by removing the least recently used model."""
        with self.lock:
            if len(self.models) >= self.max_models:
                # Find least recently used model
                lru_model_id = None
                lru_time = float('inf')
                
                for model_id, model_info in self.models.items():
                    last_used = model_info.get("last_used", 0)
                    if last_used < lru_time:
                        lru_time = last_used
                        lru_model_id = model_id
                
                # Remove least recently used model
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
        """
        with self.lock:
            if model_id in self.models:
                # Update last used time
                self.models[model_id]["last_used"] = time.time()
                
                # Return model
                return self.models[model_id]["model"]
            
            return None
    
    def put(self, model_id: str, model: T, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Put a model in cache.
        
        Args:
            model_id: Model ID
            model: Model to cache
            metadata: Optional metadata
        """
        with self.lock:
            # Make room if needed
            if len(self.models) >= self.max_models:
                self._make_room()
            
            # Add model to cache
            self.models[model_id] = {
                "model": model,
                "last_used": time.time(),
                "metadata": metadata or {}
            }
            
            logger.info(f"Added model to cache: {model_id}")
    
    def remove(self, model_id: str) -> None:
        """
        Remove a model from cache.
        
        Args:
            model_id: Model ID
        """
        with self.lock:
            self._remove_model(model_id)
    
    def clear(self) -> None:
        """Clear all models from cache."""
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
            model_id: Model ID
            factory: Factory function for creating the model
            metadata: Optional metadata
            
        Returns:
            Model
        """
        # Try to get from cache
        model = self.get(model_id)
        if model is not None:
            logger.debug(f"Model cache hit: {model_id}")
            return model
        
        # Create model
        logger.debug(f"Model cache miss: {model_id}")
        model = factory()
        
        # Put in cache
        self.put(model_id, model, metadata)
        
        return model
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Cache statistics
        """
        with self.lock:
            return {
                "size": len(self.models),
                "max_size": self.max_models,
                "ttl": self.ttl,
                "models": [
                    {
                        "id": model_id,
                        "last_used": model_info.get("last_used", 0),
                        "age": time.time() - model_info.get("last_used", 0),
                        "metadata": model_info.get("metadata", {})
                    }
                    for model_id, model_info in self.models.items()
                ]
            }
    
    def shutdown(self) -> None:
        """Shutdown the cache and cleanup resources."""
        self.running = False
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=1.0)
        
        self.clear()
        logger.info("Model cache shutdown")

# Create a singleton instance
model_cache = ModelCache()
