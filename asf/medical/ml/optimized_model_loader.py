"""
Optimized Model Loader for the Medical Research Synthesizer.
This module provides an optimized model loading system that efficiently
manages ML models in memory and supports sharing models across processes.
"""
import os
import time
import logging
import threading
import pickle
import uuid
from typing import Dict, Any, Optional, TypeVar
from pathlib import Path
import numpy as np
logger = logging.getLogger(__name__)
T = TypeVar('T')
class ModelInfo:
    """
    Information about a loaded model.
    """
    def __init__(
        self,
        model_id: str,
        model: Any,
        metadata: Dict[str, Any],
        loaded_at: float,
        last_used: float,
        shared_memory_id: Optional[str] = None
    ):
        self.model_id = model_id
        self.model = model
        self.metadata = metadata
        self.loaded_at = loaded_at
        self.last_used = last_used
        self.shared_memory_id = shared_memory_id
        self.use_count = 0
        self.lock = threading.RLock()
    def update_last_used(self):
        """Update the last used time.
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
    """
        self.last_used = time.time()
        self.use_count += 1
    def get_memory_usage(self) -> int:
        """
        Get the memory usage of the model in MB.
        Returns:
            int: Memory usage in MB
        Get the age of the model in seconds.
        Returns:
            float: Age in seconds
        Get the idle time of the model in seconds.
        Returns:
            float: Idle time in seconds
        Convert to dictionary.
        Returns:
            Dict[str, Any]: Dictionary representation
    Optimized model loader for the Medical Research Synthesizer.
    This class provides an optimized model loading system that efficiently
    manages ML models in memory and supports sharing models across processes.
        Create a singleton instance of the optimized model loader.
        Returns:
            OptimizedModelLoader: The singleton instance
        Initialize the optimized model loader.
        Args:
            max_models: Maximum number of models to keep in memory (default: 5)
            ttl: Time to live in seconds (default: 3600 = 1 hour)
            check_interval: Interval in seconds for checking expired models (default: 300 = 5 minutes)
            shared_memory_dir: Directory for shared memory files (default: system temp dir)
            use_shared_memory: Whether to use shared memory (default: False)
        if self.maintenance_task is not None:
            return
        self.running = True
        def maintenance_loop():
            while self.running:
                try:
                    self._check_expired_models()
                except Exception as e:
                    logger.error(f"Error in maintenance task: {str(e)}")
                time.sleep(self.check_interval)
        self.maintenance_task = threading.Thread(target=maintenance_loop, daemon=True)
        self.maintenance_task.start()
        logger.debug("Maintenance task started")
    def _check_expired_models(self):
        """Check for expired models and unload them.
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
        Unload a model from memory.
        Args:
            model_id: Model ID
        Register a factory function for creating a model.
        Args:
            model_id: Model ID
            factory: Factory function that creates the model
            model_type: Expected type of the model (for type checking)
            metadata: Model metadata
        Get a model from the cache.
        Args:
            model_id: Model ID
        Returns:
            Optional[Any]: Model instance or None if not found
        Put a model in the cache.
        Args:
            model_id: Model ID
            model: Model instance
            metadata: Model metadata
            use_shared_memory: Whether to use shared memory (default: self.use_shared_memory)
        Get a model from the cache or create it if not found.
        Args:
            model_id: Model ID
            factory: Factory function that creates the model
            metadata: Model metadata
            use_shared_memory: Whether to use shared memory (default: self.use_shared_memory)
        Returns:
            Any: Model instance
        Raises:
            ValueError: If the model is not found and no factory is provided
        Remove a model from the cache.
        Args:
            model_id: Model ID
        with self.lock:
            model_ids = list(self.models.keys())
            for model_id in model_ids:
                self._unload_model(model_id)
            logger.info(f"Cleared {len(model_ids)} models from cache")
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        Returns:
            Dict[str, Any]: Cache statistics
        """
        with self.lock:
            stats = {
                "max_models": self.max_models,
                "ttl": self.ttl,
                "check_interval": self.check_interval,
                "use_shared_memory": self.use_shared_memory,
                "shared_memory_dir": self.shared_memory_dir,
                "models": len(self.models),
                "factories": len(self.factories),
                "models_info": [model_info.to_dict() for model_info in self.models.values()],
                "total_memory_mb": sum(model_info.get_memory_usage() for model_info in self.models.values())
            }
            return stats
    def _evict_model(self):
        """
        Evict a model from the cache.
        This method selects the least recently used model and unloads it.
        """
        with self.lock:
            if not self.models:
                return
            lru_model_id = min(self.models.items(), key=lambda x: x[1].last_used)[0]
            self._unload_model(lru_model_id)
            logger.info(f"Evicted model: {lru_model_id}")
    def _save_to_shared_memory(self, model_id: str, model: Any) -> str:
        """
        Save a model to shared memory.
        Args:
            model_id: Model ID
            model: Model instance
        Returns:
            str: Shared memory ID
        Raises:
            Exception: If the model cannot be saved to shared memory
        """
        shared_memory_id = f"{model_id}_{uuid.uuid4().hex}.pkl"
        shared_memory_path = os.path.join(self.shared_memory_dir, shared_memory_id)
        with open(shared_memory_path, "wb") as f:
            pickle.dump(model, f)
        logger.debug(f"Saved model to shared memory: {shared_memory_path}")
        return shared_memory_id
    def _load_from_shared_memory(self, model_id: str) -> Optional[Any]:
        """
        Load a model from shared memory.
        Args:
            model_id: Model ID
        Returns:
            Optional[Any]: Model instance or None if not found
        Raises:
            Exception: If the model cannot be loaded from shared memory
        """
        shared_memory_pattern = f"{model_id}_*.pkl"
        shared_memory_dir = Path(self.shared_memory_dir)
        matching_files = list(shared_memory_dir.glob(shared_memory_pattern))
        if not matching_files:
            return None
        shared_memory_path = max(matching_files, key=lambda p: p.stat().st_mtime)
        with open(shared_memory_path, "rb") as f:
            model = pickle.load(f)
        logger.debug(f"Loaded model from shared memory: {shared_memory_path}")
        return model
    def __del__(self):
        """Clean up resources.
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
    """
        self.running = False
        if self.maintenance_task is not None:
            self.maintenance_task.join(timeout=1.0)
        self.clear()
optimized_model_loader = OptimizedModelLoader()
def lazy_load_model(model_id: str, metadata: Optional[Dict[str, Any]] = None):
    """
    Decorator for lazy loading models.
    This decorator creates a property that lazily loads a model when accessed.
    Args:
        model_id: Model ID
        metadata: Model metadata
    Returns:
        Decorated property