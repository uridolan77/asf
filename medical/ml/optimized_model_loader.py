"""Optimized Model Loader for the Medical Research Synthesizer.

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
# Import necessary modules
logger = logging.getLogger(__name__)
T = TypeVar('T')
class ModelInfo:
    """Information about a loaded model.
    
    This class stores metadata and state information about a model
    that has been loaded into the model cache.
    """
    def __init__(
        self,
        model_id: str,
            """
            __init__ function.
            
            This function provides functionality for...
            Args:
                model_id: Description of model_id
                model: Description of model
                metadata: Description of metadata
                loaded_at: Description of loaded_at
                last_used: Description of last_used
                shared_memory_id: Description of shared_memory_id
            """
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
        """Update the last used time and increment the use count.
        
        This method is called whenever the model is accessed to keep track
        of when it was last used for cache management purposes.
        """
        self.last_used = time.time()
        self.use_count += 1
    def get_memory_usage(self) -> int:
        """
        Get the memory usage of the model in MB.

        Returns:
            int: Memory usage in MB
        """
        return self.metadata.get("memory_mb", 0)

    def get_age(self) -> float:
        """
        Get the age of the model in seconds.

        Returns:
            float: Age in seconds
        """
        return time.time() - self.loaded_at

    def get_idle_time(self) -> float:
        """
        Get the idle time of the model in seconds.

        Returns:
            float: Idle time in seconds
        """
        return time.time() - self.last_used

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "model_id": self.model_id,
            "metadata": self.metadata,
            "loaded_at": self.loaded_at,
            "last_used": self.last_used,
            "use_count": self.use_count,
            "age": self.get_age(),
            "idle_time": self.get_idle_time(),
            "memory_usage": self.get_memory_usage(),
            "shared_memory_id": self.shared_memory_id
        }


class OptimizedModelLoader:
    """Optimized model loader for the Medical Research Synthesizer.
    
    This class provides an optimized model loading system that efficiently
    manages ML models in memory and supports sharing models across processes.
    """

    _instance = None

    def __new__(cls):
        """
        Create a singleton instance of the optimized model loader.

        Returns:
            OptimizedModelLoader: The singleton instance
        """
        if cls._instance is None:
            cls._instance = super(OptimizedModelLoader, cls).__new__(cls)
        return cls._instance

    def __init__(self, max_models: int = 5, ttl: int = 3600, check_interval: int = 300,
                 shared_memory_dir: Optional[str] = None, use_shared_memory: bool = False):
        """
        Initialize the optimized model loader.

        Args:
            max_models: Maximum number of models to keep in memory (default: 5)
            ttl: Time to live in seconds (default: 3600 = 1 hour)
            check_interval: Interval in seconds for checking expired models (default: 300 = 5 minutes)
            shared_memory_dir: Directory for shared memory files (default: system temp dir)
            use_shared_memory: Whether to use shared memory (default: False)
        """
        # Initialize only once (singleton pattern)
        if hasattr(self, "initialized") and self.initialized:
            return

        self.max_models = max_models
        self.ttl = ttl
        self.check_interval = check_interval
        self.shared_memory_dir = shared_memory_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "shared_memory")
        self.use_shared_memory = use_shared_memory

        # Create shared memory directory if it doesn't exist
        if self.use_shared_memory and not os.path.exists(self.shared_memory_dir):
            os.makedirs(self.shared_memory_dir, exist_ok=True)

        self.models = {}
        self.factories = {}
        self.lock = threading.RLock()
        self.maintenance_task = None
        self.running = False

        # Start maintenance task
        self._start_maintenance_task()

        self.initialized = True
        logger.info(f"Optimized model loader initialized with max_models={max_models}, ttl={ttl}s")
        if self.maintenance_task is not None:
            return
        self.running = True
        def maintenance_loop():
            """Maintenance loop for background model cleanup.
            
            This function runs periodically in the background to check for
            expired models and clean them up to free memory.
            """
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
        """
        Check for expired models and unload them.

        This method is called periodically by the maintenance task to unload
        models that have not been used for a while.
        """
        now = time.time()
        with self.lock:
            for model_id, model_info in list(self.models.items()):
                # Check if model has expired
                if now - model_info.last_used > self.ttl:
                    self._unload_model(model_id)
                    logger.info(f"Unloaded expired model: {model_id}")

    def _unload_model(self, model_id: str):
        """
        Unload a model from memory.

        Args:
            model_id: Model ID to unload
        """
        with self.lock:
            if model_id in self.models:
                model_info = self.models[model_id]
                # Clean up shared memory if used
                if model_info.shared_memory_id and self.use_shared_memory:
                    shared_memory_path = os.path.join(self.shared_memory_dir, model_info.shared_memory_id)
                    if os.path.exists(shared_memory_path):
                        try:
                            os.remove(shared_memory_path)
                            logger.debug(f"Removed shared memory file: {shared_memory_path}")
                        except Exception as e:
                            logger.warning(f"Failed to remove shared memory file: {e}")

                # Remove from cache
                del self.models[model_id]
                logger.debug(f"Unloaded model: {model_id}")

    def register_factory(self, model_id: str, factory, model_type: Optional[type] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Register a factory function for creating a model.

        Args:
            model_id: Model ID
            factory: Factory function that creates the model
            model_type: Expected type of the model (for type checking)
            metadata: Model metadata
        """
        with self.lock:
            self.factories[model_id] = {
                "factory": factory,
                "model_type": model_type,
                "metadata": metadata or {}
            }
            logger.debug(f"Registered factory for model: {model_id}")

    def get(self, model_id: str) -> Optional[Any]:
        """
        Get a model from the cache.

        Args:
            model_id: Model ID

        Returns:
            Optional[Any]: Model instance or None if not found
        """
        with self.lock:
            if model_id in self.models:
                model_info = self.models[model_id]
                model_info.update_last_used()
                return model_info.model
            return None

    def put(self, model_id: str, model: Any, metadata: Optional[Dict[str, Any]] = None, use_shared_memory: Optional[bool] = None):
        """
        Put a model in the cache.

        Args:
            model_id: Model ID
            model: Model instance
            metadata: Model metadata
            use_shared_memory: Whether to use shared memory (default: self.use_shared_memory)
        """
        use_shared_memory = use_shared_memory if use_shared_memory is not None else self.use_shared_memory
        shared_memory_id = None

        # Save to shared memory if enabled
        if use_shared_memory:
            try:
                shared_memory_id = self._save_to_shared_memory(model_id, model)
            except Exception as e:
                logger.warning(f"Failed to save model to shared memory: {e}")

        # Check if we need to evict a model
        with self.lock:
            if len(self.models) >= self.max_models:
                self._evict_model()

            # Add to cache
            now = time.time()
            self.models[model_id] = ModelInfo(
                model_id=model_id,
                model=model,
                metadata=metadata or {},
                loaded_at=now,
                last_used=now,
                shared_memory_id=shared_memory_id
            )
            logger.debug(f"Added model to cache: {model_id}")

    def get_or_create(self, model_id: str, factory=None, metadata: Optional[Dict[str, Any]] = None, use_shared_memory: Optional[bool] = None) -> Any:
        """
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
        """
        # Try to get from cache
        model = self.get(model_id)
        if model is not None:
            return model

        # Try to load from shared memory
        if self.use_shared_memory:
            try:
                model = self._load_from_shared_memory(model_id)
                if model is not None:
                    self.put(model_id, model, metadata, use_shared_memory=False)
                    return model
            except Exception as e:
                logger.warning(f"Failed to load model from shared memory: {e}")

        # Use provided factory or registered factory
        if factory is None:
            with self.lock:
                if model_id in self.factories:
                    factory_info = self.factories[model_id]
                    factory = factory_info["factory"]
                    if metadata is None:
                        metadata = factory_info["metadata"]
                else:
                    raise ValueError(f"Model not found and no factory provided: {model_id}")

        # Create model
        model = factory()
        self.put(model_id, model, metadata, use_shared_memory)
        return model

    def remove(self, model_id: str):
        """
        Remove a model from the cache.

        Args:
            model_id: Model ID to remove
        """
        self._unload_model(model_id)
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
        """
        Clean up resources when the object is garbage collected.

        This method stops the maintenance task and clears the cache.
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
    """
    def decorator(factory_method):
        """Factory method decorator for lazy model loading.
        
        This function wraps the factory method in a property that will
        load the model on first access and cache it for future use.
        
        Args:
            factory_method: Function that creates the model instance
            
        Returns:
            property: A property descriptor that loads the model on demand
        """
        def getter(self):
            """Get or create a model instance on demand.
            
            This function creates a property accessor that will load the model
            when first accessed and retrieve it from cache on subsequent accesses.
            
            Returns:
                Any: The cached or newly created model instance
            """
            # Use the model ID as is or with class name prefix
            instance_model_id = f"{self.__class__.__name__}_{model_id}"

            # Get or create the model
            return optimized_model_loader.get_or_create(
                model_id=instance_model_id,
                factory=lambda: factory_method(self),
                metadata=metadata
            )
        return property(getter)
    return decorator