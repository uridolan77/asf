"""
Optimized Model Loader for the Medical Research Synthesizer.

This module provides an optimized model loading system that efficiently
manages ML models in memory and supports sharing models across processes.
"""

import os
import time
import logging
import threading
import asyncio
import json
import pickle
import tempfile
import uuid
import gc
from typing import Dict, List, Any, Optional, Callable, Union, Type, TypeVar
from pathlib import Path
from datetime import datetime, timedelta

import torch
import numpy as np

from asf.medical.core.enhanced_cache import enhanced_cache_manager
from asf.medical.core.resource_limiter import resource_limiter

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for models
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
        """
        Initialize model information.
        
        Args:
            model_id: Model ID
            model: Model instance
            metadata: Model metadata
            loaded_at: Time when the model was loaded
            last_used: Time when the model was last used
            shared_memory_id: Shared memory ID (if model is in shared memory)
        """
        self.model_id = model_id
        self.model = model
        self.metadata = metadata
        self.loaded_at = loaded_at
        self.last_used = last_used
        self.shared_memory_id = shared_memory_id
        self.use_count = 0
        self.lock = threading.RLock()
    
    def update_last_used(self):
        """Update the last used time."""
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
            "age": self.get_age(),
            "idle_time": self.get_idle_time(),
            "use_count": self.use_count,
            "memory_mb": self.get_memory_usage(),
            "shared_memory": self.shared_memory_id is not None
        }

class OptimizedModelLoader:
    """
    Optimized model loader for the Medical Research Synthesizer.
    
    This class provides an optimized model loading system that efficiently
    manages ML models in memory and supports sharing models across processes.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        """
        Create a singleton instance of the optimized model loader.
        
        Returns:
            OptimizedModelLoader: The singleton instance
        """
        if cls._instance is None:
            cls._instance = super(OptimizedModelLoader, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        max_models: int = 5,
        ttl: int = 3600,  # 1 hour
        check_interval: int = 300,  # 5 minutes
        shared_memory_dir: Optional[str] = None,
        use_shared_memory: bool = False
    ):
        """
        Initialize the optimized model loader.
        
        Args:
            max_models: Maximum number of models to keep in memory (default: 5)
            ttl: Time to live in seconds (default: 3600 = 1 hour)
            check_interval: Interval in seconds for checking expired models (default: 300 = 5 minutes)
            shared_memory_dir: Directory for shared memory files (default: system temp dir)
            use_shared_memory: Whether to use shared memory (default: False)
        """
        if self._initialized:
            return
        
        # Get configuration from environment variables
        self.max_models = int(os.environ.get("MAX_CACHED_MODELS", max_models))
        self.ttl = int(os.environ.get("MODEL_CACHE_TTL", ttl))
        self.check_interval = int(os.environ.get("MODEL_CACHE_CHECK_INTERVAL", check_interval))
        
        # Shared memory configuration
        self.use_shared_memory = use_shared_memory
        self.shared_memory_dir = shared_memory_dir or os.environ.get("SHARED_MEMORY_DIR", tempfile.gettempdir())
        
        # Create shared memory directory if it doesn't exist
        if self.use_shared_memory:
            os.makedirs(self.shared_memory_dir, exist_ok=True)
        
        # Model cache
        self.models: Dict[str, ModelInfo] = {}
        
        # Factory registry
        self.factories: Dict[str, Dict[str, Any]] = {}
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        # Background task for maintenance
        self.maintenance_task = None
        self.running = False
        
        # Start maintenance task
        self._start_maintenance_task()
        
        self._initialized = True
        logger.info(f"Optimized model loader initialized with max_models={self.max_models}, ttl={self.ttl}s")
    
    def _start_maintenance_task(self):
        """Start the maintenance task."""
        if self.maintenance_task is not None:
            return
        
        self.running = True
        
        def maintenance_loop():
            while self.running:
                try:
                    self._check_expired_models()
                except Exception as e:
                    logger.error(f"Error in maintenance task: {str(e)}")
                
                # Sleep for check_interval seconds
                time.sleep(self.check_interval)
        
        # Start maintenance thread
        self.maintenance_task = threading.Thread(target=maintenance_loop, daemon=True)
        self.maintenance_task.start()
        
        logger.debug("Maintenance task started")
    
    def _check_expired_models(self):
        """Check for expired models and unload them."""
        with self.lock:
            # Get current time
            now = time.time()
            
            # Find expired models
            expired_models = []
            for model_id, model_info in self.models.items():
                # Check if model has expired
                if now - model_info.last_used > self.ttl:
                    expired_models.append(model_id)
            
            # Unload expired models
            for model_id in expired_models:
                self._unload_model(model_id)
                logger.info(f"Unloaded expired model: {model_id}")
    
    def _unload_model(self, model_id: str):
        """
        Unload a model from memory.
        
        Args:
            model_id: Model ID
        """
        with self.lock:
            if model_id not in self.models:
                return
            
            model_info = self.models[model_id]
            
            # Remove from models
            del self.models[model_id]
            
            # Delete shared memory file if applicable
            if model_info.shared_memory_id:
                shared_memory_path = os.path.join(self.shared_memory_dir, model_info.shared_memory_id)
                try:
                    os.remove(shared_memory_path)
                    logger.debug(f"Deleted shared memory file: {shared_memory_path}")
                except Exception as e:
                    logger.error(f"Error deleting shared memory file: {str(e)}")
            
            # Clear model reference
            model_info.model = None
            
            # Force garbage collection
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.debug(f"Unloaded model: {model_id}")
    
    def register_factory(
        self,
        model_id: str,
        factory: Callable[[], T],
        model_type: Optional[Type] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
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
                "type": model_type,
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
                logger.debug(f"Cache hit for model: {model_id}")
                return model_info.model
            
            logger.debug(f"Cache miss for model: {model_id}")
            return None
    
    def put(
        self,
        model_id: str,
        model: Any,
        metadata: Optional[Dict[str, Any]] = None,
        use_shared_memory: Optional[bool] = None
    ):
        """
        Put a model in the cache.
        
        Args:
            model_id: Model ID
            model: Model instance
            metadata: Model metadata
            use_shared_memory: Whether to use shared memory (default: self.use_shared_memory)
        """
        with self.lock:
            # Check if we need to evict a model
            if len(self.models) >= self.max_models and model_id not in self.models:
                self._evict_model()
            
            # Use shared memory if enabled
            shared_memory_id = None
            if (use_shared_memory if use_shared_memory is not None else self.use_shared_memory):
                try:
                    shared_memory_id = self._save_to_shared_memory(model_id, model)
                except Exception as e:
                    logger.error(f"Error saving model to shared memory: {str(e)}")
            
            # Create model info
            now = time.time()
            model_info = ModelInfo(
                model_id=model_id,
                model=model,
                metadata=metadata or {},
                loaded_at=now,
                last_used=now,
                shared_memory_id=shared_memory_id
            )
            
            # Add to models
            self.models[model_id] = model_info
            
            # Register memory usage
            memory_mb = model_info.get_memory_usage()
            if memory_mb > 0:
                resource_limiter.register_model_usage(model_id, memory_mb)
            
            logger.debug(f"Added model to cache: {model_id}")
    
    def get_or_create(
        self,
        model_id: str,
        factory: Optional[Callable[[], T]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        use_shared_memory: Optional[bool] = None
    ) -> Any:
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
        
        with self.lock:
            # Check again in case another thread loaded the model
            model = self.get(model_id)
            if model is not None:
                return model
            
            # Try to load from shared memory
            if (use_shared_memory if use_shared_memory is not None else self.use_shared_memory):
                try:
                    model = self._load_from_shared_memory(model_id)
                    if model is not None:
                        self.put(model_id, model, metadata, use_shared_memory=False)
                        return model
                except Exception as e:
                    logger.error(f"Error loading model from shared memory: {str(e)}")
            
            # Use provided factory or get from registry
            if factory is None:
                if model_id in self.factories:
                    factory_info = self.factories[model_id]
                    factory = factory_info["factory"]
                    
                    # Use metadata from registry if not provided
                    if metadata is None:
                        metadata = factory_info["metadata"]
                else:
                    raise ValueError(f"Model not found and no factory provided: {model_id}")
            
            # Create model
            logger.info(f"Creating model: {model_id}")
            model = factory()
            
            # Type check
            if model_id in self.factories and self.factories[model_id]["type"] is not None:
                expected_type = self.factories[model_id]["type"]
                if not isinstance(model, expected_type):
                    raise TypeError(f"Model {model_id} has wrong type: {type(model)}, expected {expected_type}")
            
            # Add to cache
            self.put(model_id, model, metadata, use_shared_memory)
            
            return model
    
    def remove(self, model_id: str):
        """
        Remove a model from the cache.
        
        Args:
            model_id: Model ID
        """
        with self.lock:
            self._unload_model(model_id)
    
    def clear(self):
        """Clear all models from the cache."""
        with self.lock:
            # Get all model IDs
            model_ids = list(self.models.keys())
            
            # Unload all models
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
            
            # Find the least recently used model
            lru_model_id = min(self.models.items(), key=lambda x: x[1].last_used)[0]
            
            # Unload the model
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
        # Generate shared memory ID
        shared_memory_id = f"{model_id}_{uuid.uuid4().hex}.pkl"
        
        # Create shared memory path
        shared_memory_path = os.path.join(self.shared_memory_dir, shared_memory_id)
        
        # Save model to shared memory
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
        # Find shared memory file
        shared_memory_pattern = f"{model_id}_*.pkl"
        shared_memory_dir = Path(self.shared_memory_dir)
        
        # Find matching files
        matching_files = list(shared_memory_dir.glob(shared_memory_pattern))
        
        if not matching_files:
            return None
        
        # Use the most recent file
        shared_memory_path = max(matching_files, key=lambda p: p.stat().st_mtime)
        
        # Load model from shared memory
        with open(shared_memory_path, "rb") as f:
            model = pickle.load(f)
        
        logger.debug(f"Loaded model from shared memory: {shared_memory_path}")
        
        return model
    
    def __del__(self):
        """Clean up resources."""
        self.running = False
        
        if self.maintenance_task is not None:
            self.maintenance_task.join(timeout=1.0)
        
        self.clear()

# Create a singleton instance
optimized_model_loader = OptimizedModelLoader()

# Decorator for lazy loading models
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
        @property
        def wrapper(self):
            # Use model cache to get or create the model
            return optimized_model_loader.get_or_create(
                model_id=model_id,
                factory=lambda: factory_method(self),
                metadata=metadata
            )
        return wrapper
    return decorator
