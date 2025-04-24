"""Model registry for the Medical Research Synthesizer.

This module provides a registry for ML models with lazy loading.
"""
import logging
import threading
from typing import Any, Optional, Type, Callable
logger = logging.getLogger(__name__)
class ModelRegistry:
    """Registry for ML models with lazy loading.
    
    This class provides a thread-safe registry for ML models,
    allowing models to be loaded only when needed and unloaded
    when not in use to save memory.
    """
    _instance = None
    _lock = threading.RLock()
    def __new__(cls):
        """
        Create a singleton instance of the model registry.
        Returns:
            ModelRegistry: The singleton instance
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelRegistry, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    def __init__(self):
        """Initialize the model registry.
        
        This method initializes the model registry as a singleton instance.
        """
        with self._lock:
            if self._initialized:
                return
            self._models = {}
            self._factories = {}
            self._initialized = True
            logger.info("Model registry initialized")
    def init(self):
        """Initialize the model registry with default models.
        
        This method can be used to pre-register commonly used models.
        """
        logger.info("Model registry initialized with default models")
    def register_model_factory(
        self,
        model_name: str,
        factory: Callable[[], Any],
        model_type: Optional[Type] = None
    ):
        """
        Register a factory function for creating a model.

        Args:
            model_name: Name of the model
            factory: Factory function that creates the model
            model_type: Expected type of the model (for type checking)
        """
        with self._lock:
            self._factories[model_name] = {
                'factory': factory,
                'type': model_type
            }
            logger.info(f"Model factory registered: {model_name}")
    def is_model_registered(self, model_name: str) -> bool:
        """
        Check if a model factory is registered.
        Args:
            model_name: Name of the model
        Returns:
            True if the model factory is registered, False otherwise
        """
        with self._lock:
            return model_name in self._factories
    def get_model(self, model_name: str) -> Any:
        """
        Get a model from the registry.
        If the model is not loaded, it will be loaded using the registered factory.
        Args:
            model_name: Name of the model
        Returns:
            The model
        Raises:
            ValueError: If the model is not registered
        """
        with self._lock:
            if model_name in self._models:
                logger.debug(f"Model already loaded: {model_name}")
                return self._models[model_name]
            if model_name not in self._factories:
                raise ValueError(f"Model not registered: {model_name}")
            logger.info(f"Loading model: {model_name}")
            factory_info = self._factories[model_name]
            model = factory_info['factory']()
            if factory_info['type'] is not None and not isinstance(model, factory_info['type']):
                raise TypeError(f"Model {model_name} has wrong type: {type(model)}, expected {factory_info['type']}")
            self._models[model_name] = model
            return model
    def unload_model(self, model_name: str) -> bool:
        """
        Unload a model from the registry.
        Args:
            model_name: Name of the model
        Returns:
            True if the model was unloaded, False otherwise
        """
        with self._lock:
            if model_name not in self._models:
                logger.debug(f"Model not loaded: {model_name}")
                return False
            model = self._models[model_name]
            if hasattr(model, 'unload_model'):
                logger.info(f"Unloading model: {model_name}")
                model.unload_model()
            del self._models[model_name]
            return True
    def unload_all_models(self):
        """Unload all models from the registry.
        
        This method unloads all loaded models to free memory.
        """
        with self._lock:
            for model_name in list(self._models.keys()):
                self.unload_model(model_name)
            logger.info("All models unloaded")
    def is_model_loaded(self, model_name: str) -> bool:
        """
        Check if a model is loaded.

        Args:
            model_name: Name of the model

        Returns:
            True if the model is loaded, False otherwise
        """
        with self._lock:
            return model_name in self._models

    def get_loaded_models(self) -> dict:
        """
        Get all loaded models.

        Returns:
            Dictionary of loaded models
        """
        with self._lock:
            return dict(self._models)

    def get_registered_models(self) -> dict:
        """
        Get all registered models.

        Returns:
            Dictionary of registered models
        """