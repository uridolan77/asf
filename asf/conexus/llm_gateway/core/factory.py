"""
Factory for LLM Gateway provider creation and management.

This module provides factory classes for creating and managing LLM provider
instances, including support for resource management and progress tracking.
"""

import logging
import asyncio
from typing import Dict, Type, Optional, List, Any, cast

from asf.conexus.llm_gateway.core.models import ProviderConfig, GatewayConfig
from asf.conexus.llm_gateway.providers.base import BaseProvider

# Import progress tracking functionality if available
try:
    from asf.conexus.llm_gateway.progress.factory import ProgressTrackingFactory
    PROGRESS_TRACKING_AVAILABLE = True
except ImportError:
    PROGRESS_TRACKING_AVAILABLE = False

logger = logging.getLogger(__name__)


class ProviderFactoryError(Exception):
    """Custom exception for errors during provider factory operations."""
    pass


class ProviderFactory:
    """
    Factory class responsible for creating instances of LLM providers.

    This factory uses a registry to map provider type strings (from configuration)
    to the actual provider implementation classes.
    """

    def __init__(self):
        """Initialize the provider factory."""
        self._provider_registry = {}
        self._provider_instances = {}
        self._instance_locks = {}
        self._allow_overwrite = False
        
        # Initialize progress tracking components if available
        self._use_progress_tracking = False
        if PROGRESS_TRACKING_AVAILABLE:
            self._progress_factory = None

    def enable_progress_tracking(self, enabled: bool = True) -> None:
        """
        Enable or disable progress tracking.
        
        Args:
            enabled: Whether to enable progress tracking
        """
        if enabled and not PROGRESS_TRACKING_AVAILABLE:
            logger.warning("Progress tracking requested but not available (missing dependencies)")
            return
            
        self._use_progress_tracking = enabled and PROGRESS_TRACKING_AVAILABLE
        logger.info(f"Progress tracking {'enabled' if self._use_progress_tracking else 'disabled'}")
        
        # Initialize progress factory if needed
        if self._use_progress_tracking and not self._progress_factory:
            self._progress_factory = ProgressTrackingFactory()

    def register_provider(self, provider_type: str, provider_class: Type[BaseProvider]):
        """
        Register a provider type string and its corresponding class.
        
        Args:
            provider_type: String identifier for the provider type
            provider_class: Class implementing the BaseProvider interface
        """
        type_lower = provider_type.lower()
        if type_lower in self._provider_registry and not self._allow_overwrite:
            raise ValueError(f"Provider type '{type_lower}' already registered and overwrite is disallowed.")
        elif type_lower in self._provider_registry and self._allow_overwrite:
            logger.warning(f"Provider type '{type_lower}' is already registered. Overwriting.")

        if not issubclass(provider_class, BaseProvider):
             raise TypeError(f"Provider class {provider_class.__name__} must inherit from BaseProvider.")

        self._provider_registry[type_lower] = provider_class
        logger.debug(f"Registered provider type '{type_lower}' with class {provider_class.__name__}")

    def get_registered_types(self) -> List[str]:
         """Returns a list of registered provider type strings."""
         return list(self._provider_registry.keys())

    async def get_provider(self, provider_id: str, provider_config: ProviderConfig, gateway_config: GatewayConfig) -> BaseProvider:
        """
        Gets or creates a provider instance with optional progress tracking.
        
        Args:
            provider_id: Unique ID for the specific provider instance
            provider_config: Configuration for this provider instance
            gateway_config: Global gateway configuration
            
        Returns:
            An initialized provider instance, potentially wrapped with progress tracking
            
        Raises:
            ProviderFactoryError: If creation or initialization fails
        """
        # Ensure lock exists for the provider_id
        if provider_id not in self._instance_locks:
            # This needs its own lock to prevent race conditions when creating instance locks
            async with asyncio.Lock():  # Temporary lock for creating the specific lock
                if provider_id not in self._instance_locks:
                    self._instance_locks[provider_id] = asyncio.Lock()

        async with self._instance_locks[provider_id]:
            if provider_id not in self._provider_instances:
                logger.info(f"Creating new provider instance for ID: {provider_id}")
                try:
                    # Get the provider type
                    provider_type = provider_config.provider_type.lower()
                    
                    # Get the provider class
                    provider_class = self._provider_registry.get(provider_type)
                    if not provider_class:
                        raise ProviderFactoryError(
                            f"Unknown provider type '{provider_type}' for ID '{provider_id}'. "
                            f"Registered: {self.get_registered_types()}"
                        )
                    
                    # Create the provider instance, potentially with progress tracking
                    if self._use_progress_tracking and PROGRESS_TRACKING_AVAILABLE:
                        logger.info(f"Creating progress-tracking provider for ID '{provider_id}'")
                        instance = self._progress_factory.create_provider(
                            provider_type=provider_type,
                            provider_config=provider_config,
                            provider_class=provider_class, 
                            gateway_config=gateway_config
                        )
                    else:
                        # Create standard provider instance
                        instance = self._create_provider_sync(
                            provider_id=provider_id,
                            provider_config=provider_config,
                            gateway_config=gateway_config,
                            provider_class=provider_class
                        )
                    
                    # Perform async initialization if the method exists
                    if hasattr(instance, "initialize_async") and callable(instance.initialize_async):
                        logger.info(f"Running async initialization for provider {provider_id}...")
                        await instance.initialize_async()
                        logger.info(f"Async initialization complete for {provider_id}.")

                    self._provider_instances[provider_id] = instance
                    
                except Exception as e:
                    # Catch errors during creation or async init
                    logger.error(f"Failed to create or initialize provider '{provider_id}': {e}", exc_info=True)
                    # Remove potentially partial instance if error occurred
                    if provider_id in self._provider_instances:
                        del self._provider_instances[provider_id]
                    raise ProviderFactoryError(f"Failed to get provider '{provider_id}': {e}") from e

            return self._provider_instances[provider_id]

    def _create_provider_sync(self, provider_id: str, provider_config: ProviderConfig, gateway_config: GatewayConfig, provider_class: Type[BaseProvider] = None) -> BaseProvider:
        """
        Synchronously instantiate a provider class.
        
        Args:
            provider_id: Unique ID for the provider instance
            provider_config: Configuration for this provider
            gateway_config: Global gateway configuration
            provider_class: Optional provider class, if not provided it will be looked up
            
        Returns:
            An initialized provider instance
            
        Raises:
            ProviderFactoryError: If creation fails
        """
        if not provider_config or not hasattr(provider_config, 'provider_type'):
            raise TypeError("Invalid provider_config. Must be ProviderConfig with 'provider_type'.")

        provider_type = provider_config.provider_type
        if not provider_type or not isinstance(provider_type, str):
            raise TypeError(f"provider_config for '{provider_id}' must have a non-empty string 'provider_type'.")

        provider_type_lower = provider_type.lower()
        
        if not provider_class:
            provider_class = self._provider_registry.get(provider_type_lower)

        if not provider_class:
            raise ProviderFactoryError(
                f"Unknown provider type '{provider_type}' for ID '{provider_id}'. "
                f"Registered: {self.get_registered_types()}"
            )

        logger.info(f"Instantiating provider ID '{provider_id}' (Type: '{provider_type}')")
        try:
            # Create the provider instance
            provider_instance = provider_class(
                provider_config=provider_config,
                gateway_config=gateway_config,
                provider_id=provider_id
            )
            logger.info(f"Successfully instantiated provider ID '{provider_id}'.")
            return provider_instance
        except Exception as e:
            # Log details but re-raise as factory error
            logger.error(f"Instantiation error for provider '{provider_id}': {e}", exc_info=True)
            raise ProviderFactoryError(f"Instantiation failed for '{provider_id}': {e}") from e

    def get_cached_instance(self, provider_id: str) -> Optional[BaseProvider]:
        """Gets a cached provider instance if it exists, without creating."""
        return self._provider_instances.get(provider_id)

    def get_all_cached_instances(self) -> List[BaseProvider]:
        """Gets all currently cached provider instances."""
        return list(self._provider_instances.values())

    async def cleanup_all(self):
        """Cleans up all cached provider instances."""
        logger.info(f"Cleaning up {len(self._provider_instances)} provider instance(s)...")
        tasks = []
        provider_ids = list(self._provider_instances.keys())  # Avoid dict size change during iteration
        for provider_id in provider_ids:
            instance = self._provider_instances.pop(provider_id, None)
            if instance and hasattr(instance, "cleanup") and callable(instance.cleanup):
                logger.debug(f"Scheduling cleanup for provider {provider_id}")
                tasks.append(instance.cleanup())
            # Remove lock too
            if provider_id in self._instance_locks:
                del self._instance_locks[provider_id]

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error during cleanup for a provider: {result}")
        logger.info("Provider instance cleanup complete.")


if PROGRESS_TRACKING_AVAILABLE:
    class EnhancedGatewayFactory:
        """
        Factory for creating enhanced LLM Gateway components.
        
        This class provides methods for creating LLM Gateway components
        with integrated progress tracking.
        """
        
        def __init__(self, cache_manager=None):
            """
            Initialize the enhanced gateway factory.
            
            Args:
                cache_manager: Cache manager for storing progress information
            """
            # Create enhanced provider factory
            self.provider_factory = ProviderFactory()
            self.provider_factory.enable_progress_tracking(True)
            
            # Create progress tracking factory
            self.progress_factory = ProgressTrackingFactory(cache_manager)
            
            logger.info("Enhanced gateway factory initialized")
        
        def create_client(self, config: GatewayConfig = None, db=None) -> Any:
            """
            Create an enhanced LLM Gateway client with progress tracking.
            
            Args:
                config: Gateway configuration
                db: Database session
                
            Returns:
                Progress-tracking LLM client
            """
            logger.info("Creating enhanced LLM Gateway client with progress tracking")
            
            # Import here to avoid circular imports
            from asf.conexus.llm_gateway.core.client import ProgressTrackingLLMClient
            
            # Create progress-tracking client
            client = ProgressTrackingLLMClient(
                config=config,
                provider_factory=self.provider_factory,
                db=db
            )
            
            return client
        
        def enable_progress_tracking(self, enabled: bool = True) -> None:
            """
            Enable or disable progress tracking.
            
            Args:
                enabled: Whether to enable progress tracking
            """
            self.provider_factory.enable_progress_tracking(enabled)
            logger.info(f"Progress tracking {'enabled' if enabled else 'disabled'}")


# Global instance of the enhanced gateway factory
_enhanced_gateway_factory = None


def get_enhanced_gateway_factory(cache_manager=None) -> Any:
    """
    Get the singleton instance of the enhanced gateway factory.
    
    Args:
        cache_manager: Cache manager for storing progress information
        
    Returns:
        Enhanced gateway factory
    """
    global _enhanced_gateway_factory
    if _enhanced_gateway_factory is None and PROGRESS_TRACKING_AVAILABLE:
        _enhanced_gateway_factory = EnhancedGatewayFactory(cache_manager)
    return _enhanced_gateway_factory