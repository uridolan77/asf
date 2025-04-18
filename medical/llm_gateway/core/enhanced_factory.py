"""
Enhanced factory for LLM Gateway components with progress tracking.

This module provides enhanced factory classes for creating LLM Gateway
components with integrated progress tracking.
"""

import logging
from typing import Optional, Dict, Any, Type

from asf.medical.llm_gateway.core.factory import ProviderFactory, ProviderFactoryError
from asf.medical.llm_gateway.core.models import GatewayConfig, ProviderConfig
from asf.medical.llm_gateway.providers.base import BaseProvider

# Import progress tracking components
from asf.medical.llm_gateway.core.progress_client import ProgressTrackingLLMClient
from asf.medical.llm_gateway.progress.factory import ProgressTrackingFactory

logger = logging.getLogger(__name__)


class EnhancedProviderFactory(ProviderFactory):
    """
    Enhanced provider factory with integrated progress tracking.
    
    This class extends the base provider factory to create providers
    with integrated progress tracking by default.
    """
    
    def __init__(self, cache_manager=None):
        """
        Initialize the enhanced provider factory.
        
        Args:
            cache_manager: Cache manager for storing progress information
        """
        super().__init__()
        
        # Create progress tracking factory
        self.progress_factory = ProgressTrackingFactory(cache_manager)
        
        # Flag to control whether to use progress tracking
        self.use_progress_tracking = True
        
        logger.info("Enhanced provider factory initialized with progress tracking")
    
    def enable_progress_tracking(self, enabled: bool = True) -> None:
        """
        Enable or disable progress tracking.
        
        Args:
            enabled: Whether to enable progress tracking
        """
        self.use_progress_tracking = enabled
        logger.info(f"Progress tracking {'enabled' if enabled else 'disabled'}")
    
    async def get_provider(self, provider_id: str, provider_config: ProviderConfig, gateway_config: GatewayConfig) -> BaseProvider:
        """
        Gets or creates a provider instance with progress tracking.
        
        Args:
            provider_id: Unique ID for the specific provider instance
            provider_config: Configuration for this provider instance
            gateway_config: Global gateway configuration
            
        Returns:
            An initialized provider instance with progress tracking
            
        Raises:
            ProviderFactoryError: If creation or initialization fails
        """
        # Check if we should use progress tracking
        if not self.use_progress_tracking:
            # Use the base implementation if progress tracking is disabled
            return await super().get_provider(provider_id, provider_config, gateway_config)
        
        # Get the provider type
        provider_type = provider_config.provider_type.lower()
        
        # Get the provider class from the registry
        provider_class = self._provider_registry.get(provider_type)
        
        if not provider_class:
            raise ProviderFactoryError(
                f"Unknown provider type '{provider_type}' for ID '{provider_id}'. "
                f"Registered: {self.get_registered_types()}"
            )
        
        # Check if we already have an instance
        if provider_id in self._provider_instances:
            return self._provider_instances[provider_id]
        
        try:
            # Create a progress-tracking provider
            provider = self.progress_factory.create_provider(
                provider_type=provider_type,
                provider_config=provider_config,
                provider_class=provider_class
            )
            
            # Store the instance
            self._provider_instances[provider_id] = provider
            
            logger.info(f"Created progress-tracking provider for ID '{provider_id}' (Type: '{provider_type}')")
            
            return provider
        except Exception as e:
            logger.error(f"Failed to create progress-tracking provider for ID '{provider_id}': {e}", exc_info=True)
            raise ProviderFactoryError(f"Failed to create provider '{provider_id}': {e}") from e


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
        self.provider_factory = EnhancedProviderFactory(cache_manager)
        
        # Create progress tracking factory
        self.progress_factory = ProgressTrackingFactory(cache_manager)
        
        logger.info("Enhanced gateway factory initialized")
    
    def create_client(self, config: GatewayConfig = None, db=None) -> ProgressTrackingLLMClient:
        """
        Create an enhanced LLM Gateway client with progress tracking.
        
        Args:
            config: Gateway configuration
            db: Database session
            
        Returns:
            Progress-tracking LLM client
        """
        logger.info("Creating enhanced LLM Gateway client with progress tracking")
        
        # Create progress-tracking client
        client = self.progress_factory.create_client(
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


# Singleton instance
_enhanced_gateway_factory = None


def get_enhanced_gateway_factory(cache_manager=None) -> EnhancedGatewayFactory:
    """
    Get the singleton instance of the enhanced gateway factory.
    
    Args:
        cache_manager: Cache manager for storing progress information
        
    Returns:
        Enhanced gateway factory
    """
    global _enhanced_gateway_factory
    if _enhanced_gateway_factory is None:
        _enhanced_gateway_factory = EnhancedGatewayFactory(cache_manager)
    return _enhanced_gateway_factory
