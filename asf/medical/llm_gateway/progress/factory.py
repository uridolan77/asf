"""
Factory for creating progress-tracking components.

This module provides a factory for creating progress-tracking components,
such as progress-tracking LLM clients and providers.
"""

import logging
from typing import Optional, Dict, Any, Type

from asf.medical.llm_gateway.core.client import LLMGatewayClient
from asf.medical.llm_gateway.core.models import GatewayConfig, ProviderConfig
from asf.medical.llm_gateway.core.factory import ProviderFactory
from asf.medical.llm_gateway.providers.base import BaseProvider

# Import progress-tracking components
from asf.medical.llm_gateway.core.progress_client import ProgressTrackingLLMClient
from asf.medical.llm_gateway.providers.progress_provider import ProgressTrackingProvider

# Import registry
from .registry import get_progress_registry

logger = logging.getLogger(__name__)


class ProgressTrackingFactory:
    """
    Factory for creating progress-tracking components.
    
    This class provides methods for creating progress-tracking LLM clients
    and providers, as well as wrapping existing components with progress tracking.
    """
    
    def __init__(self, cache_manager=None):
        """
        Initialize the progress tracking factory.
        
        Args:
            cache_manager: Cache manager for storing progress information
        """
        # Initialize progress registry
        self.progress_registry = get_progress_registry(cache_manager)
        logger.info("Progress tracking factory initialized")
    
    def create_client(
        self,
        config: GatewayConfig = None,
        provider_factory: Optional[ProviderFactory] = None,
        db=None
    ) -> ProgressTrackingLLMClient:
        """
        Create a progress-tracking LLM client.
        
        Args:
            config: Gateway configuration
            provider_factory: Provider factory
            db: Database session
            
        Returns:
            Progress-tracking LLM client
        """
        logger.info("Creating progress-tracking LLM client")
        return ProgressTrackingLLMClient(config, provider_factory, db)
    
    def wrap_client(self, client: LLMGatewayClient) -> ProgressTrackingLLMClient:
        """
        Wrap an existing LLM client with progress tracking.
        
        This method is not fully implemented as it would require more complex
        inheritance and composition patterns. For now, it's recommended to
        create a new progress-tracking client instead.
        
        Args:
            client: Existing LLM client
            
        Returns:
            Progress-tracking LLM client
        """
        logger.warning("Wrapping existing clients is not fully supported. Creating a new client instead.")
        return ProgressTrackingLLMClient(client.config, client.provider_factory, client.db)
    
    def create_provider(
        self,
        provider_type: str,
        provider_config: ProviderConfig,
        provider_class: Optional[Type[BaseProvider]] = None
    ) -> ProgressTrackingProvider:
        """
        Create a progress-tracking provider.
        
        Args:
            provider_type: Provider type
            provider_config: Provider configuration
            provider_class: Optional provider class to use
            
        Returns:
            Progress-tracking provider
        """
        logger.info(f"Creating progress-tracking provider for type: {provider_type}")
        
        # Create a subclass of ProgressTrackingProvider that inherits from the provider class
        if provider_class:
            class ProgressTrackingProviderImpl(ProgressTrackingProvider, provider_class):
                """
                Implementation of progress-tracking provider for a specific provider class.
                """
                
                def _initialize_client_internal(self) -> None:
                    """Initialize the client using the provider class's method."""
                    provider_class._initialize_client(self)
                
                async def _generate_internal(self, request):
                    """Generate a response using the provider class's method."""
                    return await provider_class.generate(self, request)
                
                async def _generate_stream_internal(self, request):
                    """Generate a streaming response using the provider class's method."""
                    async for chunk in provider_class.generate_stream(self, request):
                        yield chunk
                
                async def _cleanup_internal(self) -> None:
                    """Clean up resources using the provider class's method."""
                    await provider_class.cleanup(self)
                
                async def _health_check_internal(self) -> Dict[str, Any]:
                    """Check health using the provider class's method."""
                    return await provider_class.health_check(self)
            
            # Create an instance of the implementation
            return ProgressTrackingProviderImpl(provider_config)
        else:
            # If no provider class is provided, return a base progress-tracking provider
            # This won't be functional on its own and should be subclassed
            return ProgressTrackingProvider(provider_config)
    
    def wrap_provider(self, provider: BaseProvider) -> ProgressTrackingProvider:
        """
        Wrap an existing provider with progress tracking.
        
        This method is not fully implemented as it would require more complex
        inheritance and composition patterns. For now, it's recommended to
        create a new progress-tracking provider instead.
        
        Args:
            provider: Existing provider
            
        Returns:
            Progress-tracking provider
        """
        logger.warning("Wrapping existing providers is not fully supported. Creating a new provider instead.")
        return self.create_provider(
            provider_type=provider.provider_config.provider_type,
            provider_config=provider.provider_config,
            provider_class=provider.__class__
        )
