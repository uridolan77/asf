"""
Base provider interface for the Conexus LLM Gateway.

This module defines the contract that all provider implementations must follow.
Providers are responsible for translating between the gateway's standard 
request/response format and the specific APIs of different LLM services.
"""

import abc
import logging
from datetime import datetime
from typing import AsyncGenerator, Optional, Dict, Any, List

from asf.conexus.llm_gateway.core.models import (
    LLMRequest,
    LLMResponse,
    ProviderConfig,
    GatewayConfig,
    StreamChunk,
)

logger = logging.getLogger(__name__)


class BaseProvider(abc.ABC):
    """
    Abstract base class for all LLM Gateway provider implementations.
    
    A provider is responsible for connecting to a specific LLM service,
    sending requests, and processing responses according to the gateway's
    standardized formats.
    """

    def __init__(
        self, 
        provider_config: Dict[str, Any],
        gateway_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a provider with its configuration.
        
        Args:
            provider_config: Configuration specific to this provider
            gateway_config: General gateway configuration
        """
        self.provider_config = ProviderConfig(**provider_config)
        self.gateway_config = GatewayConfig(**(gateway_config or {}))
        self.provider_id = self.provider_config.provider_id
        self.provider_type = self.provider_config.provider_type
        self._initialized = False
        self._client = None
        
        # Extract common settings
        self.max_retries = self.gateway_config.max_retries
        self.default_timeout = self.gateway_config.default_timeout_seconds
        
        logger.info(f"Initialized {self.__class__.__name__} with ID: {self.provider_id}")
    
    async def initialize_async(self) -> None:
        """Initialize the provider (async version)."""
        if not self._initialized:
            await self._initialize_client_async()
            self._initialized = True
            logger.info(f"Provider {self.provider_id} initialized successfully")
    
    def initialize(self) -> None:
        """Initialize the provider (sync version)."""
        if not self._initialized:
            self._initialize_client()
            self._initialized = True
            logger.info(f"Provider {self.provider_id} initialized successfully")
    
    def _initialize_client(self) -> None:
        """Initialize the client for this provider (sync)."""
        # Default implementation - override in subclasses if needed
        pass
    
    async def _initialize_client_async(self) -> None:
        """Initialize the client for this provider (async)."""
        # Default implementation - override in subclasses if needed
        pass
    
    @abc.abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate a response for the given request.
        
        Args:
            request: The LLM request to process
            
        Returns:
            An LLM response
        """
        pass
    
    @abc.abstractmethod
    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[StreamChunk, None]:
        """
        Generate a streaming response for the given request.
        
        Args:
            request: The LLM request to process
            
        Yields:
            Stream chunks containing partial responses
        """
        pass
    
    @abc.abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources used by this provider."""
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check the health/status of this provider.
        
        Returns:
            Dictionary with health check results
        """
        return {
            "status": "operational",
            "provider_id": self.provider_id,
            "provider_type": self.provider_type,
            "initialized": self._initialized,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_model_info(self, model_identifier: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific model.
        
        Args:
            model_identifier: The ID of the model
            
        Returns:
            Model information or None if not found
        """
        models = self.supported_models
        return next(
            (model for model in models.get("models", []) 
             if model.get("model_id") == model_identifier),
            None
        )
    
    @property
    def supported_models(self) -> Dict[str, Any]:
        """
        Get all models supported by this provider.
        
        Returns:
            Dictionary with supported models information
        """
        # Default implementation - should be overridden by subclasses
        return {
            "provider_id": self.provider_id,
            "models": []
        }
        
    @property
    def default_model(self) -> Optional[str]:
        """Get the default model for this provider."""
        # Default implementation - should be overridden by subclasses
        models = self.supported_models.get("models", [])
        return models[0].get("model_id") if models else None