"""
Enhanced LLM Gateway service with progress tracking.

This module provides a service for interacting with the LLM Gateway
with integrated progress tracking.
"""

import logging
import yaml
import os
from typing import Dict, Any, List, Optional
from fastapi import HTTPException, status

from asf.medical.llm_gateway.core.models import (
    GatewayConfig, LLMRequest, LLMResponse, LLMConfig,
    InterventionContext, ContentItem
)
from asf.medical.llm_gateway.core.enhanced_factory import get_enhanced_gateway_factory

logger = logging.getLogger(__name__)

# Check if LLM Gateway is available
try:
    from asf.medical.llm_gateway.core.progress_client import ProgressTrackingLLMClient
    LLM_GATEWAY_AVAILABLE = True
except ImportError:
    LLM_GATEWAY_AVAILABLE = False
    logger.warning("LLM Gateway is not available. Some features will be disabled.")

# Path to the gateway configuration file
CONFIG_PATH = os.environ.get("LLM_GATEWAY_CONFIG_PATH", "config/llm_gateway_config.yaml")


class EnhancedGatewayService:
    """
    Enhanced service for interacting with the LLM Gateway with progress tracking.
    
    This service provides methods for generating text, managing providers,
    and other LLM Gateway operations with integrated progress tracking.
    """
    
    def __init__(self):
        """Initialize the enhanced gateway service."""
        self._client = None
        self._config = None
        
        # Initialize client if LLM Gateway is available
        if LLM_GATEWAY_AVAILABLE:
            self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the LLM Gateway client with progress tracking."""
        try:
            # Load config from file
            with open(CONFIG_PATH, 'r') as f:
                self._config = yaml.safe_load(f)
            
            # Create GatewayConfig from dict
            gateway_config = GatewayConfig(**self._config)
            
            # Get enhanced gateway factory
            factory = get_enhanced_gateway_factory()
            
            # Create enhanced LLM client
            self._client = factory.create_client(config=gateway_config)
            
            logger.info(f"Enhanced LLM Gateway client initialized with gateway ID: {gateway_config.gateway_id}")
        except Exception as e:
            logger.error(f"Failed to initialize enhanced LLM Gateway client: {str(e)}")
            self._client = None
    
    async def generate(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a response from an LLM with progress tracking.
        
        Args:
            request_data: Request data
            
        Returns:
            Generated response
        """
        if not LLM_GATEWAY_AVAILABLE or not self._client:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="LLM Gateway is not available. Please check your installation."
            )
        
        try:
            # Create request ID
            import uuid
            request_id = request_data.get("request_id", str(uuid.uuid4()))
            
            # Create context
            context = InterventionContext(request_id=request_id)
            
            # Create content item
            content_item = ContentItem(
                content_type="text",
                content=request_data.get("prompt", "")
            )
            
            # Create config
            config = LLMConfig(
                model_identifier=request_data.get("model"),
                provider_id=request_data.get("provider"),
                max_tokens=request_data.get("max_tokens"),
                temperature=request_data.get("temperature"),
                stream=request_data.get("stream", False),
                additional_params=request_data.get("additional_params", {})
            )
            
            # Create request
            request = LLMRequest(
                initial_context=context,
                prompt_content=content_item,
                config=config
            )
            
            # Generate response
            response = await self._client.generate(request)
            
            # Convert response to dict
            return {
                "request_id": response.final_context.request_id,
                "text": response.generated_content.content,
                "finish_reason": response.finish_reason,
                "model": response.model_info.get("model_identifier", "unknown"),
                "provider": response.model_info.get("provider_id", "unknown"),
                "usage": response.usage.dict() if response.usage else {},
                "elapsed_ms": response.performance_metrics.total_duration_ms if response.performance_metrics else 0,
                "additional_info": response.additional_info,
                "operation_id": response.final_context.intervention_data.get("operation_id", None)
            }
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate response: {str(e)}"
            )
    
    async def get_providers(self) -> List[Dict[str, Any]]:
        """
        Get a list of available providers.
        
        Returns:
            List of providers
        """
        if not LLM_GATEWAY_AVAILABLE or not self._client:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="LLM Gateway is not available. Please check your installation."
            )
        
        try:
            # Get providers from client
            providers = await self._client.get_active_providers()
            
            # Convert to list of dicts
            return [
                {
                    "provider_id": provider.provider_id,
                    "provider_type": provider.provider_type,
                    "display_name": provider.display_name,
                    "models": list(provider.supported_models.keys()),
                    "status": "available"
                }
                for provider in providers
            ]
        except Exception as e:
            logger.error(f"Error getting providers: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get providers: {str(e)}"
            )
    
    async def get_models(self, provider_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get a list of available models.
        
        Args:
            provider_id: Optional provider ID to filter by
            
        Returns:
            List of models
        """
        if not LLM_GATEWAY_AVAILABLE or not self._client:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="LLM Gateway is not available. Please check your installation."
            )
        
        try:
            # Get providers from client
            providers = await self._client.get_active_providers()
            
            # Filter by provider ID if provided
            if provider_id:
                providers = [p for p in providers if p.provider_id == provider_id]
            
            # Get models from providers
            models = []
            for provider in providers:
                for model_id, model_info in provider.supported_models.items():
                    models.append({
                        "model_id": model_id,
                        "provider_id": provider.provider_id,
                        "display_name": model_info.get("display_name", model_id),
                        "model_type": model_info.get("model_type"),
                        "context_window": model_info.get("context_window"),
                        "max_tokens": model_info.get("max_tokens"),
                        "enabled": model_info.get("enabled", True),
                        "additional_info": {
                            k: v for k, v in model_info.items()
                            if k not in ["display_name", "model_type", "context_window", "max_tokens", "enabled"]
                        }
                    })
            
            return models
        except Exception as e:
            logger.error(f"Error getting models: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get models: {str(e)}"
            )


# Singleton instance
_enhanced_gateway_service = None


def get_enhanced_gateway_service() -> EnhancedGatewayService:
    """
    Get the singleton instance of the enhanced gateway service.
    
    Returns:
        Enhanced gateway service
    """
    global _enhanced_gateway_service
    if _enhanced_gateway_service is None:
        _enhanced_gateway_service = EnhancedGatewayService()
    return _enhanced_gateway_service
