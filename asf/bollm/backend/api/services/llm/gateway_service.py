"""
LLM Gateway service for BO backend.

This module provides a service for interacting with the LLM Gateway.
"""

import os
import yaml
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from fastapi import Depends, HTTPException, status

from ...utils import handle_api_error

# Import LLM Gateway components if available
try:
    from asf.medical.llm_gateway.core.client import LLMGatewayClient
    from asf.medical.llm_gateway.core.models import (
        LLMRequest, LLMConfig, InterventionContext, ContentItem,
        GatewayConfig, ProviderConfig, MCPRole
    )
    from asf.medical.llm_gateway.core.factory import ProviderFactory
    LLM_GATEWAY_AVAILABLE = True
except ImportError:
    LLM_GATEWAY_AVAILABLE = False
    logging.warning("LLM Gateway not available. Some functionality will be limited.")

logger = logging.getLogger(__name__)

# Constants
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 
                          "config", "llm", "llm_gateway_config.yaml")

class LLMGatewayService:
    """
    Service for interacting with the LLM Gateway.
    """
    
    def __init__(self):
        """
        Initialize the LLM Gateway service.
        """
        self._client = None
        self._config = None
        
        if not LLM_GATEWAY_AVAILABLE:
            logger.warning("LLM Gateway is not available. Some functionality will be limited.")
            return
        
        try:
            # Load config from file
            with open(CONFIG_PATH, 'r') as f:
                self._config = yaml.safe_load(f)
            
            # Create GatewayConfig from dict
            gateway_config = GatewayConfig(**self._config)
            
            # Create provider factory
            provider_factory = ProviderFactory()
            
            # Create gateway client
            self._client = LLMGatewayClient(gateway_config, provider_factory)
            logger.info(f"LLM Gateway client initialized with gateway ID: {gateway_config.gateway_id}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM Gateway client: {str(e)}")
    
    async def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the LLM Gateway.
        
        Returns:
            Dictionary with gateway status information
        """
        if not LLM_GATEWAY_AVAILABLE or not self._client:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="LLM Gateway is not available. Please check your installation."
            )
        
        try:
            # Get provider statuses
            provider_statuses = []
            for provider_id in self._config.get("allowed_providers", []):
                provider_config = self._config.get("additional_config", {}).get("providers", {}).get(provider_id, {})
                provider_type = provider_config.get("provider_type", "unknown")
                display_name = provider_config.get("display_name", provider_id)
                models = list(provider_config.get("models", {}).keys())
                
                # Check provider status
                try:
                    # This would be replaced with actual status check from the gateway
                    status_info = "operational"
                    message = "Provider is operational"
                except Exception as e:
                    status_info = "error"
                    message = str(e)
                
                provider_statuses.append({
                    "provider_id": provider_id,
                    "status": status_info,
                    "provider_type": provider_type,
                    "display_name": display_name,
                    "models": models,
                    "checked_at": datetime.utcnow().isoformat(),
                    "message": message
                })
            
            return {
                "gateway_id": self._config.get("gateway_id", "unknown"),
                "status": "operational" if all(p["status"] == "operational" for p in provider_statuses) else "degraded",
                "version": "1.0.0",  # This would be replaced with actual version
                "default_provider": self._config.get("default_provider", ""),
                "active_providers": provider_statuses,
                "config_path": CONFIG_PATH,
                "checked_at": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting gateway status: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get gateway status: {str(e)}"
            )
    
    async def get_providers(self) -> List[Dict[str, Any]]:
        """
        Get all configured LLM providers.
        
        Returns:
            List of provider information dictionaries
        """
        if not LLM_GATEWAY_AVAILABLE or not self._client:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="LLM Gateway is not available. Please check your installation."
            )
        
        try:
            # Get provider statuses
            provider_statuses = []
            for provider_id in self._config.get("allowed_providers", []):
                provider_config = self._config.get("additional_config", {}).get("providers", {}).get(provider_id, {})
                provider_type = provider_config.get("provider_type", "unknown")
                display_name = provider_config.get("display_name", provider_id)
                models = list(provider_config.get("models", {}).keys())
                
                # Check provider status
                try:
                    # This would be replaced with actual status check from the gateway
                    status_info = "operational"
                    message = "Provider is operational"
                except Exception as e:
                    status_info = "error"
                    message = str(e)
                
                provider_statuses.append({
                    "provider_id": provider_id,
                    "status": status_info,
                    "provider_type": provider_type,
                    "display_name": display_name,
                    "models": models,
                    "checked_at": datetime.utcnow().isoformat(),
                    "message": message
                })
            
            return provider_statuses
        except Exception as e:
            logger.error(f"Error getting providers: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get providers: {str(e)}"
            )
    
    async def get_provider(self, provider_id: str) -> Dict[str, Any]:
        """
        Get information about a specific LLM provider.
        
        Args:
            provider_id: Provider ID
            
        Returns:
            Provider information dictionary
        """
        if not LLM_GATEWAY_AVAILABLE or not self._client:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="LLM Gateway is not available. Please check your installation."
            )
        
        try:
            # Check if provider exists
            if provider_id not in self._config.get("allowed_providers", []):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Provider '{provider_id}' not found"
                )
            
            # Get provider config
            provider_config = self._config.get("additional_config", {}).get("providers", {}).get(provider_id, {})
            provider_type = provider_config.get("provider_type", "unknown")
            display_name = provider_config.get("display_name", provider_id)
            models = list(provider_config.get("models", {}).keys())
            
            # Check provider status
            try:
                # This would be replaced with actual status check from the gateway
                status_info = "operational"
                message = "Provider is operational"
            except Exception as e:
                status_info = "error"
                message = str(e)
            
            return {
                "provider_id": provider_id,
                "status": status_info,
                "provider_type": provider_type,
                "display_name": display_name,
                "models": models,
                "checked_at": datetime.utcnow().isoformat(),
                "message": message
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting provider '{provider_id}': {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get provider '{provider_id}': {str(e)}"
            )
    
    async def update_provider(self, provider_id: str, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update configuration for a specific LLM provider.
        
        Args:
            provider_id: Provider ID
            update_data: Provider update data
            
        Returns:
            Updated provider information
        """
        if not LLM_GATEWAY_AVAILABLE or not self._client:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="LLM Gateway is not available. Please check your installation."
            )
        
        try:
            # Check if provider exists
            if provider_id not in self._config.get("allowed_providers", []) and not update_data.get("enabled"):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Provider '{provider_id}' not found"
                )
            
            # Get provider config
            provider_config = self._config.get("additional_config", {}).get("providers", {}).get(provider_id, {})
            
            # Update provider config
            if "display_name" in update_data:
                provider_config["display_name"] = update_data["display_name"]
            
            if "connection_params" in update_data:
                provider_config["connection_params"] = {
                    **provider_config.get("connection_params", {}),
                    **update_data["connection_params"]
                }
            
            if "models" in update_data:
                provider_config["models"] = update_data["models"]
            
            # Update config
            self._config["additional_config"]["providers"][provider_id] = provider_config
            
            # If provider is disabled, remove from allowed_providers
            if "enabled" in update_data:
                if update_data["enabled"] and provider_id not in self._config["allowed_providers"]:
                    self._config["allowed_providers"].append(provider_id)
                elif not update_data["enabled"] and provider_id in self._config["allowed_providers"]:
                    self._config["allowed_providers"].remove(provider_id)
            
            # Save config to file
            with open(CONFIG_PATH, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False)
            
            # Get updated provider status
            provider_type = provider_config.get("provider_type", "unknown")
            display_name = provider_config.get("display_name", provider_id)
            models = list(provider_config.get("models", {}).keys())
            
            # Check provider status
            try:
                # This would be replaced with actual status check from the gateway
                status_info = "operational"
                message = "Provider is operational"
            except Exception as e:
                status_info = "error"
                message = str(e)
            
            return {
                "provider_id": provider_id,
                "status": status_info,
                "provider_type": provider_type,
                "display_name": display_name,
                "models": models,
                "checked_at": datetime.utcnow().isoformat(),
                "message": message
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error updating provider '{provider_id}': {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to update provider '{provider_id}': {str(e)}"
            )
    
    async def test_provider(self, provider_id: str) -> Dict[str, Any]:
        """
        Test connection to a specific LLM provider.
        
        Args:
            provider_id: Provider ID
            
        Returns:
            Test results
        """
        if not LLM_GATEWAY_AVAILABLE or not self._client:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="LLM Gateway is not available. Please check your installation."
            )
        
        try:
            # Check if provider exists
            if provider_id not in self._config.get("allowed_providers", []):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Provider '{provider_id}' not found"
                )
            
            # Get provider config
            provider_config = self._config.get("additional_config", {}).get("providers", {}).get(provider_id, {})
            
            # Test provider connection
            test_start = datetime.utcnow()
            
            # Create a simple test request
            model_id = next(iter(provider_config.get("models", {}).keys()), None)
            if not model_id:
                return {
                    "success": False,
                    "message": "No models configured for this provider",
                    "provider_id": provider_id,
                    "tested_at": test_start.isoformat(),
                    "duration_ms": 0
                }
            
            # Create a test request
            llm_config = LLMConfig(model_identifier=model_id)
            context = InterventionContext(session_id=f"test-{datetime.utcnow().timestamp()}")
            
            llm_req = LLMRequest(
                prompt_content="Hello, this is a test request. Please respond with 'Test successful'.",
                config=llm_config,
                initial_context=context
            )
            
            try:
                # Send test request
                response = await self._client.generate(llm_req)
                
                test_end = datetime.utcnow()
                duration_ms = (test_end - test_start).total_seconds() * 1000
                
                return {
                    "success": True,
                    "message": "Provider connection test successful",
                    "provider_id": provider_id,
                    "model_tested": model_id,
                    "response": response.generated_content,
                    "tested_at": test_start.isoformat(),
                    "duration_ms": duration_ms
                }
            except Exception as e:
                test_end = datetime.utcnow()
                duration_ms = (test_end - test_start).total_seconds() * 1000
                
                return {
                    "success": False,
                    "message": f"Provider connection test failed: {str(e)}",
                    "provider_id": provider_id,
                    "model_tested": model_id,
                    "tested_at": test_start.isoformat(),
                    "duration_ms": duration_ms,
                    "error": str(e)
                }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error testing provider '{provider_id}': {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to test provider '{provider_id}': {str(e)}"
            )
    
    async def generate(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a response from an LLM.
        
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
            # Create LLM request
            llm_config = LLMConfig(
                model_identifier=request_data["model"],
                temperature=request_data.get("temperature"),
                max_tokens=request_data.get("max_tokens"),
                system_prompt=request_data.get("system_prompt")
            )
            
            context = InterventionContext(session_id=f"bo-{datetime.utcnow().timestamp()}")
            
            # Add system prompt if provided
            if request_data.get("system_prompt"):
                context.add_conversation_turn(MCPRole.SYSTEM.value, request_data["system_prompt"])
            
            llm_req = LLMRequest(
                prompt_content=request_data["prompt"],
                config=llm_config,
                initial_context=context,
                stream=request_data.get("stream", False)
            )
            
            # Generate response
            start_time = datetime.utcnow()
            
            if request_data.get("stream", False):
                # Streaming not implemented in this method
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Streaming is not supported in this method. Use the streaming method instead."
                )
            else:
                response = await self._client.generate(llm_req)
            
            end_time = datetime.utcnow()
            latency_ms = (end_time - start_time).total_seconds() * 1000
            
            # Create response
            return {
                "request_id": response.request_id,
                "content": response.generated_content,
                "model": request_data["model"],
                "provider_id": request_data.get("provider_id", "default"),
                "finish_reason": response.finish_reason.value,
                "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                "completion_tokens": response.usage.completion_tokens if response.usage else None,
                "total_tokens": response.usage.total_tokens if response.usage else None,
                "latency_ms": latency_ms,
                "created_at": end_time.isoformat()
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate LLM response: {str(e)}"
            )
    
    async def get_config(self) -> Dict[str, Any]:
        """
        Get the current LLM Gateway configuration.
        
        Returns:
            Gateway configuration
        """
        if not LLM_GATEWAY_AVAILABLE or not self._client:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="LLM Gateway is not available. Please check your installation."
            )
        
        return self._config
    
    async def update_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the LLM Gateway configuration.
        
        Args:
            config: New configuration
            
        Returns:
            Updated configuration
        """
        if not LLM_GATEWAY_AVAILABLE or not self._client:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="LLM Gateway is not available. Please check your installation."
            )
        
        try:
            # Save config to file
            with open(CONFIG_PATH, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            # Update instance config
            self._config = config
            
            return config
        except Exception as e:
            logger.error(f"Error updating gateway config: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to update gateway config: {str(e)}"
            )
    
    async def get_usage(self, start_date: Optional[str] = None, end_date: Optional[str] = None,
                        provider_id: Optional[str] = None, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Get usage statistics for the LLM Gateway.
        
        Args:
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            provider_id: Filter by provider ID
            model: Filter by model
            
        Returns:
            Usage statistics
        """
        if not LLM_GATEWAY_AVAILABLE or not self._client:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="LLM Gateway is not available. Please check your installation."
            )
        
        try:
            # This would be replaced with actual usage statistics from the gateway
            return {
                "total_requests": 100,
                "total_tokens": 25000,
                "prompt_tokens": 10000,
                "completion_tokens": 15000,
                "average_latency_ms": 250.5,
                "providers": {
                    "openai_gpt4_default": {
                        "requests": 75,
                        "tokens": 20000
                    },
                    "anthropic_claude3": {
                        "requests": 25,
                        "tokens": 5000
                    }
                },
                "models": {
                    "gpt-4-turbo-preview": {
                        "requests": 50,
                        "tokens": 15000
                    },
                    "gpt-3.5-turbo": {
                        "requests": 25,
                        "tokens": 5000
                    },
                    "claude-3-sonnet-20240229": {
                        "requests": 25,
                        "tokens": 5000
                    }
                },
                "period": {
                    "start_date": start_date or "2023-01-01T00:00:00Z",
                    "end_date": end_date or datetime.utcnow().isoformat()
                }
            }
        except Exception as e:
            logger.error(f"Error getting gateway usage: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get gateway usage: {str(e)}"
            )

# Singleton instance
_llm_gateway_service = None

def get_llm_gateway_service():
    """
    Get the LLM Gateway service instance.
    
    Returns:
        LLM Gateway service instance
    """
    global _llm_gateway_service
    
    if _llm_gateway_service is None:
        _llm_gateway_service = LLMGatewayService()
    
    return _llm_gateway_service
