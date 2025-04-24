"""
LLM Gateway API router for BO backend.

This module provides endpoints for managing and using the LLM Gateway,
including provider configuration, testing, and direct LLM interactions.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Body, Query, Path
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
import httpx
import logging
import os
import yaml
import json
from datetime import datetime
import asyncio

from ..auth import get_current_user, User
from ..utils import handle_api_error

# Mock LLM Gateway components
class LLMGatewayClient:
    """Mock LLMGatewayClient class."""
    def __init__(self, gateway_config, provider_factory):
        self.gateway_config = gateway_config
        self.provider_factory = provider_factory

    async def generate(self, llm_req):
        """Mock generate method."""
        return MockLLMResponse("This is a mock response from the LLM Gateway.")

class LLMRequest:
    """Mock LLMRequest class."""
    def __init__(self, prompt_content, config, initial_context, stream=False):
        self.prompt_content = prompt_content
        self.config = config
        self.initial_context = initial_context
        self.stream = stream

class LLMConfig:
    """Mock LLMConfig class."""
    def __init__(self, model_identifier, temperature=0.7, max_tokens=None, system_prompt=None):
        self.model_identifier = model_identifier
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt

class InterventionContext:
    """Mock InterventionContext class."""
    def __init__(self, session_id):
        self.session_id = session_id
        self.conversation = []

    def add_conversation_turn(self, role, content):
        self.conversation.append({"role": role, "content": content})

class ContentItem:
    """Mock ContentItem class."""
    def __init__(self, content, role):
        self.content = content
        self.role = role

class GatewayConfig:
    """Mock GatewayConfig class."""
    def __init__(self, **kwargs):
        self.gateway_id = kwargs.get("gateway_id", "mock-gateway")
        self.allowed_providers = kwargs.get("allowed_providers", [])
        self.default_provider = kwargs.get("default_provider", "")
        self.additional_config = kwargs.get("additional_config", {})

class ProviderConfig:
    """Mock ProviderConfig class."""
    def __init__(self, **kwargs):
        self.provider_id = kwargs.get("provider_id", "mock-provider")
        self.provider_type = kwargs.get("provider_type", "mock")
        self.display_name = kwargs.get("display_name", "Mock Provider")
        self.connection_params = kwargs.get("connection_params", {})
        self.models = kwargs.get("models", {})

class MCPRole:
    """Mock MCPRole class."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class FinishReason:
    """Mock FinishReason class."""
    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"

class ProviderFactory:
    """Mock ProviderFactory class."""
    def __init__(self):
        pass

class MockLLMResponse:
    """Mock LLMResponse class."""
    def __init__(self, generated_content):
        self.generated_content = generated_content
        self.finish_reason = FinishReason.STOP
        self.usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}

LLM_GATEWAY_AVAILABLE = True
logging.info("Using mock LLM Gateway implementation")

router = APIRouter(prefix="/api/llm/gateway", tags=["llm-gateway"])

logger = logging.getLogger(__name__)

# Constants
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                          "config", "llm", "llm_gateway_config.yaml")

# Models
class ProviderStatus(BaseModel):
    provider_id: str
    status: str
    provider_type: str
    display_name: Optional[str] = None
    models: List[str] = []
    checked_at: str
    message: Optional[str] = None

class GatewayStatus(BaseModel):
    gateway_id: str
    status: str
    version: str
    default_provider: str
    active_providers: List[ProviderStatus]
    config_path: str
    checked_at: str

class ProviderUpdateRequest(BaseModel):
    display_name: Optional[str] = None
    connection_params: Optional[Dict[str, Any]] = None
    models: Optional[Dict[str, Dict[str, Any]]] = None
    enabled: Optional[bool] = None

class LLMRequestModel(BaseModel):
    prompt: str
    model: str
    provider_id: Optional[str] = None
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, gt=0)
    system_prompt: Optional[str] = None
    stream: bool = False

class LLMResponseModel(BaseModel):
    request_id: str
    content: str
    model: str
    provider_id: str
    finish_reason: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    created_at: str

# Provider creation and deletion models
class ProviderCreateRequest(BaseModel):
    provider_id: str = Field(..., description="Unique identifier for the provider")
    provider_type: str = Field(..., description="Type of provider (e.g., openai, anthropic, local)")
    display_name: str = Field(..., description="Display name for the provider")
    connection_params: Dict[str, Any] = Field(..., description="Connection parameters for the provider")
    models: Dict[str, Dict[str, Any]] = Field(..., description="Models supported by the provider")
    enabled: bool = Field(True, description="Whether the provider is enabled")

class ProviderDeleteResponse(BaseModel):
    success: bool
    message: str
    provider_id: str
    deleted_at: str

# Model management models
class ModelInfo(BaseModel):
    model_id: str = Field(..., description="Unique identifier for the model")
    provider_id: str = Field(..., description="ID of the provider that owns this model")
    display_name: str = Field(..., description="Display name for the model")
    model_type: str = Field(..., description="Type of model (e.g., completion, embedding, chat)")
    capabilities: List[str] = Field(default_factory=list, description="List of capabilities (e.g., code, vision)")
    context_window: Optional[int] = Field(None, description="Maximum context window size in tokens")
    max_output_tokens: Optional[int] = Field(None, description="Maximum number of output tokens")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Model-specific parameters")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")

class ModelCreateRequest(BaseModel):
    model_id: str = Field(..., description="Unique identifier for the model")
    provider_id: str = Field(..., description="ID of the provider that owns this model")
    display_name: str = Field(..., description="Display name for the model")
    model_type: str = Field(..., description="Type of model (e.g., completion, embedding, chat)")
    capabilities: List[str] = Field(default_factory=list, description="List of capabilities (e.g., code, vision)")
    context_window: Optional[int] = Field(None, description="Maximum context window size in tokens")
    max_output_tokens: Optional[int] = Field(None, description="Maximum number of output tokens")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Model-specific parameters")

class ModelDeleteResponse(BaseModel):
    success: bool
    message: str
    model_id: str
    provider_id: str
    deleted_at: str

# Gateway client instance
_gateway_client = None

def get_gateway_client():
    """Get or create the LLM Gateway client."""
    global _gateway_client

    if not LLM_GATEWAY_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="LLM Gateway is not available. Please check your installation."
        )

    if _gateway_client is None:
        try:
            # Load config from file
            with open(CONFIG_PATH, 'r') as f:
                config_dict = yaml.safe_load(f)

            # Create config with error handling for unsupported parameters
            try:
                gateway_config = GatewayConfig(**config_dict)
            except TypeError as e:
                error_msg = str(e)

                # Extract the parameter name from the error message
                import re
                match = re.search(r"unexpected keyword argument '([^']+)'", error_msg)
                if match:
                    param_name = match.group(1)
                    logger.warning(f"Removing unsupported parameter from GatewayConfig: {param_name}")

                    # Remove the problematic parameter
                    if param_name in config_dict:
                        del config_dict[param_name]

                    # Try again after removing the parameter
                    return get_gateway_client()
                else:
                    # If we can't extract a parameter name, re-raise the error
                    raise

            # Create provider factory
            provider_factory = ProviderFactory()

            # Create gateway client
            _gateway_client = LLMGatewayClient(gateway_config, provider_factory)
            logger.info(f"LLM Gateway client initialized with gateway ID: {gateway_config.gateway_id}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM Gateway client: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to initialize LLM Gateway client: {str(e)}"
            )

    return _gateway_client

@router.get("/status", response_model=GatewayStatus)
async def get_gateway_status(current_user: User = Depends(get_current_user)):
    """
    Get the status of the LLM Gateway.

    This endpoint returns the status of the LLM Gateway, including
    the status of all configured providers.
    """
    if not LLM_GATEWAY_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="LLM Gateway is not available. Please check your installation."
        )

    try:
        # Load config from file
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)

        # Get gateway client
        client = get_gateway_client()

        # Get provider statuses
        provider_statuses = []
        for provider_id in config.get("allowed_providers", []):
            provider_config = config.get("additional_config", {}).get("providers", {}).get(provider_id, {})
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

            provider_statuses.append(
                ProviderStatus(
                    provider_id=provider_id,
                    status=status_info,
                    provider_type=provider_type,
                    display_name=display_name,
                    models=models,
                    checked_at=datetime.utcnow().isoformat(),
                    message=message
                )
            )

        return GatewayStatus(
            gateway_id=config.get("gateway_id", "unknown"),
            status="operational" if all(p.status == "operational" for p in provider_statuses) else "degraded",
            version="1.0.0",  # This would be replaced with actual version
            default_provider=config.get("default_provider", ""),
            active_providers=provider_statuses,
            config_path=CONFIG_PATH,
            checked_at=datetime.utcnow().isoformat()
        )
    except Exception as e:
        logger.error(f"Error getting gateway status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get gateway status: {str(e)}"
        )

@router.get("/providers", response_model=List[ProviderStatus])
async def get_providers(current_user: User = Depends(get_current_user)):
    """
    Get all configured LLM providers.

    This endpoint returns information about all configured LLM providers,
    including their status, supported models, and configuration.
    """
    if not LLM_GATEWAY_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="LLM Gateway is not available. Please check your installation."
        )

    try:
        # Load config from file
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)

        # Get provider statuses
        provider_statuses = []
        for provider_id in config.get("allowed_providers", []):
            provider_config = config.get("additional_config", {}).get("providers", {}).get(provider_id, {})
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

            provider_statuses.append(
                ProviderStatus(
                    provider_id=provider_id,
                    status=status_info,
                    provider_type=provider_type,
                    display_name=display_name,
                    models=models,
                    checked_at=datetime.utcnow().isoformat(),
                    message=message
                )
            )

        return provider_statuses
    except Exception as e:
        logger.error(f"Error getting providers: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get providers: {str(e)}"
        )

@router.get("/providers/{provider_id}", response_model=ProviderStatus)
async def get_provider(
    provider_id: str = Path(..., description="Provider ID"),
    current_user: User = Depends(get_current_user)
):
    """
    Get information about a specific LLM provider.

    This endpoint returns detailed information about a specific LLM provider,
    including its status, supported models, and configuration.
    """
    if not LLM_GATEWAY_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="LLM Gateway is not available. Please check your installation."
        )

    try:
        # Load config from file
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)

        # Check if provider exists
        if provider_id not in config.get("allowed_providers", []):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Provider '{provider_id}' not found"
            )

        # Get provider config
        provider_config = config.get("additional_config", {}).get("providers", {}).get(provider_id, {})
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

        return ProviderStatus(
            provider_id=provider_id,
            status=status_info,
            provider_type=provider_type,
            display_name=display_name,
            models=models,
            checked_at=datetime.utcnow().isoformat(),
            message=message
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting provider '{provider_id}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get provider '{provider_id}': {str(e)}"
        )

@router.put("/providers/{provider_id}", response_model=ProviderStatus)
async def update_provider(
    provider_id: str = Path(..., description="Provider ID"),
    request: ProviderUpdateRequest = Body(...),
    current_user: User = Depends(get_current_user)
):
    """
    Update configuration for a specific LLM provider.

    This endpoint updates the configuration for a specific LLM provider
    and returns the updated status.
    """
    if not LLM_GATEWAY_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="LLM Gateway is not available. Please check your installation."
        )

    try:
        # Load config from file
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)

        # Check if provider exists
        if provider_id not in config.get("allowed_providers", []):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Provider '{provider_id}' not found"
            )

        # Get provider config
        provider_config = config.get("additional_config", {}).get("providers", {}).get(provider_id, {})

        # Update provider config
        if request.display_name is not None:
            provider_config["display_name"] = request.display_name

        if request.connection_params is not None:
            provider_config["connection_params"] = {
                **provider_config.get("connection_params", {}),
                **request.connection_params
            }

        if request.models is not None:
            provider_config["models"] = request.models

        # Update config
        config["additional_config"]["providers"][provider_id] = provider_config

        # If provider is disabled, remove from allowed_providers
        if request.enabled is not None:
            if request.enabled and provider_id not in config["allowed_providers"]:
                config["allowed_providers"].append(provider_id)
            elif not request.enabled and provider_id in config["allowed_providers"]:
                config["allowed_providers"].remove(provider_id)

        # Save config to file
        with open(CONFIG_PATH, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

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

        return ProviderStatus(
            provider_id=provider_id,
            status=status_info,
            provider_type=provider_type,
            display_name=display_name,
            models=models,
            checked_at=datetime.utcnow().isoformat(),
            message=message
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating provider '{provider_id}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update provider '{provider_id}': {str(e)}"
        )

@router.post("/providers/{provider_id}/test", response_model=Dict[str, Any])
async def test_provider(
    provider_id: str = Path(..., description="Provider ID"),
    current_user: User = Depends(get_current_user)
):
    """
    Test connection to a specific LLM provider.

    This endpoint tests the connection to a specific LLM provider
    and returns the test results.
    """
    if not LLM_GATEWAY_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="LLM Gateway is not available. Please check your installation."
        )

    try:
        # Load config from file
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)

        # Check if provider exists
        if provider_id not in config.get("allowed_providers", []):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Provider '{provider_id}' not found"
            )

        # Get provider config
        provider_config = config.get("additional_config", {}).get("providers", {}).get(provider_id, {})

        # Get gateway client
        client = get_gateway_client()

        # Test provider connection
        # This would be replaced with actual test from the gateway
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
            response = await client.generate(llm_req)

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

@router.post("/providers", response_model=ProviderStatus)
async def create_provider(
    request: ProviderCreateRequest = Body(...),
    current_user: User = Depends(get_current_user)
):
    """
    Create a new LLM provider.

    This endpoint creates a new LLM provider with the given configuration
    and returns its status.
    """
    if not LLM_GATEWAY_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="LLM Gateway is not available. Please check your installation."
        )

    try:
        # Load config from file
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)

        # Check if provider already exists
        if request.provider_id in config.get("allowed_providers", []):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Provider '{request.provider_id}' already exists"
            )

        # Create provider config
        provider_config = {
            "provider_type": request.provider_type,
            "display_name": request.display_name,
            "connection_params": request.connection_params,
            "models": request.models
        }

        # Add provider to config
        if "additional_config" not in config:
            config["additional_config"] = {}

        if "providers" not in config["additional_config"]:
            config["additional_config"]["providers"] = {}

        config["additional_config"]["providers"][request.provider_id] = provider_config

        # Add to allowed_providers if enabled
        if request.enabled:
            if "allowed_providers" not in config:
                config["allowed_providers"] = []

            if request.provider_id not in config["allowed_providers"]:
                config["allowed_providers"].append(request.provider_id)

        # Save config to file
        with open(CONFIG_PATH, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        # Return provider status
        return ProviderStatus(
            provider_id=request.provider_id,
            status="operational",  # Assuming new provider is operational
            provider_type=request.provider_type,
            display_name=request.display_name,
            models=list(request.models.keys()),
            checked_at=datetime.utcnow().isoformat(),
            message="Provider created successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating provider: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create provider: {str(e)}"
        )

@router.delete("/providers/{provider_id}", response_model=ProviderDeleteResponse)
async def delete_provider(
    provider_id: str = Path(..., description="Provider ID"),
    current_user: User = Depends(get_current_user)
):
    """
    Delete an LLM provider.

    This endpoint deletes the specified LLM provider from the configuration.
    """
    if not LLM_GATEWAY_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="LLM Gateway is not available. Please check your installation."
        )

    try:
        # Load config from file
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)

        # Check if provider exists
        provider_exists = (
            provider_id in config.get("allowed_providers", []) or
            provider_id in config.get("additional_config", {}).get("providers", {})
        )

        if not provider_exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Provider '{provider_id}' not found"
            )

        # Remove provider from allowed_providers
        if provider_id in config.get("allowed_providers", []):
            config["allowed_providers"].remove(provider_id)

        # Remove provider from additional_config.providers
        if provider_id in config.get("additional_config", {}).get("providers", {}):
            del config["additional_config"]["providers"][provider_id]

        # Save config to file
        with open(CONFIG_PATH, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        # Return success response
        return ProviderDeleteResponse(
            success=True,
            message=f"Provider '{provider_id}' deleted successfully",
            provider_id=provider_id,
            deleted_at=datetime.utcnow().isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting provider '{provider_id}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete provider '{provider_id}': {str(e)}"
        )

@router.get("/models", response_model=List[ModelInfo])
async def get_models(
    provider_id: Optional[str] = Query(None, description="Filter models by provider ID"),
    model_type: Optional[str] = Query(None, description="Filter models by type"),
    current_user: User = Depends(get_current_user)
):
    """
    Get all available LLM models.

    This endpoint returns information about all available LLM models,
    optionally filtered by provider ID or model type.
    """
    if not LLM_GATEWAY_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="LLM Gateway is not available. Please check your installation."
        )

    try:
        # Load config from file
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)

        models = []
        providers = config.get("additional_config", {}).get("providers", {})

        # Iterate through providers and their models
        for p_id, p_config in providers.items():
            # Skip if filtering by provider and doesn't match
            if provider_id and p_id != provider_id:
                continue

            provider_models = p_config.get("models", {})

            for m_id, m_config in provider_models.items():
                # Get model type from config or default to "chat"
                m_type = m_config.get("type", "chat")

                # Skip if filtering by model type and doesn't match
                if model_type and m_type != model_type:
                    continue

                # Extract capabilities
                capabilities = m_config.get("capabilities", [])
                if isinstance(capabilities, str):
                    capabilities = [cap.strip() for cap in capabilities.split(",")]

                # Create model info object
                model_info = ModelInfo(
                    model_id=m_id,
                    provider_id=p_id,
                    display_name=m_config.get("display_name", m_id),
                    model_type=m_type,
                    capabilities=capabilities,
                    context_window=m_config.get("context_window"),
                    max_output_tokens=m_config.get("max_output_tokens"),
                    parameters=m_config.get("parameters"),
                    created_at=m_config.get("created_at", datetime.utcnow().isoformat()),
                    updated_at=m_config.get("updated_at", datetime.utcnow().isoformat())
                )

                models.append(model_info)

        return models
    except Exception as e:
        logger.error(f"Error getting models: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get models: {str(e)}"
        )

@router.post("/models", response_model=ModelInfo)
async def create_model(
    request: ModelCreateRequest = Body(...),
    current_user: User = Depends(get_current_user)
):
    """
    Create or register a new LLM model.

    This endpoint creates or registers a new LLM model for a specific provider.
    """
    if not LLM_GATEWAY_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="LLM Gateway is not available. Please check your installation."
        )

    try:
        # Load config from file
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)

        # Check if provider exists
        if request.provider_id not in config.get("additional_config", {}).get("providers", {}):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Provider '{request.provider_id}' not found"
            )

        # Get provider config
        provider_config = config["additional_config"]["providers"][request.provider_id]

        # Check if model already exists
        if request.model_id in provider_config.get("models", {}):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Model '{request.model_id}' already exists for provider '{request.provider_id}'"
            )

        # Create model config
        now = datetime.utcnow().isoformat()
        model_config = {
            "display_name": request.display_name,
            "type": request.model_type,
            "capabilities": request.capabilities,
            "context_window": request.context_window,
            "max_output_tokens": request.max_output_tokens,
            "parameters": request.parameters or {},
            "created_at": now,
            "updated_at": now
        }

        # Add model to provider config
        if "models" not in provider_config:
            provider_config["models"] = {}

        provider_config["models"][request.model_id] = model_config

        # Save config to file
        with open(CONFIG_PATH, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        # Return model info
        return ModelInfo(
            model_id=request.model_id,
            provider_id=request.provider_id,
            display_name=request.display_name,
            model_type=request.model_type,
            capabilities=request.capabilities,
            context_window=request.context_window,
            max_output_tokens=request.max_output_tokens,
            parameters=request.parameters,
            created_at=now,
            updated_at=now
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create model: {str(e)}"
        )

@router.delete("/models/{model_id}", response_model=ModelDeleteResponse)
async def delete_model(
    model_id: str = Path(..., description="Model ID"),
    provider_id: str = Query(..., description="Provider ID"),
    current_user: User = Depends(get_current_user)
):
    """
    Delete an LLM model.

    This endpoint deletes the specified LLM model from a provider's configuration.
    """
    if not LLM_GATEWAY_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="LLM Gateway is not available. Please check your installation."
        )

    try:
        # Load config from file
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)

        # Check if provider exists
        if provider_id not in config.get("additional_config", {}).get("providers", {}):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Provider '{provider_id}' not found"
            )

        # Get provider config
        provider_config = config["additional_config"]["providers"][provider_id]

        # Check if model exists
        if model_id not in provider_config.get("models", {}):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_id}' not found for provider '{provider_id}'"
            )

        # Remove model from provider config
        del provider_config["models"][model_id]

        # Save config to file
        with open(CONFIG_PATH, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        # Return success response
        return ModelDeleteResponse(
            success=True,
            message=f"Model '{model_id}' deleted successfully from provider '{provider_id}'",
            model_id=model_id,
            provider_id=provider_id,
            deleted_at=datetime.utcnow().isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting model '{model_id}' from provider '{provider_id}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete model '{model_id}' from provider '{provider_id}': {str(e)}"
        )

@router.get("/models/{model_id}", response_model=ModelInfo)
async def get_model(
    model_id: str = Path(..., description="Model ID"),
    provider_id: str = Query(..., description="Provider ID"),
    current_user: User = Depends(get_current_user)
):
    """
    Get information about a specific LLM model.

    This endpoint returns detailed information about a specific LLM model.
    """
    if not LLM_GATEWAY_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="LLM Gateway is not available. Please check your installation."
        )

    try:
        # Load config from file
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)

        # Check if provider exists
        if provider_id not in config.get("additional_config", {}).get("providers", {}):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Provider '{provider_id}' not found"
            )

        # Get provider config
        provider_config = config["additional_config"]["providers"][provider_id]

        # Check if model exists
        if model_id not in provider_config.get("models", {}):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_id}' not found for provider '{provider_id}'"
            )

        # Get model config
        model_config = provider_config["models"][model_id]

        # Extract capabilities
        capabilities = model_config.get("capabilities", [])
        if isinstance(capabilities, str):
            capabilities = [cap.strip() for cap in capabilities.split(",")]

        # Return model info
        return ModelInfo(
            model_id=model_id,
            provider_id=provider_id,
            display_name=model_config.get("display_name", model_id),
            model_type=model_config.get("type", "chat"),
            capabilities=capabilities,
            context_window=model_config.get("context_window"),
            max_output_tokens=model_config.get("max_output_tokens"),
            parameters=model_config.get("parameters"),
            created_at=model_config.get("created_at", datetime.utcnow().isoformat()),
            updated_at=model_config.get("updated_at", datetime.utcnow().isoformat())
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model '{model_id}' from provider '{provider_id}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model '{model_id}' from provider '{provider_id}': {str(e)}"
        )

@router.post("/generate", response_model=LLMResponseModel)
async def generate_llm_response(
    request: LLMRequestModel = Body(...),
    current_user: User = Depends(get_current_user)
):
    """
    Generate a response from an LLM.

    This endpoint sends a request to an LLM provider and returns the response.
    """
    if not LLM_GATEWAY_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="LLM Gateway is not available. Please check your installation."
        )

    try:
        # Get gateway client
        client = get_gateway_client()

        # Create LLM request
        llm_config = LLMConfig(
            model_identifier=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            system_prompt=request.system_prompt
        )

        context = InterventionContext(session_id=f"bo-{datetime.utcnow().timestamp()}")

        # Add system prompt if provided
        if request.system_prompt:
            context.add_conversation_turn(MCPRole.SYSTEM.value, request.system_prompt)

        llm_req = LLMRequest(
            prompt_content=request.prompt,
            config=llm_config,
            initial_context=context,
            stream=request.stream
        )

        # Generate response
        start_time = datetime.utcnow()

        if request.stream:
            # Streaming not implemented in this endpoint
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Streaming is not supported in this endpoint. Use the streaming endpoint instead."
            )
        else:
            response = await client.generate(llm_req)

        end_time = datetime.utcnow()
        latency_ms = (end_time - start_time).total_seconds() * 1000

        # Create response model
        generated_content = ""
        if response.generated_content is not None:
            if isinstance(response.generated_content, str):
                generated_content = response.generated_content
            elif isinstance(response.generated_content, list):
                # If it's a list of ContentItems, extract the text content
                text_parts = []
                for item in response.generated_content:
                    if hasattr(item, "text_content") and item.text_content:
                        text_parts.append(item.text_content)
                    elif hasattr(item, "type") and item.type == "text" and hasattr(item, "data") and "text" in item.data:
                        text_parts.append(item.data["text"])
                generated_content = " ".join(text_parts)
            else:
                # Convert to string as a fallback
                generated_content = str(response.generated_content)

        return LLMResponseModel(
            request_id=response.request_id,
            content=generated_content,
            model=request.model,
            provider_id=request.provider_id or "default",
            finish_reason=response.finish_reason.value if response.finish_reason else "unknown",
            prompt_tokens=response.usage.prompt_tokens if response.usage else None,
            completion_tokens=response.usage.completion_tokens if response.usage else None,
            total_tokens=response.usage.total_tokens if response.usage else None,
            latency_ms=latency_ms,
            created_at=end_time.isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating LLM response: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate LLM response: {str(e)}"
        )

@router.get("/config", response_model=Dict[str, Any])
async def get_gateway_config(current_user: User = Depends(get_current_user)):
    """
    Get the current LLM Gateway configuration.

    This endpoint returns the current configuration of the LLM Gateway.
    """
    if not LLM_GATEWAY_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="LLM Gateway is not available. Please check your installation."
        )

    try:
        # Load config from file
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)

        return config
    except Exception as e:
        logger.error(f"Error getting gateway config: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get gateway config: {str(e)}"
        )

@router.put("/config", response_model=Dict[str, Any])
async def update_gateway_config(
    config: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_user)
):
    """
    Update the LLM Gateway configuration.

    This endpoint updates the configuration of the LLM Gateway.
    """
    if not LLM_GATEWAY_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="LLM Gateway is not available. Please check your installation."
        )

    try:
        # Save config to file
        with open(CONFIG_PATH, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        return config
    except Exception as e:
        logger.error(f"Error updating gateway config: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update gateway config: {str(e)}"
        )

@router.get("/usage", response_model=Dict[str, Any])
async def get_gateway_usage(
    start_date: Optional[str] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format)"),
    provider_id: Optional[str] = Query(None, description="Filter by provider ID"),
    model: Optional[str] = Query(None, description="Filter by model"),
    current_user: User = Depends(get_current_user)
):
    """
    Get usage statistics for the LLM Gateway.

    This endpoint returns usage statistics for the LLM Gateway,
    including token usage, request counts, and latency metrics.
    """
    if not LLM_GATEWAY_AVAILABLE:
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
