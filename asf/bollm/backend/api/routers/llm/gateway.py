"""
LLM Gateway API router for BO backend.

This module provides endpoints for managing and using the LLM Gateway,
including provider configuration, testing, and direct LLM interactions.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Body, Query, Path
from typing import Dict, Any, List, Optional, Union
from sqlalchemy.orm import Session
import httpx
import logging
import os
import yaml
import json
from datetime import datetime
import asyncio

from api.auth import get_current_user, User
from ...dependencies import get_db
from ...utils import handle_api_error
from .models import (
    ProviderStatus, GatewayStatus, ProviderUpdateRequest,
    LLMRequestModel, LLMResponseModel
)
from repositories.model_repository import ModelRepository
from .utils import (
    load_config, save_config, format_timestamp,
    get_llm_gateway_availability, GATEWAY_CONFIG_PATH
)

# Mock LLM Gateway components
LLM_GATEWAY_AVAILABLE = get_llm_gateway_availability()
if LLM_GATEWAY_AVAILABLE:
    # Mock classes for LLM Gateway
    class LLMGatewayClient:
        def __init__(self, gateway_config, provider_factory, db=None):
            self.gateway_config = gateway_config
            self.provider_factory = provider_factory
            self.db = db

        async def generate(self, llm_req):
            return MockLLMResponse("This is a mock response from the LLM Gateway.")

    class LLMRequest:
        def __init__(self, prompt_content, config, initial_context, stream=False):
            self.prompt_content = prompt_content
            self.config = config
            self.initial_context = initial_context
            self.stream = stream

    class LLMConfig:
        def __init__(self, model_identifier, temperature=0.7, max_tokens=None, system_prompt=None):
            self.model_identifier = model_identifier
            self.temperature = temperature
            self.max_tokens = max_tokens
            self.system_prompt = system_prompt

    class InterventionContext:
        def __init__(self, session_id):
            self.session_id = session_id
            self.conversation = []

        def add_conversation_turn(self, role, content):
            self.conversation.append({"role": role, "content": content})

    class ContentItem:
        def __init__(self, content, role):
            self.content = content
            self.role = role

    class GatewayConfig:
        def __init__(self, gateway_id, allowed_providers=None, default_provider=None, additional_config=None):
            self.gateway_id = gateway_id
            self.allowed_providers = allowed_providers or []
            self.default_provider = default_provider
            self.additional_config = additional_config or {}

    class ProviderConfig:
        def __init__(self, provider_id, provider_type, display_name=None, connection_params=None, models=None):
            self.provider_id = provider_id
            self.provider_type = provider_type
            self.display_name = display_name or provider_id
            self.connection_params = connection_params or {}
            self.models = models or {}

    class MCPRole:
        SYSTEM = "system"
        USER = "user"
        ASSISTANT = "assistant"

    class ProviderFactory:
        def __init__(self):
            pass

    class MockLLMResponse:
        def __init__(self, generated_content):
            self.generated_content = generated_content

router = APIRouter(prefix="/gateway", tags=["llm-gateway"])

logger = logging.getLogger(__name__)

# Gateway client instance
_gateway_client = None

def get_gateway_client(db: Session = None):
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
            config_dict = load_config(GATEWAY_CONFIG_PATH)

            # Create GatewayConfig from dict
            gateway_config = GatewayConfig(**config_dict)

            # Create provider factory
            provider_factory = ProviderFactory()

            # Create gateway client with database session
            _gateway_client = LLMGatewayClient(gateway_config, provider_factory, db)
            logger.info(f"LLM Gateway client initialized with gateway ID: {gateway_config.gateway_id}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM Gateway client: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to initialize LLM Gateway client: {str(e)}"
            )

    return _gateway_client

@router.get("/status", response_model=GatewayStatus)
async def get_gateway_status(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
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
        config = load_config(GATEWAY_CONFIG_PATH)

        # Get gateway client
        client = get_gateway_client(db)

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
                    checked_at=format_timestamp(),
                    message=message
                )
            )

        return GatewayStatus(
            gateway_id=config.get("gateway_id", "unknown"),
            status="operational" if all(p.status == "operational" for p in provider_statuses) else "degraded",
            version="1.0.0",  # This would be replaced with actual version
            default_provider=config.get("default_provider", ""),
            active_providers=provider_statuses,
            config_path=GATEWAY_CONFIG_PATH,
            checked_at=format_timestamp()
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
        config = load_config(GATEWAY_CONFIG_PATH)

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
                    checked_at=format_timestamp(),
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
        config = load_config(GATEWAY_CONFIG_PATH)

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
            checked_at=format_timestamp(),
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
        config = load_config(GATEWAY_CONFIG_PATH)

        # Check if provider exists
        if provider_id not in config.get("allowed_providers", []) and not request.enabled:
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
        save_config(GATEWAY_CONFIG_PATH, config)

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
            checked_at=format_timestamp(),
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
    db: Session = Depends(get_db),
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
        config = load_config(GATEWAY_CONFIG_PATH)

        # Check if provider exists
        if provider_id not in config.get("allowed_providers", []):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Provider '{provider_id}' not found"
            )

        # Get provider config
        provider_config = config.get("additional_config", {}).get("providers", {}).get(provider_id, {})

        # Get gateway client
        client = get_gateway_client(db)

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

@router.post("/generate", response_model=LLMResponseModel)
async def generate_llm_response(
    request: LLMRequestModel = Body(...),
    db: Session = Depends(get_db),
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
        client = get_gateway_client(db)

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
        return LLMResponseModel(
            request_id=response.request_id,
            content=response.generated_content,
            model=request.model,
            provider_id=request.provider_id or "default",
            finish_reason=response.finish_reason.value,
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
        config = load_config(GATEWAY_CONFIG_PATH)
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
        save_config(GATEWAY_CONFIG_PATH, config)
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

@router.get("/models", response_model=List[Dict[str, Any]])
async def get_models(
    provider_id: Optional[str] = Query(None, description="Filter models by provider"),
    model_type: Optional[str] = Query(None, description="Filter models by type"),
    db: Session = Depends(get_db)
):
    """
    Get all available LLM models across providers.

    This endpoint returns information about all available LLM models,
    with options to filter by provider or model type.
    """
    logger.info(f"GET /models endpoint called with provider_id={provider_id}, model_type={model_type}")

    if not LLM_GATEWAY_AVAILABLE:
        logger.error("LLM Gateway is not available")
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="LLM Gateway is not available. Please check your installation."
        )

    try:
        # Use the ModelDAO to get models directly from the database
        # Import directly to avoid circular imports
        from asf.bollm.backend.repositories.model_dao import ModelDAO

        logger.info("Creating ModelDAO instance")
        dao = ModelDAO()
        logger.info(f"Getting models with provider_id={provider_id}, model_type={model_type}")

        # Get models from database using DAO
        models = dao.get_all_models(provider_id=provider_id, model_type=model_type)
        logger.info(f"Found {len(models)} models in the database using DAO")

        logger.info(f"Returning {len(models)} models from database")
        return models
    except Exception as e:
        logger.error(f"Error getting models: {str(e)}")
        logger.exception("Exception details:")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get models: {str(e)}"
        )

@router.get("/models/{model_id}", response_model=Dict[str, Any])
async def get_model(
    model_id: str = Path(..., description="Model ID"),
    provider_id: str = Query(..., description="Provider ID"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get information about a specific LLM model.

    This endpoint returns detailed information about a specific LLM model,
    including its capabilities, parameters, and configuration.
    """
    if not LLM_GATEWAY_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="LLM Gateway is not available. Please check your installation."
        )

    try:
        # Use DAO to get model
        from asf.bollm.backend.repositories.model_dao import ModelDAO
        dao = ModelDAO()
        model = dao.get_model_by_id(model_id, provider_id)

        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_id}' not found for provider '{provider_id}'"
            )

        # Return model info
        return model
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model '{model_id}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model '{model_id}': {str(e)}"
        )

@router.post("/models", response_model=Dict[str, Any])
async def create_model(
    model_data: Dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Create a new LLM model.

    This endpoint creates a new LLM model for a specific provider.
    """
    if not LLM_GATEWAY_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="LLM Gateway is not available. Please check your installation."
        )

    try:
        # Extract required fields
        model_id = model_data.get("model_id")
        provider_id = model_data.get("provider_id")

        if not model_id or not provider_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Both model_id and provider_id are required"
            )

        # Check if provider exists in config
        config = load_config(GATEWAY_CONFIG_PATH)
        if provider_id not in config.get("allowed_providers", []):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Provider '{provider_id}' not found"
            )

        # Use DAO to create model
        from asf.bollm.backend.repositories.model_dao import ModelDAO
        dao = ModelDAO()

        # Check if model already exists
        existing_model = dao.get_model_by_id(model_id, provider_id)
        if existing_model:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Model '{model_id}' already exists for provider '{provider_id}'"
            )

        # Create model
        created_model = dao.create_model(model_data)

        if not created_model:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create model"
            )

        # Return model info
        return created_model
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create model: {str(e)}"
        )

@router.put("/models/{model_id}", response_model=Dict[str, Any])
async def update_model(
    model_id: str = Path(..., description="Model ID"),
    model_data: Dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Update an existing LLM model.

    This endpoint updates an existing LLM model for a specific provider.
    """
    if not LLM_GATEWAY_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="LLM Gateway is not available. Please check your installation."
        )

    try:
        # Extract required fields
        provider_id = model_data.get("provider_id")

        if not provider_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="provider_id is required"
            )

        # Check if provider exists in config
        config = load_config(GATEWAY_CONFIG_PATH)
        if provider_id not in config.get("allowed_providers", []):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Provider '{provider_id}' not found"
            )

        # Use DAO to update model
        from asf.bollm.backend.repositories.model_dao import ModelDAO
        dao = ModelDAO()

        # Check if model exists
        existing_model = dao.get_model_by_id(model_id, provider_id)
        if not existing_model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_id}' not found for provider '{provider_id}'"
            )

        # Update model
        updated_model = dao.update_model(model_id, provider_id, model_data)

        if not updated_model:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update model"
            )

        # Return model info
        return updated_model
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating model '{model_id}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update model '{model_id}': {str(e)}"
        )

@router.delete("/models/{model_id}", response_model=Dict[str, Any])
async def delete_model(
    model_id: str = Path(..., description="Model ID"),
    provider_id: str = Query(..., description="Provider ID"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Delete an existing LLM model.

    This endpoint deletes an existing LLM model for a specific provider.
    """
    if not LLM_GATEWAY_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="LLM Gateway is not available. Please check your installation."
        )

    try:
        # Check if provider exists in config
        config = load_config(GATEWAY_CONFIG_PATH)
        if provider_id not in config.get("allowed_providers", []):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Provider '{provider_id}' not found"
            )

        # Use DAO to delete model
        from asf.bollm.backend.repositories.model_dao import ModelDAO
        dao = ModelDAO()

        # Check if model exists
        existing_model = dao.get_model_by_id(model_id, provider_id)
        if not existing_model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_id}' not found for provider '{provider_id}'"
            )

        # Get model info for response
        model_info = existing_model

        # Delete model
        success = dao.delete_model(model_id, provider_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete model"
            )

        # Return deleted model info
        return {
            "model_id": model_id,
            "provider_id": provider_id,
            "display_name": model_info.get("display_name", model_id),
            "message": f"Model '{model_id}' deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting model '{model_id}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete model '{model_id}': {str(e)}"
        )
