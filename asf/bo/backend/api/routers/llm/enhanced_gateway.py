"""
Enhanced LLM Gateway router with progress tracking.

This module provides API endpoints for interacting with the LLM Gateway
with integrated progress tracking.
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Body, status

from ...auth import get_current_user, User
from ...models.llm import (
    ProviderStatus,
    ModelInfo,
    ModelCreateRequest,
    ProviderUpdateRequest,
    LLMRequestModel,
    LLMResponseModel
)

from ...services.llm.enhanced_gateway_service import (
    get_enhanced_gateway_service,
    EnhancedGatewayService
)

router = APIRouter(prefix="/enhanced-gateway", tags=["llm-enhanced-gateway"])

logger = logging.getLogger(__name__)


# Dependency for getting the enhanced gateway service
def get_gateway_service() -> EnhancedGatewayService:
    """Get the enhanced gateway service."""
    return get_enhanced_gateway_service()


@router.get("/", response_model=Dict[str, Any])
async def enhanced_gateway_root(current_user: User = Depends(get_current_user)):
    """
    Root endpoint for enhanced LLM Gateway API.
    
    Returns information about available enhanced LLM Gateway endpoints.
    """
    return {
        "status": "ok",
        "message": "Enhanced LLM Gateway API is operational",
        "features": [
            "Progress tracking for all LLM operations",
            "Enhanced error handling",
            "Detailed performance metrics"
        ],
        "endpoints": [
            {
                "path": "/api/llm/enhanced-gateway/providers",
                "description": "Get a list of available providers"
            },
            {
                "path": "/api/llm/enhanced-gateway/models",
                "description": "Get a list of available models"
            },
            {
                "path": "/api/llm/enhanced-gateway/generate",
                "description": "Generate a response from an LLM"
            }
        ]
    }


@router.get("/providers", response_model=List[ProviderStatus])
async def get_providers(
    current_user: User = Depends(get_current_user),
    gateway_service: EnhancedGatewayService = Depends(get_gateway_service)
):
    """
    Get a list of available providers.
    
    Returns:
        List of providers
    """
    try:
        providers = await gateway_service.get_providers()
        
        # Convert to response model
        return [
            ProviderStatus(
                provider_id=provider["provider_id"],
                status=provider["status"],
                provider_type=provider["provider_type"],
                display_name=provider["display_name"],
                models=provider["models"],
                checked_at=format_timestamp(),
                message="Provider is operational"
            )
            for provider in providers
        ]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting providers: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get providers: {str(e)}"
        )


@router.get("/models", response_model=List[ModelInfo])
async def get_models(
    current_user: User = Depends(get_current_user),
    gateway_service: EnhancedGatewayService = Depends(get_gateway_service),
    provider_id: Optional[str] = None
):
    """
    Get a list of available models.
    
    Args:
        provider_id: Optional provider ID to filter by
        
    Returns:
        List of models
    """
    try:
        models = await gateway_service.get_models(provider_id)
        
        # Convert to response model
        return [
            ModelInfo(
                model_id=model["model_id"],
                provider_id=model["provider_id"],
                display_name=model["display_name"],
                model_type=model["model_type"],
                context_window=model["context_window"],
                max_tokens=model["max_tokens"],
                enabled=model["enabled"],
                additional_info=model["additional_info"]
            )
            for model in models
        ]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting models: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get models: {str(e)}"
        )


@router.post("/generate", response_model=LLMResponseModel)
async def generate_llm_response(
    request: LLMRequestModel = Body(...),
    current_user: User = Depends(get_current_user),
    gateway_service: EnhancedGatewayService = Depends(get_gateway_service)
):
    """
    Generate a response from an LLM with progress tracking.
    
    This endpoint sends a request to an LLM provider and returns the response.
    The progress of the operation is tracked and can be monitored through
    the progress tracking API.
    
    Args:
        request: LLM request
        
    Returns:
        LLM response
    """
    try:
        # Convert request to dict
        request_data = {
            "prompt": request.prompt,
            "model": request.model,
            "provider": request.provider,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": request.stream,
            "additional_params": request.additional_params
        }
        
        # Generate response
        response = await gateway_service.generate(request_data)
        
        # Convert to response model
        return LLMResponseModel(
            request_id=response["request_id"],
            text=response["text"],
            finish_reason=response["finish_reason"],
            model=response["model"],
            provider=response["provider"],
            usage=response["usage"],
            elapsed_ms=response["elapsed_ms"],
            additional_info={
                **response["additional_info"],
                "operation_id": response.get("operation_id")
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate response: {str(e)}"
        )


def format_timestamp() -> str:
    """Format the current timestamp."""
    from datetime import datetime
    return datetime.utcnow().isoformat()
