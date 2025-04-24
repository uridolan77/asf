"""
LLM router for BO backend.

This module provides the main router for LLM-related functionality,
which includes all sub-routers for specific LLM components.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Body, Query
from typing import Dict, Any, List, Optional
import logging

from ..auth import get_current_user, User
from ..services.llm import get_llm_service, LLMService

# Import LLM routers
from .llm.main import router as llm_main_router
from .llm.gateway import router as llm_gateway_router
from .llm.dspy import router as llm_dspy_router
from .llm.biomedlm import router as llm_biomedlm_router
from .llm.cl_peft import router as llm_cl_peft_router
from .llm.mcp import router as llm_mcp_router

# Create the main LLM router
router = APIRouter(prefix="/api/llm", tags=["llm"])

logger = logging.getLogger(__name__)

# Include sub-routers
router.include_router(llm_main_router)
router.include_router(llm_gateway_router)
router.include_router(llm_dspy_router)
router.include_router(llm_biomedlm_router)
router.include_router(llm_cl_peft_router)
router.include_router(llm_mcp_router)

# Add additional endpoints to the main router
@router.get("/status")
async def get_llm_status(
    current_user: User = Depends(get_current_user),
    llm_service: LLMService = Depends(get_llm_service)
):
    """
    Get the status of all LLM components.

    This endpoint returns the status of all LLM components,
    including LLM Gateway, DSPy, BiomedLM, CL-PEFT, and MCP.
    """
    try:
        status = await llm_service.get_status()
        return status
    except Exception as e:
        logger.error(f"Error getting LLM status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get LLM status: {str(e)}"
        )

@router.get("/models")
async def get_available_models(
    current_user: User = Depends(get_current_user),
    llm_service: LLMService = Depends(get_llm_service)
):
    """
    Get all available models from all LLM components.

    This endpoint returns all available models from all LLM components,
    including LLM Gateway, DSPy, BiomedLM, CL-PEFT, and MCP.
    """
    try:
        models = await llm_service.get_available_models()
        return models
    except Exception as e:
        logger.error(f"Error getting available models: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get available models: {str(e)}"
        )

@router.post("/generate")
async def generate_text(
    request_data: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_user),
    llm_service: LLMService = Depends(get_llm_service)
):
    """
    Generate text using the appropriate LLM component.

    This endpoint generates text using the appropriate LLM component
    based on the request data.
    """
    try:
        result = await llm_service.generate_text(request_data)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate text: {str(e)}"
        )

@router.get("/usage")
async def get_usage_statistics(
    current_user: User = Depends(get_current_user),
    llm_service: LLMService = Depends(get_llm_service)
):
    """
    Get usage statistics for all LLM components.

    This endpoint returns usage statistics for all LLM components,
    including LLM Gateway, DSPy, BiomedLM, CL-PEFT, and MCP.
    """
    try:
        usage = await llm_service.get_usage_statistics()
        return usage
    except Exception as e:
        logger.error(f"Error getting usage statistics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get usage statistics: {str(e)}"
        )
