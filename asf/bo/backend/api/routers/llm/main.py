"""
Main LLM router for BO backend.

This module provides the main router for LLM-related functionality,
which includes all sub-routers for specific LLM components.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, List, Optional
import logging

from ...auth import get_current_user, User

# Import sub-routers
from .gateway import router as gateway_router
from .dspy import router as dspy_router
from .biomedlm import router as biomedlm_router
from .debug import router as debug_router
from .cl_peft import router as cl_peft_router
from .mcp import router as mcp_router

# Create the main LLM router
router = APIRouter(prefix="/api/llm", tags=["llm"])

# Include sub-routers
router.include_router(gateway_router)
router.include_router(dspy_router)
router.include_router(biomedlm_router)
router.include_router(debug_router)
router.include_router(cl_peft_router)
router.include_router(mcp_router)

logger = logging.getLogger(__name__)

@router.get("/")
async def llm_root(current_user: User = Depends(get_current_user)):
    """
    Root endpoint for LLM API.

    Returns information about available LLM endpoints.
    """
    return {
        "status": "ok",
        "message": "LLM API is operational"
    }

@router.get("/status")
async def get_llm_status():
    """
    Get the status of all LLM components.

    Returns a comprehensive status report for all LLM components,
    including Gateway, DSPy, BiomedLM, CL-PEFT, and MCP.
    """
    # In a real implementation, we would check the actual status of each component
    # For now, we'll return a mock status
    return {
        "success": True,
        "data": {
            "overall_status": "operational",
            "components": {
                "gateway": {
                    "status": "available",
                    "details": {
                        "active_providers": ["openai", "anthropic", "local"],
                        "total_providers": 3,
                        "total_models": 12
                    }
                },
                "dspy": {
                    "status": "available",
                    "modules": ["MedicalRAG", "ContradictionDetection", "EntityExtraction"],
                    "modules_count": 3
                },
                "biomedlm": {
                    "status": "available",
                    "models": ["biomedlm-2-7b", "biomedlm-3-3b"],
                    "models_count": 2
                },
                "cl_peft": {
                    "status": "available",
                    "adapters": ["medical-qa-lora", "clinical-notes-qlora"],
                    "adapters_count": 2
                },
                "mcp": {
                    "status": "available",
                    "providers": ["openai-mcp", "anthropic-mcp", "local-mcp"],
                    "providers_count": 3
                }
            }
        }
    }

@router.get("/components")
async def get_llm_components(current_user: User = Depends(get_current_user)):
    """
    Get information about all LLM components.

    Returns a list of all available LLM components and their endpoints.
    """
    return {
        "status": "ok",
        "components": [
            {
                "name": "gateway",
                "description": "LLM Gateway for managing LLM providers and requests",
                "endpoints": [
                    "/api/llm/gateway/status",
                    "/api/llm/gateway/providers",
                    "/api/llm/gateway/generate"
                ]
            },
            {
                "name": "dspy",
                "description": "DSPy integration for advanced LLM programming",
                "endpoints": [
                    "/api/llm/dspy/modules",
                    "/api/llm/dspy/execute",
                    "/api/llm/dspy/optimize"
                ]
            },
            {
                "name": "biomedlm",
                "description": "BiomedLM integration for medical-specific LLM capabilities",
                "endpoints": [
                    "/api/llm/biomedlm/models",
                    "/api/llm/biomedlm/generate",
                    "/api/llm/biomedlm/finetune"
                ]
            },
            {
                "name": "debug",
                "description": "Debugging tools for LLM Gateway",
                "endpoints": [
                    "/api/llm/debug/config",
                    "/api/llm/debug/environment",
                    "/api/llm/debug/test-openai",
                    "/api/llm/debug/diagnostics",
                    "/api/llm/debug/logs"
                ]
            },
            {
                "name": "cl-peft",
                "description": "CL-PEFT for continual learning with parameter-efficient fine-tuning",
                "endpoints": [
                    "/api/llm/cl-peft/adapters",
                    "/api/llm/cl-peft/adapters/{adapter_id}",
                    "/api/llm/cl-peft/adapters/{adapter_id}/train",
                    "/api/llm/cl-peft/adapters/{adapter_id}/evaluate",
                    "/api/llm/cl-peft/adapters/{adapter_id}/forgetting",
                    "/api/llm/cl-peft/adapters/{adapter_id}/generate",
                    "/api/llm/cl-peft/strategies",
                    "/api/llm/cl-peft/peft-methods",
                    "/api/llm/cl-peft/base-models"
                ]
            },
            {
                "name": "mcp",
                "description": "Model Context Protocol (MCP) for standardized interaction with LLMs",
                "endpoints": [
                    "/api/llm/mcp/providers",
                    "/api/llm/mcp/providers/{provider_id}",
                    "/api/llm/mcp/providers/{provider_id}/test",
                    "/api/llm/mcp/providers/{provider_id}/models",
                    "/api/llm/mcp/generate"
                ]
            }
        ]
    }
