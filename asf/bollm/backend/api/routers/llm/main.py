"""
Main LLM router for BO backend.

This module provides the main router for LLM-related functionality,
which includes all sub-routers for specific LLM components.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, List
import logging

from ...auth import get_current_user, User

# Import sub-routers
from .gateway import router as gateway_router
from .service import router as service_router
from .dspy import router as dspy_router
from .biomedlm import router as biomedlm_router
from .debug import router as debug_router
from .cl_peft import router as cl_peft_router
from .mcp import router as mcp_router

# Create the main LLM router
router = APIRouter(prefix="/api/llm", tags=["llm"])

# Include sub-routers
router.include_router(gateway_router)
router.include_router(service_router)
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
                "name": "service",
                "description": "LLM Service Abstraction Layer for enhanced LLM capabilities",
                "endpoints": [
                    "/api/llm/service/config",
                    "/api/llm/service/health",
                    "/api/llm/service/stats",
                    "/api/llm/service/cache/clear",
                    "/api/llm/service/resilience/reset-circuit-breakers"
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
