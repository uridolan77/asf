"""
Main LLM router for BO backend.

This module provides the main router for LLM-related functionality,
which includes all sub-routers for specific LLM components.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, List
import logging

from ...auth import get_current_user, User

# Create the main LLM router
router = APIRouter(prefix="/api/llm", tags=["llm"])

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
            }
        ]
    }
