"""
DSPy API Router for DSPyDashboard integration.

This module provides endpoints for interacting with DSPy functionality,
including listing available modules and other DSPy-related operations.
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from pydantic import BaseModel

# Define router with prefix and tags
router = APIRouter(prefix="/dspy", tags=["dspy"])

class DspyModule(BaseModel):
    """Model representing a DSPy module."""
    name: str
    description: str
    category: str = "Default"
    parameters: Dict[str, Any] = {}

@router.get("/modules", response_model=List[DspyModule])
async def get_dspy_modules():
    """
    Get all available DSPy modules.
    
    Returns:
        List[DspyModule]: A list of available DSPy modules
    """
    # For now, return a placeholder list of modules
    # This can be expanded later to dynamically discover and return actual modules
    return [
        DspyModule(
            name="Prompt",
            description="Basic DSPy prompt module",
            category="Core",
            parameters={"template": "string"}
        ),
        DspyModule(
            name="ChainOfThought",
            description="Chain of thought reasoning module",
            category="Reasoning",
            parameters={"question": "string", "reasoning_steps": "integer"}
        ),
        DspyModule(
            name="RAG",
            description="Retrieval Augmented Generation module",
            category="Retrieval",
            parameters={"query": "string", "k": "integer"}
        )
    ]

@router.get("/status")
async def get_dspy_status():
    """
    Get the status of the DSPy integration.
    
    Returns:
        dict: Status information about DSPy
    """
    return {
        "status": "operational",
        "version": "0.1.0",
        "modules_available": 3
    }