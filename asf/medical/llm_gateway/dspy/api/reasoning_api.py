"""Reasoning API

This module provides FastAPI endpoints for general reasoning functionality.
"""

import logging
from typing import Dict, Any, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Path
from pydantic import BaseModel, Field

from ..client import get_enhanced_client
from ..modules.reasoning import ReasoningModule, ExpertReasoningModule

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/dspy/reasoning", tags=["reasoning"])


# Pydantic models for requests and responses
class ReasoningRequest(BaseModel):
    """Request model for general reasoning."""
    
    problem_description: str = Field(..., description="Problem or scenario description")
    include_alternatives: bool = Field(True, description="Whether to include alternative solutions")
    max_solutions: int = Field(5, description="Maximum number of solutions to include")


class ExpertReasoningRequest(BaseModel):
    """Request model for expert reasoning."""
    
    problem_description: str = Field(..., description="Problem or scenario description")
    domain_expertise: str = Field(..., description="Domain of expertise (e.g., 'software', 'finance')")
    include_alternatives: bool = Field(True, description="Whether to include alternative solutions")
    max_solutions: int = Field(5, description="Maximum number of solutions to include")


# Dependency for getting DSPy client
async def get_dspy_client_dep():
    """Dependency for getting DSPy client."""
    return await get_enhanced_client()


@router.post("/general", response_model=Dict[str, Any])
async def general_reasoning(
    request: ReasoningRequest,
    dspy_client = Depends(get_dspy_client_dep)
) -> Dict[str, Any]:
    """
    Perform general reasoning on a given problem.
    
    Args:
        request: Reasoning request
        
    Returns:
        Dict[str, Any]: Reasoning result
    """
    # Create module
    module = ReasoningModule(
        max_solutions=request.max_solutions,
        include_alternatives=request.include_alternatives
    )
    
    # Call module
    try:
        result = await dspy_client.call_module(
            module_name="general_reasoning",
            module=module,
            problem_description=request.problem_description
        )
        return result
    except Exception as e:
        logger.error(f"Error in general reasoning: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error in general reasoning: {str(e)}"
        )


@router.post("/expert", response_model=Dict[str, Any])
async def expert_reasoning(
    request: ExpertReasoningRequest,
    dspy_client = Depends(get_dspy_client_dep)
) -> Dict[str, Any]:
    """
    Get expert reasoning for a problem.
    
    Args:
        request: Expert reasoning request
        
    Returns:
        Dict[str, Any]: Expert reasoning result
    """
    # Create base reasoning module
    base_module = ReasoningModule(
        max_solutions=request.max_solutions,
        include_alternatives=request.include_alternatives
    )
    
    # Create expert module
    expert_module = ExpertReasoningModule(
        domain=request.domain_expertise,
        base_reasoning_module=base_module
    )
    
    # Call module
    try:
        result = await dspy_client.call_module(
            module_name="expert_reasoning",
            module=expert_module,
            problem_description=request.problem_description
        )
        return result
    except Exception as e:
        logger.error(f"Error in expert reasoning: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error in expert reasoning: {str(e)}"
        )


@router.get("/domains", response_model=List[str])
async def list_domains() -> List[str]:
    """
    List available domains of expertise for consultation.
    
    Returns:
        List[str]: List of available domains
    """
    # Return a list of supported domains
    return [
        "software_development",
        "data_science",
        "finance",
        "marketing",
        "legal",
        "product_management",
        "project_management",
        "human_resources",
        "operations",
        "strategy",
        "education",
        "research"
    ]


# Export router
__all__ = ['router']
