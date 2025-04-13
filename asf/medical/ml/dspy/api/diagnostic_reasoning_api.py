"""Diagnostic Reasoning API

This module provides FastAPI endpoints for diagnostic reasoning functionality.
"""

import logging
from typing import Dict, Any, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Path
from pydantic import BaseModel, Field

from ..client import get_enhanced_client
from ..modules.diagnostic_reasoning import DiagnosticReasoningModule, SpecialistConsultModule

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/dspy/diagnostic", tags=["diagnostic-reasoning"])


# Pydantic models for requests and responses
class DiagnosticReasoningRequest(BaseModel):
    """Request model for diagnostic reasoning."""
    
    case_description: str = Field(..., description="Clinical case description")
    include_rare_conditions: bool = Field(True, description="Whether to include rare conditions in differential")
    max_diagnoses: int = Field(5, description="Maximum number of diagnoses to include")


class SpecialistConsultRequest(BaseModel):
    """Request model for specialist consultation."""
    
    case_description: str = Field(..., description="Clinical case description")
    specialty: str = Field(..., description="Medical specialty (e.g., 'cardiology', 'neurology')")
    include_rare_conditions: bool = Field(True, description="Whether to include rare conditions in differential")
    max_diagnoses: int = Field(5, description="Maximum number of diagnoses to include")


# Dependency for getting DSPy client
async def get_dspy_client_dep():
    """Dependency for getting DSPy client."""
    return await get_enhanced_client()


@router.post("/reasoning", response_model=Dict[str, Any])
async def diagnostic_reasoning(
    request: DiagnosticReasoningRequest,
    dspy_client = Depends(get_dspy_client_dep)
) -> Dict[str, Any]:
    """
    Perform diagnostic reasoning on a clinical case.
    
    Args:
        request: Diagnostic reasoning request
        
    Returns:
        Dict[str, Any]: Diagnostic reasoning result
    """
    # Create module
    module = DiagnosticReasoningModule(
        max_diagnoses=request.max_diagnoses,
        include_rare_conditions=request.include_rare_conditions
    )
    
    # Call module
    try:
        result = await dspy_client.call_module(
            module_name="diagnostic_reasoning",
            module=module,
            case_description=request.case_description
        )
        return result
    except Exception as e:
        logger.error(f"Error in diagnostic reasoning: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error in diagnostic reasoning: {str(e)}"
        )


@router.post("/specialist-consult", response_model=Dict[str, Any])
async def specialist_consult(
    request: SpecialistConsultRequest,
    dspy_client = Depends(get_dspy_client_dep)
) -> Dict[str, Any]:
    """
    Get a specialist consultation for a clinical case.
    
    Args:
        request: Specialist consultation request
        
    Returns:
        Dict[str, Any]: Specialist consultation result
    """
    # Create base diagnostic module
    base_module = DiagnosticReasoningModule(
        max_diagnoses=request.max_diagnoses,
        include_rare_conditions=request.include_rare_conditions
    )
    
    # Create specialist module
    specialist_module = SpecialistConsultModule(
        specialty=request.specialty,
        base_reasoning_module=base_module
    )
    
    # Call module
    try:
        result = await dspy_client.call_module(
            module_name="specialist_consult",
            module=specialist_module,
            case_description=request.case_description
        )
        return result
    except Exception as e:
        logger.error(f"Error in specialist consultation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error in specialist consultation: {str(e)}"
        )


@router.get("/specialties", response_model=List[str])
async def list_specialties() -> List[str]:
    """
    List available medical specialties for consultation.
    
    Returns:
        List[str]: List of available specialties
    """
    # Return a list of supported specialties
    return [
        "cardiology",
        "neurology",
        "pulmonology",
        "gastroenterology",
        "endocrinology",
        "nephrology",
        "hematology",
        "oncology",
        "rheumatology",
        "infectious_disease",
        "psychiatry",
        "emergency_medicine"
    ]


# Export router
__all__ = ['router']
