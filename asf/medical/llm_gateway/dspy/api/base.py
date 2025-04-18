DSPy API Integration

This module provides FastAPI endpoints for DSPy functionality.

import logging
import asyncio
from typing import Dict, Any, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Path
from pydantic import BaseModel, Field

from client import get_dspy_client, DSPyClient
from settings import get_dspy_settings
from .modules import (
    MedicalRAGModule,
    EnhancedMedicalRAGModule,
    ContradictionDetectionModule,
    TemporalContradictionModule,
    EvidenceExtractionModule,
    MedicalSummarizationModule,
    ClinicalQAModule
)

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/dspy", tags=["dspy"])


# Pydantic models for requests and responses
class DSPyModuleListResponse(BaseModel):
    Response model for listing DSPy modules.
    # Create module
    module = ContradictionDetectionModule()
    
    # Call module
    try:
        result = module(
            statement1=request.statement1,
            statement2=request.statement2
        )
        return result
    except Exception as e:
        logger.error(f"Error in contradiction detection: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error in contradiction detection: {str(e)}"
        )


@router.post("/temporal-contradiction", response_model=Dict[str, Any])
async def temporal_contradiction(
    request: DSPyTemporalContradictionRequest,
    dspy_client: DSPyClient = Depends(get_dspy_client_dep)
) -> Dict[str, Any]:
    """
    Detect temporal contradictions between medical statements.
    
    Args:
        request: Temporal contradiction detection request
        
    Returns:
        Dict[str, Any]: Temporal contradiction detection result
    """
    # Create module
    module = TemporalContradictionModule()
    
    # Call module
    try:
        result = module(
            statement1=request.statement1,
            timestamp1=request.timestamp1,
            statement2=request.statement2,
            timestamp2=request.timestamp2
        )
        return result
    except Exception as e:
        logger.error(f"Error in temporal contradiction detection: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error in temporal contradiction detection: {str(e)}"
        )


@router.post("/evidence-extraction", response_model=Dict[str, Any])
async def evidence_extraction(
    request: DSPyEvidenceExtractionRequest,
    dspy_client: DSPyClient = Depends(get_dspy_client_dep)
) -> Dict[str, Any]:
    """
    Extract evidence from medical text.
    
    Args:
        request: Evidence extraction request
        
    Returns:
        Dict[str, Any]: Evidence extraction result
    """
    # Create module
    module = EvidenceExtractionModule()
    
    # Call module
    try:
        result = module(
            text=request.text,
            claim=request.claim
        )
        return result
    except Exception as e:
        logger.error(f"Error in evidence extraction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error in evidence extraction: {str(e)}"
        )


@router.post("/medical-summarization", response_model=Dict[str, Any])
async def medical_summarization(
    request: DSPyMedicalSummarizationRequest,
    dspy_client: DSPyClient = Depends(get_dspy_client_dep)
) -> Dict[str, Any]:
    """
    Summarize medical content.
    
    Args:
        request: Medical summarization request
        
    Returns:
        Dict[str, Any]: Summarization result
    """
    # Create module
    module = MedicalSummarizationModule()
    
    # Call module
    try:
        result = module(
            text=request.text,
            audience=request.audience
        )
        return result
    except Exception as e:
        logger.error(f"Error in medical summarization: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error in medical summarization: {str(e)}"
        )


@router.post("/clinical-qa", response_model=Dict[str, Any])
async def clinical_qa(
    request: DSPyClinicalQARequest,
    dspy_client: DSPyClient = Depends(get_dspy_client_dep)
) -> Dict[str, Any]:
    """
    Answer a clinical question.
    
    Args:
        request: Clinical QA request
        
    Returns:
        Dict[str, Any]: Clinical QA result
    """
    # Create module
    module = ClinicalQAModule()
    
    # Call module
    try:
        result = module(question=request.question)
        return result
    except Exception as e:
        logger.error(f"Error in clinical QA: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error in clinical QA: {str(e)}"
        )


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Check the health of the DSPy API.
    
    Returns:
        Dict[str, str]: Health status
    """
    return {"status": "healthy"}


@router.post("/cache/clear")
async def clear_cache(
    pattern: Optional[str] = None,
    dspy_client: DSPyClient = Depends(get_dspy_client_dep)
) -> Dict[str, str]:
    """
    Clear the DSPy cache.
    
    Args:
        pattern: Optional pattern to match cache keys
        
    Returns:
        Dict[str, str]: Clear result
    """
    await dspy_client.clear_cache(pattern=pattern)
    return {"status": "cache cleared"}


# Export router
__all__ = ['router']
