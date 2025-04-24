"""DSPy API Integration

This module provides FastAPI endpoints for DSPy functionality.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Path
from pydantic import BaseModel, Field

from client import get_dspy_client, DSPyClient
from settings import get_dspy_settings
from .modules import (
    AdvancedRAGModule,
    MultiStageRAGModule,
    ContradictionDetectionModule,
    TemporalContradictionModule,
    EvidenceExtractionModule,
    ContentSummarizationModule,
    AdvancedQAModule
)

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/dspy", tags=["dspy"])


# Pydantic models for requests and responses
class DSPyModuleListResponse(BaseModel):
    """Response model for listing DSPy modules."""
    modules: List[str] = Field(..., description="List of available DSPy modules")


# Request models
class DSPyContradictionRequest(BaseModel):
    """Request model for contradiction detection."""
    statement1: str = Field(..., description="First statement")
    statement2: str = Field(..., description="Second statement")


class DSPyTemporalContradictionRequest(BaseModel):
    """Request model for temporal contradiction detection."""
    statement1: str = Field(..., description="First statement")
    timestamp1: str = Field(..., description="Timestamp of first statement")
    statement2: str = Field(..., description="Second statement")
    timestamp2: str = Field(..., description="Timestamp of second statement")


class DSPyEvidenceExtractionRequest(BaseModel):
    """Request model for evidence extraction."""
    text: str = Field(..., description="Text to analyze")
    claim: str = Field(..., description="Claim to evaluate")


class DSPyContentSummarizationRequest(BaseModel):
    """Request model for content summarization."""
    text: str = Field(..., description="Text to summarize")
    audience: str = Field("expert", description="Target audience (expert, researcher, general)")


class DSPyAdvancedQARequest(BaseModel):
    """Request model for advanced QA."""
    question: str = Field(..., description="Question to answer")


@router.post("/contradiction", response_model=Dict[str, Any])
async def contradiction(
    request: DSPyContradictionRequest,
    dspy_client: DSPyClient = Depends(get_dspy_client)
) -> Dict[str, Any]:
    """
    Detect contradictions between statements.

    Args:
        request: Contradiction detection request

    Returns:
        Dict[str, Any]: Contradiction detection result
    """
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
    dspy_client: DSPyClient = Depends(get_dspy_client)
) -> Dict[str, Any]:
    """
    Detect temporal contradictions between statements.

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
    dspy_client: DSPyClient = Depends(get_dspy_client)
) -> Dict[str, Any]:
    """
    Extract evidence from text.

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


@router.post("/content-summarization", response_model=Dict[str, Any])
async def content_summarization(
    request: DSPyContentSummarizationRequest,
    dspy_client: DSPyClient = Depends(get_dspy_client)
) -> Dict[str, Any]:
    """
    Summarize content.

    Args:
        request: Content summarization request

    Returns:
        Dict[str, Any]: Summarization result
    """
    # Create module
    module = ContentSummarizationModule()

    # Call module
    try:
        result = module(
            text=request.text,
            audience=request.audience
        )
        return result
    except Exception as e:
        logger.error(f"Error in content summarization: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error in content summarization: {str(e)}"
        )


@router.post("/advanced-qa", response_model=Dict[str, Any])
async def advanced_qa(
    request: DSPyAdvancedQARequest,
    dspy_client: DSPyClient = Depends(get_dspy_client)
) -> Dict[str, Any]:
    """
    Answer a complex question with evidence assessment.

    Args:
        request: Advanced QA request

    Returns:
        Dict[str, Any]: Advanced QA result
    """
    # Create module
    module = AdvancedQAModule()

    # Call module
    try:
        result = module(question=request.question)
        return result
    except Exception as e:
        logger.error(f"Error in advanced QA: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error in advanced QA: {str(e)}"
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
    dspy_client: DSPyClient = Depends(get_dspy_client)
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
