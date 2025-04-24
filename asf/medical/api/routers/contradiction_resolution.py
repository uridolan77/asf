"""Contradiction resolution API endpoints.

This module provides endpoints for resolving contradictions in medical literature.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Dict, List, Optional, Any
from pydantic import BaseModel

from ...ml.services.resolution.contradiction_resolution_service import MedicalContradictionResolutionService
from ...ml.services.resolution.resolution_models import ResolutionStrategy

from ..dependencies import get_current_user

logger = logging.getLogger(__name__)

class ContradictionResolutionRequest(BaseModel):
    """Request model for contradiction resolution."""
    claim1: str
    claim2: str
    metadata1: Optional[Dict[str, Any]] = None
    metadata2: Optional[Dict[str, Any]] = None
    strategy: Optional[str] = None
    use_combined_evidence: bool = False

class ContradictionResolutionResponse(BaseModel):
    """Response model for contradiction resolution."""
    recommendation: str
    confidence: str
    confidence_score: float
    recommended_claim: Optional[str] = None
    strategy: str
    explanation: Dict[str, Any]
    timestamp: str

router = APIRouter(
    prefix="/api/v1/contradiction-resolution",
    tags=["contradiction-resolution"],
    responses={404: {"description": "Not found"}}
)

from ..dependencies import get_contradiction_service

contradiction_service = get_contradiction_service()
resolution_service = MedicalContradictionResolutionService()

@router.post("/resolve", response_model=ContradictionResolutionResponse)
async def resolve_contradiction(
    request: ContradictionResolutionRequest,
    current_user = Depends(get_current_user)
):
    """Resolve a contradiction between two medical claims.

    Args:
        request: The contradiction resolution request containing the claims and resolution options
        current_user: The authenticated user making the request

    Returns:
        ContradictionResolutionResponse: The resolution result with recommendation and explanation

    Raises:
        HTTPException: If no contradiction is detected or if an error occurs during resolution
    """
    try:
        contradiction = await contradiction_service.detect_contradiction(
            claim1=request.claim1,
            claim2=request.claim2,
            metadata1=request.metadata1,
            metadata2=request.metadata2
        )

        if not contradiction.get("contradiction_detected", False):
            raise HTTPException(
                status_code=400,
                detail="No contradiction was detected between the provided claims."
            )

        if request.use_combined_evidence:
            resolution = await resolution_service.resolve_contradiction_with_combined_evidence(contradiction)
        else:
            strategy = None
            if request.strategy:
                try:
                    strategy = ResolutionStrategy(request.strategy)
                except ValueError:
                    logger.error(f"Error: {str(e)}")
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid resolution strategy: {request.strategy}. Valid strategies are: {', '.join([s.value for s in ResolutionStrategy])}"
                    )

            resolution = await resolution_service.resolve_contradiction(
                contradiction=contradiction,
                strategy=strategy
            )

        return resolution
    except HTTPException:
        logger.error(f"Error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        logger.error(f"API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error resolving contradiction: {str(e)}")

@router.post("/batch-resolve")
async def batch_resolve_contradictions(
    contradictions: List[Dict[str, Any]],
    strategy: Optional[str] = Query(None, description="Resolution strategy to use"),
    use_combined_evidence: bool = Query(False, description="Whether to use combined evidence approach"),
    current_user = Depends(get_current_user)
):
    """Resolve multiple contradictions in batch mode.

    Args:
        contradictions: List of contradiction data to resolve
        strategy: Optional resolution strategy to apply to all contradictions
        use_combined_evidence: Whether to use the combined evidence approach for resolution
        current_user: The authenticated user making the request

    Returns:
        Dict containing total contradictions, resolved contradictions count, and resolution results

    Raises:
        HTTPException: If no contradictions are provided or if an error occurs during resolution
    """
    try:
        if not contradictions:
            raise HTTPException(status_code=400, detail="No contradictions provided")

        strategy_enum = None
        if strategy:
            try:
                strategy_enum = ResolutionStrategy(strategy)
            except ValueError:
                logger.error(f"Error: {str(e)}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid resolution strategy: {strategy}. Valid strategies are: {', '.join([s.value for s in ResolutionStrategy])}"
                )

        resolutions = []
        for contradiction in contradictions:
            if not contradiction.get("contradiction_detected", False):
                continue

            if use_combined_evidence:
                resolution = await resolution_service.resolve_contradiction_with_combined_evidence(contradiction)
            else:
                resolution = await resolution_service.resolve_contradiction(
                    contradiction=contradiction,
                    strategy=strategy_enum
                )

            resolutions.append(resolution)

        return {
            "total_contradictions": len(contradictions),
            "resolved_contradictions": len(resolutions),
            "resolutions": resolutions
        }
    except HTTPException:
        logger.error(f"Error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        logger.error(f"API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error resolving contradictions: {str(e)}")

@router.get("/history")
async def get_resolution_history(
    current_user = Depends(get_current_user)
):
    """Get the history of contradiction resolutions.

    Args:
        current_user: The authenticated user making the request

    Returns:
        Dict containing total history entries and the history data

    Raises:
        HTTPException: If an error occurs retrieving the history
    """
    try:
        history = resolution_service.get_resolution_history()
        return {
            "total_entries": len(history),
            "history": history
        }
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting resolution history: {str(e)}")

@router.post("/feedback/{contradiction_id}")
async def add_resolution_feedback(
    contradiction_id: str,
    feedback: Dict[str, Any],
    current_user = Depends(get_current_user)
):
    """Add user feedback to a previously resolved contradiction.

    Args:
        contradiction_id: The ID of the resolved contradiction to provide feedback for
        feedback: The feedback data to associate with the resolution
        current_user: The authenticated user providing the feedback

    Returns:
        Dict containing a success message

    Raises:
        HTTPException: If the contradiction ID is not found or an error occurs adding feedback
    """
    try:
        success = resolution_service.add_resolution_feedback(
            contradiction_id=contradiction_id,
            feedback=feedback
        )

        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Contradiction with ID {contradiction_id} not found in resolution history"
            )

        return {"message": f"Feedback added to contradiction {contradiction_id}"}
    except HTTPException:
        logger.error(f"Error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        logger.error(f"Database error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error adding feedback: {str(e)}")
