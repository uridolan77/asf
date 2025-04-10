"""
Contradiction detection API endpoints.

This module provides API endpoints for detecting contradictions between medical claims.
"""

import logging
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, status, Request, Response

from asf.medical.api.models.base import APIResponse, ErrorResponse
from asf.medical.api.models.contradiction import (
    ContradictionRequest,
    ContradictionResponse,
    BatchContradictionRequest,
    BatchContradictionResponse
)
from asf.medical.api.auth import get_current_active_user
from asf.medical.storage.models import User
from asf.medical.ml.services.enhanced_contradiction_service import EnhancedContradictionService
from asf.medical.core.monitoring import async_timed, log_error
from asf.medical.api.dependencies import get_contradiction_service

# Set up logging
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/contradiction", tags=["contradiction"])

@router.post("/detect", response_model=APIResponse[ContradictionResponse])
@async_timed("detect_contradiction_endpoint")
async def detect_contradiction(
    request: ContradictionRequest,
    req: Request = None,
    res: Response = None,
    current_user: User = Depends(get_current_active_user),
    contradiction_service: EnhancedContradictionService = Depends(get_contradiction_service)
):
    """
    Detect contradiction between two claims.

    This endpoint detects contradictions between two claims using rule-based approaches
    and text similarity. It provides detailed results for each detection method.

    Args:
        request: Contradiction detection request
        req: Request object
        res: Response object
        current_user: Current authenticated user

    Returns:
        Contradiction detection result
    """
    try:
        # Add request ID to response headers for tracing
        request_id = req.headers.get("X-Request-ID") if req else None
        if request_id and res:
            res.headers["X-Request-ID"] = request_id

        # Log the request
        logger.info(f"Contradiction detection request: claim1='{request.claim1[:50]}...', claim2='{request.claim2[:50]}...', user_id={current_user.id}")

        # Detect contradiction
        result = await contradiction_service.detect_contradiction(
            claim1=request.claim1,
            claim2=request.claim2,
            metadata1=request.metadata1,
            metadata2=request.metadata2,
            threshold=request.threshold,
            use_biomedlm=request.use_biomedlm,
            use_temporal=request.use_temporal,
            use_tsmixer=request.use_tsmixer
        )

        # Return result
        return APIResponse(
            status="success",
            message="Contradiction detection completed successfully",
            data=result
        )
    except Exception as e:
        # Log error
        log_error(e, {
            "endpoint": "detect_contradiction",
            "user_id": current_user.id,
            "claim1": request.claim1[:50],
            "claim2": request.claim2[:50]
        })

        # Return error response
        return ErrorResponse(
            status="error",
            message=f"Error detecting contradiction: {str(e)}",
            error_code="CONTRADICTION_DETECTION_ERROR"
        )

@router.post("/analyze-batch", response_model=APIResponse[BatchContradictionResponse])
@async_timed("analyze_batch_contradictions_endpoint")
async def analyze_batch_contradictions(
    request: BatchContradictionRequest,
    req: Request = None,
    res: Response = None,
    current_user: User = Depends(get_current_active_user),
    contradiction_service: EnhancedContradictionService = Depends(get_contradiction_service)
):
    """
    Analyze contradictions in a batch of articles.

    This endpoint analyzes contradictions between multiple articles and provides
    detailed results for each detected contradiction.

    Args:
        request: Batch contradiction analysis request
        req: Request object
        res: Response object
        current_user: Current authenticated user

    Returns:
        Batch contradiction analysis result
    """
    try:
        # Add request ID to response headers for tracing
        request_id = req.headers.get("X-Request-ID") if req else None
        if request_id and res:
            res.headers["X-Request-ID"] = request_id

        # Log the request
        logger.info(f"Batch contradiction analysis request: {len(request.articles)} articles, user_id={current_user.id}")

        # Validate request
        if len(request.articles) < 2:
            return ErrorResponse(
                status="error",
                message="At least 2 articles are required for contradiction analysis",
                error_code="INVALID_REQUEST"
            )

        # Detect contradictions
        contradictions = await contradiction_service.detect_contradictions_in_articles(
            articles=request.articles,
            threshold=request.threshold,
            use_biomedlm=request.use_biomedlm,
            use_temporal=request.use_temporal,
            use_tsmixer=request.use_tsmixer
        )

        # Create response
        response = {
            "contradictions": contradictions,
            "total_articles": len(request.articles),
            "total_contradictions": len(contradictions),
            "threshold": request.threshold
        }

        # Return result
        return APIResponse(
            status="success",
            message=f"Found {len(contradictions)} contradictions in {len(request.articles)} articles",
            data=response
        )
    except Exception as e:
        # Log error
        log_error(e, {
            "endpoint": "analyze_batch_contradictions",
            "user_id": current_user.id,
            "article_count": len(request.articles)
        })

        # Return error response
        return ErrorResponse(
            status="error",
            message=f"Error analyzing contradictions: {str(e)}",
            error_code="CONTRADICTION_ANALYSIS_ERROR"
        )
