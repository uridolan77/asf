"""
Unified contradiction detection API endpoints.

This module provides a unified API for contradiction detection, replacing
the separate contradiction and enhanced_contradiction routers.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

from asf.medical.ml.services.unified_contradiction_service import UnifiedContradictionService
from asf.medical.api.dependencies import get_current_user, get_current_active_user
from asf.medical.api.models.base import APIResponse, ErrorResponse
from asf.medical.storage.models import User
from asf.medical.core.monitoring import async_timed, log_error

# Initialize router
router = APIRouter(prefix="/contradiction", tags=["contradiction"])

# Set up logging
logger = logging.getLogger(__name__)

# Define models
class ContradictionRequest(BaseModel):
    """Request model for contradiction detection."""
    claim1: str = Field(..., description="First claim to compare")
    claim2: str = Field(..., description="Second claim to compare")
    metadata1: Optional[Dict[str, Any]] = Field(None, description="Metadata for the first claim")
    metadata2: Optional[Dict[str, Any]] = Field(None, description="Metadata for the second claim")
    threshold: float = Field(0.7, description="Contradiction detection threshold")
    use_biomedlm: bool = Field(True, description="Whether to use BioMedLM for contradiction detection")
    use_tsmixer: bool = Field(False, description="Whether to use TSMixer for contradiction detection")
    use_lorentz: bool = Field(False, description="Whether to use Lorentz embeddings for contradiction detection")
    use_temporal: bool = Field(False, description="Whether to use temporal analysis for contradiction detection")

class ContradictionResponse(BaseModel):
    """Response model for contradiction detection."""
    claim1: str = Field(..., description="First claim")
    claim2: str = Field(..., description="Second claim")
    is_contradiction: bool = Field(..., description="Whether a contradiction was detected")
    contradiction_score: float = Field(..., description="Contradiction score")
    contradiction_type: str = Field(..., description="Type of contradiction")
    confidence: str = Field(..., description="Confidence in the contradiction detection")
    explanation: str = Field(..., description="Explanation of the contradiction")
    methods_used: List[str] = Field(..., description="Methods used for contradiction detection")
    details: Dict[str, Any] = Field(..., description="Detailed results for each method")
    classification: Optional[Dict[str, Any]] = Field(None, description="Enhanced classification of the contradiction")

class ArticleContradictionRequest(BaseModel):
    """Request model for article contradiction detection."""
    articles: List[Dict[str, Any]] = Field(..., description="List of articles to analyze")
    threshold: float = Field(0.7, description="Contradiction detection threshold")
    use_biomedlm: bool = Field(True, description="Whether to use BioMedLM for contradiction detection")
    use_tsmixer: bool = Field(False, description="Whether to use TSMixer for contradiction detection")
    use_lorentz: bool = Field(False, description="Whether to use Lorentz embeddings for contradiction detection")
    use_temporal: bool = Field(False, description="Whether to use temporal analysis for contradiction detection")

class BatchClaimsRequest(BaseModel):
    """Request model for batch contradiction detection."""
    claims: List[Dict[str, str]] = Field(..., description="List of claims to analyze")
    threshold: float = Field(0.7, description="Contradiction detection threshold")
    use_biomedlm: bool = Field(True, description="Whether to use BioMedLM for contradiction detection")
    use_tsmixer: bool = Field(False, description="Whether to use TSMixer for contradiction detection")
    use_lorentz: bool = Field(False, description="Whether to use Lorentz embeddings for contradiction detection")
    use_temporal: bool = Field(False, description="Whether to use temporal analysis for contradiction detection")

# Dependency for getting the contradiction service
def get_contradiction_service() -> UnifiedContradictionService:
    """Get the unified contradiction service."""
    return UnifiedContradictionService()

@router.post("/detect", response_model=APIResponse[ContradictionResponse])
@async_timed("detect_contradiction_endpoint")
async def detect_contradiction(
    request: ContradictionRequest,
    req: Request = None,
    res: Response = None,
    current_user: User = Depends(get_current_active_user),
    contradiction_service: UnifiedContradictionService = Depends(get_contradiction_service)
):
    """
    Detect contradiction between two claims.

    This endpoint detects contradictions between two claims using multiple methods,
    including BioMedLM, TSMixer, Lorentz embeddings, and temporal analysis.

    Args:
        request: Contradiction detection request
        req: Request object
        res: Response object
        current_user: Current authenticated user
        contradiction_service: Unified contradiction service

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
            use_tsmixer=request.use_tsmixer,
            use_lorentz=request.use_lorentz,
            use_temporal=request.use_temporal
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

@router.post("/analyze-articles", response_model=APIResponse[Dict[str, Any]])
@async_timed("analyze_articles_contradiction_endpoint")
async def analyze_articles_contradiction(
    request: ArticleContradictionRequest,
    current_user: User = Depends(get_current_active_user),
    contradiction_service: UnifiedContradictionService = Depends(get_contradiction_service)
):
    """
    Analyze contradictions in a list of articles.

    This endpoint analyzes contradictions between multiple articles using multiple methods,
    including BioMedLM, TSMixer, Lorentz embeddings, and temporal analysis.

    Args:
        request: Article contradiction detection request
        current_user: Current authenticated user
        contradiction_service: Unified contradiction service

    Returns:
        Article contradiction detection result
    """
    try:
        # Validate input
        if not request.articles or len(request.articles) < 2:
            return ErrorResponse(
                status="error",
                message="At least 2 articles are required",
                error_code="INVALID_INPUT"
            )

        # Log the request
        logger.info(f"Article contradiction analysis request: {len(request.articles)} articles, user_id={current_user.id}")

        # Detect contradictions
        contradictions = await contradiction_service.detect_contradictions_in_articles(
            articles=request.articles,
            threshold=request.threshold,
            use_biomedlm=request.use_biomedlm,
            use_tsmixer=request.use_tsmixer,
            use_lorentz=request.use_lorentz,
            use_temporal=request.use_temporal
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
            "endpoint": "analyze_articles_contradiction",
            "user_id": current_user.id,
            "num_articles": len(request.articles)
        })

        # Return error response
        return ErrorResponse(
            status="error",
            message=f"Error analyzing contradictions: {str(e)}",
            error_code="CONTRADICTION_ANALYSIS_ERROR"
        )

@router.post("/analyze-batch", response_model=APIResponse[Dict[str, Any]])
@async_timed("analyze_batch_contradictions_endpoint")
async def analyze_batch_contradictions(
    request: BatchClaimsRequest,
    current_user: User = Depends(get_current_active_user),
    contradiction_service: UnifiedContradictionService = Depends(get_contradiction_service)
):
    """
    Analyze contradictions in a batch of claims.

    This endpoint analyzes contradictions between multiple claims using multiple methods,
    including BioMedLM, TSMixer, Lorentz embeddings, and temporal analysis.

    Args:
        request: Batch claims contradiction detection request
        current_user: Current authenticated user
        contradiction_service: Unified contradiction service

    Returns:
        Batch contradiction detection result
    """
    try:
        # Validate input
        if not request.claims or len(request.claims) < 2:
            return ErrorResponse(
                status="error",
                message="At least 2 claims are required",
                error_code="INVALID_INPUT"
            )

        # Log the request
        logger.info(f"Batch contradiction analysis request: {len(request.claims)} claims, user_id={current_user.id}")

        # Analyze contradictions
        contradictions = []
        for i in range(len(request.claims)):
            for j in range(i + 1, len(request.claims)):
                # Extract claims
                claim1 = request.claims[i].get("text", "")
                claim2 = request.claims[j].get("text", "")
                
                # Skip if either claim is empty
                if not claim1.strip() or not claim2.strip():
                    continue
                
                # Extract metadata
                metadata1 = request.claims[i].get("metadata", {})
                metadata2 = request.claims[j].get("metadata", {})
                
                # Detect contradiction
                result = await contradiction_service.detect_contradiction(
                    claim1=claim1,
                    claim2=claim2,
                    metadata1=metadata1,
                    metadata2=metadata2,
                    threshold=request.threshold,
                    use_biomedlm=request.use_biomedlm,
                    use_tsmixer=request.use_tsmixer,
                    use_lorentz=request.use_lorentz,
                    use_temporal=request.use_temporal
                )
                
                # Add to contradictions if contradiction detected
                if result.get("is_contradiction", False):
                    contradiction = {
                        "claim1": {
                            "text": claim1,
                            "metadata": metadata1
                        },
                        "claim2": {
                            "text": claim2,
                            "metadata": metadata2
                        },
                        "contradiction_score": result.get("contradiction_score"),
                        "contradiction_type": result.get("contradiction_type"),
                        "confidence": result.get("confidence"),
                        "explanation": result.get("explanation"),
                        "classification": result.get("classification")
                    }
                    
                    contradictions.append(contradiction)

        # Create response
        response = {
            "total_claims": len(request.claims),
            "total_contradictions": len(contradictions),
            "contradictions": contradictions,
            "threshold": request.threshold
        }

        # Return result
        return APIResponse(
            status="success",
            message=f"Found {len(contradictions)} contradictions in {len(request.claims)} claims",
            data=response
        )
    except Exception as e:
        # Log error
        log_error(e, {
            "endpoint": "analyze_batch_contradictions",
            "user_id": current_user.id,
            "num_claims": len(request.claims)
        })

        # Return error response
        return ErrorResponse(
            status="error",
            message=f"Error analyzing contradictions: {str(e)}",
            error_code="CONTRADICTION_ANALYSIS_ERROR"
        )
