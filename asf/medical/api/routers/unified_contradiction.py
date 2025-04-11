"""
Unified contradiction detection API endpoints.

This module provides a unified API for contradiction detection, replacing
the separate contradiction and enhanced_contradiction routers.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

from asf.medical.ml.services.unified_contradiction_service import UnifiedUnifiedUnifiedContradictionService
from asf.medical.api.dependencies import get_current_user, get_current_active_user
from asf.medical.api.models.base import APIResponse, ErrorResponse
from asf.medical.storage.models import User
from asf.medical.core.monitoring import async_timed, log_error

router = APIRouter(prefix="/contradiction", tags=["contradiction"])

logger = logging.getLogger(__name__)

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

def get_contradiction_service() -> UnifiedUnifiedUnifiedContradictionService:
    """Get the unified contradiction service.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
    return UnifiedUnifiedUnifiedContradictionService()

@router.post("/detect", response_model=APIResponse[ContradictionResponse])
@async_timed("detect_contradiction_endpoint")
async def detect_contradiction(
    request: ContradictionRequest,
    req: Request = None,
    res: Response = None,
    current_user: User = Depends(get_current_active_user),
    contradiction_service: UnifiedUnifiedUnifiedContradictionService = Depends(get_contradiction_service)
):