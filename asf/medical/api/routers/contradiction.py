"""API endpoints for contradiction detection.

This module provides API endpoints for detecting contradictions in medical literature.
"""

import logging
from fastapi import APIRouter, Depends, Request, Response
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

from ...ml.services.contradiction_service import ContradictionService
from ..dependencies import get_contradiction_service, get_current_active_user
from ..models.base import APIResponse
from ...storage.models import User
from ...core.observability import async_timed

router = APIRouter(prefix="/contradiction", tags=["contradiction"])

logger = logging.getLogger(__name__)

class ClaimMetadata(BaseModel):
    """Metadata for a claim."""
    source: str = Field(..., description="Source of the claim")
    date: Optional[str] = Field(None, description="Date of the claim")
    confidence: Optional[float] = Field(None, description="Confidence score of the claim")
    authors: Optional[List[str]] = Field(None, description="Authors of the claim")

class ContradictionRequest(BaseModel):
    """Request model for contradiction detection."""
    claim1: str = Field(..., description="First claim to compare")
    claim2: str = Field(..., description="Second claim to compare")
    metadata1: Optional[ClaimMetadata] = Field(None, description="Metadata for the first claim")
    metadata2: Optional[ClaimMetadata] = Field(None, description="Metadata for the second claim")
    use_biomedlm: bool = Field(True, description="Whether to use BioMedLM model")
    use_tsmixer: bool = Field(False, description="Whether to use TSMixer model")
    use_lorentz: bool = Field(False, description="Whether to use Lorentz model")
    use_temporal: bool = Field(True, description="Whether to consider temporal aspects")

@router.post("/detect", response_model=APIResponse[Dict[str, Any]])
@async_timed("detect_contradiction_endpoint")
async def detect_contradiction(
    request: ContradictionRequest,
    req: Request = None,
    res: Response = None,
    current_user: User = Depends(get_current_active_user),
    contradiction_service: ContradictionService = Depends(get_contradiction_service),
):
    """Detect contradictions between two claims.

    Args:
        request: The contradiction request containing the claims and options
        req: FastAPI request object
        res: FastAPI response object
        current_user: The current authenticated user
        contradiction_service: The contradiction service

    Returns:
        APIResponse containing the contradiction analysis results
    """
    # Extract claims and metadata from request
    claim1 = request.claim1
    claim2 = request.claim2
    metadata1 = request.metadata1
    metadata2 = request.metadata2

    # Call the contradiction service
    result = await contradiction_service.detect_contradiction(
        claim1=claim1,
        claim2=claim2,
        metadata1=metadata1,
        metadata2=metadata2,
        use_biomedlm=request.use_biomedlm,
        use_tsmixer=request.use_tsmixer,
        use_lorentz=request.use_lorentz,
        use_temporal=request.use_temporal
    )

    # Return the result
    return APIResponse(data=result)
