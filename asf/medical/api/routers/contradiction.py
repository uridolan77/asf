API endpoints for contradiction detection.

This module provides API endpoints for detecting contradictions in medical literature.

import logging
from fastapi import APIRouter, Depends, Request, Response
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

from asf.medical.ml.services.contradiction_service import ContradictionService
from asf.medical.api.dependencies import get_current_active_user
from asf.medical.api.models.base import APIResponse
from asf.medical.storage.models import User
from asf.medical.core.monitoring import async_timed

router = APIRouter(prefix="/contradiction", tags=["contradiction"])

logger = logging.getLogger(__name__)

class ClaimMetadata(BaseModel):
    Metadata for a claim.
    
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
