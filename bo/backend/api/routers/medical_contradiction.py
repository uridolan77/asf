"""
Medical Contradiction Analysis API router for BO backend.
"""
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body, status
from pydantic import BaseModel, Field

from api.services.medical_contradiction_service import MedicalContradictionService, get_medical_contradiction_service
from models.user import User
from api.dependencies import get_current_user

router = APIRouter(
    prefix="/api/medical/analysis",
    tags=["medical-analysis"],
    responses={404: {"description": "Not found"}},
)

class ContradictionAnalysisRequest(BaseModel):
    """Contradiction analysis request model."""
    query: str = Field(..., description="Search query to find articles for contradiction analysis")
    max_results: int = Field(20, description="Maximum number of contradiction pairs to return")
    threshold: float = Field(0.7, description="Minimum contradiction score threshold (0.0-1.0)")
    use_biomedlm: bool = Field(True, description="Whether to use the BioMedLM model")
    use_tsmixer: bool = Field(False, description="Whether to use the TSMixer model")
    use_lorentz: bool = Field(False, description="Whether to use the Lorentz model")

@router.get("/contradiction/models")
async def get_contradiction_models(
    contradiction_service: MedicalContradictionService = Depends(get_medical_contradiction_service)
):
    """
    Get available contradiction detection models.
    """
    return contradiction_service.get_available_models()

@router.post("/contradictions")
async def analyze_contradictions(
    request: ContradictionAnalysisRequest,
    contradiction_service: MedicalContradictionService = Depends(get_medical_contradiction_service),
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    Analyze contradictions in medical literature for a given query.
    
    This endpoint identifies pairs of medical research articles that present contradictory
    findings on the same topic. It extracts claims from each article and calculates a
    contradiction score based on semantic analysis.
    """
    user_id = current_user.id if current_user else None
    result = await contradiction_service.analyze_contradictions(
        query=request.query,
        max_results=request.max_results,
        threshold=request.threshold,
        use_biomedlm=request.use_biomedlm,
        use_tsmixer=request.use_tsmixer,
        use_lorentz=request.use_lorentz,
        user_id=user_id
    )
    
    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result["message"]
        )
        
    return result