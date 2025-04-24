"""
Enhanced Medical Contradiction Analysis API router for BO backend.
"""
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body, status
from pydantic import BaseModel, Field

from api.services.enhanced_contradiction_service import EnhancedContradictionService, get_enhanced_contradiction_service
from models.user import User
from api.dependencies import get_current_user

router = APIRouter(
    prefix="/api/medical/analysis/enhanced",
    tags=["medical-enhanced-analysis"],
    responses={404: {"description": "Not found"}},
)

class EnhancedContradictionAnalysisRequest(BaseModel):
    """Enhanced contradiction analysis request model."""
    query: str = Field(..., description="Search query to find articles for contradiction analysis")
    max_results: int = Field(20, description="Maximum number of contradiction pairs to return")
    threshold: float = Field(0.7, description="Minimum contradiction score threshold (0.0-1.0)")
    
    # Model selection
    use_biomedlm: bool = Field(True, description="Whether to use the BioMedLM model")
    use_tsmixer: bool = Field(False, description="Whether to use the TSMixer model")
    use_lorentz: bool = Field(False, description="Whether to use the Lorentz model")
    
    # Enhancement options
    include_clinical_trials: bool = Field(True, description="Whether to include clinical trials data")
    standardize_terminology: bool = Field(True, description="Whether to standardize terminology using SNOMED CT")

@router.get("/contradiction/models")
async def get_enhanced_contradiction_models(
    contradiction_service: EnhancedContradictionService = Depends(get_enhanced_contradiction_service)
):
    """
    Get available enhanced contradiction detection models and features.
    """
    return contradiction_service.get_available_models()

@router.post("/contradictions")
async def analyze_enhanced_contradictions(
    request: EnhancedContradictionAnalysisRequest,
    contradiction_service: EnhancedContradictionService = Depends(get_enhanced_contradiction_service),
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    Analyze contradictions in medical literature with enhanced features.
    
    This endpoint identifies pairs of medical research articles that present contradictory
    findings, enhanced with:
    
    1. Clinical trials data integration - finds supporting evidence from clinical trials
    2. Terminology standardization - uses SNOMED CT to standardize medical terms
    3. Semantic context - provides hierarchical and semantic context for contradictions
    
    The analysis includes normalized terminology, supporting trials, and contradiction types.
    """
    user_id = current_user.id if current_user else None
    result = await contradiction_service.analyze_contradictions(
        query=request.query,
        max_results=request.max_results,
        threshold=request.threshold,
        use_biomedlm=request.use_biomedlm,
        use_tsmixer=request.use_tsmixer,
        use_lorentz=request.use_lorentz,
        include_clinical_trials=request.include_clinical_trials,
        standardize_terminology=request.standardize_terminology,
        user_id=user_id
    )
    
    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result["message"]
        )
        
    return result