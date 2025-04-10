"""
Enhanced contradiction detection API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Dict, List, Optional, Any
from pydantic import BaseModel

from asf.medical.ml.services.enhanced_contradiction_service import EnhancedContradictionService
from asf.medical.api.dependencies import get_current_user

# Define models
class EnhancedContradictionRequest(BaseModel):
    """Request model for enhanced contradiction detection."""
    claim1: str
    claim2: str
    metadata1: Optional[Dict[str, Any]] = None
    metadata2: Optional[Dict[str, Any]] = None
    use_biomedlm: bool = True
    use_tsmixer: bool = False
    use_lorentz: bool = False
    threshold: float = 0.7

class EnhancedContradictionResponse(BaseModel):
    """Response model for enhanced contradiction detection."""
    is_contradiction: bool
    contradiction_score: Optional[float] = None
    contradiction_type: Optional[str] = None
    confidence: Optional[str] = None
    explanation: Optional[str] = None
    classification: Optional[Dict[str, Any]] = None

# Create router
router = APIRouter(
    prefix="/api/v1/enhanced-contradiction",
    tags=["enhanced-contradiction"],
    responses={404: {"description": "Not found"}}
)

# Initialize services
contradiction_service = EnhancedContradictionService()

@router.post("/detect", response_model=EnhancedContradictionResponse)
async def detect_enhanced_contradiction(
    request: EnhancedContradictionRequest,
    current_user = Depends(get_current_user)
):
    """
    Detect and classify contradiction between two claims with enhanced classification.
    
    This endpoint detects contradictions between two claims and provides enhanced
    classification including clinical significance, evidence quality, temporal factors,
    population differences, and methodological differences.
    
    Args:
        request: Enhanced contradiction request
        
    Returns:
        Enhanced contradiction detection result
    """
    try:
        # Detect contradiction
        result = await contradiction_service.detect_contradiction(
            claim1=request.claim1,
            claim2=request.claim2,
            metadata1=request.metadata1,
            metadata2=request.metadata2,
            use_biomedlm=request.use_biomedlm,
            use_tsmixer=request.use_tsmixer,
            use_lorentz=request.use_lorentz,
            threshold=request.threshold
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting contradiction: {str(e)}")

@router.post("/analyze-batch")
async def analyze_batch_contradictions(
    claims: List[Dict[str, str]],
    threshold: float = Query(0.7, description="Contradiction detection threshold"),
    current_user = Depends(get_current_user)
):
    """
    Analyze contradictions in a batch of claims with enhanced classification.
    
    This endpoint analyzes contradictions between multiple claims and provides enhanced
    classification for each contradiction.
    
    Args:
        claims: List of claims to analyze
        threshold: Contradiction detection threshold
        
    Returns:
        List of contradictions with enhanced classification
    """
    try:
        # Validate claims
        if len(claims) < 2:
            raise HTTPException(status_code=400, detail="At least 2 claims are required")
        
        # Analyze contradictions
        contradictions = []
        for i in range(len(claims)):
            for j in range(i + 1, len(claims)):
                # Extract claims
                claim1 = claims[i].get("text", "")
                claim2 = claims[j].get("text", "")
                
                # Extract metadata
                metadata1 = claims[i].get("metadata", {})
                metadata2 = claims[j].get("metadata", {})
                
                # Detect contradiction
                result = await contradiction_service.detect_contradiction(
                    claim1=claim1,
                    claim2=claim2,
                    metadata1=metadata1,
                    metadata2=metadata2,
                    threshold=threshold
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
        
        return {
            "total_claims": len(claims),
            "total_contradictions": len(contradictions),
            "contradictions": contradictions
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing contradictions: {str(e)}")
