"""
Contradiction resolution API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Dict, List, Optional, Any
from pydantic import BaseModel

from asf.medical.ml.services.enhanced_contradiction_service import EnhancedContradictionService
from asf.medical.ml.services.resolution.contradiction_resolution_service import MedicalContradictionResolutionService
from asf.medical.ml.services.resolution.resolution_models import ResolutionStrategy
from asf.medical.api.dependencies import get_current_user

# Define models
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

# Create router
router = APIRouter(
    prefix="/api/v1/contradiction-resolution",
    tags=["contradiction-resolution"],
    responses={404: {"description": "Not found"}}
)

# Initialize services
contradiction_service = EnhancedContradictionService()
resolution_service = MedicalContradictionResolutionService()

@router.post("/resolve", response_model=ContradictionResolutionResponse)
async def resolve_contradiction(
    request: ContradictionResolutionRequest,
    current_user = Depends(get_current_user)
):
    """
    Resolve a contradiction between two claims.
    
    This endpoint detects contradictions between two claims and provides a resolution
    based on evidence-based medicine principles.
    
    Args:
        request: Contradiction resolution request
        
    Returns:
        Contradiction resolution result
    """
    try:
        # Detect contradiction
        contradiction = await contradiction_service.detect_contradiction(
            claim1=request.claim1,
            claim2=request.claim2,
            metadata1=request.metadata1,
            metadata2=request.metadata2
        )
        
        # Check if contradiction was detected
        if not contradiction.get("is_contradiction", False):
            raise HTTPException(
                status_code=400,
                detail="No contradiction was detected between the provided claims."
            )
        
        # Resolve contradiction
        if request.use_combined_evidence:
            resolution = await resolution_service.resolve_contradiction_with_combined_evidence(contradiction)
        else:
            strategy = None
            if request.strategy:
                try:
                    strategy = ResolutionStrategy(request.strategy)
                except ValueError:
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
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resolving contradiction: {str(e)}")

@router.post("/batch-resolve")
async def batch_resolve_contradictions(
    contradictions: List[Dict[str, Any]],
    strategy: Optional[str] = Query(None, description="Resolution strategy to use"),
    use_combined_evidence: bool = Query(False, description="Whether to use combined evidence approach"),
    current_user = Depends(get_current_user)
):
    """
    Resolve a batch of contradictions.
    
    This endpoint resolves multiple contradictions using the specified strategy.
    
    Args:
        contradictions: List of contradictions to resolve
        strategy: Resolution strategy to use
        use_combined_evidence: Whether to use combined evidence approach
        
    Returns:
        List of contradiction resolution results
    """
    try:
        # Validate contradictions
        if not contradictions:
            raise HTTPException(status_code=400, detail="No contradictions provided")
        
        # Convert strategy string to enum if provided
        strategy_enum = None
        if strategy:
            try:
                strategy_enum = ResolutionStrategy(strategy)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid resolution strategy: {strategy}. Valid strategies are: {', '.join([s.value for s in ResolutionStrategy])}"
                )
        
        # Resolve contradictions
        resolutions = []
        for contradiction in contradictions:
            # Validate contradiction
            if not contradiction.get("is_contradiction", False):
                continue
            
            # Resolve contradiction
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
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resolving contradictions: {str(e)}")

@router.get("/history")
async def get_resolution_history(
    current_user = Depends(get_current_user)
):
    """
    Get the contradiction resolution history.
    
    This endpoint returns the history of contradiction resolutions.
    
    Returns:
        List of contradiction resolution history entries
    """
    try:
        history = resolution_service.get_resolution_history()
        return {
            "total_entries": len(history),
            "history": history
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting resolution history: {str(e)}")

@router.post("/feedback/{contradiction_id}")
async def add_resolution_feedback(
    contradiction_id: str,
    feedback: Dict[str, Any],
    current_user = Depends(get_current_user)
):
    """
    Add feedback to a contradiction resolution.
    
    This endpoint adds feedback to a contradiction resolution in the history.
    
    Args:
        contradiction_id: ID of the contradiction
        feedback: Feedback data
        
    Returns:
        Success message
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
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding feedback: {str(e)}")
