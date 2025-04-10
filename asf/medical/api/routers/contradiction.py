"""
Contradiction router for the Medical Research Synthesizer API.

This module provides endpoints for enhanced contradiction detection.
"""

import logging
import uuid
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from asf.medical.api.dependencies import (
    get_search_service, get_enhanced_contradiction_service,
    get_current_active_user
)
from asf.medical.ml.services.enhanced_contradiction_service import (
    EnhancedContradictionService, ContradictionType, ContradictionConfidence
)
from asf.medical.services.search_service import SearchService
from asf.medical.storage.models import User

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/contradiction", tags=["contradiction"])

# Define request/response models
class ContradictionRequest(BaseModel):
    """Enhanced contradiction detection request."""
    query: str = Field(..., description="Search query")
    max_results: int = Field(20, description="Maximum number of results to analyze")
    threshold: float = Field(0.7, description="Contradiction detection threshold")
    use_all_methods: bool = Field(True, description="Whether to use all available methods")

class ClaimContradictionRequest(BaseModel):
    """Claim contradiction detection request."""
    claim1: str = Field(..., description="First claim")
    claim2: str = Field(..., description="Second claim")
    metadata1: Optional[Dict[str, Any]] = Field(None, description="Metadata for first claim")
    metadata2: Optional[Dict[str, Any]] = Field(None, description="Metadata for second claim")
    use_all_methods: bool = Field(True, description="Whether to use all available methods")

class ContradictionResult(BaseModel):
    """Enhanced contradiction detection result."""
    query: str = Field(..., description="Search query")
    total_articles: int = Field(..., description="Total number of articles")
    contradictions_found: int = Field(..., description="Number of contradictions found")
    contradiction_types: Dict[str, int] = Field(..., description="Contradiction types count")
    contradictions: List[Dict[str, Any]] = Field(..., description="Detected contradictions")
    analysis_id: str = Field(..., description="Analysis ID")

class ClaimContradictionResult(BaseModel):
    """Claim contradiction detection result."""
    claim1: str = Field(..., description="First claim")
    claim2: str = Field(..., description="Second claim")
    is_contradiction: bool = Field(..., description="Whether a contradiction was detected")
    contradiction_score: float = Field(..., description="Contradiction score")
    contradiction_type: ContradictionType = Field(..., description="Type of contradiction")
    confidence: ContradictionConfidence = Field(..., description="Confidence level")
    explanation: str = Field(..., description="Explanation of the contradiction")
    methods_used: List[str] = Field(..., description="Methods used for detection")
    details: Dict[str, Any] = Field(..., description="Detailed results")

@router.post("/analyze", response_model=ContradictionResult)
async def analyze_contradictions(
    request: ContradictionRequest,
    search_service: SearchService = Depends(get_search_service),
    contradiction_service: EnhancedContradictionService = Depends(get_enhanced_contradiction_service),
    current_user: User = Depends(get_current_active_user)
):
    """
    Analyze contradictions in literature matching the query.
    
    This endpoint searches for articles and analyzes contradictions between them,
    using enhanced contradiction detection methods.
    """
    try:
        logger.info(f"Analyzing contradictions for query: {request.query}")
        
        # Search for articles
        search_result = await search_service.search(
            query=request.query,
            max_results=request.max_results,
            user_id=current_user.id
        )
        
        articles = search_result.get("results", [])
        
        if not articles:
            logger.warning(f"No articles found for query: {request.query}")
            return {
                "query": request.query,
                "total_articles": 0,
                "contradictions_found": 0,
                "contradiction_types": {},
                "contradictions": [],
                "analysis_id": str(uuid.uuid4())
            }
        
        # Detect contradictions
        contradictions = await contradiction_service.detect_contradictions_in_articles(
            articles=articles,
            threshold=request.threshold,
            use_all_methods=request.use_all_methods
        )
        
        # Count contradiction types
        contradiction_types = {}
        for contradiction in contradictions:
            contradiction_type = contradiction["contradiction_type"]
            if contradiction_type in contradiction_types:
                contradiction_types[contradiction_type] += 1
            else:
                contradiction_types[contradiction_type] = 1
        
        # Generate analysis ID
        analysis_id = str(uuid.uuid4())
        
        logger.info(f"Contradiction analysis completed: {len(contradictions)} contradictions found")
        
        return {
            "query": request.query,
            "total_articles": len(articles),
            "contradictions_found": len(contradictions),
            "contradiction_types": contradiction_types,
            "contradictions": contradictions,
            "analysis_id": analysis_id
        }
    except ValueError as e:
        logger.warning(f"Validation error in contradiction analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error analyzing contradictions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze contradictions: {str(e)}"
        )

@router.post("/detect", response_model=ClaimContradictionResult)
async def detect_contradiction(
    request: ClaimContradictionRequest,
    contradiction_service: EnhancedContradictionService = Depends(get_enhanced_contradiction_service),
    current_user: User = Depends(get_current_active_user)
):
    """
    Detect contradiction between two claims.
    
    This endpoint detects contradiction between two claims using enhanced
    contradiction detection methods.
    """
    try:
        logger.info(f"Detecting contradiction between claims")
        
        # Detect contradiction
        result = await contradiction_service.detect_contradiction(
            claim1=request.claim1,
            claim2=request.claim2,
            metadata1=request.metadata1,
            metadata2=request.metadata2,
            use_all_methods=request.use_all_methods
        )
        
        logger.info(f"Contradiction detection completed: {result['is_contradiction']}")
        
        return result
    except ValueError as e:
        logger.warning(f"Validation error in contradiction detection: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error detecting contradiction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to detect contradiction: {str(e)}"
        )
