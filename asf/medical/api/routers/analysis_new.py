"""
Analysis router for the Medical Research Synthesizer API.

This module provides endpoints for analyzing medical literature.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request, Response

from asf.medical.api.models.analysis import (
    ContradictionAnalysisRequest, 
    ContradictionAnalysisResponse,
    CAPAnalysisResponse
)
from asf.medical.api.dependencies import get_analysis_service
from asf.medical.api.auth_service import get_current_active_user
from asf.medical.services.analysis_service import AnalysisService
from asf.medical.storage.models import User

# Initialize router
router = APIRouter(prefix="/v1/analysis", tags=["Analysis"])

# Set up logging
logger = logging.getLogger(__name__)

@router.post("/contradictions", response_model=ContradictionAnalysisResponse)
async def analyze_contradictions(
    request: ContradictionAnalysisRequest,
    analysis_service: AnalysisService = Depends(get_analysis_service),
    current_user: User = Depends(get_current_active_user),
    req: Request = None,
    res: Response = None
):
    """
    Analyze contradictions in medical literature.

    This endpoint searches for articles matching the query and analyzes them
    for contradictions using various methods (BioMedLM, TSMixer, Lorentz).
    """
    try:
        # Add request ID to response headers for tracing
        request_id = req.headers.get("X-Request-ID") if req else None
        if request_id and res:
            res.headers["X-Request-ID"] = request_id
        
        # Log the request
        logger.info(
            f"Contradiction analysis request: query='{request.query}', "
            f"max_results={request.max_results}, threshold={request.threshold}, "
            f"use_biomedlm={request.use_biomedlm}, use_tsmixer={request.use_tsmixer}, "
            f"use_lorentz={request.use_lorentz}, user_id={current_user.id}"
        )
        
        # Execute the analysis
        result = await analysis_service.analyze_contradictions(
            query=request.query,
            max_results=request.max_results,
            threshold=request.threshold,
            use_biomedlm=request.use_biomedlm,
            use_tsmixer=request.use_tsmixer,
            use_lorentz=request.use_lorentz,
            user_id=current_user.id
        )
        
        # Log the result
        logger.info(f"Contradiction analysis completed: {len(result.get('contradictions', []))} contradictions found")
        
        return result
    except ValueError as e:
        # Handle validation errors
        logger.warning(f"Validation error in contradiction analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Error executing contradiction analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing contradiction analysis: {str(e)}"
        )

@router.get("/cap", response_model=CAPAnalysisResponse)
async def analyze_cap(
    analysis_service: AnalysisService = Depends(get_analysis_service),
    current_user: User = Depends(get_current_active_user),
    req: Request = None,
    res: Response = None
):
    """
    Analyze Community-Acquired Pneumonia (CAP) literature.

    This endpoint searches for CAP-related articles and analyzes them
    for treatment types, patient populations, and outcomes.
    """
    try:
        # Add request ID to response headers for tracing
        request_id = req.headers.get("X-Request-ID") if req else None
        if request_id and res:
            res.headers["X-Request-ID"] = request_id
        
        # Log the request
        logger.info(f"CAP analysis request: user_id={current_user.id}")
        
        # Execute the analysis
        result = await analysis_service.analyze_cap(user_id=current_user.id)
        
        # Log the result
        logger.info(f"CAP analysis completed")
        
        return result
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Error executing CAP analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing CAP analysis: {str(e)}"
        )

@router.get("/{analysis_id}")
async def get_analysis(
    analysis_id: str,
    analysis_service: AnalysisService = Depends(get_analysis_service),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get a stored analysis by ID.
    
    This endpoint retrieves a previously executed analysis by its ID.
    """
    try:
        result = await analysis_service.get_analysis(analysis_id)
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Analysis with ID {analysis_id} not found"
            )
        
        # Check if the user has access to this analysis
        if result.get('user_id') and result.get('user_id') != current_user.id and current_user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to access this analysis"
            )
        
        return result.get('analysis')
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving analysis: {str(e)}"
        )
