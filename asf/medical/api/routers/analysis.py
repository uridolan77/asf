"""
Analysis router for the Medical Research Synthesizer API.

This module provides endpoints for analyzing medical literature,
including contradiction detection and specialized analyses.
"""

import uuid
import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request, Response

from asf.medical.api.models.base import APIResponse, ErrorResponse
from asf.medical.api.models.analysis import (
    ContradictionAnalysisRequest, 
    CAPAnalysisResponse
)
from asf.medical.api.dependencies import get_analysis_service
from asf.medical.api.auth import get_current_active_user
from asf.medical.services.analysis_service import AnalysisService
from asf.medical.storage.models import User
from asf.medical.core.monitoring import async_timed, log_error

# Initialize router
router = APIRouter(prefix="/analysis", tags=["analysis"])

# Set up logging
logger = logging.getLogger(__name__)

@router.post("/contradictions", response_model=APIResponse[Dict[str, Any]])
@async_timed("analyze_contradictions_endpoint")
async def analyze_contradictions(
    request: ContradictionAnalysisRequest,
    analysis_service: AnalysisService = Depends(get_analysis_service),
    current_user: User = Depends(get_current_active_user),
    req: Request = None,
    res: Response = None
):
    """
    Analyze contradictions in medical literature.
    
    This endpoint searches for articles and analyzes contradictions between them,
    using various contradiction detection methods.
    """
    try:
        # Add request ID to response headers for tracing
        request_id = req.headers.get("X-Request-ID") if req else None
        if request_id and res:
            res.headers["X-Request-ID"] = request_id
        
        # Log the request
        logger.info(f"Contradiction analysis request: query='{request.query}', max_results={request.max_results}, user_id={current_user.id}")
        
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
        
        return APIResponse(
            success=True,
            message="Contradiction analysis completed successfully",
            data=result,
            meta={
                "query": request.query,
                "max_results": request.max_results,
                "threshold": request.threshold,
                "use_biomedlm": request.use_biomedlm,
                "use_tsmixer": request.use_tsmixer,
                "use_lorentz": request.use_lorentz
            }
        )
    except ValueError as e:
        # Handle validation errors
        log_error(e, {"query": request.query, "user_id": current_user.id})
        logger.warning(f"Validation error in contradiction analysis: {str(e)}")
        return ErrorResponse(
            message="Invalid contradiction analysis parameters",
            errors=[{"detail": str(e)}],
            code="VALIDATION_ERROR"
        )
    except Exception as e:
        # Handle unexpected errors
        log_error(e, {"query": request.query, "user_id": current_user.id})
        logger.error(f"Error analyzing contradictions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing contradictions: {str(e)}"
        )

@router.get("/cap", response_model=APIResponse[Dict[str, Any]])
@async_timed("analyze_cap_endpoint")
async def analyze_cap(
    analysis_service: AnalysisService = Depends(get_analysis_service),
    current_user: User = Depends(get_current_active_user)
):
    """
    Analyze Community-Acquired Pneumonia (CAP) literature.
    
    This endpoint performs a specialized analysis of the CAP literature,
    focusing on treatment approaches, patient populations, and outcomes.
    """
    try:
        # Log the request
        logger.info(f"CAP analysis request: user_id={current_user.id}")
        
        # Execute the analysis
        result = await analysis_service.analyze_cap(user_id=current_user.id)
        
        # Log the result
        logger.info(f"CAP analysis completed")
        
        return APIResponse(
            success=True,
            message="CAP analysis completed successfully",
            data=result,
            meta={
                "user_id": current_user.id
            }
        )
    except Exception as e:
        # Handle unexpected errors
        log_error(e, {"user_id": current_user.id})
        logger.error(f"Error executing CAP analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing CAP analysis: {str(e)}"
        )

@router.get("/{analysis_id}", response_model=APIResponse[Dict[str, Any]])
@async_timed("get_analysis_endpoint")
async def get_analysis(
    analysis_id: str,
    analysis_service: AnalysisService = Depends(get_analysis_service),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get a stored analysis by ID.
    
    This endpoint retrieves a previously stored analysis by its ID.
    """
    try:
        # Log the request
        logger.info(f"Get analysis request: analysis_id={analysis_id}, user_id={current_user.id}")
        
        # Get the analysis
        result = await analysis_service.get_analysis(analysis_id)
        
        if not result:
            logger.warning(f"Analysis not found: analysis_id={analysis_id}")
            return ErrorResponse(
                message="Analysis not found",
                errors=[{"detail": f"No analysis found with ID {analysis_id}"}],
                code="NOT_FOUND"
            )
        
        # Log the result
        logger.info(f"Analysis retrieved: analysis_id={analysis_id}")
        
        return APIResponse(
            success=True,
            message="Analysis retrieved successfully",
            data=result,
            meta={
                "analysis_id": analysis_id,
                "user_id": current_user.id
            }
        )
    except Exception as e:
        # Handle unexpected errors
        log_error(e, {"analysis_id": analysis_id, "user_id": current_user.id})
        logger.error(f"Error retrieving analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving analysis: {str(e)}"
        )
