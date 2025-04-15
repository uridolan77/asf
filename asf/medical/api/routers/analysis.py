"""Analysis router for the Medical Research Synthesizer API.

This module provides endpoints for analyzing medical literature,
including contradiction detection and specialized analyses.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, status, Request, Response
from typing import Dict, Any

from ..models.base import APIResponse, ErrorResponse
from ..models.analysis import (
    ContradictionAnalysisRequest,
    CAPAnalysisResponse
)
from ..dependencies import get_analysis_service
from ..auth import get_current_active_user
from ...services.analysis_service import AnalysisService
from ...storage.models import User
from ...core.observability import async_timed, log_error

router = APIRouter(prefix="/analysis", tags=["analysis"])

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
    """Analyze contradictions in medical literature based on a query.

    This endpoint performs contradiction analysis on medical literature matching
    the provided query. It identifies statements that contradict each other and
    provides explanations for the contradictions.

    Args:
        request: The contradiction analysis request containing query and parameters
        analysis_service: The analysis service for performing the analysis
        current_user: The authenticated user making the request
        req: The FastAPI request object
        res: The FastAPI response object

    Returns:
        APIResponse containing the contradiction analysis results

    Raises:
        HTTPException: If an error occurs during analysis
    """
    try:
        request_id = req.headers.get("X-Request-ID") if req else None
        if request_id and res:
            res.headers["X-Request-ID"] = request_id

        logger.info(f"Contradiction analysis request: query='{request.query}', max_results={request.max_results}, user_id={current_user.id}")

        result = await analysis_service.analyze_contradictions(
            query=request.query,
            max_results=request.max_results,
            threshold=request.threshold,
            use_biomedlm=request.use_biomedlm,
            use_tsmixer=request.use_tsmixer,
            use_lorentz=request.use_lorentz,
            user_id=current_user.id
        )

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
        logger.error(f"Error: {str(e)}")
        log_error(e, {"query": request.query, "user_id": current_user.id})
        logger.warning(f"Validation error in contradiction analysis: {str(e)}")
        return ErrorResponse(
            message="Invalid contradiction analysis parameters",
            errors=[{"detail": str(e)}],
            code="VALIDATION_ERROR"
        )
    except Exception as e:
        logger.error(f"Error: {str(e)}")
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
    """Perform CAP (Clinical Assessment Protocol) analysis.

    This endpoint performs a Clinical Assessment Protocol analysis on the user's
    data. CAP analysis helps identify clinical patterns and provides
    recommendations for treatment planning.

    Args:
        analysis_service: The analysis service for performing the analysis
        current_user: The authenticated user making the request

    Returns:
        APIResponse containing the CAP analysis results

    Raises:
        HTTPException: If an error occurs during analysis
    """
    try:
        logger.info(f"CAP analysis request: user_id={current_user.id}")

        result = await analysis_service.analyze_cap(user_id=current_user.id)

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
        logger.error(f"Error: {str(e)}")
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
    """Retrieve a previously performed analysis by ID.

    This endpoint retrieves the results of a previously performed analysis
    using its unique identifier.

    Args:
        analysis_id: The unique identifier of the analysis to retrieve
        analysis_service: The analysis service for retrieving the analysis
        current_user: The authenticated user making the request

    Returns:
        APIResponse containing the analysis results

    Raises:
        HTTPException: If the analysis is not found or an error occurs
    """
    try:
        logger.info(f"Get analysis request: analysis_id={analysis_id}, user_id={current_user.id}")

        result = await analysis_service.get_analysis(analysis_id)

        if not result:
            logger.warning(f"Analysis not found: analysis_id={analysis_id}")
            return ErrorResponse(
                message="Analysis not found",
                errors=[{"detail": f"No analysis found with ID {analysis_id}"}],
                code="NOT_FOUND"
            )

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
        logger.error(f"Error: {str(e)}")
        log_error(e, {"analysis_id": analysis_id, "user_id": current_user.id})
        logger.error(f"Error retrieving analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving analysis: {str(e)}"
        )
