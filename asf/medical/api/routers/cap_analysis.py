"""CAP analysis router for the Medical Research Synthesizer API.

This module provides endpoints for performing Clinical Assessment Protocol (CAP) analysis.
"""

import logging
from fastapi import Depends, HTTPException, status
from typing import Dict, Any

from asf.medical.api.routers.analysis_base import router, async_timed, log_error
from ..models.base import APIResponse, ErrorResponse
from ..models.analysis import CAPAnalysisResponse
from ..dependencies import get_analysis_service
from ..auth import get_current_active_user
from ...services.analysis_service import AnalysisService
from ...storage.models import User

logger = logging.getLogger(__name__)

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
