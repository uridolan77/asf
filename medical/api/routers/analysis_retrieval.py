"""Analysis retrieval router for the Medical Research Synthesizer API.

This module provides endpoints for retrieving previously performed analyses.
"""

import logging
from fastapi import Depends, HTTPException, status
from typing import Dict, Any

from asf.medical.api.routers.analysis_base import router, async_timed, log_error
from ..models.base import APIResponse, ErrorResponse
from ..dependencies import get_analysis_service
from ..auth import get_current_active_user
from ...services.analysis_service import AnalysisService
from ...storage.models import User

logger = logging.getLogger(__name__)

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
