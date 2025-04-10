"""
Search router for the Medical Research Synthesizer API.

This module provides endpoints for searching medical literature.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request, Response

from asf.medical.api.models.search import QueryRequest, PICORequest
from asf.medical.api.models.base import APIResponse, ErrorResponse
from asf.medical.api.dependencies import get_search_service
from asf.medical.api.auth import get_current_active_user
from asf.medical.services.search_service import SearchService
from asf.medical.storage.models import User
from asf.medical.core.monitoring import async_timed, log_error

# Initialize router
router = APIRouter(prefix="/search", tags=["search"])

# Set up logging
logger = logging.getLogger(__name__)

@router.post("", response_model=APIResponse[Dict[str, Any]])
@async_timed("search_endpoint")
async def search(
    request: QueryRequest,
    search_service: SearchService = Depends(get_search_service),
    current_user: User = Depends(get_current_active_user),
    req: Request = None,
    res: Response = None
):
    """
    Search PubMed with the given query and return enriched results.

    This endpoint performs a search using the enhanced NCBIClient and enriches
    the results with metadata such as impact factors, authority scores,
    and standardized dates.
    """
    try:
        # Add request ID to response headers for tracing
        request_id = req.headers.get("X-Request-ID") if req else None
        if request_id and res:
            res.headers["X-Request-ID"] = request_id
        
        # Log the request
        logger.info(f"Search request: query='{request.query}', max_results={request.max_results}, user_id={current_user.id}")
        
        # Execute the search
        result = await search_service.search(
            query=request.query,
            max_results=request.max_results,
            user_id=current_user.id
        )
        
        # Log the result
        logger.info(f"Search completed: {len(result.get('results', []))} results found")
        
        return APIResponse(
            success=True,
            message="Search completed successfully",
            data=result,
            meta={
                "query": request.query,
                "max_results": request.max_results,
                "user_id": current_user.id
            }
        )
    except ValueError as e:
        # Handle validation errors
        log_error(e, {"query": request.query, "user_id": current_user.id})
        logger.warning(f"Validation error in search: {str(e)}")
        return ErrorResponse(
            message="Invalid search parameters",
            errors=[{"detail": str(e)}],
            code="VALIDATION_ERROR"
        )
    except Exception as e:
        # Handle unexpected errors
        log_error(e, {"query": request.query, "user_id": current_user.id})
        logger.error(f"Error executing search: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing search: {str(e)}"
        )

@router.post("/pico", response_model=APIResponse[Dict[str, Any]])
@async_timed("search_pico_endpoint")
async def search_pico(
    request: PICORequest,
    search_service: SearchService = Depends(get_search_service),
    current_user: User = Depends(get_current_active_user),
    req: Request = None,
    res: Response = None
):
    """
    Search PubMed using the PICO framework.

    This endpoint builds a structured query using the PICO framework
    (Population, Intervention, Comparison, Outcome) and returns enriched results.
    """
    try:
        # Add request ID to response headers for tracing
        request_id = req.headers.get("X-Request-ID") if req else None
        if request_id and res:
            res.headers["X-Request-ID"] = request_id
        
        # Log the request
        logger.info(f"PICO search request: condition='{request.condition}', interventions={request.interventions}, outcomes={request.outcomes}, user_id={current_user.id}")
        
        # Validate inputs
        if not request.condition:
            raise ValueError("Condition is required for PICO search")
        
        # Execute the search
        result = await search_service.search_pico(
            condition=request.condition,
            interventions=request.interventions,
            outcomes=request.outcomes,
            population=request.population,
            study_design=request.study_design,
            years=request.years,
            max_results=request.max_results,
            user_id=current_user.id
        )
        
        # Log the result
        logger.info(f"PICO search completed: {len(result.get('results', []))} results found")
        
        return APIResponse(
            success=True,
            message="PICO search completed successfully",
            data=result,
            meta={
                "condition": request.condition,
                "interventions": request.interventions,
                "outcomes": request.outcomes,
                "user_id": current_user.id
            }
        )
    except ValueError as e:
        # Handle validation errors
        log_error(e, {"condition": request.condition, "user_id": current_user.id})
        logger.warning(f"Validation error in PICO search: {str(e)}")
        return ErrorResponse(
            message="Invalid PICO search parameters",
            errors=[{"detail": str(e)}],
            code="VALIDATION_ERROR"
        )
    except Exception as e:
        # Handle unexpected errors
        log_error(e, {"condition": request.condition, "user_id": current_user.id})
        logger.error(f"Error executing PICO search: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing PICO search: {str(e)}"
        )

@router.get("/{result_id}", response_model=APIResponse[Dict[str, Any]])
@async_timed("get_search_result_endpoint")
async def get_search_result(
    result_id: str,
    search_service: SearchService = Depends(get_search_service),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get a stored search result by ID.

    This endpoint retrieves a previously stored search result by its ID.
    """
    try:
        # Log the request
        logger.info(f"Get search result request: result_id={result_id}, user_id={current_user.id}")
        
        # Get the result
        result = await search_service.get_result(result_id, current_user.id)
        
        if not result:
            logger.warning(f"Search result not found: result_id={result_id}, user_id={current_user.id}")
            return ErrorResponse(
                message="Search result not found",
                errors=[{"detail": f"No result found with ID {result_id}"}],
                code="NOT_FOUND"
            )
        
        # Log the result
        logger.info(f"Search result retrieved: result_id={result_id}, user_id={current_user.id}")
        
        return APIResponse(
            success=True,
            message="Search result retrieved successfully",
            data=result,
            meta={
                "result_id": result_id,
                "user_id": current_user.id
            }
        )
    except Exception as e:
        # Handle unexpected errors
        log_error(e, {"result_id": result_id, "user_id": current_user.id})
        logger.error(f"Error retrieving search result: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving search result: {str(e)}"
        )
