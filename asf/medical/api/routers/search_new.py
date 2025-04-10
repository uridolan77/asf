"""
Search router for the Medical Research Synthesizer API.

This module provides endpoints for searching medical literature.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request, Response

from asf.medical.api.models.search import QueryRequest, SearchResponse, PICORequest
from asf.medical.api.dependencies import get_search_service
from asf.medical.api.auth import get_current_active_user
from asf.medical.services.search_service import SearchService
from asf.medical.storage.models import User

# Initialize router
router = APIRouter(prefix="/v1/search", tags=["Search"])

# Set up logging
logger = logging.getLogger(__name__)

@router.post("", response_model=SearchResponse)
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
        
        return result
    except ValueError as e:
        # Handle validation errors
        logger.warning(f"Validation error in search: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Error executing search: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing search: {str(e)}"
        )

@router.post("/pico", response_model=SearchResponse)
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
        
        return result
    except ValueError as e:
        # Handle validation errors
        logger.warning(f"Validation error in PICO search: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Error executing PICO search: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing PICO search: {str(e)}"
        )

@router.get("/{result_id}", response_model=SearchResponse)
async def get_search_result(
    result_id: str,
    search_service: SearchService = Depends(get_search_service),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get a stored search result by ID.
    
    This endpoint retrieves a previously executed search result by its ID.
    """
    try:
        result = await search_service.get_result(result_id)
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Search result with ID {result_id} not found"
            )
        
        # Check if the user has access to this result
        if result.get('user_id') and result.get('user_id') != current_user.id and current_user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to access this search result"
            )
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving search result: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving search result: {str(e)}"
        )
