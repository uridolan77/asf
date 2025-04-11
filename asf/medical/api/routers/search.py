"""
Search router for the Medical Research Synthesizer API.

This module provides endpoints for searching medical literature.
"""

import logging
import traceback
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Request, Response

from asf.medical.api.models.search import QueryRequest, PICORequest
from asf.medical.api.models.base import APIResponse, ErrorResponse
from asf.medical.api.dependencies import get_search_service
from asf.medical.api.auth import get_current_active_user
from asf.medical.services.search_service import SearchService
from asf.medical.storage.models import User
from asf.medical.core.monitoring import async_timed, log_error
from asf.medical.core.exceptions import SearchError, ValidationError

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
    Search for medical literature with the given query and return enriched results.

    This endpoint performs a search using the specified search method (PubMed, ClinicalTrials.gov, or GraphRAG)
    and enriches the results with metadata such as impact factors, authority scores,
    and standardized dates.

    The search method can be specified using the search_method parameter:
    - pubmed: Search PubMed using the NCBI API
    - clinical_trials: Search ClinicalTrials.gov
    - graph_rag: Search using GraphRAG (graph-based retrieval-augmented generation)

    GraphRAG parameters:
    - use_graph_rag: Whether to use GraphRAG for search (overrides search_method if True)
    - use_vector_search: Whether to use vector search with GraphRAG
    - use_graph_search: Whether to use graph search with GraphRAG
    """
    try:
        # Add request ID to response headers for tracing
        request_id = req.headers.get("X-Request-ID") if req else None
        if request_id and res:
            res.headers["X-Request-ID"] = request_id

        # Validate the request
        if not request.query or not request.query.strip():
            logger.warning("Empty search query")
            raise ValidationError("Search query cannot be empty")

        # Check if GraphRAG is available if requested
        if request.use_graph_rag and not search_service.is_graph_rag_available():
            logger.warning("GraphRAG requested but not available")
            if res:
                res.headers["X-GraphRAG-Available"] = "false"
        elif request.use_graph_rag and res:
            res.headers["X-GraphRAG-Available"] = "true"

        # Log the request
        logger.info(
            f"Search request: query='{request.query}', max_results={request.max_results}, "
            f"method={request.search_method}, use_graph_rag={request.use_graph_rag}, "
            f"user_id={current_user.id}"
        )

        # Execute the search
        result = await search_service.search(
            query=request.query,
            max_results=request.max_results,
            page=request.pagination.page,
            page_size=request.pagination.page_size,
            user_id=current_user.id,
            search_method=request.search_method,
            use_graph_rag=request.use_graph_rag,
            use_vector_search=request.use_vector_search,
            use_graph_search=request.use_graph_search
        )

        # Add GraphRAG info to response headers
        if res and 'graph_rag_results' in result:
            res.headers["X-GraphRAG-Used"] = "true"
        elif res and 'fallback_reason' in result and 'GraphRAG' in result.get('fallback_reason', ''):
            res.headers["X-GraphRAG-Used"] = "false"
            res.headers["X-GraphRAG-Fallback-Reason"] = result.get('fallback_reason', '')

        # Log the result
        logger.info(f"Search completed: {len(result.get('results', []))} results found")

        return APIResponse(
            success=True,
            message="Search completed successfully",
            data=result,
            meta={
                "query": request.query,
                "max_results": request.max_results,
                "pagination": {
                    "page": request.pagination.page,
                    "page_size": request.pagination.page_size,
                    "total_pages": result.get("pagination", {}).get("total_pages", 1),
                    "total_count": result.get("total_count", 0)
                },
                "user_id": current_user.id,
                "search_method": request.search_method,
                "graph_rag_used": 'graph_rag_results' in result
            }
        )
    except ValidationError as e:
        # Handle validation errors
        log_error(e, {"query": request.query, "user_id": current_user.id})
        logger.warning(f"Validation error in search: {str(e)}")
        return ErrorResponse(
            message="Invalid search parameters",
            errors=[{"detail": str(e)}],
            code="VALIDATION_ERROR"
        )
    except SearchError as e:
        # Handle search errors
        log_error(e, {"query": request.query, "user_id": current_user.id})
        logger.error(f"Search error: {str(e)}")
        logger.error(traceback.format_exc())
        return ErrorResponse(
            message="Search failed",
            errors=[{"detail": str(e)}],
            code="SEARCH_ERROR"
        )
    except Exception as e:
        # Handle unexpected errors
        log_error(e, {"query": request.query, "user_id": current_user.id})
        logger.error(f"Unexpected error in search: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
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
            page=request.pagination.page,
            page_size=request.pagination.page_size,
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
                "population": request.population,
                "study_design": request.study_design,
                "years": request.years,
                "max_results": request.max_results,
                "pagination": {
                    "page": request.pagination.page,
                    "page_size": request.pagination.page_size,
                    "total_pages": result.get("pagination", {}).get("total_pages", 1),
                    "total_count": result.get("total_count", 0)
                },
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
