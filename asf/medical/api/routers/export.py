"""
Export router for the Medical Research Synthesizer API.

This module provides endpoints for exporting search results and analyses
in various formats (JSON, CSV, Excel, PDF).
"""

import os
import tempfile
import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse

from asf.medical.api.models.export import ExportRequest
from asf.medical.api.models.base import APIResponse, ErrorResponse
from asf.medical.api.dependencies import get_synthesizer, get_search_service, get_analysis_service
from asf.medical.api.auth import get_current_active_user
from asf.medical.data_ingestion_layer.enhanced_medical_research_synthesizer import EnhancedMedicalResearchSynthesizer
from asf.medical.services.search_service import SearchService
from asf.medical.services.analysis_service import AnalysisService
from asf.medical.storage.models import User
from asf.medical.api.export_utils import (
    export_to_json, export_to_csv, export_to_excel, export_to_pdf,
    export_contradiction_analysis_to_pdf
)
from asf.medical.core.monitoring import async_timed, log_error

# Initialize router
router = APIRouter(prefix="/export", tags=["export"])

# Set up logging
logger = logging.getLogger(__name__)

@router.post("/{format}", response_model=APIResponse[Dict[str, Any]])
@async_timed("export_results_endpoint")
async def export_results(
    format: str,
    request: ExportRequest,
    background_tasks: BackgroundTasks,
    synthesizer: EnhancedMedicalResearchSynthesizer = Depends(get_synthesizer),
    search_service: SearchService = Depends(get_search_service),
    analysis_service: AnalysisService = Depends(get_analysis_service),
    current_user: User = Depends(get_current_active_user)
):
    """
    Export search results in various formats.

    This endpoint allows downloading search results in JSON, CSV, Excel, or PDF formats.
    It can use either a stored result_id or execute a new search with the provided query.
    """
    try:
        if format.lower() not in ["json", "csv", "excel", "pdf"]:
            logger.error(f"Unsupported export format: {format}")
            return ErrorResponse(
                message=f"Unsupported format: {format}",
                errors=[{"detail": f"Format must be one of: json, csv, excel, pdf"}],
                code="INVALID_FORMAT"
            )
        
        # Check if we have a result_id or query
        if request.result_id:
            # Try to get search result
            search_result = await search_service.get_result(request.result_id, current_user.id)
            
            if search_result:
                # This is a search result
                logger.info(f"Exporting stored search result: {request.result_id} (format={format})")
                
                # Get the results from the search result
                results = search_result.get("results", [])
                query_text = search_result.get("query", "Search Result")
                
                # Export the results
                response = await export_search_results(format, results, query_text)
                
                return APIResponse(
                    success=True,
                    message=f"Search result exported successfully as {format}",
                    data={"file_url": response.headers.get("location")},
                    meta={
                        "format": format,
                        "result_id": request.result_id,
                        "query": query_text,
                        "result_count": len(results)
                    }
                )
            
            # Try to get analysis result
            analysis_result = await analysis_service.get_analysis(request.result_id, current_user.id)
            
            if analysis_result:
                # This is an analysis result
                logger.info(f"Exporting stored analysis result: {request.result_id} (format={format})")
                
                # Get the analysis from the result
                analysis = analysis_result.get("analysis", {})
                query_text = analysis_result.get("query", "Analysis Result")
                
                # Export the analysis
                response = await export_contradiction_analysis(format, analysis, query_text)
                
                return APIResponse(
                    success=True,
                    message=f"Analysis result exported successfully as {format}",
                    data={"file_url": response.headers.get("location")},
                    meta={
                        "format": format,
                        "result_id": request.result_id,
                        "query": query_text
                    }
                )
            
            # Result not found
            logger.error(f"Result not found: {request.result_id}")
            return ErrorResponse(
                message="Result not found",
                errors=[{"detail": f"No result found with ID {request.result_id}"}],
                code="NOT_FOUND"
            )
        
        elif request.query:
            # Execute a new search
            logger.info(f"Executing new search for export: {request.query} (format={format})")
            
            # Execute the search
            results = synthesizer.search_and_enrich(query=request.query, max_results=request.max_results)
            
            # Export the results
            response = await export_search_results(format, results, request.query)
            
            return APIResponse(
                success=True,
                message=f"Search result exported successfully as {format}",
                data={"file_url": response.headers.get("location")},
                meta={
                    "format": format,
                    "query": request.query,
                    "result_count": len(results)
                }
            )
        
        else:
            # Neither result_id nor query provided
            logger.error("Neither result_id nor query provided")
            return ErrorResponse(
                message="Either result_id or query must be provided",
                errors=[{"detail": "Either result_id or query must be provided"}],
                code="MISSING_PARAMETER"
            )
    
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        # Log and handle other exceptions
        log_error(e, {
            "format": format,
            "result_id": request.result_id,
            "query": request.query,
            "user_id": current_user.id
        })
        logger.error(f"Error exporting results: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error exporting results: {str(e)}"
        )

async def export_search_results(format: str, results: list, query_text: str):
    """
    Export search results in the specified format.
    
    Args:
        format: The export format (json, csv, excel, pdf)
        results: The search results to export
        query_text: The query that produced the results
        
    Returns:
        The appropriate response for the requested format
    """
    try:
        if format.lower() == "json":
            return export_to_json(results, query_text)
        elif format.lower() == "csv":
            return export_to_csv(results, query_text)
        elif format.lower() == "excel":
            return export_to_excel(results, query_text)
        elif format.lower() == "pdf":
            return export_to_pdf(results, query_text)
    except Exception as e:
        logger.error(f"Error in export_search_results: {str(e)}")
        raise

async def export_contradiction_analysis(format: str, analysis: dict, query_text: str):
    """
    Export contradiction analysis in the specified format.
    
    Args:
        format: The export format (json, pdf)
        analysis: The contradiction analysis to export
        query_text: The query that produced the analysis
        
    Returns:
        The appropriate response for the requested format
    """
    try:
        if format.lower() == "json":
            return JSONResponse(content=analysis)
        elif format.lower() == "pdf":
            return export_contradiction_analysis_to_pdf(analysis, query_text)
        else:
            # For CSV and Excel, convert to JSON first
            return JSONResponse(content=analysis)
    except Exception as e:
        logger.error(f"Error in export_contradiction_analysis: {str(e)}")
        raise
