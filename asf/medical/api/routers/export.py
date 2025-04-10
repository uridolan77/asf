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

from asf.medical.api.models import ExportRequest
from asf.medical.api.dependencies import get_synthesizer, get_current_user
from asf.medical.api.auth import User
from asf.medical.data_ingestion_layer.enhanced_medical_research_synthesizer import EnhancedMedicalResearchSynthesizer
from asf.medical.api.export_utils import (
    export_to_json, export_to_csv, export_to_excel, export_to_pdf,
    export_contradiction_analysis_to_pdf
)

# Initialize router
router = APIRouter(prefix="/v1/export", tags=["Export"])

# Set up logging
logger = logging.getLogger(__name__)

# Reference to result storage from other routers (will be replaced with database in Phase 2)
from asf.medical.api.routers.search import result_storage as search_result_storage
from asf.medical.api.routers.analysis import result_storage as analysis_result_storage

@router.post("/{format}")
async def export_results(
    format: str,
    request: ExportRequest,
    background_tasks: BackgroundTasks,
    synthesizer: EnhancedMedicalResearchSynthesizer = Depends(get_synthesizer),
    current_user: User = Depends(get_current_user)
):
    """
    Export search results in various formats.

    This endpoint allows downloading search results in JSON, CSV, Excel, or PDF formats.
    It can use either a stored result_id or execute a new search with the provided query.
    """
    if format not in ["json", "csv", "excel", "pdf"]:
        logger.error(f"Unsupported export format: {format}")
        raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")

    try:
        # Get results
        if request.result_id:
            # Check if result_id is in search results
            if request.result_id in search_result_storage:
                # Use stored search results
                result_data = search_result_storage[request.result_id]
                if 'results' in result_data:
                    results = result_data['results']
                    query_text = result_data.get('query', 'Stored query')
                    logger.info(f"Exporting stored search results: {request.result_id} (format={format})")
                    return await export_search_results(format, results, query_text)
                else:
                    logger.error(f"Invalid result data for result_id: {request.result_id}")
                    raise HTTPException(status_code=400, detail="Invalid result data")
            
            # Check if result_id is in analysis results
            elif request.result_id in analysis_result_storage:
                # This is a contradiction analysis
                result_data = analysis_result_storage[request.result_id]
                if 'analysis' in result_data:
                    analysis = result_data['analysis']
                    query_text = result_data.get('query', 'Contradiction Analysis')
                    logger.info(f"Exporting stored contradiction analysis: {request.result_id} (format={format})")
                    return await export_contradiction_analysis(format, analysis, query_text)
                else:
                    logger.error(f"Invalid analysis data for result_id: {request.result_id}")
                    raise HTTPException(status_code=400, detail="Invalid analysis data")
            else:
                logger.error(f"Result not found: {request.result_id}")
                raise HTTPException(status_code=404, detail="Result not found")
        
        elif request.query:
            # Execute a new search
            logger.info(f"Executing new search for export: {request.query} (format={format})")
            results = synthesizer.search_and_enrich(query=request.query, max_results=request.max_results)
            return await export_search_results(format, results, request.query)
        
        else:
            logger.error("Neither result_id nor query provided")
            raise HTTPException(status_code=400, detail="Either result_id or query must be provided")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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
        if format == "json":
            return export_to_json(results, query_text)
        elif format == "csv":
            return export_to_csv(results, query_text)
        elif format == "excel":
            return export_to_excel(results, query_text)
        elif format == "pdf":
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
        if format == "json":
            return JSONResponse(content=analysis)
        elif format == "pdf":
            # Create a temporary file for the PDF
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            temp_file.close()
            
            # Create the PDF
            pdf_path = export_contradiction_analysis_to_pdf(analysis, query_text, temp_file.name)
            
            # Return the file
            return FileResponse(
                path=pdf_path,
                filename=f"contradiction_analysis.pdf",
                media_type="application/pdf",
                background=BackgroundTasks.add_task(lambda: os.unlink(pdf_path))
            )
        else:
            logger.error(f"Format {format} not supported for contradiction analysis")
            raise HTTPException(status_code=400, detail=f"Format {format} not supported for contradiction analysis")
    except Exception as e:
        logger.error(f"Error in export_contradiction_analysis: {str(e)}")
        raise
