"""
Export router for the Medical Research Synthesizer API.

This module provides endpoints for exporting search results and analyses
in various formats (JSON, CSV, Excel, PDF).
"""

import logging
import tempfile
import json
import os
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse

from ...tasks.export_tasks import (
    generate_pdf_report, generate_contradiction_pdf_report, get_task_result
)

from ..models.export import ExportRequest
from ..models.base import APIResponse, ErrorResponse
from ..dependencies import get_synthesizer, get_search_service, get_analysis_service
from ..auth import get_current_active_user
from ...services.search_service import SearchService
from ...services.analysis_service import AnalysisService
from ...storage.models import User
from ..export_utils import (
    export_to_json, export_to_csv, export_to_excel
)
from ...core.observability import async_timed, log_error
from ...core.exceptions import ValidationError


router = APIRouter(prefix="/export", tags=["export"])

logger = logging.getLogger(__name__)

background_tasks_status = {}

@router.post("/{format}", response_model=APIResponse[Dict[str, Any]])
@async_timed("export_results_endpoint")
async def export_results(
    format: str,
    request: ExportRequest,
    background_tasks: BackgroundTasks,
    synthesizer = Depends(get_synthesizer),
    search_service: SearchService = Depends(get_search_service),
    analysis_service: AnalysisService = Depends(get_analysis_service),
    current_user: User = Depends(get_current_active_user)
):
    """Export search results or analysis to the specified format.
    
    Args:
        format: The export format (json, csv, excel, pdf)
        request: The export request containing result_id or query
        background_tasks: FastAPI background tasks manager
        synthesizer: The search synthesizer service
        search_service: The search service
        analysis_service: The analysis service
        current_user: The current authenticated user
        
    Returns:
        APIResponse containing export status and file URL
        
    Raises:
        HTTPException: If the format is not supported, result not found, or an error occurs
    """
    try:
        if format.lower() not in ["json", "csv", "excel", "pdf"]:
            logger.error(f"Unsupported export format: {format}")
            return ErrorResponse(
                message=f"Unsupported format: {format}",
                errors=[{"detail": f"Format must be one of: json, csv, excel, pdf"}],
                code="INVALID_FORMAT"
            )

        if request.result_id:
            search_result = await search_service.get_result(request.result_id, current_user.id)

            if search_result:
                logger.info(f"Exporting stored search result: {request.result_id} (format={format})")

                results = search_result.get("results", [])
                query_text = search_result.get("query", "Search Result")

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

            analysis_result = await analysis_service.get_analysis(request.result_id, current_user.id)

            if analysis_result:
                logger.info(f"Exporting stored analysis result: {request.result_id} (format={format})")

                analysis = analysis_result.get("analysis", {})
                query_text = analysis_result.get("query", "Analysis Result")

                response = await export_contradiction_analysis(format, analysis, query_text, background_tasks)

                if isinstance(response, JSONResponse) and response.body:
                    response_data = json.loads(response.body)
                    if response_data.get("status") == "processing":
                        return APIResponse(
                            success=True,
                            message=f"PDF generation started. The file will be available shortly.",
                            data={
                                "status": "processing",
                                "task_id": response_data.get("task_id"),
                                "file_path": response_data.get("file_path")
                            },
                            meta={
                                "format": format,
                                "result_id": request.result_id,
                                "query": query_text
                            }
                        )

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

            logger.error(f"Result not found: {request.result_id}")
            return ErrorResponse(
                message="Result not found",
                errors=[{"detail": f"No result found with ID {request.result_id}"}],
                code="NOT_FOUND"
            )

        elif request.query:
            logger.info(f"Executing new search for export: {request.query} (format={format})")

            results = synthesizer.search_and_enrich(query=request.query, max_results=request.max_results)

            response = await export_search_results(format, results, request.query, background_tasks)

            if isinstance(response, JSONResponse) and response.body:
                response_data = json.loads(response.body)
                if response_data.get("status") == "processing":
                    return APIResponse(
                        success=True,
                        message=f"PDF generation started. The file will be available shortly.",
                        data={
                            "status": "processing",
                            "task_id": response_data.get("task_id"),
                            "file_path": response_data.get("file_path")
                        },
                        meta={
                            "format": format,
                            "query": request.query,
                            "result_count": len(results)
                        }
                    )

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
            logger.error("Neither result_id nor query provided")
            return ErrorResponse(
                message="Either result_id or query must be provided",
                errors=[{"detail": "Either result_id or query must be provided"}],
                code="MISSING_PARAMETER"
            )

    except HTTPException as e:
        logger.error(f"Error: {str(e)}")
        raise

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        logger.error(f"Validation error: {str(e)}")
        raise ValidationError(f"Validation failed: {str(e)}")
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

async def export_search_results(format: str, results: list, query_text: str, background_tasks: Optional[BackgroundTasks] = None):
    """Export search results to the specified format.
    
    Args:
        format: The export format (json, csv, excel, pdf)
        results: The search results to export
        query_text: The query text that generated these results
        background_tasks: Optional background tasks handler
        
    Returns:
        FileResponse or JSONResponse depending on the format
        
    Raises:
        ValueError: If the format is not supported
    """
    try:
        if format.lower() == "pdf":
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            temp_file.close()

            output_path = temp_file.name

            message = generate_pdf_report.send(results, query_text, output_path)
            task_id = message.message_id

            return JSONResponse(content={
                "status": "processing",
                "message": "PDF generation started. The file will be available shortly.",
                "task_id": task_id,
                "file_path": output_path
            })

        if format.lower() == "json":
            return export_to_json(results, query_text)
        elif format.lower() == "csv":
            return export_to_csv(results, query_text)
        elif format.lower() == "excel":
            return export_to_excel(results, query_text)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    except Exception as e:
        logger.error(f"Error in export_search_results: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Error in export_search_results: {str(e)}"
        )


async def export_contradiction_analysis(format: str, analysis: dict, query_text: str, background_tasks: Optional[BackgroundTasks] = None):
    """Export contradiction analysis to the specified format.
    
    Args:
        format: The export format (json, pdf)
        analysis: The contradiction analysis data to export
        query_text: The query text that generated this analysis
        background_tasks: Optional background tasks handler
        
    Returns:
        FileResponse or JSONResponse depending on the format
        
    Raises:
        ValueError: If the format is not supported for contradiction analysis
    """
    try:
        if format.lower() == "pdf":
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            temp_file.close()

            output_path = temp_file.name

            message = generate_contradiction_pdf_report.send(analysis, query_text, output_path)
            task_id = message.message_id

            return JSONResponse(content={
                "status": "processing",
                "message": "PDF generation started. The file will be available shortly.",
                "task_id": task_id,
                "file_path": output_path
            })

        if format.lower() == "json":
            return export_to_json(analysis.get("contradictions", []), query_text)
        else:
            raise ValueError(f"Unsupported export format for contradiction analysis: {format}")
    except Exception as e:
        logger.error(f"Error in export_contradiction_analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in export_contradiction_analysis: {str(e)}"
        )


@router.get("/status/{task_id}", response_model=APIResponse[Dict[str, Any]])
async def get_export_status(task_id: str):
    """Get the status of an export task.
    
    Args:
        task_id: The ID of the export task or file path
        
    Returns:
        APIResponse with status information and file URL if completed
    """
    try:
        task_result = get_task_result(task_id)

        if task_result:
            if task_result.get("status") == "completed":
                output_path = task_result.get("output_path")
                if output_path and os.path.exists(output_path):
                    return APIResponse(
                        success=True,
                        message="Export completed successfully",
                        data={
                            "status": "completed",
                            "file_url": f"/export/download/{os.path.basename(output_path)}"
                        }
                    )
                else:
                    return APIResponse(
                        success=False,
                        message="Export file not found",
                        data={"status": "failed", "error": "File not found"}
                    )

            return APIResponse(
                success=True,
                message=f"Export status: {task_result.get('status')}",
                data=task_result
            )

        if os.path.exists(task_id):
            return APIResponse(
                success=True,
                message="Export completed successfully",
                data={
                    "status": "completed",
                    "file_url": f"/export/download/{os.path.basename(task_id)}"
                }
            )

        return APIResponse(
            success=False,
            message="Export task not found",
            data={"status": "not_found"}
        )

    except Exception as e:
        logger.error(f"Error getting export status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting export status: {str(e)}"
        )

@router.get("/download/{file_name}", response_class=FileResponse)
async def download_export(file_name: str):
    """Download an exported file.
    
    Args:
        file_name: The name of the file to download
        
    Returns:
        FileResponse containing the requested file
        
    Raises:
        HTTPException: If the file is not found or an error occurs
    """
    try:
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, file_name)

        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found"
            )

        return FileResponse(
            path=file_path,
            filename=file_name,
            media_type="application/octet-stream"
        )

    except HTTPException:
        logger.error(f"Error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error downloading export: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error downloading export: {str(e)}"
        )
