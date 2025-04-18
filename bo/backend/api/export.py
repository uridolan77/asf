"""
Export API endpoints for the BO backend.

This module provides endpoints for exporting search results and analyses
in various formats (JSON, CSV, Excel, PDF).
"""

from fastapi import APIRouter, Depends, HTTPException, status, Body, BackgroundTasks
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import httpx
import logging
import os
from datetime import datetime

from .auth import get_current_user, User
from .utils import handle_api_error

router = APIRouter(prefix="/api/medical/export", tags=["export"])

logger = logging.getLogger(__name__)

# Environment variables
MEDICAL_API_URL = os.getenv("MEDICAL_API_URL", "http://localhost:8000")

# Models
class ExportRequest(BaseModel):
    result_id: Optional[str] = None
    analysis_id: Optional[str] = None
    query: Optional[str] = None
    max_results: int = 20
    include_abstracts: bool = True
    include_metadata: bool = True

# Track background tasks
background_tasks_status = {}

@router.post("/{format}", response_model=Dict[str, Any])
async def export_results(
    format: str,
    request: ExportRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """
    Export search results or analysis to the specified format.
    
    This endpoint exports search results or analysis results to the specified format.
    Supported formats are JSON, CSV, Excel, and PDF.
    
    For large exports or PDF generation, the export is performed as a background task.
    """
    if format not in ["json", "csv", "excel", "pdf"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format: {format}. Supported formats are: json, csv, excel, pdf"
        )
        
    if not request.result_id and not request.analysis_id and not request.query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either result_id, analysis_id, or query must be provided"
        )
    
    try:
        # For PDF exports, use background task
        if format == "pdf":
            task_id = f"export_{format}_{datetime.now().timestamp()}"
            background_tasks_status[task_id] = {"status": "processing", "progress": 0}
            
            background_tasks.add_task(
                export_in_background,
                task_id,
                format,
                request,
                current_user.token
            )
            
            return {
                "success": True,
                "message": f"Export to {format} started as background task",
                "data": {
                    "task_id": task_id,
                    "status_url": f"/api/medical/export/status/{task_id}"
                }
            }
        
        # For other formats, process synchronously
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{MEDICAL_API_URL}/api/export/{format}",
                json=request.dict(),
                headers={"Authorization": f"Bearer {current_user.token}"}
            )
            
            if response.status_code != 200:
                return handle_api_error(response)
                
            return response.json()
    except Exception as e:
        logger.error(f"Error exporting results to {format}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export results to {format}: {str(e)}"
        )

@router.get("/status/{task_id}", response_model=Dict[str, Any])
async def get_export_status(
    task_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get the status of a background export task.
    
    This endpoint retrieves the status of a background export task.
    """
    if task_id not in background_tasks_status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Export task {task_id} not found"
        )
        
    return {
        "success": True,
        "data": background_tasks_status[task_id]
    }

@router.get("/download/{task_id}", response_model=Dict[str, Any])
async def download_export(
    task_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Download a completed export.
    
    This endpoint downloads a completed export.
    """
    if task_id not in background_tasks_status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Export task {task_id} not found"
        )
        
    task_status = background_tasks_status[task_id]
    
    if task_status["status"] != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Export task {task_id} is not completed yet. Current status: {task_status['status']}"
        )
        
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{MEDICAL_API_URL}/api/export/download/{task_id}",
                headers={"Authorization": f"Bearer {current_user.token}"}
            )
            
            if response.status_code != 200:
                return handle_api_error(response)
                
            return response.json()
    except Exception as e:
        logger.error(f"Error downloading export: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to download export: {str(e)}"
        )

async def export_in_background(task_id: str, format: str, request: ExportRequest, token: str):
    """
    Perform export in background.
    
    This function is called as a background task to perform exports that may take longer,
    such as PDF generation.
    """
    try:
        background_tasks_status[task_id]["progress"] = 10
        
        async with httpx.AsyncClient() as client:
            background_tasks_status[task_id]["progress"] = 30
            
            response = await client.post(
                f"{MEDICAL_API_URL}/api/export/{format}",
                json=request.dict(),
                headers={"Authorization": f"Bearer {token}"}
            )
            
            background_tasks_status[task_id]["progress"] = 70
            
            if response.status_code != 200:
                background_tasks_status[task_id]["status"] = "failed"
                background_tasks_status[task_id]["error"] = response.text
                return
            
            result = response.json()
            background_tasks_status[task_id]["progress"] = 100
            background_tasks_status[task_id]["status"] = "completed"
            background_tasks_status[task_id]["result"] = result
            background_tasks_status[task_id]["download_url"] = result.get("data", {}).get("download_url")
    except Exception as e:
        logger.error(f"Error in background export task: {str(e)}")
        background_tasks_status[task_id]["status"] = "failed"
        background_tasks_status[task_id]["error"] = str(e)
