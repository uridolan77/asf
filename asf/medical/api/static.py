"""
Static file serving for the Medical Research Synthesizer API.

This module provides static file serving for the API, including HTML dashboards
and other static assets needed by the application.
"""

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from starlette.status import HTTP_404_NOT_FOUND

from ..core.logging_config import get_logger

logger = get_logger(__name__)

# Get the static directory
static_dir = Path(__file__).parent.parent / "static"

# Get the templates directory
templates_dir = Path(__file__).parent / "templates"

# Create the router
router = APIRouter(tags=["static"])


@router.get("/static/{file_path:path}")
async def get_static_file(file_path: str):
    """
    Get a static file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        FileResponse: The requested file
        
    Raises:
        HTTPException: If the file is not found
    """
    # Construct the full path
    full_path = static_dir / file_path

    # Check if the file exists
    if not os.path.exists(full_path) or not os.path.isfile(full_path):
        logger.warning(f"Static file not found: {file_path}")
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="File not found")

    # Return the file
    return FileResponse(full_path)


@router.get("/task-monitor")
async def get_task_monitor():
    """
    Get the task monitor page.
    
    Returns:
        FileResponse: The task monitor HTML page
        
    Raises:
        HTTPException: If the file is not found
    """
    # Construct the full path
    full_path = static_dir / "task_monitor.html"

    # Check if the file exists
    if not os.path.exists(full_path) or not os.path.isfile(full_path):
        logger.warning("Task monitor page not found")
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="File not found")

    # Return the file
    return FileResponse(full_path)


@router.get("/messaging-dashboard")
async def get_messaging_dashboard():
    """
    Get the messaging dashboard page.
    
    Returns:
        FileResponse: The messaging dashboard HTML page
        
    Raises:
        HTTPException: If the file is not found
    """
    # Construct the full path
    full_path = static_dir / "messaging_dashboard.html"

    # Check if the file exists
    if not os.path.exists(full_path) or not os.path.isfile(full_path):
        logger.warning("Messaging dashboard page not found")
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="File not found")

    # Return the file
    return FileResponse(full_path)


@router.get("/observability-dashboard")
async def get_observability_dashboard():
    """
    Get the observability dashboard page.
    
    Returns:
        FileResponse: The observability dashboard HTML page
        
    Raises:
        HTTPException: If the file is not found
    """
    # Construct the full path
    full_path = static_dir / "observability_dashboard.html"

    # Check if the file exists
    if not os.path.exists(full_path) or not os.path.isfile(full_path):
        logger.warning("Observability dashboard page not found")
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="File not found")

    # Return the file
    return FileResponse(full_path)


@router.get("/claim-visualizer")
async def get_claim_visualizer():
    """
    Get the scientific claim visualizer page.
    
    Returns:
        FileResponse: The claim visualizer HTML page
        
    Raises:
        HTTPException: If the file is not found
    """
    # Construct the full path
    full_path = templates_dir / "claim_visualizer.html"

    # Check if the file exists
    if not os.path.exists(full_path) or not os.path.isfile(full_path):
        logger.warning("Claim visualizer page not found")
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="File not found")

    # Return the file
    return FileResponse(full_path)
