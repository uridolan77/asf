"""
Resource Monitoring API Router

This module provides API endpoints for monitoring and managing resource usage.
"""

import logging
import psutil
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from asf.medical.api.auth import get_current_user
from asf.medical.core.resource_limiter import resource_limiter

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/v1/resources",
    tags=["resources"],
    dependencies=[Depends(get_current_user)],
    responses={404: {"description": "Not found"}},
)

class ResourceUsage(BaseModel):
    """Resource usage information."""
    cpu_percent: float
    memory_percent: float
    gpu_percent: float
    concurrent_tasks: int
    max_cpu_percent: float
    max_memory_percent: float
    max_gpu_percent: float
    max_concurrent_tasks: int

class ResourceLimits(BaseModel):
    """Resource limits."""
    max_cpu_percent: float
    max_memory_percent: float
    max_gpu_percent: float
    max_concurrent_tasks: int

class ResourceResponse(BaseModel):
    """Resource response."""
    status: str
    message: str
    data: Dict[str, Any] = {}

# Routes
@router.get("/usage", response_model=ResourceUsage)
async def get_resource_usage():
    """
    Get current resource usage.
    
    Returns:
        Resource usage information
    Get current resource limits.
    
    Returns:
        Resource limits
    Update resource limits.
    
    Args:
        limits: New resource limits
        
    Returns:
        Response with status and message
    Get system information.
    
    Returns:
        System information