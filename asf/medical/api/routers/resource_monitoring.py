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

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/v1/resources",
    tags=["resources"],
    dependencies=[Depends(get_current_user)],
    responses={404: {"description": "Not found"}},
)

# Models
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
    """
    try:
        # Get resource usage
        usage = resource_limiter.get_resource_usage()
        
        # Add limits
        usage["max_cpu_percent"] = resource_limiter.max_cpu_percent
        usage["max_memory_percent"] = resource_limiter.max_memory_percent
        usage["max_gpu_percent"] = resource_limiter.max_gpu_percent
        usage["max_concurrent_tasks"] = resource_limiter.max_concurrent_tasks
        
        return ResourceUsage(**usage)
    except Exception as e:
        logger.error(f"Error getting resource usage: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting resource usage: {str(e)}",
        )

@router.get("/limits", response_model=ResourceLimits)
async def get_resource_limits():
    """
    Get current resource limits.
    
    Returns:
        Resource limits
    """
    try:
        return ResourceLimits(
            max_cpu_percent=resource_limiter.max_cpu_percent,
            max_memory_percent=resource_limiter.max_memory_percent,
            max_gpu_percent=resource_limiter.max_gpu_percent,
            max_concurrent_tasks=resource_limiter.max_concurrent_tasks,
        )
    except Exception as e:
        logger.error(f"Error getting resource limits: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting resource limits: {str(e)}",
        )

@router.put("/limits", response_model=ResourceResponse)
async def update_resource_limits(limits: ResourceLimits):
    """
    Update resource limits.
    
    Args:
        limits: New resource limits
        
    Returns:
        Response with status and message
    """
    try:
        # Update limits
        resource_limiter.max_cpu_percent = limits.max_cpu_percent
        resource_limiter.max_memory_percent = limits.max_memory_percent
        resource_limiter.max_gpu_percent = limits.max_gpu_percent
        resource_limiter.max_concurrent_tasks = limits.max_concurrent_tasks
        
        return ResourceResponse(
            status="success",
            message="Resource limits updated",
            data={
                "max_cpu_percent": resource_limiter.max_cpu_percent,
                "max_memory_percent": resource_limiter.max_memory_percent,
                "max_gpu_percent": resource_limiter.max_gpu_percent,
                "max_concurrent_tasks": resource_limiter.max_concurrent_tasks,
            }
        )
    except Exception as e:
        logger.error(f"Error updating resource limits: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating resource limits: {str(e)}",
        )

@router.get("/system", response_model=Dict[str, Any])
async def get_system_info():
    """
    Get system information.
    
    Returns:
        System information
    """
    try:
        # Get CPU info
        cpu_count = psutil.cpu_count(logical=False)
        cpu_count_logical = psutil.cpu_count(logical=True)
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Get memory info
        memory = psutil.virtual_memory()
        
        # Get disk info
        disk = psutil.disk_usage("/")
        
        # Get GPU info
        gpu_info = {}
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info["available"] = True
                gpu_info["count"] = torch.cuda.device_count()
                gpu_info["name"] = torch.cuda.get_device_name(0)
                gpu_info["memory_allocated"] = torch.cuda.memory_allocated(0)
                gpu_info["memory_reserved"] = torch.cuda.memory_reserved(0)
                gpu_info["max_memory_allocated"] = torch.cuda.max_memory_allocated(0)
                gpu_info["max_memory_reserved"] = torch.cuda.max_memory_reserved(0)
            else:
                gpu_info["available"] = False
        except ImportError:
            gpu_info["available"] = False
        
        return {
            "cpu": {
                "count": cpu_count,
                "count_logical": cpu_count_logical,
                "percent": cpu_percent,
            },
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "percent": memory.percent,
            },
            "disk": {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": disk.percent,
            },
            "gpu": gpu_info,
        }
    except Exception as e:
        logger.error(f"Error getting system info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting system info: {str(e)}",
        )
