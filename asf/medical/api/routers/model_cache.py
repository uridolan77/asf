"""
Model Cache API Router

This module provides API endpoints for monitoring and managing the model cache.
"""

import logging
from typing import Dict, Any, List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from asf.medical.api.dependencies import get_admin_user
from asf.medical.ml.model_cache import model_cache

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/v1/model-cache",
    tags=["model-cache"],
    dependencies=[Depends(get_admin_user)],
    responses={404: {"description": "Not found"}},
)

class ModelCacheStats(BaseModel):
    """Model cache statistics."""
    size: int
    max_size: int
    ttl: int
    models: List[Dict[str, Any]]

class ModelCacheResponse(BaseModel):
    """Model cache response."""
    status: str
    message: str
    data: Dict[str, Any] = {}

@router.get("/stats", response_model=ModelCacheStats)
async def get_model_cache_stats():
    """Retrieve current model cache statistics.
    
    Returns:
        ModelCacheStats: Statistics about the current model cache state
        
    Raises:
        HTTPException: If an error occurs retrieving cache stats
    """
    try:
        stats = model_cache.get_stats()
        return ModelCacheStats(**stats)
    except Exception as e:
        logger.error(f"Error getting model cache stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting model cache stats: {str(e)}",
        )

@router.delete("/models/{model_id}", response_model=ModelCacheResponse)
async def remove_model(model_id: str):
    """Remove a specific model from the cache.
    
    Args:
        model_id: The ID of the model to remove
        
    Returns:
        ModelCacheResponse: Success message confirming removal
        
    Raises:
        HTTPException: If an error occurs removing the model
    """
    try:
        model_cache.remove(model_id)
        return ModelCacheResponse(
            status="success",
            message=f"Model removed from cache: {model_id}",
        )
    except Exception as e:
        logger.error(f"Error removing model from cache: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error removing model from cache: {str(e)}",
        )

@router.delete("/clear", response_model=ModelCacheResponse)
async def clear_model_cache():
    """Clear the entire model cache.
    
    Returns:
        ModelCacheResponse: Success message confirming cache clearance
        
    Raises:
        HTTPException: If an error occurs clearing the cache
    """
    try:
        model_cache.clear()
        return ModelCacheResponse(
            status="success",
            message="Model cache cleared",
        )
    except Exception as e:
        logger.error(f"Error clearing model cache: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error clearing model cache: {str(e)}",
        )
