"""
Cache Management API Routes

This module provides API routes for managing the LLM Gateway cache system.
"""

import os
import logging
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query, Body, Path
from pydantic import BaseModel, Field

from asf.medical.llm_gateway.cache.cache_manager import get_cache_manager
from asf.medical.llm_gateway.gateway import get_gateway
from asf.medical.llm_gateway.cache.cache_warming import warm_cache_with_queries

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cache", tags=["cache"])

class CacheWarmingQuery(BaseModel):
    """Model for a cache warming query."""
    
    prompt: str = Field(..., description="The prompt to execute")
    model: str = Field(..., description="Model identifier")
    temperature: float = Field(0.0, description="Temperature setting (0-1)")
    max_tokens: int = Field(500, description="Maximum tokens to generate")
    system_prompt: Optional[str] = Field(None, description="System prompt")

class CacheWarmingRequest(BaseModel):
    """Model for cache warming request."""
    
    queries: List[CacheWarmingQuery] = Field(..., description="List of queries to warm the cache with")
    concurrency: int = Field(2, description="Maximum number of concurrent requests")

class CacheClearRequest(BaseModel):
    """Model for cache clearing request."""
    
    confirm: bool = Field(..., description="Confirmation flag (must be true)")

@router.get("/stats")
async def get_cache_stats() -> Dict[str, Any]:
    """
    Get cache statistics.
    
    Returns:
        Dictionary with cache statistics
    """
    cache_manager = get_cache_manager()
    stats = cache_manager.get_stats()
    return {"status": "ok", "stats": stats}

@router.post("/clear")
async def clear_cache(request: CacheClearRequest) -> Dict[str, Any]:
    """
    Clear the cache.
    
    Args:
        request: Cache clear request
        
    Returns:
        Status message
    """
    if not request.confirm:
        raise HTTPException(status_code=400, detail="Confirmation required to clear cache")
        
    cache_manager = get_cache_manager()
    await cache_manager.clear()
    
    return {
        "status": "ok",
        "message": "Cache cleared successfully"
    }

@router.post("/warm")
async def warm_cache(request: CacheWarmingRequest) -> Dict[str, Any]:
    """
    Warm the cache with the provided queries.
    
    Args:
        request: Cache warming request
        
    Returns:
        Dictionary with warming results
    """
    gateway = get_gateway()
    
    # Convert Pydantic models to dictionaries
    queries = [q.dict() for q in request.queries]
    
    # Warm the cache
    results = await warm_cache_with_queries(
        gateway=gateway,
        queries=queries,
        concurrency=request.concurrency
    )
    
    return results

@router.get("/config")
async def get_cache_config() -> Dict[str, Any]:
    """
    Get the current cache configuration.
    
    Returns:
        Dictionary with cache configuration
    """
    cache_manager = get_cache_manager()
    
    # Get cache directory from environment or default
    cache_dir = os.environ.get("LLM_CACHE_DIR")
    if not cache_dir:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".llm_gateway", "cache")
    
    # Get cache configuration
    config = {
        "enabled": cache_manager.enable_caching,
        "persistence": {
            "type": cache_manager._get_persistence_type(),
            "location": cache_manager._get_persistence_location()
        },
        "embedding_provider": cache_manager._get_embedding_provider_info(),
        "similarity_threshold": getattr(cache_manager.semantic_cache, 'similarity_threshold', None),
        "max_entries": getattr(cache_manager.semantic_cache, 'max_entries', None),
        "ttl_seconds": getattr(cache_manager.semantic_cache, 'ttl_seconds', None)
    }
    
    return {
        "status": "ok",
        "config": config
    }