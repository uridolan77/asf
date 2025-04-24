"""
Health check router for the Conexus LLM Gateway.

This module provides endpoints for checking the health and status
of the LLM Gateway and its components.
"""

import logging
import platform
import time
from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query

from asf.conexus.llm_gateway.core.client import get_client
from asf.conexus.llm_gateway.db.database import get_db_engine, get_db_status
from asf.conexus.llm_gateway.observability.metrics import get_or_create_gauge, gauge_set

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/llm/gateway/health", tags=["llm-gateway-health"])

# Start time of the server for uptime calculation
START_TIME = time.time()


@router.get("/status")
async def health_status() -> Dict[str, Any]:
    """
    Get the overall health status of the LLM Gateway.
    
    Returns basic information about the gateway's health and operational status.
    """
    # Calculate uptime
    uptime_seconds = time.time() - START_TIME
    
    # Record uptime in metrics
    gauge_set("llm_gateway_uptime_seconds", uptime_seconds)
    
    # Create basic status response
    status = {
        "status": "operational",
        "version": "1.0.0",  # TODO: Get from package version
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": int(uptime_seconds),
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform()
        }
    }
    
    # Get database status
    try:
        db_status = await get_db_status()
        status["database"] = db_status
        if not db_status.get("connected", False):
            status["status"] = "degraded"
    except Exception as e:
        logger.error(f"Error getting database status: {e}")
        status["database"] = {"connected": False, "error": str(e)}
        status["status"] = "degraded"
    
    return status


@router.get("/providers")
async def provider_health(
    provider_id: str = Query(None, description="Filter by provider ID")
) -> List[Dict[str, Any]]:
    """
    Get the health status of LLM providers.
    
    Args:
        provider_id: Optional provider ID to filter by
        
    Returns:
        Health status information for each provider
    """
    client = get_client()
    
    try:
        # Get provider health information
        provider_statuses = await client.get_provider_health(provider_id=provider_id)
        
        # Record provider statuses in metrics
        for provider in provider_statuses:
            status = provider.get("status", "unknown")
            provider_id = provider.get("provider_id", "unknown")
            gauge_set(
                "llm_gateway_provider_status", 
                1.0 if status == "operational" else 0.0,
                {"provider_id": provider_id, "status": status}
            )
        
        return provider_statuses
        
    except Exception as e:
        logger.error(f"Error checking provider health: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check provider health: {str(e)}"
        )


@router.get("/detailed")
async def detailed_health() -> Dict[str, Any]:
    """
    Get detailed health information about all components of the LLM Gateway.
    
    Returns comprehensive status information about all subsystems.
    """
    client = get_client()
    
    detailed_status = {
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": int(time.time() - START_TIME),
    }
    
    # Get basic status
    try:
        basic_status = await health_status()
        detailed_status.update(basic_status)
    except Exception as e:
        logger.error(f"Error getting basic health status: {e}")
        detailed_status["status"] = "error"
        detailed_status["basic_status_error"] = str(e)
    
    # Get provider status
    try:
        providers = await provider_health(provider_id=None)
        detailed_status["providers"] = providers
        
        # If any provider is down, mark overall status as degraded
        if detailed_status["status"] == "operational" and any(
            p.get("status", "") != "operational" for p in providers
        ):
            detailed_status["status"] = "degraded"
            
    except Exception as e:
        logger.error(f"Error getting provider health status: {e}")
        detailed_status["providers_error"] = str(e)
        detailed_status["status"] = "degraded"
    
    # Get caching status if available
    try:
        cache_status = await client.get_cache_status()
        detailed_status["cache"] = cache_status
    except Exception as e:
        logger.error(f"Error getting cache status: {e}")
        detailed_status["cache"] = {"status": "error", "error": str(e)}
    
    return detailed_status