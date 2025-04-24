"""
Resource Monitoring API Router

This module provides API endpoints for monitoring and managing resource usage.
"""
import logging
import psutil
import platform
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, status, HTTPException, Query, Path
from pydantic import BaseModel
from asf.medical.api.auth import get_current_user
from asf.medical.llm_gateway.core.client import LLMGatewayClient
from asf.medical.llm_gateway.core.resource_init import get_resource_stats
from asf.medical.llm_gateway.core.errors import ResourceError
from asf.medical.core.service_initialization import get_service

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
    gpu_percent: float = 0.0
    concurrent_tasks: int
    max_cpu_percent: float
    max_memory_percent: float
    max_gpu_percent: float = 0.0
    max_concurrent_tasks: int
    llm_metrics: Dict[str, Any] = {}
    timestamp: datetime = datetime.now()

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

class LLMResourcePool(BaseModel):
    """LLM Resource Pool information."""
    pool_id: str
    provider_id: Optional[str] = None
    resource_type: str
    total_resources: int
    available_resources: int
    in_use_resources: int
    waiting_requests: int
    circuit_breaker_status: str = "closed"
    health_status: str = "healthy"
    metrics: Dict[str, Any] = {}

class LLMResourceMetrics(BaseModel):
    """LLM Resource Metrics."""
    resource_pools: List[LLMResourcePool]
    total_pools: int
    total_resources: int
    available_resources: int
    in_use_resources: int
    waiting_requests: int
    unhealthy_pools: int

class LLMUsageStats(BaseModel):
    """LLM Usage Statistics."""
    total_requests: int = 0
    total_tokens: int = 0
    total_errors: int = 0
    average_response_time_ms: float = 0.0
    requests_per_minute: float = 0.0
    tokens_per_minute: float = 0.0
    tokens_per_request: float = 0.0
    error_rate: float = 0.0
    provider_stats: Dict[str, Dict[str, Any]] = {}
    model_stats: Dict[str, Dict[str, Any]] = {}

# Routes
@router.get("/usage", response_model=ResourceUsage)
async def get_resource_usage():
    """
    Get current resource usage.

    Returns:
        Resource usage information
    """
    try:
        # Get system resource usage
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Get LLM resource stats
        llm_stats = get_resource_stats()
        
        # Determine concurrent tasks from LLM stats
        concurrent_tasks = 0
        for pool_id, pool_stats in llm_stats.items():
            if not pool_id.startswith('_'):  # Skip aggregates
                concurrent_tasks += pool_stats.get("in_use", 0)
        
        # Define resource limits (could be retrieved from config)
        max_cpu_percent = 90.0
        max_memory_percent = 85.0
        max_gpu_percent = 95.0
        max_concurrent_tasks = 100
        
        # Calculate LLM-specific metrics for the response
        llm_metrics = {
            "total_pools": sum(1 for k in llm_stats.keys() if not k.startswith('_')),
            "in_use_resources": sum(pool_stats.get("in_use", 0) for k, pool_stats in llm_stats.items() if not k.startswith('_')),
            "available_resources": sum(pool_stats.get("available", 0) for k, pool_stats in llm_stats.items() if not k.startswith('_')),
            "waiting_requests": sum(pool_stats.get("waiting_requests", 0) for k, pool_stats in llm_stats.items() if not k.startswith('_')),
            "error_count": sum(pool_stats.get("error_count", 0) for k, pool_stats in llm_stats.items() if not k.startswith('_')),
        }
        
        # Calculate resource health scores
        llm_metrics["health_score"] = 1.0
        for pool_id, pool_stats in llm_stats.items():
            if not pool_id.startswith('_') and pool_stats.get("health_check", {}).get("status") != "healthy":
                llm_metrics["health_score"] = 0.5
                break
        
        return ResourceUsage(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            concurrent_tasks=concurrent_tasks,
            max_cpu_percent=max_cpu_percent,
            max_memory_percent=max_memory_percent,
            max_gpu_percent=max_gpu_percent,
            max_concurrent_tasks=max_concurrent_tasks,
            llm_metrics=llm_metrics,
            timestamp=datetime.now()
        )
    except Exception as e:
        logger.error(f"Error getting resource usage: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get resource usage: {str(e)}"
        )

@router.get("/limits", response_model=ResourceLimits)
async def get_resource_limits():
    """
    Get current resource limits.

    Returns:
        Resource limits
    """
    # Return hardcoded limits for now, these could be retrieved from a configuration
    return ResourceLimits(
        max_cpu_percent=90.0,
        max_memory_percent=85.0,
        max_gpu_percent=95.0,
        max_concurrent_tasks=100
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
        # Here we would update the limits in a configuration store
        # For now, we'll just return success
        return ResourceResponse(
            status="success",
            message="Resource limits updated successfully",
            data=limits.dict()
        )
    except Exception as e:
        logger.error(f"Error updating resource limits: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update resource limits: {str(e)}"
        )

@router.get("/system", response_model=Dict[str, Any])
async def get_system_info():
    """
    Get system information.

    Returns:
        System information
    """
    try:
        # Collect system information
        system_info = {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "physical_cores": psutil.cpu_count(logical=False),
            "total_cores": psutil.cpu_count(logical=True),
            "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat()
        }
        
        # Add disk information
        disk_info = []
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_info.append({
                    "device": partition.device,
                    "mountpoint": partition.mountpoint,
                    "filesystem": partition.fstype,
                    "total_gb": round(usage.total / (1024**3), 2),
                    "used_percent": usage.percent
                })
            except PermissionError:
                # This can happen if the disk isn't ready
                continue
        
        system_info["disk_info"] = disk_info
        
        return system_info
    except Exception as e:
        logger.error(f"Error getting system info: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system info: {str(e)}"
        )

@router.get("/llm/metrics", response_model=LLMResourceMetrics, summary="Get LLM resource metrics")
async def get_llm_resource_metrics():
    """
    Get metrics for all LLM resource pools.
    
    This endpoint returns metrics for all resource pools managed by the LLM Gateway,
    including session pools, token rate limits, and other resources.
    
    Returns:
        LLMResourceMetrics object containing detailed metrics for all resource pools
    
    Raises:
        HTTPException: If the LLM Gateway service is not available or an error occurs
    """
    try:
        # Get resource stats from the resource management layer
        resource_stats = get_resource_stats()
        
        # Process the stats into our response model
        pools = []
        total_resources = 0
        available_resources = 0
        in_use_resources = 0
        waiting_requests = 0
        unhealthy_pools = 0
        
        for pool_id, pool_stats in resource_stats.items():
            # Skip internal pools or aggregate stats
            if pool_id == "_aggregate" or pool_id.startswith("_"):
                continue
                
            provider_id = pool_stats.get("provider_id", None)
            resource_type = pool_stats.get("resource_type", "unknown")
            
            # Calculate counts
            pool_total = pool_stats.get("total", 0)
            pool_available = pool_stats.get("available", 0)
            pool_in_use = pool_stats.get("in_use", 0)
            pool_waiting = pool_stats.get("waiting_requests", 0)
            
            # Determine health status
            circuit_breaker = pool_stats.get("circuit_breaker", {})
            circuit_breaker_status = "open" if circuit_breaker.get("is_open", False) else "closed"
            
            health_check = pool_stats.get("health_check", {})
            health_status = health_check.get("status", "unknown")
            
            # Define the pool metrics
            pool = LLMResourcePool(
                pool_id=pool_id,
                provider_id=provider_id,
                resource_type=resource_type,
                total_resources=pool_total,
                available_resources=pool_available,
                in_use_resources=pool_in_use,
                waiting_requests=pool_waiting,
                circuit_breaker_status=circuit_breaker_status,
                health_status=health_status,
                metrics={
                    "average_wait_time_ms": pool_stats.get("average_wait_time_ms", 0),
                    "average_use_time_ms": pool_stats.get("average_use_time_ms", 0),
                    "peak_concurrent_use": pool_stats.get("peak_concurrent_use", 0),
                    "circuit_breaker": circuit_breaker,
                    "health_check": health_check,
                    "error_count": pool_stats.get("error_count", 0),
                    "success_count": pool_stats.get("success_count", 0),
                    "last_error": pool_stats.get("last_error", None)
                }
            )
            
            pools.append(pool)
            
            # Update totals
            total_resources += pool_total
            available_resources += pool_available
            in_use_resources += pool_in_use
            waiting_requests += pool_waiting
            
            # Check if pool is unhealthy
            if health_status != "healthy" or circuit_breaker_status == "open":
                unhealthy_pools += 1
        
        return LLMResourceMetrics(
            resource_pools=pools,
            total_pools=len(pools),
            total_resources=total_resources,
            available_resources=available_resources,
            in_use_resources=in_use_resources,
            waiting_requests=waiting_requests,
            unhealthy_pools=unhealthy_pools
        )
    
    except ResourceError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM resource management error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error getting LLM resource metrics: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get LLM resource metrics: {str(e)}"
        )

@router.get("/llm/usage", response_model=LLMUsageStats, summary="Get LLM usage statistics")
async def get_llm_usage_stats():
    """
    Get aggregated usage statistics for LLM services.
    
    This endpoint returns usage statistics aggregated across all LLM providers,
    including token usage, request counts, error rates, and response times.
    
    Returns:
        LLMUsageStats object containing aggregated usage statistics
        
    Raises:
        HTTPException: If the LLM Gateway service is not available or an error occurs
    """
    try:
        # Get the LLM gateway client service
        llm_gateway: Optional[LLMGatewayClient] = get_service("llm_gateway")
        if not llm_gateway:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="LLM Gateway service not available"
            )
        
        # Get usage statistics from the LLM Gateway
        usage_stats = await llm_gateway.get_usage_stats()
        
        # Process provider stats
        provider_stats = {}
        model_stats = {}
        
        # Aggregate metrics
        total_requests = 0
        total_tokens = 0
        total_errors = 0
        weighted_response_time = 0
        
        # Process provider-specific stats
        for provider_id, stats in usage_stats.get("providers", {}).items():
            provider_requests = stats.get("request_count", 0)
            provider_tokens = stats.get("token_count", 0)
            provider_errors = stats.get("error_count", 0)
            
            total_requests += provider_requests
            total_tokens += provider_tokens
            total_errors += provider_errors
            
            # Calculate weighted response time
            if provider_requests > 0 and stats.get("avg_response_time_ms") is not None:
                weighted_response_time += stats["avg_response_time_ms"] * provider_requests
            
            # Add to provider stats
            provider_stats[provider_id] = {
                "requests": provider_requests,
                "tokens": provider_tokens,
                "errors": provider_errors,
                "avg_response_time_ms": stats.get("avg_response_time_ms"),
                "tokens_per_request": provider_tokens / max(1, provider_requests),
                "error_rate": provider_errors / max(1, provider_requests) * 100
            }
        
        # Process model-specific stats
        for model_id, stats in usage_stats.get("models", {}).items():
            model_requests = stats.get("request_count", 0)
            model_tokens = stats.get("token_count", 0)
            model_errors = stats.get("error_count", 0)
            
            # Add to model stats
            model_stats[model_id] = {
                "requests": model_requests,
                "tokens": model_tokens,
                "errors": model_errors,
                "avg_response_time_ms": stats.get("avg_response_time_ms"),
                "tokens_per_request": model_tokens / max(1, model_requests),
                "error_rate": model_errors / max(1, model_requests) * 100
            }
        
        # Calculate final metrics
        average_response_time = weighted_response_time / max(1, total_requests)
        requests_per_minute = usage_stats.get("requests_per_minute", 0)
        tokens_per_minute = usage_stats.get("tokens_per_minute", 0)
        error_rate = (total_errors / max(1, total_requests)) * 100
        tokens_per_request = total_tokens / max(1, total_requests)
        
        return LLMUsageStats(
            total_requests=total_requests,
            total_tokens=total_tokens,
            total_errors=total_errors,
            average_response_time_ms=average_response_time,
            requests_per_minute=requests_per_minute,
            tokens_per_minute=tokens_per_minute,
            tokens_per_request=tokens_per_request,
            error_rate=error_rate,
            provider_stats=provider_stats,
            model_stats=model_stats
        )
    except ResourceError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM resource error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error getting LLM usage statistics: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get LLM usage statistics: {str(e)}"
        )

@router.get("/llm/providers", response_model=Dict[str, Any], summary="Get LLM provider health")
async def get_llm_provider_health():
    """
    Get health statistics for all LLM providers.
    
    This endpoint returns health information for all LLM providers,
    including connection status, circuit breaker states, and resource usage.
    
    Returns:
        Dictionary with provider health statistics
    
    Raises:
        HTTPException: If the LLM Gateway service is not available or an error occurs
    """
    try:
        # Get the LLM gateway client service
        llm_gateway: Optional[LLMGatewayClient] = get_service("llm_gateway")
        if not llm_gateway:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="LLM Gateway service not available"
            )
        
        # Get health check for all providers
        health_stats = await llm_gateway.health_check()
        return {
            "status": "success",
            "providers": health_stats
        }
    
    except ResourceError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM resource error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error getting LLM provider health: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get LLM provider health: {str(e)}"
        )

@router.get("/llm/pools/{provider_id}", response_model=List[LLMResourcePool], summary="Get provider resource pools")
async def get_provider_resource_pools(provider_id: str = Path(..., description="LLM provider ID")):
    """
    Get resource pools for a specific LLM provider.
    
    This endpoint returns details about all resource pools associated with
    the specified provider, such as connection pools and rate limiters.
    
    Args:
        provider_id: The ID of the LLM provider
        
    Returns:
        List of resource pools for the provider
        
    Raises:
        HTTPException: If the provider doesn't exist or an error occurs
    """
    try:
        # Get resource stats
        resource_stats = get_resource_stats()
        
        # Filter pools for the specified provider
        provider_pools = []
        
        for pool_id, pool_stats in resource_stats.items():
            if pool_stats.get("provider_id") == provider_id:
                # Skip internal pools
                if pool_id.startswith("_"):
                    continue
                    
                resource_type = pool_stats.get("resource_type", "unknown")
                
                # Calculate counts
                pool_total = pool_stats.get("total", 0)
                pool_available = pool_stats.get("available", 0)
                pool_in_use = pool_stats.get("in_use", 0)
                pool_waiting = pool_stats.get("waiting_requests", 0)
                
                # Determine health status
                circuit_breaker = pool_stats.get("circuit_breaker", {})
                circuit_breaker_status = "open" if circuit_breaker.get("is_open", False) else "closed"
                
                health_check = pool_stats.get("health_check", {})
                health_status = health_check.get("status", "unknown")
                
                # Create and add the pool
                pool = LLMResourcePool(
                    pool_id=pool_id,
                    provider_id=provider_id,
                    resource_type=resource_type,
                    total_resources=pool_total,
                    available_resources=pool_available,
                    in_use_resources=pool_in_use,
                    waiting_requests=pool_waiting,
                    circuit_breaker_status=circuit_breaker_status,
                    health_status=health_status,
                    metrics={
                        "average_wait_time_ms": pool_stats.get("average_wait_time_ms", 0),
                        "average_use_time_ms": pool_stats.get("average_use_time_ms", 0),
                        "peak_concurrent_use": pool_stats.get("peak_concurrent_use", 0),
                        "circuit_breaker": circuit_breaker,
                        "health_check": health_check,
                        "error_count": pool_stats.get("error_count", 0),
                        "success_count": pool_stats.get("success_count", 0),
                        "last_error": pool_stats.get("last_error", None)
                    }
                )
                
                provider_pools.append(pool)
        
        if not provider_pools:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No resource pools found for provider: {provider_id}"
            )
            
        return provider_pools
        
    except ResourceError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM resource error: {str(e)}"
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error getting provider resource pools: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get provider resource pools: {str(e)}"
        )

@router.get("/llm/pools/{provider_id}/{pool_id}", response_model=Dict[str, Any], summary="Get specific resource pool details")
async def get_resource_pool_details(
    provider_id: str = Path(..., description="LLM provider ID"),
    pool_id: str = Path(..., description="Resource pool ID")
):
    """
    Get detailed metrics for a specific resource pool.
    
    This endpoint returns comprehensive metrics and diagnostics for a specific 
    resource pool, including historical metrics, error details, and configuration.
    
    Args:
        provider_id: The ID of the LLM provider
        pool_id: The ID of the resource pool
        
    Returns:
        Detailed metrics and information about the resource pool
        
    Raises:
        HTTPException: If the provider or pool doesn't exist, or an error occurs
    """
    try:
        # Get resource stats
        resource_stats = get_resource_stats()
        
        # Check if the pool exists
        if pool_id not in resource_stats:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Resource pool not found: {pool_id}"
            )
            
        pool_stats = resource_stats[pool_id]
        
        # Verify the provider matches
        if pool_stats.get("provider_id") != provider_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Resource pool {pool_id} does not belong to provider {provider_id}"
            )
            
        # Return the full pool stats with some additional calculated metrics
        result = {
            **pool_stats,
            "utilization_percent": (pool_stats.get("in_use", 0) / pool_stats.get("total", 1)) * 100 
                if pool_stats.get("total", 0) > 0 else 0,
            "health_summary": "healthy" 
                if (pool_stats.get("circuit_breaker", {}).get("is_open", False) == False and 
                    pool_stats.get("health_check", {}).get("status", "unknown") == "healthy") 
                else "unhealthy"
        }
        
        return result
        
    except ResourceError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM resource error: {str(e)}"
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error getting resource pool details: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get resource pool details: {str(e)}"
        )