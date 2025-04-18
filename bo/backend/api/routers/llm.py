"""
LLM Gateway API endpoints.

This module provides API endpoints for interacting with the LLM Gateway,
including provider management, request handling, and monitoring.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Path, BackgroundTasks
from pydantic import BaseModel, Field

from api.auth.dependencies import get_current_user
from api.models.user import User
from api.services.llm.gateway_service import get_gateway_service
from api.services.llm.metrics_service import get_metrics_service
from api.services.llm.prometheus_service import get_prometheus_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/llm", tags=["llm"])


class ProviderStatus(BaseModel):
    """Provider status response model."""
    provider_id: str
    status: str
    provider_type: str
    checked_at: str
    message: str
    circuit_breaker: Dict[str, Any]
    models: Optional[List[str]] = None
    session_count: Optional[int] = None


class ProviderMetrics(BaseModel):
    """Provider metrics response model."""
    provider_id: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_tokens: int
    input_tokens: int
    output_tokens: int
    average_latency_ms: float
    period_start: str
    period_end: str


class CircuitBreakerHistoryEntry(BaseModel):
    """Circuit breaker history entry model."""
    timestamp: str
    state: str
    failure_count: int
    recovery_time: Optional[str] = None


class ProviderConfig(BaseModel):
    """Provider configuration model."""
    provider_id: str
    provider_type: str
    display_name: str
    description: Optional[str] = None
    models: List[str]
    connection_params: Dict[str, Any]
    enabled: bool = True


@router.get("/providers", response_model=List[Dict[str, Any]])
async def get_providers(
    current_user: User = Depends(get_current_user)
):
    """
    Get all available LLM providers.
    
    Returns:
        List of provider information
    """
    gateway_service = get_gateway_service()
    providers = await gateway_service.get_providers()
    return providers


@router.get("/providers/{provider_id}", response_model=Dict[str, Any])
async def get_provider(
    provider_id: str = Path(..., description="Provider ID"),
    current_user: User = Depends(get_current_user)
):
    """
    Get information about a specific LLM provider.
    
    Args:
        provider_id: Provider ID
        
    Returns:
        Provider information
    """
    gateway_service = get_gateway_service()
    provider = await gateway_service.get_provider(provider_id)
    
    if not provider:
        raise HTTPException(status_code=404, detail=f"Provider {provider_id} not found")
    
    return provider


@router.get("/providers/{provider_id}/status", response_model=ProviderStatus)
async def get_provider_status(
    provider_id: str = Path(..., description="Provider ID"),
    current_user: User = Depends(get_current_user)
):
    """
    Get the current status of an LLM provider.
    
    Args:
        provider_id: Provider ID
        
    Returns:
        Provider status
    """
    gateway_service = get_gateway_service()
    status = await gateway_service.get_provider_status(provider_id)
    
    if not status:
        raise HTTPException(status_code=404, detail=f"Provider {provider_id} not found")
    
    return status


@router.get("/providers/{provider_id}/metrics", response_model=ProviderMetrics)
async def get_provider_metrics(
    provider_id: str = Path(..., description="Provider ID"),
    period: str = Query("day", description="Time period (hour, day, week, month)"),
    current_user: User = Depends(get_current_user)
):
    """
    Get usage metrics for an LLM provider.
    
    Args:
        provider_id: Provider ID
        period: Time period (hour, day, week, month)
        
    Returns:
        Provider metrics
    """
    metrics_service = get_metrics_service()
    metrics = await metrics_service.get_provider_metrics(provider_id, period)
    
    if not metrics:
        raise HTTPException(status_code=404, detail=f"Metrics for provider {provider_id} not found")
    
    return metrics


@router.get("/providers/{provider_id}/circuit-breaker/history", response_model=List[CircuitBreakerHistoryEntry])
async def get_circuit_breaker_history(
    provider_id: str = Path(..., description="Provider ID"),
    period: str = Query("day", description="Time period (hour, day, week, month)"),
    current_user: User = Depends(get_current_user)
):
    """
    Get circuit breaker history for an LLM provider.
    
    Args:
        provider_id: Provider ID
        period: Time period (hour, day, week, month)
        
    Returns:
        Circuit breaker history
    """
    # Calculate time range based on period
    now = datetime.utcnow()
    if period == "hour":
        start_time = now - timedelta(hours=1)
    elif period == "day":
        start_time = now - timedelta(days=1)
    elif period == "week":
        start_time = now - timedelta(weeks=1)
    elif period == "month":
        start_time = now - timedelta(days=30)
    else:
        start_time = now - timedelta(days=1)  # Default to day
    
    # Get Prometheus service
    prometheus_service = get_prometheus_service()
    
    # Query circuit breaker history
    history = await prometheus_service.get_circuit_breaker_history(
        provider_id=provider_id,
        start_time=start_time,
        end_time=now
    )
    
    return history


@router.post("/providers/{provider_id}/reset-circuit-breaker")
async def reset_circuit_breaker(
    provider_id: str = Path(..., description="Provider ID"),
    current_user: User = Depends(get_current_user)
):
    """
    Reset the circuit breaker for an LLM provider.
    
    Args:
        provider_id: Provider ID
        
    Returns:
        Success message
    """
    gateway_service = get_gateway_service()
    success = await gateway_service.reset_circuit_breaker(provider_id)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Provider {provider_id} not found")
    
    return {"message": f"Circuit breaker for provider {provider_id} has been reset"}


@router.post("/providers/{provider_id}/refresh-sessions")
async def refresh_sessions(
    provider_id: str = Path(..., description="Provider ID"),
    current_user: User = Depends(get_current_user)
):
    """
    Refresh the session pool for an LLM provider.
    
    Args:
        provider_id: Provider ID
        
    Returns:
        Success message
    """
    gateway_service = get_gateway_service()
    success = await gateway_service.refresh_sessions(provider_id)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Provider {provider_id} not found")
    
    return {"message": f"Session pool for provider {provider_id} has been refreshed"}


@router.get("/providers/{provider_id}/connection-pool", response_model=Dict[str, Any])
async def get_connection_pool_stats(
    provider_id: str = Path(..., description="Provider ID"),
    current_user: User = Depends(get_current_user)
):
    """
    Get connection pool statistics for an LLM provider.
    
    Args:
        provider_id: Provider ID
        
    Returns:
        Connection pool statistics
    """
    gateway_service = get_gateway_service()
    stats = await gateway_service.get_connection_pool_stats(provider_id)
    
    if not stats:
        raise HTTPException(status_code=404, detail=f"Connection pool stats for provider {provider_id} not found")
    
    return stats


@router.get("/metrics/summary", response_model=Dict[str, Any])
async def get_metrics_summary(
    period: str = Query("day", description="Time period (hour, day, week, month)"),
    current_user: User = Depends(get_current_user)
):
    """
    Get a summary of LLM Gateway metrics.
    
    Args:
        period: Time period (hour, day, week, month)
        
    Returns:
        Metrics summary
    """
    metrics_service = get_metrics_service()
    summary = await metrics_service.get_metrics_summary(period)
    
    return summary


@router.get("/metrics/errors", response_model=Dict[str, Any])
async def get_error_metrics(
    period: str = Query("day", description="Time period (hour, day, week, month)"),
    current_user: User = Depends(get_current_user)
):
    """
    Get error metrics for the LLM Gateway.
    
    Args:
        period: Time period (hour, day, week, month)
        
    Returns:
        Error metrics
    """
    metrics_service = get_metrics_service()
    errors = await metrics_service.get_error_metrics(period)
    
    return errors


@router.get("/metrics/latency", response_model=Dict[str, Any])
async def get_latency_metrics(
    period: str = Query("day", description="Time period (hour, day, week, month)"),
    current_user: User = Depends(get_current_user)
):
    """
    Get latency metrics for the LLM Gateway.
    
    Args:
        period: Time period (hour, day, week, month)
        
    Returns:
        Latency metrics
    """
    metrics_service = get_metrics_service()
    latency = await metrics_service.get_latency_metrics(period)
    
    return latency


@router.get("/metrics/tokens", response_model=Dict[str, Any])
async def get_token_metrics(
    period: str = Query("day", description="Time period (hour, day, week, month)"),
    current_user: User = Depends(get_current_user)
):
    """
    Get token usage metrics for the LLM Gateway.
    
    Args:
        period: Time period (hour, day, week, month)
        
    Returns:
        Token usage metrics
    """
    metrics_service = get_metrics_service()
    tokens = await metrics_service.get_token_metrics(period)
    
    return tokens


@router.get("/websocket/stats", response_model=Dict[str, Any])
async def get_websocket_stats(
    current_user: User = Depends(get_current_user)
):
    """
    Get WebSocket connection statistics.
    
    Returns:
        WebSocket statistics
    """
    from api.websockets.mcp_manager import mcp_manager
    
    stats = mcp_manager.get_connection_stats()
    
    return stats
