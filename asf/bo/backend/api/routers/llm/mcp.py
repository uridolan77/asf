"""
MCP (Model Context Protocol) Router

This module provides FastAPI routes for managing MCP providers,
configurations, and monitoring.
"""

from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, Body, status
from pydantic import BaseModel, Field
from datetime import datetime

# Import dependencies from the new consolidated structure
try:
    # Updated imports for the consolidated LLM Gateway structure
    from asf.medical.llm_gateway.mcp import (
        Transport, TransportConfig, 
        StdioConfig, GrpcConfig, HttpConfig,
        TransportFactory
    )
    from asf.medical.llm_gateway.providers import MCPProvider
    from asf.medical.llm_gateway.resilience import CircuitBreaker
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# Import services
from ...services.llm.llm_service import get_llm_service
from ...services.llm.gateway_service import get_llm_gateway_service

# Import user dependencies
from ...dependencies import get_current_user, get_admin_user
from models.user import User  # Fix: Import User from correct location

router = APIRouter(prefix="/mcp", tags=["mcp"])

# --- Models ---

class MCPTransportConfig(BaseModel):
    """Base model for MCP transport configuration"""
    transport_type: str = Field(..., description="Transport type (stdio, grpc, http)")


class StdioConfig(MCPTransportConfig):
    """Configuration for stdio transport"""
    transport_type: str = "stdio"
    command: str = Field(..., description="Command to execute")
    args: List[str] = Field(default_factory=list, description="Command arguments")
    env: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    cwd: Optional[str] = Field(None, description="Working directory")


class GrpcConfig(MCPTransportConfig):
    """Configuration for gRPC transport"""
    transport_type: str = "grpc"
    endpoint: str = Field(..., description="gRPC server endpoint (host:port)")
    use_tls: bool = Field(default=False, description="Whether to use TLS")
    ca_cert: Optional[str] = Field(None, description="CA certificate file for TLS")
    client_cert: Optional[str] = Field(None, description="Client certificate file for mTLS")
    client_key: Optional[str] = Field(None, description="Client key file for mTLS")
    metadata: Dict[str, str] = Field(default_factory=dict, description="Additional metadata for requests")


class HttpConfig(MCPTransportConfig):
    """Configuration for HTTP transport"""
    transport_type: str = "http"
    base_url: str = Field(..., description="Base URL for HTTP requests")
    headers: Dict[str, str] = Field(default_factory=dict, description="HTTP headers")
    verify_ssl: bool = Field(default=True, description="Whether to verify SSL certificates")


class MCPProviderConfig(BaseModel):
    """Configuration for an MCP provider"""
    provider_id: str = Field(..., description="Unique identifier for the provider")
    display_name: str = Field(..., description="Display name for the provider")
    transport_config: MCPTransportConfig = Field(..., description="Transport configuration")
    enable_streaming: bool = Field(default=True, description="Whether to enable streaming")
    timeout_seconds: int = Field(default=60, description="Operation timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum number of retries")
    models: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Models supported by the provider")
    enabled: bool = Field(default=True, description="Whether the provider is enabled")


class MCPProviderStatus(BaseModel):
    """Status of an MCP provider"""
    provider_id: str
    display_name: str
    status: str
    transport_type: str
    checked_at: str
    message: Optional[str] = None
    circuit_breaker: Dict[str, Any] = Field(default_factory=dict)
    models: List[str] = Field(default_factory=list)


class MCPProviderTestResult(BaseModel):
    """Result of testing an MCP provider"""
    provider_id: str
    success: bool
    message: str
    latency_ms: float
    tested_at: str


class MCPUsageStats(BaseModel):
    """Usage statistics for MCP providers"""
    provider_id: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_tokens: int
    average_latency_ms: float
    period_start: str
    period_end: str


# --- Routes ---

@router.get("/info")
async def get_mcp_info(current_user: User = Depends(get_current_user)):
    """
    Get information about the MCP implementation.

    Returns:
        Information about the MCP implementation
    """
    return {
        "name": "Model Context Protocol (MCP)",
        "version": "1.0",
        "description": "A comprehensive, production-grade MCP gateway provider implementation with advanced features for scalability, reliability, and observability.",
        "features": [
            "Robust Connection Management",
            "Streaming & Non-Streaming Support",
            "Advanced Resilience",
            "Comprehensive Observability",
            "Configuration Management",
            "Transport Flexibility"
        ],
        "transport_types": [
            "stdio",
            "gRPC",
            "HTTP/REST"
        ],
        "documentation_url": "https://github.com/anthropics/anthropic-sdk-typescript/tree/main/packages/mcp"
    }


@router.get("/providers", response_model=List[MCPProviderStatus])
async def get_mcp_providers(current_user: User = Depends(get_current_user)):
    """
    Get all registered MCP providers.

    Returns:
        List of MCP providers with their status
    """
    try:
        llm_gateway = get_llm_gateway_service()
        providers = await llm_gateway.get_mcp_providers()

        return providers
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get MCP providers: {str(e)}"
        )


@router.post("/providers", status_code=status.HTTP_201_CREATED)
async def register_mcp_provider(
    config: MCPProviderConfig = Body(...),
    current_user: User = Depends(get_admin_user)
):
    """
    Register a new MCP provider.

    Args:
        config: Configuration for the MCP provider

    Returns:
        The registered provider
    """
    try:
        llm_gateway = get_llm_gateway_service()
        provider = await llm_gateway.register_mcp_provider(config)

        return {
            "success": True,
            "message": f"MCP provider '{config.provider_id}' registered successfully",
            "provider": provider
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register MCP provider: {str(e)}"
        )


@router.put("/providers/{provider_id}")
async def update_mcp_provider(
    provider_id: str,
    config: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_admin_user)
):
    """
    Update an existing MCP provider.

    Args:
        provider_id: ID of the provider to update
        config: Updated configuration for the provider

    Returns:
        The updated provider
    """
    try:
        llm_gateway = get_llm_gateway_service()
        provider = await llm_gateway.update_mcp_provider(provider_id, config)

        return {
            "success": True,
            "message": f"MCP provider '{provider_id}' updated successfully",
            "provider": provider
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update MCP provider: {str(e)}"
        )


@router.delete("/providers/{provider_id}")
async def delete_mcp_provider(
    provider_id: str,
    current_user: User = Depends(get_admin_user)
):
    """
    Delete an MCP provider.

    Args:
        provider_id: ID of the provider to delete

    Returns:
        Confirmation of deletion
    """
    try:
        llm_gateway = get_llm_gateway_service()
        await llm_gateway.delete_mcp_provider(provider_id)

        return {
            "success": True,
            "message": f"MCP provider '{provider_id}' deleted successfully",
            "provider_id": provider_id,
            "deleted_at": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete MCP provider: {str(e)}"
        )


@router.post("/providers/{provider_id}/test", response_model=MCPProviderTestResult)
async def test_mcp_provider(
    provider_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Test connection to an MCP provider.

    Args:
        provider_id: ID of the provider to test

    Returns:
        Test result
    """
    try:
        start_time = datetime.now()
        llm_gateway = get_llm_gateway_service()
        result = await llm_gateway.test_mcp_provider(provider_id)
        end_time = datetime.now()

        latency_ms = (end_time - start_time).total_seconds() * 1000

        return {
            "provider_id": provider_id,
            "success": result["success"],
            "message": result["message"],
            "latency_ms": latency_ms,
            "tested_at": start_time.isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to test MCP provider '{provider_id}': {str(e)}"
        )


@router.get("/providers/{provider_id}/status", response_model=MCPProviderStatus)
async def get_mcp_provider_status(
    provider_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get status of an MCP provider.

    Args:
        provider_id: ID of the provider

    Returns:
        Provider status
    """
    try:
        llm_gateway = get_llm_gateway_service()
        status = await llm_gateway.get_mcp_provider_status(provider_id)

        return status
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get status for MCP provider '{provider_id}': {str(e)}"
        )


@router.get("/providers/{provider_id}/usage", response_model=MCPUsageStats)
async def get_mcp_provider_usage(
    provider_id: str,
    period: str = "day",  # day, week, month
    current_user: User = Depends(get_current_user)
):
    """
    Get usage statistics for an MCP provider.

    Args:
        provider_id: ID of the provider
        period: Time period for statistics (day, week, month)

    Returns:
        Usage statistics
    """
    try:
        llm_gateway = get_llm_gateway_service()
        usage = await llm_gateway.get_mcp_provider_usage(provider_id, period)

        return usage
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get usage statistics for MCP provider '{provider_id}': {str(e)}"
        )


@router.post("/generate")
async def generate_with_mcp(
    request: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_user)
):
    """
    Generate a response using an MCP provider.

    Args:
        request: Generation request

    Returns:
        Generated response
    """
    try:
        llm_gateway = get_llm_gateway_service()
        response = await llm_gateway.generate_with_mcp(request)

        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate response: {str(e)}"
        )
