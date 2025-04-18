"""
Grafana integration API endpoints.

This module provides API endpoints for integrating with Grafana,
including dashboard provisioning and alert management.
"""

import logging
from typing import Dict, List, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Path, BackgroundTasks
from pydantic import BaseModel

from api.auth.dependencies import get_current_user
from models.user import User
from api.services.llm.grafana_service import get_grafana_service
from api.services.llm.gateway_service import get_llm_gateway_service as get_gateway_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/grafana", tags=["grafana"])


class DashboardURL(BaseModel):
    """Dashboard URL response model."""
    dashboard_id: str
    url: Optional[str] = None


class DashboardURLs(BaseModel):
    """Dashboard URLs response model."""
    dashboards: List[DashboardURL]


class ProvisionDashboardResponse(BaseModel):
    """Provision dashboard response model."""
    success: bool
    dashboard_id: Optional[str] = None
    url: Optional[str] = None
    message: Optional[str] = None


@router.get("/dashboards", response_model=DashboardURLs)
async def get_dashboard_urls(
    current_user: User = Depends(get_current_user)
):
    """
    Get URLs for all dashboards.

    Returns:
        Dashboard URLs
    """
    grafana_service = get_grafana_service()
    urls = await grafana_service.get_dashboard_urls()

    dashboards = []
    for name, url in urls.items():
        dashboards.append(DashboardURL(dashboard_id=name, url=url))

    return DashboardURLs(dashboards=dashboards)


@router.get("/dashboards/overview", response_model=DashboardURL)
async def get_overview_dashboard_url(
    current_user: User = Depends(get_current_user)
):
    """
    Get the URL for the overview dashboard.

    Returns:
        Dashboard URL
    """
    grafana_service = get_grafana_service()
    url = await grafana_service.get_overview_dashboard_url()

    return DashboardURL(dashboard_id="mcp_overview", url=url)


@router.get("/dashboards/performance", response_model=DashboardURL)
async def get_performance_dashboard_url(
    current_user: User = Depends(get_current_user)
):
    """
    Get the URL for the performance dashboard.

    Returns:
        Dashboard URL
    """
    grafana_service = get_grafana_service()
    url = await grafana_service.get_performance_dashboard_url()

    return DashboardURL(dashboard_id="mcp_performance", url=url)


@router.get("/dashboards/errors", response_model=DashboardURL)
async def get_errors_dashboard_url(
    current_user: User = Depends(get_current_user)
):
    """
    Get the URL for the errors dashboard.

    Returns:
        Dashboard URL
    """
    grafana_service = get_grafana_service()
    url = await grafana_service.get_errors_dashboard_url()

    return DashboardURL(dashboard_id="mcp_errors", url=url)


@router.get("/dashboards/provider/{provider_id}", response_model=DashboardURL)
async def get_provider_dashboard_url(
    provider_id: str = Path(..., description="Provider ID"),
    current_user: User = Depends(get_current_user)
):
    """
    Get the URL for a provider dashboard.

    Args:
        provider_id: Provider ID

    Returns:
        Dashboard URL
    """
    grafana_service = get_grafana_service()
    url = await grafana_service.get_provider_dashboard_url(provider_id)

    return DashboardURL(dashboard_id=f"mcp-provider-{provider_id}", url=url)


@router.post("/dashboards/provider/{provider_id}", response_model=ProvisionDashboardResponse)
async def provision_provider_dashboard(
    provider_id: str = Path(..., description="Provider ID"),
    current_user: User = Depends(get_current_user)
):
    """
    Provision a dashboard for a specific provider.

    Args:
        provider_id: Provider ID

    Returns:
        Provision result
    """
    gateway_service = get_gateway_service()
    grafana_service = get_grafana_service()

    # Get provider details
    provider = await gateway_service.get_provider(provider_id)

    if not provider:
        raise HTTPException(status_code=404, detail=f"Provider {provider_id} not found")

    # Provision dashboard
    result = await grafana_service.provision_provider_dashboard(
        provider_id=provider_id,
        provider_type=provider.get("provider_type", "unknown"),
        display_name=provider.get("display_name")
    )

    if not result:
        return ProvisionDashboardResponse(
            success=False,
            message=f"Failed to provision dashboard for provider {provider_id}"
        )

    # Get dashboard URL
    url = await grafana_service.get_provider_dashboard_url(provider_id)

    return ProvisionDashboardResponse(
        success=True,
        dashboard_id=f"mcp-provider-{provider_id}",
        url=url,
        message=f"Dashboard provisioned for provider {provider_id}"
    )


@router.post("/setup", response_model=Dict[str, Any])
async def setup_grafana(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """
    Set up Grafana with datasources, dashboards, and alert rules.

    Returns:
        Setup result
    """
    grafana_service = get_grafana_service()

    # Run setup in background
    background_tasks.add_task(grafana_service.setup_grafana)

    return {
        "message": "Grafana setup started in background",
        "status": "pending"
    }
