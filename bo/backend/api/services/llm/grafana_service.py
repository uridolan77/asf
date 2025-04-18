"""
Grafana service for LLM Gateway.

This service provides methods for integrating with Grafana,
including dashboard provisioning and alert management.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from config.grafana import (
    get_grafana_client,
    load_dashboard_template,
    provision_dashboard,
    provision_prometheus_datasource,
    provision_alert_rules,
    setup_grafana,
    DASHBOARD_IDS
)

logger = logging.getLogger(__name__)


class GrafanaService:
    """
    Service for integrating with Grafana.
    """
    
    async def provision_provider_dashboard(
        self,
        provider_id: str,
        provider_type: str,
        display_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Provision a dashboard for a specific provider.
        
        Args:
            provider_id: Provider ID
            provider_type: Provider type
            display_name: Provider display name
            
        Returns:
            Dashboard provisioning result or None if failed
        """
        try:
            # Set variables for template
            variables = {
                "provider": provider_id,
                "provider_type": provider_type,
                "display_name": display_name or provider_id
            }
            
            # Provision dashboard
            result = await provision_dashboard("mcp_provider_template", variables)
            
            if result:
                logger.info(f"Provisioned dashboard for provider {provider_id}")
            else:
                logger.error(f"Failed to provision dashboard for provider {provider_id}")
            
            return result
        except Exception as e:
            logger.error(f"Error provisioning dashboard for provider {provider_id}: {str(e)}")
            return None
    
    async def get_dashboard_url(self, dashboard_id: str) -> Optional[str]:
        """
        Get the URL for a dashboard.
        
        Args:
            dashboard_id: Dashboard ID
            
        Returns:
            Dashboard URL or None if not found
        """
        try:
            client = get_grafana_client()
            dashboard = await client.get_dashboard(dashboard_id)
            
            if not dashboard:
                return None
            
            return f"{client.base_url}/d/{dashboard_id}"
        except Exception as e:
            logger.error(f"Error getting dashboard URL for {dashboard_id}: {str(e)}")
            return None
    
    async def get_provider_dashboard_url(self, provider_id: str) -> Optional[str]:
        """
        Get the URL for a provider dashboard.
        
        Args:
            provider_id: Provider ID
            
        Returns:
            Dashboard URL or None if not found
        """
        dashboard_id = f"mcp-provider-{provider_id}"
        return await self.get_dashboard_url(dashboard_id)
    
    async def get_overview_dashboard_url(self) -> Optional[str]:
        """
        Get the URL for the overview dashboard.
        
        Returns:
            Dashboard URL or None if not found
        """
        return await self.get_dashboard_url(DASHBOARD_IDS["mcp_overview"])
    
    async def get_performance_dashboard_url(self) -> Optional[str]:
        """
        Get the URL for the performance dashboard.
        
        Returns:
            Dashboard URL or None if not found
        """
        return await self.get_dashboard_url(DASHBOARD_IDS["mcp_performance"])
    
    async def get_errors_dashboard_url(self) -> Optional[str]:
        """
        Get the URL for the errors dashboard.
        
        Returns:
            Dashboard URL or None if not found
        """
        return await self.get_dashboard_url(DASHBOARD_IDS["mcp_errors"])
    
    async def setup_grafana(self) -> bool:
        """
        Set up Grafana with datasources, dashboards, and alert rules.
        
        Returns:
            True if successful, False otherwise
        """
        return await setup_grafana()
    
    async def get_dashboard_urls(self) -> Dict[str, Optional[str]]:
        """
        Get URLs for all dashboards.
        
        Returns:
            Dictionary of dashboard IDs to URLs
        """
        urls = {}
        
        for name, dashboard_id in DASHBOARD_IDS.items():
            urls[name] = await self.get_dashboard_url(dashboard_id)
        
        return urls


# Global instance
_grafana_service = None


def get_grafana_service() -> GrafanaService:
    """
    Get the global Grafana service instance.
    
    Returns:
        GrafanaService instance
    """
    global _grafana_service
    
    if _grafana_service is None:
        _grafana_service = GrafanaService()
    
    return _grafana_service
