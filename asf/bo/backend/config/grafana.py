"""
Grafana configuration and utilities.

This module provides configuration and utilities for integrating with Grafana,
including dashboard provisioning, API access, and alert management.
"""

import os
import logging
import json
from typing import Dict, List, Any, Optional
import httpx
from datetime import datetime

logger = logging.getLogger(__name__)

# Grafana configuration
GRAFANA_URL = os.getenv("GRAFANA_URL", "http://localhost:3000")
GRAFANA_API_KEY = os.getenv("GRAFANA_API_KEY", "")
GRAFANA_USERNAME = os.getenv("GRAFANA_USERNAME", "admin")
GRAFANA_PASSWORD = os.getenv("GRAFANA_PASSWORD", "admin")
GRAFANA_ORG_ID = os.getenv("GRAFANA_ORG_ID", "1")

# Dashboard templates directory
DASHBOARD_TEMPLATES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dashboards"
)

# Dashboard IDs
DASHBOARD_IDS = {
    "mcp_overview": "mcp-overview",
    "mcp_provider": "mcp-provider",
    "mcp_performance": "mcp-performance",
    "mcp_errors": "mcp-errors",
    "mcp_circuit_breaker": "mcp-circuit-breaker",
    "mcp_connection_pool": "mcp-connection-pool",
    "mcp_websocket": "mcp-websocket",
    "system_overview": "system-overview"
}

# Alert rule IDs
ALERT_RULE_IDS = {
    "high_error_rate": "mcp-high-error-rate",
    "circuit_breaker_open": "mcp-circuit-breaker-open",
    "high_latency": "mcp-high-latency",
    "connection_pool_exhausted": "mcp-connection-pool-exhausted",
    "websocket_disconnected": "mcp-websocket-disconnected"
}


class GrafanaClient:
    """
    Client for interacting with Grafana API.
    """
    
    def __init__(
        self,
        base_url: str = GRAFANA_URL,
        api_key: str = GRAFANA_API_KEY,
        username: str = GRAFANA_USERNAME,
        password: str = GRAFANA_PASSWORD,
        org_id: str = GRAFANA_ORG_ID
    ):
        """
        Initialize the Grafana client.
        
        Args:
            base_url: Grafana base URL
            api_key: Grafana API key
            username: Grafana username (used if API key is not provided)
            password: Grafana password (used if API key is not provided)
            org_id: Grafana organization ID
        """
        self.base_url = base_url
        self.api_key = api_key
        self.username = username
        self.password = password
        self.org_id = org_id
        self.token = None
    
    async def _get_headers(self) -> Dict[str, str]:
        """
        Get headers for Grafana API requests.
        
        Returns:
            Headers dictionary
        """
        if self.api_key:
            return {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        
        if not self.token:
            # Get token using basic auth
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/auth/login",
                    json={
                        "username": self.username,
                        "password": self.password
                    }
                )
                
                if response.status_code != 200:
                    logger.error(f"Failed to authenticate with Grafana: {response.text}")
                    return {"Content-Type": "application/json"}
                
                self.token = response.json().get("token")
        
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
    
    async def get_dashboard(self, uid: str) -> Optional[Dict[str, Any]]:
        """
        Get a dashboard by UID.
        
        Args:
            uid: Dashboard UID
            
        Returns:
            Dashboard data or None if not found
        """
        try:
            headers = await self._get_headers()
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/api/dashboards/uid/{uid}",
                    headers=headers
                )
                
                if response.status_code == 404:
                    return None
                
                if response.status_code != 200:
                    logger.error(f"Failed to get dashboard {uid}: {response.text}")
                    return None
                
                return response.json()
        except Exception as e:
            logger.error(f"Error getting dashboard {uid}: {str(e)}")
            return None
    
    async def create_or_update_dashboard(
        self,
        dashboard: Dict[str, Any],
        folder_id: Optional[int] = None,
        overwrite: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Create or update a dashboard.
        
        Args:
            dashboard: Dashboard definition
            folder_id: Folder ID to store the dashboard in
            overwrite: Whether to overwrite existing dashboard
            
        Returns:
            Result of the operation or None if failed
        """
        try:
            headers = await self._get_headers()
            
            # Prepare payload
            payload = {
                "dashboard": dashboard,
                "overwrite": overwrite
            }
            
            if folder_id is not None:
                payload["folderId"] = folder_id
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/dashboards/db",
                    headers=headers,
                    json=payload
                )
                
                if response.status_code not in (200, 201):
                    logger.error(f"Failed to create/update dashboard: {response.text}")
                    return None
                
                return response.json()
        except Exception as e:
            logger.error(f"Error creating/updating dashboard: {str(e)}")
            return None
    
    async def get_alert_rules(self) -> List[Dict[str, Any]]:
        """
        Get all alert rules.
        
        Returns:
            List of alert rules
        """
        try:
            headers = await self._get_headers()
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/api/ruler/grafana/api/v1/rules",
                    headers=headers
                )
                
                if response.status_code != 200:
                    logger.error(f"Failed to get alert rules: {response.text}")
                    return []
                
                return response.json()
        except Exception as e:
            logger.error(f"Error getting alert rules: {str(e)}")
            return []
    
    async def create_or_update_alert_rule(
        self,
        rule: Dict[str, Any],
        namespace: str = "MCP",
        group: str = "mcp_alerts"
    ) -> bool:
        """
        Create or update an alert rule.
        
        Args:
            rule: Alert rule definition
            namespace: Rule namespace
            group: Rule group
            
        Returns:
            True if successful, False otherwise
        """
        try:
            headers = await self._get_headers()
            
            # Check if rule group exists
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/api/ruler/grafana/api/v1/rules/{namespace}",
                    headers=headers
                )
                
                if response.status_code == 404:
                    # Create new group
                    payload = {
                        group: [rule]
                    }
                    
                    response = await client.post(
                        f"{self.base_url}/api/ruler/grafana/api/v1/rules/{namespace}",
                        headers=headers,
                        json=payload
                    )
                else:
                    # Update existing group
                    groups = response.json()
                    
                    if group in groups:
                        # Update existing rule in group
                        rules = groups[group]
                        
                        # Find rule by name
                        rule_index = -1
                        for i, r in enumerate(rules):
                            if r.get("name") == rule.get("name"):
                                rule_index = i
                                break
                        
                        if rule_index >= 0:
                            rules[rule_index] = rule
                        else:
                            rules.append(rule)
                        
                        payload = {
                            group: rules
                        }
                    else:
                        # Create new group
                        payload = {
                            group: [rule]
                        }
                    
                    response = await client.post(
                        f"{self.base_url}/api/ruler/grafana/api/v1/rules/{namespace}",
                        headers=headers,
                        json=payload
                    )
                
                if response.status_code not in (200, 201, 202):
                    logger.error(f"Failed to create/update alert rule: {response.text}")
                    return False
                
                return True
        except Exception as e:
            logger.error(f"Error creating/updating alert rule: {str(e)}")
            return False
    
    async def get_datasources(self) -> List[Dict[str, Any]]:
        """
        Get all datasources.
        
        Returns:
            List of datasources
        """
        try:
            headers = await self._get_headers()
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/api/datasources",
                    headers=headers
                )
                
                if response.status_code != 200:
                    logger.error(f"Failed to get datasources: {response.text}")
                    return []
                
                return response.json()
        except Exception as e:
            logger.error(f"Error getting datasources: {str(e)}")
            return []
    
    async def create_or_update_datasource(self, datasource: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Create or update a datasource.
        
        Args:
            datasource: Datasource definition
            
        Returns:
            Result of the operation or None if failed
        """
        try:
            headers = await self._get_headers()
            
            # Check if datasource exists
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/api/datasources/name/{datasource['name']}",
                    headers=headers
                )
                
                if response.status_code == 404:
                    # Create new datasource
                    response = await client.post(
                        f"{self.base_url}/api/datasources",
                        headers=headers,
                        json=datasource
                    )
                else:
                    # Update existing datasource
                    existing = response.json()
                    datasource["id"] = existing["id"]
                    
                    response = await client.put(
                        f"{self.base_url}/api/datasources/{existing['id']}",
                        headers=headers,
                        json=datasource
                    )
                
                if response.status_code not in (200, 201):
                    logger.error(f"Failed to create/update datasource: {response.text}")
                    return None
                
                return response.json()
        except Exception as e:
            logger.error(f"Error creating/updating datasource: {str(e)}")
            return None


# Create global client instance
_grafana_client = None


def get_grafana_client() -> GrafanaClient:
    """
    Get the global Grafana client instance.
    
    Returns:
        GrafanaClient instance
    """
    global _grafana_client
    
    if _grafana_client is None:
        _grafana_client = GrafanaClient()
    
    return _grafana_client


async def load_dashboard_template(template_name: str) -> Optional[Dict[str, Any]]:
    """
    Load a dashboard template from file.
    
    Args:
        template_name: Template name (without .json extension)
        
    Returns:
        Dashboard template or None if not found
    """
    try:
        template_path = os.path.join(DASHBOARD_TEMPLATES_DIR, f"{template_name}.json")
        
        if not os.path.exists(template_path):
            logger.error(f"Dashboard template not found: {template_path}")
            return None
        
        with open(template_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading dashboard template {template_name}: {str(e)}")
        return None


async def provision_dashboard(
    template_name: str,
    variables: Optional[Dict[str, Any]] = None,
    folder_id: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """
    Provision a dashboard from a template.
    
    Args:
        template_name: Template name (without .json extension)
        variables: Variables to replace in the template
        folder_id: Folder ID to store the dashboard in
        
    Returns:
        Result of the operation or None if failed
    """
    try:
        # Load template
        template = await load_dashboard_template(template_name)
        
        if not template:
            return None
        
        # Replace variables
        if variables:
            template_str = json.dumps(template)
            
            for key, value in variables.items():
                template_str = template_str.replace(f"${{{key}}}", str(value))
            
            template = json.loads(template_str)
        
        # Set dashboard metadata
        if "id" in template:
            del template["id"]
        
        template["version"] = 1
        template["refresh"] = "1m"
        template["time"] = {
            "from": "now-6h",
            "to": "now"
        }
        
        # Create or update dashboard
        client = get_grafana_client()
        result = await client.create_or_update_dashboard(template, folder_id)
        
        return result
    except Exception as e:
        logger.error(f"Error provisioning dashboard {template_name}: {str(e)}")
        return None


async def provision_prometheus_datasource() -> Optional[Dict[str, Any]]:
    """
    Provision Prometheus datasource.
    
    Returns:
        Result of the operation or None if failed
    """
    try:
        # Define datasource
        datasource = {
            "name": "Prometheus",
            "type": "prometheus",
            "url": os.getenv("PROMETHEUS_URL", "http://localhost:9090"),
            "access": "proxy",
            "isDefault": True,
            "jsonData": {
                "httpMethod": "POST",
                "timeInterval": "15s"
            }
        }
        
        # Create or update datasource
        client = get_grafana_client()
        result = await client.create_or_update_datasource(datasource)
        
        return result
    except Exception as e:
        logger.error(f"Error provisioning Prometheus datasource: {str(e)}")
        return None


async def provision_alert_rules() -> bool:
    """
    Provision alert rules.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        client = get_grafana_client()
        
        # High error rate alert
        high_error_rate_rule = {
            "name": "MCP High Error Rate",
            "uid": ALERT_RULE_IDS["high_error_rate"],
            "condition": "B",
            "data": [
                {
                    "refId": "A",
                    "queryType": "range",
                    "relativeTimeRange": {
                        "from": 600,
                        "to": 0
                    },
                    "datasourceUid": "P8E80F9AEF21F6940",
                    "model": {
                        "expr": "sum(rate(llm_gateway_errors_total[5m])) / sum(rate(llm_gateway_requests_total[5m])) * 100",
                        "instant": False,
                        "intervalMs": 1000,
                        "maxDataPoints": 43200,
                        "refId": "A"
                    }
                },
                {
                    "refId": "B",
                    "queryType": "threshold",
                    "relativeTimeRange": {
                        "from": 600,
                        "to": 0
                    },
                    "datasourceUid": "__expr__",
                    "model": {
                        "conditions": [
                            {
                                "evaluator": {
                                    "params": [
                                        5
                                    ],
                                    "type": "gt"
                                },
                                "operator": {
                                    "type": "and"
                                },
                                "query": {
                                    "params": [
                                        "A"
                                    ]
                                },
                                "reducer": {
                                    "params": [],
                                    "type": "avg"
                                },
                                "type": "query"
                            }
                        ],
                        "refId": "B"
                    }
                }
            ],
            "noDataState": "NoData",
            "execErrState": "Error",
            "for": "5m",
            "annotations": {
                "description": "MCP error rate is above 5% for 5 minutes",
                "summary": "High MCP error rate"
            },
            "labels": {
                "severity": "warning",
                "service": "mcp"
            }
        }
        
        # Circuit breaker open alert
        circuit_breaker_rule = {
            "name": "MCP Circuit Breaker Open",
            "uid": ALERT_RULE_IDS["circuit_breaker_open"],
            "condition": "B",
            "data": [
                {
                    "refId": "A",
                    "queryType": "range",
                    "relativeTimeRange": {
                        "from": 600,
                        "to": 0
                    },
                    "datasourceUid": "P8E80F9AEF21F6940",
                    "model": {
                        "expr": "llm_gateway_circuit_breaker_state > 0",
                        "instant": False,
                        "intervalMs": 1000,
                        "maxDataPoints": 43200,
                        "refId": "A"
                    }
                },
                {
                    "refId": "B",
                    "queryType": "threshold",
                    "relativeTimeRange": {
                        "from": 600,
                        "to": 0
                    },
                    "datasourceUid": "__expr__",
                    "model": {
                        "conditions": [
                            {
                                "evaluator": {
                                    "params": [
                                        0
                                    ],
                                    "type": "gt"
                                },
                                "operator": {
                                    "type": "and"
                                },
                                "query": {
                                    "params": [
                                        "A"
                                    ]
                                },
                                "reducer": {
                                    "params": [],
                                    "type": "last"
                                },
                                "type": "query"
                            }
                        ],
                        "refId": "B"
                    }
                }
            ],
            "noDataState": "NoData",
            "execErrState": "Error",
            "for": "1m",
            "annotations": {
                "description": "MCP circuit breaker is open",
                "summary": "MCP circuit breaker open"
            },
            "labels": {
                "severity": "critical",
                "service": "mcp"
            }
        }
        
        # High latency alert
        high_latency_rule = {
            "name": "MCP High Latency",
            "uid": ALERT_RULE_IDS["high_latency"],
            "condition": "B",
            "data": [
                {
                    "refId": "A",
                    "queryType": "range",
                    "relativeTimeRange": {
                        "from": 600,
                        "to": 0
                    },
                    "datasourceUid": "P8E80F9AEF21F6940",
                    "model": {
                        "expr": "histogram_quantile(0.95, sum(rate(llm_gateway_request_duration_seconds_bucket[5m])) by (le))",
                        "instant": False,
                        "intervalMs": 1000,
                        "maxDataPoints": 43200,
                        "refId": "A"
                    }
                },
                {
                    "refId": "B",
                    "queryType": "threshold",
                    "relativeTimeRange": {
                        "from": 600,
                        "to": 0
                    },
                    "datasourceUid": "__expr__",
                    "model": {
                        "conditions": [
                            {
                                "evaluator": {
                                    "params": [
                                        10
                                    ],
                                    "type": "gt"
                                },
                                "operator": {
                                    "type": "and"
                                },
                                "query": {
                                    "params": [
                                        "A"
                                    ]
                                },
                                "reducer": {
                                    "params": [],
                                    "type": "avg"
                                },
                                "type": "query"
                            }
                        ],
                        "refId": "B"
                    }
                }
            ],
            "noDataState": "NoData",
            "execErrState": "Error",
            "for": "5m",
            "annotations": {
                "description": "MCP p95 latency is above 10 seconds for 5 minutes",
                "summary": "High MCP latency"
            },
            "labels": {
                "severity": "warning",
                "service": "mcp"
            }
        }
        
        # Create or update alert rules
        success1 = await client.create_or_update_alert_rule(high_error_rate_rule)
        success2 = await client.create_or_update_alert_rule(circuit_breaker_rule)
        success3 = await client.create_or_update_alert_rule(high_latency_rule)
        
        return success1 and success2 and success3
    except Exception as e:
        logger.error(f"Error provisioning alert rules: {str(e)}")
        return False


async def setup_grafana() -> bool:
    """
    Set up Grafana with datasources, dashboards, and alert rules.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Provision Prometheus datasource
        datasource_result = await provision_prometheus_datasource()
        
        if not datasource_result:
            logger.warning("Failed to provision Prometheus datasource")
        
        # Provision dashboards
        dashboard_results = []
        
        # MCP Overview dashboard
        overview_result = await provision_dashboard("mcp_overview")
        dashboard_results.append(overview_result is not None)
        
        # MCP Provider dashboard template
        provider_result = await provision_dashboard("mcp_provider_template")
        dashboard_results.append(provider_result is not None)
        
        # MCP Performance dashboard
        performance_result = await provision_dashboard("mcp_performance")
        dashboard_results.append(performance_result is not None)
        
        # MCP Errors dashboard
        errors_result = await provision_dashboard("mcp_errors")
        dashboard_results.append(errors_result is not None)
        
        # Provision alert rules
        alert_result = await provision_alert_rules()
        
        if not alert_result:
            logger.warning("Failed to provision alert rules")
        
        # Return overall success
        return all(dashboard_results) and alert_result
    except Exception as e:
        logger.error(f"Error setting up Grafana: {str(e)}")
        return False
