"""
Prometheus service for LLM Gateway metrics.

This service provides access to Prometheus metrics for the LLM Gateway,
including circuit breaker history, error trends, and performance metrics.
"""

import logging
import aiohttp
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union

logger = logging.getLogger(__name__)


class PrometheusService:
    """
    Service for accessing Prometheus metrics for the LLM Gateway.
    """
    
    def __init__(self, prometheus_url: str = "http://localhost:9090"):
        """
        Initialize the Prometheus service.
        
        Args:
            prometheus_url: URL of the Prometheus server
        """
        self.prometheus_url = prometheus_url
        self.api_url = f"{prometheus_url}/api/v1"
    
    async def query(self, query: str, time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Execute a Prometheus query.
        
        Args:
            query: PromQL query
            time: Query time (optional)
            
        Returns:
            Query result
        """
        params = {"query": query}
        
        if time:
            params["time"] = time.timestamp()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_url}/query", params=params) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Prometheus query failed: {error_text}")
                        return {"status": "error", "data": {"result": []}}
                    
                    return await response.json()
        except Exception as e:
            logger.error(f"Error querying Prometheus: {str(e)}")
            return {"status": "error", "data": {"result": []}}
    
    async def query_range(
        self,
        query: str,
        start: datetime,
        end: datetime,
        step: str = "15s"
    ) -> Dict[str, Any]:
        """
        Execute a Prometheus range query.
        
        Args:
            query: PromQL query
            start: Start time
            end: End time
            step: Query resolution step
            
        Returns:
            Query result
        """
        params = {
            "query": query,
            "start": start.timestamp(),
            "end": end.timestamp(),
            "step": step
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_url}/query_range", params=params) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Prometheus range query failed: {error_text}")
                        return {"status": "error", "data": {"result": []}}
                    
                    return await response.json()
        except Exception as e:
            logger.error(f"Error querying Prometheus range: {str(e)}")
            return {"status": "error", "data": {"result": []}}
    
    async def get_circuit_breaker_history(
        self,
        provider_id: str,
        start_time: datetime,
        end_time: datetime,
        step: str = "1m"
    ) -> List[Dict[str, Any]]:
        """
        Get circuit breaker history for a provider.
        
        Args:
            provider_id: Provider ID
            start_time: Start time
            end_time: End time
            step: Query resolution step
            
        Returns:
            Circuit breaker history
        """
        # Query circuit breaker state
        state_query = f'llm_gateway_circuit_breaker_state{{provider="{provider_id}"}}'
        state_result = await self.query_range(state_query, start_time, end_time, step)
        
        # Query failure count
        failures_query = f'llm_gateway_circuit_breaker_failures_total{{provider="{provider_id}"}}'
        failures_result = await self.query_range(failures_query, start_time, end_time, step)
        
        # Process results
        history = []
        
        if state_result.get("status") == "success" and failures_result.get("status") == "success":
            state_data = state_result.get("data", {}).get("result", [])
            failures_data = failures_result.get("data", {}).get("result", [])
            
            if state_data and failures_data:
                state_values = state_data[0].get("values", [])
                failures_values = failures_data[0].get("values", [])
                
                # Create a mapping of timestamps to failure counts
                failures_map = {int(ts): float(value) for ts, value in failures_values}
                
                for ts, value in state_values:
                    timestamp = datetime.fromtimestamp(int(ts))
                    state = "open" if float(value) > 0 else "closed"
                    failure_count = int(failures_map.get(int(ts), 0))
                    
                    history.append({
                        "timestamp": timestamp.isoformat(),
                        "state": state,
                        "failure_count": failure_count,
                        "recovery_time": None  # We don't have this information from Prometheus
                    })
        
        # If no data from Prometheus, try to generate some mock data for testing
        if not history and start_time and end_time:
            # For testing/development only
            if not self.prometheus_url.startswith("http://localhost"):
                return []
            
            # Generate mock data
            current = start_time
            step_delta = timedelta(minutes=5)
            failure_count = 0
            state = "closed"
            
            while current <= end_time:
                # Simulate some failures
                if current.minute % 30 == 0:
                    failure_count += 1
                
                # Simulate circuit breaker opening
                if failure_count >= 5:
                    state = "open"
                    recovery_time = (current + timedelta(minutes=15)).isoformat()
                else:
                    state = "closed"
                    recovery_time = None
                
                history.append({
                    "timestamp": current.isoformat(),
                    "state": state,
                    "failure_count": failure_count,
                    "recovery_time": recovery_time
                })
                
                # Reset after some time
                if current.minute % 45 == 0:
                    failure_count = 0
                
                current += step_delta
        
        return history
    
    async def get_error_trends(
        self,
        provider_id: Optional[str] = None,
        start_time: datetime = None,
        end_time: datetime = None,
        step: str = "1h"
    ) -> Dict[str, Any]:
        """
        Get error trends for providers.
        
        Args:
            provider_id: Provider ID (optional, if None get all providers)
            start_time: Start time
            end_time: End time
            step: Query resolution step
            
        Returns:
            Error trends
        """
        if not start_time:
            start_time = datetime.utcnow() - timedelta(days=1)
        
        if not end_time:
            end_time = datetime.utcnow()
        
        # Build query
        provider_filter = f'provider="{provider_id}"' if provider_id else ""
        query = f'sum by (error_type) (rate(llm_gateway_errors_total{{{provider_filter}}}[1h]))'
        
        # Execute query
        result = await self.query_range(query, start_time, end_time, step)
        
        # Process results
        trends = {
            "timestamps": [],
            "error_types": {},
            "total_errors": 0
        }
        
        if result.get("status") == "success":
            data = result.get("data", {}).get("result", [])
            
            for series in data:
                error_type = series.get("metric", {}).get("error_type", "unknown")
                values = series.get("values", [])
                
                trends["error_types"][error_type] = []
                
                for ts, value in values:
                    timestamp = datetime.fromtimestamp(int(ts)).isoformat()
                    
                    if timestamp not in trends["timestamps"]:
                        trends["timestamps"].append(timestamp)
                    
                    error_count = float(value)
                    trends["error_types"][error_type].append(error_count)
                    trends["total_errors"] += error_count
        
        return trends
    
    async def get_performance_metrics(
        self,
        provider_id: Optional[str] = None,
        start_time: datetime = None,
        end_time: datetime = None,
        step: str = "1h"
    ) -> Dict[str, Any]:
        """
        Get performance metrics for providers.
        
        Args:
            provider_id: Provider ID (optional, if None get all providers)
            start_time: Start time
            end_time: End time
            step: Query resolution step
            
        Returns:
            Performance metrics
        """
        if not start_time:
            start_time = datetime.utcnow() - timedelta(days=1)
        
        if not end_time:
            end_time = datetime.utcnow()
        
        # Build queries
        provider_filter = f'provider="{provider_id}"' if provider_id else ""
        
        latency_query = f'histogram_quantile(0.95, sum by (le) (rate(llm_gateway_request_duration_seconds_bucket{{{provider_filter}}}[1h])))'
        requests_query = f'sum(rate(llm_gateway_requests_total{{{provider_filter}}}[1h]))'
        tokens_query = f'sum(rate(llm_gateway_tokens_total{{{provider_filter}}}[1h]))'
        
        # Execute queries
        latency_result = await self.query_range(latency_query, start_time, end_time, step)
        requests_result = await self.query_range(requests_query, start_time, end_time, step)
        tokens_result = await self.query_range(tokens_query, start_time, end_time, step)
        
        # Process results
        metrics = {
            "timestamps": [],
            "latency_p95": [],
            "requests_per_second": [],
            "tokens_per_second": []
        }
        
        # Process latency
        if latency_result.get("status") == "success":
            data = latency_result.get("data", {}).get("result", [])
            
            if data:
                values = data[0].get("values", [])
                
                for ts, value in values:
                    timestamp = datetime.fromtimestamp(int(ts)).isoformat()
                    
                    if timestamp not in metrics["timestamps"]:
                        metrics["timestamps"].append(timestamp)
                    
                    metrics["latency_p95"].append(float(value))
        
        # Process requests
        if requests_result.get("status") == "success":
            data = requests_result.get("data", {}).get("result", [])
            
            if data:
                values = data[0].get("values", [])
                
                for ts, value in values:
                    timestamp = datetime.fromtimestamp(int(ts)).isoformat()
                    
                    if timestamp not in metrics["timestamps"]:
                        metrics["timestamps"].append(timestamp)
                    
                    metrics["requests_per_second"].append(float(value))
        
        # Process tokens
        if tokens_result.get("status") == "success":
            data = tokens_result.get("data", {}).get("result", [])
            
            if data:
                values = data[0].get("values", [])
                
                for ts, value in values:
                    timestamp = datetime.fromtimestamp(int(ts)).isoformat()
                    
                    if timestamp not in metrics["timestamps"]:
                        metrics["timestamps"].append(timestamp)
                    
                    metrics["tokens_per_second"].append(float(value))
        
        return metrics
    
    async def get_connection_pool_metrics(
        self,
        provider_id: Optional[str] = None,
        transport_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get connection pool metrics for providers.
        
        Args:
            provider_id: Provider ID (optional, if None get all providers)
            transport_type: Transport type (optional, if None get all types)
            
        Returns:
            Connection pool metrics
        """
        # Build filters
        filters = []
        
        if provider_id:
            filters.append(f'provider="{provider_id}"')
        
        if transport_type:
            filters.append(f'transport_type="{transport_type}"')
        
        filter_str = ",".join(filters)
        filter_expr = f"{{{filter_str}}}" if filter_str else ""
        
        # Build queries
        size_query = f'llm_gateway_connection_pool_size{filter_expr}'
        active_query = f'llm_gateway_connection_pool_active{filter_expr}'
        errors_query = f'sum by (provider, transport_type, error_type) (llm_gateway_connection_errors_total{filter_expr})'
        
        # Execute queries
        size_result = await self.query(size_query)
        active_result = await self.query(active_query)
        errors_result = await self.query(errors_query)
        
        # Process results
        metrics = {
            "pools": [],
            "total_size": 0,
            "total_active": 0,
            "total_errors": 0,
            "error_types": {}
        }
        
        # Process pool size
        if size_result.get("status") == "success":
            data = size_result.get("data", {}).get("result", [])
            
            for series in data:
                provider = series.get("metric", {}).get("provider", "unknown")
                transport = series.get("metric", {}).get("transport_type", "unknown")
                value = float(series.get("value", [0, "0"])[1])
                
                pool = {
                    "provider": provider,
                    "transport_type": transport,
                    "size": value,
                    "active": 0,
                    "errors": 0
                }
                
                metrics["pools"].append(pool)
                metrics["total_size"] += value
        
        # Process active connections
        if active_result.get("status") == "success":
            data = active_result.get("data", {}).get("result", [])
            
            for series in data:
                provider = series.get("metric", {}).get("provider", "unknown")
                transport = series.get("metric", {}).get("transport_type", "unknown")
                value = float(series.get("value", [0, "0"])[1])
                
                # Find matching pool
                for pool in metrics["pools"]:
                    if pool["provider"] == provider and pool["transport_type"] == transport:
                        pool["active"] = value
                        metrics["total_active"] += value
                        break
        
        # Process errors
        if errors_result.get("status") == "success":
            data = errors_result.get("data", {}).get("result", [])
            
            for series in data:
                provider = series.get("metric", {}).get("provider", "unknown")
                transport = series.get("metric", {}).get("transport_type", "unknown")
                error_type = series.get("metric", {}).get("error_type", "unknown")
                value = float(series.get("value", [0, "0"])[1])
                
                # Find matching pool
                for pool in metrics["pools"]:
                    if pool["provider"] == provider and pool["transport_type"] == transport:
                        pool["errors"] += value
                        metrics["total_errors"] += value
                        break
                
                # Track error types
                if error_type not in metrics["error_types"]:
                    metrics["error_types"][error_type] = 0
                
                metrics["error_types"][error_type] += value
        
        return metrics


# Global instance
_prometheus_service = None


def get_prometheus_service() -> PrometheusService:
    """
    Get the global Prometheus service instance.
    
    Returns:
        PrometheusService instance
    """
    global _prometheus_service
    
    if _prometheus_service is None:
        # Get configuration from environment or use default
        import os
        prometheus_url = os.environ.get("PROMETHEUS_URL", "http://localhost:9090")
        _prometheus_service = PrometheusService(prometheus_url)
    
    return _prometheus_service
