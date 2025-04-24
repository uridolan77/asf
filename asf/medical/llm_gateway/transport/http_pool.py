"""
HTTP Connection Pool implementation using the resource management layer.

This module provides a specialized resource pool for managing HTTP client connections
with features like connection reuse, health checking, and automatic reconnection.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, Tuple, List
import uuid

import httpx

from ..core.resource_manager import ResourcePool, ResourceAcquisitionContext, ResourceType
from ..core.models import ResourcePoolConfig, ResourceLimits
from ..core.errors import ResourceError, HttpTransportError

logger = logging.getLogger(__name__)

class HttpClientPool(ResourcePool[httpx.AsyncClient]):
    """Resource pool for HTTP client connections."""
    
    def __init__(
        self, 
        config: ResourcePoolConfig,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        limits: Optional[httpx.Limits] = None,
        headers: Optional[Dict[str, str]] = None,
        verify_ssl: bool = True
    ):
        # Override resource type to ensure correct classification
        config.resource_type = ResourceType.HTTP_CONNECTION
        
        super().__init__(config)
        
        # HTTP client configuration
        self.base_url = base_url
        self.timeout = timeout
        self.limits = limits or httpx.Limits(max_keepalive_connections=10, max_connections=20)
        self.headers = headers or {}
        self.verify_ssl = verify_ssl
    
    async def create_resource(self) -> httpx.AsyncClient:
        """Create a new HTTP client."""
        try:
            client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                limits=self.limits,
                headers=self.headers,
                verify=self.verify_ssl
            )
            
            return client
        except Exception as e:
            raise ResourceError(
                f"Failed to create HTTP client: {str(e)}",
                resource_type=ResourceType.HTTP_CONNECTION.value,
                operation="create",
                cause=e
            )
    
    async def validate_resource(self, client: httpx.AsyncClient) -> bool:
        """Check if HTTP client is still usable."""
        # If client is closed, it's not valid
        if client.is_closed:
            return False
        
        # Optionally perform a health check request
        if self.base_url and self.config.enable_health_checks:
            try:
                # Simple HEAD request to check connectivity
                # You might want to hit a specific health check endpoint instead
                resp = await client.head("/", timeout=5.0)
                return resp.status_code < 500
            except Exception as e:
                logger.warning(f"HTTP client health check failed: {str(e)}")
                return False
        
        return True
    
    async def cleanup_resource(self, client: httpx.AsyncClient) -> None:
        """Close the HTTP client properly."""
        try:
            await client.aclose()
        except Exception as e:
            logger.warning(f"Error closing HTTP client: {str(e)}")


class HttpTransportManager:
    """Manager for HTTP transport operations."""
    
    def __init__(self):
        self.client_pools: Dict[str, HttpClientPool] = {}
    
    async def get_client(
        self, 
        base_url: str, 
        timeout: float = 30.0,
        pool_id: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        verify_ssl: bool = True,
        max_pool_size: int = 10,
        min_pool_size: int = 1,
    ) -> ResourceAcquisitionContext[httpx.AsyncClient]:
        """Get an HTTP client from a pool, creating the pool if needed."""
        # Generate pool ID if not provided
        if not pool_id:
            sanitized_url = base_url.replace("://", "_").replace(".", "_").replace("/", "_")
            pool_id = f"http_pool_{sanitized_url}_{uuid.uuid4().hex[:8]}"
        
        # Create pool if it doesn't exist
        if pool_id not in self.client_pools:
            config = ResourcePoolConfig(
                resource_type=ResourceType.HTTP_CONNECTION,
                pool_id=pool_id,
                limits=ResourceLimits(
                    max_pool_size=max_pool_size,
                    min_pool_size=min_pool_size,
                    max_idle_time_seconds=300,  # 5 minutes
                    circuit_breaker_threshold=5,
                    acquisition_timeout_seconds=10.0
                ),
                enable_health_checks=True,
                health_check_interval_seconds=60
            )
            
            # Create HTTP client pool
            client_pool = HttpClientPool(
                config=config,
                base_url=base_url,
                timeout=timeout,
                headers=headers,
                verify_ssl=verify_ssl
            )
            
            await client_pool.start()
            self.client_pools[pool_id] = client_pool
        
        # Get client from pool
        return await self.client_pools[pool_id].acquire()
    
    async def shutdown(self):
        """Shutdown all HTTP client pools."""
        for pool_id, pool in list(self.client_pools.items()):
            await pool.stop()
        
        self.client_pools.clear()


# Create a singleton instance
http_transport_manager = HttpTransportManager()


async def initialize_http_transport():
    """Initialize the HTTP transport system."""
    pass


async def shutdown_http_transport():
    """Shutdown the HTTP transport system."""
    await http_transport_manager.shutdown()


async def http_request(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    json: Any = None,
    data: Any = None,
    params: Any = None,
    timeout: float = 30.0,
    verify_ssl: bool = True,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> httpx.Response:
    """Make an HTTP request using the connection pool.
    
    Args:
        url: The URL to request
        method: HTTP method to use
        headers: HTTP headers
        json: JSON body
        data: Form data
        params: Query parameters
        timeout: Request timeout in seconds
        verify_ssl: Whether to verify SSL certificates
        max_retries: Maximum number of retries
        retry_delay: Delay between retries in seconds
    
    Returns:
        HTTPx Response object
    
    Raises:
        HttpTransportError: If the request fails
    """
    parsed_url = httpx.URL(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    path = str(parsed_url).replace(base_url, "")
    
    for attempt in range(max_retries + 1):
        client_context = None
        try:
            # Get client from pool
            client_context = await http_transport_manager.get_client(
                base_url=base_url,
                timeout=timeout,
                headers=headers,
                verify_ssl=verify_ssl
            )
            
            async with client_context as client:
                response = await client.request(
                    method=method,
                    url=path,
                    json=json,
                    data=data,
                    params=params,
                    timeout=timeout
                )
                
                # Handle 429 Too Many Requests with retries
                if response.status_code == 429 and attempt < max_retries:
                    # Get retry-after header if available
                    retry_after = response.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        wait_time = float(retry_after)
                    else:
                        wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    
                    logger.warning(f"Rate limited. Retrying in {wait_time:.2f}s...")
                    await asyncio.sleep(wait_time)
                    continue
                
                # Handle server errors with retries
                if response.status_code >= 500 and attempt < max_retries:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Server error {response.status_code}. Retrying in {wait_time:.2f}s...")
                    await asyncio.sleep(wait_time)
                    continue
                
                return response
                
        except httpx.TimeoutException as e:
            if attempt >= max_retries:
                raise HttpTransportError(
                    f"HTTP request timed out after {timeout}s",
                    status_code=None,
                    url=url,
                    retryable=True,
                    cause=e
                )
            
            wait_time = retry_delay * (2 ** attempt)
            logger.warning(f"Request timeout. Retrying in {wait_time:.2f}s...")
            await asyncio.sleep(wait_time)
        
        except httpx.HTTPError as e:
            if attempt >= max_retries:
                raise HttpTransportError(
                    f"HTTP error: {str(e)}",
                    status_code=getattr(e, "status_code", None),
                    url=url,
                    retryable=True,
                    cause=e
                )
            
            wait_time = retry_delay * (2 ** attempt)
            logger.warning(f"HTTP error: {str(e)}. Retrying in {wait_time:.2f}s...")
            await asyncio.sleep(wait_time)
        
        except Exception as e:
            raise HttpTransportError(
                f"Request failed: {str(e)}",
                url=url,
                retryable=False,
                cause=e
            )
        
        finally:
            # Note: We don't need to explicitly release the client
            # as the context manager will handle that
            pass
    
    # This should never be reached due to the exception in the last retry attempt
    raise HttpTransportError(
        f"HTTP request failed after {max_retries} retries",
        url=url,
        retryable=False
    )