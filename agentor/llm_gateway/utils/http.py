import aiohttp
import logging
from typing import Dict, Any, Optional
import json

logger = logging.getLogger(__name__)


class AsyncHTTPClient:
    """Asynchronous HTTP client with connection pooling."""
    
    def __init__(self, limit_per_host: int = 100, timeout: int = 30):
        """Initialize the HTTP client.
        
        Args:
            limit_per_host: Maximum number of connections per host
            timeout: Timeout in seconds
        """
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit_per_host=limit_per_host),
            timeout=aiohttp.ClientTimeout(total=timeout)
        )
    
    async def close(self):
        """Close the HTTP client."""
        await self.session.close()
    
    async def get(self, url: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Send a GET request.
        
        Args:
            url: The URL to request
            headers: The request headers
            
        Returns:
            The response as a dictionary
        """
        async with self.session.get(url, headers=headers) as response:
            response.raise_for_status()
            return await response.json()
    
    async def post(
        self,
        url: str,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Send a POST request.
        
        Args:
            url: The URL to request
            data: The request data
            headers: The request headers
            
        Returns:
            The response as a dictionary
        """
        async with self.session.post(url, json=data, headers=headers) as response:
            response.raise_for_status()
            return await response.json()
    
    async def put(
        self,
        url: str,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Send a PUT request.
        
        Args:
            url: The URL to request
            data: The request data
            headers: The request headers
            
        Returns:
            The response as a dictionary
        """
        async with self.session.put(url, json=data, headers=headers) as response:
            response.raise_for_status()
            return await response.json()
    
    async def delete(self, url: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Send a DELETE request.
        
        Args:
            url: The URL to request
            headers: The request headers
            
        Returns:
            The response as a dictionary
        """
        async with self.session.delete(url, headers=headers) as response:
            response.raise_for_status()
            return await response.json()
