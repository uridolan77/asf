"""
Base client for external API clients.

This module provides a base class for external API clients with retry logic
and circuit breaker pattern.
"""

import logging
import json
from typing import Dict, Any, Optional, List, Union, TypeVar, Generic, Type, Callable
from abc import ABC, abstractmethod

import httpx
import aiohttp
from tenacity import RetryError

from asf.medical.core.exceptions import ExternalServiceError
from asf.medical.core.retry import with_retry, with_tenacity_retry
from asf.medical.core.circuit_breaker import with_circuit_breaker
from asf.medical.core.observability import trace_external_call

# Configure logging
logger = logging.getLogger(__name__)

# Type variables
T = TypeVar("T")
ResponseType = TypeVar("ResponseType")


class BaseClient(ABC, Generic[ResponseType]):
    """
    Base class for external API clients.
    
    This class provides common functionality for external API clients,
    including retry logic and circuit breaker pattern.
    """
    
    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        max_retries: int = 3,
        circuit_breaker_name: Optional[str] = None,
    ):
        """
        Initialize the client.
        
        Args:
            base_url: Base URL for the API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            circuit_breaker_name: Name of the circuit breaker
        """
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.circuit_breaker_name = circuit_breaker_name or self.__class__.__name__
        
        # Initialize HTTP clients
        self._sync_client = httpx.Client(
            base_url=base_url,
            timeout=timeout,
        )
        
        # Headers for requests
        self.headers: Dict[str, str] = {
            "Accept": "application/json",
            "User-Agent": f"ASF-Medical-Research-Synthesizer/{self.__class__.__name__}",
        }
    
    async def _create_async_client(self) -> aiohttp.ClientSession:
        """
        Create an async HTTP client.
        
        Returns:
            aiohttp.ClientSession: Async HTTP client
        """
        return aiohttp.ClientSession(
            base_url=self.base_url,
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            headers=self.headers,
        )
    
    @with_retry(max_attempts=3, exception_types=ExternalServiceError)
    @with_circuit_breaker(name="base_client", exception_types=ExternalServiceError)
    def get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> ResponseType:
        """
        Make a GET request to the API.
        
        Args:
            path: API endpoint path
            params: Query parameters
            headers: Additional headers
            
        Returns:
            Response data
            
        Raises:
            ExternalServiceError: If the request fails
        """
        with trace_external_call(self.__class__.__name__, "GET", path):
            try:
                # Merge headers
                request_headers = {**self.headers}
                if headers:
                    request_headers.update(headers)
                
                # Make the request
                response = self._sync_client.get(
                    path,
                    params=params,
                    headers=request_headers,
                )
                
                # Check for errors
                response.raise_for_status()
                
                # Parse the response
                return self._parse_response(response.text)
            except httpx.HTTPError as e:
                logger.error(f"HTTP error in {self.__class__.__name__}.get: {e}")
                raise ExternalServiceError(self.__class__.__name__, str(e))
            except Exception as e:
                logger.error(f"Error in {self.__class__.__name__}.get: {e}")
                raise ExternalServiceError(self.__class__.__name__, str(e))
    
    @with_retry(max_attempts=3, exception_types=ExternalServiceError)
    @with_circuit_breaker(name="base_client", exception_types=ExternalServiceError)
    def post(
        self,
        path: str,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> ResponseType:
        """
        Make a POST request to the API.
        
        Args:
            path: API endpoint path
            data: Form data
            json_data: JSON data
            params: Query parameters
            headers: Additional headers
            
        Returns:
            Response data
            
        Raises:
            ExternalServiceError: If the request fails
        """
        with trace_external_call(self.__class__.__name__, "POST", path):
            try:
                # Merge headers
                request_headers = {**self.headers}
                if headers:
                    request_headers.update(headers)
                
                # Make the request
                response = self._sync_client.post(
                    path,
                    data=data,
                    json=json_data,
                    params=params,
                    headers=request_headers,
                )
                
                # Check for errors
                response.raise_for_status()
                
                # Parse the response
                return self._parse_response(response.text)
            except httpx.HTTPError as e:
                logger.error(f"HTTP error in {self.__class__.__name__}.post: {e}")
                raise ExternalServiceError(self.__class__.__name__, str(e))
            except Exception as e:
                logger.error(f"Error in {self.__class__.__name__}.post: {e}")
                raise ExternalServiceError(self.__class__.__name__, str(e))
    
    @with_tenacity_retry(max_attempts=3)
    async def async_get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> ResponseType:
        """
        Make an async GET request to the API.
        
        Args:
            path: API endpoint path
            params: Query parameters
            headers: Additional headers
            
        Returns:
            Response data
            
        Raises:
            ExternalServiceError: If the request fails
        """
        with trace_external_call(self.__class__.__name__, "GET", path):
            try:
                # Create async client
                async with await self._create_async_client() as session:
                    # Merge headers
                    request_headers = {**self.headers}
                    if headers:
                        request_headers.update(headers)
                    
                    # Make the request
                    async with session.get(
                        path,
                        params=params,
                        headers=request_headers,
                    ) as response:
                        # Check for errors
                        if response.status >= 400:
                            error_text = await response.text()
                            raise ExternalServiceError(
                                self.__class__.__name__,
                                f"HTTP error {response.status}: {error_text}"
                            )
                        
                        # Parse the response
                        response_text = await response.text()
                        return self._parse_response(response_text)
            except aiohttp.ClientError as e:
                logger.error(f"HTTP error in {self.__class__.__name__}.async_get: {e}")
                raise ExternalServiceError(self.__class__.__name__, str(e))
            except Exception as e:
                logger.error(f"Error in {self.__class__.__name__}.async_get: {e}")
                raise ExternalServiceError(self.__class__.__name__, str(e))
    
    @with_tenacity_retry(max_attempts=3)
    async def async_post(
        self,
        path: str,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> ResponseType:
        """
        Make an async POST request to the API.
        
        Args:
            path: API endpoint path
            data: Form data
            json_data: JSON data
            params: Query parameters
            headers: Additional headers
            
        Returns:
            Response data
            
        Raises:
            ExternalServiceError: If the request fails
        """
        with trace_external_call(self.__class__.__name__, "POST", path):
            try:
                # Create async client
                async with await self._create_async_client() as session:
                    # Merge headers
                    request_headers = {**self.headers}
                    if headers:
                        request_headers.update(headers)
                    
                    # Make the request
                    async with session.post(
                        path,
                        data=data,
                        json=json_data,
                        params=params,
                        headers=request_headers,
                    ) as response:
                        # Check for errors
                        if response.status >= 400:
                            error_text = await response.text()
                            raise ExternalServiceError(
                                self.__class__.__name__,
                                f"HTTP error {response.status}: {error_text}"
                            )
                        
                        # Parse the response
                        response_text = await response.text()
                        return self._parse_response(response_text)
            except aiohttp.ClientError as e:
                logger.error(f"HTTP error in {self.__class__.__name__}.async_post: {e}")
                raise ExternalServiceError(self.__class__.__name__, str(e))
            except Exception as e:
                logger.error(f"Error in {self.__class__.__name__}.async_post: {e}")
                raise ExternalServiceError(self.__class__.__name__, str(e))
    
    @abstractmethod
    def _parse_response(self, response_text: str) -> ResponseType:
        """
        Parse the response text.
        
        Args:
            response_text: Response text
            
        Returns:
            Parsed response
        """
        pass
    
    def close(self) -> None:
        """Close the HTTP client."""
        self._sync_client.close()
    
    def __del__(self) -> None:
        """Close the HTTP client when the object is deleted."""
        try:
            self.close()
        except Exception:
            pass


class JSONClient(BaseClient[Dict[str, Any]]):
    """
    Client for JSON APIs.
    
    This class provides a client for APIs that return JSON responses.
    """
    
    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        max_retries: int = 3,
        circuit_breaker_name: Optional[str] = None,
    ):
        """
        Initialize the client.
        
        Args:
            base_url: Base URL for the API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            circuit_breaker_name: Name of the circuit breaker
        """
        super().__init__(
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            circuit_breaker_name=circuit_breaker_name,
        )
        
        # Set JSON content type
        self.headers["Content-Type"] = "application/json"
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse the response text as JSON.
        
        Args:
            response_text: Response text
            
        Returns:
            Parsed JSON
            
        Raises:
            ExternalServiceError: If the response is not valid JSON
        """
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON response: {e}")
            raise ExternalServiceError(self.__class__.__name__, f"Invalid JSON response: {e}")


# Export classes
__all__ = [
    "BaseClient",
    "JSONClient",
]
