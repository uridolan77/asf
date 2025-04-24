Base client for external API clients.

This module provides a base class for external API clients with retry logic
and circuit breaker pattern.

import logging
import json
from abc import ABC, abstractmethod

import httpx
import aiohttp

from asf.medical.core.exceptions import ExternalServiceError
from asf.medical.core.retry import with_retry, with_tenacity_retry
from asf.medical.core.circuit_breaker import with_circuit_breaker
from asf.medical.core.observability import trace_external_call

logger = logging.getLogger(__name__)

T = TypeVar("T")
ResponseType = TypeVar("ResponseType")


class BaseClient(ABC, Generic[ResponseType]):
    Base class for external API clients.
    
    This class provides common functionality for external API clients,
    including retry logic and circuit breaker pattern.
    
    def __init__(
        self,
        base_url: str,
            """
            __init__ function.
            
            This function provides functionality for...
            Args:
                base_url: Description of base_url
                timeout: Description of timeout
                max_retries: Description of max_retries
                circuit_breaker_name: Description of circuit_breaker_name
            """
        timeout: float = 30.0,
        max_retries: int = 3,
        circuit_breaker_name: Optional[str] = None,
    ):
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.circuit_breaker_name = circuit_breaker_name or self.__class__.__name__
        
        self._sync_client = httpx.Client(
            base_url=base_url,
            timeout=timeout,
        )
        
        self.headers: Dict[str, str] = {
            "Accept": "application/json",
            "User-Agent": f"ASF-Medical-Research-Synthesizer/{self.__class__.__name__}",
        }
    
    async def _create_async_client(self) -> aiohttp.ClientSession:
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
            """
            get function.
            
            This function provides functionality for...
            Args:
                path: Description of path
                params: Description of params
                headers: Description of headers
            
            Returns:
                Description of return value
            """
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> ResponseType:
        with trace_external_call(self.__class__.__name__, "GET", path):
            try:
                request_headers = {**self.headers}
                if headers:
                    request_headers.update(headers)
                
                response = self._sync_client.get(
                    path,
                    params=params,
                    headers=request_headers,
                )
                
                response.raise_for_status()
                
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
            """
            post function.
            
            This function provides functionality for...
            Args:
                path: Description of path
                data: Description of data
                json_data: Description of json_data
                params: Description of params
                headers: Description of headers
            
            Returns:
                Description of return value
            """
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> ResponseType:
        with trace_external_call(self.__class__.__name__, "POST", path):
            try:
                request_headers = {**self.headers}
                if headers:
                    request_headers.update(headers)
                
                response = self._sync_client.post(
                    path,
                    data=data,
                    json=json_data,
                    params=params,
                    headers=request_headers,
                )
                
                response.raise_for_status()
                
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
        with trace_external_call(self.__class__.__name__, "GET", path):
            try:
                async with await self._create_async_client() as session:
                    request_headers = {**self.headers}
                    if headers:
                        request_headers.update(headers)
                    
                    async with session.get(
                        path,
                        params=params,
                        headers=request_headers,
                    ) as response:
                        if response.status >= 400:
                            error_text = await response.text()
                            raise ExternalServiceError(
                                self.__class__.__name__,
                                f"HTTP error {response.status}: {error_text}"
                            )
                        
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
        with trace_external_call(self.__class__.__name__, "POST", path):
            try:
                async with await self._create_async_client() as session:
                    request_headers = {**self.headers}
                    if headers:
                        request_headers.update(headers)
                    
                    async with session.post(
                        path,
                        data=data,
                        json=json_data,
                        params=params,
                        headers=request_headers,
                    ) as response:
                        if response.status >= 400:
                            error_text = await response.text()
                            raise ExternalServiceError(
                                self.__class__.__name__,
                                f"HTTP error {response.status}: {error_text}"
                            )
                        
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
        """Close the HTTP client.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
        self._sync_client.close()
    
    def __del__(self) -> None:
        """Close the HTTP client when the object is deleted.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
        try:
            self.close()
        except Exception:
            pass


class JSONClient(BaseClient[Dict[str, Any]]):
    Client for JSON APIs.
    
    This class provides a client for APIs that return JSON responses.
    
    def __init__(
        self,
        base_url: str,
            """
            __init__ function.
            
            This function provides functionality for...
            Args:
                base_url: Description of base_url
                timeout: Description of timeout
                max_retries: Description of max_retries
                circuit_breaker_name: Description of circuit_breaker_name
            """
        timeout: float = 30.0,
        max_retries: int = 3,
        circuit_breaker_name: Optional[str] = None,
    ):
        super().__init__(
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            circuit_breaker_name=circuit_breaker_name,
        )
        
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


__all__ = [
    "BaseClient",
    "JSONClient",
]
