"""
ClinicalTrials.gov client for the Medical Research Synthesizer.
This module provides a client for interacting with the ClinicalTrials.gov API.
"""
import logging
import hashlib
from typing import List, Optional
import httpx
from functools import wraps
from asf.medical.core.enhanced_cache import enhanced_cache_manager as cache_manager, enhanced_cached as cached
from asf.medical.core.rate_limiter import AsyncRateLimiter
from asf.medical.core.exceptions import ExternalServiceError
logger = logging.getLogger(__name__)
class ClinicalTrialsClientError(ExternalServiceError):
    """
    Exception raised when the ClinicalTrials.gov API returns an error.
    """
    def __init__(self, message: str, status_code: Optional[int] = None):
        """
        Initialize the exception.
        Args:
            message: Error message
            status_code: HTTP status code (optional)
        """
        self.status_code = status_code
        super().__init__("ClinicalTrials.gov", message)
def clinical_trials_cache(ttl: int = 3600, prefix: str = "ct", data_type: str = "search"):
    """
    Decorator for caching ClinicalTrials.gov API responses.
    Args:
        ttl: Time-to-live in seconds (default: 3600 = 1 hour)
        prefix: Cache key prefix (default: "ct")
        data_type: Type of data being cached (default: "search")
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            key_parts = [
                prefix,
                data_type,
                func.__name__,
                *[str(arg) for arg in args[1:]],  # Skip self
                *[f"{k}={v}" for k, v in sorted(kwargs.items())]
            ]
            key = hashlib.md5("_".join(key_parts).encode()).hexdigest()
            cached_result = cache_manager.get(key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            logger.debug(f"Cache miss for {func.__name__}")
            result = await func(*args, **kwargs)
            cache_manager.set(key, result, ttl)
            return result
        return wrapper
    return decorator
class ClinicalTrialsClient:
    """
    Client for interacting with the ClinicalTrials.gov API.
    This client provides methods for searching clinical trials and retrieving trial details.
    Features:
    - Retry logic for API calls
    - Rate limiting to prevent API throttling
    - Caching of API responses
    - Comprehensive error handling
    """
    def __init__(
        self,
        base_url: str = "https://clinicaltrials.gov/api/v2",
        timeout: float = 30.0,
        max_retries: int = 3,
        requests_per_second: float = 5.0,
        burst_size: int = 10,
        cache_ttl: int = 3600  # 1 hour
    ):
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.requests_per_second = requests_per_second
        self.cache_ttl = cache_ttl
        self.client = httpx.AsyncClient(timeout=timeout)
        self.rate_limiter = AsyncRateLimiter(requests_per_second, burst_size)
    async def close(self):
        """
        Make a request to the ClinicalTrials.gov API with retry logic and rate limiting.
        Args:
            endpoint: API endpoint
            params: Request parameters
        Returns:
            Response data
        Raises:
            ClinicalTrialsClientError: If the request fails after retries
        Search for clinical trials.
        Args:
            query: Search query
            max_results: Maximum number of results to return (default: 20)
            status: Trial status (e.g., "Recruiting", "Completed")
            phase: Trial phase (e.g., "Phase 1", "Phase 2")
            study_type: Study type (e.g., "Interventional", "Observational")
            min_date: Minimum start date (YYYY/MM/DD format)
            max_date: Maximum start date (YYYY/MM/DD format)
        Returns:
            List of trial summaries
        Raises:
            ClinicalTrialsClientError: If the search fails
        Get details for a specific clinical trial.
        Args:
            nct_id: NCT ID (e.g., "NCT01234567")
        Returns:
            Trial details
        Raises:
            ClinicalTrialsClientError: If the study cannot be retrieved
        Get details for multiple clinical trials in parallel.
        Args:
            nct_ids: List of NCT IDs
            max_concurrent: Maximum number of concurrent requests (default: 5)
        Returns:
            Dictionary mapping NCT IDs to study details
        Raises:
            ClinicalTrialsClientError: If the batch operation fails
        """