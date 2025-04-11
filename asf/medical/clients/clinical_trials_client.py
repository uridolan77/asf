"""
ClinicalTrials.gov client for the Medical Research Synthesizer.

This module provides a client for interacting with the ClinicalTrials.gov API.
"""

import asyncio
import json
import logging
import hashlib
import random
from typing import Dict, List, Optional, Any, Union
import httpx
from functools import wraps

from asf.medical.core.cache import cache_manager
from asf.medical.core.rate_limiter import AsyncRateLimiter
from asf.medical.core.exceptions import ExternalServiceError

# Set up logging
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
            # Generate cache key
            key_parts = [
                prefix,
                data_type,
                func.__name__,
                *[str(arg) for arg in args[1:]],  # Skip self
                *[f"{k}={v}" for k, v in sorted(kwargs.items())]
            ]
            key = hashlib.md5("_".join(key_parts).encode()).hexdigest()

            # Try to get from cache
            cached_result = cache_manager.get(key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result

            # Call the function
            logger.debug(f"Cache miss for {func.__name__}")
            result = await func(*args, **kwargs)

            # Store in cache
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
        """
        Initialize the ClinicalTrials.gov client.

        Args:
            base_url: Base URL for ClinicalTrials.gov API (default: "https://clinicaltrials.gov/api/v2")
            timeout: Request timeout in seconds (default: 30.0)
            max_retries: Maximum number of retry attempts (default: 3)
            requests_per_second: Maximum number of requests per second (default: 5.0)
            burst_size: Maximum number of requests that can be made in a burst (default: 10)
            cache_ttl: Default cache TTL in seconds (default: 3600 = 1 hour)
        """
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.requests_per_second = requests_per_second
        self.cache_ttl = cache_ttl

        # Initialize HTTP client
        self.client = httpx.AsyncClient(timeout=timeout)

        # Initialize rate limiter
        self.rate_limiter = AsyncRateLimiter(requests_per_second, burst_size)

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Make a request to the ClinicalTrials.gov API with retry logic and rate limiting.

        Args:
            endpoint: API endpoint
            params: Request parameters

        Returns:
            Response data

        Raises:
            ClinicalTrialsClientError: If the request fails after retries
        """
        # Apply rate limiting
        await self.rate_limiter.acquire()

        # Prepare request URL and parameters
        url = f"{self.base_url}/{endpoint}"
        if params is None:
            params = {}

        logger.debug(f"Making request to {url} with params {params}")

        # Initialize retry counter
        retry_count = 0
        max_retries = self.max_retries
        last_error = None

        while retry_count <= max_retries:
            try:
                # Make the request
                response = await self.client.get(url, params=params)

                # Check for HTTP errors
                if response.status_code >= 400:
                    error_msg = f"HTTP error {response.status_code}: {response.text}"
                    logger.warning(error_msg)

                    # Handle specific status codes
                    if response.status_code == 429:  # Too Many Requests
                        retry_count += 1
                        if retry_count <= max_retries:
                            # Exponential backoff with jitter
                            wait_time = min(30, (2 ** retry_count) + (random.random() * 0.5))
                            logger.warning(f"Rate limited. Retrying in {wait_time:.2f} seconds (attempt {retry_count}/{max_retries})")
                            await asyncio.sleep(wait_time)
                            continue

                    # For other errors, raise immediately
                    raise ClinicalTrialsClientError(error_msg, response.status_code)

                # Parse JSON response
                try:
                    return response.json()
                except json.JSONDecodeError as e:
                    error_msg = f"Invalid JSON response: {str(e)}"
                    logger.error(error_msg)
                    raise ClinicalTrialsClientError(error_msg)

            except (httpx.HTTPError, asyncio.TimeoutError) as e:
                retry_count += 1
                last_error = e

                if retry_count <= max_retries:
                    # Exponential backoff with jitter
                    wait_time = min(30, (2 ** retry_count) + (random.random() * 0.5))
                    logger.warning(f"Request failed: {str(e)}. Retrying in {wait_time:.2f} seconds (attempt {retry_count}/{max_retries})")
                    await asyncio.sleep(wait_time)
                else:
                    # Max retries reached
                    error_msg = f"Request failed after {max_retries} retries: {str(e)}"
                    logger.error(error_msg)
                    raise ClinicalTrialsClientError(error_msg)

        # This should not be reached, but just in case
        if last_error:
            raise ClinicalTrialsClientError(f"Request failed: {str(last_error)}")
        else:
            raise ClinicalTrialsClientError("Request failed for unknown reason")

    @clinical_trials_cache(ttl=3600, prefix="ct", data_type="search")
    async def search(
        self,
        query: str,
        max_results: int = 20,
        status: Optional[str] = None,
        phase: Optional[str] = None,
        study_type: Optional[str] = None,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
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
        """
        if not query or not query.strip():
            raise ClinicalTrialsClientError("Search query cannot be empty")

        if max_results < 1:
            raise ClinicalTrialsClientError("max_results must be at least 1")

        params = {
            "query.term": query.strip(),
            "pageSize": max_results,
            "format": "json"
        }

        # Add filters if provided
        if status:
            params["filter.overall_status"] = status

        if phase:
            params["filter.phase"] = phase

        if study_type:
            params["filter.study_type"] = study_type

        # Add date range if provided
        if min_date and max_date:
            params["filter.start_date"] = f"{min_date},{max_date}"

        try:
            logger.info(f"Searching ClinicalTrials.gov for '{query}' (max_results={max_results})")
            data = await self._make_request("studies", params)

            # Extract study summaries
            studies = []
            if "studies" in data:
                for study in data["studies"]:
                    # Extract basic study information
                    study_summary = {
                        "nct_id": study.get("protocolSection", {}).get("identificationModule", {}).get("nctId", ""),
                        "title": study.get("protocolSection", {}).get("identificationModule", {}).get("officialTitle", ""),
                        "brief_title": study.get("protocolSection", {}).get("identificationModule", {}).get("briefTitle", ""),
                        "status": study.get("protocolSection", {}).get("statusModule", {}).get("overallStatus", ""),
                        "phase": study.get("protocolSection", {}).get("designModule", {}).get("phases", []),
                        "study_type": study.get("protocolSection", {}).get("designModule", {}).get("studyType", ""),
                        "conditions": study.get("protocolSection", {}).get("conditionsModule", {}).get("conditions", []),
                        "interventions": [],
                        "source": "ClinicalTrials.gov"
                    }

                    # Extract interventions
                    interventions_module = study.get("protocolSection", {}).get("armsInterventionsModule", {})
                    if "interventions" in interventions_module:
                        for intervention in interventions_module["interventions"]:
                            study_summary["interventions"].append({
                                "name": intervention.get("name", ""),
                                "type": intervention.get("type", "")
                            })

                    studies.append(study_summary)

            logger.info(f"Found {len(studies)} studies matching '{query}'")
            return studies
        except ClinicalTrialsClientError as e:
            # Re-raise with more context
            logger.error(f"Error searching ClinicalTrials.gov for '{query}': {str(e)}")
            raise ClinicalTrialsClientError(f"Failed to search ClinicalTrials.gov: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error searching ClinicalTrials.gov for '{query}': {str(e)}")
            raise ClinicalTrialsClientError(f"Unexpected error searching ClinicalTrials.gov: {str(e)}")

    @clinical_trials_cache(ttl=86400, prefix="ct", data_type="study")  # Cache for 24 hours
    async def get_study(self, nct_id: str) -> Dict[str, Any]:
        """
        Get details for a specific clinical trial.

        Args:
            nct_id: NCT ID (e.g., "NCT01234567")

        Returns:
            Trial details

        Raises:
            ClinicalTrialsClientError: If the study cannot be retrieved
        """
        if not nct_id or not nct_id.strip():
            raise ClinicalTrialsClientError("NCT ID cannot be empty")

        # Normalize NCT ID (ensure it starts with NCT)
        normalized_nct_id = nct_id.strip().upper()
        if not normalized_nct_id.startswith("NCT"):
            raise ClinicalTrialsClientError(f"Invalid NCT ID format: {nct_id}. Must start with 'NCT'")

        try:
            logger.info(f"Fetching study details for {normalized_nct_id}")
            data = await self._make_request(f"studies/{normalized_nct_id}")

            # Extract study details
            study = data.get("study", {})
            protocol_section = study.get("protocolSection", {})

            # Basic information
            identification_module = protocol_section.get("identificationModule", {})
            status_module = protocol_section.get("statusModule", {})
            design_module = protocol_section.get("designModule", {})
            conditions_module = protocol_section.get("conditionsModule", {})
            description_module = protocol_section.get("descriptionModule", {})

            # Eligibility
            eligibility_module = protocol_section.get("eligibilityModule", {})

            # Interventions
            arms_interventions_module = protocol_section.get("armsInterventionsModule", {})

            # Outcomes
            outcomes_module = protocol_section.get("outcomesModule", {})

            # Create detailed study object
            study_details = {
                "nct_id": identification_module.get("nctId", ""),
                "title": identification_module.get("officialTitle", ""),
                "brief_title": identification_module.get("briefTitle", ""),
                "status": status_module.get("overallStatus", ""),
                "phase": design_module.get("phases", []),
                "study_type": design_module.get("studyType", ""),
                "conditions": conditions_module.get("conditions", []),
                "brief_summary": description_module.get("briefSummary", ""),
                "detailed_description": description_module.get("detailedDescription", ""),
                "start_date": status_module.get("startDateStruct", {}).get("date", ""),
                "completion_date": status_module.get("completionDateStruct", {}).get("date", ""),
                "eligibility": {
                    "criteria": eligibility_module.get("eligibilityCriteria", ""),
                    "gender": eligibility_module.get("gender", ""),
                    "minimum_age": eligibility_module.get("minimumAge", ""),
                    "maximum_age": eligibility_module.get("maximumAge", "")
                },
                "interventions": [],
                "outcomes": [],
                "source": "ClinicalTrials.gov"
            }

            # Extract interventions
            if "interventions" in arms_interventions_module:
                for intervention in arms_interventions_module["interventions"]:
                    study_details["interventions"].append({
                        "name": intervention.get("name", ""),
                        "type": intervention.get("type", ""),
                        "description": intervention.get("description", "")
                    })

            # Extract outcomes
            if "primaryOutcomes" in outcomes_module:
                for outcome in outcomes_module["primaryOutcomes"]:
                    study_details["outcomes"].append({
                        "type": "primary",
                        "measure": outcome.get("measure", ""),
                        "description": outcome.get("description", ""),
                        "time_frame": outcome.get("timeFrame", "")
                    })

            if "secondaryOutcomes" in outcomes_module:
                for outcome in outcomes_module["secondaryOutcomes"]:
                    study_details["outcomes"].append({
                        "type": "secondary",
                        "measure": outcome.get("measure", ""),
                        "description": outcome.get("description", ""),
                        "time_frame": outcome.get("timeFrame", "")
                    })

            logger.info(f"Successfully retrieved details for study {normalized_nct_id}")
            return study_details
        except ClinicalTrialsClientError as e:
            # Re-raise with more context
            logger.error(f"Error getting study {normalized_nct_id} from ClinicalTrials.gov: {str(e)}")
            raise ClinicalTrialsClientError(f"Failed to get study details: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error getting study {normalized_nct_id} from ClinicalTrials.gov: {str(e)}")
            raise ClinicalTrialsClientError(f"Unexpected error getting study details: {str(e)}")

    @clinical_trials_cache(ttl=3600, prefix="ct", data_type="batch")
    async def batch_get_studies(self, nct_ids: List[str], max_concurrent: int = 5) -> Dict[str, Dict[str, Any]]:
        """
        Get details for multiple clinical trials in parallel.

        Args:
            nct_ids: List of NCT IDs
            max_concurrent: Maximum number of concurrent requests (default: 5)

        Returns:
            Dictionary mapping NCT IDs to study details

        Raises:
            ClinicalTrialsClientError: If the batch operation fails
        """
        if not nct_ids:
            raise ClinicalTrialsClientError("NCT IDs list cannot be empty")

        # Validate and normalize NCT IDs
        normalized_nct_ids = []
        for nct_id in nct_ids:
            if not nct_id or not nct_id.strip():
                continue

            normalized_nct_id = nct_id.strip().upper()
            if not normalized_nct_id.startswith("NCT"):
                logger.warning(f"Skipping invalid NCT ID: {nct_id}. Must start with 'NCT'")
                continue

            normalized_nct_ids.append(normalized_nct_id)

        if not normalized_nct_ids:
            raise ClinicalTrialsClientError("No valid NCT IDs provided")

        logger.info(f"Batch fetching {len(normalized_nct_ids)} studies")

        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)

        # Define a worker function to process each NCT ID
        async def fetch_study(nct_id: str) -> tuple[str, Optional[Dict[str, Any]]]:
            async with semaphore:
                try:
                    study = await self.get_study(nct_id)
                    return nct_id, study
                except Exception as e:
                    logger.error(f"Error fetching study {nct_id}: {str(e)}")
                    return nct_id, None

        # Create tasks for all NCT IDs
        tasks = [fetch_study(nct_id) for nct_id in normalized_nct_ids]

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)

        # Build result dictionary
        studies = {}
        for nct_id, study in results:
            if study is not None:
                studies[nct_id] = study

        logger.info(f"Successfully fetched {len(studies)}/{len(normalized_nct_ids)} studies")
        return studies
