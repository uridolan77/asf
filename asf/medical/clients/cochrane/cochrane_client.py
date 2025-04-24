"""Cochrane Library client for the Medical Research Synthesizer.

This module provides a client for interacting with the Cochrane Library API,
accessing systematic reviews, meta-analyses, and clinical guidelines.
"""
import asyncio
import json
import logging
import hashlib
import time
import re
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from medical.core.rate_limiter import AsyncRateLimiter
from medical.core.enhanced_cache import enhanced_cache_manager, enhanced_cached
from medical.core.exceptions import ValidationError

# Configure logging
logger = logging.getLogger(__name__)

class CochraneClientError(Exception):
    """Exception raised for Cochrane client errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

class PICOElement:
    """Base class for PICO elements."""
    
    def __init__(self, description: str, confidence: float = 0.0):
        self.description = description
        self.confidence = confidence

class EvidenceGrade:
    """Class representing GRADE evidence rating."""
    
    def __init__(self, grade: str, explanation: str = "", confidence: float = 0.0):
        self.grade = grade
        self.explanation = explanation
        self.confidence = confidence

class CochraneClient:
    """Client for interacting with the Cochrane Library API."""
    
    def __init__(
        self, 
        base_url: str = "https://www.cochranelibrary.com",
        timeout: float = 30.0,
        max_retries: int = 3,
        api_key: Optional[str] = None,
        requests_per_second: float = 2.0,
        burst_size: int = 5,
        use_cache: bool = True,
        cache_ttl: int = 86400  # 24 hours default cache
    ):
        """Initialize the Cochrane Library client.
        
        Args:
            base_url: Base URL for the Cochrane Library
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            api_key: API key for Cochrane Library (if available)
            requests_per_second: Maximum number of requests per second
            burst_size: Maximum burst size for rate limiting
            use_cache: Whether to use caching
            cache_ttl: Cache TTL in seconds
        """
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.api_key = api_key
        self.use_cache = use_cache
        self.cache_ttl = cache_ttl
        
        # Initialize rate limiter
        self.rate_limiter = AsyncRateLimiter(
            requests_per_second=requests_per_second,
            burst_size=burst_size
        )
        
        # Initialize HTTP client
        self.client = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True
        )
        
        # Set default headers
        self.headers = {
            "Accept": "application/json",
            "User-Agent": "ASF-Medical-Research-Synthesizer/CochraneClient",
        }
        
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        
        logger.info(f"Initialized Cochrane client with base URL: {base_url}")
        
    async def check_api_status(self) -> Dict[str, Any]:
        """Check the status of the Cochrane Library API.
        
        Returns:
            Dictionary with API status information
        """
        try:
            # Try a simple request to check if the API is responsive
            response = await self._make_request(
                f"{self.base_url}/rest/api/status",
                params=None,
                method="GET",
                return_json=True
            )
            
            # If we got here, the API is responsive
            status = {
                "status": "ok",
                "message": "Cochrane Library API is responsive",
                "details": response if isinstance(response, dict) else {}
            }
            logger.info(f"Cochrane API status check: {status['status']}")
            return status
        except CochraneClientError as e:
            # API returned an error
            status = {
                "status": "error",
                "message": f"Cochrane Library API returned an error: {str(e)}",
                "error_code": e.status_code
            }
            logger.warning(f"Cochrane API status check: {status['status']} - {status['message']}")
            return status
        except Exception as e:
            # Other error (network, etc)
            status = {
                "status": "error",
                "message": f"Error connecting to Cochrane Library API: {str(e)}",
                "error_type": type(e).__name__
            }
            logger.error(f"Cochrane API status check: {status['status']} - {status['message']}")
            return status
    
    def get_rate_limit(self) -> Dict[str, Any]:
        """Get the current rate limit settings.
        
        Returns:
            Dictionary with rate limit settings
        """
        return {
            "requests_per_second": self.rate_limiter.requests_per_second,
            "burst_size": self.rate_limiter.burst_size
        }
    
    def update_rate_limit(self, requests_per_second: Optional[float] = None) -> Dict[str, Any]:
        """Update the rate limit settings.
        
        Args:
            requests_per_second: New requests per second limit
            
        Returns:
            Dictionary with updated rate limit settings
        """
        if requests_per_second is not None:
            if requests_per_second <= 0:
                raise ValueError("requests_per_second must be positive")
            self.rate_limiter.update_rate(requests_per_second)
            logger.info(f"Updated rate limit to {requests_per_second} requests per second")
        
        return self.get_rate_limit()
    
    def update_cache_settings(self, use_cache: Optional[bool] = None, cache_ttl: Optional[int] = None) -> Dict[str, Any]:
        """Update the cache settings.
        
        Args:
            use_cache: Whether to use caching
            cache_ttl: Cache TTL in seconds
            
        Returns:
            Dictionary with updated cache settings
        """
        if use_cache is not None:
            self.use_cache = use_cache
        
        if cache_ttl is not None:
            if cache_ttl < 0:
                raise ValueError("cache_ttl must be non-negative")
            self.cache_ttl = cache_ttl
        
        logger.info(f"Updated cache settings: use_cache={self.use_cache}, cache_ttl={self.cache_ttl}")
        
        return {
            "use_cache": self.use_cache,
            "cache_ttl": self.cache_ttl
        }
    
    async def count_cached_items(self, pattern: Optional[str] = None) -> Dict[str, Any]:
        """Count the number of cached items.
        
        Args:
            pattern: Pattern to match cache keys
            
        Returns:
            Dictionary with cache count information
        """
        if pattern is None:
            pattern = "cochrane:*"
        elif not pattern.startswith("cochrane:"):
            pattern = f"cochrane:{pattern}"
        
        try:
            count = await enhanced_cache_manager.count(pattern)
            return {
                "pattern": pattern,
                "count": count
            }
        except Exception as e:
            logger.error(f"Error counting cached items: {str(e)}")
            return {
                "pattern": pattern,
                "count": 0,
                "error": str(e)
            }
    
    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration of the client.
        
        Returns:
            Dictionary with current configuration
        """
        return {
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "has_api_key": self.api_key is not None,
            "use_cache": self.use_cache,
            "cache_ttl": self.cache_ttl,
            "rate_limit": self.get_rate_limit()
        }
    
    async def clear_cache(self, pattern: Optional[str] = None) -> int:
        """Clear the cache.
        
        Args:
            pattern: Pattern to match cache keys to clear
            
        Returns:
            Number of cache keys cleared
        """
        if pattern is None:
            pattern = "cochrane:*"
        elif not pattern.startswith("cochrane:"):
            pattern = f"cochrane:{pattern}"
        
        try:
            count = await enhanced_cache_manager.delete_pattern(pattern)
            logger.info(f"Cleared {count} items from cache with pattern {pattern}")
            return count
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            return 0
    
    async def _make_request(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        method: str = "GET",
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        return_json: bool = True,
        use_cache: Optional[bool] = None
    ) -> Any:
        """Make a request to the Cochrane Library API with rate limiting and retries.
        
        Args:
            url: URL to request
            params: Query parameters
            method: HTTP method (GET, POST, etc.)
            data: Request body for POST/PUT requests
            headers: Additional headers
            return_json: Whether to return JSON
            use_cache: Whether to use cache (overrides client setting)
            
        Returns:
            Response data (JSON or text)
            
        Raises:
            CochraneClientError: If the request fails
        """
        use_cache_for_request = self.use_cache if use_cache is None else use_cache
        
        # Generate cache key if needed
        cache_key = None
        if use_cache_for_request and method.upper() == "GET":
            cache_key = f"cochrane:{hashlib.md5(f'{url}:{json.dumps(params or {})}'.encode()).hexdigest()}"
            
            # Try to get from cache
            cached_data = await enhanced_cache_manager.get(cache_key)
            if cached_data is not None:
                logger.debug(f"Cache hit for {url}")
                return cached_data
        
        # Apply rate limiting
        await self.rate_limiter.acquire()
        
        # Set headers
        request_headers = self.headers.copy()
        if headers:
            request_headers.update(headers)
        
        # Make request with retries
        for attempt in range(self.max_retries + 1):
            try:
                if method.upper() == "GET":
                    response = await self.client.get(url, params=params, headers=request_headers)
                elif method.upper() == "POST":
                    response = await self.client.post(url, params=params, json=data, headers=request_headers)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                # Handle response
                response.raise_for_status()
                
                if return_json:
                    try:
                        result = response.json()
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse JSON response from {url}")
                        result = response.text
                else:
                    result = response.text
                
                # Cache result if needed
                if cache_key and use_cache_for_request:
                    await enhanced_cache_manager.set(cache_key, result, ttl=self.cache_ttl)
                
                return result
                
            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                error_text = e.response.text
                
                # Handle specific status codes
                if status_code == 429:  # Too Many Requests
                    if attempt < self.max_retries:
                        wait_time = min(2 ** attempt, 60)  # Exponential backoff with cap
                        logger.warning(f"Rate limited by Cochrane API. Retrying in {wait_time} seconds.")
                        await asyncio.sleep(wait_time)
                        continue
                
                # For other status codes, build a detailed error message
                try:
                    error_json = e.response.json()
                    error_message = error_json.get("message", str(e))
                except (json.JSONDecodeError, AttributeError):
                    error_message = error_text or str(e)
                
                logger.error(f"HTTP error {status_code} from Cochrane API: {error_message}")
                raise CochraneClientError(f"HTTP error {status_code}: {error_message}", status_code=status_code)
                
            except httpx.RequestError as e:
                # Network errors, timeouts, etc.
                if attempt < self.max_retries:
                    wait_time = min(2 ** attempt, 60)
                    logger.warning(f"Request error: {str(e)}. Retrying in {wait_time} seconds.")
                    await asyncio.sleep(wait_time)
                    continue
                
                logger.error(f"Request failed after {self.max_retries} retries: {str(e)}")
                raise CochraneClientError(f"Request failed: {str(e)}")
        
        # This should never be reached, but just in case
        raise CochraneClientError(f"Request failed after {self.max_retries} retries")
    
    @enhanced_cached(prefix="cochrane:search", ttl=3600)  # Cache for 1 hour
    async def search(
        self,
        query: str,
        max_results: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        page: int = 1
    ) -> Dict[str, Any]:
        """Search the Cochrane Library.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            filters: Search filters (e.g., publication_year, type)
            page: Page number
            
        Returns:
            Dictionary with search results
        """
        if not query:
            raise ValidationError("Query cannot be empty")
        
        logger.info(f"Searching Cochrane Library for: {query}")
        
        # Prepare parameters
        params = {
            "q": query,
            "page": page,
            "pageSize": min(max_results, 100),  # Respect API limits
        }
        
        # Add filters if provided
        if filters:
            for key, value in filters.items():
                if value is not None:
                    params[key] = value
        
        # Make request
        response = await self._make_request(
            f"{self.base_url}/rest/search",
            params=params,
            method="GET",
            return_json=True
        )
        
        # Process results
        if isinstance(response, dict):
            # Format the results in a standardized way
            return {
                "query": query,
                "total_results": response.get("totalResults", 0),
                "page": page,
                "page_size": min(max_results, 100),
                "results": response.get("results", [])
            }
        else:
            logger.warning(f"Unexpected response format from Cochrane API: {type(response)}")
            return {
                "query": query,
                "total_results": 0,
                "page": page,
                "page_size": min(max_results, 100),
                "results": []
            }
    
    @enhanced_cached(prefix="cochrane:review", ttl=86400)  # Cache for 24 hours
    async def get_review(self, review_id: str) -> Dict[str, Any]:
        """Get a Cochrane review by ID.
        
        Args:
            review_id: The ID of the review
            
        Returns:
            Dictionary with review details
        """
        if not review_id:
            raise ValidationError("Review ID cannot be empty")
        
        logger.info(f"Getting Cochrane review: {review_id}")
        
        # Make request
        response = await self._make_request(
            f"{self.base_url}/rest/reviews/{review_id}",
            method="GET",
            return_json=True
        )
        
        return response
    
    @enhanced_cached(prefix="cochrane:extract_pico", ttl=86400)  # Cache for 24 hours
    async def extract_pico(self, review_id: str) -> Dict[str, Any]:
        """Extract PICO elements from a Cochrane review.
        
        Args:
            review_id: The ID of the review
            
        Returns:
            Dictionary with PICO elements
        """
        if not review_id:
            raise ValidationError("Review ID cannot be empty")
        
        logger.info(f"Extracting PICO from Cochrane review: {review_id}")
        
        # Get the full review first
        review = await self.get_review(review_id)
        
        # Extract PICO elements
        pico = {
            "population": [],
            "intervention": [],
            "comparison": [],
            "outcome": []
        }
        
        # Process the review to extract PICO elements
        # This is a simplified implementation - in a real client, you would
        # use NLP or other techniques to extract PICO elements from the review text
        
        # Extract population
        if "participants" in review:
            pico["population"].append(PICOElement(review["participants"], 0.8).__dict__)
        
        # Extract interventions
        if "interventions" in review:
            pico["intervention"].append(PICOElement(review["interventions"], 0.8).__dict__)
        
        # Extract comparisons
        if "comparisons" in review:
            pico["comparison"].append(PICOElement(review["comparisons"], 0.8).__dict__)
        
        # Extract outcomes
        if "outcomes" in review:
            pico["outcome"].append(PICOElement(review["outcomes"], 0.8).__dict__)
        
        return {
            "review_id": review_id,
            "pico": pico
        }
    
    @enhanced_cached(prefix="cochrane:evidence", ttl=86400)  # Cache for 24 hours
    async def extract_evidence(self, review_id: str) -> Dict[str, Any]:
        """Extract evidence ratings from a Cochrane review.
        
        Args:
            review_id: The ID of the review
            
        Returns:
            Dictionary with evidence information
        """
        if not review_id:
            raise ValidationError("Review ID cannot be empty")
        
        logger.info(f"Extracting evidence from Cochrane review: {review_id}")
        
        # Get the full review first
        review = await self.get_review(review_id)
        
        # Process the review to extract evidence ratings
        # In a real implementation, this would analyze the text to find GRADE ratings
        evidence = {
            "certainty_ratings": [],
            "summary_of_findings": {},
            "grade_assessment": {}
        }
        
        # Look for evidence certainty ratings in the review
        if "qualityOfEvidence" in review:
            evidence["certainty_ratings"].append(
                EvidenceGrade(
                    grade=review["qualityOfEvidence"], 
                    explanation="Extracted from review quality of evidence field", 
                    confidence=0.9
                ).__dict__
            )
        
        # Extract summary of findings if available
        if "summaryOfFindings" in review:
            evidence["summary_of_findings"] = review["summaryOfFindings"]
        
        # Extract GRADE assessment
        # This would be more complex in a real implementation
        if "gradeAssessment" in review:
            evidence["grade_assessment"] = review["gradeAssessment"]
        
        return {
            "review_id": review_id,
            "evidence": evidence
        }
    
    @enhanced_cached(prefix="cochrane:search_pico", ttl=3600)  # Cache for 1 hour
    async def search_by_pico(
        self,
        population: Optional[str] = None,
        intervention: Optional[str] = None,
        comparison: Optional[str] = None,
        outcome: Optional[str] = None,
        max_results: int = 20,
        page: int = 1
    ) -> Dict[str, Any]:
        """Search the Cochrane Library by PICO elements.
        
        Args:
            population: Population or patient group
            intervention: Intervention
            comparison: Comparison or control
            outcome: Outcome
            max_results: Maximum number of results to return
            page: Page number
            
        Returns:
            Dictionary with search results
        """
        # Build query string from PICO elements
        query_parts = []
        
        if population:
            query_parts.append(f"population:({population})")
        
        if intervention:
            query_parts.append(f"intervention:({intervention})")
        
        if comparison:
            query_parts.append(f"comparison:({comparison})")
        
        if outcome:
            query_parts.append(f"outcome:({outcome})")
        
        if not query_parts:
            raise ValidationError("At least one PICO element must be provided")
        
        query = " AND ".join(query_parts)
        logger.info(f"Searching Cochrane Library with PICO query: {query}")
        
        return await self.search(
            query=query,
            max_results=max_results,
            filters={"type": "systematic-review"},
            page=page
        )
    
    async def get_recent_reviews(self, count: int = 10) -> Dict[str, Any]:
        """Get recent Cochrane reviews.
        
        Args:
            count: Number of recent reviews to retrieve
            
        Returns:
            Dictionary with recent reviews
        """
        logger.info(f"Getting {count} recent Cochrane reviews")
        
        # Search for recent reviews
        return await self.search(
            query="*",
            max_results=count,
            filters={"type": "systematic-review", "sort": "date-desc"},
            page=1
        )
    
    async def get_top_cited_reviews(self, count: int = 10) -> Dict[str, Any]:
        """Get top cited Cochrane reviews.
        
        Args:
            count: Number of top cited reviews to retrieve
            
        Returns:
            Dictionary with top cited reviews
        """
        logger.info(f"Getting {count} top cited Cochrane reviews")
        
        # Search for top cited reviews
        return await self.search(
            query="*",
            max_results=count,
            filters={"type": "systematic-review", "sort": "citations-desc"},
            page=1
        )
    
    @enhanced_cached(prefix="cochrane:topics", ttl=86400*7)  # Cache for 1 week
    async def get_topics(self) -> List[Dict[str, Any]]:
        """Get list of Cochrane topics/categories.
        
        Returns:
            List of topics
        """
        logger.info("Getting Cochrane topics")
        
        # Make request
        response = await self._make_request(
            f"{self.base_url}/rest/topics",
            method="GET",
            return_json=True
        )
        
        if isinstance(response, dict) and "topics" in response:
            return response["topics"]
        elif isinstance(response, list):
            return response
        else:
            logger.warning(f"Unexpected response format from Cochrane API for topics: {type(response)}")
            return []
    
    @enhanced_cached(prefix="cochrane:html_content", ttl=86400)  # Cache for 24 hours
    async def get_review_html_content(self, review_id: str) -> Dict[str, str]:
        """Get HTML content of a Cochrane review.
        
        Args:
            review_id: The ID of the review
            
        Returns:
            Dictionary with HTML sections of the review
        """
        if not review_id:
            raise ValidationError("Review ID cannot be empty")
        
        logger.info(f"Getting HTML content for Cochrane review: {review_id}")
        
        # Make request
        html_content = await self._make_request(
            f"{self.base_url}/cdsr/doi/10.1002/14651858.{review_id}/full",
            method="GET",
            return_json=False,
            use_cache=True
        )
        
        # Parse HTML content using BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract different sections
        sections = {}
        
        # Abstract
        abstract_elem = soup.select_one("#abstract-content")
        if abstract_elem:
            sections["abstract"] = abstract_elem.get_text(strip=True)
        
        # Background
        background_elem = soup.select_one("#background-section")
        if background_elem:
            sections["background"] = background_elem.get_text(strip=True)
        
        # Methods
        methods_elem = soup.select_one("#methods-section")
        if methods_elem:
            sections["methods"] = methods_elem.get_text(strip=True)
        
        # Results
        results_elem = soup.select_one("#results-section")
        if results_elem:
            sections["results"] = results_elem.get_text(strip=True)
        
        # Discussion
        discussion_elem = soup.select_one("#discussion-section")
        if discussion_elem:
            sections["discussion"] = discussion_elem.get_text(strip=True)
        
        # Authors' conclusions
        conclusions_elem = soup.select_one("#conclusions-section")
        if conclusions_elem:
            sections["conclusions"] = conclusions_elem.get_text(strip=True)
        
        return {
            "review_id": review_id,
            "sections": sections
        }
    
    async def get_review_with_full_content(self, review_id: str) -> Dict[str, Any]:
        """Get a Cochrane review with full content including metadata, PICO, and evidence.
        
        Args:
            review_id: The ID of the review
            
        Returns:
            Dictionary with complete review information
        """
        logger.info(f"Getting complete data for Cochrane review: {review_id}")
        
        # Get review metadata
        metadata = await self.get_review(review_id)
        
        # Get PICO elements
        pico = await self.extract_pico(review_id)
        
        # Get evidence ratings
        evidence = await self.extract_evidence(review_id)
        
        # Get HTML content sections
        content = await self.get_review_html_content(review_id)
        
        # Combine all information
        result = {
            "review_id": review_id,
            "metadata": metadata,
            "pico": pico.get("pico", {}),
            "evidence": evidence.get("evidence", {}),
            "content": content.get("sections", {})
        }
        
        return result
    
    async def close(self):
        """Close the client."""
        await self.client.aclose()
        logger.info("Closed Cochrane client")