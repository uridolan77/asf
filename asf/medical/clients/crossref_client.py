"""
CrossRef client for the Medical Research Synthesizer.
This module provides a client for interacting with the CrossRef API.
"""
import logging
from typing import List
import httpx
logger = logging.getLogger(__name__)
class CrossRefClient:
    """
    Client for interacting with the CrossRef API.
    This client provides methods for retrieving citation data for articles.
    """
    def __init__(
        self,
        email: str,
        base_url: str = "https://api.crossref.org"
    ):
        self.email = email
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
        self.requests_per_second = 2
        self.last_request_time = 0
    async def close(self):
        """
        Implement rate limiting for CrossRef API.
        CrossRef recommends no more than 2 requests per second.
        Make a request to the CrossRef API.
        Args:
            endpoint: API endpoint
            params: Request parameters
        Returns:
            Response data
        Raises:
            httpx.HTTPError: If the request fails
        Get article metadata by DOI.
        Args:
            doi: DOI (e.g., "10.1056/NEJMoa2001017")
        Returns:
            Article metadata
        Search for articles.
        Args:
            query: Search query
            max_results: Maximum number of results to return (default: 20)
            filter: Filters to apply (e.g., {"type": "journal-article"})
        Returns:
            List of article summaries
        """