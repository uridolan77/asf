"""
UMLS client for the Medical Research Synthesizer.
This module provides a client for interacting with the UMLS API.
"""
import logging
from typing import List
import httpx
logger = logging.getLogger(__name__)
class UMLSClient:
    """
    Client for interacting with the UMLS API.
    This client provides methods for searching UMLS concepts and retrieving concept details.
    """
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://uts-ws.nlm.nih.gov/rest"
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
        self.tgt = None
        self.tgt_expires = 0
        self.requests_per_second = 5
        self.last_request_time = 0
    async def close(self):
        Implement rate limiting for UMLS API.
        UMLS recommends no more than 5 requests per second.
        Get a Ticket Granting Ticket (TGT) from the UMLS API.
        Returns:
            TGT URL
        Raises:
            httpx.HTTPError: If the request fails
        Get a Service Ticket (ST) from the UMLS API.
        Returns:
            Service Ticket
        Raises:
            httpx.HTTPError: If the request fails
        Make a request to the UMLS API.
        Args:
            endpoint: API endpoint
            params: Request parameters
        Returns:
            Response data
        Raises:
            httpx.HTTPError: If the request fails
        Search for UMLS concepts.
        Args:
            query: Search query
            search_type: Search type (default: "words")
            max_results: Maximum number of results to return (default: 20)
        Returns:
            List of concept summaries
        Get details for a specific UMLS concept.
        Args:
            concept_id: Concept ID (e.g., "C0012634")
        Returns:
            Concept details
        Get relations for a specific UMLS concept.
        Args:
            concept_id: Concept ID (e.g., "C0012634")
        Returns:
            List of relations