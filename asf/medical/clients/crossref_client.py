"""
CrossRef client for the Medical Research Synthesizer.

This module provides a client for interacting with the CrossRef API.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Union
import httpx
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# Set up logging
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
        """
        Initialize the CrossRef client.
        
        Args:
            email: Email address for CrossRef API (required for polite pool)
            base_url: Base URL for CrossRef API (default: "https://api.crossref.org")
        """
        self.email = email
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # Rate limiting
        self.requests_per_second = 2
        self.last_request_time = 0
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def _rate_limit(self):
        """
        Implement rate limiting for CrossRef API.
        
        CrossRef recommends no more than 2 requests per second.
        """
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        min_interval = 1.0 / self.requests_per_second
        
        if time_since_last_request < min_interval:
            await asyncio.sleep(min_interval - time_since_last_request)
        
        self.last_request_time = time.time()
    
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((httpx.HTTPError, asyncio.TimeoutError))
    )
    async def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Make a request to the CrossRef API.
        
        Args:
            endpoint: API endpoint
            params: Request parameters
            
        Returns:
            Response data
            
        Raises:
            httpx.HTTPError: If the request fails
        """
        # Apply rate limiting
        await self._rate_limit()
        
        # Make request
        url = f"{self.base_url}/{endpoint}"
        
        # Add email to params for polite pool
        if params is None:
            params = {}
        
        headers = {
            "User-Agent": f"MedicalResearchSynthesizer/1.0 (mailto:{self.email})"
        }
        
        logger.debug(f"Making request to {url} with params {params}")
        
        response = await self.client.get(url, params=params, headers=headers)
        response.raise_for_status()
        
        return response.json()
    
    async def get_by_doi(self, doi: str) -> Dict[str, Any]:
        """
        Get article metadata by DOI.
        
        Args:
            doi: DOI (e.g., "10.1056/NEJMoa2001017")
            
        Returns:
            Article metadata
        """
        try:
            data = await self._make_request(f"works/{doi}")
            
            # Extract article metadata
            message = data.get("message", {})
            
            article = {
                "doi": message.get("DOI", ""),
                "title": message.get("title", [""])[0] if message.get("title") else "",
                "journal": message.get("container-title", [""])[0] if message.get("container-title") else "",
                "publisher": message.get("publisher", ""),
                "type": message.get("type", ""),
                "citation_count": message.get("is-referenced-by-count", 0),
                "references_count": message.get("references-count", 0),
                "publication_date": None,
                "authors": [],
                "source": "CrossRef"
            }
            
            # Extract publication date
            if "published" in message and "date-parts" in message["published"]:
                date_parts = message["published"]["date-parts"][0]
                if len(date_parts) >= 3:
                    article["publication_date"] = f"{date_parts[0]}-{date_parts[1]:02d}-{date_parts[2]:02d}"
                elif len(date_parts) >= 2:
                    article["publication_date"] = f"{date_parts[0]}-{date_parts[1]:02d}"
                elif len(date_parts) >= 1:
                    article["publication_date"] = f"{date_parts[0]}"
            
            # Extract authors
            if "author" in message:
                for author in message["author"]:
                    author_name = ""
                    if "given" in author and "family" in author:
                        author_name = f"{author['family']}, {author['given']}"
                    elif "family" in author:
                        author_name = author["family"]
                    
                    if author_name:
                        article["authors"].append(author_name)
            
            return article
        except Exception as e:
            logger.error(f"Error getting article from CrossRef: {str(e)}")
            raise
    
    async def search(
        self,
        query: str,
        max_results: int = 20,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for articles.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return (default: 20)
            filter: Filters to apply (e.g., {"type": "journal-article"})
            
        Returns:
            List of article summaries
        """
        params = {
            "query": query,
            "rows": max_results
        }
        
        # Add filters if provided
        if filter:
            for key, value in filter.items():
                params[f"filter.{key}"] = value
        
        try:
            data = await self._make_request("works", params)
            
            # Extract article summaries
            articles = []
            if "message" in data and "items" in data["message"]:
                for item in data["message"]["items"]:
                    article = {
                        "doi": item.get("DOI", ""),
                        "title": item.get("title", [""])[0] if item.get("title") else "",
                        "journal": item.get("container-title", [""])[0] if item.get("container-title") else "",
                        "publisher": item.get("publisher", ""),
                        "type": item.get("type", ""),
                        "citation_count": item.get("is-referenced-by-count", 0),
                        "references_count": item.get("references-count", 0),
                        "publication_date": None,
                        "authors": [],
                        "source": "CrossRef"
                    }
                    
                    # Extract publication date
                    if "published" in item and "date-parts" in item["published"]:
                        date_parts = item["published"]["date-parts"][0]
                        if len(date_parts) >= 3:
                            article["publication_date"] = f"{date_parts[0]}-{date_parts[1]:02d}-{date_parts[2]:02d}"
                        elif len(date_parts) >= 2:
                            article["publication_date"] = f"{date_parts[0]}-{date_parts[1]:02d}"
                        elif len(date_parts) >= 1:
                            article["publication_date"] = f"{date_parts[0]}"
                    
                    # Extract authors
                    if "author" in item:
                        for author in item["author"]:
                            author_name = ""
                            if "given" in author and "family" in author:
                                author_name = f"{author['family']}, {author['given']}"
                            elif "family" in author:
                                author_name = author["family"]
                            
                            if author_name:
                                article["authors"].append(author_name)
                    
                    articles.append(article)
            
            return articles
        except Exception as e:
            logger.error(f"Error searching CrossRef: {str(e)}")
            raise
