"""
NCBI client for the Medical Research Synthesizer.

This module provides a client for interacting with the NCBI E-utilities API.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Union
import httpx
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

from asf.medical.core.config import settings

# Set up logging
logger = logging.getLogger(__name__)

class NCBIClient:
    """
    Client for interacting with the NCBI E-utilities API.
    
    This client provides methods for searching PubMed and retrieving article details.
    """
    
    def __init__(
        self,
        email: Optional[str] = None,
        api_key: Optional[str] = None,
        tool: str = "MedicalResearchSynthesizer",
        base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    ):
        """
        Initialize the NCBI client.
        
        Args:
            email: Email address for NCBI API (required)
            api_key: API key for NCBI API (optional)
            tool: Tool name for NCBI API (default: "MedicalResearchSynthesizer")
            base_url: Base URL for NCBI API (default: "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/")
        """
        self.email = email or settings.NCBI_EMAIL
        self.api_key = api_key or (settings.NCBI_API_KEY.get_secret_value() if settings.NCBI_API_KEY else None)
        self.tool = tool
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # Rate limiting
        self.requests_per_second = 10 if self.api_key else 3
        self.last_request_time = 0
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def _rate_limit(self):
        """
        Implement rate limiting for NCBI API.
        
        NCBI allows 10 requests per second with an API key, or 3 without.
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
    async def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a request to the NCBI API.
        
        Args:
            endpoint: API endpoint
            params: Request parameters
            
        Returns:
            Response data
            
        Raises:
            httpx.HTTPError: If the request fails
        """
        # Add common parameters
        params.update({
            "tool": self.tool,
            "email": self.email,
            "retmode": "json"
        })
        
        # Add API key if available
        if self.api_key:
            params["api_key"] = self.api_key
        
        # Apply rate limiting
        await self._rate_limit()
        
        # Make request
        url = f"{self.base_url}{endpoint}"
        logger.debug(f"Making request to {url} with params {params}")
        
        response = await self.client.get(url, params=params)
        response.raise_for_status()
        
        return response.json()
    
    async def search(
        self,
        query: str,
        db: str = "pubmed",
        max_results: int = 20,
        sort: str = "relevance",
        min_date: Optional[str] = None,
        max_date: Optional[str] = None
    ) -> List[str]:
        """
        Search the NCBI database.
        
        Args:
            query: Search query
            db: Database to search (default: "pubmed")
            max_results: Maximum number of results to return (default: 20)
            sort: Sort order (default: "relevance")
            min_date: Minimum date (YYYY/MM/DD format)
            max_date: Maximum date (YYYY/MM/DD format)
            
        Returns:
            List of PMIDs
        """
        params = {
            "db": db,
            "term": query,
            "retmax": max_results,
            "sort": sort
        }
        
        # Add date range if provided
        if min_date and max_date:
            params["datetype"] = "pdat"  # Publication date
            params["mindate"] = min_date
            params["maxdate"] = max_date
        
        try:
            data = await self._make_request("esearch.fcgi", params)
            return data["esearchresult"]["idlist"]
        except Exception as e:
            logger.error(f"Error searching NCBI: {str(e)}")
            raise
    
    async def fetch_article_details(
        self,
        pmids: Union[str, List[str]],
        db: str = "pubmed"
    ) -> List[Dict[str, Any]]:
        """
        Fetch article details from NCBI.
        
        Args:
            pmids: PMID or list of PMIDs
            db: Database to search (default: "pubmed")
            
        Returns:
            List of article details
        """
        if isinstance(pmids, str):
            pmids = [pmids]
        
        if not pmids:
            return []
        
        params = {
            "db": db,
            "id": ",".join(pmids)
        }
        
        try:
            data = await self._make_request("esummary.fcgi", params)
            
            # Extract article details
            articles = []
            for pmid in pmids:
                if pmid in data["result"]:
                    article_data = data["result"][pmid]
                    
                    # Extract authors
                    authors = []
                    if "authors" in article_data:
                        for author in article_data["authors"]:
                            if "name" in author:
                                authors.append(author["name"])
                    
                    # Extract publication date
                    pub_date = None
                    if "pubdate" in article_data:
                        pub_date = article_data["pubdate"]
                    
                    # Create article object
                    article = {
                        "pmid": pmid,
                        "title": article_data.get("title", ""),
                        "journal": article_data.get("fulljournalname", ""),
                        "publication_date": pub_date,
                        "authors": authors,
                        "doi": article_data.get("elocationid", "").replace("doi: ", "") if "elocationid" in article_data else None,
                        "abstract": "",  # Abstract is not included in esummary, need to use efetch
                        "keywords": [],  # Keywords are not included in esummary, need to use efetch
                        "publication_types": article_data.get("pubtype", []),
                        "source": "PubMed"
                    }
                    
                    articles.append(article)
            
            return articles
        except Exception as e:
            logger.error(f"Error fetching article details from NCBI: {str(e)}")
            raise
    
    async def fetch_article_abstracts(
        self,
        pmids: Union[str, List[str]],
        db: str = "pubmed"
    ) -> Dict[str, str]:
        """
        Fetch article abstracts from NCBI.
        
        Args:
            pmids: PMID or list of PMIDs
            db: Database to search (default: "pubmed")
            
        Returns:
            Dictionary mapping PMIDs to abstracts
        """
        if isinstance(pmids, str):
            pmids = [pmids]
        
        if not pmids:
            return {}
        
        params = {
            "db": db,
            "id": ",".join(pmids),
            "retmode": "xml"  # XML format is required for abstracts
        }
        
        try:
            # For abstracts, we need to use efetch and parse XML
            await self._rate_limit()
            url = f"{self.base_url}efetch.fcgi"
            
            # Add common parameters
            params.update({
                "tool": self.tool,
                "email": self.email
            })
            
            # Add API key if available
            if self.api_key:
                params["api_key"] = self.api_key
            
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            
            # Parse XML response
            # This is a simplified parser, a real implementation would use a proper XML parser
            xml_text = response.text
            abstracts = {}
            
            for pmid in pmids:
                # Find the abstract for this PMID
                pmid_start = xml_text.find(f"<PMID>{pmid}</PMID>")
                if pmid_start == -1:
                    continue
                
                # Find the abstract text
                abstract_start = xml_text.find("<AbstractText>", pmid_start)
                if abstract_start == -1:
                    continue
                
                abstract_end = xml_text.find("</AbstractText>", abstract_start)
                if abstract_end == -1:
                    continue
                
                # Extract the abstract text
                abstract_text = xml_text[abstract_start + 14:abstract_end]
                abstracts[pmid] = abstract_text
            
            return abstracts
        except Exception as e:
            logger.error(f"Error fetching article abstracts from NCBI: {str(e)}")
            raise
    
    async def fetch_complete_articles(
        self,
        pmids: Union[str, List[str]],
        db: str = "pubmed"
    ) -> List[Dict[str, Any]]:
        """
        Fetch complete article details including abstracts from NCBI.
        
        Args:
            pmids: PMID or list of PMIDs
            db: Database to search (default: "pubmed")
            
        Returns:
            List of complete article details
        """
        if isinstance(pmids, str):
            pmids = [pmids]
        
        if not pmids:
            return []
        
        # Fetch basic article details
        articles = await self.fetch_article_details(pmids, db)
        
        # Fetch abstracts
        abstracts = await self.fetch_article_abstracts(pmids, db)
        
        # Merge abstracts into articles
        for article in articles:
            pmid = article["pmid"]
            if pmid in abstracts:
                article["abstract"] = abstracts[pmid]
        
        return articles
