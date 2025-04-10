"""
NCBI client for the Medical Research Synthesizer.

This module provides a client for interacting with the NCBI E-utilities API.
"""

import asyncio
import json
import logging
import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Any, Union, Tuple
import httpx
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

from asf.medical.core.config import settings
from asf.medical.core.rate_limiter import AsyncRateLimiter

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
        base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/",
        max_batch_size: int = 200,
        timeout: float = 30.0
    ):
        """
        Initialize the NCBI client.

        Args:
            email: Email address for NCBI API (required)
            api_key: API key for NCBI API (optional)
            tool: Tool name for NCBI API (default: "MedicalResearchSynthesizer")
            base_url: Base URL for NCBI API (default: "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/")
            max_batch_size: Maximum batch size for requests (default: 200)
            timeout: Request timeout in seconds (default: 30.0)
        """
        self.email = email or settings.NCBI_EMAIL
        self.api_key = api_key or (settings.NCBI_API_KEY.get_secret_value() if settings.NCBI_API_KEY else None)
        self.tool = tool
        self.base_url = base_url
        self.max_batch_size = max_batch_size
        self.client = httpx.AsyncClient(timeout=timeout)

        # Rate limiting
        requests_per_second = 10 if self.api_key else 3
        self.rate_limiter = AsyncRateLimiter(requests_per_second=requests_per_second)

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def _batch_pmids(self, pmids: List[str], batch_size: Optional[int] = None) -> List[List[str]]:
        """
        Split a list of PMIDs into batches.

        Args:
            pmids: List of PMIDs
            batch_size: Batch size (default: self.max_batch_size)

        Returns:
            List of PMID batches
        """
        batch_size = batch_size or self.max_batch_size
        return [pmids[i:i + batch_size] for i in range(0, len(pmids), batch_size)]

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((httpx.HTTPError, asyncio.TimeoutError))
    )
    async def _make_request(self, endpoint: str, params: Dict[str, Any], return_json: bool = True) -> Union[Dict[str, Any], str]:
        """
        Make a request to the NCBI API.

        Args:
            endpoint: API endpoint
            params: Request parameters
            return_json: Whether to return JSON (default: True)

        Returns:
            Response data as JSON or text

        Raises:
            httpx.HTTPError: If the request fails
        """
        # Add common parameters
        params.update({
            "tool": self.tool,
            "email": self.email,
        })

        # Add retmode if not specified
        if "retmode" not in params:
            params["retmode"] = "json" if return_json else "xml"

        # Add API key if available
        if self.api_key:
            params["api_key"] = self.api_key

        # Apply rate limiting
        await self.rate_limiter.acquire()

        # Make request
        url = f"{self.base_url}{endpoint}"
        logger.debug(f"Making request to {url} with params {params}")

        response = await self.client.get(url, params=params)
        response.raise_for_status()

        return response.json() if return_json else response.text

    async def search_pubmed(
        self,
        query: str,
        max_results: int = 20,
        sort: str = "relevance",
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
        use_history: bool = False
    ) -> Dict[str, Any]:
        """
        Search PubMed with the given query.

        Args:
            query: Search query
            max_results: Maximum number of results to return (default: 20)
            sort: Sort order (default: "relevance")
            min_date: Minimum date (YYYY/MM/DD format)
            max_date: Maximum date (YYYY/MM/DD format)
            use_history: Whether to use the Entrez History server (default: False)

        Returns:
            Search results including PMIDs and metadata
        """
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "sort": sort,
            "usehistory": "y" if use_history else "n"
        }

        # Add date range if provided
        if min_date and max_date:
            params["datetype"] = "pdat"  # Publication date
            params["mindate"] = min_date
            params["maxdate"] = max_date

        try:
            data = await self._make_request("esearch.fcgi", params)
            return data
        except Exception as e:
            logger.error(f"Error searching PubMed: {str(e)}")
            raise

    async def fetch_article_details(
        self,
        pmids: Union[str, List[str]],
        db: str = "pubmed",
        batch_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch article details from NCBI.

        Args:
            pmids: PMID or list of PMIDs
            db: Database to search (default: "pubmed")
            batch_size: Batch size for requests (default: self.max_batch_size)

        Returns:
            List of article details
        """
        if isinstance(pmids, str):
            pmids = [pmids]

        if not pmids:
            return []

        # Split PMIDs into batches
        batches = await self._batch_pmids(pmids, batch_size)
        all_articles = []

        # Process each batch
        for batch in batches:
            params = {
                "db": db,
                "id": ",".join(batch)
            }

            try:
                data = await self._make_request("esummary.fcgi", params)

                # Extract article details
                batch_articles = []
                for pmid in batch:
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

                        batch_articles.append(article)

                all_articles.extend(batch_articles)
            except Exception as e:
                logger.error(f"Error fetching article details from NCBI (batch {len(all_articles)//batch_size + 1}): {str(e)}")
                # Continue with the next batch instead of failing completely
                continue

        return all_articles

    async def fetch_article_abstracts(
        self,
        pmids: Union[str, List[str]],
        db: str = "pubmed",
        batch_size: Optional[int] = None
    ) -> Dict[str, str]:
        """
        Fetch article abstracts from NCBI.

        Args:
            pmids: PMID or list of PMIDs
            db: Database to search (default: "pubmed")
            batch_size: Batch size for requests (default: self.max_batch_size)

        Returns:
            Dictionary mapping PMIDs to abstracts
        """
        if isinstance(pmids, str):
            pmids = [pmids]

        if not pmids:
            return {}

        # Split PMIDs into batches
        batches = await self._batch_pmids(pmids, batch_size)
        all_abstracts = {}

        # Process each batch
        for batch in batches:
            params = {
                "db": db,
                "id": ",".join(batch),
                "retmode": "xml"  # XML format is required for abstracts
            }

            try:
                # For abstracts, we need to use efetch and parse XML
                xml_text = await self._make_request("efetch.fcgi", params, return_json=False)

                try:
                    # Parse XML using ElementTree
                    root = ET.fromstring(xml_text)

                    # Find all PubmedArticle elements
                    for article in root.findall("./PubmedArticle"):
                        # Get PMID
                        pmid_elem = article.find(".//PMID")
                        if pmid_elem is None:
                            continue

                        pmid = pmid_elem.text

                        # Get abstract
                        abstract_elems = article.findall(".//AbstractText")
                        if not abstract_elems:
                            continue

                        # Combine all abstract sections
                        abstract_text = " ".join(elem.text or "" for elem in abstract_elems)

                        # Add to abstracts dictionary
                        all_abstracts[pmid] = abstract_text
                except ET.ParseError as e:
                    logger.error(f"Error parsing XML response: {str(e)}")

                    # Fallback to simple string parsing if XML parsing fails
                    for pmid in batch:
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
                        all_abstracts[pmid] = abstract_text
            except Exception as e:
                logger.error(f"Error fetching article abstracts from NCBI (batch {len(all_abstracts)//batch_size + 1}): {str(e)}")
                # Continue with the next batch instead of failing completely
                continue

        return all_abstracts

    async def fetch_pubmed_abstracts(
        self,
        id_list: Optional[List[str]] = None,
        query: Optional[str] = None,
        max_results: int = 20,
        batch_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch PubMed abstracts by ID list or query.

        Args:
            id_list: List of PMIDs (optional)
            query: Search query (optional)
            max_results: Maximum number of results to return (default: 20)
            batch_size: Batch size for requests (default: self.max_batch_size)

        Returns:
            List of articles with abstracts
        """
        # Get PMIDs from query if not provided
        if id_list is None and query is not None:
            search_result = await self.search_pubmed(query, max_results=max_results)
            id_list = search_result.get("esearchresult", {}).get("idlist", [])

        if not id_list:
            return []

        # Fetch article details and abstracts in parallel
        details_task = asyncio.create_task(self.fetch_article_details(id_list, batch_size=batch_size))
        abstracts_task = asyncio.create_task(self.fetch_article_abstracts(id_list, batch_size=batch_size))

        # Wait for both tasks to complete
        articles, abstracts = await asyncio.gather(details_task, abstracts_task)

        # Merge abstracts into articles
        for article in articles:
            pmid = article["pmid"]
            if pmid in abstracts:
                article["abstract"] = abstracts[pmid]

        return articles

    async def search_and_fetch_pubmed(
        self,
        query: str,
        max_results: int = 20,
        batch_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search PubMed and fetch abstracts in one step.

        Args:
            query: Search query
            max_results: Maximum number of results to return (default: 20)
            batch_size: Batch size for requests (default: self.max_batch_size)

        Returns:
            List of articles with abstracts
        """
        return await self.fetch_pubmed_abstracts(
            query=query,
            max_results=max_results,
            batch_size=batch_size
        )
