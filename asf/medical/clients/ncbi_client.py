"""
NCBI client for the Medical Research Synthesizer.

This module provides a client for interacting with the NCBI E-utilities API.
"""

import asyncio
import json
import logging
import hashlib
import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Any, Union
import httpx

from asf.medical.core.config import settings
from asf.medical.core.rate_limiter import AsyncRateLimiter
from asf.medical.core.cache import cache_manager, cached
from asf.medical.core.exceptions import ExternalServiceError, ValidationError

# Set up logging
logger = logging.getLogger(__name__)

class NCBIClientError(ExternalServiceError):
    """Exception raised for NCBI client errors."""

    def __init__(self, message: str = "NCBI API error", status_code: Optional[int] = None):
        self.status_code = status_code
        super().__init__("NCBI API", message)


class NCBIClient:
    """
    Client for interacting with the NCBI E-utilities API.

    This client provides methods for searching PubMed and retrieving article details.
    It includes retry logic, rate limiting, and caching to improve reliability and performance.
    """

    def __init__(
        self,
        email: Optional[str] = None,
        api_key: Optional[str] = None,
        tool: str = "MedicalResearchSynthesizer",
        base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/",
        max_batch_size: int = 200,
        timeout: float = 30.0,
        max_retries: int = 3,
        cache_ttl: int = 3600,  # 1 hour cache TTL by default
        use_cache: bool = True
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
            max_retries: Maximum number of retries for failed requests (default: 3)
            cache_ttl: Cache TTL in seconds (default: 3600)
            use_cache: Whether to use caching (default: True)

        Raises:
            ValidationError: If email is not provided and not in settings
            ConfigurationError: If API configuration is invalid
        """
        # Validate email
        self.email = email or settings.NCBI_EMAIL
        if not self.email:
            raise ValidationError("Email is required for NCBI API")

        # Get API key from settings if not provided
        self.api_key = api_key or (settings.NCBI_API_KEY.get_secret_value() if settings.NCBI_API_KEY else None)

        # Set other parameters
        self.tool = tool
        self.base_url = base_url
        self.max_batch_size = max_batch_size
        self.max_retries = max_retries
        self.cache_ttl = cache_ttl
        self.use_cache = use_cache

        # Initialize HTTP client with timeout and limits
        self.client = httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )

        # Configure rate limiting based on whether we have an API key
        # NCBI allows 10 requests/second with an API key, 3 without
        requests_per_second = 10 if self.api_key else 3
        self.rate_limiter = AsyncRateLimiter(requests_per_second=requests_per_second)

        logger.info(f"Initialized NCBI client with email={self.email}, api_key={'*****' if self.api_key else 'None'}, cache_ttl={cache_ttl}s")

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def check_api_status(self) -> Dict[str, Any]:
        """
        Check the status of the NCBI API.

        Returns:
            Dictionary with API status information
        """
        try:
            # Make a simple request to check if the API is available
            params = {
                "db": "pubmed",
                "term": "test",
                "retmax": 1
            }

            start_time = time.time()
            data = await self._make_request("esearch.fcgi", params)
            response_time = time.time() - start_time

            # Check if the response is valid
            if "esearchresult" in data:
                status = "ok"
                message = "API is available"
            else:
                status = "warning"
                message = "API response is missing expected fields"

            return {
                "status": status,
                "message": message,
                "response_time": round(response_time, 3),
                "api_key_used": self.api_key is not None,
                "rate_limit": self.rate_limiter.requests_per_second,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        except NCBIClientError as e:
            return {
                "status": "error",
                "message": f"API error: {str(e)}",
                "status_code": getattr(e, 'status_code', None),
                "api_key_used": self.api_key is not None,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Unexpected error: {str(e)}",
                "api_key_used": self.api_key is not None,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

    def get_rate_limit(self) -> Dict[str, Any]:
        """
        Get the current rate limit settings.

        Returns:
            Dictionary with rate limit information
        """
        return {
            "requests_per_second": self.rate_limiter.requests_per_second,
            "api_key_used": self.api_key is not None,
            "max_batch_size": self.max_batch_size,
            "max_retries": self.max_retries
        }

    def update_rate_limit(self, requests_per_second: Optional[float] = None) -> Dict[str, Any]:
        """
        Update the rate limit settings.

        Args:
            requests_per_second: New rate limit in requests per second

        Returns:
            Dictionary with updated rate limit information

        Raises:
            ValidationError: If requests_per_second is invalid
        """
        if requests_per_second is not None:
            if requests_per_second <= 0:
                raise ValidationError("requests_per_second must be positive")

            # Update the rate limiter
            old_rate = self.rate_limiter.requests_per_second
            self.rate_limiter.requests_per_second = requests_per_second
            logger.info(f"Updated rate limit from {old_rate} to {requests_per_second} requests per second")

        # Return the updated rate limit
        return self.get_rate_limit()

    def update_cache_settings(self, use_cache: Optional[bool] = None, cache_ttl: Optional[int] = None) -> Dict[str, Any]:
        """
        Update the cache settings.

        Args:
            use_cache: Whether to use caching
            cache_ttl: Cache TTL in seconds

        Returns:
            Dictionary with updated cache settings

        Raises:
            ValidationError: If cache_ttl is invalid
        """
        if use_cache is not None:
            old_use_cache = self.use_cache
            self.use_cache = use_cache
            logger.info(f"Updated use_cache from {old_use_cache} to {use_cache}")

        if cache_ttl is not None:
            if cache_ttl < 0:
                raise ValidationError("cache_ttl must be non-negative")

            old_cache_ttl = self.cache_ttl
            self.cache_ttl = cache_ttl
            logger.info(f"Updated cache_ttl from {old_cache_ttl} to {cache_ttl} seconds")

        # Return the updated cache settings
        return {
            "use_cache": self.use_cache,
            "cache_ttl": self.cache_ttl
        }

    async def count_cached_items(self, pattern: Optional[str] = None) -> Dict[str, Any]:
        """
        Count the number of cached items for this client.

        Args:
            pattern: Optional pattern to match cache keys (e.g., "pubmed_search:*")
                     If None, counts all NCBI-related cache entries

        Returns:
            Dictionary with cache counts
        """
        if not self.use_cache:
            logger.warning("Cache is disabled for this client")
            return {"enabled": False, "total": 0}

        try:
            # If no pattern is provided, count all NCBI-related cache entries
            if pattern is None:
                patterns = [
                    "ncbi:*",              # Direct API requests
                    "pubmed_search:*",      # Search results
                    "pubmed_abstracts:*",   # Article abstracts
                    "pubmed_search_fetch:*", # Combined search and fetch
                    "article_details:*",     # Article details
                    "article_abstracts:*"    # Article abstracts
                ]

                counts = {}
                total = 0
                for p in patterns:
                    count = await cache_manager.count_pattern(p)
                    counts[p] = count
                    total += count

                return {
                    "enabled": True,
                    "total": total,
                    "patterns": counts,
                    "ttl": self.cache_ttl
                }
            else:
                # Count cache entries matching the specified pattern
                count = await cache_manager.count_pattern(pattern)
                return {
                    "enabled": True,
                    "total": count,
                    "pattern": pattern,
                    "ttl": self.cache_ttl
                }
        except Exception as e:
            logger.error(f"Error counting cached items: {str(e)}")
            return {"enabled": self.use_cache, "error": str(e), "total": 0}

    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration of the client.

        Returns:
            Dictionary with client configuration
        """
        return {
            "email": self.email,
            "api_key_used": self.api_key is not None,
            "tool": self.tool,
            "base_url": self.base_url,
            "max_batch_size": self.max_batch_size,
            "max_retries": self.max_retries,
            "use_cache": self.use_cache,
            "cache_ttl": self.cache_ttl,
            "rate_limit": self.rate_limiter.requests_per_second
        }

    @cached(prefix="ncbi_einfo", data_type="search")
    async def get_database_info(self, db: Optional[str] = None, version: str = "2.0") -> Dict[str, Any]:
        """
        Get information about NCBI databases or a specific database.

        This method uses the EInfo E-utility to retrieve information about available
        Entrez databases or details about a specific database including available
        search fields and links.

        Args:
            db: Database name to get information about. If None, returns a list of all databases.
            version: Version of EInfo XML to return. Default is "2.0" which includes
                     additional fields like IsTruncatable and IsRangeable.

        Returns:
            Dictionary with database information

        Raises:
            NCBIClientError: If the request fails
        """
        params = {}

        if db:
            params["db"] = db

        if version:
            params["version"] = version

        try:
            logger.info(f"Getting database info for {db if db else 'all databases'}")
            result = await self._make_request("einfo.fcgi", params)
            return result
        except Exception as e:
            logger.error(f"Error getting database info: {str(e)}")
            raise NCBIClientError(f"Failed to get database info: {str(e)}")

    @cached(prefix="ncbi_espell", data_type="search")
    async def get_spelling_suggestions(self, term: str, db: str = "pubmed") -> Dict[str, Any]:
        """
        Get spelling suggestions for a search term.

        This method uses the ESpell E-utility to retrieve spelling suggestions
        for terms within a search query.

        Args:
            term: Search term to check spelling for
            db: Database to check spelling against (default: "pubmed")

        Returns:
            Dictionary with spelling suggestions

        Raises:
            ValidationError: If term is empty
            NCBIClientError: If the request fails
        """
        # Validate inputs
        if not term or not term.strip():
            raise ValidationError("Search term cannot be empty")

        params = {
            "db": db,
            "term": term.strip()
        }

        try:
            logger.info(f"Getting spelling suggestions for '{term}' in {db}")
            result = await self._make_request("espell.fcgi", params)
            return result
        except Exception as e:
            logger.error(f"Error getting spelling suggestions: {str(e)}")
            raise NCBIClientError(f"Failed to get spelling suggestions: {str(e)}")

    @cached(prefix="ncbi_egquery", data_type="search")
    async def search_all_databases(self, term: str) -> Dict[str, Any]:
        """
        Search across all Entrez databases with a single query.

        This method uses the EGQuery E-utility to retrieve the number of records
        matching a search term in all available Entrez databases.

        Args:
            term: Search term to use across all databases

        Returns:
            Dictionary with search results for each database

        Raises:
            ValidationError: If term is empty
            NCBIClientError: If the request fails
        """
        # Validate inputs
        if not term or not term.strip():
            raise ValidationError("Search term cannot be empty")

        params = {
            "term": term.strip()
        }

        try:
            logger.info(f"Searching all databases for '{term}'")
            result = await self._make_request("egquery.fcgi", params)
            return result
        except Exception as e:
            logger.error(f"Error searching all databases: {str(e)}")
            raise NCBIClientError(f"Failed to search all databases: {str(e)}")

    async def match_citations(self, citations: List[Dict[str, str]]) -> Dict[str, str]:
        """
        Match citation strings to PubMed IDs.

        This method uses the ECitMatch E-utility to retrieve PubMed IDs (PMIDs)
        that correspond to a set of input citation strings.

        Args:
            citations: List of citation dictionaries, each containing:
                - journal: Journal title
                - year: Publication year
                - volume: Volume number
                - first_page: First page of article
                - author: First author name (last name, initials)
                - key: Optional user-provided key for the citation

        Returns:
            Dictionary mapping user keys to PMIDs

        Raises:
            ValidationError: If citations list is empty or invalid
            NCBIClientError: If the request fails
        """
        # Validate inputs
        if not citations:
            raise ValidationError("Citations list cannot be empty")

        # Format citation strings
        citation_strings = []
        for citation in citations:
            # Validate required fields
            required_fields = ["journal", "year", "volume", "first_page", "author"]
            missing_fields = [field for field in required_fields if field not in citation or not citation[field]]
            if missing_fields:
                raise ValidationError(f"Citation missing required fields: {', '.join(missing_fields)}")

            # Format citation string: journal|year|volume|first_page|author|key|
            key = citation.get("key", f"cit_{len(citation_strings)+1}")
            citation_str = f"{citation['journal']}|{citation['year']}|{citation['volume']}|{citation['first_page']}|{citation['author']}|{key}|"
            citation_strings.append(citation_str)

        # Join citation strings with carriage returns
        bdata = "%0D".join(citation_strings)

        params = {
            "db": "pubmed",
            "retmode": "xml",
            "bdata": bdata
        }

        try:
            logger.info(f"Matching {len(citations)} citations to PMIDs")
            result = await self._make_request("ecitmatch.cgi", params, return_json=False)

            # Parse the results
            # The result is a simple text string with each line containing the original citation string
            # followed by the PMID (or "NOT_FOUND" if no match)
            matches = {}
            lines = result.strip().split("\n")

            for line in lines:
                parts = line.split("|")
                if len(parts) >= 7:  # Original citation has 6 parts plus PMID
                    key = parts[5]
                    pmid = parts[6].strip()
                    matches[key] = pmid if pmid != "NOT_FOUND" else None

            return matches
        except Exception as e:
            logger.error(f"Error matching citations: {str(e)}")
            raise NCBIClientError(f"Failed to match citations: {str(e)}")

    @cached(prefix="ncbi_elink", data_type="search")
    async def get_links(self,
                       ids: Union[str, List[str]],
                       dbfrom: str,
                       db: str,
                       linkname: Optional[str] = None,
                       cmd: str = "neighbor") -> Dict[str, Any]:
        """
        Get links between database records.

        This method uses the ELink E-utility to retrieve links between records
        in different Entrez databases or within the same database.

        Args:
            ids: UID or list of UIDs to find links for
            dbfrom: Source database containing the input UIDs
            db: Target database to find links in
            linkname: Name of the link to retrieve (optional)
            cmd: ELink command (default: "neighbor")
                 Options include:
                 - neighbor: Get linked UIDs
                 - neighbor_score: Get linked UIDs with similarity scores
                 - neighbor_history: Post results to History server
                 - acheck: List all available links
                 - ncheck: Check for existence of links
                 - lcheck: Check for existence of external links
                 - llinks: List URLs for LinkOut providers
                 - llinkslib: List URLs for all LinkOut providers including libraries
                 - prlinks: List URLs for primary LinkOut providers

        Returns:
            Dictionary with link information

        Raises:
            ValidationError: If inputs are invalid
            NCBIClientError: If the request fails
        """
        # Validate inputs
        if not ids:
            raise ValidationError("IDs list cannot be empty")

        if not dbfrom or not db:
            raise ValidationError("Source and target databases must be specified")

        # Convert single ID to list
        if isinstance(ids, str):
            id_list = [ids]
        else:
            id_list = ids

        # Prepare parameters
        params = {
            "dbfrom": dbfrom,
            "db": db,
            "id": ",".join(id_list),
            "cmd": cmd
        }

        # Add optional linkname if provided
        if linkname:
            params["linkname"] = linkname

        try:
            logger.info(f"Getting links from {dbfrom} to {db} for {len(id_list)} IDs")
            result = await self._make_request("elink.fcgi", params)
            return result
        except Exception as e:
            logger.error(f"Error getting links: {str(e)}")
            raise NCBIClientError(f"Failed to get links: {str(e)}")

    @cached(prefix="ncbi_elink_history", data_type="search")
    async def get_links_and_post_to_history(self,
                                           ids: Union[str, List[str]],
                                           dbfrom: str,
                                           db: str,
                                           linkname: Optional[str] = None) -> Dict[str, Any]:
        """
        Get links between database records and post results to History server.

        This is a convenience method that uses ELink with cmd=neighbor_history
        to retrieve links and post the results to the History server.

        Args:
            ids: UID or list of UIDs to find links for
            dbfrom: Source database containing the input UIDs
            db: Target database to find links in
            linkname: Name of the link to retrieve (optional)

        Returns:
            Dictionary with WebEnv and QueryKey for the posted results

        Raises:
            ValidationError: If inputs are invalid
            NCBIClientError: If the request fails
        """
        return await self.get_links(ids, dbfrom, db, linkname, cmd="neighbor_history")

    @cached(prefix="ncbi_elink_scores", data_type="search")
    async def get_links_with_scores(self,
                                   ids: Union[str, List[str]],
                                   dbfrom: str,
                                   db: str,
                                   linkname: Optional[str] = None) -> Dict[str, Any]:
        """
        Get links between database records with similarity scores.

        This is a convenience method that uses ELink with cmd=neighbor_score
        to retrieve links with similarity scores.

        Args:
            ids: UID or list of UIDs to find links for
            dbfrom: Source database containing the input UIDs
            db: Target database to find links in
            linkname: Name of the link to retrieve (optional)

        Returns:
            Dictionary with links and scores

        Raises:
            ValidationError: If inputs are invalid
            NCBIClientError: If the request fails
        """
        return await self.get_links(ids, dbfrom, db, linkname, cmd="neighbor_score")

    @cached(prefix="ncbi_elink_check", data_type="search")
    async def check_links(self,
                         ids: Union[str, List[str]],
                         dbfrom: str,
                         db: Optional[str] = None) -> Dict[str, Any]:
        """
        Check for the existence of links for a set of UIDs.

        This method uses the ELink E-utility with cmd=acheck to check for
        the existence of links for a set of UIDs.

        Args:
            ids: UID or list of UIDs to check links for
            dbfrom: Source database containing the input UIDs
            db: Target database to check links in (optional)

        Returns:
            Dictionary with link information

        Raises:
            ValidationError: If inputs are invalid
            NCBIClientError: If the request fails
        """
        # Validate inputs
        if not ids:
            raise ValidationError("IDs list cannot be empty")

        if not dbfrom:
            raise ValidationError("Source database must be specified")

        # Convert single ID to list
        if isinstance(ids, str):
            id_list = [ids]
        else:
            id_list = ids

        # Prepare parameters
        params = {
            "dbfrom": dbfrom,
            "id": ",".join(id_list),
            "cmd": "acheck"
        }

        # Add optional target database if provided
        if db:
            params["db"] = db

        try:
            logger.info(f"Checking links from {dbfrom} for {len(id_list)} IDs")
            result = await self._make_request("elink.fcgi", params)
            return result
        except Exception as e:
            logger.error(f"Error checking links: {str(e)}")
            raise NCBIClientError(f"Failed to check links: {str(e)}")

    @cached(prefix="ncbi_linkout", data_type="search")
    async def get_linkout_urls(self,
                              ids: Union[str, List[str]],
                              dbfrom: str,
                              include_libraries: bool = False) -> Dict[str, Any]:
        """
        Get LinkOut URLs for a set of UIDs.

        This method uses the ELink E-utility with cmd=llinks or cmd=llinkslib
        to retrieve LinkOut URLs for a set of UIDs.

        Args:
            ids: UID or list of UIDs to get LinkOut URLs for
            dbfrom: Source database containing the input UIDs
            include_libraries: Whether to include library LinkOut providers (default: False)

        Returns:
            Dictionary with LinkOut URLs

        Raises:
            ValidationError: If inputs are invalid
            NCBIClientError: If the request fails
        """
        # Validate inputs
        if not ids:
            raise ValidationError("IDs list cannot be empty")

        if not dbfrom:
            raise ValidationError("Source database must be specified")

        # Convert single ID to list
        if isinstance(ids, str):
            id_list = [ids]
        else:
            id_list = ids

        # Prepare parameters
        params = {
            "dbfrom": dbfrom,
            "id": ",".join(id_list),
            "cmd": "llinkslib" if include_libraries else "llinks"
        }

        try:
            logger.info(f"Getting LinkOut URLs for {len(id_list)} IDs from {dbfrom}")
            result = await self._make_request("elink.fcgi", params)
            return result
        except Exception as e:
            logger.error(f"Error getting LinkOut URLs: {str(e)}")
            raise NCBIClientError(f"Failed to get LinkOut URLs: {str(e)}")

    async def get_fulltext_url(self, pmid: str) -> str:
        """
        Get the full-text URL for a PubMed article.

        This method uses the ELink E-utility with cmd=prlinks and retmode=ref
        to retrieve the full-text URL for a PubMed article and redirects to it.

        Args:
            pmid: PubMed ID to get full-text URL for

        Returns:
            Full-text URL

        Raises:
            ValidationError: If PMID is invalid
            NCBIClientError: If the request fails
        """
        # Validate inputs
        if not pmid or not pmid.strip():
            raise ValidationError("PMID cannot be empty")

        # Prepare parameters
        params = {
            "dbfrom": "pubmed",
            "id": pmid.strip(),
            "cmd": "prlinks",
            "retmode": "ref"
        }

        try:
            # This is a special case where we want to get the redirect URL
            # rather than following it, so we need to use a different approach
            url = f"{self.base_url}elink.fcgi"

            # Add common parameters
            params.update({
                "tool": self.tool,
                "email": self.email,
            })

            # Add API key if available
            if self.api_key:
                params["api_key"] = self.api_key

            # Apply rate limiting
            await self.rate_limiter.acquire()

            # Make request with redirect=False to get the redirect URL
            async with httpx.AsyncClient(follow_redirects=False) as client:
                response = await client.get(url, params=params)

                # Check for redirect
                if response.status_code in (301, 302, 303, 307, 308):
                    redirect_url = response.headers.get("location")
                    if redirect_url:
                        logger.info(f"Got full-text URL for PMID {pmid}: {redirect_url}")
                        return redirect_url

                # If no redirect, check for error
                if response.status_code >= 400:
                    error_msg = f"NCBI API error: {response.status_code} {response.reason_phrase}"
                    logger.warning(f"{error_msg} for {url}")
                    raise NCBIClientError(error_msg, response.status_code)

                # If no redirect and no error, return the response URL
                logger.warning(f"No redirect found for PMID {pmid}, returning original URL")
                return str(response.url)
        except httpx.RequestError as e:
            error_msg = f"Network error: {str(e)}"
            logger.error(f"{error_msg} for PMID {pmid}")
            raise NCBIClientError(f"Failed to get full-text URL: {str(e)}")
        except Exception as e:
            logger.error(f"Error getting full-text URL for PMID {pmid}: {str(e)}")
            raise NCBIClientError(f"Failed to get full-text URL: {str(e)}")

    @cached(prefix="pubmed_related", data_type="search")
    async def get_related_articles(self, pmid: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """
        Get related articles for a PubMed article.

        This method combines ELink and EFetch to retrieve related articles
        for a PubMed article with their details.

        Args:
            pmid: PubMed ID to find related articles for
            max_results: Maximum number of related articles to return (default: 20)

        Returns:
            List of related articles with details

        Raises:
            ValidationError: If PMID is invalid
            NCBIClientError: If the request fails
        """
        # Validate inputs
        if not pmid or not pmid.strip():
            raise ValidationError("PMID cannot be empty")

        if max_results < 1:
            raise ValidationError("max_results must be at least 1")

        try:
            # Get related articles with scores
            logger.info(f"Getting related articles for PMID {pmid}")
            links_result = await self.get_links_with_scores(pmid, "pubmed", "pubmed")

            # Extract PMIDs of related articles
            related_pmids = []
            try:
                linksets = links_result.get("linksets", [])
                if linksets and "linksetdbs" in linksets[0]:
                    for linksetdb in linksets[0]["linksetdbs"]:
                        if linksetdb.get("linkname") == "pubmed_pubmed":
                            links = linksetdb.get("links", [])
                            # Sort by score if available
                            if links and "score" in links[0]:
                                links.sort(key=lambda x: int(x.get("score", 0)), reverse=True)
                            # Extract PMIDs
                            related_pmids = [link.get("id") for link in links[:max_results] if "id" in link]
            except Exception as e:
                logger.error(f"Error parsing related articles: {str(e)}")

            if not related_pmids:
                logger.warning(f"No related articles found for PMID {pmid}")
                return []

            # Fetch details for related articles
            logger.info(f"Fetching details for {len(related_pmids)} related articles")
            articles = await self.fetch_pubmed_abstracts(id_list=related_pmids)

            return articles
        except NCBIClientError as e:
            # Re-raise with more context
            logger.error(f"Error getting related articles for PMID {pmid}: {str(e)}")
            raise NCBIClientError(f"Failed to get related articles: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error getting related articles for PMID {pmid}: {str(e)}")
            raise NCBIClientError(f"Unexpected error getting related articles: {str(e)}")

    @cached(prefix="pubmed_cited_by", data_type="search")
    async def get_citing_articles(self, pmid: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """
        Get articles that cite a PubMed article.

        This method combines ELink and EFetch to retrieve articles that cite
        a PubMed article with their details.

        Args:
            pmid: PubMed ID to find citing articles for
            max_results: Maximum number of citing articles to return (default: 20)

        Returns:
            List of citing articles with details

        Raises:
            ValidationError: If PMID is invalid
            NCBIClientError: If the request fails
        """
        # Validate inputs
        if not pmid or not pmid.strip():
            raise ValidationError("PMID cannot be empty")

        if max_results < 1:
            raise ValidationError("max_results must be at least 1")

        try:
            # Get citing articles
            logger.info(f"Getting citing articles for PMID {pmid}")
            links_result = await self.get_links(pmid, "pubmed", "pubmed", linkname="pubmed_pubmed_citedin")

            # Extract PMIDs of citing articles
            citing_pmids = []
            try:
                linksets = links_result.get("linksets", [])
                if linksets and "linksetdbs" in linksets[0]:
                    for linksetdb in linksets[0]["linksetdbs"]:
                        if linksetdb.get("linkname") == "pubmed_pubmed_citedin":
                            links = linksetdb.get("links", [])
                            # Extract PMIDs
                            citing_pmids = [link.get("id") for link in links[:max_results] if "id" in link]
            except Exception as e:
                logger.error(f"Error parsing citing articles: {str(e)}")

            if not citing_pmids:
                logger.warning(f"No citing articles found for PMID {pmid}")
                return []

            # Fetch details for citing articles
            logger.info(f"Fetching details for {len(citing_pmids)} citing articles")
            articles = await self.fetch_pubmed_abstracts(id_list=citing_pmids)

            return articles
        except NCBIClientError as e:
            # Re-raise with more context
            logger.error(f"Error getting citing articles for PMID {pmid}: {str(e)}")
            raise NCBIClientError(f"Failed to get citing articles: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error getting citing articles for PMID {pmid}: {str(e)}")
            raise NCBIClientError(f"Unexpected error getting citing articles: {str(e)}")

    @cached(prefix="pubmed_mesh", data_type="search")
    async def get_mesh_terms(self, pmid: str) -> List[Dict[str, str]]:
        """
        Get MeSH terms for a PubMed article.

        This method uses EFetch to retrieve MeSH terms for a PubMed article.

        Args:
            pmid: PubMed ID to get MeSH terms for

        Returns:
            List of MeSH terms with their qualifiers

        Raises:
            ValidationError: If PMID is invalid
            NCBIClientError: If the request fails
        """
        # Validate inputs
        if not pmid or not pmid.strip():
            raise ValidationError("PMID cannot be empty")

        try:
            # Fetch article in XML format
            logger.info(f"Getting MeSH terms for PMID {pmid}")
            params = {
                "db": "pubmed",
                "id": pmid.strip(),
                "retmode": "xml"
            }

            xml_text = await self._make_request("efetch.fcgi", params, return_json=False)

            # Parse XML to extract MeSH terms
            mesh_terms = []
            try:
                root = ET.fromstring(xml_text)

                # Find all MeshHeading elements
                for mesh_heading in root.findall(".//MeshHeading"):
                    # Get descriptor name
                    descriptor = mesh_heading.find("DescriptorName")
                    if descriptor is None:
                        continue

                    descriptor_name = descriptor.text
                    descriptor_ui = descriptor.get("UI", "")
                    is_major = descriptor.get("MajorTopicYN", "N") == "Y"

                    # Get qualifiers
                    qualifiers = []
                    for qualifier in mesh_heading.findall("QualifierName"):
                        qualifier_name = qualifier.text
                        qualifier_ui = qualifier.get("UI", "")
                        qualifier_major = qualifier.get("MajorTopicYN", "N") == "Y"

                        qualifiers.append({
                            "name": qualifier_name,
                            "ui": qualifier_ui,
                            "major": qualifier_major
                        })

                    # Add MeSH term to list
                    mesh_terms.append({
                        "descriptor": descriptor_name,
                        "descriptor_ui": descriptor_ui,
                        "is_major": is_major,
                        "qualifiers": qualifiers
                    })
            except ET.ParseError as e:
                logger.error(f"Error parsing XML for PMID {pmid}: {str(e)}")
                raise NCBIClientError(f"Failed to parse XML: {str(e)}")

            return mesh_terms
        except NCBIClientError as e:
            # Re-raise with more context
            logger.error(f"Error getting MeSH terms for PMID {pmid}: {str(e)}")
            raise NCBIClientError(f"Failed to get MeSH terms: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error getting MeSH terms for PMID {pmid}: {str(e)}")
            raise NCBIClientError(f"Unexpected error getting MeSH terms: {str(e)}")

    @cached(prefix="pubmed_journal_info", data_type="search")
    async def get_journal_info(self, journal: str) -> Dict[str, Any]:
        """
        Get information about a journal from the NLM Catalog.

        This method uses ESearch and ESummary to retrieve information about a journal.

        Args:
            journal: Journal title or abbreviation

        Returns:
            Dictionary with journal information

        Raises:
            ValidationError: If journal is invalid
            NCBIClientError: If the request fails
        """
        # Validate inputs
        if not journal or not journal.strip():
            raise ValidationError("Journal name cannot be empty")

        try:
            # Search for the journal in the NLM Catalog
            logger.info(f"Searching for journal '{journal}' in NLM Catalog")
            search_params = {
                "db": "nlmcatalog",
                "term": f"{journal.strip()}[Title] AND ncbijournals[Filter]",
                "retmax": 1
            }

            search_result = await self._make_request("esearch.fcgi", search_params)

            # Extract journal ID
            journal_ids = search_result.get("esearchresult", {}).get("idlist", [])
            if not journal_ids:
                logger.warning(f"No journal found for '{journal}'")
                return {}

            # Get journal details
            logger.info(f"Getting details for journal ID {journal_ids[0]}")
            summary_params = {
                "db": "nlmcatalog",
                "id": journal_ids[0],
                "version": "2.0"
            }

            journal_info = await self._make_request("esummary.fcgi", summary_params)

            return journal_info
        except NCBIClientError as e:
            # Re-raise with more context
            logger.error(f"Error getting journal info for '{journal}': {str(e)}")
            raise NCBIClientError(f"Failed to get journal info: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error getting journal info for '{journal}': {str(e)}")
            raise NCBIClientError(f"Unexpected error getting journal info: {str(e)}")

    @cached(prefix="sequence_search", data_type="search")
    async def search_sequence_database(self,
                                     query: str,
                                     db: str = "nucleotide",
                                     max_results: int = 20,
                                     return_type: str = "gb",
                                     return_mode: str = "text") -> Dict[str, Any]:
        """
        Search a sequence database and retrieve sequences.

        This method combines ESearch and EFetch to search a sequence database
        and retrieve the sequences in the specified format.

        Args:
            query: Search query
            db: Database to search (default: "nucleotide")
                Options include: "nucleotide", "protein", "genome", "gene"
            max_results: Maximum number of results to return (default: 20)
            return_type: Return type (default: "gb")
                Options for nucleotide: "gb" (GenBank), "fasta", "gbwithparts", "gbc"
                Options for protein: "gp" (GenPept), "fasta", "gpc"
            return_mode: Return mode (default: "text")
                Options: "text", "xml", "asn.1"

        Returns:
            Dictionary with search results and sequences

        Raises:
            ValidationError: If inputs are invalid
            NCBIClientError: If the request fails
        """
        # Validate inputs
        if not query or not query.strip():
            raise ValidationError("Search query cannot be empty")

        if max_results < 1:
            raise ValidationError("max_results must be at least 1")

        # Validate database
        valid_dbs = ["nucleotide", "protein", "genome", "gene"]
        if db not in valid_dbs:
            raise ValidationError(f"Invalid database: {db}. Must be one of {valid_dbs}")

        try:
            # Search the database
            logger.info(f"Searching {db} database for '{query}'")
            search_params = {
                "db": db,
                "term": query.strip(),
                "retmax": max_results,
                "usehistory": "y"
            }

            search_result = await self._make_request("esearch.fcgi", search_params)

            # Extract IDs and WebEnv/QueryKey
            id_list = search_result.get("esearchresult", {}).get("idlist", [])
            web_env = search_result.get("esearchresult", {}).get("webenv")
            query_key = search_result.get("esearchresult", {}).get("querykey")
            count = int(search_result.get("esearchresult", {}).get("count", "0"))

            if not id_list:
                logger.warning(f"No results found for '{query}' in {db} database")
                return {"count": 0, "ids": [], "sequences": ""}

            # Fetch sequences
            logger.info(f"Fetching {len(id_list)} sequences from {db} database")
            fetch_params = {
                "db": db,
                "WebEnv": web_env,
                "query_key": query_key,
                "rettype": return_type,
                "retmode": return_mode
            }

            sequences = await self._make_request("efetch.fcgi", fetch_params, return_json=False)

            return {
                "count": count,
                "ids": id_list,
                "sequences": sequences
            }
        except NCBIClientError as e:
            # Re-raise with more context
            logger.error(f"Error searching {db} database for '{query}': {str(e)}")
            raise NCBIClientError(f"Failed to search {db} database: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error searching {db} database for '{query}': {str(e)}")
            raise NCBIClientError(f"Unexpected error searching {db} database: {str(e)}")

    @cached(prefix="sequence_fetch", data_type="search")
    async def fetch_sequence(self,
                           id: str,
                           db: str = "nucleotide",
                           return_type: str = "gb",
                           return_mode: str = "text",
                           strand: Optional[int] = None,
                           seq_start: Optional[int] = None,
                           seq_stop: Optional[int] = None) -> str:
        """
        Fetch a sequence from a sequence database.

        This method uses EFetch to retrieve a sequence in the specified format.

        Args:
            id: Sequence ID (GI number or accession)
            db: Database to fetch from (default: "nucleotide")
                Options include: "nucleotide", "protein", "genome", "gene"
            return_type: Return type (default: "gb")
                Options for nucleotide: "gb" (GenBank), "fasta", "gbwithparts", "gbc"
                Options for protein: "gp" (GenPept), "fasta", "gpc"
            return_mode: Return mode (default: "text")
                Options: "text", "xml", "asn.1"
            strand: Strand of DNA to retrieve (1 for plus, 2 for minus)
            seq_start: First sequence base to retrieve (1-based)
            seq_stop: Last sequence base to retrieve (1-based)

        Returns:
            Sequence in the specified format

        Raises:
            ValidationError: If inputs are invalid
            NCBIClientError: If the request fails
        """
        # Validate inputs
        if not id or not id.strip():
            raise ValidationError("Sequence ID cannot be empty")

        # Validate database
        valid_dbs = ["nucleotide", "protein", "genome", "gene"]
        if db not in valid_dbs:
            raise ValidationError(f"Invalid database: {db}. Must be one of {valid_dbs}")

        # Validate strand
        if strand is not None and strand not in (1, 2):
            raise ValidationError("Strand must be 1 (plus) or 2 (minus)")

        # Validate sequence range
        if (seq_start is not None and seq_stop is None) or (seq_start is None and seq_stop is not None):
            raise ValidationError("Both seq_start and seq_stop must be provided together")

        if seq_start is not None and seq_stop is not None:
            if seq_start < 1 or seq_stop < seq_start:
                raise ValidationError("Invalid sequence range: seq_start must be >= 1 and seq_stop must be >= seq_start")

        try:
            # Prepare parameters
            params = {
                "db": db,
                "id": id.strip(),
                "rettype": return_type,
                "retmode": return_mode
            }

            # Add optional parameters
            if strand is not None:
                params["strand"] = strand

            if seq_start is not None and seq_stop is not None:
                params["seq_start"] = seq_start
                params["seq_stop"] = seq_stop

            # Fetch sequence
            logger.info(f"Fetching sequence {id} from {db} database")
            sequence = await self._make_request("efetch.fcgi", params, return_json=False)

            return sequence
        except NCBIClientError as e:
            # Re-raise with more context
            logger.error(f"Error fetching sequence {id} from {db} database: {str(e)}")
            raise NCBIClientError(f"Failed to fetch sequence: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error fetching sequence {id} from {db} database: {str(e)}")
            raise NCBIClientError(f"Unexpected error fetching sequence: {str(e)}")

    @cached(prefix="taxonomy_fetch", data_type="search")
    async def get_taxonomy(self, id: Union[str, int]) -> Dict[str, Any]:
        """
        Get taxonomy information for a taxon ID or organism name.

        This method uses ESearch and ESummary to retrieve taxonomy information.

        Args:
            id: Taxonomy ID or organism name

        Returns:
            Dictionary with taxonomy information

        Raises:
            ValidationError: If ID is invalid
            NCBIClientError: If the request fails
        """
        # Validate inputs
        if not id:
            raise ValidationError("Taxonomy ID or name cannot be empty")

        try:
            # Check if ID is a number (taxon ID) or string (organism name)
            if isinstance(id, int) or (isinstance(id, str) and id.isdigit()):
                # Fetch taxonomy directly by ID
                logger.info(f"Getting taxonomy information for taxon ID {id}")
                params = {
                    "db": "taxonomy",
                    "id": str(id),
                    "version": "2.0"
                }

                taxonomy = await self._make_request("esummary.fcgi", params)
                return taxonomy
            else:
                # Search for organism name
                logger.info(f"Searching taxonomy database for '{id}'")
                search_params = {
                    "db": "taxonomy",
                    "term": str(id).strip(),
                    "retmax": 1
                }

                search_result = await self._make_request("esearch.fcgi", search_params)

                # Extract taxon ID
                taxon_ids = search_result.get("esearchresult", {}).get("idlist", [])
                if not taxon_ids:
                    logger.warning(f"No taxonomy found for '{id}'")
                    return {}

                # Get taxonomy details
                logger.info(f"Getting taxonomy information for taxon ID {taxon_ids[0]}")
                summary_params = {
                    "db": "taxonomy",
                    "id": taxon_ids[0],
                    "version": "2.0"
                }

                taxonomy = await self._make_request("esummary.fcgi", summary_params)
                return taxonomy
        except NCBIClientError as e:
            # Re-raise with more context
            logger.error(f"Error getting taxonomy for '{id}': {str(e)}")
            raise NCBIClientError(f"Failed to get taxonomy: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error getting taxonomy for '{id}': {str(e)}")
            raise NCBIClientError(f"Unexpected error getting taxonomy: {str(e)}")

    # History Server Methods

    async def create_history_session(self) -> Dict[str, str]:
        """
        Create a new History server session.

        This method creates a new WebEnv on the NCBI History server by performing
        a simple search and returning the WebEnv and query_key.

        Returns:
            Dictionary with WebEnv and query_key

        Raises:
            NCBIClientError: If the request fails
        """
        try:
            # Perform a simple search to create a new WebEnv
            logger.info("Creating new History server session")
            params = {
                "db": "pubmed",
                "term": "1:1[uid]",  # Search for PMID 1, which is guaranteed to exist
                "usehistory": "y"
            }

            result = await self._make_request("esearch.fcgi", params)

            # Extract WebEnv and query_key
            web_env = result.get("esearchresult", {}).get("webenv")
            query_key = result.get("esearchresult", {}).get("querykey")

            if not web_env:
                logger.error("Failed to create History server session: No WebEnv returned")
                raise NCBIClientError("Failed to create History server session: No WebEnv returned")

            logger.info(f"Created new History server session with WebEnv={web_env[:10]}... and query_key={query_key}")
            return {
                "web_env": web_env,
                "query_key": query_key
            }
        except NCBIClientError as e:
            # Re-raise with more context
            logger.error(f"Error creating History server session: {str(e)}")
            raise NCBIClientError(f"Failed to create History server session: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error creating History server session: {str(e)}")
            raise NCBIClientError(f"Unexpected error creating History server session: {str(e)}")

    async def post_ids_to_history(self, ids: Union[str, List[str]], db: str, web_env: Optional[str] = None) -> Dict[str, str]:
        """
        Post a list of IDs to the History server.

        This method uses EPost to upload a list of UIDs to the Entrez History server.
        If a WebEnv is provided, the IDs will be appended to the existing session.
        Otherwise, a new session will be created.

        Args:
            ids: UID or list of UIDs to post
            db: Database containing the UIDs
            web_env: Existing WebEnv to append to (optional)

        Returns:
            Dictionary with WebEnv and query_key

        Raises:
            ValidationError: If inputs are invalid
            NCBIClientError: If the request fails
        """
        # Validate inputs
        if not ids:
            raise ValidationError("IDs list cannot be empty")

        if not db:
            raise ValidationError("Database must be specified")

        # Convert single ID to list
        if isinstance(ids, str):
            id_list = [ids]
        else:
            id_list = ids

        try:
            # Prepare parameters
            params = {
                "db": db,
                "id": ",".join(id_list)
            }

            # Add WebEnv if provided
            if web_env:
                params["WebEnv"] = web_env

            # Post IDs to History server
            logger.info(f"Posting {len(id_list)} IDs to History server")
            result = await self._make_request("epost.fcgi", params)

            # Extract WebEnv and query_key
            new_web_env = result.get("webenv")
            new_query_key = result.get("querykey")

            if not new_web_env or not new_query_key:
                logger.error("Failed to post IDs to History server: No WebEnv or query_key returned")
                raise NCBIClientError("Failed to post IDs to History server: No WebEnv or query_key returned")

            logger.info(f"Posted {len(id_list)} IDs to History server with WebEnv={new_web_env[:10]}... and query_key={new_query_key}")
            return {
                "web_env": new_web_env,
                "query_key": new_query_key
            }
        except NCBIClientError as e:
            # Re-raise with more context
            logger.error(f"Error posting IDs to History server: {str(e)}")
            raise NCBIClientError(f"Failed to post IDs to History server: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error posting IDs to History server: {str(e)}")
            raise NCBIClientError(f"Unexpected error posting IDs to History server: {str(e)}")

    async def search_and_post_to_history(self,
                                        query: str,
                                        db: str = "pubmed",
                                        web_env: Optional[str] = None,
                                        query_key: Optional[str] = None,
                                        **search_params) -> Dict[str, Any]:
        """
        Search a database and post results to the History server.

        This method uses ESearch with usehistory=y to search a database and post
        the results to the History server. If a WebEnv is provided, the results
        will be appended to the existing session.

        Args:
            query: Search query
            db: Database to search (default: "pubmed")
            web_env: Existing WebEnv to append to (optional)
            query_key: Existing query_key to use (optional)
            **search_params: Additional search parameters
                - sort: Sort order (e.g., "relevance", "pub_date")
                - field: Field to search in (e.g., "title", "author")
                - datetype: Type of date (e.g., "pdat", "edat")
                - reldate: Relative date (e.g., 30 for last 30 days)
                - mindate/maxdate: Date range (format: YYYY/MM/DD)
                - retmax: Maximum number of results to return

        Returns:
            Dictionary with search results, WebEnv, and query_key

        Raises:
            ValidationError: If inputs are invalid
            NCBIClientError: If the request fails
        """
        # Validate inputs
        if not query or not query.strip():
            raise ValidationError("Search query cannot be empty")

        if not db:
            raise ValidationError("Database must be specified")

        try:
            # Prepare parameters
            params = {
                "db": db,
                "term": query.strip(),
                "usehistory": "y"
            }

            # Add WebEnv and query_key if provided
            if web_env:
                params["WebEnv"] = web_env

            if query_key:
                params["query_key"] = query_key

            # Add additional search parameters
            params.update(search_params)

            # Search and post to History server
            logger.info(f"Searching {db} for '{query}' and posting to History server")
            result = await self._make_request("esearch.fcgi", params)

            # Extract WebEnv and query_key
            new_web_env = result.get("esearchresult", {}).get("webenv")
            new_query_key = result.get("esearchresult", {}).get("querykey")
            count = int(result.get("esearchresult", {}).get("count", "0"))

            if not new_web_env or not new_query_key:
                logger.error("Failed to search and post to History server: No WebEnv or query_key returned")
                raise NCBIClientError("Failed to search and post to History server: No WebEnv or query_key returned")

            logger.info(f"Found {count} results for '{query}' in {db} and posted to History server with WebEnv={new_web_env[:10]}... and query_key={new_query_key}")
            return {
                "count": count,
                "web_env": new_web_env,
                "query_key": new_query_key,
                "result": result
            }
        except NCBIClientError as e:
            # Re-raise with more context
            logger.error(f"Error searching and posting to History server: {str(e)}")
            raise NCBIClientError(f"Failed to search and post to History server: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error searching and posting to History server: {str(e)}")
            raise NCBIClientError(f"Unexpected error searching and posting to History server: {str(e)}")

    async def fetch_from_history(self,
                               web_env: str,
                               query_key: str,
                               db: str,
                               retstart: int = 0,
                               retmax: int = 20,
                               rettype: Optional[str] = None,
                               retmode: Optional[str] = None) -> Any:
        """
        Fetch records from the History server.

        This method uses EFetch to retrieve records from a set of UIDs stored
        on the History server.

        Args:
            web_env: WebEnv string
            query_key: Query key
            db: Database to fetch from
            retstart: First record to retrieve (default: 0)
            retmax: Maximum number of records to retrieve (default: 20)
            rettype: Retrieval type (e.g., "abstract", "medline", "gb")
            retmode: Retrieval mode (e.g., "text", "xml", "json")

        Returns:
            Fetched records in the specified format

        Raises:
            ValidationError: If inputs are invalid
            NCBIClientError: If the request fails
        """
        # Validate inputs
        if not web_env or not query_key:
            raise ValidationError("WebEnv and query_key must be provided")

        if not db:
            raise ValidationError("Database must be specified")

        if retstart < 0:
            raise ValidationError("retstart must be non-negative")

        if retmax < 1:
            raise ValidationError("retmax must be at least 1")

        try:
            # Prepare parameters
            params = {
                "db": db,
                "WebEnv": web_env,
                "query_key": query_key,
                "retstart": retstart,
                "retmax": retmax
            }

            # Add optional parameters
            if rettype:
                params["rettype"] = rettype

            if retmode:
                params["retmode"] = retmode

            # Fetch from History server
            logger.info(f"Fetching {retmax} records from {db} (starting at {retstart}) using History server")
            result = await self._make_request("efetch.fcgi", params, return_json=(retmode != "xml" and retmode != "text"))

            logger.info(f"Successfully fetched records from History server")
            return result
        except NCBIClientError as e:
            # Re-raise with more context
            logger.error(f"Error fetching from History server: {str(e)}")
            raise NCBIClientError(f"Failed to fetch from History server: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error fetching from History server: {str(e)}")
            raise NCBIClientError(f"Unexpected error fetching from History server: {str(e)}")

    async def get_summary_from_history(self,
                                     web_env: str,
                                     query_key: str,
                                     db: str,
                                     retstart: int = 0,
                                     retmax: int = 20,
                                     version: str = "2.0") -> Dict[str, Any]:
        """
        Get document summaries from the History server.

        This method uses ESummary to retrieve document summaries from a set of UIDs
        stored on the History server.

        Args:
            web_env: WebEnv string
            query_key: Query key
            db: Database to fetch from
            retstart: First record to retrieve (default: 0)
            retmax: Maximum number of records to retrieve (default: 20)
            version: ESummary version (default: "2.0")

        Returns:
            Document summaries

        Raises:
            ValidationError: If inputs are invalid
            NCBIClientError: If the request fails
        """
        # Validate inputs
        if not web_env or not query_key:
            raise ValidationError("WebEnv and query_key must be provided")

        if not db:
            raise ValidationError("Database must be specified")

        if retstart < 0:
            raise ValidationError("retstart must be non-negative")

        if retmax < 1:
            raise ValidationError("retmax must be at least 1")

        try:
            # Prepare parameters
            params = {
                "db": db,
                "WebEnv": web_env,
                "query_key": query_key,
                "retstart": retstart,
                "retmax": retmax,
                "version": version
            }

            # Get summaries from History server
            logger.info(f"Getting summaries for {retmax} records from {db} (starting at {retstart}) using History server")
            result = await self._make_request("esummary.fcgi", params)

            logger.info(f"Successfully got summaries from History server")
            return result
        except NCBIClientError as e:
            # Re-raise with more context
            logger.error(f"Error getting summaries from History server: {str(e)}")
            raise NCBIClientError(f"Failed to get summaries from History server: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error getting summaries from History server: {str(e)}")
            raise NCBIClientError(f"Unexpected error getting summaries from History server: {str(e)}")

    # Batch Operations Methods

    async def batch_fetch_articles(self,
                                 pmids: List[str],
                                 batch_size: int = 200,
                                 include_abstracts: bool = True,
                                 max_workers: int = 5) -> List[Dict[str, Any]]:
        """
        Fetch article details for a large list of PMIDs in batches.

        This method efficiently fetches article details for a large list of PMIDs
        by splitting them into batches and processing them concurrently.

        Args:
            pmids: List of PMIDs to fetch
            batch_size: Number of PMIDs per batch (default: 200)
            include_abstracts: Whether to include abstracts (default: True)
            max_workers: Maximum number of concurrent workers (default: 5)

        Returns:
            List of article details

        Raises:
            ValidationError: If inputs are invalid
            NCBIClientError: If the fetch fails
        """
        # Validate inputs
        if not pmids:
            raise ValidationError("PMIDs list cannot be empty")

        if batch_size < 1:
            raise ValidationError("batch_size must be at least 1")

        if max_workers < 1:
            raise ValidationError("max_workers must be at least 1")

        # Split PMIDs into batches
        batches = await self._batch_pmids(pmids, batch_size)
        total_batches = len(batches)

        logger.info(f"Fetching {len(pmids)} articles in {total_batches} batches of size {batch_size}")

        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_workers)

        # Define a worker function to process each batch
        async def process_batch(batch_index: int, batch_pmids: List[str]) -> List[Dict[str, Any]]:
            async with semaphore:
                try:
                    logger.info(f"Processing batch {batch_index+1}/{total_batches} with {len(batch_pmids)} PMIDs")

                    # Fetch article details
                    articles = await self.fetch_article_details(batch_pmids)

                    # Fetch abstracts if requested
                    if include_abstracts and articles:
                        abstracts = await self.fetch_article_abstracts(batch_pmids)

                        # Merge abstracts into articles
                        for article in articles:
                            pmid = article["pmid"]
                            if pmid in abstracts:
                                article["abstract"] = abstracts[pmid]

                    logger.info(f"Completed batch {batch_index+1}/{total_batches} with {len(articles)} articles")
                    return articles
                except Exception as e:
                    logger.error(f"Error processing batch {batch_index+1}/{total_batches}: {str(e)}")
                    # Return empty list for this batch to continue with other batches
                    return []

        # Process all batches concurrently
        tasks = [process_batch(i, batch) for i, batch in enumerate(batches)]
        batch_results = await asyncio.gather(*tasks)

        # Combine results from all batches
        all_articles = []
        for batch_result in batch_results:
            all_articles.extend(batch_result)

        logger.info(f"Fetched {len(all_articles)}/{len(pmids)} articles in total")
        return all_articles

    async def batch_search_and_fetch(self,
                                    query: str,
                                    db: str = "pubmed",
                                    max_results: int = 1000,
                                    batch_size: int = 200,
                                    max_workers: int = 5,
                                    **search_params) -> Dict[str, Any]:
        """
        Search a database and fetch results in batches.

        This method efficiently searches a database and fetches results in batches
        by using the History server and processing batches concurrently.

        Args:
            query: Search query
            db: Database to search (default: "pubmed")
            max_results: Maximum number of results to fetch (default: 1000)
            batch_size: Number of records per batch (default: 200)
            max_workers: Maximum number of concurrent workers (default: 5)
            **search_params: Additional search parameters
                - sort: Sort order (e.g., "relevance", "pub_date")
                - field: Field to search in (e.g., "title", "author")
                - datetype: Type of date (e.g., "pdat", "edat")
                - reldate: Relative date (e.g., 30 for last 30 days)
                - mindate/maxdate: Date range (format: YYYY/MM/DD)

        Returns:
            Dictionary with search results and fetched records

        Raises:
            ValidationError: If inputs are invalid
            NCBIClientError: If the search or fetch fails
        """
        # Validate inputs
        if not query or not query.strip():
            raise ValidationError("Search query cannot be empty")

        if not db:
            raise ValidationError("Database must be specified")

        if max_results < 1:
            raise ValidationError("max_results must be at least 1")

        if batch_size < 1:
            raise ValidationError("batch_size must be at least 1")

        if max_workers < 1:
            raise ValidationError("max_workers must be at least 1")

        try:
            # Search and post to History server
            logger.info(f"Searching {db} for '{query}' with max_results={max_results}")
            search_result = await self.search_and_post_to_history(
                query=query,
                db=db,
                retmax=min(max_results, 100000),  # NCBI limits to 100,000 results
                **search_params
            )

            count = search_result["count"]
            web_env = search_result["web_env"]
            query_key = search_result["query_key"]

            logger.info(f"Found {count} results for '{query}' in {db}")

            # Limit max_results to the actual count
            max_results = min(max_results, count)

            # Calculate number of batches
            num_batches = (max_results + batch_size - 1) // batch_size

            # Create a semaphore to limit concurrency
            semaphore = asyncio.Semaphore(max_workers)

            # Define a worker function to process each batch
            async def process_batch(batch_index: int) -> List[Dict[str, Any]]:
                async with semaphore:
                    try:
                        retstart = batch_index * batch_size
                        current_batch_size = min(batch_size, max_results - retstart)

                        if current_batch_size <= 0:
                            return []

                        logger.info(f"Fetching batch {batch_index+1}/{num_batches} (records {retstart+1}-{retstart+current_batch_size})")

                        # Determine appropriate rettype and retmode based on database
                        rettype, retmode = None, None
                        if db == "pubmed":
                            rettype, retmode = "medline", "text"
                        elif db in ("nucleotide", "protein"):
                            rettype, retmode = "gb", "text"

                        # Fetch batch from History server
                        batch_result = await self.fetch_from_history(
                            web_env=web_env,
                            query_key=query_key,
                            db=db,
                            retstart=retstart,
                            retmax=current_batch_size,
                            rettype=rettype,
                            retmode=retmode
                        )

                        # For PubMed, parse the Medline format
                        if db == "pubmed" and rettype == "medline" and retmode == "text":
                            # Simple parsing of Medline format
                            records = []
                            current_record = {}
                            lines = batch_result.strip().split("\n")

                            for line in lines:
                                if line.startswith("PMID- "):
                                    if current_record:
                                        records.append(current_record)
                                    current_record = {"pmid": line[6:].strip()}
                                elif line.startswith("TI  - "):
                                    current_record["title"] = line[6:].strip()
                                elif line.startswith("AB  - "):
                                    current_record["abstract"] = line[6:].strip()
                                elif line.startswith("AU  - "):
                                    if "authors" not in current_record:
                                        current_record["authors"] = []
                                    current_record["authors"].append(line[6:].strip())
                                elif line.startswith("DP  - "):
                                    current_record["publication_date"] = line[6:].strip()

                            if current_record:
                                records.append(current_record)

                            logger.info(f"Parsed {len(records)} records from batch {batch_index+1}/{num_batches}")
                            return records
                        else:
                            # For other databases or formats, return the raw result
                            return batch_result
                    except Exception as e:
                        logger.error(f"Error processing batch {batch_index+1}/{num_batches}: {str(e)}")
                        # Return empty list for this batch to continue with other batches
                        return []

            # Process all batches concurrently
            tasks = [process_batch(i) for i in range(num_batches)]
            batch_results = await asyncio.gather(*tasks)

            # Combine results from all batches
            all_records = []
            for batch_result in batch_results:
                if isinstance(batch_result, list):
                    all_records.extend(batch_result)
                else:
                    # If not a list, it's probably a raw result from a non-PubMed database
                    all_records.append(batch_result)

            logger.info(f"Fetched {len(all_records) if isinstance(all_records, list) else 'raw'} records in total")

            return {
                "count": count,
                "fetched": min(max_results, count),
                "records": all_records
            }
        except NCBIClientError as e:
            # Re-raise with more context
            logger.error(f"Error in batch search and fetch for '{query}' in {db}: {str(e)}")
            raise NCBIClientError(f"Failed to batch search and fetch: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error in batch search and fetch for '{query}' in {db}: {str(e)}")
            raise NCBIClientError(f"Unexpected error in batch search and fetch: {str(e)}")

    async def batch_fetch_sequences(self,
                                  ids: List[str],
                                  db: str = "nucleotide",
                                  batch_size: int = 50,
                                  return_type: str = "fasta",
                                  return_mode: str = "text",
                                  max_workers: int = 3) -> Dict[str, Any]:
        """
        Fetch sequences for a large list of IDs in batches.

        This method efficiently fetches sequences for a large list of IDs
        by splitting them into batches and processing them concurrently.

        Args:
            ids: List of sequence IDs to fetch
            db: Database to fetch from (default: "nucleotide")
            batch_size: Number of IDs per batch (default: 50)
            return_type: Return type (default: "fasta")
            return_mode: Return mode (default: "text")
            max_workers: Maximum number of concurrent workers (default: 3)

        Returns:
            Dictionary with fetched sequences

        Raises:
            ValidationError: If inputs are invalid
            NCBIClientError: If the fetch fails
        """
        # Validate inputs
        if not ids:
            raise ValidationError("IDs list cannot be empty")

        if batch_size < 1:
            raise ValidationError("batch_size must be at least 1")

        if max_workers < 1:
            raise ValidationError("max_workers must be at least 1")

        # Split IDs into batches
        batches = [ids[i:i+batch_size] for i in range(0, len(ids), batch_size)]
        total_batches = len(batches)

        logger.info(f"Fetching {len(ids)} sequences in {total_batches} batches of size {batch_size}")

        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_workers)

        # Define a worker function to process each batch
        async def process_batch(batch_index: int, batch_ids: List[str]) -> Dict[str, str]:
            async with semaphore:
                try:
                    logger.info(f"Processing batch {batch_index+1}/{total_batches} with {len(batch_ids)} IDs")

                    # Post IDs to History server
                    history_result = await self.post_ids_to_history(batch_ids, db)
                    web_env = history_result["web_env"]
                    query_key = history_result["query_key"]

                    # Fetch sequences from History server
                    sequences = await self.fetch_from_history(
                        web_env=web_env,
                        query_key=query_key,
                        db=db,
                        rettype=return_type,
                        retmode=return_mode
                    )

                    logger.info(f"Completed batch {batch_index+1}/{total_batches}")

                    # For FASTA format, parse the sequences
                    if return_type == "fasta" and return_mode == "text":
                        result = {}
                        current_id = None
                        current_seq = []

                        for line in sequences.strip().split("\n"):
                            if line.startswith(">"):
                                # Save previous sequence
                                if current_id and current_seq:
                                    result[current_id] = "\n".join(current_seq)

                                # Start new sequence
                                header = line[1:].strip()
                                current_id = header.split()[0]  # Extract ID from header
                                current_seq = [line]
                            elif current_id:
                                current_seq.append(line)

                        # Save last sequence
                        if current_id and current_seq:
                            result[current_id] = "\n".join(current_seq)

                        return result
                    else:
                        # For other formats, return the raw result
                        return {"batch_result": sequences}
                except Exception as e:
                    logger.error(f"Error processing batch {batch_index+1}/{total_batches}: {str(e)}")
                    # Return empty dict for this batch to continue with other batches
                    return {}

        # Process all batches concurrently
        tasks = [process_batch(i, batch) for i, batch in enumerate(batches)]
        batch_results = await asyncio.gather(*tasks)

        # Combine results from all batches
        all_sequences = {}
        for batch_result in batch_results:
            all_sequences.update(batch_result)

        logger.info(f"Fetched sequences for {len(all_sequences)} IDs in total")
        return {
            "count": len(all_sequences),
            "sequences": all_sequences
        }

    # Advanced Search Methods

    async def advanced_search(self,
                           db: str = "pubmed",
                           **search_criteria) -> Dict[str, Any]:
        """
        Perform an advanced search with multiple criteria.

        This method builds a complex search query from multiple search criteria
        and performs a search using ESearch.

        Args:
            db: Database to search (default: "pubmed")
            **search_criteria: Search criteria as keyword arguments
                For PubMed:
                - title: Search in title field
                - abstract: Search in abstract field
                - author: Search for author
                - journal: Search for journal
                - publication_date: Publication date range (e.g., "2020:2023")
                - publication_type: Publication type (e.g., "review")
                - mesh_terms: MeSH terms
                - keywords: Keywords
                - affiliation: Author affiliation
                - doi: DOI
                - free_text: Free text search
                - filters: List of filters to apply (e.g., ["free full text"])

                For sequence databases:
                - organism: Organism name
                - gene: Gene name
                - protein: Protein name
                - sequence_length: Sequence length range (e.g., "1000:2000")
                - molecule_type: Molecule type (e.g., "mrna")
                - source: Source database
                - free_text: Free text search

        Returns:
            Dictionary with search results

        Raises:
            ValidationError: If inputs are invalid
            NCBIClientError: If the search fails
        """
        # Validate inputs
        if not db:
            raise ValidationError("Database must be specified")

        if not search_criteria:
            raise ValidationError("At least one search criterion must be provided")

        try:
            # Build search query based on database
            query_parts = []

            if db == "pubmed":
                # PubMed-specific fields
                if "title" in search_criteria:
                    query_parts.append(f"{search_criteria['title']}[Title]")

                if "abstract" in search_criteria:
                    query_parts.append(f"{search_criteria['abstract']}[Abstract]")

                if "author" in search_criteria:
                    query_parts.append(f"{search_criteria['author']}[Author]")

                if "journal" in search_criteria:
                    query_parts.append(f"{search_criteria['journal']}[Journal]")

                if "publication_date" in search_criteria:
                    query_parts.append(f"{search_criteria['publication_date']}[Publication Date]")

                if "publication_type" in search_criteria:
                    query_parts.append(f"{search_criteria['publication_type']}[Publication Type]")

                if "mesh_terms" in search_criteria:
                    query_parts.append(f"{search_criteria['mesh_terms']}[MeSH Terms]")

                if "keywords" in search_criteria:
                    query_parts.append(f"{search_criteria['keywords']}[Keywords]")

                if "affiliation" in search_criteria:
                    query_parts.append(f"{search_criteria['affiliation']}[Affiliation]")

                if "doi" in search_criteria:
                    query_parts.append(f"{search_criteria['doi']}[DOI]")

                # Add filters
                if "filters" in search_criteria and isinstance(search_criteria["filters"], list):
                    for filter_name in search_criteria["filters"]:
                        query_parts.append(f"{filter_name}[Filter]")
            elif db in ("nucleotide", "protein", "gene"):
                # Sequence database fields
                if "organism" in search_criteria:
                    query_parts.append(f"{search_criteria['organism']}[Organism]")

                if "gene" in search_criteria:
                    query_parts.append(f"{search_criteria['gene']}[Gene Name]")

                if "protein" in search_criteria:
                    query_parts.append(f"{search_criteria['protein']}[Protein Name]")

                if "sequence_length" in search_criteria:
                    query_parts.append(f"{search_criteria['sequence_length']}[Sequence Length]")

                if "molecule_type" in search_criteria:
                    query_parts.append(f"{search_criteria['molecule_type']}[Molecule Type]")

                if "source" in search_criteria:
                    query_parts.append(f"{search_criteria['source']}[Source]")

            # Add free text search (applies to all databases)
            if "free_text" in search_criteria:
                query_parts.append(search_criteria["free_text"])

            # Combine query parts with AND
            query = " AND ".join(query_parts)

            # Extract search parameters
            search_params = {}
            for param in ["retmax", "retstart", "sort", "field", "datetype", "reldate", "mindate", "maxdate"]:
                if param in search_criteria:
                    search_params[param] = search_criteria[param]

            # Perform search
            logger.info(f"Performing advanced search in {db} with query: {query}")
            result = await self.search_pubmed(query, db=db, **search_params)

            return result
        except NCBIClientError as e:
            # Re-raise with more context
            logger.error(f"Error performing advanced search in {db}: {str(e)}")
            raise NCBIClientError(f"Failed to perform advanced search: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error performing advanced search in {db}: {str(e)}")
            raise NCBIClientError(f"Unexpected error performing advanced search: {str(e)}")

    async def date_range_search(self,
                              query: str,
                              db: str = "pubmed",
                              date_type: str = "pdat",
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None,
                              relative_date: Optional[int] = None,
                              **search_params) -> Dict[str, Any]:
        """
        Search with date range constraints.

        This method performs a search with date range constraints using ESearch.

        Args:
            query: Search query
            db: Database to search (default: "pubmed")
            date_type: Type of date (default: "pdat")
                Options for PubMed: "pdat" (publication date), "edat" (entrez date), "mdat" (modification date)
            start_date: Start date in YYYY/MM/DD format (optional)
            end_date: End date in YYYY/MM/DD format (optional)
            relative_date: Relative date in days (optional)
            **search_params: Additional search parameters
                - retmax: Maximum number of results to return
                - sort: Sort order (e.g., "relevance", "pub_date")
                - field: Field to search in (e.g., "title", "author")

        Returns:
            Dictionary with search results

        Raises:
            ValidationError: If inputs are invalid
            NCBIClientError: If the search fails
        """
        # Validate inputs
        if not query or not query.strip():
            raise ValidationError("Search query cannot be empty")

        if not db:
            raise ValidationError("Database must be specified")

        if not date_type:
            raise ValidationError("Date type must be specified")

        if (start_date and not end_date) or (not start_date and end_date):
            raise ValidationError("Both start_date and end_date must be provided together")

        if start_date and end_date and relative_date:
            raise ValidationError("Cannot specify both date range and relative date")

        if not start_date and not end_date and not relative_date:
            raise ValidationError("Either date range or relative date must be specified")

        try:
            # Prepare parameters
            params = {
                "db": db,
                "term": query.strip(),
                "datetype": date_type
            }

            # Add date range or relative date
            if start_date and end_date:
                params["mindate"] = start_date
                params["maxdate"] = end_date
            elif relative_date:
                params["reldate"] = relative_date

            # Add additional search parameters
            params.update(search_params)

            # Perform search
            logger.info(f"Performing date range search in {db} with query: {query}")
            result = await self._make_request("esearch.fcgi", params)

            return result
        except NCBIClientError as e:
            # Re-raise with more context
            logger.error(f"Error performing date range search in {db}: {str(e)}")
            raise NCBIClientError(f"Failed to perform date range search: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error performing date range search in {db}: {str(e)}")
            raise NCBIClientError(f"Unexpected error performing date range search: {str(e)}")

    async def field_search(self,
                         terms: Dict[str, str],
                         db: str = "pubmed",
                         operator: str = "AND",
                         **search_params) -> Dict[str, Any]:
        """
        Search with field-specific terms.

        This method builds a search query with field-specific terms and performs
        a search using ESearch.

        Args:
            terms: Dictionary mapping fields to search terms
            db: Database to search (default: "pubmed")
            operator: Operator to combine terms (default: "AND")
                Options: "AND", "OR", "NOT"
            **search_params: Additional search parameters
                - retmax: Maximum number of results to return
                - sort: Sort order (e.g., "relevance", "pub_date")
                - datetype: Type of date (e.g., "pdat", "edat")
                - reldate: Relative date (e.g., 30 for last 30 days)
                - mindate/maxdate: Date range (format: YYYY/MM/DD)

        Returns:
            Dictionary with search results

        Raises:
            ValidationError: If inputs are invalid
            NCBIClientError: If the search fails
        """
        # Validate inputs
        if not terms:
            raise ValidationError("Terms dictionary cannot be empty")

        if not db:
            raise ValidationError("Database must be specified")

        if operator not in ("AND", "OR", "NOT"):
            raise ValidationError("Operator must be one of: AND, OR, NOT")

        try:
            # Build search query
            query_parts = []

            for field, term in terms.items():
                if not term or not term.strip():
                    continue

                # Format field name for search
                field_tag = field
                if not field.endswith("]"):
                    field_tag = f"[{field}]"

                query_parts.append(f"{term.strip()}{field_tag}")

            if not query_parts:
                raise ValidationError("No valid terms provided")

            # Combine query parts with operator
            query = f" {operator} ".join(query_parts)

            # Perform search
            logger.info(f"Performing field search in {db} with query: {query}")
            result = await self.search_pubmed(query, db=db, **search_params)

            return result
        except NCBIClientError as e:
            # Re-raise with more context
            logger.error(f"Error performing field search in {db}: {str(e)}")
            raise NCBIClientError(f"Failed to perform field search: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error performing field search in {db}: {str(e)}")
            raise NCBIClientError(f"Unexpected error performing field search: {str(e)}")

    async def proximity_search(self,
                             terms: List[str],
                             field: str = "Title/Abstract",
                             distance: int = 5,
                             **search_params) -> Dict[str, Any]:
        """
        Perform a proximity search in PubMed.

        This method performs a proximity search for multiple terms appearing
        within a specified distance of each other in a field.

        Args:
            terms: List of terms to search for
            field: Field to search in (default: "Title/Abstract")
                Options: "Title", "Title/Abstract", "Abstract"
            distance: Maximum distance between terms (default: 5)
            **search_params: Additional search parameters
                - retmax: Maximum number of results to return
                - sort: Sort order (e.g., "relevance", "pub_date")
                - datetype: Type of date (e.g., "pdat", "edat")
                - reldate: Relative date (e.g., 30 for last 30 days)
                - mindate/maxdate: Date range (format: YYYY/MM/DD)

        Returns:
            Dictionary with search results

        Raises:
            ValidationError: If inputs are invalid
            NCBIClientError: If the search fails
        """
        # Validate inputs
        if not terms or len(terms) < 2:
            raise ValidationError("At least two terms must be provided for proximity search")

        if field not in ("Title", "Title/Abstract", "Abstract"):
            raise ValidationError("Field must be one of: Title, Title/Abstract, Abstract")

        if distance < 1:
            raise ValidationError("Distance must be at least 1")

        try:
            # Build proximity search query
            # Format: "term1 term2 term3"[field:~distance]
            terms_str = " ".join(term.strip() for term in terms if term and term.strip())
            query = f"\"{terms_str}\"[{field}:~{distance}]"

            # Perform search
            logger.info(f"Performing proximity search in PubMed with query: {query}")
            result = await self.search_pubmed(query, **search_params)

            return result
        except NCBIClientError as e:
            # Re-raise with more context
            logger.error(f"Error performing proximity search: {str(e)}")
            raise NCBIClientError(f"Failed to perform proximity search: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error performing proximity search: {str(e)}")
            raise NCBIClientError(f"Unexpected error performing proximity search: {str(e)}")

    # PubMed Central (PMC) Methods

    @cached(prefix="pmc_search", data_type="search")
    async def search_pmc(self,
                       query: str,
                       max_results: int = 20,
                       sort: str = "relevance",
                       min_date: Optional[str] = None,
                       max_date: Optional[str] = None,
                       use_history: bool = False) -> Dict[str, Any]:
        """
        Search PubMed Central (PMC) with the given query.

        This method uses ESearch to search the PMC database.

        Args:
            query: Search query
            max_results: Maximum number of results to return (default: 20)
            sort: Sort order (default: "relevance")
                Options: "relevance", "pub_date", "journal", "title"
            min_date: Minimum date (YYYY/MM/DD format)
            max_date: Maximum date (YYYY/MM/DD format)
            use_history: Whether to use the Entrez History server (default: False)

        Returns:
            Search results including PMCIDs and metadata

        Raises:
            ValidationError: If the query is invalid
            NCBIClientError: If the search fails
        """
        # Validate inputs
        if not query or not query.strip():
            raise ValidationError("Search query cannot be empty")

        if max_results < 1:
            raise ValidationError("max_results must be at least 1")

        # Prepare search parameters
        params = {
            "db": "pmc",
            "term": query.strip(),
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
            # Make the request
            logger.info(f"Searching PMC for '{query}' (max_results={max_results})")
            data = await self._make_request("esearch.fcgi", params)

            # Log the results
            count = int(data.get("esearchresult", {}).get("count", "0"))
            logger.info(f"Found {count} results for '{query}' in PMC")

            return data
        except NCBIClientError as e:
            # Re-raise with more context
            logger.error(f"Error searching PMC for '{query}': {str(e)}")
            raise NCBIClientError(f"Failed to search PMC: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error searching PMC for '{query}': {str(e)}")
            raise NCBIClientError(f"Unexpected error searching PMC: {str(e)}")

    @cached(prefix="pmc_fetch", data_type="search")
    async def fetch_pmc_article(self, pmcid: str, format: str = "xml") -> str:
        """
        Fetch a full-text article from PubMed Central (PMC).

        This method uses EFetch to retrieve a full-text article from PMC.

        Args:
            pmcid: PMC ID (e.g., "PMC1234567" or just "1234567")
            format: Format to retrieve (default: "xml")
                Options: "xml", "medline", "pdf"

        Returns:
            Full-text article in the specified format

        Raises:
            ValidationError: If the PMCID is invalid
            NCBIClientError: If the fetch fails
        """
        # Validate inputs
        if not pmcid or not pmcid.strip():
            raise ValidationError("PMCID cannot be empty")

        # Normalize PMCID (add PMC prefix if missing)
        normalized_pmcid = pmcid.strip()
        if not normalized_pmcid.startswith("PMC"):
            normalized_pmcid = f"PMC{normalized_pmcid}"

        # Validate format
        valid_formats = ["xml", "medline", "pdf"]
        if format not in valid_formats:
            raise ValidationError(f"Invalid format: {format}. Must be one of {valid_formats}")

        # Prepare parameters
        params = {
            "db": "pmc",
            "id": normalized_pmcid
        }

        # Set rettype and retmode based on format
        if format == "xml":
            params["rettype"] = "xml"
            params["retmode"] = "xml"
        elif format == "medline":
            params["rettype"] = "medline"
            params["retmode"] = "text"
        elif format == "pdf":
            # For PDF, we need to use a different approach
            # We'll get the PDF URL and then download it
            return await self._fetch_pmc_pdf(normalized_pmcid)

        try:
            # Make the request
            logger.info(f"Fetching PMC article {normalized_pmcid} in {format} format")
            result = await self._make_request("efetch.fcgi", params, return_json=False)

            return result
        except NCBIClientError as e:
            # Re-raise with more context
            logger.error(f"Error fetching PMC article {normalized_pmcid}: {str(e)}")
            raise NCBIClientError(f"Failed to fetch PMC article: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error fetching PMC article {normalized_pmcid}: {str(e)}")
            raise NCBIClientError(f"Unexpected error fetching PMC article: {str(e)}")

    async def _fetch_pmc_pdf(self, pmcid: str) -> str:
        """
        Fetch a PDF from PubMed Central (PMC).

        This is a helper method for fetch_pmc_article() to handle PDF format.

        Args:
            pmcid: PMC ID (with PMC prefix)

        Returns:
            URL to the PDF file

        Raises:
            NCBIClientError: If the fetch fails
        """
        try:
            # First, get the article in XML format to extract the PDF URL
            xml_content = await self.fetch_pmc_article(pmcid, format="xml")

            # Parse the XML to find the PDF URL
            try:
                root = ET.fromstring(xml_content)

                # Look for the PDF URL in the XML
                # The exact path depends on the XML structure, which may vary
                pdf_url = None

                # Try different paths to find the PDF URL
                # Method 1: Look for a specific element with the PDF URL
                for elem in root.findall(".//self-uri[@content-type='pdf']"):
                    if "href" in elem.attrib:
                        pdf_url = elem.attrib["href"]
                        break

                # Method 2: Look for a specific element with the PDF URL
                if not pdf_url:
                    for elem in root.findall(".//supplementary-material[@content-type='pdf']"):
                        if "href" in elem.attrib:
                            pdf_url = elem.attrib["href"]
                            break

                # Method 3: Construct the PDF URL from the PMCID
                if not pdf_url:
                    pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/{pmcid}.pdf"

                return pdf_url
            except ET.ParseError as e:
                logger.error(f"Error parsing XML for PMC article {pmcid}: {str(e)}")
                # Fallback to constructing the PDF URL from the PMCID
                return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/{pmcid}.pdf"
        except Exception as e:
            logger.error(f"Error fetching PDF for PMC article {pmcid}: {str(e)}")
            raise NCBIClientError(f"Failed to fetch PDF for PMC article: {str(e)}")

    @cached(prefix="pmc_ids", data_type="search")
    async def convert_pmid_to_pmcid(self, pmids: Union[str, List[str]]) -> Dict[str, str]:
        """
        Convert PubMed IDs (PMIDs) to PubMed Central IDs (PMCIDs).

        This method uses the NCBI ID Converter API to convert PMIDs to PMCIDs.

        Args:
            pmids: PMID or list of PMIDs to convert

        Returns:
            Dictionary mapping PMIDs to PMCIDs (None if no PMCID exists)

        Raises:
            ValidationError: If the PMIDs are invalid
            NCBIClientError: If the conversion fails
        """
        # Validate inputs
        if isinstance(pmids, str):
            pmids = [pmids]

        if not pmids:
            raise ValidationError("PMIDs list cannot be empty")

        # Check for invalid PMIDs
        invalid_pmids = [pmid for pmid in pmids if not pmid or not pmid.strip()]
        if invalid_pmids:
            raise ValidationError(f"Invalid PMIDs: {invalid_pmids}")

        try:
            # Prepare parameters
            params = {
                "dbfrom": "pubmed",
                "db": "pmc",
                "id": ",".join(pmid.strip() for pmid in pmids)
            }

            # Make the request
            logger.info(f"Converting {len(pmids)} PMIDs to PMCIDs")
            result = await self._make_request("elink.fcgi", params)

            # Parse the result to extract PMCIDs
            pmid_to_pmcid = {}

            try:
                linksets = result.get("linksets", [])
                for linkset in linksets:
                    pmid = linkset.get("ids", [None])[0]  # Get the source PMID
                    if not pmid:
                        continue

                    # Initialize with None (no PMCID found)
                    pmid_to_pmcid[pmid] = None

                    # Look for PMCIDs in the linkset
                    for linksetdb in linkset.get("linksetdbs", []):
                        if linksetdb.get("linkname") == "pubmed_pmc":
                            links = linksetdb.get("links", [])
                            if links:
                                # Get the first PMCID (there should only be one)
                                pmcid = links[0]
                                pmid_to_pmcid[pmid] = f"PMC{pmcid}"
                                break
            except Exception as e:
                logger.error(f"Error parsing PMID to PMCID conversion result: {str(e)}")

            # Add entries for PMIDs that were not found in the result
            for pmid in pmids:
                if pmid not in pmid_to_pmcid:
                    pmid_to_pmcid[pmid] = None

            logger.info(f"Converted {len(pmids)} PMIDs to PMCIDs, found {sum(1 for pmcid in pmid_to_pmcid.values() if pmcid)} matches")
            return pmid_to_pmcid
        except NCBIClientError as e:
            # Re-raise with more context
            logger.error(f"Error converting PMIDs to PMCIDs: {str(e)}")
            raise NCBIClientError(f"Failed to convert PMIDs to PMCIDs: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error converting PMIDs to PMCIDs: {str(e)}")
            raise NCBIClientError(f"Unexpected error converting PMIDs to PMCIDs: {str(e)}")

    @cached(prefix="pmc_ids", data_type="search")
    async def convert_pmcid_to_pmid(self, pmcids: Union[str, List[str]]) -> Dict[str, str]:
        """
        Convert PubMed Central IDs (PMCIDs) to PubMed IDs (PMIDs).

        This method uses the NCBI ID Converter API to convert PMCIDs to PMIDs.

        Args:
            pmcids: PMCID or list of PMCIDs to convert

        Returns:
            Dictionary mapping PMCIDs to PMIDs (None if no PMID exists)

        Raises:
            ValidationError: If the PMCIDs are invalid
            NCBIClientError: If the conversion fails
        """
        # Validate inputs
        if isinstance(pmcids, str):
            pmcids = [pmcids]

        if not pmcids:
            raise ValidationError("PMCIDs list cannot be empty")

        # Normalize PMCIDs (add PMC prefix if missing)
        normalized_pmcids = []
        for pmcid in pmcids:
            if not pmcid or not pmcid.strip():
                raise ValidationError(f"Invalid PMCID: {pmcid}")

            normalized_pmcid = pmcid.strip()
            if not normalized_pmcid.startswith("PMC"):
                normalized_pmcid = f"PMC{normalized_pmcid}"

            normalized_pmcids.append(normalized_pmcid)

        try:
            # Prepare parameters
            params = {
                "dbfrom": "pmc",
                "db": "pubmed",
                "id": ",".join(pmcid.replace("PMC", "") for pmcid in normalized_pmcids)  # Remove PMC prefix for the API
            }

            # Make the request
            logger.info(f"Converting {len(normalized_pmcids)} PMCIDs to PMIDs")
            result = await self._make_request("elink.fcgi", params)

            # Parse the result to extract PMIDs
            pmcid_to_pmid = {}

            try:
                linksets = result.get("linksets", [])
                for linkset in linksets:
                    pmcid_num = linkset.get("ids", [None])[0]  # Get the source PMCID (without PMC prefix)
                    if not pmcid_num:
                        continue

                    pmcid = f"PMC{pmcid_num}"

                    # Initialize with None (no PMID found)
                    pmcid_to_pmid[pmcid] = None

                    # Look for PMIDs in the linkset
                    for linksetdb in linkset.get("linksetdbs", []):
                        if linksetdb.get("linkname") == "pmc_pubmed":
                            links = linksetdb.get("links", [])
                            if links:
                                # Get the first PMID (there should only be one)
                                pmid = links[0]
                                pmcid_to_pmid[pmcid] = pmid
                                break
            except Exception as e:
                logger.error(f"Error parsing PMCID to PMID conversion result: {str(e)}")

            # Add entries for PMCIDs that were not found in the result
            for pmcid in normalized_pmcids:
                if pmcid not in pmcid_to_pmid:
                    pmcid_to_pmid[pmcid] = None

            logger.info(f"Converted {len(normalized_pmcids)} PMCIDs to PMIDs, found {sum(1 for pmid in pmcid_to_pmid.values() if pmid)} matches")
            return pmcid_to_pmid
        except NCBIClientError as e:
            # Re-raise with more context
            logger.error(f"Error converting PMCIDs to PMIDs: {str(e)}")
            raise NCBIClientError(f"Failed to convert PMCIDs to PMIDs: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error converting PMCIDs to PMIDs: {str(e)}")
            raise NCBIClientError(f"Unexpected error converting PMCIDs to PMIDs: {str(e)}")

    @cached(prefix="pmc_extract", data_type="search")
    async def extract_pmc_article_sections(self, pmcid: str) -> Dict[str, Any]:
        """
        Extract sections from a PMC article.

        This method fetches a PMC article in XML format and extracts its sections,
        including title, abstract, introduction, methods, results, discussion, and
        references.

        Args:
            pmcid: PMC ID (e.g., "PMC1234567" or just "1234567")

        Returns:
            Dictionary with article sections

        Raises:
            ValidationError: If the PMCID is invalid
            NCBIClientError: If the fetch fails
        """
        # Validate inputs
        if not pmcid or not pmcid.strip():
            raise ValidationError("PMCID cannot be empty")

        try:
            # Fetch the article in XML format
            xml_content = await self.fetch_pmc_article(pmcid, format="xml")

            # Parse the XML to extract sections
            try:
                root = ET.fromstring(xml_content)

                # Initialize result dictionary
                article = {
                    "pmcid": pmcid,
                    "title": "",
                    "abstract": "",
                    "sections": [],
                    "references": []
                }

                # Extract article title
                title_elem = root.find(".//article-title")
                if title_elem is not None and title_elem.text:
                    article["title"] = self._get_element_text(title_elem)

                # Extract abstract
                abstract_elem = root.find(".//abstract")
                if abstract_elem is not None:
                    article["abstract"] = self._get_element_text(abstract_elem)

                # Extract sections
                body_elem = root.find(".//body")
                if body_elem is not None:
                    for section_elem in body_elem.findall(".//sec"):
                        section = self._parse_section(section_elem)
                        if section:
                            article["sections"].append(section)

                # Extract references
                ref_list_elem = root.find(".//ref-list")
                if ref_list_elem is not None:
                    for ref_elem in ref_list_elem.findall(".//ref"):
                        reference = self._parse_reference(ref_elem)
                        if reference:
                            article["references"].append(reference)

                return article
            except ET.ParseError as e:
                logger.error(f"Error parsing XML for PMC article {pmcid}: {str(e)}")
                raise NCBIClientError(f"Failed to parse XML for PMC article: {str(e)}")
        except NCBIClientError as e:
            # Re-raise with more context
            logger.error(f"Error extracting sections from PMC article {pmcid}: {str(e)}")
            raise NCBIClientError(f"Failed to extract sections from PMC article: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error extracting sections from PMC article {pmcid}: {str(e)}")
            raise NCBIClientError(f"Unexpected error extracting sections from PMC article: {str(e)}")

    def _get_element_text(self, element: ET.Element) -> str:
        """
        Get the text content of an XML element, including its children.

        Args:
            element: XML element

        Returns:
            Text content of the element
        """
        if element is None:
            return ""

        # Get the element's text
        text = element.text or ""

        # Add text from child elements
        for child in element:
            # Add the child's text
            child_text = self._get_element_text(child)
            if child_text:
                text += " " + child_text

            # Add the tail text (text after the child element)
            if child.tail:
                text += child.tail

        return text.strip()

    def _parse_section(self, section_elem: ET.Element) -> Dict[str, Any]:
        """
        Parse a section element from a PMC article.

        Args:
            section_elem: Section element

        Returns:
            Dictionary with section title and content
        """
        if section_elem is None:
            return None

        # Initialize section dictionary
        section = {
            "title": "",
            "content": "",
            "subsections": []
        }

        # Extract section title
        title_elem = section_elem.find("./title")
        if title_elem is not None:
            section["title"] = self._get_element_text(title_elem)

        # Extract section content (paragraphs)
        paragraphs = []
        for p_elem in section_elem.findall("./p"):
            paragraph = self._get_element_text(p_elem)
            if paragraph:
                paragraphs.append(paragraph)

        section["content"] = "\n\n".join(paragraphs)

        # Extract subsections
        for subsec_elem in section_elem.findall("./sec"):
            subsection = self._parse_section(subsec_elem)
            if subsection:
                section["subsections"].append(subsection)

        return section

    def _parse_reference(self, ref_elem: ET.Element) -> Dict[str, str]:
        """
        Parse a reference element from a PMC article.

        Args:
            ref_elem: Reference element

        Returns:
            Dictionary with reference details
        """
        if ref_elem is None:
            return None

        # Initialize reference dictionary
        reference = {
            "id": ref_elem.get("id", ""),
            "title": "",
            "authors": [],
            "journal": "",
            "year": "",
            "volume": "",
            "issue": "",
            "pages": "",
            "doi": "",
            "pmid": ""
        }

        # Extract reference details
        element_citation = ref_elem.find(".//element-citation") or ref_elem.find(".//mixed-citation")
        if element_citation is not None:
            # Extract title
            article_title = element_citation.find(".//article-title")
            if article_title is not None:
                reference["title"] = self._get_element_text(article_title)

            # Extract authors
            for person_group in element_citation.findall(".//person-group"):
                for name in person_group.findall(".//name"):
                    surname = name.find("./surname")
                    given_names = name.find("./given-names")

                    if surname is not None and given_names is not None:
                        author = f"{self._get_element_text(surname)} {self._get_element_text(given_names)}"
                        reference["authors"].append(author)

            # Extract journal
            source = element_citation.find(".//source")
            if source is not None:
                reference["journal"] = self._get_element_text(source)

            # Extract year
            year = element_citation.find(".//year")
            if year is not None:
                reference["year"] = self._get_element_text(year)

            # Extract volume
            volume = element_citation.find(".//volume")
            if volume is not None:
                reference["volume"] = self._get_element_text(volume)

            # Extract issue
            issue = element_citation.find(".//issue")
            if issue is not None:
                reference["issue"] = self._get_element_text(issue)

            # Extract pages
            fpage = element_citation.find(".//fpage")
            lpage = element_citation.find(".//lpage")
            if fpage is not None and lpage is not None:
                reference["pages"] = f"{self._get_element_text(fpage)}-{self._get_element_text(lpage)}"
            elif fpage is not None:
                reference["pages"] = self._get_element_text(fpage)

            # Extract DOI
            pub_id_doi = element_citation.find(".//pub-id[@pub-id-type='doi']")
            if pub_id_doi is not None:
                reference["doi"] = self._get_element_text(pub_id_doi)

            # Extract PMID
            pub_id_pmid = element_citation.find(".//pub-id[@pub-id-type='pmid']")
            if pub_id_pmid is not None:
                reference["pmid"] = self._get_element_text(pub_id_pmid)

        return reference

    @cached(prefix="pmc_search_fetch", data_type="search")
    async def search_and_fetch_pmc(self,
                                 query: str,
                                 max_results: int = 20,
                                 include_full_text: bool = False,
                                 **search_params) -> List[Dict[str, Any]]:
        """
        Search PMC and fetch articles in one step.

        This method combines search_pmc and fetch_pmc_article to search PMC
        and fetch the articles in one step.

        Args:
            query: Search query
            max_results: Maximum number of results to return (default: 20)
            include_full_text: Whether to include full text (default: False)
            **search_params: Additional search parameters
                - sort: Sort order (e.g., "relevance", "pub_date")
                - min_date/max_date: Date range (format: YYYY/MM/DD)

        Returns:
            List of articles with metadata and optionally full text

        Raises:
            ValidationError: If the query is invalid
            NCBIClientError: If the search or fetch fails
        """
        # Validate inputs
        if not query or not query.strip():
            raise ValidationError("Search query cannot be empty")

        if max_results < 1:
            raise ValidationError("max_results must be at least 1")

        try:
            # Search PMC
            logger.info(f"Searching PMC for '{query}' (max_results={max_results})")
            search_result = await self.search_pmc(query, max_results=max_results, **search_params)

            # Extract PMCIDs
            pmcids = search_result.get("esearchresult", {}).get("idlist", [])
            if not pmcids:
                logger.warning(f"No results found for '{query}' in PMC")
                return []

            # Fetch articles
            articles = []
            for pmcid in pmcids:
                try:
                    # Normalize PMCID (add PMC prefix if missing)
                    normalized_pmcid = pmcid
                    if not normalized_pmcid.startswith("PMC"):
                        normalized_pmcid = f"PMC{normalized_pmcid}"

                    # Get article metadata
                    article = {
                        "pmcid": normalized_pmcid
                    }

                    # Include full text if requested
                    if include_full_text:
                        # Extract article sections
                        article_data = await self.extract_pmc_article_sections(normalized_pmcid)
                        article.update(article_data)
                    else:
                        # Just get the abstract
                        xml_content = await self.fetch_pmc_article(normalized_pmcid, format="xml")
                        root = ET.fromstring(xml_content)

                        # Extract title
                        title_elem = root.find(".//article-title")
                        if title_elem is not None:
                            article["title"] = self._get_element_text(title_elem)

                        # Extract abstract
                        abstract_elem = root.find(".//abstract")
                        if abstract_elem is not None:
                            article["abstract"] = self._get_element_text(abstract_elem)

                    # Get PMID if available
                    try:
                        pmcid_to_pmid = await self.convert_pmcid_to_pmid(normalized_pmcid)
                        pmid = pmcid_to_pmid.get(normalized_pmcid)
                        if pmid:
                            article["pmid"] = pmid
                    except Exception as e:
                        logger.warning(f"Error getting PMID for {normalized_pmcid}: {str(e)}")

                    articles.append(article)
                except Exception as e:
                    logger.error(f"Error fetching PMC article {pmcid}: {str(e)}")
                    # Continue with the next article instead of failing completely
                    continue

            logger.info(f"Fetched {len(articles)} articles from PMC")
            return articles
        except NCBIClientError as e:
            # Re-raise with more context
            logger.error(f"Error in search_and_fetch_pmc for '{query}': {str(e)}")
            raise NCBIClientError(f"Failed to search and fetch PMC articles: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error in search_and_fetch_pmc for '{query}': {str(e)}")
            raise NCBIClientError(f"Unexpected error searching and fetching PMC articles: {str(e)}")

    @cached(prefix="pmc_batch_fetch", data_type="search")
    async def batch_fetch_pmc_articles(self,
                                     pmcids: List[str],
                                     include_full_text: bool = False,
                                     batch_size: int = 10,
                                     max_workers: int = 3) -> List[Dict[str, Any]]:
        """
        Fetch PMC articles in batches.

        This method efficiently fetches PMC articles in batches by splitting them
        into batches and processing them concurrently.

        Args:
            pmcids: List of PMCIDs to fetch
            include_full_text: Whether to include full text (default: False)
            batch_size: Number of PMCIDs per batch (default: 10)
            max_workers: Maximum number of concurrent workers (default: 3)

        Returns:
            List of articles with metadata and optionally full text

        Raises:
            ValidationError: If the PMCIDs are invalid
            NCBIClientError: If the fetch fails
        """
        # Validate inputs
        if not pmcids:
            raise ValidationError("PMCIDs list cannot be empty")

        if batch_size < 1:
            raise ValidationError("batch_size must be at least 1")

        if max_workers < 1:
            raise ValidationError("max_workers must be at least 1")

        # Normalize PMCIDs (add PMC prefix if missing)
        normalized_pmcids = []
        for pmcid in pmcids:
            if not pmcid or not pmcid.strip():
                continue

            normalized_pmcid = pmcid.strip()
            if not normalized_pmcid.startswith("PMC"):
                normalized_pmcid = f"PMC{normalized_pmcid}"

            normalized_pmcids.append(normalized_pmcid)

        if not normalized_pmcids:
            logger.warning("No valid PMCIDs provided")
            return []

        # Split PMCIDs into batches
        batches = [normalized_pmcids[i:i+batch_size] for i in range(0, len(normalized_pmcids), batch_size)]
        total_batches = len(batches)

        logger.info(f"Fetching {len(normalized_pmcids)} PMC articles in {total_batches} batches of size {batch_size}")

        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_workers)

        # Define a worker function to process each batch
        async def process_batch(batch_index: int, batch_pmcids: List[str]) -> List[Dict[str, Any]]:
            async with semaphore:
                try:
                    logger.info(f"Processing batch {batch_index+1}/{total_batches} with {len(batch_pmcids)} PMCIDs")

                    # Fetch articles
                    batch_articles = []
                    for pmcid in batch_pmcids:
                        try:
                            # Get article metadata
                            article = {
                                "pmcid": pmcid
                            }

                            # Include full text if requested
                            if include_full_text:
                                # Extract article sections
                                article_data = await self.extract_pmc_article_sections(pmcid)
                                article.update(article_data)
                            else:
                                # Just get the abstract
                                xml_content = await self.fetch_pmc_article(pmcid, format="xml")
                                root = ET.fromstring(xml_content)

                                # Extract title
                                title_elem = root.find(".//article-title")
                                if title_elem is not None:
                                    article["title"] = self._get_element_text(title_elem)

                                # Extract abstract
                                abstract_elem = root.find(".//abstract")
                                if abstract_elem is not None:
                                    article["abstract"] = self._get_element_text(abstract_elem)

                            # Get PMID if available
                            try:
                                pmcid_to_pmid = await self.convert_pmcid_to_pmid(pmcid)
                                pmid = pmcid_to_pmid.get(pmcid)
                                if pmid:
                                    article["pmid"] = pmid
                            except Exception as e:
                                logger.warning(f"Error getting PMID for {pmcid}: {str(e)}")

                            batch_articles.append(article)
                        except Exception as e:
                            logger.error(f"Error fetching PMC article {pmcid}: {str(e)}")
                            # Continue with the next article instead of failing completely
                            continue

                    logger.info(f"Completed batch {batch_index+1}/{total_batches} with {len(batch_articles)} articles")
                    return batch_articles
                except Exception as e:
                    logger.error(f"Error processing batch {batch_index+1}/{total_batches}: {str(e)}")
                    # Return empty list for this batch to continue with other batches
                    return []

        # Process all batches concurrently
        tasks = [process_batch(i, batch) for i, batch in enumerate(batches)]
        batch_results = await asyncio.gather(*tasks)

        # Combine results from all batches
        all_articles = []
        for batch_result in batch_results:
            all_articles.extend(batch_result)

        logger.info(f"Fetched {len(all_articles)}/{len(normalized_pmcids)} PMC articles in total")
        return all_articles

    async def _batch_pmids(self, pmids: List[str], batch_size: Optional[int] = None) -> List[List[str]]:
        """
        Split a list of PMIDs into batches.

        Args:
            pmids: List of PMIDs
            batch_size: Batch size (default: self.max_batch_size)

        Returns:
            List of PMID batches

        Raises:
            ValidationError: If pmids is not a list or contains invalid PMIDs
        """
        # Validate inputs
        if not isinstance(pmids, list):
            raise ValidationError("pmids must be a list")

        # Filter out empty PMIDs
        valid_pmids = [pmid for pmid in pmids if pmid and pmid.strip()]
        if len(valid_pmids) < len(pmids):
            logger.warning(f"Filtered out {len(pmids) - len(valid_pmids)} empty PMIDs")

        if not valid_pmids:
            logger.warning("No valid PMIDs provided")
            return []

        # Use default batch size if not provided
        batch_size = batch_size or self.max_batch_size

        # Validate batch size
        if batch_size < 1:
            logger.warning(f"Invalid batch size {batch_size}, using default {self.max_batch_size}")
            batch_size = self.max_batch_size

        # Split PMIDs into batches
        batches = [valid_pmids[i:i + batch_size] for i in range(0, len(valid_pmids), batch_size)]
        logger.debug(f"Split {len(valid_pmids)} PMIDs into {len(batches)} batches of size {batch_size}")
        return batches

    async def _make_request(self, endpoint: str, params: Dict[str, Any], return_json: bool = True) -> Union[Dict[str, Any], str]:
        """
        Make a request to the NCBI API with retry logic and caching.

        Args:
            endpoint: API endpoint
            params: Request parameters
            return_json: Whether to return JSON (default: True)

        Returns:
            Response data as JSON or text

        Raises:
            NCBIClientError: If the request fails after retries
        """
        # Generate cache key if caching is enabled
        cache_key = None
        if self.use_cache:
            # Create a deterministic cache key from endpoint and params
            param_str = json.dumps(params, sort_keys=True)
            cache_key = f"ncbi:{endpoint}:{hashlib.md5(param_str.encode()).hexdigest()}"

            # Try to get from cache
            cached_result = await cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {endpoint}")
                return cached_result

        # Add common parameters
        request_params = params.copy()  # Make a copy to avoid modifying the original
        request_params.update({
            "tool": self.tool,
            "email": self.email,
        })

        # Add retmode if not specified
        if "retmode" not in request_params:
            request_params["retmode"] = "json" if return_json else "xml"

        # Add API key if available
        if self.api_key:
            request_params["api_key"] = self.api_key

        # Apply rate limiting
        await self.rate_limiter.acquire()

        # Prepare request
        url = f"{self.base_url}{endpoint}"
        logger.debug(f"Making request to {url} with params {request_params}")

        # Implement retry logic
        retries = 0
        max_retries = self.max_retries
        backoff_factor = 1.5

        while True:
            try:
                # Make request
                response = await self.client.get(url, params=request_params)

                # Check for HTTP errors
                if response.status_code >= 400:
                    error_msg = f"NCBI API error: {response.status_code} {response.reason_phrase}"
                    logger.warning(f"{error_msg} for {url}")

                    # Handle specific error codes
                    if response.status_code == 429:
                        error_msg = "Rate limit exceeded"
                    elif response.status_code == 400:
                        error_msg = f"Bad request: {response.text}"
                    elif response.status_code == 404:
                        error_msg = f"Resource not found: {endpoint}"
                    elif response.status_code >= 500:
                        error_msg = f"NCBI server error: {response.status_code}"

                    # Retry on server errors and rate limits
                    if (response.status_code >= 500 or response.status_code == 429) and retries < max_retries:
                        retries += 1
                        wait_time = backoff_factor ** retries
                        logger.warning(f"Retrying in {wait_time:.1f}s ({retries}/{max_retries})")
                        await asyncio.sleep(wait_time)
                        continue

                    # If we've exhausted retries or it's a client error, raise an exception
                    raise NCBIClientError(error_msg, response.status_code)

                # Parse response
                result = response.json() if return_json else response.text

                # Cache the result if caching is enabled
                if self.use_cache and cache_key:
                    await cache_manager.set(cache_key, result, ttl=self.cache_ttl, data_type="search")

                return result

            except httpx.RequestError as e:
                # Handle network errors
                error_msg = f"Network error: {str(e)}"
                logger.warning(f"{error_msg} for {url}")

                # Retry if we haven't exhausted retries
                if retries < max_retries:
                    retries += 1
                    wait_time = backoff_factor ** retries
                    logger.warning(f"Retrying in {wait_time:.1f}s ({retries}/{max_retries})")
                    await asyncio.sleep(wait_time)
                else:
                    # If we've exhausted retries, raise an exception
                    raise NCBIClientError(f"Failed to connect to NCBI API after {max_retries} retries: {str(e)}")
            except json.JSONDecodeError as e:
                # Handle JSON parsing errors
                error_msg = f"Invalid JSON response: {str(e)}"
                logger.error(f"{error_msg} for {url}")
                raise NCBIClientError(error_msg)
            except Exception as e:
                # Handle other errors
                error_msg = f"Unexpected error: {str(e)}"
                logger.error(f"{error_msg} for {url}")
                raise NCBIClientError(error_msg)

    @cached(prefix="pubmed_search", data_type="search")
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

        Raises:
            ValidationError: If the query is invalid
            NCBIClientError: If the search fails
        """
        # Validate inputs
        if not query or not query.strip():
            raise ValidationError("Search query cannot be empty")

        if max_results < 1:
            raise ValidationError("max_results must be at least 1")

        # Prepare search parameters
        params = {
            "db": "pubmed",
            "term": query.strip(),
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
            # Make the request
            logger.info(f"Searching PubMed for '{query}' (max_results={max_results})")
            data = await self._make_request("esearch.fcgi", params)

            # Log the results
            count = int(data.get("esearchresult", {}).get("count", "0"))
            logger.info(f"Found {count} results for '{query}'")

            return data
        except NCBIClientError as e:
            # Re-raise with more context
            logger.error(f"Error searching PubMed for '{query}': {str(e)}")
            raise NCBIClientError(f"Failed to search PubMed: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error searching PubMed for '{query}': {str(e)}")
            raise NCBIClientError(f"Unexpected error searching PubMed: {str(e)}")

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

        Raises:
            ValidationError: If the PMIDs are invalid
            NCBIClientError: If the fetch fails
        """
        # Validate inputs
        if isinstance(pmids, str):
            pmids = [pmids]

        if not pmids:
            return []

        # Check for invalid PMIDs
        invalid_pmids = [pmid for pmid in pmids if not pmid or not pmid.strip()]
        if invalid_pmids:
            raise ValidationError(f"Invalid PMIDs: {invalid_pmids}")

        # Split PMIDs into batches
        batches = await self._batch_pmids(pmids, batch_size)
        all_articles = []
        errors = []

        # Process each batch
        for batch_index, batch in enumerate(batches):
            # Check if we have cached results for this batch
            batch_key = f"article_details:{db}:{','.join(batch)}"
            if self.use_cache:
                cached_batch = await cache_manager.get(batch_key)
                if cached_batch is not None:
                    logger.debug(f"Cache hit for article details batch {batch_index+1}/{len(batches)}")
                    all_articles.extend(cached_batch)
                    continue

            params = {
                "db": db,
                "id": ",".join(batch)
            }

            try:
                logger.info(f"Fetching details for {len(batch)} articles (batch {batch_index+1}/{len(batches)})")
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
                            "source": "PubMed",
                            "fetched_at": hashlib.md5(json.dumps(article_data, sort_keys=True).encode()).hexdigest()[:8]
                        }

                        batch_articles.append(article)
                    else:
                        logger.warning(f"PMID {pmid} not found in response")
                        errors.append(f"PMID {pmid} not found")

                # Cache the batch results
                if self.use_cache and batch_articles:
                    await cache_manager.set(batch_key, batch_articles, ttl=self.cache_ttl, data_type="search")

                all_articles.extend(batch_articles)
                logger.info(f"Fetched details for {len(batch_articles)} articles in batch {batch_index+1}/{len(batches)}")
            except NCBIClientError as e:
                error_msg = f"Error fetching article details (batch {batch_index+1}/{len(batches)}): {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
                # Continue with the next batch instead of failing completely
                continue
            except Exception as e:
                error_msg = f"Unexpected error fetching article details (batch {batch_index+1}/{len(batches)}): {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
                # Continue with the next batch instead of failing completely
                continue

        # Log summary
        logger.info(f"Fetched details for {len(all_articles)}/{len(pmids)} articles with {len(errors)} errors")

        # If we have errors but also some results, just log the errors
        if errors and all_articles:
            for error in errors:
                logger.warning(f"Partial failure in fetch_article_details: {error}")
        # If we have errors and no results, raise an exception
        elif errors and not all_articles:
            raise NCBIClientError(f"Failed to fetch any article details: {'; '.join(errors[:3])}{'...' if len(errors) > 3 else ''}")

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

        Raises:
            ValidationError: If the PMIDs are invalid
            NCBIClientError: If the fetch fails
        """
        # Validate inputs
        if isinstance(pmids, str):
            pmids = [pmids]

        if not pmids:
            return {}

        # Check for invalid PMIDs
        invalid_pmids = [pmid for pmid in pmids if not pmid or not pmid.strip()]
        if invalid_pmids:
            raise ValidationError(f"Invalid PMIDs: {invalid_pmids}")

        # Split PMIDs into batches
        batches = await self._batch_pmids(pmids, batch_size)
        all_abstracts = {}
        errors = []

        # Process each batch
        for batch_index, batch in enumerate(batches):
            # Check if we have cached results for this batch
            batch_key = f"article_abstracts:{db}:{','.join(batch)}"
            if self.use_cache:
                cached_batch = await cache_manager.get(batch_key)
                if cached_batch is not None:
                    logger.debug(f"Cache hit for article abstracts batch {batch_index+1}/{len(batches)}")
                    all_abstracts.update(cached_batch)
                    continue

            params = {
                "db": db,
                "id": ",".join(batch),
                "retmode": "xml"  # XML format is required for abstracts
            }

            try:
                # For abstracts, we need to use efetch and parse XML
                logger.info(f"Fetching abstracts for {len(batch)} articles (batch {batch_index+1}/{len(batches)})")
                xml_text = await self._make_request("efetch.fcgi", params, return_json=False)
                batch_abstracts = {}

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
                        batch_abstracts[pmid] = abstract_text
                except ET.ParseError as e:
                    logger.error(f"Error parsing XML response: {str(e)}")
                    logger.warning("Falling back to simple string parsing")

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
                        batch_abstracts[pmid] = abstract_text

                # Cache the batch results
                if self.use_cache and batch_abstracts:
                    await cache_manager.set(batch_key, batch_abstracts, ttl=self.cache_ttl, data_type="search")

                # Update all abstracts
                all_abstracts.update(batch_abstracts)
                logger.info(f"Fetched abstracts for {len(batch_abstracts)}/{len(batch)} articles in batch {batch_index+1}/{len(batches)}")

                # Log missing abstracts
                missing_abstracts = [pmid for pmid in batch if pmid not in batch_abstracts]
                if missing_abstracts:
                    logger.warning(f"Missing abstracts for {len(missing_abstracts)}/{len(batch)} articles in batch {batch_index+1}/{len(batches)}")
                    for pmid in missing_abstracts[:5]:  # Log only the first 5 to avoid flooding the logs
                        logger.debug(f"Missing abstract for PMID {pmid}")
                    if len(missing_abstracts) > 5:
                        logger.debug(f"... and {len(missing_abstracts) - 5} more")
            except NCBIClientError as e:
                error_msg = f"Error fetching article abstracts (batch {batch_index+1}/{len(batches)}): {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
                # Continue with the next batch instead of failing completely
                continue
            except Exception as e:
                error_msg = f"Unexpected error fetching article abstracts (batch {batch_index+1}/{len(batches)}): {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
                # Continue with the next batch instead of failing completely
                continue

        # Log summary
        logger.info(f"Fetched abstracts for {len(all_abstracts)}/{len(pmids)} articles with {len(errors)} errors")

        # If we have errors but also some results, just log the errors
        if errors and all_abstracts:
            for error in errors:
                logger.warning(f"Partial failure in fetch_article_abstracts: {error}")
        # If we have errors and no results, raise an exception
        elif errors and not all_abstracts:
            raise NCBIClientError(f"Failed to fetch any article abstracts: {'; '.join(errors[:3])}{'...' if len(errors) > 3 else ''}")

        return all_abstracts

    @cached(prefix="pubmed_abstracts", data_type="search")
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

        Raises:
            ValidationError: If both id_list and query are None, or if inputs are invalid
            NCBIClientError: If the fetch fails
        """
        # Validate inputs
        if id_list is None and query is None:
            raise ValidationError("Either id_list or query must be provided")

        if max_results < 1:
            raise ValidationError("max_results must be at least 1")

        # Get PMIDs from query if not provided
        if id_list is None and query is not None:
            try:
                logger.info(f"Searching PubMed for '{query}' to get PMIDs")
                search_result = await self.search_pubmed(query, max_results=max_results)
                id_list = search_result.get("esearchresult", {}).get("idlist", [])
                logger.info(f"Found {len(id_list)} PMIDs for query '{query}'")
            except Exception as e:
                logger.error(f"Error searching PubMed for '{query}': {str(e)}")
                raise NCBIClientError(f"Failed to search PubMed: {str(e)}")

        if not id_list:
            logger.warning(f"No PMIDs found for query '{query}'")
            return []

        try:
            # Fetch article details and abstracts in parallel
            logger.info(f"Fetching details and abstracts for {len(id_list)} articles")
            details_task = asyncio.create_task(self.fetch_article_details(id_list, batch_size=batch_size))
            abstracts_task = asyncio.create_task(self.fetch_article_abstracts(id_list, batch_size=batch_size))

            # Wait for both tasks to complete
            articles, abstracts = await asyncio.gather(details_task, abstracts_task)

            # Merge abstracts into articles
            for article in articles:
                pmid = article["pmid"]
                if pmid in abstracts:
                    article["abstract"] = abstracts[pmid]
                else:
                    logger.debug(f"No abstract found for PMID {pmid}")

            logger.info(f"Successfully fetched {len(articles)} articles with {len(abstracts)} abstracts")
            return articles
        except NCBIClientError as e:
            # Re-raise with more context
            logger.error(f"Error fetching PubMed abstracts: {str(e)}")
            raise NCBIClientError(f"Failed to fetch PubMed abstracts: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error fetching PubMed abstracts: {str(e)}")
            raise NCBIClientError(f"Unexpected error fetching PubMed abstracts: {str(e)}")

    @cached(prefix="pubmed_search_fetch", data_type="search")
    async def search_and_fetch_pubmed(
        self,
        query: str,
        max_results: int = 20,
        batch_size: Optional[int] = None,
        sort: str = "relevance",
        min_date: Optional[str] = None,
        max_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search PubMed and fetch abstracts in one step.

        This is a convenience method that combines search_pubmed and fetch_pubmed_abstracts.

        Args:
            query: Search query
            max_results: Maximum number of results to return (default: 20)
            batch_size: Batch size for requests (default: self.max_batch_size)
            sort: Sort order (default: "relevance")
            min_date: Minimum date (YYYY/MM/DD format)
            max_date: Maximum date (YYYY/MM/DD format)

        Returns:
            List of articles with abstracts

        Raises:
            ValidationError: If the query is invalid
            NCBIClientError: If the search or fetch fails
        """
        # Validate inputs
        if not query or not query.strip():
            raise ValidationError("Search query cannot be empty")

        if max_results < 1:
            raise ValidationError("max_results must be at least 1")

        try:
            logger.info(f"Searching and fetching PubMed articles for '{query}' (max_results={max_results})")

            # First search PubMed to get PMIDs
            search_result = await self.search_pubmed(
                query=query,
                max_results=max_results,
                sort=sort,
                min_date=min_date,
                max_date=max_date
            )

            id_list = search_result.get("esearchresult", {}).get("idlist", [])
            if not id_list:
                logger.warning(f"No results found for query '{query}'")
                return []

            # Then fetch the articles with abstracts
            articles = await self.fetch_pubmed_abstracts(
                id_list=id_list,
                batch_size=batch_size
            )

            logger.info(f"Successfully searched and fetched {len(articles)} articles for query '{query}'")
            return articles
        except ValidationError as e:
            # Re-raise validation errors
            raise
        except NCBIClientError as e:
            # Re-raise with more context
            logger.error(f"Error in search_and_fetch_pubmed for '{query}': {str(e)}")
            raise NCBIClientError(f"Failed to search and fetch PubMed articles: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error in search_and_fetch_pubmed for '{query}': {str(e)}")
            raise NCBIClientError(f"Unexpected error searching and fetching PubMed articles: {str(e)}")

    async def clear_cache(self, pattern: Optional[str] = None) -> int:
        """
        Clear the cache for this client.

        Args:
            pattern: Optional pattern to match cache keys (e.g., "pubmed_search:*")
                     If None, clears all NCBI-related cache entries

        Returns:
            Number of cache entries cleared
        """
        if not self.use_cache:
            logger.warning("Cache is disabled for this client")
            return 0

        try:
            # If no pattern is provided, clear all NCBI-related cache entries
            if pattern is None:
                patterns = [
                    "ncbi:*",              # Direct API requests
                    "pubmed_search:*",      # Search results
                    "pubmed_abstracts:*",   # Article abstracts
                    "pubmed_search_fetch:*", # Combined search and fetch
                    "article_details:*",     # Article details
                    "article_abstracts:*"    # Article abstracts
                ]

                total_cleared = 0
                for p in patterns:
                    cleared = await cache_manager.delete_pattern(p)
                    logger.info(f"Cleared {cleared} cache entries matching pattern '{p}'")
                    total_cleared += cleared

                logger.info(f"Cleared {total_cleared} NCBI-related cache entries in total")
                return total_cleared
            else:
                # Clear cache entries matching the specified pattern
                cleared = await cache_manager.delete_pattern(pattern)
                logger.info(f"Cleared {cleared} cache entries matching pattern '{pattern}'")
                return cleared
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            return 0

    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics for this client.

        Returns:
            Dictionary with cache statistics
        """
        if not self.use_cache:
            logger.warning("Cache is disabled for this client")
            return {"enabled": False}

        try:
            # Get general cache stats
            stats = await cache_manager.get_stats()

            # Add NCBI-specific stats
            patterns = [
                "ncbi:*",              # Direct API requests
                "pubmed_search:*",      # Search results
                "pubmed_abstracts:*",   # Article abstracts
                "pubmed_search_fetch:*", # Combined search and fetch
                "article_details:*",     # Article details
                "article_abstracts:*"    # Article abstracts
            ]

            pattern_counts = {}
            for pattern in patterns:
                count = await cache_manager.count_pattern(pattern)
                pattern_counts[pattern] = count

            stats["ncbi_patterns"] = pattern_counts
            stats["ncbi_total"] = sum(pattern_counts.values())
            stats["enabled"] = True
            stats["ttl"] = self.cache_ttl

            return stats
        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            return {"enabled": self.use_cache, "error": str(e)}
