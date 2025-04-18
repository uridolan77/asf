"""NCBI client for the Medical Research Synthesizer.

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
from ...core.config import settings
from ...core.enhanced_cache import enhanced_cache_manager, enhanced_cached
from ...core.exceptions import ExternalServiceError, ValidationError
logger = logging.getLogger(__name__)
class NCBIClientError(ExternalServiceError):
    """Exception raised for NCBI client errors."""
    def __init__(self, message: str = "NCBI API error", status_code: Optional[int] = None):
        self.status_code = status_code
        super().__init__("NCBI API", message)
class NCBIClient:
    """Client for interacting with the NCBI E-utilities API.

    This client provides methods for searching PubMed and retrieving article details.
    It includes retry logic, rate limiting, and caching to improve reliability and performance.
    """

    def __init__(self, email: Optional[str] = None, api_key: Optional[str] = None,
                 tool: str = "MedicalResearchSynthesizer",
                 base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/",
                 max_batch_size: int = 200, timeout: float = 30.0,
                 max_retries: int = 3, cache_ttl: int = 3600,
                 use_cache: bool = True):
        """Initialize the NCBI client.

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
        # Implementation goes here

    async def aclose(self):
        """Close the client session."""
        await self.client.aclose()
    async def check_api_status(self) -> Dict[str, Any]:
        """Check the status of the NCBI API.

        Returns:
            Dictionary with API status information
        """
        try:
            params = {
                "db": "pubmed",
                "term": "test",
                "retmax": 1
            }
            start_time = time.time()
            data = await self._make_request("esearch.fcgi", params)
            response_time = time.time() - start_time
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
            old_rate = self.rate_limiter.requests_per_second
            self.rate_limiter.requests_per_second = requests_per_second
            logger.info(f"Updated rate limit from {old_rate} to {requests_per_second} requests per second")
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
        return {
            "use_cache": self.use_cache,
            "cache_ttl": self.cache_ttl
        }
    async def count_cached_items(self, pattern: Optional[str] = None) -> Dict[str, Any]:
        if not self.use_cache:
            logger.warning("Cache is disabled for this client")
            return {"enabled": False, "total": 0}
        try:
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
                    count = await enhanced_cache_manager.count_pattern(p)
                    counts[p] = count
                    total += count
                return {
                    "enabled": True,
                    "total": total,
                    "patterns": counts,
                    "ttl": self.cache_ttl
                }
            else:
                count = await enhanced_cache_manager.count_pattern(pattern)
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
    @enhanced_cached(key_prefix="ncbi_einfo")
    async def get_database_info(self, db: Optional[str] = None, version: str = "2.0") -> Dict[str, Any]:
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
    @enhanced_cached(key_prefix="ncbi_espell")
    async def get_spelling_suggestions(self, term: str, db: str = "pubmed") -> Dict[str, Any]:
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
    @enhanced_cached(key_prefix="ncbi_egquery")
    async def search_all_databases(self, term: str) -> Dict[str, Any]:
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
        if not citations:
            raise ValidationError("Citations list cannot be empty")
        citation_strings = []
        for citation in citations:
            required_fields = ["journal", "year", "volume", "first_page", "author"]
            missing_fields = [field for field in required_fields if field not in citation or not citation[field]]
            if missing_fields:
                raise ValidationError(f"Citation missing required fields: {', '.join(missing_fields)}")
            key = citation.get("key", f"cit_{len(citation_strings)+1}")
            citation_str = f"{citation['journal']}|{citation['year']}|{citation['volume']}|{citation['first_page']}|{citation['author']}|{key}|"
            citation_strings.append(citation_str)
        bdata = "%0D".join(citation_strings)
        params = {
            "db": "pubmed",
            "retmode": "xml",
            "bdata": bdata
        }
        try:
            logger.info(f"Matching {len(citations)} citations to PMIDs")
            result = await self._make_request("ecitmatch.cgi", params, return_json=False)
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
    @enhanced_cached(key_prefix="ncbi_elink")
    async def get_links(self,
                       ids: Union[str, List[str]],
                       dbfrom: str,
                       db: str,
                       linkname: Optional[str] = None,
                       cmd: str = "neighbor") -> Dict[str, Any]:
        if not ids:
            raise ValidationError("IDs list cannot be empty")
        if not dbfrom or not db:
            raise ValidationError("Source and target databases must be specified")
        if isinstance(ids, str):
            id_list = [ids]
        else:
            id_list = ids
        params = {
            "dbfrom": dbfrom,
            "db": db,
            "id": ",".join(id_list),
            "cmd": cmd
        }
        if linkname:
            params["linkname"] = linkname
        try:
            logger.info(f"Getting links from {dbfrom} to {db} for {len(id_list)} IDs")
            result = await self._make_request("elink.fcgi", params)
            return result
        except Exception as e:
            logger.error(f"Error getting links: {str(e)}")
            raise NCBIClientError(f"Failed to get links: {str(e)}")
    @enhanced_cached(key_prefix="ncbi_elink_history")
    async def get_links_and_post_to_history(self,
                                           ids: Union[str, List[str]],
                                           dbfrom: str,
                                           db: str,
                                           linkname: Optional[str] = None) -> Dict[str, Any]:
        return await self.get_links(ids, dbfrom, db, linkname, cmd="neighbor_history")
    @enhanced_cached(key_prefix="ncbi_elink_scores")
    async def get_links_with_scores(self,
                                   ids: Union[str, List[str]],
                                   dbfrom: str,
                                   db: str,
                                   linkname: Optional[str] = None) -> Dict[str, Any]:
        return await self.get_links(ids, dbfrom, db, linkname, cmd="neighbor_score")
    @enhanced_cached(key_prefix="ncbi_elink_check")
    async def check_links(self,
                         ids: Union[str, List[str]],
                         dbfrom: str,
                         db: Optional[str] = None) -> Dict[str, Any]:
        if not ids:
            raise ValidationError("IDs list cannot be empty")
        if not dbfrom:
            raise ValidationError("Source database must be specified")
        if isinstance(ids, str):
            id_list = [ids]
        else:
            id_list = ids
        params = {
            "dbfrom": dbfrom,
            "id": ",".join(id_list),
            "cmd": "acheck"
        }
        if db:
            params["db"] = db
        try:
            logger.info(f"Checking links from {dbfrom} for {len(id_list)} IDs")
            result = await self._make_request("elink.fcgi", params)
            return result
        except Exception as e:
            logger.error(f"Error checking links: {str(e)}")
            raise NCBIClientError(f"Failed to check links: {str(e)}")
    @enhanced_cached(key_prefix="ncbi_linkout")
    async def get_linkout_urls(self,
                              ids: Union[str, List[str]],
                              dbfrom: str,
                              include_libraries: bool = False) -> Dict[str, Any]:
        if not ids:
            raise ValidationError("IDs list cannot be empty")
        if not dbfrom:
            raise ValidationError("Source database must be specified")
        if isinstance(ids, str):
            id_list = [ids]
        else:
            id_list = ids
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
        if not pmid or not pmid.strip():
            raise ValidationError("PMID cannot be empty")
        params = {
            "dbfrom": "pubmed",
            "id": pmid.strip(),
            "cmd": "prlinks",
            "retmode": "ref"
        }
        try:
            url = f"{self.base_url}elink.fcgi"
            params.update({
                "tool": self.tool,
                "email": self.email,
            })
            if self.api_key:
                params["api_key"] = self.api_key
            await self.rate_limiter.acquire()
            async with httpx.AsyncClient(follow_redirects=False) as client:
                response = await client.get(url, params=params)
                if response.status_code in (301, 302, 303, 307, 308):
                    redirect_url = response.headers.get("location")
                    if redirect_url:
                        logger.info(f"Got full-text URL for PMID {pmid}: {redirect_url}")
                        return redirect_url
                if response.status_code >= 400:
                    error_msg = f"NCBI API error: {response.status_code} {response.reason_phrase}"
                    logger.warning(f"{error_msg} for {url}")
                    raise NCBIClientError(error_msg, response.status_code)
                logger.warning(f"No redirect found for PMID {pmid}, returning original URL")
                return str(response.url)
        except httpx.RequestError as e:
            error_msg = f"Network error: {str(e)}"
            logger.error(f"{error_msg} for PMID {pmid}")
            raise NCBIClientError(f"Failed to get full-text URL: {str(e)}")
        except Exception as e:
            logger.error(f"Error getting full-text URL for PMID {pmid}: {str(e)}")
            raise NCBIClientError(f"Failed to get full-text URL: {str(e)}")
    @enhanced_cached(key_prefix="pubmed_related")
    async def get_related_articles(self, pmid: str, max_results: int = 20) -> List[Dict[str, Any]]:
        if not pmid or not pmid.strip():
            raise ValidationError("PMID cannot be empty")
        if max_results < 1:
            raise ValidationError("max_results must be at least 1")
        try:
            logger.info(f"Getting related articles for PMID {pmid}")
            links_result = await self.get_links_with_scores(pmid, "pubmed", "pubmed")
            related_pmids = []
            try:
                linksets = links_result.get("linksets", [])
                if linksets and "linksetdbs" in linksets[0]:
                    for linksetdb in linksets[0]["linksetdbs"]:
                        if linksetdb.get("linkname") == "pubmed_pubmed":
                            links = linksetdb.get("links", [])
                            if links and "score" in links[0]:
                                links.sort(key=lambda x: int(x.get("score", 0)), reverse=True)
                            related_pmids = [link.get("id") for link in links[:max_results] if "id" in link]
            except Exception as e:
                logger.error(f"Error parsing related articles: {str(e)}")
            if not related_pmids:
                logger.warning(f"No related articles found for PMID {pmid}")
                return []
            logger.info(f"Fetching details for {len(related_pmids)} related articles")
            articles = await self.fetch_pubmed_abstracts(id_list=related_pmids)
            return articles
        except NCBIClientError as e:
            logger.error(f"Error getting related articles for PMID {pmid}: {str(e)}")
            raise NCBIClientError(f"Failed to get related articles: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
            logger.error(f"Unexpected error getting related articles for PMID {pmid}: {str(e)}")
            raise NCBIClientError(f"Unexpected error getting related articles: {str(e)}")
    @enhanced_cached(key_prefix="pubmed_cited_by")
    async def get_citing_articles(self, pmid: str, max_results: int = 20) -> List[Dict[str, Any]]:
        if not pmid or not pmid.strip():
            raise ValidationError("PMID cannot be empty")
        if max_results < 1:
            raise ValidationError("max_results must be at least 1")
        try:
            logger.info(f"Getting citing articles for PMID {pmid}")
            links_result = await self.get_links(pmid, "pubmed", "pubmed", linkname="pubmed_pubmed_citedin")
            citing_pmids = []
            try:
                linksets = links_result.get("linksets", [])
                if linksets and "linksetdbs" in linksets[0]:
                    for linksetdb in linksets[0]["linksetdbs"]:
                        if linksetdb.get("linkname") == "pubmed_pubmed_citedin":
                            links = linksetdb.get("links", [])
                            citing_pmids = [link.get("id") for link in links[:max_results] if "id" in link]
            except Exception as e:
                logger.error(f"Error parsing citing articles: {str(e)}")
            if not citing_pmids:
                logger.warning(f"No citing articles found for PMID {pmid}")
                return []
            logger.info(f"Fetching details for {len(citing_pmids)} citing articles")
            articles = await self.fetch_pubmed_abstracts(id_list=citing_pmids)
            return articles
        except NCBIClientError as e:
            logger.error(f"Error getting citing articles for PMID {pmid}: {str(e)}")
            raise NCBIClientError(f"Failed to get citing articles: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
            logger.error(f"Unexpected error getting citing articles for PMID {pmid}: {str(e)}")
            raise NCBIClientError(f"Unexpected error getting citing articles: {str(e)}")
    @enhanced_cached(key_prefix="pubmed_mesh")
    async def get_mesh_terms(self, pmid: str) -> List[Dict[str, str]]:
        if not pmid or not pmid.strip():
            raise ValidationError("PMID cannot be empty")
        try:
            logger.info(f"Getting MeSH terms for PMID {pmid}")
            params = {
                "db": "pubmed",
                "id": pmid.strip(),
                "retmode": "xml"
            }
            xml_text = await self._make_request("efetch.fcgi", params, return_json=False)
            mesh_terms = []
            try:
                root = ET.fromstring(xml_text)
                for mesh_heading in root.findall(".//MeshHeading"):
                    descriptor = mesh_heading.find("DescriptorName")
                    if descriptor is None:
                        continue
                    descriptor_name = descriptor.text
                    descriptor_ui = descriptor.get("UI", "")
                    is_major = descriptor.get("MajorTopicYN", "N") == "Y"
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
            logger.error(f"Error getting MeSH terms for PMID {pmid}: {str(e)}")
            raise NCBIClientError(f"Failed to get MeSH terms: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
            logger.error(f"Unexpected error getting MeSH terms for PMID {pmid}: {str(e)}")
            raise NCBIClientError(f"Unexpected error getting MeSH terms: {str(e)}")
    @enhanced_cached(key_prefix="pubmed_journal_info")
    async def get_journal_info(self, journal: str) -> Dict[str, Any]:
        if not journal or not journal.strip():
            raise ValidationError("Journal name cannot be empty")
        try:
            logger.info(f"Searching for journal '{journal}' in NLM Catalog")
            search_params = {
                "db": "nlmcatalog",
                "term": f"{journal.strip()}[Title] AND ncbijournals[Filter]",
                "retmax": 1
            }
            search_result = await self._make_request("esearch.fcgi", search_params)
            journal_ids = search_result.get("esearchresult", {}).get("idlist", [])
            if not journal_ids:
                logger.warning(f"No journal found for '{journal}'")
                return {}
            logger.info(f"Getting details for journal ID {journal_ids[0]}")
            summary_params = {
                "db": "nlmcatalog",
                "id": journal_ids[0],
                "version": "2.0"
            }
            journal_info = await self._make_request("esummary.fcgi", summary_params)
            return journal_info
        except NCBIClientError as e:
            logger.error(f"Error getting journal info for '{journal}': {str(e)}")
            raise NCBIClientError(f"Failed to get journal info: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
            logger.error(f"Unexpected error getting journal info for '{journal}': {str(e)}")
            raise NCBIClientError(f"Unexpected error getting journal info: {str(e)}")
    @enhanced_cached(key_prefix="sequence_search")
    async def search_sequence_database(self,
                                     query: str,
                                     db: str = "nucleotide",
                                     max_results: int = 20,
                                     return_type: str = "gb",
                                     return_mode: str = "text") -> Dict[str, Any]:
        if not query or not query.strip():
            raise ValidationError("Search query cannot be empty")
        if max_results < 1:
            raise ValidationError("max_results must be at least 1")
        valid_dbs = ["nucleotide", "protein", "genome", "gene"]
        if db not in valid_dbs:
            raise ValidationError(f"Invalid database: {db}. Must be one of {valid_dbs}")
        try:
            logger.info(f"Searching {db} database for '{query}'")
            search_params = {
                "db": db,
                "term": query.strip(),
                "retmax": max_results,
                "usehistory": "y"
            }
            search_result = await self._make_request("esearch.fcgi", search_params)
            id_list = search_result.get("esearchresult", {}).get("idlist", [])
            web_env = search_result.get("esearchresult", {}).get("webenv")
            query_key = search_result.get("esearchresult", {}).get("querykey")
            count = int(search_result.get("esearchresult", {}).get("count", "0"))
            if not id_list:
                logger.warning(f"No results found for '{query}' in {db} database")
                return {"count": 0, "ids": [], "sequences": ""}
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
            logger.error(f"Error searching {db} database for '{query}': {str(e)}")
            raise NCBIClientError(f"Failed to search {db} database: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
            logger.error(f"Unexpected error searching {db} database for '{query}': {str(e)}")
            raise NCBIClientError(f"Unexpected error searching {db} database: {str(e)}")
    @enhanced_cached(key_prefix="sequence_fetch")
    async def fetch_sequence(self,
                           id: str,
                           db: str = "nucleotide",
                           return_type: str = "gb",
                           return_mode: str = "text",
                           strand: Optional[int] = None,
                           seq_start: Optional[int] = None,
                           seq_stop: Optional[int] = None) -> str:
        if not id or not id.strip():
            raise ValidationError("Sequence ID cannot be empty")
        valid_dbs = ["nucleotide", "protein", "genome", "gene"]
        if db not in valid_dbs:
            raise ValidationError(f"Invalid database: {db}. Must be one of {valid_dbs}")
        if strand is not None and strand not in (1, 2):
            raise ValidationError("Strand must be 1 (plus) or 2 (minus)")
        if (seq_start is not None and seq_stop is None) or (seq_start is None and seq_stop is not None):
            raise ValidationError("Both seq_start and seq_stop must be provided together")
        if seq_start is not None and seq_stop is not None:
            if seq_start < 1 or seq_stop < seq_start:
                raise ValidationError("Invalid sequence range: seq_start must be >= 1 and seq_stop must be >= seq_start")
        try:
            params = {
                "db": db,
                "id": id.strip(),
                "rettype": return_type,
                "retmode": return_mode
            }
            if strand is not None:
                params["strand"] = strand
            if seq_start is not None and seq_stop is not None:
                params["seq_start"] = seq_start
                params["seq_stop"] = seq_stop
            logger.info(f"Fetching sequence {id} from {db} database")
            sequence = await self._make_request("efetch.fcgi", params, return_json=False)
            return sequence
        except NCBIClientError as e:
            logger.error(f"Error fetching sequence {id} from {db} database: {str(e)}")
            raise NCBIClientError(f"Failed to fetch sequence: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
            logger.error(f"Unexpected error fetching sequence {id} from {db} database: {str(e)}")
            raise NCBIClientError(f"Unexpected error fetching sequence: {str(e)}")
    @enhanced_cached(key_prefix="taxonomy_fetch")
    async def get_taxonomy(self, id: Union[str, int]) -> Dict[str, Any]:
        if not id:
            raise ValidationError("Taxonomy ID or name cannot be empty")
        try:
            if isinstance(id, int) or (isinstance(id, str) and id.isdigit()):
                logger.info(f"Getting taxonomy information for taxon ID {id}")
                params = {
                    "db": "taxonomy",
                    "id": str(id),
                    "version": "2.0"
                }
                taxonomy = await self._make_request("esummary.fcgi", params)
                return taxonomy
            else:
                logger.info(f"Searching taxonomy database for '{id}'")
                search_params = {
                    "db": "taxonomy",
                    "term": str(id).strip(),
                    "retmax": 1
                }
                search_result = await self._make_request("esearch.fcgi", search_params)
                taxon_ids = search_result.get("esearchresult", {}).get("idlist", [])
                if not taxon_ids:
                    logger.warning(f"No taxonomy found for '{id}'")
                    return {}
                logger.info(f"Getting taxonomy information for taxon ID {taxon_ids[0]}")
                summary_params = {
                    "db": "taxonomy",
                    "id": taxon_ids[0],
                    "version": "2.0"
                }
                taxonomy = await self._make_request("esummary.fcgi", summary_params)
                return taxonomy
        except NCBIClientError as e:
            logger.error(f"Error getting taxonomy for '{id}': {str(e)}")
            raise NCBIClientError(f"Failed to get taxonomy: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
            logger.error(f"Unexpected error getting taxonomy for '{id}': {str(e)}")
            raise NCBIClientError(f"Unexpected error getting taxonomy: {str(e)}")
    async def create_history_session(self) -> Dict[str, str]:
        try:
            logger.info("Creating new History server session")
            params = {
                "db": "pubmed",
                "term": "1:1[uid]",  # Search for PMID 1, which is guaranteed to exist
                "usehistory": "y"
            }
            result = await self._make_request("esearch.fcgi", params)
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
            logger.error(f"Error creating History server session: {str(e)}")
            raise NCBIClientError(f"Failed to create History server session: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
            logger.error(f"Unexpected error creating History server session: {str(e)}")
            raise NCBIClientError(f"Unexpected error creating History server session: {str(e)}")
    async def post_ids_to_history(self, ids: Union[str, List[str]], db: str, web_env: Optional[str] = None) -> Dict[str, str]:
        if not ids:
            raise ValidationError("IDs list cannot be empty")
        if not db:
            raise ValidationError("Database must be specified")
        if isinstance(ids, str):
            id_list = [ids]
        else:
            id_list = ids
        try:
            params = {
                "db": db,
                "id": ",".join(id_list)
            }
            if web_env:
                params["WebEnv"] = web_env
            logger.info(f"Posting {len(id_list)} IDs to History server")
            result = await self._make_request("epost.fcgi", params)
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
            logger.error(f"Error posting IDs to History server: {str(e)}")
            raise NCBIClientError(f"Failed to post IDs to History server: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
            logger.error(f"Unexpected error posting IDs to History server: {str(e)}")
            raise NCBIClientError(f"Unexpected error posting IDs to History server: {str(e)}")
    async def search_and_post_to_history(self,
                                        query: str,
                                        db: str = "pubmed",
                                        web_env: Optional[str] = None,
                                        query_key: Optional[str] = None,
                                        **search_params) -> Dict[str, Any]:
        if not query or not query.strip():
            raise ValidationError("Search query cannot be empty")
        if not db:
            raise ValidationError("Database must be specified")
        try:
            params = {
                "db": db,
                "term": query.strip(),
                "usehistory": "y"
            }
            if web_env:
                params["WebEnv"] = web_env
            if query_key:
                params["query_key"] = query_key
            params.update(search_params)
            logger.info(f"Searching {db} for '{query}' and posting to History server")
            result = await self._make_request("esearch.fcgi", params)
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
            logger.error(f"Error searching and posting to History server: {str(e)}")
            raise NCBIClientError(f"Failed to search and post to History server: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
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
        if not web_env or not query_key:
            raise ValidationError("WebEnv and query_key must be provided")
        if not db:
            raise ValidationError("Database must be specified")
        if retstart < 0:
            raise ValidationError("retstart must be non-negative")
        if retmax < 1:
            raise ValidationError("retmax must be at least 1")
        try:
            params = {
                "db": db,
                "WebEnv": web_env,
                "query_key": query_key,
                "retstart": retstart,
                "retmax": retmax
            }
            if rettype:
                params["rettype"] = rettype
            if retmode:
                params["retmode"] = retmode
            logger.info(f"Fetching {retmax} records from {db} (starting at {retstart}) using History server")
            result = await self._make_request("efetch.fcgi", params, return_json=(retmode != "xml" and retmode != "text"))
            logger.info(f"Successfully fetched records from History server")
            return result
        except NCBIClientError as e:
            logger.error(f"Error fetching from History server: {str(e)}")
            raise NCBIClientError(f"Failed to fetch from History server: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
            logger.error(f"Unexpected error fetching from History server: {str(e)}")
            raise NCBIClientError(f"Unexpected error fetching from History server: {str(e)}")
    async def get_summary_from_history(self,
                                     web_env: str,
                                     query_key: str,
                                     db: str,
                                     retstart: int = 0,
                                     retmax: int = 20,
                                     version: str = "2.0") -> Dict[str, Any]:
        if not web_env or not query_key:
            raise ValidationError("WebEnv and query_key must be provided")
        if not db:
            raise ValidationError("Database must be specified")
        if retstart < 0:
            raise ValidationError("retstart must be non-negative")
        if retmax < 1:
            raise ValidationError("retmax must be at least 1")
        try:
            params = {
                "db": db,
                "WebEnv": web_env,
                "query_key": query_key,
                "retstart": retstart,
                "retmax": retmax,
                "version": version
            }
            logger.info(f"Getting summaries for {retmax} records from {db} (starting at {retstart}) using History server")
            result = await self._make_request("esummary.fcgi", params)
            logger.info(f"Successfully got summaries from History server")
            return result
        except NCBIClientError as e:
            logger.error(f"Error getting summaries from History server: {str(e)}")
            raise NCBIClientError(f"Failed to get summaries from History server: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
            logger.error(f"Unexpected error getting summaries from History server: {str(e)}")
            raise NCBIClientError(f"Unexpected error getting summaries from History server: {str(e)}")
    async def batch_fetch_articles(self,
                                 pmids: List[str],
                                 batch_size: int = 200,
                                 include_abstracts: bool = True,
                                 max_workers: int = 5) -> List[Dict[str, Any]]:
        if not pmids:
            raise ValidationError("PMIDs list cannot be empty")
        if batch_size < 1:
            raise ValidationError("batch_size must be at least 1")
        if max_workers < 1:
            raise ValidationError("max_workers must be at least 1")
        batches = await self._batch_pmids(pmids, batch_size)
        total_batches = len(batches)
        logger.info(f"Fetching {len(pmids)} articles in {total_batches} batches of size {batch_size}")
        semaphore = asyncio.Semaphore(max_workers)
        async def process_batch(batch_index: int, batch_pmids: List[str]) -> List[Dict[str, Any]]:
            async with semaphore:
                try:
                    logger.info(f"Processing batch {batch_index+1}/{total_batches} with {len(batch_pmids)} PMIDs")
                    articles = await self.fetch_article_details(batch_pmids)
                    if include_abstracts and articles:
                        abstracts = await self.fetch_article_abstracts(batch_pmids)
                        for article in articles:
                            pmid = article["pmid"]
                            if pmid in abstracts:
                                article["abstract"] = abstracts[pmid]
                    logger.info(f"Completed batch {batch_index+1}/{total_batches} with {len(articles)} articles")
                    return articles
                except Exception as e:
                    logger.error(f"Error processing batch {batch_index+1}/{total_batches}: {str(e)}")
                    return []
        tasks = [process_batch(i, batch) for i, batch in enumerate(batches)]
        batch_results = await asyncio.gather(*tasks)
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
            max_results = min(max_results, count)
            num_batches = (max_results + batch_size - 1) // batch_size
            semaphore = asyncio.Semaphore(max_workers)
            async def process_batch(batch_index: int) -> List[Dict[str, Any]]:
                async with semaphore:
                    try:
                        retstart = batch_index * batch_size
                        current_batch_size = min(batch_size, max_results - retstart)
                        if current_batch_size <= 0:
                            return []
                        logger.info(f"Fetching batch {batch_index+1}/{num_batches} (records {retstart+1}-{retstart+current_batch_size})")
                        rettype, retmode = None, None
                        if db == "pubmed":
                            rettype, retmode = "medline", "text"
                        elif db in ("nucleotide", "protein"):
                            rettype, retmode = "gb", "text"
                        batch_result = await self.fetch_from_history(
                            web_env=web_env,
                            query_key=query_key,
                            db=db,
                            retstart=retstart,
                            retmax=current_batch_size,
                            rettype=rettype,
                            retmode=retmode
                        )
                        if db == "pubmed" and rettype == "medline" and retmode == "text":
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
                            return batch_result
                    except Exception as e:
                        logger.error(f"Error processing batch {batch_index+1}/{num_batches}: {str(e)}")
                        return []
            tasks = [process_batch(i) for i in range(num_batches)]
            batch_results = await asyncio.gather(*tasks)
            all_records = []
            for batch_result in batch_results:
                if isinstance(batch_result, list):
                    all_records.extend(batch_result)
                else:
                    all_records.append(batch_result)
            logger.info(f"Fetched {len(all_records) if isinstance(all_records, list) else 'raw'} records in total")
            return {
                "count": count,
                "fetched": min(max_results, count),
                "records": all_records
            }
        except NCBIClientError as e:
            logger.error(f"Error in batch search and fetch for '{query}' in {db}: {str(e)}")
            raise NCBIClientError(f"Failed to batch search and fetch: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
            logger.error(f"Unexpected error in batch search and fetch for '{query}' in {db}: {str(e)}")
            raise NCBIClientError(f"Unexpected error in batch search and fetch: {str(e)}")
    async def batch_fetch_sequences(self,
                                  ids: List[str],
                                  db: str = "nucleotide",
                                  batch_size: int = 50,
                                  return_type: str = "fasta",
                                  return_mode: str = "text",
                                  max_workers: int = 3) -> Dict[str, Any]:
        if not ids:
            raise ValidationError("IDs list cannot be empty")
        if batch_size < 1:
            raise ValidationError("batch_size must be at least 1")
        if max_workers < 1:
            raise ValidationError("max_workers must be at least 1")
        batches = [ids[i:i+batch_size] for i in range(0, len(ids), batch_size)]
        total_batches = len(batches)
        logger.info(f"Fetching {len(ids)} sequences in {total_batches} batches of size {batch_size}")
        semaphore = asyncio.Semaphore(max_workers)
        async def process_batch(batch_index: int, batch_ids: List[str]) -> Dict[str, str]:
            async with semaphore:
                try:
                    logger.info(f"Processing batch {batch_index+1}/{total_batches} with {len(batch_ids)} IDs")
                    history_result = await self.post_ids_to_history(batch_ids, db)
                    web_env = history_result["web_env"]
                    query_key = history_result["query_key"]
                    sequences = await self.fetch_from_history(
                        web_env=web_env,
                        query_key=query_key,
                        db=db,
                        rettype=return_type,
                        retmode=return_mode
                    )
                    logger.info(f"Completed batch {batch_index+1}/{total_batches}")
                    if return_type == "fasta" and return_mode == "text":
                        result = {}
                        current_id = None
                        current_seq = []
                        for line in sequences.strip().split("\n"):
                            if line.startswith(">"):
                                if current_id and current_seq:
                                    result[current_id] = "\n".join(current_seq)
                                header = line[1:].strip()
                                current_id = header.split()[0]  # Extract ID from header
                                current_seq = [line]
                            elif current_id:
                                current_seq.append(line)
                        if current_id and current_seq:
                            result[current_id] = "\n".join(current_seq)
                        return result
                    else:
                        return {"batch_result": sequences}
                except Exception as e:
                    logger.error(f"Error processing batch {batch_index+1}/{total_batches}: {str(e)}")
                    return {}
        tasks = [process_batch(i, batch) for i, batch in enumerate(batches)]
        batch_results = await asyncio.gather(*tasks)
        all_sequences = {}
        for batch_result in batch_results:
            all_sequences.update(batch_result)
        logger.info(f"Fetched sequences for {len(all_sequences)} IDs in total")
        return {
            "count": len(all_sequences),
            "sequences": all_sequences
        }
    async def advanced_search(self,
                           db: str = "pubmed",
                           **search_criteria) -> Dict[str, Any]:
        if not db:
            raise ValidationError("Database must be specified")
        if not search_criteria:
            raise ValidationError("At least one search criterion must be provided")
        try:
            query_parts = []
            if db == "pubmed":
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
                if "filters" in search_criteria and isinstance(search_criteria["filters"], list):
                    for filter_name in search_criteria["filters"]:
                        query_parts.append(f"{filter_name}[Filter]")
            elif db in ("nucleotide", "protein", "gene"):
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
            if "free_text" in search_criteria:
                query_parts.append(search_criteria["free_text"])
            query = " AND ".join(query_parts)
            search_params = {}
            for param in ["retmax", "retstart", "sort", "field", "datetype", "reldate", "mindate", "maxdate"]:
                if param in search_criteria:
                    search_params[param] = search_criteria[param]
            logger.info(f"Performing advanced search in {db} with query: {query}")
            result = await self.search_pubmed(query, db=db, **search_params)
            return result
        except NCBIClientError as e:
            logger.error(f"Error performing advanced search in {db}: {str(e)}")
            raise NCBIClientError(f"Failed to perform advanced search: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
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
            params = {
                "db": db,
                "term": query.strip(),
                "datetype": date_type
            }
            if start_date and end_date:
                params["mindate"] = start_date
                params["maxdate"] = end_date
            elif relative_date:
                params["reldate"] = relative_date
            params.update(search_params)
            logger.info(f"Performing date range search in {db} with query: {query}")
            result = await self._make_request("esearch.fcgi", params)
            return result
        except NCBIClientError as e:
            logger.error(f"Error performing date range search in {db}: {str(e)}")
            raise NCBIClientError(f"Failed to perform date range search: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
            logger.error(f"Unexpected error performing date range search in {db}: {str(e)}")
            raise NCBIClientError(f"Unexpected error performing date range search: {str(e)}")
    async def field_search(self,
                         terms: Dict[str, str],
                         db: str = "pubmed",
                         operator: str = "AND",
                         **search_params) -> Dict[str, Any]:
        if not terms:
            raise ValidationError("Terms dictionary cannot be empty")
        if not db:
            raise ValidationError("Database must be specified")
        if operator not in ("AND", "OR", "NOT"):
            raise ValidationError("Operator must be one of: AND, OR, NOT")
        try:
            query_parts = []
            for field, term in terms.items():
                if not term or not term.strip():
                    continue
                field_tag = field
                if not field.endswith("]"):
                    field_tag = f"[{field}]"
                query_parts.append(f"{term.strip()}{field_tag}")
            if not query_parts:
                raise ValidationError("No valid terms provided")
            query = f" {operator} ".join(query_parts)
            logger.info(f"Performing field search in {db} with query: {query}")
            result = await self.search_pubmed(query, db=db, **search_params)
            return result
        except NCBIClientError as e:
            logger.error(f"Error performing field search in {db}: {str(e)}")
            raise NCBIClientError(f"Failed to perform field search: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
            logger.error(f"Unexpected error performing field search in {db}: {str(e)}")
            raise NCBIClientError(f"Unexpected error performing field search: {str(e)}")
    async def proximity_search(self,
                             terms: List[str],
                             field: str = "Title/Abstract",
                             distance: int = 5,
                             **search_params) -> Dict[str, Any]:
        if not terms or len(terms) < 2:
            raise ValidationError("At least two terms must be provided for proximity search")
        if field not in ("Title", "Title/Abstract", "Abstract"):
            raise ValidationError("Field must be one of: Title, Title/Abstract, Abstract")
        if distance < 1:
            raise ValidationError("Distance must be at least 1")
        try:
            terms_str = " ".join(term.strip() for term in terms if term and term.strip())
            query = f"\"{terms_str}\"[{field}:~{distance}]"
            logger.info(f"Performing proximity search in PubMed with query: {query}")
            result = await self.search_pubmed(query, **search_params)
            return result
        except NCBIClientError as e:
            logger.error(f"Error performing proximity search: {str(e)}")
            raise NCBIClientError(f"Failed to perform proximity search: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
            logger.error(f"Unexpected error performing proximity search: {str(e)}")
            raise NCBIClientError(f"Unexpected error performing proximity search: {str(e)}")
    @enhanced_cached(key_prefix="pmc_search")
    async def search_pmc(self,
                       query: str,
                       max_results: int = 20,
                       sort: str = "relevance",
                       min_date: Optional[str] = None,
                       max_date: Optional[str] = None,
                       use_history: bool = False) -> Dict[str, Any]:
        if not query or not query.strip():
            raise ValidationError("Search query cannot be empty")
        if max_results < 1:
            raise ValidationError("max_results must be at least 1")
        params = {
            "db": "pmc",
            "term": query.strip(),
            "retmax": max_results,
            "sort": sort,
            "usehistory": "y" if use_history else "n"
        }
        if min_date and max_date:
            params["datetype"] = "pdat"  # Publication date
            params["mindate"] = min_date
            params["maxdate"] = max_date
        try:
            logger.info(f"Searching PMC for '{query}' (max_results={max_results})")
            data = await self._make_request("esearch.fcgi", params)
            count = int(data.get("esearchresult", {}).get("count", "0"))
            logger.info(f"Found {count} results for '{query}' in PMC")
            return data
        except NCBIClientError as e:
            logger.error(f"Error searching PMC for '{query}': {str(e)}")
            raise NCBIClientError(f"Failed to search PMC: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
            logger.error(f"Unexpected error searching PMC for '{query}': {str(e)}")
            raise NCBIClientError(f"Unexpected error searching PMC: {str(e)}")
    @enhanced_cached(key_prefix="pmc_fetch")
    async def fetch_pmc_article(self, pmcid: str, format: str = "xml") -> str:
        if not pmcid or not pmcid.strip():
            raise ValidationError("PMCID cannot be empty")
        normalized_pmcid = pmcid.strip()
        if not normalized_pmcid.startswith("PMC"):
            normalized_pmcid = f"PMC{normalized_pmcid}"
        valid_formats = ["xml", "medline", "pdf"]
        if format not in valid_formats:
            raise ValidationError(f"Invalid format: {format}. Must be one of {valid_formats}")
        params = {
            "db": "pmc",
            "id": normalized_pmcid
        }
        if format == "xml":
            params["rettype"] = "xml"
            params["retmode"] = "xml"
        elif format == "medline":
            params["rettype"] = "medline"
            params["retmode"] = "text"
        elif format == "pdf":
            return await self._fetch_pmc_pdf(normalized_pmcid)
        try:
            logger.info(f"Fetching PMC article {normalized_pmcid} in {format} format")
            result = await self._make_request("efetch.fcgi", params, return_json=False)
            return result
        except NCBIClientError as e:
            logger.error(f"Error fetching PMC article {normalized_pmcid}: {str(e)}")
            raise NCBIClientError(f"Failed to fetch PMC article: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
            logger.error(f"Unexpected error fetching PMC article {normalized_pmcid}: {str(e)}")
            raise NCBIClientError(f"Unexpected error fetching PMC article: {str(e)}")
    async def _fetch_pmc_pdf(self, pmcid: str) -> str:
        try:
            xml_content = await self.fetch_pmc_article(pmcid, format="xml")
            try:
                root = ET.fromstring(xml_content)
                pdf_url = None
                for elem in root.findall(".//self-uri[@content-type='pdf']"):
                    if "href" in elem.attrib:
                        pdf_url = elem.attrib["href"]
                        break
                if not pdf_url:
                    for elem in root.findall(".//supplementary-material[@content-type='pdf']"):
                        if "href" in elem.attrib:
                            pdf_url = elem.attrib["href"]
                            break
                if not pdf_url:
                    pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/{pmcid}.pdf"
                return pdf_url
            except ET.ParseError as e:
                logger.error(f"Error parsing XML for PMC article {pmcid}: {str(e)}")
                return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/{pmcid}.pdf"
        except Exception as e:
            logger.error(f"Error fetching PDF for PMC article {pmcid}: {str(e)}")
            raise NCBIClientError(f"Failed to fetch PDF for PMC article: {str(e)}")
    @enhanced_cached(key_prefix="pmc_ids")
    async def convert_pmid_to_pmcid(self, pmids: Union[str, List[str]]) -> Dict[str, str]:
        if isinstance(pmids, str):
            pmids = [pmids]
        if not pmids:
            raise ValidationError("PMIDs list cannot be empty")
        invalid_pmids = [pmid for pmid in pmids if not pmid or not pmid.strip()]
        if invalid_pmids:
            raise ValidationError(f"Invalid PMIDs: {invalid_pmids}")
        try:
            params = {
                "dbfrom": "pubmed",
                "db": "pmc",
                "id": ",".join(pmid.strip() for pmid in pmids)
            }
            logger.info(f"Converting {len(pmids)} PMIDs to PMCIDs")
            result = await self._make_request("elink.fcgi", params)
            pmid_to_pmcid = {}
            try:
                linksets = result.get("linksets", [])
                for linkset in linksets:
                    pmid = linkset.get("ids", [None])[0]  # Get the source PMID
                    if not pmid:
                        continue
                    pmid_to_pmcid[pmid] = None
                    for linksetdb in linkset.get("linksetdbs", []):
                        if linksetdb.get("linkname") == "pubmed_pmc":
                            links = linksetdb.get("links", [])
                            if links:
                                pmcid = links[0]
                                pmid_to_pmcid[pmid] = f"PMC{pmcid}"
                                break
            except Exception as e:
                logger.error(f"Error parsing PMID to PMCID conversion result: {str(e)}")
            for pmid in pmids:
                if pmid not in pmid_to_pmcid:
                    pmid_to_pmcid[pmid] = None
            logger.info(f"Converted {len(pmids)} PMIDs to PMCIDs, found {sum(1 for pmcid in pmid_to_pmcid.values() if pmcid)} matches")
            return pmid_to_pmcid
        except NCBIClientError as e:
            logger.error(f"Error converting PMIDs to PMCIDs: {str(e)}")
            raise NCBIClientError(f"Failed to convert PMIDs to PMCIDs: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
            logger.error(f"Unexpected error converting PMIDs to PMCIDs: {str(e)}")
            raise NCBIClientError(f"Unexpected error converting PMIDs to PMCIDs: {str(e)}")
    @enhanced_cached(key_prefix="pmc_ids")
    async def convert_pmcid_to_pmid(self, pmcids: Union[str, List[str]]) -> Dict[str, str]:
        if isinstance(pmcids, str):
            pmcids = [pmcids]
        if not pmcids:
            raise ValidationError("PMCIDs list cannot be empty")
        normalized_pmcids = []
        for pmcid in pmcids:
            if not pmcid or not pmcid.strip():
                raise ValidationError(f"Invalid PMCID: {pmcid}")
            normalized_pmcid = pmcid.strip()
            if not normalized_pmcid.startswith("PMC"):
                normalized_pmcid = f"PMC{normalized_pmcid}"
            normalized_pmcids.append(normalized_pmcid)
        try:
            params = {
                "dbfrom": "pmc",
                "db": "pubmed",
                "id": ",".join(pmcid.replace("PMC", "") for pmcid in normalized_pmcids)  # Remove PMC prefix for the API
            }
            logger.info(f"Converting {len(normalized_pmcids)} PMCIDs to PMIDs")
            result = await self._make_request("elink.fcgi", params)
            pmcid_to_pmid = {}
            try:
                linksets = result.get("linksets", [])
                for linkset in linksets:
                    pmcid_num = linkset.get("ids", [None])[0]  # Get the source PMCID (without PMC prefix)
                    if not pmcid_num:
                        continue
                    pmcid = f"PMC{pmcid_num}"
                    pmcid_to_pmid[pmcid] = None
                    for linksetdb in linkset.get("linksetdbs", []):
                        if linksetdb.get("linkname") == "pmc_pubmed":
                            links = linksetdb.get("links", [])
                            if links:
                                pmid = links[0]
                                pmcid_to_pmid[pmcid] = pmid
                                break
            except Exception as e:
                logger.error(f"Error parsing PMCID to PMID conversion result: {str(e)}")
            for pmcid in normalized_pmcids:
                if pmcid not in pmcid_to_pmid:
                    pmcid_to_pmid[pmcid] = None
            logger.info(f"Converted {len(normalized_pmcids)} PMCIDs to PMIDs, found {sum(1 for pmid in pmcid_to_pmid.values() if pmid)} matches")
            return pmcid_to_pmid
        except NCBIClientError as e:
            logger.error(f"Error converting PMCIDs to PMIDs: {str(e)}")
            raise NCBIClientError(f"Failed to convert PMCIDs to PMIDs: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
            logger.error(f"Unexpected error converting PMCIDs to PMIDs: {str(e)}")
            raise NCBIClientError(f"Unexpected error converting PMCIDs to PMIDs: {str(e)}")
    @enhanced_cached(key_prefix="pmc_extract")
    async def extract_pmc_article_sections(self, pmcid: str) -> Dict[str, Any]:
        if not pmcid or not pmcid.strip():
            raise ValidationError("PMCID cannot be empty")
        try:
            xml_content = await self.fetch_pmc_article(pmcid, format="xml")
            try:
                root = ET.fromstring(xml_content)
                article = {
                    "pmcid": pmcid,
                    "title": "",
                    "abstract": "",
                    "sections": [],
                    "references": []
                }
                title_elem = root.find(".//article-title")
                if title_elem is not None and title_elem.text:
                    article["title"] = self._get_element_text(title_elem)
                abstract_elem = root.find(".//abstract")
                if abstract_elem is not None:
                    article["abstract"] = self._get_element_text(abstract_elem)
                body_elem = root.find(".//body")
                if body_elem is not None:
                    for section_elem in body_elem.findall(".//sec"):
                        section = self._parse_section(section_elem)
                        if section:
                            article["sections"].append(section)
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
            logger.error(f"Error extracting sections from PMC article {pmcid}: {str(e)}")
            raise NCBIClientError(f"Failed to extract sections from PMC article: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
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
        text = element.text or ""
        for child in element:
            child_text = self._get_element_text(child)
            if child_text:
                text += " " + child_text
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
        section = {
            "title": "",
            "content": "",
            "subsections": []
        }
        title_elem = section_elem.find("./title")
        if title_elem is not None:
            section["title"] = self._get_element_text(title_elem)
        paragraphs = []
        for p_elem in section_elem.findall("./p"):
            paragraph = self._get_element_text(p_elem)
            if paragraph:
                paragraphs.append(paragraph)
        section["content"] = "\n\n".join(paragraphs)
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
        element_citation = ref_elem.find(".//element-citation") or ref_elem.find(".//mixed-citation")
        if element_citation is not None:
            article_title = element_citation.find(".//article-title")
            if article_title is not None:
                reference["title"] = self._get_element_text(article_title)
            for person_group in element_citation.findall(".//person-group"):
                for name in person_group.findall(".//name"):
                    surname = name.find("./surname")
                    given_names = name.find("./given-names")
                    if surname is not None and given_names is not None:
                        author = f"{self._get_element_text(surname)} {self._get_element_text(given_names)}"
                        reference["authors"].append(author)
            source = element_citation.find(".//source")
            if source is not None:
                reference["journal"] = self._get_element_text(source)
            year = element_citation.find(".//year")
            if year is not None:
                reference["year"] = self._get_element_text(year)
            volume = element_citation.find(".//volume")
            if volume is not None:
                reference["volume"] = self._get_element_text(volume)
            issue = element_citation.find(".//issue")
            if issue is not None:
                reference["issue"] = self._get_element_text(issue)
            fpage = element_citation.find(".//fpage")
            lpage = element_citation.find(".//lpage")
            if fpage is not None and lpage is not None:
                reference["pages"] = f"{self._get_element_text(fpage)}-{self._get_element_text(lpage)}"
            elif fpage is not None:
                reference["pages"] = self._get_element_text(fpage)
            pub_id_doi = element_citation.find(".//pub-id[@pub-id-type='doi']")
            if pub_id_doi is not None:
                reference["doi"] = self._get_element_text(pub_id_doi)
            pub_id_pmid = element_citation.find(".//pub-id[@pub-id-type='pmid']")
            if pub_id_pmid is not None:
                reference["pmid"] = self._get_element_text(pub_id_pmid)
        return reference
    @enhanced_cached(key_prefix="pmc_search_fetch")
    async def search_and_fetch_pmc(self,
                                 query: str,
                                 max_results: int = 20,
                                 include_full_text: bool = False,
                                 **search_params) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            raise ValidationError("Search query cannot be empty")
        if max_results < 1:
            raise ValidationError("max_results must be at least 1")
        try:
            logger.info(f"Searching PMC for '{query}' (max_results={max_results})")
            search_result = await self.search_pmc(query, max_results=max_results, **search_params)
            pmcids = search_result.get("esearchresult", {}).get("idlist", [])
            if not pmcids:
                logger.warning(f"No results found for '{query}' in PMC")
                return []
            articles = []
            for pmcid in pmcids:
                try:
                    normalized_pmcid = pmcid
                    if not normalized_pmcid.startswith("PMC"):
                        normalized_pmcid = f"PMC{normalized_pmcid}"
                    article = {
                        "pmcid": normalized_pmcid
                    }
                    if include_full_text:
                        article_data = await self.extract_pmc_article_sections(normalized_pmcid)
                        article.update(article_data)
                    else:
                        xml_content = await self.fetch_pmc_article(normalized_pmcid, format="xml")
                        root = ET.fromstring(xml_content)
                        title_elem = root.find(".//article-title")
                        if title_elem is not None:
                            article["title"] = self._get_element_text(title_elem)
                        abstract_elem = root.find(".//abstract")
                        if abstract_elem is not None:
                            article["abstract"] = self._get_element_text(abstract_elem)
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
                    continue
            logger.info(f"Fetched {len(articles)} articles from PMC")
            return articles
        except NCBIClientError as e:
            logger.error(f"Error in search_and_fetch_pmc for '{query}': {str(e)}")
            raise NCBIClientError(f"Failed to search and fetch PMC articles: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
            logger.error(f"Unexpected error in search_and_fetch_pmc for '{query}': {str(e)}")
            raise NCBIClientError(f"Unexpected error searching and fetching PMC articles: {str(e)}")
    @enhanced_cached(key_prefix="pmc_batch_fetch")
    async def batch_fetch_pmc_articles(self,
                                     pmcids: List[str],
                                     include_full_text: bool = False,
                                     batch_size: int = 10,
                                     max_workers: int = 3) -> List[Dict[str, Any]]:
        if not pmcids:
            raise ValidationError("PMCIDs list cannot be empty")
        if batch_size < 1:
            raise ValidationError("batch_size must be at least 1")
        if max_workers < 1:
            raise ValidationError("max_workers must be at least 1")
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
        batches = [normalized_pmcids[i:i+batch_size] for i in range(0, len(normalized_pmcids), batch_size)]
        total_batches = len(batches)
        logger.info(f"Fetching {len(normalized_pmcids)} PMC articles in {total_batches} batches of size {batch_size}")
        semaphore = asyncio.Semaphore(max_workers)
        async def process_batch(batch_index: int, batch_pmcids: List[str]) -> List[Dict[str, Any]]:
            async with semaphore:
                try:
                    logger.info(f"Processing batch {batch_index+1}/{total_batches} with {len(batch_pmcids)} PMCIDs")
                    batch_articles = []
                    for pmcid in batch_pmcids:
                        try:
                            article = {
                                "pmcid": pmcid
                            }
                            if include_full_text:
                                article_data = await self.extract_pmc_article_sections(pmcid)
                                article.update(article_data)
                            else:
                                xml_content = await self.fetch_pmc_article(pmcid, format="xml")
                                root = ET.fromstring(xml_content)
                                title_elem = root.find(".//article-title")
                                if title_elem is not None:
                                    article["title"] = self._get_element_text(title_elem)
                                abstract_elem = root.find(".//abstract")
                                if abstract_elem is not None:
                                    article["abstract"] = self._get_element_text(abstract_elem)
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
                            continue
                    logger.info(f"Completed batch {batch_index+1}/{total_batches} with {len(batch_articles)} articles")
                    return batch_articles
                except Exception as e:
                    logger.error(f"Error processing batch {batch_index+1}/{total_batches}: {str(e)}")
                    return []
        tasks = [process_batch(i, batch) for i, batch in enumerate(batches)]
        batch_results = await asyncio.gather(*tasks)
        all_articles = []
        for batch_result in batch_results:
            all_articles.extend(batch_result)
        logger.info(f"Fetched {len(all_articles)}/{len(normalized_pmcids)} PMC articles in total")
        return all_articles
    async def _batch_pmids(self, pmids: List[str], batch_size: Optional[int] = None) -> List[List[str]]:
        if not isinstance(pmids, list):
            raise ValidationError("pmids must be a list")
        valid_pmids = [pmid for pmid in pmids if pmid and pmid.strip()]
        if len(valid_pmids) < len(pmids):
            logger.warning(f"Filtered out {len(pmids) - len(valid_pmids)} empty PMIDs")
        if not valid_pmids:
            logger.warning("No valid PMIDs provided")
            return []
        batch_size = batch_size or self.max_batch_size
        if batch_size < 1:
            logger.warning(f"Invalid batch size {batch_size}, using default {self.max_batch_size}")
            batch_size = self.max_batch_size
        batches = [valid_pmids[i:i + batch_size] for i in range(0, len(valid_pmids), batch_size)]
        logger.debug(f"Split {len(valid_pmids)} PMIDs into {len(batches)} batches of size {batch_size}")
        return batches
    async def _make_request(self, endpoint: str, params: Dict[str, Any], return_json: bool = True) -> Union[Dict[str, Any], str]:
        cache_key = None
        if self.use_cache:
            param_str = json.dumps(params, sort_keys=True)
            cache_key = f"ncbi:{endpoint}:{hashlib.md5(param_str.encode()).hexdigest()}"
            cached_result = await enhanced_cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {endpoint}")
                return cached_result
        request_params = params.copy()  # Make a copy to avoid modifying the original
        request_params.update({
            "tool": self.tool,
            "email": self.email,
        })
        if "retmode" not in request_params:
            request_params["retmode"] = "json" if return_json else "xml"
        if self.api_key:
            request_params["api_key"] = self.api_key
        await self.rate_limiter.acquire()
        url = f"{self.base_url}{endpoint}"
        logger.debug(f"Making request to {url} with params {request_params}")
        retries = 0
        max_retries = self.max_retries
        backoff_factor = 1.5
        while True:
            try:
                response = await self.client.get(url, params=request_params)
                if response.status_code >= 400:
                    error_msg = f"NCBI API error: {response.status_code} {response.reason_phrase}"
                    logger.warning(f"{error_msg} for {url}")
                    if response.status_code == 429:
                        error_msg = "Rate limit exceeded"
                    elif response.status_code == 400:
                        error_msg = f"Bad request: {response.text}"
                    elif response.status_code == 404:
                        error_msg = f"Resource not found: {endpoint}"
                    elif response.status_code >= 500:
                        error_msg = f"NCBI server error: {response.status_code}"
                    if (response.status_code >= 500 or response.status_code == 429) and retries < max_retries:
                        retries += 1
                        wait_time = backoff_factor ** retries
                        logger.warning(f"Retrying in {wait_time:.1f}s ({retries}/{max_retries})")
                        await asyncio.sleep(wait_time)
                        continue
                    raise NCBIClientError(error_msg, response.status_code)
                result = response.json() if return_json else response.text
                if self.use_cache and cache_key:
                    await enhanced_cache_manager.set(cache_key, result, ttl=self.cache_ttl)
                return result
            except httpx.RequestError as e:
                error_msg = f"Network error: {str(e)}"
                logger.warning(f"{error_msg} for {url}")
                if retries < max_retries:
                    retries += 1
                    wait_time = backoff_factor ** retries
                    logger.warning(f"Retrying in {wait_time:.1f}s ({retries}/{max_retries})")
                    await asyncio.sleep(wait_time)
                else:
                    raise NCBIClientError(f"Failed to connect to NCBI API after {max_retries} retries: {str(e)}")
            except json.JSONDecodeError as e:
                error_msg = f"Invalid JSON response: {str(e)}"
                logger.error(f"{error_msg} for {url}")
                raise NCBIClientError(error_msg)
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                logger.error(f"{error_msg} for {url}")
                raise NCBIClientError(error_msg)
    @enhanced_cached(key_prefix="pubmed_search")
    async def search_pubmed(
        self,
        query: str,
        max_results: int = 20,
        sort: str = "relevance",
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
        use_history: bool = False
    ) -> Dict[str, Any]:
        if not query or not query.strip():
            raise ValidationError("Search query cannot be empty")
        if max_results < 1:
            raise ValidationError("max_results must be at least 1")
        params = {
            "db": "pubmed",
            "term": query.strip(),
            "retmax": max_results,
            "sort": sort,
            "usehistory": "y" if use_history else "n"
        }
        if min_date and max_date:
            params["datetype"] = "pdat"  # Publication date
            params["mindate"] = min_date
            params["maxdate"] = max_date
        try:
            logger.info(f"Searching PubMed for '{query}' (max_results={max_results})")
            data = await self._make_request("esearch.fcgi", params)
            count = int(data.get("esearchresult", {}).get("count", "0"))
            logger.info(f"Found {count} results for '{query}'")
            return data
        except NCBIClientError as e:
            logger.error(f"Error searching PubMed for '{query}': {str(e)}")
            raise NCBIClientError(f"Failed to search PubMed: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
            logger.error(f"Unexpected error searching PubMed for '{query}': {str(e)}")
            raise NCBIClientError(f"Unexpected error searching PubMed: {str(e)}")
    async def fetch_article_details(
        self,
        pmids: Union[str, List[str]],
        db: str = "pubmed",
        batch_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        if isinstance(pmids, str):
            pmids = [pmids]
        if not pmids:
            return []
        invalid_pmids = [pmid for pmid in pmids if not pmid or not pmid.strip()]
        if invalid_pmids:
            raise ValidationError(f"Invalid PMIDs: {invalid_pmids}")
        batches = await self._batch_pmids(pmids, batch_size)
        all_articles = []
        errors = []
        for batch_index, batch in enumerate(batches):
            batch_key = f"article_details:{db}:{','.join(batch)}"
            if self.use_cache:
                cached_batch = await enhanced_cache_manager.get(batch_key)
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
                batch_articles = []
                for pmid in batch:
                    if pmid in data["result"]:
                        article_data = data["result"][pmid]
                        authors = []
                        if "authors" in article_data:
                            for author in article_data["authors"]:
                                if "name" in author:
                                    authors.append(author["name"])
                        pub_date = None
                        if "pubdate" in article_data:
                            pub_date = article_data["pubdate"]
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
                if self.use_cache and batch_articles:
                    await enhanced_cache_manager.set(batch_key, batch_articles, ttl=self.cache_ttl)
                all_articles.extend(batch_articles)
                logger.info(f"Fetched details for {len(batch_articles)} articles in batch {batch_index+1}/{len(batches)}")
            except NCBIClientError as e:
                error_msg = f"Error fetching article details (batch {batch_index+1}/{len(batches)}): {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
                continue
            except Exception as e:
                error_msg = f"Unexpected error fetching article details (batch {batch_index+1}/{len(batches)}): {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
                continue
        logger.info(f"Fetched details for {len(all_articles)}/{len(pmids)} articles with {len(errors)} errors")
        if errors and all_articles:
            for error in errors:
                logger.warning(f"Partial failure in fetch_article_details: {error}")
        elif errors and not all_articles:
            raise NCBIClientError(f"Failed to fetch any article details: {'; '.join(errors[:3])}{'...' if len(errors) > 3 else ''}")
        return all_articles
    async def fetch_article_abstracts(
        self,
        pmids: Union[str, List[str]],
        db: str = "pubmed",
        batch_size: Optional[int] = None
    ) -> Dict[str, str]:
        if isinstance(pmids, str):
            pmids = [pmids]
        if not pmids:
            return {}
        invalid_pmids = [pmid for pmid in pmids if not pmid or not pmid.strip()]
        if invalid_pmids:
            raise ValidationError(f"Invalid PMIDs: {invalid_pmids}")
        batches = await self._batch_pmids(pmids, batch_size)
        all_abstracts = {}
        errors = []
        for batch_index, batch in enumerate(batches):
            batch_key = f"article_abstracts:{db}:{','.join(batch)}"
            if self.use_cache:
                cached_batch = await enhanced_cache_manager.get(batch_key)
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
                logger.info(f"Fetching abstracts for {len(batch)} articles (batch {batch_index+1}/{len(batches)})")
                xml_text = await self._make_request("efetch.fcgi", params, return_json=False)
                batch_abstracts = {}
                try:
                    root = ET.fromstring(xml_text)
                    for article in root.findall("./PubmedArticle"):
                        pmid_elem = article.find(".//PMID")
                        if pmid_elem is None:
                            continue
                        pmid = pmid_elem.text
                        abstract_elems = article.findall(".//AbstractText")
                        if not abstract_elems:
                            continue
                        abstract_text = " ".join(elem.text or "" for elem in abstract_elems)
                        batch_abstracts[pmid] = abstract_text
                except ET.ParseError as e:
                    logger.error(f"Error parsing XML response: {str(e)}")
                    logger.warning("Falling back to simple string parsing")
                    for pmid in batch:
                        pmid_start = xml_text.find(f"<PMID>{pmid}</PMID>")
                        if pmid_start == -1:
                            continue
                        abstract_start = xml_text.find("<AbstractText>", pmid_start)
                        if abstract_start == -1:
                            continue
                        abstract_end = xml_text.find("</AbstractText>", abstract_start)
                        if abstract_end == -1:
                            continue
                        abstract_text = xml_text[abstract_start + 14:abstract_end]
                        batch_abstracts[pmid] = abstract_text
                if self.use_cache and batch_abstracts:
                    await enhanced_cache_manager.set(batch_key, batch_abstracts, ttl=self.cache_ttl)
                all_abstracts.update(batch_abstracts)
                logger.info(f"Fetched abstracts for {len(batch_abstracts)}/{len(batch)} articles in batch {batch_index+1}/{len(batches)}")
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
                continue
            except Exception as e:
                error_msg = f"Unexpected error fetching article abstracts (batch {batch_index+1}/{len(batches)}): {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
                continue
        logger.info(f"Fetched abstracts for {len(all_abstracts)}/{len(pmids)} articles with {len(errors)} errors")
        if errors and all_abstracts:
            for error in errors:
                logger.warning(f"Partial failure in fetch_article_abstracts: {error}")
        elif errors and not all_abstracts:
            raise NCBIClientError(f"Failed to fetch any article abstracts: {'; '.join(errors[:3])}{'...' if len(errors) > 3 else ''}")
        return all_abstracts
    @enhanced_cached(key_prefix="pubmed_abstracts")
    async def fetch_pubmed_abstracts(
        self,
        id_list: Optional[List[str]] = None,
        query: Optional[str] = None,
        max_results: int = 20,
        batch_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        if id_list is None and query is None:
            raise ValidationError("Either id_list or query must be provided")
        if max_results < 1:
            raise ValidationError("max_results must be at least 1")
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
            logger.info(f"Fetching details and abstracts for {len(id_list)} articles")
            details_task = asyncio.create_task(self.fetch_article_details(id_list, batch_size=batch_size))
            abstracts_task = asyncio.create_task(self.fetch_article_abstracts(id_list, batch_size=batch_size))
            articles, abstracts = await asyncio.gather(details_task, abstracts_task)
            for article in articles:
                pmid = article["pmid"]
                if pmid in abstracts:
                    article["abstract"] = abstracts[pmid]
                else:
                    logger.debug(f"No abstract found for PMID {pmid}")
            logger.info(f"Successfully fetched {len(articles)} articles with {len(abstracts)} abstracts")
            return articles
        except NCBIClientError as e:
            logger.error(f"Error fetching PubMed abstracts: {str(e)}")
            raise NCBIClientError(f"Failed to fetch PubMed abstracts: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
            logger.error(f"Unexpected error fetching PubMed abstracts: {str(e)}")
            raise NCBIClientError(f"Unexpected error fetching PubMed abstracts: {str(e)}")
    @enhanced_cached(key_prefix="pubmed_search_fetch")
    async def search_and_fetch_pubmed(
        self,
        query: str,
        max_results: int = 20,
        batch_size: Optional[int] = None,
        sort: str = "relevance",
        min_date: Optional[str] = None,
        max_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            raise ValidationError("Search query cannot be empty")
        if max_results < 1:
            raise ValidationError("max_results must be at least 1")
        try:
            logger.info(f"Searching and fetching PubMed articles for '{query}' (max_results={max_results})")
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
            articles = await self.fetch_pubmed_abstracts(
                id_list=id_list,
                batch_size=batch_size
            )
            logger.info(f"Successfully searched and fetched {len(articles)} articles for query '{query}'")
            return articles
        except ValidationError as e:
            raise
        except NCBIClientError as e:
            logger.error(f"Error in search_and_fetch_pubmed for '{query}': {str(e)}")
            raise NCBIClientError(f"Failed to search and fetch PubMed articles: {str(e)}", getattr(e, 'status_code', None))
        except Exception as e:
            logger.error(f"Unexpected error in search_and_fetch_pubmed for '{query}': {str(e)}")
            raise NCBIClientError(f"Unexpected error searching and fetching PubMed articles: {str(e)}")
    async def clear_cache(self, pattern: Optional[str] = None) -> int:
        if not self.use_cache:
            logger.warning("Cache is disabled for this client")
            return 0
        try:
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
                    cleared = await enhanced_cache_manager.delete_pattern(p)
                    logger.info(f"Cleared {cleared} cache entries matching pattern '{p}'")
                    total_cleared += cleared
                logger.info(f"Cleared {total_cleared} NCBI-related cache entries in total")
                return total_cleared
            else:
                cleared = await enhanced_cache_manager.delete_pattern(pattern)
                logger.info(f"Cleared {cleared} cache entries matching pattern '{pattern}'")
                return cleared
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            return 0
    async def get_cache_stats(self) -> Dict[str, Any]:
        if not self.use_cache:
            logger.warning("Cache is disabled for this client")
            return {"enabled": False}
        try:
            stats = await enhanced_cache_manager.get_stats()
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
                count = await enhanced_cache_manager.count_pattern(pattern)
                pattern_counts[pattern] = count
            stats["ncbi_patterns"] = pattern_counts
            stats["ncbi_total"] = sum(pattern_counts.values())
            stats["enabled"] = True
            stats["ttl"] = self.cache_ttl
            return stats
        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            return {"enabled": self.use_cache, "error": str(e)}