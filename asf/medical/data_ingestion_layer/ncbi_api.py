"""
NCBI E-utilities API Module for Medical Research Synthesizer
This module provides comprehensive access to NCBI's Entrez Programming Utilities (E-utilities),
enabling retrieval of biomedical literature and data from NCBI databases like PubMed.
The module implements all core E-utilities functions while strictly adhering to NCBI's usage
guidelines and rate limits.
Prerequisites:
- NCBI API key (optional but recommended): https://ncbiinsights.ncbi.nlm.nih.gov/2017/11/02/new-api-keys-for-the-e-utilities/
- Python 3.6+
- Required packages: requests, dotenv
Usage:
1. Store your NCBI API key in a .env file (optional but recommended)
2. Use the NCBIClient class to interact with the E-utilities
"""
import time
import json
import xml.etree.ElementTree as ET
import logging
import requests
from typing import Dict, List, Optional, Union, Any, Tuple
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ncbi_api')
class NCBIClient:
    """Client for interacting with the NCBI E-utilities API."""
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 tool: str = "medical_research_synthesizer",
                 email: Optional[str] = None,
                 rate_limit_per_second: int = 3,
                 retry_attempts: int = 3,
                 retry_delay: float = 1.0):
        """
        Initialize the NCBI E-utilities client.
        Args:
            api_key: NCBI API key. If not provided, will attempt to load from environment variables.
            tool: Identifier for your application, required by NCBI.
            email: Developer's email, required by NCBI.
            rate_limit_per_second: Maximum number of requests per second (3 without API key, 10 with key).
            retry_attempts: Number of retry attempts for failed requests.
            retry_delay: Initial delay between retries in seconds (increases exponentially).
        Get a list of valid Entrez databases.
        This method uses cached values to avoid an API call on initialization.
        Use einfo() directly to get up-to-date database information.
        Returns:
            List[str]: List of valid Entrez database names
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        min_interval = 1.0 / self.rate_limit
        if time_since_last_request < min_interval:
            delay = min_interval - time_since_last_request
            logger.debug(f"Rate limiting: Sleeping for {delay:.3f} seconds")
            time.sleep(delay)
        self.last_request_time = time.time()
    def _build_base_params(self) -> Dict[str, str]:
        """Build the base parameters required for all E-utilities requests."""
        params = {
            'tool': self.tool
        }
        if self.email:
            params['email'] = self.email
        if self.api_key:
            params['api_key'] = self.api_key
        return params
    def _make_request(self, 
                      endpoint: str, 
                      params: Dict[str, Any], 
                      method: str = 'GET') -> Optional[requests.Response]:
        """
        Make a request to an E-utilities endpoint with rate limiting and retry logic.
        Args:
            endpoint: The E-utilities endpoint (e.g., 'esearch.fcgi', 'efetch.fcgi')
            params: Dictionary of parameters for the request
            method: HTTP method to use ('GET' or 'POST')
        Returns:
            Optional[requests.Response]: Response object if successful, None otherwise
        Check if the provided database name is valid.
        Args:
            db: The database name to validate
        Returns:
            bool: True if the database is valid, False otherwise
        Search for terms in the specified NCBI database.
        Args:
            db: The NCBI database to search (e.g., 'pubmed', 'gene')
            term: The search term
            use_history: Whether to use the Entrez History server
            retmax: Maximum number of results to return (max 10,000)
            retstart: Index of the first result to return
            sort: Sort order (e.g., 'relevance', 'pub_date' for PubMed)
            field: Field to search in (e.g., 'title', 'abstract')
            idtype: Type of identifier to return for sequence DBs ('acc' for accession numbers)
            datetype: Type of date used to limit a search (e.g., 'mdat', 'pdat', 'edat')
            reldate: Only items with specified date within last N days
            mindate: Start date for date range (YYYY/MM/DD format)
            maxdate: End date for date range (YYYY/MM/DD format)
            retmode: Format of the returned output ('json' or 'xml')
            **kwargs: Additional parameters for the ESearch utility
        Returns:
            Optional[Dict]: Dictionary containing search results if successful, None otherwise
        Upload a list of UIDs to the Entrez History server.
        Args:
            db: The NCBI database the UIDs belong to
            id_list: List of UIDs to upload
            use_existing_web_env: If True, will append to existing WebEnv
        Returns:
            Optional[Dict[str, str]]: Dictionary with WebEnv and query_key if successful
        Fetch records from NCBI databases.
        Args:
            db: The NCBI database to fetch from
            id_list: List of UIDs to fetch (alternative to using history server)
            web_env: History server Web Environment (defaults to stored value if None)
            query_key: History server Query Key (defaults to stored value if None)
            rettype: Type of data to return (e.g., 'abstract', 'fasta', 'gb')
            retmode: Format of the returned data (e.g., 'text', 'xml', 'json')
            retmax: Maximum number of records to return
            retstart: Index of the first record to return
            strand: Strand of DNA to retrieve (1=plus, 2=minus) for sequence databases
            seq_start: First sequence base to retrieve (1-based index)
            seq_stop: Last sequence base to retrieve
            complexity: Data content to return (0=entire blob, 1=bioseq, etc.)
            **kwargs: Additional parameters for the EFetch utility
        Returns:
            Optional[str]: Fetched records if successful, None otherwise
        Retrieve document summaries for a list of UIDs.
        Args:
            db: The NCBI database to fetch from
            id_list: List of UIDs to fetch summaries for (alternative to using history server)
            web_env: History server Web Environment (defaults to stored value if None)
            query_key: History server Query Key (defaults to stored value if None)
            retmode: Format of the returned data ('json' or 'xml')
            retmax: Maximum number of summaries to return
            retstart: Index of the first summary to return
            version: Used to specify ESummary version ('2.0' for enhanced output)
            **kwargs: Additional parameters for the ESummary utility
        Returns:
            Optional[Dict]: Document summaries if successful, None otherwise
        Find links between records in NCBI databases.
        Args:
            dbfrom: Source database
            db: Target database
            id_list: List of UIDs to find links for (alternative to using history server)
            web_env: History server Web Environment (defaults to stored value if None)
            query_key: History server Query Key (defaults to stored value if None)
            linkname: Name of the Entrez link to retrieve (e.g., 'pubmed_protein')
            cmd: ELink command (e.g., 'neighbor', 'neighbor_history', 'acheck')
            retmode: Format of the returned data ('json' or 'xml')
            term: Entrez query to limit the output linked UIDs
            holding: Name of LinkOut provider to filter results by
            datetype: Type of date used to limit links (for PubMed)
            reldate: Only return links to items with specified date within last N days
            mindate: Start date for date range (YYYY/MM/DD format)
            maxdate: End date for date range (YYYY/MM/DD format)
            **kwargs: Additional parameters for the ELink utility
        Returns:
            Optional[Dict]: Link information if successful, None otherwise
        Retrieve information about NCBI Entrez databases.
        Args:
            db: Specific database to get information about (None for all databases)
            retmode: Format of the returned data ('json' or 'xml')
            version: Used to specify EInfo version ('2.0' for enhanced output)
        Returns:
            Optional[Dict]: Database information if successful, None otherwise
        Get spelling suggestions for search terms.
        Args:
            db: The NCBI database to get spelling suggestions for
            term: The search term to check
        Returns:
            Optional[Dict]: Spelling suggestions if successful, None otherwise
        Match citation strings to PubMed IDs.
        Args:
            citation_list: List of dictionaries with journal, year, volume, first_page, author, and key
            db: Must be 'pubmed'
            retmode: Must be 'xml'
        Returns:
            Optional[Dict]: Dictionary mapping citation keys to PMIDs
        Download a large dataset in batches.
        Args:
            db: The NCBI database to fetch from
            query: Search query to fetch results for (alternative to id_list)
            id_list: List of UIDs to fetch (alternative to query)
            rettype: Type of data to return
            retmode: Format of the returned data
            batch_size: Number of records to download in each batch
            max_records: Maximum total number of records to download (None for all)
        Yields:
            str: Batches of downloaded records
        Convenience method for searching PubMed.
        Args:
            query: The search query
            max_results: Maximum number of results to return
            use_history: Whether to use the Entrez History server
            sort: Sort order (e.g., 'relevance', 'pub_date')
            **kwargs: Additional parameters for the ESearch utility
        Returns:
            Optional[Dict]: Search results if successful, None otherwise
        Fetch and parse PubMed abstracts.
        Args:
            id_list: List of PubMed IDs (alternative to using history server)
            max_results: Maximum number of abstracts to return
            parse_xml: Whether to parse the XML into a structured format
            **kwargs: Additional parameters for the EFetch utility
        Returns:
            Optional[Union[List[Dict], str]]: List of parsed abstracts or raw XML
        Convenience method to search PubMed and fetch abstracts in one step.
        Args:
            query: The search query
            max_results: Maximum number of results to return
            parse_xml: Whether to parse the XML into a structured format
        Returns:
            Optional[Union[List[Dict], str]]: List of parsed abstracts or raw XML
        Get articles that cite the specified PubMed article.
        Args:
            pmid: PubMed ID of the article
            max_results: Maximum number of results to return
        Returns:
            Optional[List[Dict]]: List of parsed abstracts that cite the article
        Get articles similar to the specified PubMed article.
        Args:
            pmid: PubMed ID of the article
            max_results: Maximum number of results to return
        Returns:
            Optional[List[Dict]]: List of parsed abstracts similar to the article
        Search for genes in the Gene database.
        Args:
            query: The search query
            organism: Organism name or tax ID to restrict results
            max_results: Maximum number of results to return
        Returns:
            Optional[Dict]: Gene search results
        Fetch gene summaries from the Gene database.
        Args:
            id_list: List of Gene IDs (alternative to using history server)
            max_results: Maximum number of summaries to return
        Returns:
            Optional[Dict]: Gene summaries
        Specialized method to search for Community-Acquired Pneumonia literature.
        Args:
            specific_aspects: List of specific aspects of CAP to focus on (e.g., ['treatment', 'diagnosis'])
            years_range: Tuple of (start_year, end_year) to limit the search
            max_results: Maximum number of results to return
        Returns:
            Optional[List[Dict]]: List of parsed abstracts if successful, None otherwise