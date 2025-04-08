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

import os
import time
import json
import xml.etree.ElementTree as ET
import logging
from urllib.parse import quote_plus, urlencode
import requests
from typing import Dict, List, Optional, Union, Any, Tuple, Iterator
from dotenv import load_dotenv

# Configure logging
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
        """
        # Load environment variables
        load_dotenv()
        
        # Set API credentials
        self.api_key = api_key or os.getenv('NCBI_API_KEY')
        self.email = email or os.getenv('NCBI_EMAIL')
        self.tool = tool
        
        if not self.email:
            logger.warning("No email provided. This is required by NCBI usage policy.")
        
        # Set rate limiting
        self.rate_limit = min(rate_limit_per_second, 10) if self.api_key else min(rate_limit_per_second, 3)
        self.last_request_time = 0
        
        # Set retry configuration
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        
        # NCBI E-utilities base URL
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        
        # Current history server WebEnv and query_key from the last search
        self.web_env = None
        self.query_key = None
        
        # Valid Entrez databases
        self.valid_databases = self._get_valid_databases()
    
    def _get_valid_databases(self) -> List[str]:
        """
        Get a list of valid Entrez databases.
        This method uses cached values to avoid an API call on initialization.
        Use einfo() directly to get up-to-date database information.
        
        Returns:
            List[str]: List of valid Entrez database names
        """
        # Hardcoded list of common databases to avoid an API call on initialization
        # This list can be updated by calling einfo()
        return [
            'pubmed', 'protein', 'nuccore', 'nucleotide', 'nucgss', 'nucest',
            'structure', 'genome', 'assembly', 'genomeprj', 'bioproject',
            'biosample', 'blastdbinfo', 'books', 'cdd', 'clinvar', 'clone',
            'gap', 'gapplus', 'grasp', 'dbvar', 'gene', 'gds', 'geoprofiles',
            'homologene', 'medgen', 'mesh', 'ncbisearch', 'nlmcatalog', 'omim',
            'orgtrack', 'pmc', 'popset', 'probe', 'proteinclusters', 'pcassay',
            'pccompound', 'pcsubstance', 'seqannot', 'snp', 'sra', 'taxonomy',
            'biocollections', 'gtr'
        ]
    
    def _enforce_rate_limit(self) -> None:
        """Enforce NCBI's rate limit by adding appropriate delay between requests."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        # Calculate required delay to maintain rate limit
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
        """
        # Enforce rate limit
        self._enforce_rate_limit()
        
        # Add base parameters
        all_params = {**self._build_base_params(), **params}
        
        # Build the full URL
        url = f"{self.base_url}{endpoint}"
        
        # Implement retry logic
        for attempt in range(self.retry_attempts):
            try:
                if method.upper() == 'GET':
                    response = requests.get(url, params=all_params)
                elif method.upper() == 'POST':
                    response = requests.post(url, data=all_params)
                else:
                    logger.error(f"Unsupported HTTP method: {method}")
                    return None
                
                response.raise_for_status()
                return response
                
            except requests.exceptions.RequestException as e:
                # Check if this is the last attempt
                if attempt == self.retry_attempts - 1:
                    logger.error(f"Request failed after {self.retry_attempts} attempts: {str(e)}")
                    return None
                
                # Calculate delay with exponential backoff
                delay = self.retry_delay * (2 ** attempt)
                logger.warning(f"Request failed (attempt {attempt+1}/{self.retry_attempts}), "
                               f"retrying in {delay:.2f} seconds: {str(e)}")
                time.sleep(delay)
    
    def validate_database(self, db: str) -> bool:
        """
        Check if the provided database name is valid.
        
        Args:
            db: The database name to validate
            
        Returns:
            bool: True if the database is valid, False otherwise
        """
        if db not in self.valid_databases:
            logger.warning(f"Potentially invalid database name: {db}. "
                          f"Use einfo() for a list of valid databases.")
            return False
        return True
    
    def esearch(self, 
                db: str, 
                term: str, 
                use_history: bool = False,
                retmax: int = 20,
                retstart: int = 0,
                sort: Optional[str] = None,
                field: Optional[str] = None,
                idtype: Optional[str] = None,
                datetype: Optional[str] = None,
                reldate: Optional[int] = None,
                mindate: Optional[str] = None,
                maxdate: Optional[str] = None,
                retmode: str = 'json',
                **kwargs) -> Optional[Dict]:
        """
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
        """
        # Validate database
        self.validate_database(db)
        
        # Build search query
        formatted_term = term
        if field:
            formatted_term = f"{formatted_term}[{field}]"
        
        # Set up parameters
        params = {
            'db': db,
            'term': formatted_term,
            'retmode': retmode,
            'retmax': min(retmax, 10000),  # Maximum allowed by NCBI
            'retstart': retstart
        }
        
        # Add optional parameters if provided
        if sort:
            params['sort'] = sort
            
        if idtype:
            params['idtype'] = idtype
            
        if datetype:
            params['datetype'] = datetype
            
            # Date parameters require datetype
            if reldate is not None:
                params['reldate'] = reldate
                
            if mindate and maxdate:
                params['mindate'] = mindate
                params['maxdate'] = maxdate
            
        # Add history server parameters if requested
        if use_history:
            params['usehistory'] = 'y'
            
        # Add any additional parameters
        params.update(kwargs)
        
        # Make the request
        response = self._make_request('esearch.fcgi', params)
        
        if not response:
            return None
            
        try:
            if retmode == 'json':
                data = response.json()
                
                # Store WebEnv and query_key if history was used
                if use_history and 'esearchresult' in data:
                    self.web_env = data['esearchresult'].get('webenv')
                    self.query_key = data['esearchresult'].get('querykey')
                    logger.debug(f"Stored WebEnv: {self.web_env} and query_key: {self.query_key}")
                    
                return data
            else:
                # Parse XML response
                root = ET.fromstring(response.text)
                
                # Store WebEnv and query_key if history was used
                if use_history:
                    web_env_elem = root.find('./WebEnv')
                    query_key_elem = root.find('./QueryKey')
                    
                    if web_env_elem is not None and query_key_elem is not None:
                        self.web_env = web_env_elem.text
                        self.query_key = query_key_elem.text
                        
                return {'root': root}
                
        except (json.JSONDecodeError, ET.ParseError) as e:
            logger.error(f"Failed to parse response: {str(e)}")
            return None
    
    def epost(self, 
              db: str, 
              id_list: List[str],
              use_existing_web_env: bool = False) -> Optional[Dict[str, str]]:
        """
        Upload a list of UIDs to the Entrez History server.
        
        Args:
            db: The NCBI database the UIDs belong to
            id_list: List of UIDs to upload
            use_existing_web_env: If True, will append to existing WebEnv
            
        Returns:
            Optional[Dict[str, str]]: Dictionary with WebEnv and query_key if successful
        """
        # Validate database
        self.validate_database(db)
        
        # Set up parameters
        params = {
            'db': db,
            'id': ','.join(id_list)
        }
        
        # Add WebEnv if appending to existing set
        if use_existing_web_env and self.web_env:
            params['WebEnv'] = self.web_env
        
        # Use POST method for potentially large ID lists
        method = 'POST'
        
        # Make the request
        response = self._make_request('epost.fcgi', params, method=method)
        
        if not response:
            return None
            
        try:
            # Parse XML response
            root = ET.fromstring(response.text)
            
            # Extract WebEnv and query_key
            web_env_elem = root.find('./WebEnv')
            query_key_elem = root.find('./QueryKey')
            
            if web_env_elem is not None and query_key_elem is not None:
                self.web_env = web_env_elem.text
                self.query_key = query_key_elem.text
                
                return {
                    'web_env': self.web_env,
                    'query_key': self.query_key
                }
            else:
                logger.error("Failed to extract WebEnv or query_key from EPost response")
                return None
                
        except ET.ParseError as e:
            logger.error(f"Failed to parse XML response: {str(e)}")
            return None
    
    def efetch(self, 
               db: str, 
               id_list: Optional[List[str]] = None,
               web_env: Optional[str] = None,
               query_key: Optional[str] = None,
               rettype: str = 'abstract',
               retmode: str = 'text',
               retmax: int = 20,
               retstart: int = 0,
               strand: Optional[str] = None,
               seq_start: Optional[int] = None,
               seq_stop: Optional[int] = None,
               complexity: Optional[int] = None,
               **kwargs) -> Optional[str]:
        """
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
        """
        # Validate database
        self.validate_database(db)
        
        # Set up parameters
        params = {
            'db': db,
            'rettype': rettype,
            'retmode': retmode,
            'retmax': retmax,
            'retstart': retstart
        }
        
        # Use either direct ID list or history server
        if id_list:
            # For larger lists, use POST method (handled in _make_request)
            method = 'POST' if len(id_list) > 200 else 'GET'
            params['id'] = ','.join(id_list)
        elif web_env or self.web_env:
            method = 'GET'
            params['WebEnv'] = web_env or self.web_env
            params['query_key'] = query_key or self.query_key
        else:
            logger.error("Either id_list or web_env/query_key must be provided")
            return None
        
        # Add sequence-specific parameters if provided (for sequence databases)
        if db in ['nuccore', 'nucgss', 'nucest', 'protein', 'genome', 'popset']:
            if strand:
                params['strand'] = strand
                
            if seq_start is not None:
                params['seq_start'] = seq_start
                
            if seq_stop is not None:
                params['seq_stop'] = seq_stop
                
            if complexity is not None:
                params['complexity'] = complexity
            
        # Add any additional parameters
        params.update(kwargs)
        
        # Make the request
        response = self._make_request('efetch.fcgi', params, method=method)
        
        if not response:
            return None
            
        return response.text
    
    def esummary(self, 
                 db: str, 
                 id_list: Optional[List[str]] = None,
                 web_env: Optional[str] = None,
                 query_key: Optional[str] = None,
                 retmode: str = 'json',
                 retmax: int = 20,
                 retstart: int = 0,
                 version: Optional[str] = None,
                 **kwargs) -> Optional[Dict]:
        """
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
        """
        # Validate database
        self.validate_database(db)
        
        # Set up parameters
        params = {
            'db': db,
            'retmode': retmode,
            'retmax': retmax,
            'retstart': retstart
        }
        
        # Add version parameter for enhanced output
        if version:
            params['version'] = version
        
        # Use either direct ID list or history server
        if id_list:
            # For larger lists, use POST method (handled in _make_request)
            method = 'POST' if len(id_list) > 200 else 'GET'
            params['id'] = ','.join(id_list)
        elif web_env or self.web_env:
            method = 'GET'
            params['WebEnv'] = web_env or self.web_env
            params['query_key'] = query_key or self.query_key
        else:
            logger.error("Either id_list or web_env/query_key must be provided")
            return None
            
        # Add any additional parameters
        params.update(kwargs)
        
        # Make the request
        response = self._make_request('esummary.fcgi', params, method=method)
        
        if not response:
            return None
            
        try:
            if retmode == 'json':
                return response.json()
            else:
                # Parse XML response
                root = ET.fromstring(response.text)
                return {'root': root}  # Return parsed XML tree
                
        except (json.JSONDecodeError, ET.ParseError) as e:
            logger.error(f"Failed to parse response: {str(e)}")
            return None
    
    def elink(self, 
              dbfrom: str, 
              db: str,
              id_list: Optional[List[str]] = None,
              web_env: Optional[str] = None,
              query_key: Optional[str] = None,
              linkname: Optional[str] = None,
              cmd: str = 'neighbor',
              retmode: str = 'json',
              term: Optional[str] = None,
              holding: Optional[str] = None,
              datetype: Optional[str] = None,
              reldate: Optional[int] = None,
              mindate: Optional[str] = None,
              maxdate: Optional[str] = None,
              **kwargs) -> Optional[Dict]:
        """
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
        """
        # Validate databases
        self.validate_database(dbfrom)
        self.validate_database(db)
        
        # Set up parameters
        params = {
            'dbfrom': dbfrom,
            'db': db,
            'cmd': cmd,
        }
        
        # Add retmode if applicable
        if cmd in ['neighbor', 'neighbor_score']:
            params['retmode'] = retmode
        
        # Add link name if provided
        if linkname:
            params['linkname'] = linkname
            
        # Add term if provided
        if term:
            params['term'] = term
            
        # Add holding parameter if provided
        if holding:
            params['holding'] = holding
            
        # Add date parameters if provided (only for PubMed)
        if dbfrom == 'pubmed' and datetype:
            params['datetype'] = datetype
            
            if reldate is not None:
                params['reldate'] = reldate
                
            if mindate and maxdate:
                params['mindate'] = mindate
                params['maxdate'] = maxdate
        
        # Use either direct ID list or history server
        if id_list:
            # For larger lists, use POST method (handled in _make_request)
            method = 'POST' if len(id_list) > 200 else 'GET'
            
            # For one-to-one linking, use multiple id parameters
            if kwargs.get('preserve_one_to_one', False):
                # Create a new parameter dictionary with multiple id parameters
                multi_id_params = {}
                for k, v in params.items():
                    multi_id_params[k] = v
                    
                for uid in id_list:
                    multi_id_params['id'] = uid
                    
                # Remove the preserving flag
                kwargs.pop('preserve_one_to_one', None)
                
                # Update with any remaining kwargs
                multi_id_params.update(kwargs)
                
                # Make the request with multiple id parameters
                response = self._make_request('elink.fcgi', multi_id_params, method=method)
            else:
                # Regular batch mode
                params['id'] = ','.join(id_list)
                params.update(kwargs)
                response = self._make_request('elink.fcgi', params, method=method)
        elif web_env or self.web_env:
            method = 'GET'
            params['WebEnv'] = web_env or self.web_env
            params['query_key'] = query_key or self.query_key
            params.update(kwargs)
            response = self._make_request('elink.fcgi', params, method=method)
        else:
            logger.error("Either id_list or web_env/query_key must be provided")
            return None
        
        if not response:
            return None
            
        try:
            if retmode == 'json' and cmd in ['neighbor', 'neighbor_score']:
                return response.json()
            else:
                # Parse XML response
                root = ET.fromstring(response.text)
                
                # For neighbor_history command, update webenv and query_key
                if cmd == 'neighbor_history':
                    web_env_elem = root.find('.//WebEnv')
                    query_key_elem = root.find('.//QueryKey')
                    
                    if web_env_elem is not None and query_key_elem is not None:
                        self.web_env = web_env_elem.text
                        self.query_key = query_key_elem.text
                
                return {'root': root}  # Return parsed XML tree
                
        except (json.JSONDecodeError, ET.ParseError) as e:
            logger.error(f"Failed to parse response: {str(e)}")
            return None
    
    def einfo(self, 
              db: Optional[str] = None, 
              retmode: str = 'json',
              version: Optional[str] = None) -> Optional[Dict]:
        """
        Retrieve information about NCBI Entrez databases.
        
        Args:
            db: Specific database to get information about (None for all databases)
            retmode: Format of the returned data ('json' or 'xml')
            version: Used to specify EInfo version ('2.0' for enhanced output)
            
        Returns:
            Optional[Dict]: Database information if successful, None otherwise
        """
        # Set up parameters
        params = {
            'retmode': retmode
        }
        
        if db:
            # Validate database
            self.validate_database(db)
            params['db'] = db
            
        if version:
            params['version'] = version
            
        # Make the request
        response = self._make_request('einfo.fcgi', params)
        
        if not response:
            return None
            
        try:
            if retmode == 'json':
                data = response.json()
                
                # Update valid_databases list if retrieving info for all databases
                if not db and 'einforesult' in data and 'dblist' in data['einforesult']:
                    self.valid_databases = data['einforesult']['dblist']
                    
                return data
            else:
                # Parse XML response
                root = ET.fromstring(response.text)
                
                # Update valid_databases list if retrieving info for all databases
                if not db:
                    db_list = [db_elem.text for db_elem in root.findall('.//DbName')]
                    if db_list:
                        self.valid_databases = db_list
                        
                return {'root': root}  # Return parsed XML tree
                
        except (json.JSONDecodeError, ET.ParseError) as e:
            logger.error(f"Failed to parse response: {str(e)}")
            return None
    
    def espell(self, 
               db: str, 
               term: str) -> Optional[Dict]:
        """
        Get spelling suggestions for search terms.
        
        Args:
            db: The NCBI database to get spelling suggestions for
            term: The search term to check
            
        Returns:
            Optional[Dict]: Spelling suggestions if successful, None otherwise
        """
        # Validate database
        self.validate_database(db)
        
        # Set up parameters
        params = {
            'db': db,
            'term': term
        }
        
        # Make the request
        response = self._make_request('espell.fcgi', params)
        
        if not response:
            return None
            
        try:
            # Parse XML response
            root = ET.fromstring(response.text)
            
            # Extract spelling suggestions
            corrected_query = root.find('./CorrectedQuery')
            
            if corrected_query is not None:
                return {
                    'original_term': term,
                    'corrected_term': corrected_query.text
                }
            else:
                return {
                    'original_term': term,
                    'corrected_term': None  # No correction needed
                }
                
        except ET.ParseError as e:
            logger.error(f"Failed to parse XML response: {str(e)}")
            return None
    
    def ecitmatch(self,
                 citation_list: List[Dict[str, str]],
                 db: str = 'pubmed',
                 retmode: str = 'xml') -> Optional[Dict]:
        """
        Match citation strings to PubMed IDs.
        
        Args:
            citation_list: List of dictionaries with journal, year, volume, first_page, author, and key
            db: Must be 'pubmed'
            retmode: Must be 'xml'
            
        Returns:
            Optional[Dict]: Dictionary mapping citation keys to PMIDs
        """
        if db != 'pubmed':
            logger.error("ECitMatch only supports 'pubmed' database")
            return None
            
        if retmode != 'xml':
            logger.error("ECitMatch only supports 'xml' retmode")
            return None
            
        # Format citation strings
        citation_strings = []
        for citation in citation_list:
            # Required format: journal|year|volume|first_page|author|your_key|
            citation_str = "|".join([
                citation.get('journal', ''),
                citation.get('year', ''),
                citation.get('volume', ''),
                citation.get('first_page', ''),
                citation.get('author', ''),
                citation.get('key', '')
            ]) + "|"
            citation_strings.append(citation_str)
            
        # Set up parameters
        params = {
            'db': db,
            'retmode': retmode,
            'bdata': "\n".join(citation_strings)
        }
        
        # Make the request using POST
        response = self._make_request('ecitmatch.cgi', params, method='POST')
        
        if not response:
            return None
            
        try:
            # Parse the response (pipe-separated values)
            results = {}
            lines = response.text.strip().split('\n')
            
            for line in lines:
                parts = line.split('|')
                if len(parts) >= 7:  # journal|year|volume|page|author|key|pmid
                    key = parts[5]
                    pmid = parts[6]
                    results[key] = pmid if pmid.strip() else None
                    
            return results
                
        except Exception as e:
            logger.error(f"Failed to parse ECitMatch response: {str(e)}")
            return None
    
    def download_large_dataset(self,
                              db: str,
                              query: str = None,
                              id_list: List[str] = None,
                              rettype: str = 'abstract',
                              retmode: str = 'text',
                              batch_size: int = 500,
                              max_records: int = None) -> Iterator[str]:
        """
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
        """
        # Validate database
        self.validate_database(db)
        
        if query and id_list:
            logger.error("Provide either query or id_list, not both")
            return
            
        # If query is provided, perform search and use history server
        if query:
            search_results = self.esearch(
                db=db,
                term=query,
                use_history=True,
                retmax=0  # We just need the count and history server info
            )
            
            if not search_results or 'esearchresult' not in search_results:
                logger.error("Search failed")
                return
                
            count = int(search_results['esearchresult'].get('count', 0))
            
            if count == 0:
                logger.info(f"No results found for query: {query}")
                return
                
            web_env = self.web_env
            query_key = self.query_key
            
            if not web_env or not query_key:
                logger.error("Failed to get WebEnv and query_key")
                return
                
            # Limit total records if specified
            total_records = min(count, max_records) if max_records else count
            
            logger.info(f"Downloading {total_records} records in batches of {batch_size}")
            
            # Download in batches
            for start in range(0, total_records, batch_size):
                end = min(start + batch_size, total_records)
                logger.info(f"Downloading batch {start+1} to {end}")
                
                data = self.efetch(
                    db=db,
                    web_env=web_env,
                    query_key=query_key,
                    rettype=rettype,
                    retmode=retmode,
                    retstart=start,
                    retmax=batch_size
                )
                
                if data:
                    yield data
                else:
                    logger.error(f"Failed to fetch batch starting at {start}")
                    return
                    
        # If id_list is provided, process it in batches
        elif id_list:
            # Limit total records if specified
            total_ids = min(len(id_list), max_records) if max_records else len(id_list)
            id_list = id_list[:total_ids]
            
            logger.info(f"Downloading {total_ids} records in batches of {batch_size}")
            
            # Process in batches
            for i in range(0, total_ids, batch_size):
                batch = id_list[i:i+batch_size]
                logger.info(f"Downloading batch {i+1} to {i+len(batch)}")
                
                data = self.efetch(
                    db=db,
                    id_list=batch,
                    rettype=rettype,
                    retmode=retmode
                )
                
                if data:
                    yield data
                else:
                    logger.error(f"Failed to fetch batch starting at {i}")
                    return
        else:
            logger.error("Either query or id_list must be provided")
            return
    
    def search_pubmed(self, 
                      query: str, 
                      max_results: int = 20,
                      use_history: bool = True,
                      sort: str = 'relevance',
                      **kwargs) -> Optional[Dict]:
        """
        Convenience method for searching PubMed.
        
        Args:
            query: The search query
            max_results: Maximum number of results to return
            use_history: Whether to use the Entrez History server
            sort: Sort order (e.g., 'relevance', 'pub_date')
            **kwargs: Additional parameters for the ESearch utility
            
        Returns:
            Optional[Dict]: Search results if successful, None otherwise
        """
        return self.esearch(
            db='pubmed',
            term=query,
            use_history=use_history,
            retmax=max_results,
            sort=sort,
            **kwargs
        )
    
    def fetch_pubmed_abstracts(self, 
                              id_list: Optional[List[str]] = None,
                              max_results: int = 20,
                              parse_xml: bool = True,
                              **kwargs) -> Optional[Union[List[Dict], str]]:
        """
        Fetch and parse PubMed abstracts.
        
        Args:
            id_list: List of PubMed IDs (alternative to using history server)
            max_results: Maximum number of abstracts to return
            parse_xml: Whether to parse the XML into a structured format
            **kwargs: Additional parameters for the EFetch utility
            
        Returns:
            Optional[Union[List[Dict], str]]: List of parsed abstracts or raw XML
        """
        # Fetch abstracts in XML format
        abstracts_xml = self.efetch(
            db='pubmed',
            id_list=id_list,
            rettype='abstract',
            retmode='xml',
            retmax=max_results,
            **kwargs
        )
        
        if not abstracts_xml:
            return None
            
        # Return raw XML if not parsing
        if not parse_xml:
            return abstracts_xml
            
        try:
            # Parse XML response
            root = ET.fromstring(abstracts_xml)
            
            # Extract and parse abstracts
            articles = root.findall('.//PubmedArticle')
            results = []
            
            for article in articles:
                # Extract PMID
                pmid_elem = article.find('.//PMID')
                pmid = pmid_elem.text if pmid_elem is not None else None
                
                # Extract title
                title_elem = article.find('.//ArticleTitle')
                title = title_elem.text if title_elem is not None else None
                
                # Extract abstract
                abstract_elems = article.findall('.//AbstractText')
                abstract = ' '.join([elem.text or '' for elem in abstract_elems if elem.text]) if abstract_elems else None
                
                # Extract mesh terms
                mesh_terms = []
                mesh_elems = article.findall('.//MeshHeading')
                for mesh_elem in mesh_elems:
                    descriptor = mesh_elem.find('./DescriptorName')
                    if descriptor is not None and descriptor.text:
                        mesh_terms.append(descriptor.text)
                
                # Extract authors
                author_elems = article.findall('.//Author')
                authors = []
                
                for author_elem in author_elems:
                    last_name = author_elem.find('./LastName')
                    fore_name = author_elem.find('./ForeName')
                    
                    if last_name is not None:
                        author_name = last_name.text
                        if fore_name is not None and fore_name.text:
                            author_name = f"{fore_name.text} {author_name}"
                        authors.append(author_name)
                
                # Extract publication date
                pub_date_elem = article.find('.//PubDate')
                pub_year = pub_date_elem.find('./Year') if pub_date_elem is not None else None
                pub_month = pub_date_elem.find('./Month') if pub_date_elem is not None else None
                pub_day = pub_date_elem.find('./Day') if pub_date_elem is not None else None
                
                pub_date = None
                if pub_year is not None and pub_year.text:
                    pub_date = pub_year.text
                    if pub_month is not None and pub_month.text:
                        pub_date = f"{pub_date}-{pub_month.text}"
                        if pub_day is not None and pub_day.text:
                            pub_date = f"{pub_date}-{pub_day.text}"
                
                # Extract journal info
                journal_elem = article.find('.//Journal')
                journal_title = None
                
                if journal_elem is not None:
                    journal_title_elem = journal_elem.find('./Title')
                    journal_title = journal_title_elem.text if journal_title_elem is not None else None
                
                # Extract DOI
                article_id_elements = article.findall('.//ArticleId')
                doi = None
                for id_elem in article_id_elements:
                    if id_elem.get('IdType') == 'doi':
                        doi = id_elem.text
                        break
                
                # Construct article dictionary
                article_dict = {
                    'pmid': pmid,
                    'title': title,
                    'abstract': abstract,
                    'authors': authors,
                    'publication_date': pub_date,
                    'journal': journal_title,
                    'mesh_terms': mesh_terms,
                    'doi': doi
                }
                
                results.append(article_dict)
            
            return results
            
        except ET.ParseError as e:
            logger.error(f"Failed to parse XML response: {str(e)}")
            return None
    
    def search_and_fetch_pubmed(self, 
                               query: str, 
                               max_results: int = 20,
                               parse_xml: bool = True) -> Optional[Union[List[Dict], str]]:
        """
        Convenience method to search PubMed and fetch abstracts in one step.
        
        Args:
            query: The search query
            max_results: Maximum number of results to return
            parse_xml: Whether to parse the XML into a structured format
            
        Returns:
            Optional[Union[List[Dict], str]]: List of parsed abstracts or raw XML
        """
        # Search PubMed
        search_results = self.search_pubmed(query, max_results=max_results, use_history=True)
        
        if not search_results or 'esearchresult' not in search_results:
            return None
            
        # Check if any results were found
        count = int(search_results['esearchresult'].get('count', 0))
        
        if count == 0:
            logger.info(f"No results found for query: {query}")
            return []
            
        # If using history server, fetch abstracts using WebEnv and query_key
        if self.web_env and self.query_key:
            return self.fetch_pubmed_abstracts(max_results=max_results, parse_xml=parse_xml)
        else:
            # Otherwise, extract PMIDs from search results and fetch directly
            id_list = search_results['esearchresult'].get('idlist', [])
            return self.fetch_pubmed_abstracts(id_list=id_list, max_results=max_results, parse_xml=parse_xml)
    
    def get_cited_by(self, 
                    pmid: str, 
                    max_results: int = 20) -> Optional[List[Dict]]:
        """
        Get articles that cite the specified PubMed article.
        
        Args:
            pmid: PubMed ID of the article
            max_results: Maximum number of results to return
            
        Returns:
            Optional[List[Dict]]: List of parsed abstracts that cite the article
        """
        # Link to cited-by articles
        link_results = self.elink(
            dbfrom='pubmed',
            db='pubmed',
            id_list=[pmid],
            linkname='pubmed_pubmed_citedin',
            cmd='neighbor_history'
        )
        
        if not link_results or not self.web_env or not self.query_key:
            return None
            
        # Fetch the cited-by articles
        return self.fetch_pubmed_abstracts(max_results=max_results)
    
    def get_similar_articles(self, 
                            pmid: str, 
                            max_results: int = 20) -> Optional[List[Dict]]:
        """
        Get articles similar to the specified PubMed article.
        
        Args:
            pmid: PubMed ID of the article
            max_results: Maximum number of results to return
            
        Returns:
            Optional[List[Dict]]: List of parsed abstracts similar to the article
        """
        # Link to similar articles
        link_results = self.elink(
            dbfrom='pubmed',
            db='pubmed',
            id_list=[pmid],
            linkname='pubmed_pubmed',
            cmd='neighbor_history'
        )
        
        if not link_results or not self.web_env or not self.query_key:
            return None
            
        # Fetch the similar articles
        return self.fetch_pubmed_abstracts(max_results=max_results)
    
    def search_gene(self, 
                   query: str, 
                   organism: Optional[str] = None,
                   max_results: int = 20) -> Optional[Dict]:
        """
        Search for genes in the Gene database.
        
        Args:
            query: The search query
            organism: Organism name or tax ID to restrict results
            max_results: Maximum number of results to return
            
        Returns:
            Optional[Dict]: Gene search results
        """
        # Construct the query
        if organism:
            query = f"{query} AND {organism}[Organism]"
            
        # Search the Gene database
        return self.esearch(
            db='gene',
            term=query,
            retmax=max_results,
            use_history=True
        )
    
    def fetch_gene_summaries(self, 
                            id_list: Optional[List[str]] = None,
                            max_results: int = 20) -> Optional[Dict]:
        """
        Fetch gene summaries from the Gene database.
        
        Args:
            id_list: List of Gene IDs (alternative to using history server)
            max_results: Maximum number of summaries to return
            
        Returns:
            Optional[Dict]: Gene summaries
        """
        # Fetch gene summaries
        return self.esummary(
            db='gene',
            id_list=id_list,
            retmax=max_results,
            version='2.0'  # Use version 2.0 for enhanced output
        )
    
    def search_cap_literature(self, 
                             specific_aspects: Optional[List[str]] = None,
                             years_range: Optional[Tuple[int, int]] = None,
                             max_results: int = 50) -> Optional[List[Dict]]:
        """
        Specialized method to search for Community-Acquired Pneumonia literature.
        
        Args:
            specific_aspects: List of specific aspects of CAP to focus on (e.g., ['treatment', 'diagnosis'])
            years_range: Tuple of (start_year, end_year) to limit the search
            max_results: Maximum number of results to return
            
        Returns:
            Optional[List[Dict]]: List of parsed abstracts if successful, None otherwise
        """
        # Construct the base query
        query = "(community acquired pneumonia[Title/Abstract] OR CAP[Title/Abstract])"
        
        # Add specific aspects if provided
        if specific_aspects:
            aspects_query = " OR ".join([f"{aspect}[Title/Abstract]" for aspect in specific_aspects])
            query = f"{query} AND ({aspects_query})"
        
        # Add year range if provided
        if years_range:
            start_year, end_year = years_range
            query = f"{query} AND ({start_year}:{end_year}[PDAT])"
        
        # Perform the search and fetch results
        return self.search_and_fetch_pubmed(query, max_results=max_results)


# Example usage
if __name__ == "__main__":
    # Initialize client
    client = NCBIClient()
    
    # Search for articles about COVID-19 treatment
    results = client.search_and_fetch_pubmed("COVID-19 treatment", max_results=5)
    
    # Display results
    if results:
        for i, article in enumerate(results, 1):
            print(f"\n{i}. {article['title']}")
            print(f"Authors: {', '.join(article['authors'])}")
            print(f"Journal: {article['journal']}, Date: {article['publication_date']}")
            if article['abstract']:
                print(f"Abstract snippet: {article['abstract'][:200]}...")
            else:
                print("No abstract available")
    else:
        print("No results found or error occurred")



# # Initialize client with your credentials
# client = NCBIClient(
#     api_key="your_api_key",
#     email="your_email@example.com",
#     tool="your_application_name"
# )

# # Search and fetch articles about COVID-19 treatment guidelines
# covid_articles = client.search_and_fetch_pubmed(
#     "COVID-19 treatment guidelines",
#     max_results=25
# )

# # Download a large dataset in batches
# for batch in client.download_large_dataset(
#     db="pubmed",
#     query="cancer immunotherapy clinical trial",
#     rettype="abstract",
#     batch_size=100,
#     max_records=1000
# ):
#     # Process each batch
#     print(f"Retrieved batch with {len(batch.split('<PubmedArticle>'))-1} articles")

# # Find articles that cite a specific paper
# citing_articles = client.get_cited_by("33306283")  # PMC ID for a COVID paper

# # Convert citation strings to PubMed IDs
# citations = [
#     {"journal": "Science", "year": "2020", "volume": "370", 
#      "first_page": "1022", "author": "Cohen", "key": "covid1"},
#     {"journal": "Nature", "year": "2021", "volume": "591", 
#      "first_page": "520", "author": "Planas", "key": "covid2"}
# ]
# pmids = client.ecitmatch(citation_list=citations)
