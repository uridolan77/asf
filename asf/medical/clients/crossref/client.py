"""
Core CrossRef client implementation for the Medical Research Synthesizer.
This module provides a comprehensive client for interacting with the CrossRef API.
"""
import logging
import time
from typing import Dict, List, Optional, Any, Union
from crossrefapi import Works, Journals, Members, Types, Prefixes, Funders

from .works import WorksService
from .journals import JournalsService
from .utils import retry_function
from .exceptions import CrossRefError, CrossRefAPIError, CrossRefRateLimitError

logger = logging.getLogger(__name__)

class CrossRefClient:
    """
    Client for interacting with the CrossRef API.
    This client provides methods for retrieving citation data for articles,
    journal information, publisher information, and more.
    
    The client uses the crossrefapi package to interact with the CrossRef API
    and provides a convenient interface for common use cases.
    """
    
    def __init__(
        self,
        email: str = None,
        plus_api_token: str = None,
        base_url: str = "https://api.crossref.org"
    ):
        """
        Initialize the CrossRef client.
        
        Args:
            email: Email address to use in the User-Agent header (polite pool)
            plus_api_token: CrossRef Plus API token for higher rate limits (if available)
            base_url: Base URL for the CrossRef API
        """
        self.email = email
        self.plus_api_token = plus_api_token
        self.base_url = base_url
        self.user_agent = f"Medical-Research-Synthesizer/1.0 (mailto:{email})" if email else None
        
        # Initialize API clients
        self.works_api = Works(base_url=base_url, mailto=email, token=plus_api_token)
        self.journals_api = Journals(base_url=base_url, mailto=email, token=plus_api_token)
        self.members_api = Members(base_url=base_url, mailto=email, token=plus_api_token)
        self.types_api = Types(base_url=base_url, mailto=email, token=plus_api_token)
        self.prefixes_api = Prefixes(base_url=base_url, mailto=email, token=plus_api_token)
        self.funders_api = Funders(base_url=base_url, mailto=email, token=plus_api_token)
        
        # Rate limiting properties
        self.requests_per_second = 2  # Standard rate limit
        self.last_request_time = 0
        
        # Initialize specialized services
        self.works = WorksService(self)
        self.journals = JournalsService(self)
    
    def _implement_rate_limiting(self) -> None:
        """
        Implement rate limiting for CrossRef API.
        CrossRef recommends no more than 2 requests per second.
        """
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < (1.0 / self.requests_per_second):
            sleep_time = (1.0 / self.requests_per_second) - time_since_last_request
            logger.debug(f"Rate limiting: Sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    @retry_function
    def get_work_by_doi(self, doi: str) -> Optional[Dict]:
        """
        Get article metadata by DOI.
        
        Args:
            doi: DOI (e.g., "10.1056/NEJMoa2001017")
            
        Returns:
            Dict: Article metadata or None if not found
        """
        self._implement_rate_limiting()
        try:
            result = self.works_api.doi(doi)
            return result
        except Exception as e:
            logger.error(f"Error retrieving work by DOI {doi}: {str(e)}")
            return None
    
    @retry_function
    def search_works(
        self, 
        query: str = None, 
        max_results: int = 20,
        filter: Dict = None,
        sort: str = None,
        order: str = None,
        offset: int = 0,
        **kwargs
    ) -> List[Dict]:
        """
        Search for articles.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return (default: 20)
            filter: Filters to apply (e.g., {"type": "journal-article"})
            sort: Sort field (e.g., "score", "updated", "deposited", "indexed", "published")
            order: Sort order (e.g., "asc", "desc")
            offset: Result offset (for pagination)
            **kwargs: Additional parameters to pass to the CrossRef API
            
        Returns:
            List[Dict]: List of article summaries
        """
        self._implement_rate_limiting()
        
        try:
            query_args = {}
            if query:
                query_args['query'] = query
            if filter:
                for k, v in filter.items():
                    query_args[f'filter.{k}'] = v
            if sort:
                query_args['sort'] = sort
            if order:
                query_args['order'] = order
            
            # Add any additional parameters
            query_args.update(kwargs)
            
            # Set up pagination
            query_args['rows'] = min(max_results, 1000)  # CrossRef API has a limit of 1000 results per page
            query_args['offset'] = offset
            
            # Execute the query
            results = self.works_api.query(**query_args)
            
            # Collect the results
            items = []
            for idx, item in enumerate(results):
                if idx >= max_results:
                    break
                items.append(item)
            
            return items
        except Exception as e:
            logger.error(f"Error searching works: {str(e)}")
            return []
    
    @retry_function
    def get_journal_by_issn(self, issn: str) -> Optional[Dict]:
        """
        Get journal metadata by ISSN.
        
        Args:
            issn: ISSN (e.g., "1476-4687")
            
        Returns:
            Dict: Journal metadata or None if not found
        """
        self._implement_rate_limiting()
        try:
            result = self.journals_api.journal(issn)
            return result
        except Exception as e:
            logger.error(f"Error retrieving journal by ISSN {issn}: {str(e)}")
            return None
    
    @retry_function
    def search_journals(
        self, 
        query: str = None, 
        max_results: int = 20,
        **kwargs
    ) -> List[Dict]:
        """
        Search for journals.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return (default: 20)
            **kwargs: Additional parameters to pass to the CrossRef API
            
        Returns:
            List[Dict]: List of journal summaries
        """
        self._implement_rate_limiting()
        
        try:
            # Execute the query
            results = self.journals_api.query(query, **kwargs)
            
            # Collect the results
            items = []
            for idx, item in enumerate(results):
                if idx >= max_results:
                    break
                items.append(item)
            
            return items
        except Exception as e:
            logger.error(f"Error searching journals: {str(e)}")
            return []
    
    @retry_function
    def get_member_by_id(self, member_id: Union[str, int]) -> Optional[Dict]:
        """
        Get publisher information by member ID.
        
        Args:
            member_id: CrossRef member ID (e.g., "311")
            
        Returns:
            Dict: Publisher information or None if not found
        """
        self._implement_rate_limiting()
        try:
            result = self.members_api.member(member_id)
            return result
        except Exception as e:
            logger.error(f"Error retrieving member by ID {member_id}: {str(e)}")
            return None
    
    @retry_function
    def get_funder_by_id(self, funder_id: str) -> Optional[Dict]:
        """
        Get funder information by ID.
        
        Args:
            funder_id: Funder ID (e.g., "10.13039/100000001" for NSF)
            
        Returns:
            Dict: Funder information or None if not found
        """
        self._implement_rate_limiting()
        try:
            result = self.funders_api.funder(funder_id)
            return result
        except Exception as e:
            logger.error(f"Error retrieving funder by ID {funder_id}: {str(e)}")
            return None
    
    @retry_function
    def search_funders(
        self, 
        query: str = None, 
        max_results: int = 20,
        **kwargs
    ) -> List[Dict]:
        """
        Search for funding organizations.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return (default: 20)
            **kwargs: Additional parameters to pass to the CrossRef API
            
        Returns:
            List[Dict]: List of funder summaries
        """
        self._implement_rate_limiting()
        
        try:
            # Execute the query
            results = self.funders_api.query(query, **kwargs)
            
            # Collect the results
            items = []
            for idx, item in enumerate(results):
                if idx >= max_results:
                    break
                items.append(item)
            
            return items
        except Exception as e:
            logger.error(f"Error searching funders: {str(e)}")
            return []
    
    @retry_function
    def get_citation_count(self, doi: str) -> int:
        """
        Get the number of citations for a work identified by its DOI.
        
        Args:
            doi: DOI of the work to check
            
        Returns:
            int: Number of citations or 0 if not found/error
        """
        self._implement_rate_limiting()
        try:
            # Use the 'is-referenced-by-count' field
            work = self.works_api.doi(doi)
            if work and 'is-referenced-by-count' in work:
                return work['is-referenced-by-count']
            return 0
        except Exception as e:
            logger.error(f"Error retrieving citation count for DOI {doi}: {str(e)}")
            return 0
    
    @retry_function
    def get_references(self, doi: str) -> List[Dict]:
        """
        Get the references cited by a work identified by its DOI.
        
        Args:
            doi: DOI of the work to check
            
        Returns:
            List[Dict]: List of references or empty list if not found/error
        """
        self._implement_rate_limiting()
        try:
            work = self.works_api.doi(doi)
            if work and 'reference' in work:
                return work['reference']
            return []
        except Exception as e:
            logger.error(f"Error retrieving references for DOI {doi}: {str(e)}")
            return []
            
    def get_all_types(self) -> List[Dict]:
        """
        Get all work types from the CrossRef API.
        
        Returns:
            List[Dict]: List of work types
        """
        self._implement_rate_limiting()
        try:
            types = list(self.types_api.all())
            return types
        except Exception as e:
            logger.error(f"Error retrieving work types: {str(e)}")
            return []
            
    def get_prefix_info(self, prefix: str) -> Optional[Dict]:
        """
        Get information about a DOI prefix.
        
        Args:
            prefix: DOI prefix (e.g., "10.1016")
            
        Returns:
            Dict: Prefix information or None if not found
        """
        self._implement_rate_limiting()
        try:
            result = self.prefixes_api.prefix(prefix)
            return result
        except Exception as e:
            logger.error(f"Error retrieving prefix info for {prefix}: {str(e)}")
            return None