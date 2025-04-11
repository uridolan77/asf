ClinicalTrials.gov API Client

This module provides a comprehensive client for the ClinicalTrials.gov API v2.

import requests
import time
import logging
from typing import Dict, List, Optional, Any, Union

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

logger = logging.getLogger(__name__)

class ClinicalTrialsClient:
    A comprehensive connector for the ClinicalTrials.gov API Version 2.0
    
    API Base URL: https://clinicaltrials.gov
    Version: 2.0
    
    Documentation: https://clinicaltrials.gov/data-api/api
    
    BASE_URL = "https://clinicaltrials.gov/api/v2"
    DEFAULT_PAGE_SIZE = 100  # API default is 100, max is 1000
    MAX_PAGE_SIZE = 1000
    DEFAULT_TIMEOUT = 30  # seconds
    
    def __init__(
        self, 
        max_retries: int = 3, 
            """
            __init__ function.
            
            This function provides functionality for...
            Args:
                max_retries: Description of max_retries
                backoff_factor: Description of backoff_factor
                timeout: Description of timeout
            """
        backoff_factor: float = 0.5,
        timeout: int = DEFAULT_TIMEOUT
    ):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.timeout = timeout
        self.session = requests.Session()
    
    def get_api_version(self) -> Dict[str, Any]:
        """
        Get the API version information
        
        Returns:
            Dict containing API version information
        """
        return self._make_request("GET", "/version")
    
    def search_studies(
        self,
        query: str = "",
            """
            search_studies function.
            
            This function provides functionality for...
            Args:
                query: Description of query
                fields: Description of fields
                page_size: Description of page_size
                page: Description of page
                format: Description of format
                search_areas: Description of search_areas
                min_rank: Description of min_rank
                max_rank: Description of max_rank
                country_codes: Description of country_codes
                status: Description of status
                study_type: Description of study_type
                phase: Description of phase
                to_dataframe: Description of to_dataframe
            
            Returns:
                Description of return value
            """
        fields: List[str] = None,
        page_size: int = DEFAULT_PAGE_SIZE,
        page: int = 1,
        format: str = "json",
        search_areas: List[str] = None,
        min_rank: Optional[int] = None,
        max_rank: Optional[int] = None,
        country_codes: List[str] = None,
        status: List[str] = None,
        study_type: List[str] = None,
        phase: List[str] = None,
        to_dataframe: bool = False
    ) -> Union[Dict[str, Any], 'pd.DataFrame']:
        params = {
            "query": query,
            "pageSize": min(page_size, self.MAX_PAGE_SIZE),
            "page": page,
            "format": format
        }
        
        if fields:
            params["fields"] = ",".join(fields)
        
        if search_areas:
            params["searchAreas"] = ",".join(search_areas)
        
        if min_rank is not None:
            params["minRank"] = min_rank
            
        if max_rank is not None:
            params["maxRank"] = max_rank
            
        if country_codes:
            params["countryCode"] = ",".join(country_codes)
            
        if status:
            params["status"] = ",".join(status)
            
        if study_type:
            params["studyType"] = ",".join(study_type)
            
        if phase:
            params["phase"] = ",".join(phase)
        
        results = self._make_request("GET", "/studies", params=params)
        
        if to_dataframe and HAS_PANDAS:
            if "studies" in results and results["studies"]:
                return pd.DataFrame(results["studies"])
            else:
                return pd.DataFrame()
        
        return results
    
    def get_study(
        self, 
        nct_id: str,
            """
            get_study function.
            
            This function provides functionality for...
            Args:
                nct_id: Description of nct_id
                fields: Description of fields
                format: Description of format
            
            Returns:
                Description of return value
            """
        fields: List[str] = None,
        format: str = "json"
    ) -> Dict[str, Any]:
        params = {"format": format}
        
        if fields:
            params["fields"] = ",".join(fields)
        
        return self._make_request("GET", f"/studies/{nct_id}", params=params)
    
    def get_study_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about study fields
        
        Returns:
            Dict containing study metadata
        """
        return self._make_request("GET", "/studies/metadata")
    
    def get_field_values(self, field: str) -> Dict[str, Any]:
        """
        Get possible values for a specific field
        
        Args:
            field: The field to get values for
            
        Returns:
            Dict containing field values
        """
        return self._make_request("GET", f"/studies/field_values/{field}")
    
    def build_advanced_query(
        self,
        condition: Optional[str] = None,
            """
            build_advanced_query function.
            
            This function provides functionality for...
            Args:
                condition: Description of condition
                intervention: Description of intervention
                title: Description of title
                outcome: Description of outcome
                sponsor: Description of sponsor
                location: Description of location
                status: Description of status
                phase: Description of phase
                study_type: Description of study_type
                gender: Description of gender
                min_age: Description of min_age
                max_age: Description of max_age
                start_date: Description of start_date
                completion_date: Description of completion_date
            
            Returns:
                Description of return value
            """
        intervention: Optional[str] = None,
        title: Optional[str] = None,
        outcome: Optional[str] = None,
        sponsor: Optional[str] = None,
        location: Optional[str] = None,
        status: Optional[List[str]] = None,
        phase: Optional[List[str]] = None,
        study_type: Optional[List[str]] = None,
        gender: Optional[str] = None,
        min_age: Optional[int] = None,
        max_age: Optional[int] = None,
        start_date: Optional[str] = None,
        completion_date: Optional[str] = None
    ) -> str:
        query_parts = []
        
        if condition:
            query_parts.append(f"CONDITION:{condition}")
        
        if intervention:
            query_parts.append(f"INTERVENTION:{intervention}")
        
        if title:
            query_parts.append(f"TITLE:{title}")
        
        if outcome:
            query_parts.append(f"OUTCOME:{outcome}")
        
        if sponsor:
            query_parts.append(f"SPONSOR:{sponsor}")
        
        if location:
            query_parts.append(f"LOCATION:{location}")
        
        if status:
            status_query = " OR ".join([f"STATUS:{s}" for s in status])
            query_parts.append(f"({status_query})")
        
        if phase:
            phase_query = " OR ".join([f"PHASE:{p}" for p in phase])
            query_parts.append(f"({phase_query})")
        
        if study_type:
            type_query = " OR ".join([f"STUDY_TYPE:{t}" for t in study_type])
            query_parts.append(f"({type_query})")
        
        if gender:
            query_parts.append(f"GENDER:{gender}")
        
        if min_age is not None:
            query_parts.append(f"MIN_AGE:{min_age}")
        
        if max_age is not None:
            query_parts.append(f"MAX_AGE:{max_age}")
        
        if start_date:
            query_parts.append(f"START_DATE:{start_date}")
        
        if completion_date:
            query_parts.append(f"COMPLETION_DATE:{completion_date}")
        
        return " AND ".join(query_parts)
    
    def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        params: Dict[str, Any] = None,
        data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        url = f"{self.BASE_URL}{endpoint}"
        
        retries = 0
        
        while retries <= self.max_retries:
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=data,
                    timeout=self.timeout
                )
                
                response.raise_for_status()
                
                if response.content:
                    return response.json()
                else:
                    return {}
                
            except requests.exceptions.RequestException as e:
                retries += 1
                
                if retries > self.max_retries:
                    logger.error(f"Request failed after {self.max_retries} retries: {e}")
                    raise
                
                backoff_time = self.backoff_factor * (2 ** (retries - 1))
                logger.warning(f"Request failed, retrying in {backoff_time:.2f} seconds: {e}")
                
                time.sleep(backoff_time)
