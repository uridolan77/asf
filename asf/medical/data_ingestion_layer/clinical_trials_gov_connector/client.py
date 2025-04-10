"""
ClinicalTrials.gov API Client

This module provides a comprehensive client for the ClinicalTrials.gov API v2.
"""

import requests
import json
import time
import logging
from typing import Dict, List, Optional, Any, Union
import urllib.parse

# Optional pandas support
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# Set up logging
logger = logging.getLogger(__name__)

class ClinicalTrialsClient:
    """
    A comprehensive connector for the ClinicalTrials.gov API Version 2.0
    
    API Base URL: https://clinicaltrials.gov
    Version: 2.0
    
    Documentation: https://clinicaltrials.gov/data-api/api
    """
    
    BASE_URL = "https://clinicaltrials.gov/api/v2"
    DEFAULT_PAGE_SIZE = 100  # API default is 100, max is 1000
    MAX_PAGE_SIZE = 1000
    DEFAULT_TIMEOUT = 30  # seconds
    
    def __init__(
        self, 
        max_retries: int = 3, 
        backoff_factor: float = 0.5,
        timeout: int = DEFAULT_TIMEOUT
    ):
        """
        Initialize the ClinicalTrials API connector
        
        Args:
            max_retries: Maximum number of retry attempts for failed requests
            backoff_factor: Exponential backoff factor for retry attempts
            timeout: Request timeout in seconds
        """
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
        """
        Search for studies using the ClinicalTrials.gov API
        
        Args:
            query: Search query string
            fields: List of fields to include in the response
            page_size: Number of results per page
            page: Page number
            format: Response format (json or csv)
            search_areas: Areas to search in (e.g., "Condition", "Intervention")
            min_rank: Minimum rank for results
            max_rank: Maximum rank for results
            country_codes: List of country codes to filter by
            status: List of study statuses to filter by
            study_type: List of study types to filter by
            phase: List of study phases to filter by
            to_dataframe: Whether to convert results to a pandas DataFrame
            
        Returns:
            Dict containing search results or pandas DataFrame if to_dataframe=True
        """
        # Build query parameters
        params = {
            "query": query,
            "pageSize": min(page_size, self.MAX_PAGE_SIZE),
            "page": page,
            "format": format
        }
        
        # Add optional parameters
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
        
        # Make request
        results = self._make_request("GET", "/studies", params=params)
        
        # Convert to DataFrame if requested
        if to_dataframe and HAS_PANDAS:
            if "studies" in results and results["studies"]:
                return pd.DataFrame(results["studies"])
            else:
                return pd.DataFrame()
        
        return results
    
    def get_study(
        self, 
        nct_id: str,
        fields: List[str] = None,
        format: str = "json"
    ) -> Dict[str, Any]:
        """
        Get details for a specific study by NCT ID
        
        Args:
            nct_id: The NCT ID of the study
            fields: List of fields to include in the response
            format: Response format (json or csv)
            
        Returns:
            Dict containing study details
        """
        # Build query parameters
        params = {"format": format}
        
        # Add optional parameters
        if fields:
            params["fields"] = ",".join(fields)
        
        # Make request
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
        """
        Build an advanced query string for the ClinicalTrials.gov API
        
        Args:
            condition: Medical condition
            intervention: Treatment or intervention
            title: Words in the study title
            outcome: Outcome measures
            sponsor: Study sponsor
            location: Study location
            status: Study status (e.g., "Recruiting", "Completed")
            phase: Study phase (e.g., "Phase 1", "Phase 2")
            study_type: Type of study (e.g., "Interventional", "Observational")
            gender: Participant gender ("Male", "Female", "All")
            min_age: Minimum participant age in years
            max_age: Maximum participant age in years
            start_date: Study start date (YYYY-MM-DD)
            completion_date: Study completion date (YYYY-MM-DD)
            
        Returns:
            Advanced query string
        """
        query_parts = []
        
        # Add condition
        if condition:
            query_parts.append(f"CONDITION:{condition}")
        
        # Add intervention
        if intervention:
            query_parts.append(f"INTERVENTION:{intervention}")
        
        # Add title
        if title:
            query_parts.append(f"TITLE:{title}")
        
        # Add outcome
        if outcome:
            query_parts.append(f"OUTCOME:{outcome}")
        
        # Add sponsor
        if sponsor:
            query_parts.append(f"SPONSOR:{sponsor}")
        
        # Add location
        if location:
            query_parts.append(f"LOCATION:{location}")
        
        # Add status
        if status:
            status_query = " OR ".join([f"STATUS:{s}" for s in status])
            query_parts.append(f"({status_query})")
        
        # Add phase
        if phase:
            phase_query = " OR ".join([f"PHASE:{p}" for p in phase])
            query_parts.append(f"({phase_query})")
        
        # Add study type
        if study_type:
            type_query = " OR ".join([f"STUDY_TYPE:{t}" for t in study_type])
            query_parts.append(f"({type_query})")
        
        # Add gender
        if gender:
            query_parts.append(f"GENDER:{gender}")
        
        # Add age range
        if min_age is not None:
            query_parts.append(f"MIN_AGE:{min_age}")
        
        if max_age is not None:
            query_parts.append(f"MAX_AGE:{max_age}")
        
        # Add dates
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
        """
        Make a request to the ClinicalTrials.gov API
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Query parameters
            data: Request body data
            
        Returns:
            Response data as a dictionary
        """
        url = f"{self.BASE_URL}{endpoint}"
        
        # Initialize retry counter
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
                
                # Check for successful response
                response.raise_for_status()
                
                # Parse response
                if response.content:
                    return response.json()
                else:
                    return {}
                
            except requests.exceptions.RequestException as e:
                retries += 1
                
                # If we've reached max retries, raise the exception
                if retries > self.max_retries:
                    logger.error(f"Request failed after {self.max_retries} retries: {e}")
                    raise
                
                # Calculate backoff time
                backoff_time = self.backoff_factor * (2 ** (retries - 1))
                logger.warning(f"Request failed, retrying in {backoff_time:.2f} seconds: {e}")
                
                # Wait before retrying
                time.sleep(backoff_time)
