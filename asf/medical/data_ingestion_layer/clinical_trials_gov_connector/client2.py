import requests
import json
from typing import Dict, List, Optional, Any
import urllib.parse
import time

class ClinicalTrialsConnector:
    """
    A comprehensive connector for the ClinicalTrials.gov API Version 2.0
    
    API Base URL: https://clinicaltrials.gov
    Version: 2.0
    """
    
    BASE_URL = "https://clinicaltrials.gov/api/v2"
    
    def __init__(self, max_retries: int = 3, backoff_factor: float = 0.5):
        """
        Initialize the ClinicalTrials API connector
        
        :param max_retries: Maximum number of retry attempts for failed requests
        :param backoff_factor: Exponential backoff factor for retry attempts
        """
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.session = requests.Session()
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make a request to the ClinicalTrials.gov API with exponential backoff
        
        :param endpoint: API endpoint to query
        :param params: Optional query parameters
        :return: JSON response from the API
        """
        url = f"{self.BASE_URL}{endpoint}"
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, params=params)
                
                # Raise an exception for HTTP errors
                response.raise_for_status()
                
                # Parse and return JSON response
                return response.json()
            
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise
                
                # Exponential backoff
                wait_time = (2 ** attempt) * self.backoff_factor
                time.sleep(wait_time)
    
    def get_api_version(self) -> Dict[str, Any]:
        """
        Retrieve the current API version and data timestamp
        
        :return: Dictionary containing API version information
        """
        return self._make_request("/version")
    
    def search_studies(
        self, 
        query: Optional[str] = None, 
        page_size: int = 10, 
        page_token: Optional[str] = None,
        search_areas: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Search for clinical studies with advanced filtering options
        
        :param query: Search query string
        :param page_size: Number of results per page (max 1000)
        :param page_token: Token for pagination
        :param search_areas: List of specific search areas to query
        :return: Search results dictionary
        """
        params = {
            "pageSize": min(page_size, 1000)
        }
        
        # Add optional parameters
        if query:
            params["query"] = query
        if page_token:
            params["pageToken"] = page_token
        if search_areas:
            params["searchAreas"] = ",".join(search_areas)
        
        return self._make_request("/studies/search", params)
    
    def get_study_by_nct_id(self, nct_id: str, fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Retrieve a specific study by its NCT ID
        
        :param nct_id: National Clinical Trial identifier
        :param fields: Optional list of specific fields to retrieve
        :return: Detailed study information
        """
        endpoint = f"/studies/{nct_id}"
        params = {}
        
        if fields:
            params["fields"] = ",".join(fields)
        
        return self._make_request(endpoint, params)
    
    def get_study_metadata(self) -> Dict[str, Any]:
        """
        Retrieve metadata about the study data structure
        
        :return: Metadata describing study record fields and structure
        """
        return self._make_request("/studies/metadata")
    
    def get_search_areas(self) -> Dict[str, Any]:
        """
        Retrieve available search areas for querying studies
        
        :return: Dictionary of searchable fields and their properties
        """
        return self._make_request("/studies/search-areas")
    
    def get_statistical_field_values(self, field: str) -> Dict[str, Any]:
        """
        Retrieve statistical information about distinct values for a specific field
        
        :param field: Field to retrieve statistical values for
        :return: Statistical field values
        """
        endpoint = f"/stats/field/values/{urllib.parse.quote(field)}"
        return self._make_request(endpoint)
    
    def construct_complex_query(
        self, 
        conditions: Optional[List[Dict[str, Any]]] = None,
        operators: Optional[List[str]] = None
    ) -> str:
        """
        Construct a complex search query using advanced operators
        
        :param conditions: List of query conditions
        :param operators: List of query operators
        :return: Constructed query string
        """
        # Example of advanced query construction
        # Supports AREA, RANGE, COVERAGE, and EXPANSION operators
        if not conditions:
            return ""
        
        query_parts = []
        for condition in conditions:
            # Basic structure: AREA(field, value) OPERATOR
            query_part = f"AREA({condition['field']}, {condition['value']})"
            
            # Add optional operators
            if 'operator' in condition:
                query_part += f" {condition['operator']}"
            
            query_parts.append(query_part)
        
        return " AND ".join(query_parts)

# Example usage
def main():
    # Initialize the connector
    connector = ClinicalTrialsConnector()
    
    # Get API version
    version_info = connector.get_api_version()
    print("API Version:", version_info)
    
    # Example: Search for COVID-19 vaccine trials
    search_results = connector.search_studies(
        query="COVID-19 vaccine",
        page_size=20,
        search_areas=["Condition", "Intervention"]
    )
    print("Search Results:", json.dumps(search_results, indent=2))
    
    # Example: Get metadata
    metadata = connector.get_study_metadata()
    print("Study Metadata:", json.dumps(metadata, indent=2))

if __name__ == "__main__":
    main()