#!/usr/bin/env python3
"""
ClinicalTrials.gov API Client

A comprehensive client for interacting with the ClinicalTrials.gov Data API.
This client supports all major endpoints and includes features like pagination,
caching, error handling, and data transformation.

Official API documentation: https://clinicaltrials.gov/data-api/api-docs
"""

import os
import json
import time
import hashlib
import requests
from typing import Dict, List, Optional, Union, Any, Iterator
from functools import lru_cache
from datetime import datetime, timedelta
import logging
import sqlite3
from pathlib import Path
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("clinicaltrials")

# Constants
API_BASE_URL = "https://clinicaltrials.gov/api"
DEFAULT_CACHE_TTL = 86400  # 24 hours in seconds
DEFAULT_CACHE_SIZE = 1000  # Number of items to cache in memory


class CtCache:
    """Cache for ClinicalTrials.gov API data to improve performance."""
    
    def __init__(self, cache_dir: Optional[str] = None, ttl: int = DEFAULT_CACHE_TTL, 
                 memory_size: int = DEFAULT_CACHE_SIZE):
        """
        Initialize the cache.
        
        Args:
            cache_dir: Directory to store persistent cache. If None, only memory cache is used.
            ttl: Time-to-live for cached items in seconds. Default is 24 hours.
            memory_size: Number of items to cache in memory.
        """
        self.ttl = ttl
        self.memory_size = memory_size
        self.memory_cache = {}
        self.disk_cache_enabled = cache_dir is not None
        
        if self.disk_cache_enabled:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize SQLite cache if disk cache is enabled
            self.db_path = self.cache_dir / "clinicaltrials_cache.db"
            self._init_db()
    
    def _init_db(self) -> None:
        """Initialize the SQLite database for persistent caching."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create cache table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS api_cache (
            key TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL
        )
        ''')
        
        # Create index on timestamp for cleanup
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON api_cache (created_at)')
        
        conn.commit()
        conn.close()
    
    def _generate_key(self, prefix: str, *args) -> str:
        """Generate a unique cache key based on the function arguments."""
        key_str = prefix + ":" + json.dumps(args, sort_keys=True)
        return hashlib.md5(key_str.encode('utf-8')).hexdigest()
    
    def get(self, prefix: str, *args) -> Optional[Dict]:
        """
        Get an item from the cache.
        
        Args:
            prefix: A string prefix to categorize the cache entry
            *args: Arguments to generate the unique key
            
        Returns:
            The cached data or None if not found or expired
        """
        key = self._generate_key(prefix, *args)
        
        # Check memory cache first
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            if time.time() - entry["timestamp"] < self.ttl:
                return entry["data"]
            else:
                # Remove expired item from memory cache
                del self.memory_cache[key]
        
        # If not in memory and disk cache is enabled, check disk
        if self.disk_cache_enabled:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get entry and check if it's expired
            cursor.execute(
                "SELECT data, created_at FROM api_cache WHERE key = ?", 
                (key,)
            )
            result = cursor.fetchone()
            
            if result:
                data_json, created_at = result
                created_timestamp = datetime.fromisoformat(created_at).timestamp()
                
                if time.time() - created_timestamp < self.ttl:
                    data = json.loads(data_json)
                    
                    # Add to memory cache for faster access next time
                    self._add_to_memory_cache(key, data)
                    
                    conn.close()
                    return data
                else:
                    # Remove expired entry
                    cursor.execute("DELETE FROM api_cache WHERE key = ?", (key,))
                    conn.commit()
            
            conn.close()
        
        return None
    
    def _add_to_memory_cache(self, key: str, data: Dict) -> None:
        """Add an item to the memory cache, managing cache size."""
        if len(self.memory_cache) >= self.memory_size:
            # Simple strategy: remove oldest item
            oldest_key = min(self.memory_cache.keys(), 
                            key=lambda k: self.memory_cache[k]["timestamp"])
            del self.memory_cache[oldest_key]
        
        self.memory_cache[key] = {
            "data": data,
            "timestamp": time.time()
        }
    
    def set(self, prefix: str, data: Dict, *args) -> None:
        """
        Store an item in the cache.
        
        Args:
            prefix: A string prefix to categorize the cache entry
            data: The data to cache
            *args: Arguments to generate the unique key
        """
        key = self._generate_key(prefix, *args)
        
        # Add to memory cache
        self._add_to_memory_cache(key, data)
        
        # If disk cache is enabled, store there too
        if self.disk_cache_enabled:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Store the data with the current timestamp
            now = datetime.now().isoformat()
            cursor.execute(
                "INSERT OR REPLACE INTO api_cache (key, data, created_at) VALUES (?, ?, ?)",
                (key, json.dumps(data), now)
            )
            
            conn.commit()
            conn.close()
    
    def clear_expired(self) -> int:
        """
        Clear expired items from the cache.
        
        Returns:
            Number of items removed
        """
        # Clear expired items from memory cache
        now = time.time()
        expired_keys = [k for k, v in self.memory_cache.items() 
                       if now - v["timestamp"] >= self.ttl]
        
        for key in expired_keys:
            del self.memory_cache[key]
        
        memory_cleared = len(expired_keys)
        
        # Clear expired items from disk cache if enabled
        disk_cleared = 0
        if self.disk_cache_enabled:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Calculate the expiration timestamp
            expiration_time = (datetime.now() - timedelta(seconds=self.ttl)).isoformat()
            
            # Delete expired entries
            cursor.execute(
                "DELETE FROM api_cache WHERE created_at < ?", 
                (expiration_time,)
            )
            
            disk_cleared = cursor.rowcount
            conn.commit()
            conn.close()
        
        return memory_cleared + disk_cleared
    
    def clear_all(self) -> int:
        """
        Clear all items from the cache.
        
        Returns:
            Number of items removed
        """
        memory_cleared = len(self.memory_cache)
        self.memory_cache = {}
        
        disk_cleared = 0
        if self.disk_cache_enabled:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM api_cache")
            disk_cleared = cursor.rowcount
            
            conn.commit()
            conn.close()
        
        return memory_cleared + disk_cleared


class ClinicalTrialsClient:
    """
    Client for accessing the ClinicalTrials.gov Data API.
    
    This class provides methods for accessing various endpoints of the
    ClinicalTrials.gov Data API, including studies, conditions, interventions,
    sponsors, and more.
    """
    
    def __init__(self, api_key: Optional[str] = None, 
                 base_url: str = API_BASE_URL,
                 cache_dir: Optional[str] = None,
                 use_cache: bool = True):
        """
        Initialize the ClinicalTrials.gov API client.
        
        Args:
            api_key: API key for authentication (if required)
            base_url: Base URL for the API
            cache_dir: Directory to store cache data
            use_cache: Whether to use caching
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.use_cache = use_cache
        
        # Set up caching if enabled
        self.cache = None
        if use_cache:
            cache_ttl = int(os.environ.get("CT_CACHE_TTL", DEFAULT_CACHE_TTL))
            self.cache = CtCache(cache_dir=cache_dir, ttl=cache_ttl)
        
        # Session for connection pooling
        self.session = requests.Session()
        
        # Set default headers
        self.headers = {
            "Accept": "application/json",
        }
        
        # Add API key to headers if provided
        if self.api_key:
            self.headers["api_key"] = self.api_key
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None, 
                     method: str = "GET") -> Dict:
        """
        Make a request to the ClinicalTrials.gov API.
        
        Args:
            endpoint: API endpoint to call
            params: Optional query parameters
            method: HTTP method (GET, POST, etc.)
            
        Returns:
            JSON response as dictionary
        """
        if params is None:
            params = {}
        
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        # Check cache first if using cache and it's a GET request
        if self.use_cache and method.upper() == "GET":
            cached_data = self.cache.get("request", endpoint, params)
            if cached_data:
                return cached_data
        
        try:
            # Make the request
            if method.upper() == "GET":
                response = self.session.get(url, headers=self.headers, params=params)
            elif method.upper() == "POST":
                response = self.session.post(url, headers=self.headers, json=params)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            data = response.json()
            
            # Cache the response if using cache and it's a GET request
            if self.use_cache and method.upper() == "GET":
                self.cache.set("request", data, endpoint, params)
            
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            
            # Check if we got a response with error details
            response_text = getattr(e.response, 'text', None)
            if response_text:
                try:
                    error_data = json.loads(response_text)
                    logger.error(f"API error details: {error_data}")
                except json.JSONDecodeError:
                    logger.error(f"API error response: {response_text}")
            
            # Raise a more informative error
            raise RuntimeError(f"ClinicalTrials.gov API request failed: {e}") from e
    
    def get_study(self, nct_id: str) -> Dict:
        """
        Get detailed information about a specific study by NCT ID.
        
        Args:
            nct_id: The ClinicalTrials.gov identifier (NCT number)
            
        Returns:
            Study details as dictionary
        """
        endpoint = f"v2/studies/{nct_id}"
        return self._make_request(endpoint)
    
    def search_studies(self, query: Optional[str] = None, 
                      fields: Optional[List[str]] = None,
                      min_rank: int = 1, max_rank: int = 10, 
                      **kwargs) -> Dict:
        """
        Search for studies based on criteria.
        
        Args:
            query: Search query string
            fields: List of fields to include in response
            min_rank: Starting rank for paginated results
            max_rank: Ending rank for paginated results
            **kwargs: Additional search parameters
            
        Returns:
            Search results as dictionary
        """
        endpoint = "v2/studies"
        
        # Build parameters
        params = {
            "minRank": min_rank,
            "maxRank": max_rank
        }
        
        if query:
            params["query"] = query
        
        if fields:
            params["fields"] = ",".join(fields)
        
        # Add additional parameters
        params.update(kwargs)
        
        return self._make_request(endpoint, params=params)
    
    def search_studies_iterator(self, query: Optional[str] = None, 
                               fields: Optional[List[str]] = None,
                               page_size: int = 100, max_pages: Optional[int] = None,
                               **kwargs) -> Iterator[Dict]:
        """
        Search for studies and iterate through all pages of results.
        
        Args:
            query: Search query string
            fields: List of fields to include in response
            page_size: Number of results per page
            max_pages: Maximum number of pages to retrieve (None for all)
            **kwargs: Additional search parameters
            
        Returns:
            Iterator over study records
        """
        min_rank = 1
        page = 0
        
        while True:
            max_rank = min_rank + page_size - 1
            
            # If max_pages is set and we've reached it, stop
            if max_pages and page >= max_pages:
                break
            
            # Get the current page of results
            results = self.search_studies(
                query=query,
                fields=fields,
                min_rank=min_rank,
                max_rank=max_rank,
                **kwargs
            )
            
            # Extract studies from the response
            studies = results.get("studies", [])
            
            # If no studies were returned, we've reached the end
            if not studies:
                break
            
            # Yield each study
            for study in studies:
                yield study
            
            # Update for next page
            min_rank += page_size
            page += 1
    
    def get_full_studies(self, expression: str, min_rank: int = 1, max_rank: int = 10) -> Dict:
        """
        Get full study information based on a search expression.
        
        Args:
            expression: Search expression (same format as website search)
            min_rank: Starting rank for paginated results
            max_rank: Ending rank for paginated results
            
        Returns:
            Full study details as dictionary
        """
        endpoint = f"query/full_studies"
        
        params = {
            "expr": expression,
            "min_rnk": min_rank,
            "max_rnk": max_rank,
            "fmt": "json"
        }
        
        return self._make_request(endpoint, params=params)
    
    def get_study_fields(self, expression: str, fields: List[str], 
                        min_rank: int = 1, max_rank: int = 10) -> Dict:
        """
        Get specific fields for studies matching an expression.
        
        Args:
            expression: Search expression
            fields: List of fields to retrieve
            min_rank: Starting rank for paginated results
            max_rank: Ending rank for paginated results
            
        Returns:
            Study fields as dictionary
        """
        endpoint = f"query/study_fields"
        
        params = {
            "expr": expression,
            "fields": ",".join(fields),
            "min_rnk": min_rank,
            "max_rnk": max_rank,
            "fmt": "json"
        }
        
        return self._make_request(endpoint, params=params)
    
    def get_study_statistics(self, **kwargs) -> Dict:
        """
        Get aggregate statistics about studies in the database.
        
        Args:
            **kwargs: Additional parameters for filtering
            
        Returns:
            Statistics as dictionary
        """
        endpoint = "v2/stats"
        return self._make_request(endpoint, params=kwargs)
    
    def get_conditions(self, term: Optional[str] = None, 
                      min_rank: int = 1, max_rank: int = 100) -> Dict:
        """
        Get conditions from the ClinicalTrials.gov database.
        
        Args:
            term: Search term for condition name
            min_rank: Starting rank for paginated results
            max_rank: Ending rank for paginated results
            
        Returns:
            Conditions as dictionary
        """
        endpoint = "v2/conditions"
        
        params = {
            "minRank": min_rank,
            "maxRank": max_rank
        }
        
        if term:
            params["term"] = term
        
        return self._make_request(endpoint, params=params)
    
    def get_condition(self, condition_id: str) -> Dict:
        """
        Get details for a specific condition by ID.
        
        Args:
            condition_id: The condition identifier
            
        Returns:
            Condition details as dictionary
        """
        endpoint = f"v2/conditions/{condition_id}"
        return self._make_request(endpoint)
    
    def get_interventions(self, term: Optional[str] = None, 
                          min_rank: int = 1, max_rank: int = 100) -> Dict:
        """
        Get interventions from the ClinicalTrials.gov database.
        
        Args:
            term: Search term for intervention name
            min_rank: Starting rank for paginated results
            max_rank: Ending rank for paginated results
            
        Returns:
            Interventions as dictionary
        """
        endpoint = "v2/interventions"
        
        params = {
            "minRank": min_rank,
            "maxRank": max_rank
        }
        
        if term:
            params["term"] = term
        
        return self._make_request(endpoint, params=params)
    
    def get_intervention(self, intervention_id: str) -> Dict:
        """
        Get details for a specific intervention by ID.
        
        Args:
            intervention_id: The intervention identifier
            
        Returns:
            Intervention details as dictionary
        """
        endpoint = f"v2/interventions/{intervention_id}"
        return self._make_request(endpoint)
    
    def get_sponsors(self, term: Optional[str] = None, 
                    min_rank: int = 1, max_rank: int = 100) -> Dict:
        """
        Get sponsors from the ClinicalTrials.gov database.
        
        Args:
            term: Search term for sponsor name
            min_rank: Starting rank for paginated results
            max_rank: Ending rank for paginated results
            
        Returns:
            Sponsors as dictionary
        """
        endpoint = "v2/sponsors"
        
        params = {
            "minRank": min_rank,
            "maxRank": max_rank
        }
        
        if term:
            params["term"] = term
        
        return self._make_request(endpoint, params=params)
    
    def get_sponsor(self, sponsor_id: str) -> Dict:
        """
        Get details for a specific sponsor by ID.
        
        Args:
            sponsor_id: The sponsor identifier
            
        Returns:
            Sponsor details as dictionary
        """
        endpoint = f"v2/sponsors/{sponsor_id}"
        return self._make_request(endpoint)
    
    def get_locations(self, term: Optional[str] = None, 
                     min_rank: int = 1, max_rank: int = 100) -> Dict:
        """
        Get locations from the ClinicalTrials.gov database.
        
        Args:
            term: Search term for location name
            min_rank: Starting rank for paginated results
            max_rank: Ending rank for paginated results
            
        Returns:
            Locations as dictionary
        """
        endpoint = "v2/locations"
        
        params = {
            "minRank": min_rank,
            "maxRank": max_rank
        }
        
        if term:
            params["term"] = term
        
        return self._make_request(endpoint, params=params)
    
    def get_location(self, location_id: str) -> Dict:
        """
        Get details for a specific location by ID.
        
        Args:
            location_id: The location identifier
            
        Returns:
            Location details as dictionary
        """
        endpoint = f"v2/locations/{location_id}"
        return self._make_request(endpoint)
    
    def get_countries(self, term: Optional[str] = None, 
                     min_rank: int = 1, max_rank: int = 100) -> Dict:
        """
        Get countries from the ClinicalTrials.gov database.
        
        Args:
            term: Search term for country name
            min_rank: Starting rank for paginated results
            max_rank: Ending rank for paginated results
            
        Returns:
            Countries as dictionary
        """
        endpoint = "v2/countries"
        
        params = {
            "minRank": min_rank,
            "maxRank": max_rank
        }
        
        if term:
            params["term"] = term
        
        return self._make_request(endpoint, params=params)
    
    def get_country(self, country_id: str) -> Dict:
        """
        Get details for a specific country by ID.
        
        Args:
            country_id: The country identifier
            
        Returns:
            Country details as dictionary
        """
        endpoint = f"v2/countries/{country_id}"
        return self._make_request(endpoint)
    
    def get_study_as_dataframe(self, nct_id: str) -> pd.DataFrame:
        """
        Get a study and convert key fields to a pandas DataFrame.
        
        Args:
            nct_id: The ClinicalTrials.gov identifier (NCT number)
            
        Returns:
            Study data as a DataFrame
        """
        study = self.get_study(nct_id)
        
        # Extract the study data
        study_data = study.get("data", {}).get("study", {})
        
        # Build a DataFrame from the study data
        # Focus on most important fields for the summary
        data = {
            "NCT ID": [study_data.get("nctId", "")],
            "Title": [study_data.get("briefTitle", "")],
            "Status": [study_data.get("overallStatus", "")],
            "Phase": [study_data.get("phase", "")],
            "Enrollment": [study_data.get("enrollment", "")],
            "Start Date": [study_data.get("startDate", "")],
            "Completion Date": [study_data.get("completionDate", "")],
            "Sponsor": [study_data.get("leadSponsor", {}).get("name", "")]
        }
        
        df = pd.DataFrame(data)
        return df
    
    def search_to_dataframe(self, query: Optional[str] = None, 
                           max_results: int = 100, **kwargs) -> pd.DataFrame:
        """
        Search for studies and convert to a pandas DataFrame.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to include
            **kwargs: Additional search parameters
            
        Returns:
            Search results as a DataFrame
        """
        # Fields to include in the DataFrame
        fields = kwargs.pop("fields", None) or [
            "NCTId", "BriefTitle", "OverallStatus", "Phase", 
            "EnrollmentCount", "StartDate", "CompletionDate", "LeadSponsorName"
        ]
        
        # Calculate the number of pages needed based on max_results
        page_size = 100  # API typically allows up to 100 per page
        max_pages = (max_results + page_size - 1) // page_size
        
        # Use the iterator to get all studies
        studies_iter = self.search_studies_iterator(
            query=query,
            fields=fields,
            page_size=page_size,
            max_pages=max_pages,
            **kwargs
        )
        
        # Collect the studies
        studies = []
        for i, study in enumerate(studies_iter):
            if i >= max_results:
                break
            studies.append(study)
        
        # Convert to DataFrame
        if not studies:
            # Return empty DataFrame with columns
            return pd.DataFrame(columns=fields)
        
        # Create DataFrame
        df = pd.DataFrame(studies)
        
        # Rename columns for better readability
        column_map = {
            "NCTId": "NCT ID",
            "BriefTitle": "Title",
            "OverallStatus": "Status",
            "Phase": "Phase",
            "EnrollmentCount": "Enrollment",
            "StartDate": "Start Date",
            "CompletionDate": "Completion Date",
            "LeadSponsorName": "Sponsor"
        }
        
        # Apply column renaming for columns that exist in the DataFrame
        for old_name, new_name in column_map.items():
            if old_name in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)
        
        return df


class StudyAnalyzer:
    """
    Analyze clinical trial data from ClinicalTrials.gov.
    
    This class provides methods for analyzing clinical trial data,
    including trend analysis, statistical summaries, and visualization.
    """
    
    def __init__(self, client: ClinicalTrialsClient):
        """
        Initialize the StudyAnalyzer.
        
        Args:
            client: A ClinicalTrialsClient instance
        """
        self.client = client
    
    def get_phase_distribution(self, query: Optional[str] = None, 
                              max_results: int = 1000) -> Dict:
        """
        Analyze the distribution of studies by phase.
        
        Args:
            query: Search query to filter studies
            max_results: Maximum number of results to analyze
            
        Returns:
            Distribution of studies by phase
        """
        # Fields needed for analysis
        fields = ["NCTId", "Phase"]
        
        # Collect studies
        studies_iter = self.client.search_studies_iterator(
            query=query,
            fields=fields,
            page_size=100,
            max_pages=(max_results + 99) // 100
        )
        
        # Count phases
        phase_counts = {}
        
        for i, study in enumerate(studies_iter):
            if i >= max_results:
                break
            
            phase = study.get("Phase", "Not Specified")
            if not phase:
                phase = "Not Specified"
            
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        
        # Calculate percentages
        total = sum(phase_counts.values())
        result = {
            "total_studies": total,
            "distribution": {
                phase: {
                    "count": count,
                    "percentage": round(count / total * 100, 2) if total > 0 else 0
                }
                for phase, count in phase_counts.items()
            }
        }
        
        return result
    
    def get_status_trends(self, query: Optional[str] = None, 
                         max_results: int = 1000) -> Dict:
        """
        Analyze trends in study status.
        
        Args:
            query: Search query to filter studies
            max_results: Maximum number of results to analyze
            
        Returns:
            Trends in study status over time
        """
        # Fields needed for analysis
        fields = ["NCTId", "OverallStatus", "StartDate", "CompletionDate"]
        
        # Collect studies
        studies_iter = self.client.search_studies_iterator(
            query=query,
            fields=fields,
            page_size=100,
            max_pages=(max_results + 99) // 100
        )
        
        # Process studies
        status_by_year = {}
        
        for i, study in enumerate(studies_iter):
            if i >= max_results:
                break
            
            start_date = study.get("StartDate", "")
            status = study.get("OverallStatus", "Unknown")
            
            # Extract year from start date
            if start_date:
                try:
                    # Handle different date formats
                    if "T" in start_date:
                        # ISO format
                        year = start_date.split("T")[0].split("-")[0]
                    else:
                        # Simple date
                        year = start_date.split("-")[0]
                    
                    # Skip if not a valid year
                    if not year.isdigit():
                        continue
                    
                    # Initialize year entry if not exists
                    if year not in status_by_year:
                        status_by_year[year] = {}
                    
                    # Update status count
                    status_by_year[year][status] = status_by_year[year].get(status, 0) + 1
                    
                except (IndexError, ValueError):
                    # Skip if date parsing fails
                    continue
        
        # Convert to result format
        result = {
            "total_studies": i + 1,
            "trend_by_year": {
                year: statuses
                for year, statuses in sorted(status_by_year.items())
            }
        }
        
        return result
    
    def get_condition_summary(self, query: Optional[str] = None, 
                            max_results: int = 1000) -> Dict:
        """
        Summarize studies by condition.
        
        Args:
            query: Search query to filter studies
            max_results: Maximum number of results to analyze
            
        Returns:
            Summary of studies by condition
        """
        # Fields needed for analysis
        fields = ["NCTId", "Condition", "OverallStatus"]
        
        # Collect studies
        studies_iter = self.client.search_studies_iterator(
            query=query,
            fields=fields,
            page_size=100,
            max_pages=(max_results + 99) // 100
        )
        
        # Process studies
        condition_counts = {}
        
        for i, study in enumerate(studies_iter):
            if i >= max_results:
                break
            
            conditions = study.get("Condition", [])
            status = study.get("OverallStatus", "Unknown")
            
            for condition in conditions:
                if condition not in condition_counts:
                    condition_counts[condition] = {
                        "total": 0,
                        "status_counts": {}
                    }
                
                condition_counts[condition]["total"] += 1
                condition_counts[condition]["status_counts"][status] = (
                    condition_counts[condition]["status_counts"].get(status, 0) + 1
                )
        
        # Sort conditions by total count
        sorted_conditions = sorted(
            condition_counts.items(),
            key=lambda x: x[1]["total"],
            reverse=True
        )
        
        # Format results
        result = {
            "total_studies": i + 1,
            "condition_summary": {
                condition: info
                for condition, info in sorted_conditions[:50]  # Top 50 conditions
            }
        }
        
        return result
    
    def create_study_timeline(self, nct_id: str) -> Dict:
        """
        Create a timeline of key events for a specific study.
        
        Args:
            nct_id: The ClinicalTrials.gov identifier (NCT number)
            
        Returns:
            Timeline of study events
        """
        # Get full study details
        study = self.client.get_study(nct_id)
        study_data = study.get("data", {}).get("study", {})
        
        # Extract relevant dates
        timeline_events = []
        
        # Study dates
        for event_type, date_field in [
            ("Study Start", "startDate"),
            ("Primary Completion", "primaryCompletionDate"),
            ("Study Completion", "completionDate"),
            ("First Posted", "studyFirstPostDate"),
            ("Last Update Posted", "lastUpdatePostDate"),
            ("Results First Posted", "resultsFirstPostDate")
        ]:
            date_value = study_data.get(date_field)
            if date_value:
                timeline_events.append({
                    "event": event_type,
                    "date": date_value,
                    "description": f"{event_type} date for the study"
                })
        
        # Status changes from history
        history = study_data.get("documentHistory", [])
        for entry in history:
            if "statusVerifiedDate" in entry:
                timeline_events.append({
                    "event": "Status Verified",
                    "date": entry.get("statusVerifiedDate"),
                    "description": f"Status verified as '{entry.get('status', 'Unknown')}'"
                })
        
        # Sort events by date
        def parse_date(date_str):
            try:
                # Handle different date formats
                if "T" in date_str:
                    # ISO format
                    return datetime.fromisoformat(date_str.split("T")[0])
                else:
                    # Simple date
                    parts = date_str.split("-")
                    if len(parts) == 3:
                        return datetime(int(parts[0]), int(parts[1]), int(parts[2]))
                    elif len(parts) == 2:
                        return datetime(int(parts[0]), int(parts[1]), 1)
                    else:
                        return datetime(int(parts[0]), 1, 1)
            except:
                # Default to distant past if parsing fails
                return datetime(1900, 1, 1)
        
        timeline_events.sort(key=lambda e: parse_date(e["date"]))
        
        # Format the final timeline
        result = {
            "nct_id": nct_id,
            "study_title": study_data.get("briefTitle", ""),
            "timeline": timeline_events
        }
        
        return result
    
    def find_related_studies(self, nct_id: str, max_results: int = 20) -> List[Dict]:
        """
        Find studies related to a specific study.
        
        Args:
            nct_id: The ClinicalTrials.gov identifier (NCT number)
            max_results: Maximum number of related studies to return
            
        Returns:
            List of related studies
        """
        # Get the original study to extract key information
        original_study = self.client.get_study(nct_id)
        original_data = original_study.get("data", {}).get("study", {})
        
        # Extract key information for finding related studies
        conditions = original_data.get("condition", [])
        interventions = [i.get("name", "") for i in original_data.get("intervention", [])]
        sponsors = [original_data.get("leadSponsor", {}).get("name", "")]
        
        # Build a search query based on conditions and interventions
        search_terms = []
        
        # Add conditions
        if conditions:
            condition_query = " OR ".join([f'CONDITION:"{c}"' for c in conditions[:3]])
            search_terms.append(f"({condition_query})")
        
        # Add interventions
        if interventions:
            intervention_query = " OR ".join([f'INTERVENTION:"{i}"' for i in interventions[:3]])
            search_terms.append(f"({intervention_query})")
        
        # Combine search terms
        search_query = " AND ".join(search_terms)
        
        # Search for related studies
        related = self.client.search_studies(
            query=search_query,
            fields=["NCTId", "BriefTitle", "OverallStatus", "Phase", "LeadSponsorName"],
            min_rank=1,
            max_rank=max_results + 1  # Add 1 to account for the original study
        )
        
        # Process results to exclude the original study
        related_studies = []
        for study in related.get("studies", []):
            if study.get("NCTId") != nct_id:
                related_studies.append({
                    "nct_id": study.get("NCTId"),
                    "title": study.get("BriefTitle", ""),
                    "status": study.get("OverallStatus", ""),
                    "phase": study.get("Phase", ""),
                    "sponsor": study.get("LeadSponsorName", "")
                })
            
            # Limit to requested number
            if len(related_studies) >= max_results:
                break
        
        return related_studies


def example_usage():
    """Demonstrate the usage of the ClinicalTrials.gov API client."""
    # Initialize the client
    client = ClinicalTrialsClient(cache_dir="./ctgov_cache")
    
    # Search for diabetes studies
    print("Searching for diabetes studies...")
    search_results = client.search_studies(
        query="diabetes",
        fields=["NCTId", "BriefTitle", "OverallStatus"],
        min_rank=1,
        max_rank=5
    )
    
    # Print search results
    for study in search_results.get("studies", []):
        print(f"{study.get('NCTId')}: {study.get('BriefTitle')} - {study.get('OverallStatus')}")
    
    # Get details for a specific study
    print("\nGetting details for a specific study...")
    study = client.get_study("NCT03980509")  # AMPLITUDE-O study
    study_data = study.get("data", {}).get("study", {})
    print(f"Title: {study_data.get('briefTitle')}")
    print(f"Status: {study_data.get('overallStatus')}")
    print(f"Phase: {study_data.get('phase')}")
    
    # Use the analyzer
    print("\nAnalyzing phase distribution...")
    analyzer = StudyAnalyzer(client)
    phase_distribution = analyzer.get_phase_distribution(query="cancer", max_results=100)
    
    # Print phase distribution
    print(f"Total studies analyzed: {phase_distribution['total_studies']}")
    for phase, info in phase_distribution["distribution"].items():
        print(f"{phase}: {info['count']} studies ({info['percentage']}%)")
    
    # Get a study timeline
    print("\nCreating study timeline...")
    timeline = analyzer.create_study_timeline("NCT03980509")
    print(f"Timeline for {timeline['study_title']}:")
    for event in timeline["timeline"]:
        print(f"{event['date']}: {event['event']}")


if __name__ == "__main__":
    example_usage()