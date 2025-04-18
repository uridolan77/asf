"""
ClinicalTrials.gov Data Extractor

This module extracts clinical trial data from ClinicalTrials.gov using the
ClinicalTrials.gov API. It supports searching by terms, date ranges, and other criteria.
"""

import os
import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import aiohttp
import pandas as pd
from pathlib import Path

logger = logging.getLogger("biomedical_etl.extractors.clinicaltrials")

class ClinicalTrialsExtractor:
    """
    Extractor for ClinicalTrials.gov data using the ClinicalTrials.gov API.
    
    This class provides methods for searching and retrieving clinical trial data
    from ClinicalTrials.gov. It supports batch processing, caching, and error handling.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        batch_size: int = 100,
        max_trials: int = 5000,
        cache_dir: str = "./cache/clinicaltrials"
    ):
        """
        Initialize the ClinicalTrials.gov extractor.
        
        Args:
            api_key: API key for the ClinicalTrials.gov API (optional)
            batch_size: Number of trials to fetch in each batch
            max_trials: Maximum number of trials to fetch in total
            cache_dir: Directory to store cache files
        """
        self.api_key = api_key
        self.batch_size = batch_size
        self.max_trials = max_trials
        self.cache_dir = cache_dir
        
        # Base URL for ClinicalTrials.gov API
        self.base_url = "https://clinicaltrials.gov/api/v2"
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info(f"Initialized ClinicalTrials.gov extractor with batch_size={batch_size}, max_trials={max_trials}")
    
    async def extract(
        self,
        search_terms: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract clinical trial data from ClinicalTrials.gov.
        
        Args:
            search_terms: List of search terms to use for querying ClinicalTrials.gov
            start_date: Start date for search (format: YYYY-MM-DD)
            end_date: End date for search (format: YYYY-MM-DD)
            
        Returns:
            List of clinical trial dictionaries
        """
        logger.info(f"Extracting ClinicalTrials.gov trials for {len(search_terms)} search terms")
        
        all_trials = []
        
        # Create a session for connection pooling
        async with aiohttp.ClientSession() as session:
            for term in search_terms:
                try:
                    # Generate cache key based on search parameters
                    cache_key = self._generate_cache_key(term, start_date, end_date)
                    cache_path = Path(self.cache_dir) / f"{cache_key}.json"
                    
                    # Check if we have cached results
                    if cache_path.exists():
                        logger.info(f"Loading cached results for '{term}'")
                        trials = self._load_from_cache(cache_path)
                    else:
                        logger.info(f"Searching ClinicalTrials.gov for '{term}'")
                        trials = await self._search_and_fetch(session, term, start_date, end_date)
                        
                        # Cache the results
                        self._save_to_cache(trials, cache_path)
                    
                    logger.info(f"Found {len(trials)} trials for '{term}'")
                    all_trials.extend(trials)
                    
                except Exception as e:
                    logger.error(f"Error extracting trials for '{term}': {str(e)}")
        
        # Deduplicate trials based on NCT ID
        deduplicated_trials = self._deduplicate_trials(all_trials)
        
        logger.info(f"Extracted {len(deduplicated_trials)} unique ClinicalTrials.gov trials")
        return deduplicated_trials
    
    async def _search_and_fetch(
        self,
        session: aiohttp.ClientSession,
        term: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search ClinicalTrials.gov and fetch trial details.
        
        Args:
            session: aiohttp ClientSession
            term: Search term
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            List of clinical trial dictionaries
        """
        # Step 1: Search for trial NCT IDs
        nct_ids = await self._search_trials(session, term, start_date, end_date)
        
        # Step 2: Fetch trial details in batches
        trials = []
        for i in range(0, min(len(nct_ids), self.max_trials), self.batch_size):
            batch_nct_ids = nct_ids[i:i+self.batch_size]
            logger.info(f"Fetching details for {len(batch_nct_ids)} trials (batch {i//self.batch_size + 1})")
            
            batch_trials = await self._fetch_trial_details(session, batch_nct_ids)
            trials.extend(batch_trials)
            
            # Add a small delay to respect API rate limits
            await asyncio.sleep(0.5)
        
        return trials
    
    async def _search_trials(
        self,
        session: aiohttp.ClientSession,
        term: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[str]:
        """
        Search ClinicalTrials.gov for trials matching the criteria.
        
        Args:
            session: aiohttp ClientSession
            term: Search term
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            List of NCT IDs
        """
        # Build query parameters
        params = {
            "query.term": term,
            "pageSize": min(1000, self.max_trials),  # API has a limit of 1000 per page
            "format": "json"
        }
        
        # Add date range if provided
        if start_date or end_date:
            date_filter = []
            
            if start_date:
                date_filter.append(f"AREA[StartDate]RANGE[{start_date},")
                if end_date:
                    date_filter[-1] += f"{end_date}]"
                else:
                    date_filter[-1] += "MAX]"
            elif end_date:
                date_filter.append(f"AREA[StartDate]RANGE[MIN,{end_date}]")
            
            params["filter.filter"] = " AND ".join(date_filter)
        
        # Add API key if available
        headers = {}
        if self.api_key:
            headers["api_key"] = self.api_key
        
        # Make the request
        url = f"{self.base_url}/studies"
        async with session.get(url, params=params, headers=headers) as response:
            response.raise_for_status()
            data = await response.json()
        
        # Extract NCT IDs from the response
        studies = data.get("studies", [])
        nct_ids = [study.get("protocolSection", {}).get("identificationModule", {}).get("nctId") 
                 for study in studies if study.get("protocolSection")]
        
        # Filter out None values
        nct_ids = [nct_id for nct_id in nct_ids if nct_id]
        
        return nct_ids
    
    async def _fetch_trial_details(
        self,
        session: aiohttp.ClientSession,
        nct_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Fetch details for a batch of trials.
        
        Args:
            session: aiohttp ClientSession
            nct_ids: List of NCT IDs to fetch
            
        Returns:
            List of clinical trial dictionaries
        """
        if not nct_ids:
            return []
        
        # For V2 API, we need to make individual requests for each NCT ID
        # In a production system, we would optimize this with concurrent requests
        
        trials = []
        for nct_id in nct_ids:
            try:
                trial = await self._fetch_single_trial(session, nct_id)
                if trial:
                    trials.append(trial)
            except Exception as e:
                logger.error(f"Error fetching trial {nct_id}: {str(e)}")
            
            # Add a small delay to respect API rate limits
            await asyncio.sleep(0.1)
        
        return trials
    
    async def _fetch_single_trial(
        self,
        session: aiohttp.ClientSession,
        nct_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch details for a single trial.
        
        Args:
            session: aiohttp ClientSession
            nct_id: NCT ID to fetch
            
        Returns:
            Clinical trial dictionary or None if not found
        """
        # Build request parameters
        params = {
            "format": "json"
        }
        
        # Add API key if available
        headers = {}
        if self.api_key:
            headers["api_key"] = self.api_key
        
        # Make the request
        url = f"{self.base_url}/studies/{nct_id}"
        async with session.get(url, params=params, headers=headers) as response:
            if response.status == 404:
                logger.warning(f"Trial {nct_id} not found")
                return None
            
            response.raise_for_status()
            data = await response.json()
        
        # Extract and format trial data
        return self._format_trial_data(data)
    
    def _format_trial_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format trial data from the API response.
        
        Args:
            data: API response data
            
        Returns:
            Formatted clinical trial dictionary
        """
        protocol = data.get("protocolSection", {})
        derived = data.get("derivedSection", {})
        
        # Extract identification information
        identification = protocol.get("identificationModule", {})
        nct_id = identification.get("nctId", "")
        title = identification.get("briefTitle", "")
        official_title = identification.get("officialTitle", "")
        
        # Extract status information
        status_module = protocol.get("statusModule", {})
        status = status_module.get("overallStatus", "")
        phase = status_module.get("phase", "")
        start_date = status_module.get("startDateStruct", {}).get("date", "")
        completion_date = status_module.get("completionDateStruct", {}).get("date", "")
        
        # Extract sponsor information
        sponsor_module = protocol.get("sponsorCollaboratorsModule", {})
        lead_sponsor = sponsor_module.get("leadSponsor", {}).get("name", "")
        
        # Extract condition information
        condition_module = protocol.get("conditionsModule", {})
        conditions = condition_module.get("conditions", [])
        keywords = condition_module.get("keywords", [])
        
        # Extract intervention information
        intervention_module = protocol.get("interventionsModule", {})
        interventions = []
        
        for intervention in intervention_module.get("interventions", []):
            intervention_type = intervention.get("type", "")
            intervention_name = intervention.get("name", "")
            
            if intervention_type and intervention_name:
                interventions.append({
                    "type": intervention_type,
                    "name": intervention_name,
                    "description": intervention.get("description", "")
                })
        
        # Extract design information
        design_module = protocol.get("designModule", {})
        study_type = design_module.get("studyType", "")
        enrollment = design_module.get("enrollmentInfo", {}).get("count", 0)
        
        # Extract eligibility information
        eligibility_module = protocol.get("eligibilityModule", {})
        eligibility_criteria = eligibility_module.get("eligibilityCriteria", "")
        gender = eligibility_module.get("gender", "")
        minimum_age = eligibility_module.get("minimumAge", "")
        maximum_age = eligibility_module.get("maximumAge", "")
        
        # Extract facility information
        contact_module = protocol.get("contactsLocationsModule", {})
        locations = []
        
        for facility in contact_module.get("locations", []):
            facility_name = facility.get("facility", {}).get("name", "")
            city = facility.get("facility", {}).get("city", "")
            state = facility.get("facility", {}).get("state", "")
            country = facility.get("facility", {}).get("country", "")
            
            if facility_name and country:
                locations.append({
                    "name": facility_name,
                    "city": city,
                    "state": state,
                    "country": country
                })
        
        # Create formatted trial dictionary
        trial = {
            "nct_id": nct_id,
            "title": title,
            "official_title": official_title,
            "status": status,
            "phase": phase,
            "start_date": start_date,
            "completion_date": completion_date,
            "lead_sponsor": lead_sponsor,
            "conditions": conditions,
            "keywords": keywords,
            "interventions": interventions,
            "study_type": study_type,
            "enrollment": enrollment,
            "eligibility_criteria": eligibility_criteria,
            "gender": gender,
            "minimum_age": minimum_age,
            "maximum_age": maximum_age,
            "locations": locations
        }
        
        # Extract description information
        description_module = protocol.get("descriptionModule", {})
        trial["brief_summary"] = description_module.get("briefSummary", "")
        trial["detailed_description"] = description_module.get("detailedDescription", "")
        
        return trial
    
    def _generate_cache_key(self, term: str, start_date: Optional[str], end_date: Optional[str]) -> str:
        """
        Generate a cache key based on search parameters.
        
        Args:
            term: Search term
            start_date: Start date
            end_date: End date
            
        Returns:
            Cache key string
        """
        import hashlib
        
        # Create a string with all parameters
        key_str = f"term={term}"
        
        if start_date:
            key_str += f"&start_date={start_date}"
        
        if end_date:
            key_str += f"&end_date={end_date}"
        
        # Generate a hash of the parameter string
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _save_to_cache(self, trials: List[Dict[str, Any]], cache_path: Path) -> None:
        """
        Save trials to cache.
        
        Args:
            trials: List of trial dictionaries
            cache_path: Path to save the cache file
        """
        with open(cache_path, 'w') as f:
            json.dump(trials, f)
    
    def _load_from_cache(self, cache_path: Path) -> List[Dict[str, Any]]:
        """
        Load trials from cache.
        
        Args:
            cache_path: Path to the cache file
            
        Returns:
            List of trial dictionaries
        """
        with open(cache_path, 'r') as f:
            return json.load(f)
    
    def _deduplicate_trials(self, trials: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate trials based on NCT ID.
        
        Args:
            trials: List of trial dictionaries
            
        Returns:
            Deduplicated list of trial dictionaries
        """
        seen_nct_ids = set()
        deduplicated = []
        
        for trial in trials:
            nct_id = trial.get("nct_id")
            if nct_id and nct_id not in seen_nct_ids:
                seen_nct_ids.add(nct_id)
                deduplicated.append(trial)
        
        return deduplicated