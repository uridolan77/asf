"""
ClinicalTrials.gov client for the Medical Research Synthesizer.

This module provides a client for interacting with the ClinicalTrials.gov API.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Union
import httpx
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

from asf.medical.core.config import settings

# Set up logging
logger = logging.getLogger(__name__)

class ClinicalTrialsClient:
    """
    Client for interacting with the ClinicalTrials.gov API.
    
    This client provides methods for searching clinical trials and retrieving trial details.
    """
    
    def __init__(
        self,
        base_url: str = "https://clinicaltrials.gov/api/v2"
    ):
        """
        Initialize the ClinicalTrials.gov client.
        
        Args:
            base_url: Base URL for ClinicalTrials.gov API (default: "https://clinicaltrials.gov/api/v2")
        """
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # Rate limiting
        self.requests_per_second = 5
        self.last_request_time = 0
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def _rate_limit(self):
        """
        Implement rate limiting for ClinicalTrials.gov API.
        
        ClinicalTrials.gov recommends no more than 5 requests per second.
        """
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        min_interval = 1.0 / self.requests_per_second
        
        if time_since_last_request < min_interval:
            await asyncio.sleep(min_interval - time_since_last_request)
        
        self.last_request_time = time.time()
    
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((httpx.HTTPError, asyncio.TimeoutError))
    )
    async def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Make a request to the ClinicalTrials.gov API.
        
        Args:
            endpoint: API endpoint
            params: Request parameters
            
        Returns:
            Response data
            
        Raises:
            httpx.HTTPError: If the request fails
        """
        # Apply rate limiting
        await self._rate_limit()
        
        # Make request
        url = f"{self.base_url}/{endpoint}"
        logger.debug(f"Making request to {url} with params {params}")
        
        response = await self.client.get(url, params=params)
        response.raise_for_status()
        
        return response.json()
    
    async def search(
        self,
        query: str,
        max_results: int = 20,
        status: Optional[str] = None,
        phase: Optional[str] = None,
        study_type: Optional[str] = None,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for clinical trials.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return (default: 20)
            status: Trial status (e.g., "Recruiting", "Completed")
            phase: Trial phase (e.g., "Phase 1", "Phase 2")
            study_type: Study type (e.g., "Interventional", "Observational")
            min_date: Minimum start date (YYYY/MM/DD format)
            max_date: Maximum start date (YYYY/MM/DD format)
            
        Returns:
            List of trial summaries
        """
        params = {
            "query.term": query,
            "pageSize": max_results,
            "format": "json"
        }
        
        # Add filters if provided
        if status:
            params["filter.overall_status"] = status
        
        if phase:
            params["filter.phase"] = phase
        
        if study_type:
            params["filter.study_type"] = study_type
        
        # Add date range if provided
        if min_date and max_date:
            params["filter.start_date"] = f"{min_date},{max_date}"
        
        try:
            data = await self._make_request("studies", params)
            
            # Extract study summaries
            studies = []
            if "studies" in data:
                for study in data["studies"]:
                    # Extract basic study information
                    study_summary = {
                        "nct_id": study.get("protocolSection", {}).get("identificationModule", {}).get("nctId", ""),
                        "title": study.get("protocolSection", {}).get("identificationModule", {}).get("officialTitle", ""),
                        "brief_title": study.get("protocolSection", {}).get("identificationModule", {}).get("briefTitle", ""),
                        "status": study.get("protocolSection", {}).get("statusModule", {}).get("overallStatus", ""),
                        "phase": study.get("protocolSection", {}).get("designModule", {}).get("phases", []),
                        "study_type": study.get("protocolSection", {}).get("designModule", {}).get("studyType", ""),
                        "conditions": study.get("protocolSection", {}).get("conditionsModule", {}).get("conditions", []),
                        "interventions": [],
                        "source": "ClinicalTrials.gov"
                    }
                    
                    # Extract interventions
                    interventions_module = study.get("protocolSection", {}).get("armsInterventionsModule", {})
                    if "interventions" in interventions_module:
                        for intervention in interventions_module["interventions"]:
                            study_summary["interventions"].append({
                                "name": intervention.get("name", ""),
                                "type": intervention.get("type", "")
                            })
                    
                    studies.append(study_summary)
            
            return studies
        except Exception as e:
            logger.error(f"Error searching ClinicalTrials.gov: {str(e)}")
            raise
    
    async def get_study(self, nct_id: str) -> Dict[str, Any]:
        """
        Get details for a specific clinical trial.
        
        Args:
            nct_id: NCT ID (e.g., "NCT01234567")
            
        Returns:
            Trial details
        """
        try:
            data = await self._make_request(f"studies/{nct_id}")
            
            # Extract study details
            study = data.get("study", {})
            protocol_section = study.get("protocolSection", {})
            
            # Basic information
            identification_module = protocol_section.get("identificationModule", {})
            status_module = protocol_section.get("statusModule", {})
            design_module = protocol_section.get("designModule", {})
            conditions_module = protocol_section.get("conditionsModule", {})
            description_module = protocol_section.get("descriptionModule", {})
            
            # Eligibility
            eligibility_module = protocol_section.get("eligibilityModule", {})
            
            # Interventions
            arms_interventions_module = protocol_section.get("armsInterventionsModule", {})
            
            # Outcomes
            outcomes_module = protocol_section.get("outcomesModule", {})
            
            # Create detailed study object
            study_details = {
                "nct_id": identification_module.get("nctId", ""),
                "title": identification_module.get("officialTitle", ""),
                "brief_title": identification_module.get("briefTitle", ""),
                "status": status_module.get("overallStatus", ""),
                "phase": design_module.get("phases", []),
                "study_type": design_module.get("studyType", ""),
                "conditions": conditions_module.get("conditions", []),
                "brief_summary": description_module.get("briefSummary", ""),
                "detailed_description": description_module.get("detailedDescription", ""),
                "start_date": status_module.get("startDateStruct", {}).get("date", ""),
                "completion_date": status_module.get("completionDateStruct", {}).get("date", ""),
                "eligibility": {
                    "criteria": eligibility_module.get("eligibilityCriteria", ""),
                    "gender": eligibility_module.get("gender", ""),
                    "minimum_age": eligibility_module.get("minimumAge", ""),
                    "maximum_age": eligibility_module.get("maximumAge", "")
                },
                "interventions": [],
                "outcomes": [],
                "source": "ClinicalTrials.gov"
            }
            
            # Extract interventions
            if "interventions" in arms_interventions_module:
                for intervention in arms_interventions_module["interventions"]:
                    study_details["interventions"].append({
                        "name": intervention.get("name", ""),
                        "type": intervention.get("type", ""),
                        "description": intervention.get("description", "")
                    })
            
            # Extract outcomes
            if "primaryOutcomes" in outcomes_module:
                for outcome in outcomes_module["primaryOutcomes"]:
                    study_details["outcomes"].append({
                        "type": "primary",
                        "measure": outcome.get("measure", ""),
                        "description": outcome.get("description", ""),
                        "time_frame": outcome.get("timeFrame", "")
                    })
            
            if "secondaryOutcomes" in outcomes_module:
                for outcome in outcomes_module["secondaryOutcomes"]:
                    study_details["outcomes"].append({
                        "type": "secondary",
                        "measure": outcome.get("measure", ""),
                        "description": outcome.get("description", ""),
                        "time_frame": outcome.get("timeFrame", "")
                    })
            
            return study_details
        except Exception as e:
            logger.error(f"Error getting study from ClinicalTrials.gov: {str(e)}")
            raise
