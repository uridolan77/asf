"""
UMLS client for the Medical Research Synthesizer.

This module provides a client for interacting with the UMLS API.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Union
import httpx
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# Set up logging
logger = logging.getLogger(__name__)

class UMLSClient:
    """
    Client for interacting with the UMLS API.
    
    This client provides methods for searching UMLS concepts and retrieving concept details.
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://uts-ws.nlm.nih.gov/rest"
    ):
        """
        Initialize the UMLS client.
        
        Args:
            api_key: API key for UMLS API
            base_url: Base URL for UMLS API (default: "https://uts-ws.nlm.nih.gov/rest")
        """
        self.api_key = api_key
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
        self.tgt = None
        self.tgt_expires = 0
        
        # Rate limiting
        self.requests_per_second = 5
        self.last_request_time = 0
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def _rate_limit(self):
        """
        Implement rate limiting for UMLS API.
        
        UMLS recommends no more than 5 requests per second.
        """
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        min_interval = 1.0 / self.requests_per_second
        
        if time_since_last_request < min_interval:
            await asyncio.sleep(min_interval - time_since_last_request)
        
        self.last_request_time = time.time()
    
    async def _get_tgt(self) -> str:
        """
        Get a Ticket Granting Ticket (TGT) from the UMLS API.
        
        Returns:
            TGT URL
            
        Raises:
            httpx.HTTPError: If the request fails
        """
        # Check if we already have a valid TGT
        if self.tgt and time.time() < self.tgt_expires:
            return self.tgt
        
        # Get a new TGT
        url = f"{self.base_url}/auth/tgt"
        data = {
            "apikey": self.api_key
        }
        
        response = await self.client.post(url, data=data)
        response.raise_for_status()
        
        # Extract TGT URL from response
        tgt_url = response.headers.get("location")
        if not tgt_url:
            raise ValueError("No TGT URL in response")
        
        # Set TGT and expiration time (8 hours)
        self.tgt = tgt_url
        self.tgt_expires = time.time() + 8 * 60 * 60
        
        return tgt_url
    
    async def _get_service_ticket(self) -> str:
        """
        Get a Service Ticket (ST) from the UMLS API.
        
        Returns:
            Service Ticket
            
        Raises:
            httpx.HTTPError: If the request fails
        """
        # Get TGT URL
        tgt_url = await self._get_tgt()
        
        # Get Service Ticket
        data = {
            "service": f"{self.base_url}/search"
        }
        
        response = await self.client.post(tgt_url, data=data)
        response.raise_for_status()
        
        return response.text
    
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((httpx.HTTPError, asyncio.TimeoutError))
    )
    async def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Make a request to the UMLS API.
        
        Args:
            endpoint: API endpoint
            params: Request parameters
            
        Returns:
            Response data
            
        Raises:
            httpx.HTTPError: If the request fails
        """
        # Get Service Ticket
        ticket = await self._get_service_ticket()
        
        # Apply rate limiting
        await self._rate_limit()
        
        # Make request
        url = f"{self.base_url}/{endpoint}"
        
        # Add ticket to params
        if params is None:
            params = {}
        params["ticket"] = ticket
        
        logger.debug(f"Making request to {url} with params {params}")
        
        response = await self.client.get(url, params=params)
        response.raise_for_status()
        
        return response.json()
    
    async def search(
        self,
        query: str,
        search_type: str = "words",
        max_results: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search for UMLS concepts.
        
        Args:
            query: Search query
            search_type: Search type (default: "words")
            max_results: Maximum number of results to return (default: 20)
            
        Returns:
            List of concept summaries
        """
        params = {
            "string": query,
            "searchType": search_type,
            "pageSize": max_results,
            "returnIdType": "concept"
        }
        
        try:
            data = await self._make_request("search/current", params)
            
            # Extract concept summaries
            concepts = []
            if "result" in data and "results" in data["result"]:
                for result in data["result"]["results"]:
                    concept = {
                        "ui": result.get("ui", ""),
                        "name": result.get("name", ""),
                        "uri": result.get("uri", ""),
                        "source": "UMLS"
                    }
                    concepts.append(concept)
            
            return concepts
        except Exception as e:
            logger.error(f"Error searching UMLS: {str(e)}")
            raise
    
    async def get_concept(self, concept_id: str) -> Dict[str, Any]:
        """
        Get details for a specific UMLS concept.
        
        Args:
            concept_id: Concept ID (e.g., "C0012634")
            
        Returns:
            Concept details
        """
        try:
            data = await self._make_request(f"content/current/CUI/{concept_id}")
            
            # Extract concept details
            result = data.get("result", {})
            
            concept = {
                "ui": result.get("ui", ""),
                "name": result.get("name", ""),
                "semantic_types": [],
                "definitions": [],
                "synonyms": [],
                "source": "UMLS"
            }
            
            # Get semantic types
            semantic_types_url = result.get("semanticTypes", "")
            if semantic_types_url:
                semantic_types_data = await self._make_request(semantic_types_url.replace(self.base_url + "/", ""))
                if "result" in semantic_types_data:
                    for semantic_type in semantic_types_data["result"]:
                        concept["semantic_types"].append({
                            "ui": semantic_type.get("semanticType", {}).get("ui", ""),
                            "name": semantic_type.get("semanticType", {}).get("name", "")
                        })
            
            # Get definitions
            definitions_url = result.get("definitions", "")
            if definitions_url:
                definitions_data = await self._make_request(definitions_url.replace(self.base_url + "/", ""))
                if "result" in definitions_data:
                    for definition in definitions_data["result"]:
                        concept["definitions"].append({
                            "value": definition.get("value", ""),
                            "source": definition.get("source", "")
                        })
            
            # Get synonyms (atoms)
            atoms_url = result.get("atoms", "")
            if atoms_url:
                atoms_data = await self._make_request(atoms_url.replace(self.base_url + "/", ""))
                if "result" in atoms_data:
                    for atom in atoms_data["result"]:
                        concept["synonyms"].append({
                            "name": atom.get("name", ""),
                            "source": atom.get("rootSource", "")
                        })
            
            return concept
        except Exception as e:
            logger.error(f"Error getting concept from UMLS: {str(e)}")
            raise
    
    async def get_relations(self, concept_id: str) -> List[Dict[str, Any]]:
        """
        Get relations for a specific UMLS concept.
        
        Args:
            concept_id: Concept ID (e.g., "C0012634")
            
        Returns:
            List of relations
        """
        try:
            data = await self._make_request(f"content/current/CUI/{concept_id}/relations")
            
            # Extract relations
            relations = []
            if "result" in data:
                for relation in data["result"]:
                    relations.append({
                        "relation_label": relation.get("relationLabel", ""),
                        "additional_relation_label": relation.get("additionalRelationLabel", ""),
                        "related_concept": {
                            "ui": relation.get("relatedConcept", {}).get("ui", ""),
                            "name": relation.get("relatedConcept", {}).get("name", "")
                        },
                        "source": "UMLS"
                    })
            
            return relations
        except Exception as e:
            logger.error(f"Error getting relations from UMLS: {str(e)}")
            raise
