"""
Client library for the Medical Research Synthesizer API.

This module provides a client library for interacting with the Medical Research Synthesizer API.
"""

import os
import json
import logging
import httpx
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIResponse(BaseModel):
    """API response model."""
    success: bool = Field(..., description="Whether the request was successful")
    message: str = Field(..., description="Response message")
    data: Optional[Any] = Field(None, description="Response data")
    errors: Optional[List[Dict[str, Any]]] = Field(None, description="Error details")
    meta: Optional[Dict[str, Any]] = Field(None, description="Metadata")

class MedicalResearchSynthesizerClient:
    """
    Client for the Medical Research Synthesizer API.
    
    This class provides methods for interacting with the Medical Research Synthesizer API.
    """
    
    def __init__(
        self, 
        base_url: str = "http://localhost:8000",
        api_version: str = "v1",
        token: Optional[str] = None
    ):
        """
        Initialize the client.
        
        Args:
            base_url: Base URL of the API
            api_version: API version
            token: Authentication token
        """
        self.base_url = base_url
        self.api_version = api_version
        self.token = token
        self.client = httpx.AsyncClient(
            base_url=f"{base_url}/{api_version}",
            timeout=60.0
        )
    
    async def close(self):
        """Close the client."""
        await self.client.aclose()
    
    async def _request(
        self, 
        method: str, 
        path: str, 
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> APIResponse:
        """
        Make a request to the API.
        
        Args:
            method: HTTP method
            path: API path
            data: Request data
            params: Query parameters
            
        Returns:
            API response
            
        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        # Set up headers
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        
        # Make the request
        try:
            response = await self.client.request(
                method=method,
                url=path,
                json=data,
                params=params,
                headers=headers
            )
            response.raise_for_status()
            return APIResponse(**response.json())
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            try:
                error_data = e.response.json()
                return APIResponse(
                    success=False,
                    message=error_data.get("detail", str(e)),
                    errors=[{"detail": error_data.get("detail", str(e))}],
                    meta={"status_code": e.response.status_code}
                )
            except json.JSONDecodeError:
                return APIResponse(
                    success=False,
                    message=str(e),
                    errors=[{"detail": str(e)}],
                    meta={"status_code": e.response.status_code}
                )
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            return APIResponse(
                success=False,
                message=str(e),
                errors=[{"detail": str(e)}],
                meta={}
            )
    
    async def login(self, email: str, password: str) -> APIResponse:
        """
        Log in to the API.
        
        Args:
            email: User email
            password: User password
            
        Returns:
            API response with access token
        """
        # Login is a special case because it uses form data
        try:
            response = await self.client.post(
                "/auth/token",
                data={"username": email, "password": password}
            )
            response.raise_for_status()
            data = response.json()
            
            # Set the token
            self.token = data["access_token"]
            
            return APIResponse(
                success=True,
                message="Login successful",
                data=data,
                meta={}
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            try:
                error_data = e.response.json()
                return APIResponse(
                    success=False,
                    message=error_data.get("detail", str(e)),
                    errors=[{"detail": error_data.get("detail", str(e))}],
                    meta={"status_code": e.response.status_code}
                )
            except json.JSONDecodeError:
                return APIResponse(
                    success=False,
                    message=str(e),
                    errors=[{"detail": str(e)}],
                    meta={"status_code": e.response.status_code}
                )
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            return APIResponse(
                success=False,
                message=str(e),
                errors=[{"detail": str(e)}],
                meta={}
            )
    
    async def get_current_user(self) -> APIResponse:
        """
        Get the current user.
        
        Returns:
            API response with user information
        """
        return await self._request("GET", "/auth/me")
    
    async def search(
        self, 
        query: str, 
        max_results: int = 20
    ) -> APIResponse:
        """
        Search for medical literature.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            API response with search results
        """
        return await self._request(
            "POST", 
            "/search", 
            data={"query": query, "max_results": max_results}
        )
    
    async def search_pico(
        self,
        condition: str,
        interventions: List[str],
        outcomes: List[str],
        population: Optional[str] = None,
        study_design: Optional[str] = None,
        years: Optional[int] = None,
        max_results: int = 20
    ) -> APIResponse:
        """
        Search for medical literature using the PICO framework.
        
        Args:
            condition: Medical condition
            interventions: List of interventions
            outcomes: List of outcomes
            population: Population (optional)
            study_design: Study design (optional)
            years: Number of years to search (optional)
            max_results: Maximum number of results
            
        Returns:
            API response with search results
        """
        return await self._request(
            "POST",
            "/search/pico",
            data={
                "condition": condition,
                "interventions": interventions,
                "outcomes": outcomes,
                "population": population,
                "study_design": study_design,
                "years": years,
                "max_results": max_results
            }
        )
    
    async def analyze_contradictions(
        self,
        query: str,
        max_results: int = 20,
        threshold: float = 0.7,
        use_biomedlm: bool = True,
        use_tsmixer: bool = False,
        use_lorentz: bool = False
    ) -> APIResponse:
        """
        Analyze contradictions in medical literature.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            threshold: Contradiction detection threshold
            use_biomedlm: Whether to use BioMedLM for contradiction detection
            use_tsmixer: Whether to use TSMixer for temporal contradiction detection
            use_lorentz: Whether to use Lorentz embeddings for hierarchical contradiction detection
            
        Returns:
            API response with contradiction analysis
        """
        return await self._request(
            "POST",
            "/analysis/contradictions",
            data={
                "query": query,
                "max_results": max_results,
                "threshold": threshold,
                "use_biomedlm": use_biomedlm,
                "use_tsmixer": use_tsmixer,
                "use_lorentz": use_lorentz
            }
        )
    
    async def analyze_cap(self) -> APIResponse:
        """
        Analyze Community-Acquired Pneumonia (CAP) literature.
        
        Returns:
            API response with CAP analysis
        """
        return await self._request("GET", "/analysis/cap")
    
    async def screen_articles(
        self,
        query: str,
        max_results: int = 20,
        stage: str = "screening",
        criteria: Optional[Dict[str, List[str]]] = None
    ) -> APIResponse:
        """
        Screen articles according to PRISMA guidelines.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            stage: Screening stage (identification, screening, eligibility)
            criteria: Custom screening criteria
            
        Returns:
            API response with screening results
        """
        return await self._request(
            "POST",
            "/screening/prisma",
            data={
                "query": query,
                "max_results": max_results,
                "stage": stage,
                "criteria": criteria
            }
        )
    
    async def assess_bias(
        self,
        query: str,
        max_results: int = 20,
        domains: Optional[List[str]] = None
    ) -> APIResponse:
        """
        Assess risk of bias in articles.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            domains: Bias domains to assess
            
        Returns:
            API response with bias assessment
        """
        return await self._request(
            "POST",
            "/screening/bias-assessment",
            data={
                "query": query,
                "max_results": max_results,
                "domains": domains
            }
        )
    
    async def create_knowledge_base(
        self,
        name: str,
        query: str,
        update_schedule: str = "weekly"
    ) -> APIResponse:
        """
        Create a new knowledge base.
        
        Args:
            name: Knowledge base name
            query: Search query
            update_schedule: Update schedule (daily, weekly, monthly)
            
        Returns:
            API response with knowledge base information
        """
        return await self._request(
            "POST",
            "/knowledge-base",
            data={
                "name": name,
                "query": query,
                "update_schedule": update_schedule
            }
        )
    
    async def list_knowledge_bases(self) -> APIResponse:
        """
        List all knowledge bases.
        
        Returns:
            API response with knowledge base list
        """
        return await self._request("GET", "/knowledge-base")
    
    async def get_knowledge_base(self, kb_id: str) -> APIResponse:
        """
        Get a knowledge base by ID.
        
        Args:
            kb_id: Knowledge base ID
            
        Returns:
            API response with knowledge base information
        """
        return await self._request("GET", f"/knowledge-base/{kb_id}")
    
    async def update_knowledge_base(self, kb_id: str) -> APIResponse:
        """
        Update a knowledge base.
        
        Args:
            kb_id: Knowledge base ID
            
        Returns:
            API response with update status
        """
        return await self._request("POST", f"/knowledge-base/{kb_id}/update")
    
    async def delete_knowledge_base(self, kb_id: str) -> APIResponse:
        """
        Delete a knowledge base.
        
        Args:
            kb_id: Knowledge base ID
            
        Returns:
            API response with deletion status
        """
        return await self._request("DELETE", f"/knowledge-base/{kb_id}")
    
    async def export_results(
        self,
        format: str,
        result_id: Optional[str] = None,
        query: Optional[str] = None,
        max_results: int = 20
    ) -> APIResponse:
        """
        Export search results.
        
        Args:
            format: Export format (json, csv, excel, pdf)
            result_id: Result ID (optional)
            query: Search query (optional)
            max_results: Maximum number of results
            
        Returns:
            API response with export information
        """
        data = {}
        if result_id:
            data["result_id"] = result_id
        elif query:
            data["query"] = query
            data["max_results"] = max_results
        else:
            raise ValueError("Either result_id or query must be provided")
        
        return await self._request(
            "POST",
            f"/export/{format}",
            data=data
        )
