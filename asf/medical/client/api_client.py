"""
Client library for the Medical Research Synthesizer API.
This module provides a client library for interacting with the Medical Research Synthesizer API.
"""
import json
import logging
import httpx
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
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
        Initialize the client.
        Args:
            base_url: Base URL of the API
            api_version: API version
            token: Authentication token
        await self.client.aclose()
    """        
    async def _request(
        self, 
        method: str, 
        path: str, 
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> APIResponse:
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
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
        try:
            response = await self.client.post(
                "/auth/token",
                data={"username": email, "password": password}
            )
            response.raise_for_status()
            data = response.json()
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
        return await self._request("GET", "/auth/me")
    async def search(
        self, 
        query: str, 
        max_results: int = 20
    ) -> APIResponse:
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
        return await self._request("GET", "/analysis/cap")
    async def screen_articles(
        self,
        query: str,
        max_results: int = 20,
        stage: str = "screening",
        criteria: Optional[Dict[str, List[str]]] = None
    ) -> APIResponse:
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
        return await self._request("GET", "/knowledge-base")
    async def get_knowledge_base(self, kb_id: str) -> APIResponse:
        return await self._request("GET", f"/knowledge-base/{kb_id}")
    async def update_knowledge_base(self, kb_id: str) -> APIResponse:
        return await self._request("POST", f"/knowledge-base/{kb_id}/update")
    async def delete_knowledge_base(self, kb_id: str) -> APIResponse:
        return await self._request("DELETE", f"/knowledge-base/{kb_id}")
    async def export_results(
        self,
        format: str,
        result_id: Optional[str] = None,
        query: Optional[str] = None,
        max_results: int = 20
    ) -> APIResponse:
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