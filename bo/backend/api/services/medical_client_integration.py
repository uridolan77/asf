"""
Integration service for the medical client API.
"""
import os
import sys
import logging
import json
import httpx
from typing import Dict, Any, Optional, List
from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

# Import our own models
from models.user import User
from config.database import get_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalClientIntegration:
    """
    Integration service for the Medical Client API.
    This service acts as a bridge between our frontend and the medical API.
    """
    def __init__(self, api_base_url: str = None, api_key: str = None):
        # Get configuration from environment variables if not provided
        self.api_base_url = api_base_url or os.getenv("MEDICAL_API_BASE_URL", "http://localhost:8008/api")
        self.api_key = api_key or os.getenv("MEDICAL_API_KEY", "default_api_key")
        
    async def _make_request(self, method: str, endpoint: str, data: Dict[str, Any] = None, 
                           params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Make a request to the medical API.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint
            data: Request body data
            params: Query parameters
            
        Returns:
            Response from the API
        """
        url = f"{self.api_base_url}/{endpoint.lstrip('/')}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            async with httpx.AsyncClient() as client:
                if method.upper() == "GET":
                    response = await client.get(url, headers=headers, params=params)
                elif method.upper() == "POST":
                    response = await client.post(url, headers=headers, json=data)
                elif method.upper() == "PUT":
                    response = await client.put(url, headers=headers, json=data)
                elif method.upper() == "DELETE":
                    response = await client.delete(url, headers=headers, params=params)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                response.raise_for_status()
                return response.json()
                
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            try:
                error_detail = e.response.json().get("detail", str(e))
            except:
                error_detail = str(e)
                
            if status_code == 401:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication failed with the medical API"
                )
            elif status_code == 403:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Not authorized to access this resource in the medical API"
                )
            elif status_code == 404:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Resource not found in the medical API: {error_detail}"
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Medical API error: {error_detail}"
                )
                
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Could not connect to the medical API: {str(e)}"
            )
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error communicating with the medical API: {str(e)}"
            )
    
    # Knowledge Base Management Methods
    async def create_knowledge_base(self, db: Session, user_id: int, name: str, 
                                   query: str, update_schedule: str) -> Dict[str, Any]:
        """
        Create a new knowledge base.
        
        Args:
            db: Database session
            user_id: User ID
            name: Knowledge base name
            query: Search query
            update_schedule: Update frequency (daily, weekly, monthly)
            
        Returns:
            Response from the API with knowledge base details
        """
        # Log the KB creation in our system
        # Here we could store the mapping between our users and the KB
        # in our database for access control and user-specific views
        
        # Call the medical API
        return await self._make_request(
            method="POST",
            endpoint="/medical/knowledge-base",
            data={
                "name": name,
                "query": query,
                "update_schedule": update_schedule
            }
        )
    
    async def list_knowledge_bases(self, db: Session, user_id: int) -> Dict[str, Any]:
        """
        List all knowledge bases.
        
        Args:
            db: Database session
            user_id: User ID
            
        Returns:
            Response from the API with list of knowledge bases
        """
        # Get knowledge bases from the medical API
        return await self._make_request(
            method="GET",
            endpoint="/medical/knowledge-base"
        )
    
    async def get_knowledge_base(self, db: Session, user_id: int, kb_id: str) -> Dict[str, Any]:
        """
        Get details of a specific knowledge base.
        
        Args:
            db: Database session
            user_id: User ID
            kb_id: Knowledge base ID
            
        Returns:
            Response from the API with knowledge base details
        """
        # Get knowledge base details from the medical API
        return await self._make_request(
            method="GET",
            endpoint=f"/medical/knowledge-base/{kb_id}"
        )
    
    async def update_knowledge_base(self, db: Session, user_id: int, kb_id: str) -> Dict[str, Any]:
        """
        Update a knowledge base.
        
        Args:
            db: Database session
            user_id: User ID
            kb_id: Knowledge base ID
            
        Returns:
            Response from the API with update status
        """
        # Update the knowledge base through the medical API
        return await self._make_request(
            method="POST",
            endpoint=f"/medical/knowledge-base/{kb_id}/update"
        )
    
    async def delete_knowledge_base(self, db: Session, user_id: int, kb_id: str) -> Dict[str, Any]:
        """
        Delete a knowledge base.
        
        Args:
            db: Database session
            user_id: User ID
            kb_id: Knowledge base ID
            
        Returns:
            Response from the API with deletion status
        """
        # Delete the knowledge base through the medical API
        return await self._make_request(
            method="DELETE",
            endpoint=f"/medical/knowledge-base/{kb_id}"
        )
    
    # Additional Medical Research Methods
    
    async def search_medical(self, query: str, max_results: int = 20) -> Dict[str, Any]:
        """
        Search for medical research.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            Response from the API with search results
        """
        return await self._make_request(
            method="POST",
            endpoint="/medical/search",
            data={
                "query": query,
                "max_results": max_results
            }
        )
    
    async def search_pico(self, condition: str, interventions: List[str], 
                         outcomes: List[str], population: Optional[str] = None,
                         study_design: Optional[str] = None, 
                         years: Optional[int] = None,
                         max_results: int = 20) -> Dict[str, Any]:
        """
        Search for medical research using PICO framework.
        
        Args:
            condition: Medical condition
            interventions: List of interventions
            outcomes: List of outcomes
            population: Target population
            study_design: Study design type
            years: Publication years to include
        """
    async def update_knowledge_base(self, db: Session, user_id: int, kb_id: str) -> Dict[str, Any]:
        """
        Trigger an update for a knowledge base.
        
        Args:
            db: Database session
            user_id: User ID
            kb_id: Knowledge base ID
            
        Returns:
            Update status
        """
        await self._authenticate(db, user_id)
        
        result = await self._request("POST", f"/knowledge-base/{kb_id}/update")
        
        # Log the update for auditing
        logger.info(f"Knowledge base update triggered: {kb_id} by user_id: {user_id}")
        
        return {
            "success": result.get("success", False),
            "message": result.get("message", ""),
            "data": result.get("data", {})
        }
    
    async def delete_knowledge_base(self, db: Session, user_id: int, kb_id: str) -> Dict[str, Any]:
        """
        Delete a knowledge base.
        
        Args:
            db: Database session
            user_id: User ID
            kb_id: Knowledge base ID
            
        Returns:
            Deletion status
        """
        await self._authenticate(db, user_id)
        
        result = await self._request("DELETE", f"/knowledge-base/{kb_id}")
        
        # Log the deletion for auditing
        logger.info(f"Knowledge base deleted: {kb_id} by user_id: {user_id}")
        
        return {
            "success": result.get("success", False),
            "message": result.get("message", ""),
            "data": result.get("data", {})
        }
    
    async def search_pico(
        self,
        db: Session,
        user_id: int,
        condition: str,
        interventions: List[str],
        outcomes: List[str],
        population: Optional[str] = None,
        study_design: Optional[str] = None,
        years: Optional[int] = None,
        max_results: int = 20
    ) -> Dict[str, Any]:
        """
        Perform a PICO search using the Medical Research API.
        
        Args:
            db: Database session
            user_id: User ID
            condition: Medical condition
            interventions: List of interventions
            outcomes: List of outcomes
            population: Target population
            study_design: Study design filter
            years: Publication years filter
            max_results: Maximum number of results
            
        Returns:
            Search results
        """
        await self._authenticate(db, user_id)
        
        result = await self._request(
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
        
        # Log the search for analytics
        logger.info(f"PICO search performed by user_id: {user_id} for condition: {condition}")
        
        return {
            "success": result.get("success", False),
            "message": result.get("message", ""),
            "data": result.get("data", {})
        }

async def get_medical_client(
    db: Session = Depends(get_db),
    current_user: User = None
) -> MedicalClientIntegration:
    """
    Factory function to create and provide a Medical client integration instance.
    This is designed to be used as a FastAPI dependency.
    
    Args:
        db: Database session
        current_user: Currently authenticated user
        
    Returns:
        Configured MedicalClientIntegration instance
    """
    async with MedicalClientIntegration() as client:
        yield client