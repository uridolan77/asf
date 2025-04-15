"""
Medical Search API service for integrating with the search_service.py module
from the medical package.
"""
import os
import sys
import logging
from typing import Dict, Any, List, Optional, Union
from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

# Add the parent directory to sys.path to import the medical module
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import from the medical module
from asf.medical.services.search_service import SearchService, SearchMethod
from asf.medical.clients.ncbi.ncbi_client import NCBIClient
from asf.medical.clients.clinical_trials_client import ClinicalTrialsClient
from asf.medical.storage.repositories.result_repository import ResultRepository
from asf.medical.storage.repositories.query_repository import QueryRepository
from asf.medical.graph.graph_rag import GraphRAG

from config.database import get_db
from models.user import User

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalSearchService:
    """
    Service for interacting with the medical module's SearchService.
    This provides a bridge between the BO frontend and the Medical Research search functionality.
    """
    def __init__(self):
        """Initialize with direct access to the medical module's SearchService"""
        # Initialize the required dependencies for SearchService
        self.ncbi_client = NCBIClient()
        self.clinical_trials_client = ClinicalTrialsClient()
        self.query_repository = QueryRepository()
        self.result_repository = ResultRepository()
        
        # Optionally initialize GraphRAG if available
        try:
            self.graph_rag = GraphRAG()
            has_graph_rag = True
        except Exception as e:
            logger.warning(f"Failed to initialize GraphRAG: {str(e)}")
            self.graph_rag = None
            has_graph_rag = False
        
        self.search_service = SearchService(
            ncbi_client=self.ncbi_client,
            clinical_trials_client=self.clinical_trials_client,
            query_repository=self.query_repository,
            result_repository=self.result_repository,
            graph_rag=self.graph_rag if has_graph_rag else None
        )
        
    async def search(
        self, 
        query: str, 
        max_results: int = 100, 
        page: int = 1, 
        page_size: int = 20,
        user_id: Optional[int] = None,
        search_method: str = "pubmed",
        use_graph_rag: bool = False
    ) -> Dict[str, Any]:
        """
        Search for medical literature.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            page: Page number
            page_size: Number of results per page
            user_id: BO user ID
            search_method: Search method (pubmed, clinical_trials, graph_rag)
            use_graph_rag: Whether to use GraphRAG for enhanced search
            
        Returns:
            Search results
        """
        try:
            results = await self.search_service.search(
                query=query,
                max_results=max_results,
                page=page,
                page_size=page_size,
                user_id=user_id,
                search_method=search_method,
                use_graph_rag=use_graph_rag
            )
            
            return {
                "success": True,
                "message": f"Found {results.get('total_count', 0)} results for query: {query}",
                "data": results
            }
        except Exception as e:
            logger.error(f"Error searching medical literature: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to search medical literature: {str(e)}",
                "data": None
            }
    
    async def search_pico(
        self,
        condition: str,
        interventions: List[str] = [],
        outcomes: List[str] = [],
        population: Optional[str] = None,
        study_design: Optional[str] = None,
        years: int = 5,
        max_results: int = 100,
        page: int = 1,
        page_size: int = 20,
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Search for medical literature using PICO framework.
        
        Args:
            condition: Medical condition
            interventions: List of interventions
            outcomes: List of outcomes
            population: Patient population
            study_design: Study design
            years: Number of years to search
            max_results: Maximum number of results to return
            page: Page number
            page_size: Number of results per page
            user_id: BO user ID
            
        Returns:
            Search results using PICO framework
        """
        try:
            results = await self.search_service.search_pico(
                condition=condition,
                interventions=interventions,
                outcomes=outcomes,
                population=population,
                study_design=study_design,
                years=years,
                max_results=max_results,
                page=page,
                page_size=page_size,
                user_id=user_id
            )
            
            return {
                "success": True,
                "message": f"Found {results.get('total_count', 0)} results for PICO search on condition: {condition}",
                "data": results
            }
        except Exception as e:
            logger.error(f"Error executing PICO search: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to execute PICO search: {str(e)}",
                "data": None
            }
    
    async def get_result(self, result_id: str) -> Dict[str, Any]:
        """
        Get search results by ID.
        
        Args:
            result_id: Search result ID
            
        Returns:
            Search results
        """
        try:
            result = await self.search_service.get_result(result_id=result_id)
            
            if not result:
                return {
                    "success": False,
                    "message": f"Search result with ID '{result_id}' not found",
                    "data": None
                }
                
            return {
                "success": True,
                "message": "Search result retrieved successfully",
                "data": result
            }
        except Exception as e:
            logger.error(f"Error retrieving search result: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to retrieve search result: {str(e)}",
                "data": None
            }
    
    def get_available_search_methods(self) -> Dict[str, Any]:
        """
        Get available search methods.
        
        Returns:
            List of available search methods
        """
        methods = [{"id": "pubmed", "name": "PubMed"}, 
                  {"id": "clinical_trials", "name": "ClinicalTrials.gov"}]
        
        if self.search_service.is_graph_rag_available():
            methods.append({"id": "graph_rag", "name": "GraphRAG (Enhanced)"})
            
        return {
            "success": True,
            "message": f"Found {len(methods)} available search methods",
            "data": {
                "methods": methods,
                "default_method": "pubmed"
            }
        }

# Dependency to get the medical search service
def get_medical_search_service() -> MedicalSearchService:
    """Factory function to create and provide a MedicalSearchService instance."""
    return MedicalSearchService()