"""
Medical Search API router for BO backend.
"""
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from api.services.medical_search_service import MedicalSearchService, get_medical_search_service
from config.database import get_db
from models.user import User
from api.dependencies import get_current_user

router = APIRouter(
    prefix="/api/medical/search",
    tags=["medical-search"],
    responses={404: {"description": "Not found"}},
)

class PICOSearchRequest(BaseModel):
    """PICO search request model."""
    condition: str = Field(..., description="Medical condition")
    interventions: List[str] = Field(default=[], description="List of interventions")
    outcomes: List[str] = Field(default=[], description="List of outcomes")
    population: Optional[str] = Field(None, description="Patient population")
    study_design: Optional[str] = Field(None, description="Study design")
    years: int = Field(5, description="Number of years to search")
    max_results: int = Field(100, description="Maximum number of results")
    page: int = Field(1, description="Page number")
    page_size: int = Field(20, description="Number of results per page")

@router.get("/methods")
async def get_search_methods(
    search_service: MedicalSearchService = Depends(get_medical_search_service)
):
    """
    Get available search methods.
    """
    return search_service.get_available_search_methods()

@router.get("/")
async def search_medical_literature(
    query: str = Query(..., description="Search query"),
    max_results: int = Query(100, description="Maximum number of results"),
    page: int = Query(1, description="Page number"),
    page_size: int = Query(20, description="Results per page"),
    search_method: str = Query("pubmed", description="Search method (pubmed, clinical_trials, graph_rag)"),
    use_graph_rag: bool = Query(False, description="Use GraphRAG for enhanced search"),
    search_service: MedicalSearchService = Depends(get_medical_search_service),
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    Search for medical literature based on a text query.
    """
    user_id = current_user.id if current_user else None
    result = await search_service.search(
        query=query,
        max_results=max_results,
        page=page,
        page_size=page_size,
        user_id=user_id,
        search_method=search_method,
        use_graph_rag=use_graph_rag
    )
    
    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result["message"]
        )
        
    return result

@router.post("/pico")
async def search_medical_literature_pico(
    request: PICOSearchRequest,
    search_service: MedicalSearchService = Depends(get_medical_search_service),
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    Search for medical literature using PICO framework.
    """
    user_id = current_user.id if current_user else None
    result = await search_service.search_pico(
        condition=request.condition,
        interventions=request.interventions,
        outcomes=request.outcomes,
        population=request.population,
        study_design=request.study_design,
        years=request.years,
        max_results=request.max_results,
        page=request.page,
        page_size=request.page_size,
        user_id=user_id
    )
    
    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result["message"]
        )
        
    return result

@router.get("/results/{result_id}")
async def get_search_result(
    result_id: str = Path(..., description="Search result ID"),
    search_service: MedicalSearchService = Depends(get_medical_search_service)
):
    """
    Get a search result by ID.
    """
    result = await search_service.get_result(result_id=result_id)
    
    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=result["message"]
        )
        
    return result