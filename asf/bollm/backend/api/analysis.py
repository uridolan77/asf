"""
Analysis API endpoints for the BO backend.

This module provides endpoints for analyzing medical literature,
including contradiction detection and specialized analyses.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Body
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import httpx
import logging
import os
from datetime import datetime

from .auth import get_current_user
from asf.bollm.backend.models.user import User
from .utils import handle_api_error

router = APIRouter(prefix="/api/medical/analysis", tags=["analysis"])

logger = logging.getLogger(__name__)

# Environment variables
MEDICAL_API_URL = os.getenv("MEDICAL_API_URL", "http://localhost:8000")

# Models
class ContradictionAnalysisRequest(BaseModel):
    query: str
    max_results: int = 20
    threshold: float = 0.7
    use_biomedlm: bool = True
    use_tsmixer: bool = False
    use_lorentz: bool = False

class AnalysisHistoryRequest(BaseModel):
    page: int = 1
    page_size: int = 10
    analysis_type: Optional[str] = None

@router.post("/contradictions", response_model=Dict[str, Any])
async def analyze_contradictions(
    request: ContradictionAnalysisRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Analyze contradictions in medical literature based on a query.
    
    This endpoint performs contradiction analysis on medical literature matching
    the provided query. It identifies statements that contradict each other and
    provides explanations for the contradictions.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{MEDICAL_API_URL}/api/analysis/contradictions",
                json=request.dict(),
                headers={"Authorization": f"Bearer {current_user.token}"}
            )
            
            if response.status_code != 200:
                return handle_api_error(response)
                
            return response.json()
    except Exception as e:
        logger.error(f"Error analyzing contradictions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze contradictions: {str(e)}"
        )

@router.get("/cap", response_model=Dict[str, Any])
async def analyze_cap(
    current_user: User = Depends(get_current_user)
):
    """
    Perform CAP (Community-Acquired Pneumonia) analysis.
    
    This endpoint performs analysis on Community-Acquired Pneumonia literature
    to identify treatment patterns, patient populations, and outcomes.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{MEDICAL_API_URL}/api/analysis/cap",
                headers={"Authorization": f"Bearer {current_user.token}"}
            )
            
            if response.status_code != 200:
                return handle_api_error(response)
                
            return response.json()
    except Exception as e:
        logger.error(f"Error performing CAP analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to perform CAP analysis: {str(e)}"
        )

@router.get("/{analysis_id}", response_model=Dict[str, Any])
async def get_analysis(
    analysis_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Retrieve a previously performed analysis by ID.
    
    This endpoint retrieves the results of a previously performed analysis
    using its unique identifier.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{MEDICAL_API_URL}/api/analysis/{analysis_id}",
                headers={"Authorization": f"Bearer {current_user.token}"}
            )
            
            if response.status_code != 200:
                return handle_api_error(response)
                
            return response.json()
    except Exception as e:
        logger.error(f"Error retrieving analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve analysis: {str(e)}"
        )

@router.get("/history", response_model=Dict[str, Any])
async def get_analysis_history(
    page: int = 1,
    page_size: int = 10,
    analysis_type: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Retrieve analysis history for the current user.
    
    This endpoint retrieves the history of analyses performed by the current user,
    with optional filtering by analysis type.
    """
    try:
        params = {"page": page, "page_size": page_size}
        if analysis_type:
            params["analysis_type"] = analysis_type
            
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{MEDICAL_API_URL}/api/analysis/history",
                params=params,
                headers={"Authorization": f"Bearer {current_user.token}"}
            )
            
            if response.status_code != 200:
                return handle_api_error(response)
                
            return response.json()
    except Exception as e:
        logger.error(f"Error retrieving analysis history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve analysis history: {str(e)}"
        )
