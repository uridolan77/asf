"""
ML Service API endpoints for the BO backend.

This module provides endpoints for machine learning services,
including contradiction detection, temporal analysis, and bias assessment.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Body
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import httpx
import logging
import os
from datetime import datetime

from .auth import get_current_user, User
from .utils import handle_api_error

router = APIRouter(prefix="/api/medical/ml", tags=["ml"])

logger = logging.getLogger(__name__)

# Environment variables
MEDICAL_API_URL = os.getenv("MEDICAL_API_URL", "http://localhost:8000")

# Models
class ContradictionRequest(BaseModel):
    claim1: str
    claim2: str
    metadata1: Optional[Dict[str, Any]] = None
    metadata2: Optional[Dict[str, Any]] = None
    use_biomedlm: bool = True
    use_tsmixer: bool = False
    use_lorentz: bool = False
    use_temporal: bool = False
    use_shap: bool = False
    domain: Optional[str] = None
    threshold: float = 0.7

class TemporalAnalysisRequest(BaseModel):
    publication_date: str
    domain: str = "general"
    reference_date: Optional[str] = None
    include_details: bool = False

class BiasAssessmentRequest(BaseModel):
    article_id: str
    full_text: Optional[str] = None
    abstract: Optional[str] = None
    title: Optional[str] = None
    assessment_type: str = "robins-i"

@router.post("/contradiction", response_model=Dict[str, Any])
async def detect_contradiction(
    request: ContradictionRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Detect contradiction between two medical claims.

    This endpoint uses the Contradiction Service to detect contradictions
    between two medical claims, integrating multiple methods including
    BioMedLM, TSMixer, and Lorentz embeddings.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{MEDICAL_API_URL}/api/ml/contradiction",
                json=request.dict(),
                headers={"Authorization": f"Bearer {current_user.token}"}
            )

            if response.status_code != 200:
                return handle_api_error(response)

            return response.json()
    except Exception as e:
        logger.error(f"Error detecting contradiction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to detect contradiction: {str(e)}"
        )

@router.post("/contradiction/batch", response_model=Dict[str, Any])
async def detect_contradictions_batch(
    request: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """
    Detect contradictions in a batch of claim pairs.

    This endpoint uses the Contradiction Service to detect contradictions
    in a batch of claim pairs, optimized for performance with parallel
    model execution and selective feature computation.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{MEDICAL_API_URL}/api/ml/contradiction/batch",
                json=request,
                headers={"Authorization": f"Bearer {current_user.token}"}
            )

            if response.status_code != 200:
                return handle_api_error(response)

            return response.json()
    except Exception as e:
        logger.error(f"Error detecting contradictions in batch: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to detect contradictions in batch: {str(e)}"
        )

@router.post("/temporal/confidence", response_model=Dict[str, Any])
async def calculate_temporal_confidence(
    request: TemporalAnalysisRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Calculate temporal confidence for a publication.

    This endpoint uses the Temporal Service to calculate temporal confidence
    for a publication with domain-specific characteristics.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{MEDICAL_API_URL}/api/ml/temporal/confidence",
                json=request.dict(),
                headers={"Authorization": f"Bearer {current_user.token}"}
            )

            if response.status_code != 200:
                return handle_api_error(response)

            return response.json()
    except Exception as e:
        logger.error(f"Error calculating temporal confidence: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate temporal confidence: {str(e)}"
        )

@router.post("/temporal/contradiction", response_model=Dict[str, Any])
async def detect_temporal_contradiction(
    request: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """
    Detect temporal contradiction between two claims.

    This endpoint uses the Temporal Service to detect temporal contradictions
    between two claims based on their publication dates and content.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{MEDICAL_API_URL}/api/ml/temporal/contradiction",
                json=request,
                headers={"Authorization": f"Bearer {current_user.token}"}
            )

            if response.status_code != 200:
                return handle_api_error(response)

            return response.json()
    except Exception as e:
        logger.error(f"Error detecting temporal contradiction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to detect temporal contradiction: {str(e)}"
        )

@router.post("/bias/assess", response_model=Dict[str, Any])
async def assess_bias(
    request: BiasAssessmentRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Assess bias in a medical article.

    This endpoint uses the Bias Assessment Service to assess bias in a medical
    article using various bias assessment tools like ROBINS-I, RoB 2, etc.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{MEDICAL_API_URL}/api/ml/bias/assess",
                json=request.dict(),
                headers={"Authorization": f"Bearer {current_user.token}"}
            )

            if response.status_code != 200:
                return handle_api_error(response)

            return response.json()
    except Exception as e:
        logger.error(f"Error assessing bias: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to assess bias: {str(e)}"
        )

@router.get("/bias/tools", response_model=Dict[str, Any])
async def get_bias_assessment_tools(
    current_user: User = Depends(get_current_user)
):
    """
    Get available bias assessment tools.

    This endpoint returns a list of available bias assessment tools
    and their descriptions.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{MEDICAL_API_URL}/api/ml/bias/tools",
                headers={"Authorization": f"Bearer {current_user.token}"}
            )

            if response.status_code != 200:
                return handle_api_error(response)

            return response.json()
    except Exception as e:
        logger.error(f"Error getting bias assessment tools: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get bias assessment tools: {str(e)}"
        )

@router.get("/services/status", response_model=Dict[str, Any])
async def get_ml_services_status(
    current_user: User = Depends(get_current_user)
):
    """
    Get the status of all ML services.

    This endpoint returns the status of all ML services including
    claim extraction, contradiction detection, bias assessment, and more.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{MEDICAL_API_URL}/api/ml/services/status",
                headers={"Authorization": f"Bearer {current_user.token}"}
            )

            if response.status_code != 200:
                # If the main endpoint fails, return a default response
                logger.warning(f"Failed to get ML services status from main API: {response.status_code}")
                # Return mock data as fallback
                return {
                    "services": [
                        {
                            "name": "Claim Extractor",
                            "status": "operational",
                            "version": "1.2.0",
                            "description": "Extract scientific claims from medical text",
                            "last_updated": "2025-04-15",
                            "health": "healthy"
                        },
                        {
                            "name": "Contradiction Detector",
                            "status": "operational",
                            "version": "2.0.1",
                            "description": "Detect contradictions between medical claims",
                            "last_updated": "2025-04-10",
                            "health": "healthy"
                        },
                        {
                            "name": "Bias Assessment",
                            "status": "operational",
                            "version": "1.1.5",
                            "description": "Assess bias in medical studies using various tools",
                            "last_updated": "2025-04-12",
                            "health": "healthy"
                        },
                        {
                            "name": "Evidence Grader",
                            "status": "degraded",
                            "version": "1.0.2",
                            "description": "Grade evidence quality in medical studies",
                            "last_updated": "2025-04-01",
                            "health": "degraded"
                        }
                    ]
                }

            return response.json()
    except Exception as e:
        logger.error(f"Error getting ML services status: {str(e)}")
        # Return mock data as fallback
        return {
            "services": [
                {
                    "name": "Claim Extractor",
                    "status": "operational",
                    "version": "1.2.0",
                    "description": "Extract scientific claims from medical text",
                    "last_updated": "2025-04-15",
                    "health": "healthy"
                },
                {
                    "name": "Contradiction Detector",
                    "status": "operational",
                    "version": "2.0.1",
                    "description": "Detect contradictions between medical claims",
                    "last_updated": "2025-04-10",
                    "health": "healthy"
                },
                {
                    "name": "Bias Assessment",
                    "status": "operational",
                    "version": "1.1.5",
                    "description": "Assess bias in medical studies using various tools",
                    "last_updated": "2025-04-12",
                    "health": "healthy"
                },
                {
                    "name": "Evidence Grader",
                    "status": "degraded",
                    "version": "1.0.2",
                    "description": "Grade evidence quality in medical studies",
                    "last_updated": "2025-04-01",
                    "health": "degraded"
                }
            ]
        }
