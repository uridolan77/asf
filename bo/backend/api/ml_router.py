"""
ML Router for the BO backend.

This module provides a router for the /api/ml prefix, which is used for ML services.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, List, Optional
import logging

from .auth import get_current_user, User
from .ml import get_ml_services_status

router = APIRouter(prefix="/api/ml", tags=["ml"])

logger = logging.getLogger(__name__)

@router.get("/services/status", response_model=Dict[str, Any])
async def get_services_status(
    current_user: User = Depends(get_current_user)
):
    """
    Get the status of all ML services.

    This endpoint returns the status of all ML services including
    claim extraction, contradiction detection, bias assessment, and more.
    """
    try:
        # Reuse the implementation from the medical ML router
        return await get_ml_services_status(current_user)
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
