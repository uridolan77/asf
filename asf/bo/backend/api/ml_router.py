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
    # Reuse the implementation from the medical ML router
    return await get_ml_services_status(current_user)
