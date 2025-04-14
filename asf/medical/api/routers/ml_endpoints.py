"""
ML model endpoints for the Medical Research Synthesizer API.

This module provides FastAPI routers for machine learning model endpoints
including claim extraction, biomedical language models, and other ML features.
"""

from fastapi import APIRouter, Depends
from asf.medical.api.claim_extractor_api import claim_api

# Create the router
router = APIRouter(prefix="/api/ml", tags=["ml-models"])

# Include the claim extractor API blueprint
router.include_router(claim_api)

# Define root route for ML models section
@router.get("/")
async def ml_models_root():
    """
    Root endpoint for ML models API.
    
    Returns:
        Information about available ML model endpoints
    """
    return {
        "status": "ok", 
        "models": [
            {
                "name": "claim_extractor",
                "description": "Extract scientific claims from medical text",
                "endpoints": [
                    "/extract-claims",
                    "/extract-claims-from-abstract",
                    "/extract-claims-from-full-text",
                    "/batch-extract-claims"
                ]
            }
        ]
    }