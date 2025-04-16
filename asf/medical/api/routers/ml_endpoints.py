"""
ML model endpoints for the Medical Research Synthesizer API.

This module provides FastAPI routers for machine learning model endpoints
including claim extraction, biomedical language models, and other ML features.
"""

from typing import List, Optional
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException
from ..claim_extractor_api import claim_api
from ...core.logging_config import get_logger
from ..dependencies import get_current_active_user
from ...storage.models import User

logger = get_logger(__name__)

# Define the models for ML service status
class MLServiceStatus(BaseModel):
    """Status of an ML service"""
    name: str
    status: str
    version: str
    description: str
    last_updated: str
    health: str
    metrics: Optional[dict] = None

class MLServicesStatusResponse(BaseModel):
    """Response model for ML services status"""
    services: List[MLServiceStatus]

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
            },
            {
                "name": "ml_services",
                "description": "ML services status and management",
                "endpoints": [
                    "/services/status",
                    "/services/metrics"
                ]
            }
        ]
    }

@router.get("/services/status", response_model=MLServicesStatusResponse)
async def get_ml_services_status(current_user: User = Depends(get_current_active_user)):
    """
    Get the status of all ML services.
    
    This endpoint returns the status of all ML services including 
    claim extraction, contradiction detection, bias assessment, and more.
    
    Args:
        current_user: The authenticated user
        
    Returns:
        MLServicesStatusResponse: Status of all ML services
    """
    try:
        # In a real implementation, this would query the actual services
        # For now, we return mock data
        services = [
            MLServiceStatus(
                name="Claim Extractor",
                status="operational",
                version="1.2.0",
                description="Extract scientific claims from medical text",
                last_updated="2025-04-15",
                health="healthy",
                metrics={
                    "avg_response_time": 0.45,
                    "requests_per_minute": 12.5,
                    "error_rate": 0.02
                }
            ),
            MLServiceStatus(
                name="Contradiction Detector",
                status="operational",
                version="2.0.1",
                description="Detect contradictions between medical claims",
                last_updated="2025-04-10",
                health="healthy",
                metrics={
                    "avg_response_time": 1.2,
                    "requests_per_minute": 8.3,
                    "error_rate": 0.05
                }
            ),
            MLServiceStatus(
                name="Bias Assessment",
                status="operational",
                version="1.1.5",
                description="Assess bias in medical studies using various tools",
                last_updated="2025-04-12",
                health="healthy",
                metrics={
                    "avg_response_time": 0.9,
                    "requests_per_minute": 5.7,
                    "error_rate": 0.03
                }
            ),
            MLServiceStatus(
                name="Evidence Grader",
                status="degraded",
                version="1.0.2",
                description="Grade evidence quality in medical studies",
                last_updated="2025-04-01",
                health="degraded",
                metrics={
                    "avg_response_time": 2.1,
                    "requests_per_minute": 3.2,
                    "error_rate": 0.08
                }
            )
        ]
        
        return MLServicesStatusResponse(services=services)
    except Exception as e:
        logger.error(f"Error getting ML services status: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving ML services status")

@router.get("/services/metrics")
async def get_ml_services_metrics(current_user: User = Depends(get_current_active_user)):
    """
    Get detailed metrics for all ML services.
    
    This endpoint returns detailed performance metrics for all ML services.
    
    Args:
        current_user: The authenticated user
        
    Returns:
        Dict: Detailed metrics for all ML services
    """
    try:
        # In a real implementation, this would query the actual metrics
        # For now, we return mock data
        return {
            "status": "ok",
            "period": "last_24h",
            "services": {
                "claim_extractor": {
                    "request_count": 1250,
                    "avg_response_time": 0.45,
                    "p95_response_time": 0.98,
                    "error_rate": 0.02,
                    "throughput": 12.5,
                    "resource_utilization": {
                        "cpu": 45.2,
                        "memory": 1.2,
                        "gpu": 35.8
                    }
                },
                "contradiction_detector": {
                    "request_count": 850,
                    "avg_response_time": 1.2,
                    "p95_response_time": 2.5,
                    "error_rate": 0.05,
                    "throughput": 8.3,
                    "resource_utilization": {
                        "cpu": 65.3,
                        "memory": 2.8,
                        "gpu": 72.1
                    }
                },
                "bias_assessment": {
                    "request_count": 580,
                    "avg_response_time": 0.9,
                    "p95_response_time": 1.8,
                    "error_rate": 0.03,
                    "throughput": 5.7,
                    "resource_utilization": {
                        "cpu": 38.5,
                        "memory": 1.5,
                        "gpu": 42.3
                    }
                },
                "evidence_grader": {
                    "request_count": 320,
                    "avg_response_time": 2.1,
                    "p95_response_time": 4.2,
                    "error_rate": 0.08,
                    "throughput": 3.2,
                    "resource_utilization": {
                        "cpu": 52.6,
                        "memory": 2.1,
                        "gpu": 48.7
                    }
                }
            }
        }
    except Exception as e:
        logger.error(f"Error getting ML services metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving ML services metrics")