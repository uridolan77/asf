FastAPI Application with DSPy Integration

This example demonstrates how to create a FastAPI application that uses the DSPy integration.

import logging
from typing import Dict, Any

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from asf.medical.ml.dspy.dspy_api import router as dspy_router
from asf.medical.ml.dspy.dspy_client import get_dspy_client, DSPyClient
from asf.medical.ml.dspy.modules.medical_rag import MedicalRAGModule
from asf.medical.ml.dspy.modules.contradiction_detection import ContradictionDetectionModule
from asf.medical.ml.dspy.modules.evidence_extraction import EvidenceExtractionModule
from asf.medical.ml.dspy.modules.medical_summarization import MedicalSummarizationModule
from asf.medical.ml.dspy.modules.clinical_qa import ClinicalQAModule

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Medical DSPy API",
    description="API for medical research using DSPy",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include DSPy router
app.include_router(dspy_router)


# Dependency for getting DSPy client
async def get_dspy_client_dep() -> DSPyClient:
    """Dependency for getting DSPy client."""
    return await get_dspy_client()


# Register default modules on startup
@app.on_event("startup")
async def startup_event():
    """Register default modules on startup."""
    logger.info("Registering default modules...")
    
    # Get DSPy client
    dspy_client = await get_dspy_client()
    
    # Register Medical RAG module
    medical_rag = MedicalRAGModule(k=5)
    await dspy_client.register_module(
        name="medical_rag",
        module=medical_rag,
        description="Medical RAG module for answering medical questions"
    )
    
    # Register Contradiction Detection module
    contradiction_module = ContradictionDetectionModule()
    await dspy_client.register_module(
        name="contradiction_detection",
        module=contradiction_module,
        description="Module for detecting contradictions between medical statements"
    )
    
    # Register Evidence Extraction module
    evidence_module = EvidenceExtractionModule()
    await dspy_client.register_module(
        name="evidence_extraction",
        module=evidence_module,
        description="Module for extracting evidence from medical text"
    )
    
    # Register Medical Summarization module
    summarization_module = MedicalSummarizationModule()
    await dspy_client.register_module(
        name="medical_summarization",
        module=summarization_module,
        description="Module for summarizing medical content"
    )
    
    # Register Clinical QA module
    clinical_qa_module = ClinicalQAModule()
    await dspy_client.register_module(
        name="clinical_qa",
        module=clinical_qa_module,
        description="Module for answering clinical questions"
    )
    
    logger.info("Default modules registered successfully")


# Shutdown DSPy client on app shutdown
@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown DSPy client on app shutdown."""
    logger.info("Shutting down DSPy client...")
    
    # Get DSPy client
    dspy_client = await get_dspy_client()
    
    # Shutdown client
    await dspy_client.shutdown()
    
    logger.info("DSPy client shut down successfully")


# Additional routes
class HealthResponse(BaseModel):
    Response model for health check.
    Check the health of the API.
    
    Returns:
        HealthResponse: Health status
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0"
    )


# Run the app
if __name__ == "__main__":
    import uvicorn
    
    # Run with uvicorn
    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
