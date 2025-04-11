"""
DSPy API Integration

This module provides FastAPI endpoints for DSPy functionality.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Path
from pydantic import BaseModel, Field

from .dspy_client import get_dspy_client, DSPyClient
from .dspy_settings import get_dspy_settings
from .modules import (
    MedicalRAGModule,
    EnhancedMedicalRAGModule,
    ContradictionDetectionModule,
    TemporalContradictionModule,
    EvidenceExtractionModule,
    MedicalSummarizationModule,
    ClinicalQAModule
)

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/dspy", tags=["dspy"])


# Pydantic models for requests and responses
class DSPyModuleListResponse(BaseModel):
    """Response model for listing DSPy modules."""
    modules: List[Dict[str, Any]] = Field(..., description="List of registered modules")


class DSPyModuleRegisterRequest(BaseModel):
    """Request model for registering a DSPy module."""
    name: str = Field(..., description="Name for the module")
    module_type: str = Field(..., description="Type of module to register")
    description: Optional[str] = Field(None, description="Description of the module")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    config: Optional[Dict[str, Any]] = Field(None, description="Module configuration")


class DSPyModuleRegisterResponse(BaseModel):
    """Response model for registering a DSPy module."""
    name: str = Field(..., description="Name of the registered module")
    module_type: str = Field(..., description="Type of the registered module")
    description: str = Field(..., description="Description of the module")
    registered_at: str = Field(..., description="Timestamp of registration")


class DSPyModuleCallRequest(BaseModel):
    """Request model for calling a DSPy module."""
    module_name: str = Field(..., description="Name of the module to call")
    parameters: Dict[str, Any] = Field(..., description="Parameters for the module call")


class DSPyModuleCallResponse(BaseModel):
    """Response model for calling a DSPy module."""
    module_name: str = Field(..., description="Name of the called module")
    result: Dict[str, Any] = Field(..., description="Result of the module call")
    execution_time: float = Field(..., description="Execution time in seconds")


class DSPyMedicalRAGRequest(BaseModel):
    """Request model for medical RAG."""
    question: str = Field(..., description="Medical question to answer")
    k: Optional[int] = Field(5, description="Number of passages to retrieve")


class DSPyContradictionRequest(BaseModel):
    """Request model for contradiction detection."""
    statement1: str = Field(..., description="First medical statement")
    statement2: str = Field(..., description="Second medical statement")


class DSPyTemporalContradictionRequest(BaseModel):
    """Request model for temporal contradiction detection."""
    statement1: str = Field(..., description="First medical statement")
    timestamp1: str = Field(..., description="Timestamp of the first statement")
    statement2: str = Field(..., description="Second medical statement")
    timestamp2: str = Field(..., description="Timestamp of the second statement")


class DSPyEvidenceExtractionRequest(BaseModel):
    """Request model for evidence extraction."""
    text: str = Field(..., description="Medical text to analyze")
    claim: str = Field(..., description="The claim to find evidence for")


class DSPyMedicalSummarizationRequest(BaseModel):
    """Request model for medical summarization."""
    text: str = Field(..., description="Medical text to summarize")
    audience: Optional[str] = Field("clinician", description="Target audience")


class DSPyClinicalQARequest(BaseModel):
    """Request model for clinical QA."""
    question: str = Field(..., description="Clinical question to answer")


# Dependency for getting DSPy client
async def get_dspy_client_dep() -> DSPyClient:
    """Dependency for getting DSPy client."""
    return await get_dspy_client()


@router.get("/modules", response_model=DSPyModuleListResponse)
async def list_modules(
    dspy_client: DSPyClient = Depends(get_dspy_client_dep)
) -> DSPyModuleListResponse:
    """
    List all registered DSPy modules.
    
    Returns:
        DSPyModuleListResponse: List of registered modules
    """
    modules = dspy_client.list_modules()
    return DSPyModuleListResponse(modules=modules)


@router.post("/modules", response_model=DSPyModuleRegisterResponse)
async def register_module(
    request: DSPyModuleRegisterRequest,
    dspy_client: DSPyClient = Depends(get_dspy_client_dep)
) -> DSPyModuleRegisterResponse:
    """
    Register a new DSPy module.
    
    Args:
        request: Module registration request
        
    Returns:
        DSPyModuleRegisterResponse: Registration result
    """
    # Create module based on type
    config = request.config or {}
    
    try:
        if request.module_type == "MedicalRAGModule":
            module = MedicalRAGModule(**config)
        elif request.module_type == "EnhancedMedicalRAGModule":
            module = EnhancedMedicalRAGModule(**config)
        elif request.module_type == "ContradictionDetectionModule":
            module = ContradictionDetectionModule(**config)
        elif request.module_type == "TemporalContradictionModule":
            module = TemporalContradictionModule(**config)
        elif request.module_type == "EvidenceExtractionModule":
            module = EvidenceExtractionModule(**config)
        elif request.module_type == "MedicalSummarizationModule":
            module = MedicalSummarizationModule(**config)
        elif request.module_type == "ClinicalQAModule":
            module = ClinicalQAModule(**config)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported module type: {request.module_type}"
            )
    except Exception as e:
        logger.error(f"Error creating module: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Error creating module: {str(e)}"
        )
    
    # Register module
    await dspy_client.register_module(
        name=request.name,
        module=module,
        description=request.description or f"DSPy module: {request.module_type}",
        metadata=request.metadata
    )
    
    # Get module info
    modules = dspy_client.list_modules()
    module_info = next((m for m in modules if m["name"] == request.name), None)
    
    if not module_info:
        raise HTTPException(
            status_code=500,
            detail="Module registration failed"
        )
    
    return DSPyModuleRegisterResponse(
        name=module_info["name"],
        module_type=module_info["type"],
        description=module_info["description"],
        registered_at=module_info["registered_at"]
    )


@router.post("/modules/call", response_model=DSPyModuleCallResponse)
async def call_module(
    request: DSPyModuleCallRequest,
    dspy_client: DSPyClient = Depends(get_dspy_client_dep)
) -> DSPyModuleCallResponse:
    """
    Call a registered DSPy module.
    
    Args:
        request: Module call request
        
    Returns:
        DSPyModuleCallResponse: Call result
    """
    import time
    
    # Check if module exists
    module = dspy_client.get_module(request.module_name)
    if not module:
        raise HTTPException(
            status_code=404,
            detail=f"Module not found: {request.module_name}"
        )
    
    # Call module
    start_time = time.time()
    try:
        result = await dspy_client.call_module(
            module_name=request.module_name,
            **request.parameters
        )
    except Exception as e:
        logger.error(f"Error calling module: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error calling module: {str(e)}"
        )
    execution_time = time.time() - start_time
    
    return DSPyModuleCallResponse(
        module_name=request.module_name,
        result=result,
        execution_time=execution_time
    )


@router.post("/medical-rag", response_model=Dict[str, Any])
async def medical_rag(
    request: DSPyMedicalRAGRequest,
    dspy_client: DSPyClient = Depends(get_dspy_client_dep)
) -> Dict[str, Any]:
    """
    Perform medical RAG to answer a question.
    
    Args:
        request: Medical RAG request
        
    Returns:
        Dict[str, Any]: RAG result
    """
    # Create module
    module = MedicalRAGModule(k=request.k)
    
    # Call module
    try:
        result = module(question=request.question)
        return result
    except Exception as e:
        logger.error(f"Error in medical RAG: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error in medical RAG: {str(e)}"
        )


@router.post("/contradiction-detection", response_model=Dict[str, Any])
async def contradiction_detection(
    request: DSPyContradictionRequest,
    dspy_client: DSPyClient = Depends(get_dspy_client_dep)
) -> Dict[str, Any]:
    """
    Detect contradictions between medical statements.
    
    Args:
        request: Contradiction detection request
        
    Returns:
        Dict[str, Any]: Contradiction detection result
    """
    # Create module
    module = ContradictionDetectionModule()
    
    # Call module
    try:
        result = module(
            statement1=request.statement1,
            statement2=request.statement2
        )
        return result
    except Exception as e:
        logger.error(f"Error in contradiction detection: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error in contradiction detection: {str(e)}"
        )


@router.post("/temporal-contradiction", response_model=Dict[str, Any])
async def temporal_contradiction(
    request: DSPyTemporalContradictionRequest,
    dspy_client: DSPyClient = Depends(get_dspy_client_dep)
) -> Dict[str, Any]:
    """
    Detect temporal contradictions between medical statements.
    
    Args:
        request: Temporal contradiction detection request
        
    Returns:
        Dict[str, Any]: Temporal contradiction detection result
    """
    # Create module
    module = TemporalContradictionModule()
    
    # Call module
    try:
        result = module(
            statement1=request.statement1,
            timestamp1=request.timestamp1,
            statement2=request.statement2,
            timestamp2=request.timestamp2
        )
        return result
    except Exception as e:
        logger.error(f"Error in temporal contradiction detection: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error in temporal contradiction detection: {str(e)}"
        )


@router.post("/evidence-extraction", response_model=Dict[str, Any])
async def evidence_extraction(
    request: DSPyEvidenceExtractionRequest,
    dspy_client: DSPyClient = Depends(get_dspy_client_dep)
) -> Dict[str, Any]:
    """
    Extract evidence from medical text.
    
    Args:
        request: Evidence extraction request
        
    Returns:
        Dict[str, Any]: Evidence extraction result
    """
    # Create module
    module = EvidenceExtractionModule()
    
    # Call module
    try:
        result = module(
            text=request.text,
            claim=request.claim
        )
        return result
    except Exception as e:
        logger.error(f"Error in evidence extraction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error in evidence extraction: {str(e)}"
        )


@router.post("/medical-summarization", response_model=Dict[str, Any])
async def medical_summarization(
    request: DSPyMedicalSummarizationRequest,
    dspy_client: DSPyClient = Depends(get_dspy_client_dep)
) -> Dict[str, Any]:
    """
    Summarize medical content.
    
    Args:
        request: Medical summarization request
        
    Returns:
        Dict[str, Any]: Summarization result
    """
    # Create module
    module = MedicalSummarizationModule()
    
    # Call module
    try:
        result = module(
            text=request.text,
            audience=request.audience
        )
        return result
    except Exception as e:
        logger.error(f"Error in medical summarization: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error in medical summarization: {str(e)}"
        )


@router.post("/clinical-qa", response_model=Dict[str, Any])
async def clinical_qa(
    request: DSPyClinicalQARequest,
    dspy_client: DSPyClient = Depends(get_dspy_client_dep)
) -> Dict[str, Any]:
    """
    Answer a clinical question.
    
    Args:
        request: Clinical QA request
        
    Returns:
        Dict[str, Any]: Clinical QA result
    """
    # Create module
    module = ClinicalQAModule()
    
    # Call module
    try:
        result = module(question=request.question)
        return result
    except Exception as e:
        logger.error(f"Error in clinical QA: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error in clinical QA: {str(e)}"
        )


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Check the health of the DSPy API.
    
    Returns:
        Dict[str, str]: Health status
    """
    return {"status": "healthy"}


@router.post("/cache/clear")
async def clear_cache(
    pattern: Optional[str] = None,
    dspy_client: DSPyClient = Depends(get_dspy_client_dep)
) -> Dict[str, str]:
    """
    Clear the DSPy cache.
    
    Args:
        pattern: Optional pattern to match cache keys
        
    Returns:
        Dict[str, str]: Clear result
    """
    await dspy_client.clear_cache(pattern=pattern)
    return {"status": "cache cleared"}


# Export router
__all__ = ['router']
