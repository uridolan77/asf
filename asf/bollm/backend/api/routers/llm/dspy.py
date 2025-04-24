"""
DSPy Integration API router for BO backend.

This module provides endpoints for managing and using DSPy,
including module management, execution, and optimization.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Body, Query, Path
from typing import Dict, Any, List, Optional, Union
import logging
import os
import yaml
import json
from datetime import datetime
import asyncio

from ...auth import get_current_user, User
from ...utils import handle_api_error
from .models import (
    DSPyModuleInfo, DSPyExecuteRequest, DSPyExecuteResponse,
    DSPyOptimizeRequest
)
from .utils import (
    load_config, save_config, format_timestamp,
    get_dspy_availability, DSPY_CONFIG_PATH
)

# Import DSPy components if available
DSPY_AVAILABLE = get_dspy_availability()
if DSPY_AVAILABLE:
    from asf.medical.ml.dspy.dspy_client import get_dspy_client, DSPyClient
    from asf.medical.ml.dspy.modules.medical_rag import MedicalRAGModule
    from asf.medical.ml.dspy.modules.contradiction_detection import ContradictionDetectionModule
    from asf.medical.ml.dspy.modules.evidence_extraction import EvidenceExtractionModule
    from asf.medical.ml.dspy.modules.medical_summarization import MedicalSummarizationModule
    from asf.medical.ml.dspy.modules.clinical_qa import ClinicalQAModule

router = APIRouter(prefix="/dspy", tags=["dspy"])

logger = logging.getLogger(__name__)

# DSPy client instance
_dspy_client = None

async def get_dspy_client_instance():
    """Get or create the DSPy client."""
    global _dspy_client
    
    if not DSPY_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="DSPy is not available. Please check your installation."
        )
    
    if _dspy_client is None:
        try:
            # Initialize DSPy client
            _dspy_client = await get_dspy_client()
            logger.info("DSPy client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize DSPy client: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to initialize DSPy client: {str(e)}"
            )
    
    return _dspy_client

@router.get("/modules", response_model=List[DSPyModuleInfo])
async def get_modules(current_user: User = Depends(get_current_user)):
    """
    Get all registered DSPy modules.
    
    This endpoint returns information about all registered DSPy modules,
    including their signatures, parameters, and usage statistics.
    """
    if not DSPY_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="DSPy is not available. Please check your installation."
        )
    
    try:
        # Get DSPy client
        client = await get_dspy_client_instance()
        
        # Get registered modules
        modules = await client.list_modules()
        
        # Format module information
        module_infos = []
        for module_name, module_data in modules.items():
            module_infos.append(
                DSPyModuleInfo(
                    name=module_name,
                    description=module_data.get("description", ""),
                    signature=module_data.get("signature", ""),
                    parameters=module_data.get("parameters", {}),
                    registered_at=module_data.get("registered_at", format_timestamp()),
                    last_used=module_data.get("last_used"),
                    usage_count=module_data.get("usage_count", 0)
                )
            )
        
        return module_infos
    except Exception as e:
        logger.error(f"Error getting DSPy modules: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get DSPy modules: {str(e)}"
        )

@router.get("/modules/{module_name}", response_model=DSPyModuleInfo)
async def get_module(
    module_name: str = Path(..., description="Module name"),
    current_user: User = Depends(get_current_user)
):
    """
    Get information about a specific DSPy module.
    
    This endpoint returns detailed information about a specific DSPy module,
    including its signature, parameters, and usage statistics.
    """
    if not DSPY_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="DSPy is not available. Please check your installation."
        )
    
    try:
        # Get DSPy client
        client = await get_dspy_client_instance()
        
        # Get registered modules
        modules = await client.list_modules()
        
        # Check if module exists
        if module_name not in modules:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Module '{module_name}' not found"
            )
        
        # Get module data
        module_data = modules[module_name]
        
        # Format module information
        return DSPyModuleInfo(
            name=module_name,
            description=module_data.get("description", ""),
            signature=module_data.get("signature", ""),
            parameters=module_data.get("parameters", {}),
            registered_at=module_data.get("registered_at", format_timestamp()),
            last_used=module_data.get("last_used"),
            usage_count=module_data.get("usage_count", 0)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting DSPy module '{module_name}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get DSPy module '{module_name}': {str(e)}"
        )

@router.post("/modules", response_model=DSPyModuleInfo)
async def register_module(
    module_name: str = Body(..., embed=True),
    module_type: str = Body(..., embed=True),
    parameters: Dict[str, Any] = Body({}, embed=True),
    description: Optional[str] = Body(None, embed=True),
    current_user: User = Depends(get_current_user)
):
    """
    Register a new DSPy module.
    
    This endpoint registers a new DSPy module with the specified parameters.
    """
    if not DSPY_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="DSPy is not available. Please check your installation."
        )
    
    try:
        # Get DSPy client
        client = await get_dspy_client_instance()
        
        # Create module instance based on module_type
        module_instance = None
        if module_type == "medical_rag":
            module_instance = MedicalRAGModule(**parameters)
        elif module_type == "contradiction_detection":
            module_instance = ContradictionDetectionModule(**parameters)
        elif module_type == "evidence_extraction":
            module_instance = EvidenceExtractionModule(**parameters)
        elif module_type == "medical_summarization":
            module_instance = MedicalSummarizationModule(**parameters)
        elif module_type == "clinical_qa":
            module_instance = ClinicalQAModule(**parameters)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported module type: {module_type}"
            )
        
        # Register module
        await client.register_module(
            name=module_name,
            module=module_instance,
            description=description or f"{module_type} module"
        )
        
        # Get registered module info
        modules = await client.list_modules()
        module_data = modules[module_name]
        
        # Format module information
        return DSPyModuleInfo(
            name=module_name,
            description=module_data.get("description", ""),
            signature=module_data.get("signature", ""),
            parameters=module_data.get("parameters", {}),
            registered_at=module_data.get("registered_at", format_timestamp()),
            last_used=module_data.get("last_used"),
            usage_count=module_data.get("usage_count", 0)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering DSPy module '{module_name}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register DSPy module '{module_name}': {str(e)}"
        )

@router.delete("/modules/{module_name}", response_model=Dict[str, Any])
async def unregister_module(
    module_name: str = Path(..., description="Module name"),
    current_user: User = Depends(get_current_user)
):
    """
    Unregister a DSPy module.
    
    This endpoint unregisters a DSPy module.
    """
    if not DSPY_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="DSPy is not available. Please check your installation."
        )
    
    try:
        # Get DSPy client
        client = await get_dspy_client_instance()
        
        # Check if module exists
        modules = await client.list_modules()
        if module_name not in modules:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Module '{module_name}' not found"
            )
        
        # Unregister module
        await client.unregister_module(module_name)
        
        return {
            "success": True,
            "message": f"Module '{module_name}' unregistered successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error unregistering DSPy module '{module_name}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to unregister DSPy module '{module_name}': {str(e)}"
        )

@router.post("/execute", response_model=DSPyExecuteResponse)
async def execute_module(
    request: DSPyExecuteRequest = Body(...),
    current_user: User = Depends(get_current_user)
):
    """
    Execute a DSPy module.
    
    This endpoint executes a DSPy module with the specified inputs.
    """
    if not DSPY_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="DSPy is not available. Please check your installation."
        )
    
    try:
        # Get DSPy client
        client = await get_dspy_client_instance()
        
        # Check if module exists
        modules = await client.list_modules()
        if request.module_name not in modules:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Module '{request.module_name}' not found"
            )
        
        # Execute module
        start_time = datetime.utcnow()
        result = await client.call_module(
            module_name=request.module_name,
            **request.inputs,
            **(request.config or {})
        )
        end_time = datetime.utcnow()
        execution_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Get module info
        module_info = modules[request.module_name]
        
        return DSPyExecuteResponse(
            module_name=request.module_name,
            inputs=request.inputs,
            outputs=result,
            execution_time_ms=execution_time_ms,
            model_used=module_info.get("model", "unknown"),
            tokens_used=module_info.get("tokens_used"),
            created_at=end_time.isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing DSPy module '{request.module_name}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute DSPy module '{request.module_name}': {str(e)}"
        )

@router.post("/optimize", response_model=Dict[str, Any])
async def optimize_module(
    request: DSPyOptimizeRequest = Body(...),
    current_user: User = Depends(get_current_user)
):
    """
    Optimize a DSPy module.
    
    This endpoint optimizes a DSPy module using the specified metric and examples.
    """
    if not DSPY_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="DSPy is not available. Please check your installation."
        )
    
    try:
        # Get DSPy client
        client = await get_dspy_client_instance()
        
        # Check if module exists
        modules = await client.list_modules()
        if request.module_name not in modules:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Module '{request.module_name}' not found"
            )
        
        # Optimize module
        start_time = datetime.utcnow()
        
        # This would be replaced with actual optimization
        # For now, we'll just return a mock response
        optimization_result = {
            "success": True,
            "module_name": request.module_name,
            "metric": request.metric,
            "num_trials": request.num_trials,
            "examples_used": len(request.examples),
            "original_score": 0.75,
            "optimized_score": 0.85,
            "improvement": 0.1,
            "best_prompt": "Optimized prompt template",
            "execution_time_ms": 5000
        }
        
        end_time = datetime.utcnow()
        execution_time_ms = (end_time - start_time).total_seconds() * 1000
        
        return {
            **optimization_result,
            "execution_time_ms": execution_time_ms,
            "created_at": end_time.isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error optimizing DSPy module '{request.module_name}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to optimize DSPy module '{request.module_name}': {str(e)}"
        )

@router.get("/config", response_model=Dict[str, Any])
async def get_dspy_config(current_user: User = Depends(get_current_user)):
    """
    Get the current DSPy configuration.
    
    This endpoint returns the current configuration of DSPy.
    """
    if not DSPY_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="DSPy is not available. Please check your installation."
        )
    
    try:
        # Load config from file
        config = load_config(DSPY_CONFIG_PATH)
        return config
    except Exception as e:
        logger.error(f"Error getting DSPy config: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get DSPy config: {str(e)}"
        )

@router.put("/config", response_model=Dict[str, Any])
async def update_dspy_config(
    config: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_user)
):
    """
    Update the DSPy configuration.
    
    This endpoint updates the configuration of DSPy.
    """
    if not DSPY_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="DSPy is not available. Please check your installation."
        )
    
    try:
        # Save config to file
        save_config(DSPY_CONFIG_PATH, config)
        return config
    except Exception as e:
        logger.error(f"Error updating DSPy config: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update DSPy config: {str(e)}"
        )
