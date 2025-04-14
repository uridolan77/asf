"""
API Endpoints for ML model management and fine-tuning.

This module provides FastAPI endpoints for managing ML models,
monitoring model performance, and handling fine-tuning requests.
"""

import os
import json
import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Body, Query, Path, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from asf.medical.core.logging_config import get_logger
from asf.medical.ml.models.model_registry import (
    get_model_registry, ModelMetrics, ModelStatus, ModelMetadata
)
from asf.medical.ml.preprocessing.data_pipeline import (
    DataSchema, DataPipelineFactory, create_medical_claim_schema
)
from asf.medical.ml.services.enhanced_contradiction_classifier import EnhancedContradictionClassifier
from asf.medical.api.dependencies import get_current_user
from asf.medical.storage.models import User
from asf.medical.ml.services.gpt4_contradiction_classifier import (
    GPT4ContradictionClassifier,
    QuantizationMode
)
from asf.medical.ml.models.lora_adapter import (
    LoraAdapter, LoraAdapterConfig, get_adapter_registry
)

logger = get_logger(__name__)

router = APIRouter(
    prefix="/ml",
    tags=["ml"],
    responses={404: {"description": "Not found"}},
)

# Initialize services
model_registry = get_model_registry()
contradiction_classifier = EnhancedContradictionClassifier()

@router.get("/models", response_model=Dict[str, List[Dict[str, Any]]])
async def list_models():
    """
    List all available ML models in the registry.
    
    Returns:
        Dict mapping model names to lists of versions with metadata.
    """
    try:
        models = model_registry.list_models()
        # Convert model metadata to dictionaries
        result = {}
        for name, versions in models.items():
            result[name] = [version.dict() for version in versions]
        return result
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")

@router.get("/models/{name}", response_model=List[Dict[str, Any]])
async def get_model_versions(name: str = Path(..., description="Name of the model")):
    """
    Get all versions of a specific model.
    
    Args:
        name: Name of the model.
        
    Returns:
        List of model versions with metadata.
    """
    try:
        versions = model_registry.list_versions(name)
        if not versions:
            raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
        return [version.dict() for version in versions]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model versions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting model versions: {str(e)}")

@router.get("/models/{name}/production", response_model=Dict[str, Any])
async def get_production_model(name: str = Path(..., description="Name of the model")):
    """
    Get the current production version of a model.
    
    Args:
        name: Name of the model.
        
    Returns:
        Model metadata for the production version.
    """
    try:
        model = model_registry.get_production_model(name)
        if not model:
            raise HTTPException(status_code=404, detail=f"No production model found for '{name}'")
        return model.dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting production model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting production model: {str(e)}")

@router.post("/models/{name}/{version}/status", response_model=Dict[str, Any])
async def update_model_status(
    name: str = Path(..., description="Name of the model"),
    version: str = Path(..., description="Version of the model"),
    status: ModelStatus = Body(..., description="New model status")
):
    """
    Update the status of a model version.
    
    Args:
        name: Name of the model.
        version: Version of the model.
        status: New status for the model.
        
    Returns:
        Updated model metadata.
    """
    try:
        updated = model_registry.update_model_status(name, version, status)
        if not updated:
            raise HTTPException(
                status_code=404, 
                detail=f"Model '{name}' version '{version}' not found"
            )
        return updated.dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating model status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating model status: {str(e)}")

@router.post("/models/{name}/{version}/metrics", response_model=Dict[str, Any])
async def update_model_metrics(
    name: str = Path(..., description="Name of the model"),
    version: str = Path(..., description="Version of the model"),
    metrics: ModelMetrics = Body(..., description="New model metrics")
):
    """
    Update the metrics of a model version.
    
    Args:
        name: Name of the model.
        version: Version of the model.
        metrics: New metrics for the model.
        
    Returns:
        Updated model metadata.
    """
    try:
        updated = model_registry.update_model_metrics(name, version, metrics)
        if not updated:
            raise HTTPException(
                status_code=404, 
                detail=f"Model '{name}' version '{version}' not found"
            )
        return updated.dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating model metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating model metrics: {str(e)}")

@router.post("/models/{name}/{version}/drift-check", response_model=Dict[str, Any])
async def check_model_drift(
    name: str = Path(..., description="Name of the model"),
    version: str = Path(..., description="Version of the model")
):
    """
    Check for model drift in a specific model version.
    
    Args:
        name: Name of the model.
        version: Version of the model.
        
    Returns:
        Dictionary with drift detection results.
    """
    try:
        drift_results = model_registry.check_model_drift(name, version)
        return drift_results
    except Exception as e:
        logger.error(f"Error checking model drift: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error checking model drift: {str(e)}")

@router.post("/models/{name}/retrain", response_model=Dict[str, Any])
async def schedule_model_retraining(
    name: str = Path(..., description="Name of the model"),
    background_tasks: BackgroundTasks,
    training_config: Dict[str, Any] = Body(..., description="Training configuration"),
    version: Optional[str] = Query(None, description="Base version to retrain (uses latest if not specified)")
):
    """
    Schedule retraining for a model.
    
    Args:
        name: Name of the model to retrain.
        training_config: Configuration for model retraining.
        version: Optional base version to retrain. Uses latest version if not specified.
        background_tasks: FastAPI background tasks.
        
    Returns:
        Dictionary with job ID and status.
    """
    try:
        # Get base version if not specified
        if not version:
            model = model_registry.get_model(name)
            if not model:
                raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
            version = model.version
        
        # Schedule retraining
        job_id = model_registry.schedule_retraining(name, version, training_config)
        
        # In a real implementation, this would trigger an actual training job
        # For demo purposes, just log the request
        logger.info(f"Scheduled retraining for model {name} version {version}")
        
        return {
            "status": "scheduled",
            "job_id": job_id,
            "model_name": name,
            "base_version": version,
            "scheduled_at": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error scheduling model retraining: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error scheduling model retraining: {str(e)}"
        )

@router.post("/contradiction/classify", response_model=Dict[str, Any])
async def classify_contradiction(
    contradiction: Dict[str, Any] = Body(..., description="Contradiction data to classify")
):
    """
    Classify a contradiction using the enhanced ML model.
    
    Args:
        contradiction: Dictionary containing contradiction information.
            Must include 'claim1', 'claim2', and optionally 'metadata1' and 'metadata2'.
            
    Returns:
        Classification result with detailed dimensions.
    """
    try:
        # Ensure required fields are present
        if 'claim1' not in contradiction or 'claim2' not in contradiction:
            raise HTTPException(
                status_code=400,
                detail="Missing required fields: both 'claim1' and 'claim2' are required"
            )
            
        # Process the contradiction
        result = await contradiction_classifier.classify_contradiction(contradiction)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error classifying contradiction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error classifying contradiction: {str(e)}"
        )

@router.post("/contradiction/model/{model_name}/retrain", response_model=Dict[str, Any])
async def retrain_contradiction_model(
    model_name: str = Path(..., description="Name of the model to retrain"),
    training_data: Dict[str, Any] = Body(..., description="Training data for retraining"),
    hyperparameters: Optional[Dict[str, Any]] = Body(None, description="Optional hyperparameters for training")
):
    """
    Retrain a specific contradiction classification model.
    
    Args:
        model_name: Name of the model to retrain.
        training_data: Training data for the model.
        hyperparameters: Optional hyperparameters for training.
        
    Returns:
        Training result including job ID and status.
    """
    try:
        # Validate model name
        valid_model_names = [
            "contradiction_type_classifier",
            "clinical_significance_classifier", 
            "evidence_quality_classifier",
            "temporal_classifier",
            "population_classifier",
            "methodological_classifier"
        ]
        
        if model_name not in valid_model_names:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model name. Must be one of: {', '.join(valid_model_names)}"
            )
            
        # Forward retraining request to the classifier
        result = await contradiction_classifier.retrain_model(
            model_name=model_name,
            training_data=training_data,
            hyperparameters=hyperparameters
        )
        
        if result.get("status") == "error":
            raise HTTPException(
                status_code=500,
                detail=f"Error retraining model: {result.get('error', 'Unknown error')}"
            )
            
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retraining model {model_name}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retraining model {model_name}: {str(e)}"
        )

@router.post("/data/validate", response_model=Dict[str, Any])
async def validate_data(
    data: Dict[str, Any] = Body(..., description="Data to validate"),
    schema_type: str = Query("medical_claim", description="Type of schema to use for validation")
):
    """
    Validate data against a schema.
    
    Args:
        data: Data to validate.
        schema_type: Type of schema to use.
        
    Returns:
        Validation results.
    """
    try:
        # Get appropriate schema
        if schema_type == "medical_claim":
            schema = create_medical_claim_schema()
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported schema type: {schema_type}"
            )
            
        # Create appropriate pipeline
        pipeline = DataPipelineFactory.create_pipeline(
            schema=schema,
            pipeline_type="text" if schema_type == "medical_claim" else "default"
        )
        
        # Process the data
        result = pipeline.process(data)
        
        # Return validation results and processed data
        return {
            "valid": result.validation_results["valid"],
            "errors": result.validation_results["errors"],
            "warnings": result.validation_results["warnings"],
            "processed_data": result.data,
            "quality_metrics": {
                "completeness": result.quality_metrics.completeness,
                "validation_errors": len(result.quality_metrics.validation_errors)
            }
        }
    except Exception as e:
        logger.error(f"Error validating data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error validating data: {str(e)}"
        )

@router.post("/data/preprocess", response_model=Dict[str, Any])
async def preprocess_data(
    data: Dict[str, Any] = Body(..., description="Data to preprocess"),
    pipeline_type: str = Query("text", description="Type of preprocessing pipeline to use"),
    schema_type: str = Query("medical_claim", description="Type of schema to use")
):
    """
    Preprocess data for ML model ingestion.
    
    Args:
        data: Data to preprocess.
        pipeline_type: Type of preprocessing pipeline.
        schema_type: Type of schema to use.
        
    Returns:
        Processed data with features.
    """
    try:
        # Get appropriate schema
        if schema_type == "medical_claim":
            schema = create_medical_claim_schema()
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported schema type: {schema_type}"
            )
            
        # Create appropriate pipeline
        pipeline = DataPipelineFactory.create_pipeline(
            schema=schema,
            pipeline_type=pipeline_type,
            use_spacy=True if pipeline_type == "text" else False
        )
        
        # Process the data
        result = pipeline.process(data)
        
        # Return processed data with features
        return {
            "valid": result.validation_results["valid"],
            "processed_data": result.data,
            "quality_metrics": {
                "completeness": result.quality_metrics.completeness
            },
            "data_hash": result.data_hash,
            "processing_timestamp": result.processing_timestamp.isoformat()
        }
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error preprocessing data: {str(e)}"
        )

# Advanced ML Endpoints for Medical Research Synthesizer API

advanced_router = APIRouter(
    prefix="/v1/ml",
    tags=["machine-learning"]
)

# Initialize service singletons
advanced_contradiction_service = None
legacy_contradiction_service = None

# Model selection environment variable
USE_ADVANCED_MODELS = os.environ.get("USE_ADVANCED_MODELS", "true").lower() == "true"
USE_MODEL_FALLBACK = os.environ.get("USE_MODEL_FALLBACK", "true").lower() == "true"
GPT4O_MINI_QUANTIZATION = os.environ.get("GPT4O_MINI_QUANTIZATION", "int4").lower()

class ContradictionRequest(BaseModel):
    """Request model for contradiction detection."""
    claim1: str = Field(..., description="First claim text")
    claim2: str = Field(..., description="Second claim text")
    metadata1: Optional[Dict[str, Any]] = Field(
        default={}, 
        description="Metadata for first claim (study type, sample size, etc.)"
    )
    metadata2: Optional[Dict[str, Any]] = Field(
        default={}, 
        description="Metadata for second claim (study type, sample size, etc.)"
    )
    use_advanced_model: Optional[bool] = Field(
        default=None,
        description="Whether to use GPT-4o-Mini with LoRA (overrides environment setting)"
    )

class AdapterInfo(BaseModel):
    """Information about a LoRA adapter."""
    name: str
    version: str
    component: str
    status: str
    description: Optional[str] = None

class AdapterUpdateRequest(BaseModel):
    """Request model for adapter updates."""
    component: str = Field(..., description="Component to update (e.g., contradiction_type)")
    adapter_name: str = Field(..., description="Name of the adapter")
    version: str = Field(..., description="Version of the adapter")

def get_contradiction_service(use_advanced: bool = None):
    """
    Get the contradiction service based on configuration.
    
    Args:
        use_advanced: Override for environment variable.
        
    Returns:
        Contradiction service instance.
    """
    global advanced_contradiction_service, legacy_contradiction_service
    
    # Use parameter if provided, otherwise use environment setting
    should_use_advanced = use_advanced if use_advanced is not None else USE_ADVANCED_MODELS
    
    try:
        if should_use_advanced:
            # Initialize advanced service if needed
            if advanced_contradiction_service is None:
                # Determine quantization mode
                quant_mode = QuantizationMode.INT4
                if GPT4O_MINI_QUANTIZATION == "int8":
                    quant_mode = QuantizationMode.INT8
                elif GPT4O_MINI_QUANTIZATION == "none":
                    quant_mode = QuantizationMode.NONE
                
                logger.info(f"Initializing GPT-4o-Mini contradiction service with {quant_mode} quantization")
                advanced_contradiction_service = GPT4ContradictionClassifier(
                    use_cache=True,
                    use_adapters=True,
                    quantization_mode=quant_mode,
                    fallback_to_base=USE_MODEL_FALLBACK
                )
            
            return advanced_contradiction_service
        else:
            # Initialize legacy service if needed
            if legacy_contradiction_service is None:
                logger.info("Initializing legacy contradiction service")
                legacy_contradiction_service = EnhancedContradictionClassifier(use_cache=True)
            
            return legacy_contradiction_service
    except Exception as e:
        logger.error(f"Error initializing contradiction service: {str(e)}")
        # Always fall back to legacy service if there's an initialization error
        if legacy_contradiction_service is None:
            legacy_contradiction_service = EnhancedContradictionClassifier(use_cache=True)
        
        return legacy_contradiction_service

@advanced_router.post("/contradiction/detect", response_model=Dict[str, Any])
async def detect_contradiction(
    request: ContradictionRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Detect and classify contradictions between two medical claims.
    
    This endpoint uses either the GPT-4o-Mini model with LoRA adapters (advanced)
    or the BioMedLM-based classifier (legacy) depending on configuration.
    
    Args:
        request: Contradiction request with claims and metadata.
        current_user: Current authenticated user.
        
    Returns:
        Contradiction classification results with all dimensions.
    """
    try:
        # Get appropriate service
        service = get_contradiction_service(request.use_advanced_model)
        
        # Process request
        data = {
            "claim1": request.claim1,
            "claim2": request.claim2,
            "metadata1": request.metadata1 or {},
            "metadata2": request.metadata2 or {}
        }
        
        # Classify contradiction
        result = await service.classify_contradiction(data)
        
        # Add model type to response
        if isinstance(service, GPT4ContradictionClassifier):
            result["model_type"] = "gpt4o_mini_lora"
        else:
            result["model_type"] = "biomedlm"
        
        return result
    except Exception as e:
        logger.error(f"Error in contradiction detection: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in contradiction detection: {str(e)}"
        )

@advanced_router.get("/contradiction/advanced/status")
async def get_advanced_model_status(
    current_user: User = Depends(get_current_user)
):
    """
    Get status and information about the advanced contradiction model.
    
    Args:
        current_user: Current authenticated user.
        
    Returns:
        Status information for the advanced model.
    """
    try:
        if advanced_contradiction_service is None:
            # Try to initialize the service
            service = get_contradiction_service(True)
        else:
            service = advanced_contradiction_service
        
        # Get information
        info = service.get_info()
        
        # Add environment settings
        info["environment"] = {
            "use_advanced_models": USE_ADVANCED_MODELS,
            "use_model_fallback": USE_MODEL_FALLBACK,
            "quantization": GPT4O_MINI_QUANTIZATION
        }
        
        return info
    except Exception as e:
        logger.error(f"Error getting advanced model status: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "initialized": advanced_contradiction_service is not None
        }

@advanced_router.post("/contradiction/advanced/adapters/update")
async def update_adapter(
    request: AdapterUpdateRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Update a specific adapter for the advanced contradiction model.
    
    Args:
        request: Adapter update request.
        current_user: Current authenticated user.
        
    Returns:
        Update results.
    """
    try:
        # Ensure service is initialized
        service = get_contradiction_service(True)
        
        if not isinstance(service, GPT4ContradictionClassifier):
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "status": "error",
                    "error": "Advanced model not enabled"
                }
            )
        
        # Update adapter
        result = await service.update_adapter(
            component=request.component,
            adapter_name=request.adapter_name,
            version=request.version
        )
        
        return result
    except Exception as e:
        logger.error(f"Error updating adapter: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating adapter: {str(e)}"
        )

@advanced_router.post("/contradiction/advanced/adapters/reload")
async def reload_all_adapters(
    current_user: User = Depends(get_current_user)
):
    """
    Reload all adapters for the advanced contradiction model.
    
    Args:
        current_user: Current authenticated user.
        
    Returns:
        Reload results.
    """
    try:
        # Ensure service is initialized
        service = get_contradiction_service(True)
        
        if not isinstance(service, GPT4ContradictionClassifier):
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "status": "error",
                    "error": "Advanced model not enabled"
                }
            )
        
        # Reload adapters
        result = await service.reload_all_adapters()
        
        return result
    except Exception as e:
        logger.error(f"Error reloading adapters: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reloading adapters: {str(e)}"
        )

@advanced_router.get("/contradiction/benchmark/compare")
async def benchmark_comparison(
    claim1: str = Query(..., description="First claim text"),
    claim2: str = Query(..., description="Second claim text"),
    current_user: User = Depends(get_current_user)
):
    """
    Compare results between legacy and advanced contradiction models.
    
    Args:
        claim1: First claim text.
        claim2: Second claim text.
        current_user: Current authenticated user.
        
    Returns:
        Comparison results.
    """
    try:
        # Ensure both services are initialized
        legacy_service = get_contradiction_service(False)
        advanced_service = get_contradiction_service(True)
        
        # Create request data
        data = {
            "claim1": claim1,
            "claim2": claim2,
            "metadata1": {},
            "metadata2": {}
        }
        
        # Process with both services
        start_legacy = datetime.now()
        legacy_result = await legacy_service.classify_contradiction(data)
        legacy_time = (datetime.now() - start_legacy).total_seconds() * 1000
        
        start_advanced = datetime.now()
        advanced_result = await advanced_service.classify_contradiction(data)
        advanced_time = (datetime.now() - start_advanced).total_seconds() * 1000
        
        # Create comparison
        comparison = {
            "input": {
                "claim1": claim1,
                "claim2": claim2
            },
            "legacy_model": {
                "contradiction_type": legacy_result["contradiction_type"],
                "confidence": legacy_result["contradiction_probability"],
                "processing_time_ms": legacy_time,
                "result": legacy_result
            },
            "advanced_model": {
                "contradiction_type": advanced_result["contradiction_type"],
                "confidence": advanced_result["contradiction_probability"],
                "processing_time_ms": advanced_time,
                "result": advanced_result
            },
            "agreement": legacy_result["contradiction_type"] == advanced_result["contradiction_type"],
            "performance_diff_percent": (
                (legacy_time - advanced_time) / legacy_time * 100 
                if legacy_time > 0 else 0
            )
        }
        
        return comparison
    except Exception as e:
        logger.error(f"Error in benchmark comparison: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in benchmark comparison: {str(e)}"
        )

# Include the advanced router
router.include_router(advanced_router)