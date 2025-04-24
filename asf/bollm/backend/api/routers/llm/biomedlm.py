"""
BiomedLM API router for BO backend.

This module provides endpoints for managing and using BiomedLM,
including model management, generation, and fine-tuning.
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
    BiomedLMModelInfo, BiomedLMGenerateRequest, BiomedLMGenerateResponse,
    BiomedLMFinetuneRequest
)
from .utils import (
    load_config, save_config, format_timestamp,
    get_biomedlm_availability, BIOMEDLM_CONFIG_PATH
)

# Import BiomedLM components if available
BIOMEDLM_AVAILABLE = get_biomedlm_availability()
if BIOMEDLM_AVAILABLE:
    from asf.medical.ml.models.biomedlm_adapter import BiomedLMAdapter

router = APIRouter(prefix="/biomedlm", tags=["biomedlm"])

logger = logging.getLogger(__name__)

# BiomedLM adapter instance
_biomedlm_adapter = None

def get_biomedlm_adapter():
    """Get or create the BiomedLM adapter."""
    global _biomedlm_adapter
    
    if not BIOMEDLM_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="BiomedLM is not available. Please check your installation."
        )
    
    if _biomedlm_adapter is None:
        try:
            # Load config from file
            config = load_config(BIOMEDLM_CONFIG_PATH)
            
            # Create BiomedLM adapter
            _biomedlm_adapter = BiomedLMAdapter(
                model_name=config.get("model_name", "biomedlm-2-7b"),
                model_path=os.path.expandvars(config.get("model_path", "")),
                use_gpu=config.get("use_gpu", True),
                precision=config.get("precision", "fp16")
            )
            logger.info(f"BiomedLM adapter initialized with model: {config.get('model_name')}")
        except Exception as e:
            logger.error(f"Failed to initialize BiomedLM adapter: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to initialize BiomedLM adapter: {str(e)}"
            )
    
    return _biomedlm_adapter

@router.get("/models", response_model=List[BiomedLMModelInfo])
async def get_models(current_user: User = Depends(get_current_user)):
    """
    Get all available BiomedLM models.
    
    This endpoint returns information about all available BiomedLM models,
    including base models and fine-tuned adapters.
    """
    if not BIOMEDLM_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="BiomedLM is not available. Please check your installation."
        )
    
    try:
        # Load config from file
        config = load_config(BIOMEDLM_CONFIG_PATH)
        
        # Get base model info
        base_model = BiomedLMModelInfo(
            model_id=config.get("model_name", "biomedlm-2-7b"),
            display_name="BiomedLM Base Model",
            description="Base BiomedLM model for medical text generation",
            base_model=config.get("model_name", "biomedlm-2-7b"),
            parameters=2700000000,  # 2.7B parameters
            adapter_type=None,
            fine_tuned_for=None,
            created_at="2023-01-01T00:00:00Z",  # Placeholder
            last_used=format_timestamp(),
            usage_count=100  # Placeholder
        )
        
        # Get adapter models
        adapter_models = []
        for adapter_name, adapter_config in config.get("lora_adapters", {}).items():
            adapter_models.append(
                BiomedLMModelInfo(
                    model_id=f"{config.get('model_name', 'biomedlm-2-7b')}-{adapter_name}",
                    display_name=f"BiomedLM {adapter_name.replace('_', ' ').title()}",
                    description=adapter_config.get("description", ""),
                    base_model=config.get("model_name", "biomedlm-2-7b"),
                    parameters=2700000000,  # 2.7B parameters
                    adapter_type="lora",
                    fine_tuned_for=[adapter_name.replace("_", " ")],
                    created_at="2023-01-01T00:00:00Z",  # Placeholder
                    last_used=format_timestamp(),
                    usage_count=50  # Placeholder
                )
            )
        
        return [base_model] + adapter_models
    except Exception as e:
        logger.error(f"Error getting BiomedLM models: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get BiomedLM models: {str(e)}"
        )

@router.get("/models/{model_id}", response_model=BiomedLMModelInfo)
async def get_model(
    model_id: str = Path(..., description="Model ID"),
    current_user: User = Depends(get_current_user)
):
    """
    Get information about a specific BiomedLM model.
    
    This endpoint returns detailed information about a specific BiomedLM model,
    including its parameters, adapters, and usage statistics.
    """
    if not BIOMEDLM_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="BiomedLM is not available. Please check your installation."
        )
    
    try:
        # Load config from file
        config = load_config(BIOMEDLM_CONFIG_PATH)
        
        # Check if model is the base model
        if model_id == config.get("model_name", "biomedlm-2-7b"):
            return BiomedLMModelInfo(
                model_id=model_id,
                display_name="BiomedLM Base Model",
                description="Base BiomedLM model for medical text generation",
                base_model=model_id,
                parameters=2700000000,  # 2.7B parameters
                adapter_type=None,
                fine_tuned_for=None,
                created_at="2023-01-01T00:00:00Z",  # Placeholder
                last_used=format_timestamp(),
                usage_count=100  # Placeholder
            )
        
        # Check if model is an adapter
        for adapter_name, adapter_config in config.get("lora_adapters", {}).items():
            adapter_id = f"{config.get('model_name', 'biomedlm-2-7b')}-{adapter_name}"
            if model_id == adapter_id:
                return BiomedLMModelInfo(
                    model_id=adapter_id,
                    display_name=f"BiomedLM {adapter_name.replace('_', ' ').title()}",
                    description=adapter_config.get("description", ""),
                    base_model=config.get("model_name", "biomedlm-2-7b"),
                    parameters=2700000000,  # 2.7B parameters
                    adapter_type="lora",
                    fine_tuned_for=[adapter_name.replace("_", " ")],
                    created_at="2023-01-01T00:00:00Z",  # Placeholder
                    last_used=format_timestamp(),
                    usage_count=50  # Placeholder
                )
        
        # If model not found
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_id}' not found"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting BiomedLM model '{model_id}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get BiomedLM model '{model_id}': {str(e)}"
        )

@router.post("/generate", response_model=BiomedLMGenerateResponse)
async def generate_text(
    request: BiomedLMGenerateRequest = Body(...),
    current_user: User = Depends(get_current_user)
):
    """
    Generate text using a BiomedLM model.
    
    This endpoint generates text using a BiomedLM model with the specified parameters.
    """
    if not BIOMEDLM_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="BiomedLM is not available. Please check your installation."
        )
    
    try:
        # Get BiomedLM adapter
        adapter = get_biomedlm_adapter()
        
        # Load config from file
        config = load_config(BIOMEDLM_CONFIG_PATH)
        
        # Check if model exists
        models = await get_models(current_user)
        model_ids = [model.model_id for model in models]
        if request.model_id not in model_ids:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{request.model_id}' not found"
            )
        
        # Check if model is an adapter
        adapter_name = None
        if request.model_id != config.get("model_name", "biomedlm-2-7b"):
            # Extract adapter name from model_id
            adapter_name = request.model_id.replace(f"{config.get('model_name', 'biomedlm-2-7b')}-", "")
        
        # Prepare generation parameters
        generation_params = {
            "max_new_tokens": request.max_tokens or config.get("max_new_tokens", 512),
            "temperature": request.temperature or config.get("temperature", 0.2),
            "top_p": request.top_p or config.get("top_p", 0.95),
            "top_k": request.top_k or config.get("top_k", 50),
            "repetition_penalty": request.repetition_penalty or config.get("repetition_penalty", 1.1),
            "do_sample": request.do_sample if request.do_sample is not None else config.get("do_sample", True)
        }
        
        # Generate text
        start_time = datetime.utcnow()
        
        # This would be replaced with actual generation
        # For now, we'll use a mock implementation
        if adapter_name:
            # Load adapter
            # adapter.load_adapter(adapter_name)
            pass
        
        # Generate text
        generated_text = f"This is a mock response from BiomedLM model {request.model_id} for prompt: {request.prompt}"
        
        end_time = datetime.utcnow()
        generation_time_ms = (end_time - start_time).total_seconds() * 1000
        
        return BiomedLMGenerateResponse(
            model_id=request.model_id,
            prompt=request.prompt,
            generated_text=generated_text,
            generation_time_ms=generation_time_ms,
            tokens_generated=len(generated_text.split()),
            created_at=end_time.isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating text with BiomedLM model '{request.model_id}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate text with BiomedLM model '{request.model_id}': {str(e)}"
        )

@router.post("/finetune", response_model=Dict[str, Any])
async def finetune_model(
    request: BiomedLMFinetuneRequest = Body(...),
    current_user: User = Depends(get_current_user)
):
    """
    Fine-tune a BiomedLM model.
    
    This endpoint fine-tunes a BiomedLM model for a specific task.
    """
    if not BIOMEDLM_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="BiomedLM is not available. Please check your installation."
        )
    
    try:
        # Get BiomedLM adapter
        adapter = get_biomedlm_adapter()
        
        # Load config from file
        config = load_config(BIOMEDLM_CONFIG_PATH)
        
        # Check if model exists
        models = await get_models(current_user)
        model_ids = [model.model_id for model in models]
        if request.model_id not in model_ids:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{request.model_id}' not found"
            )
        
        # Prepare fine-tuning parameters
        fine_tuning_params = {
            "learning_rate": request.learning_rate or config.get("fine_tuning", {}).get("learning_rate", 3e-4),
            "batch_size": request.batch_size or config.get("fine_tuning", {}).get("batch_size", 8),
            "num_epochs": request.num_epochs or config.get("fine_tuning", {}).get("num_epochs", 3),
            "lora_r": request.lora_r or config.get("fine_tuning", {}).get("lora_r", 16),
            "lora_alpha": request.lora_alpha or config.get("fine_tuning", {}).get("lora_alpha", 32),
            "lora_dropout": request.lora_dropout or config.get("fine_tuning", {}).get("lora_dropout", 0.05)
        }
        
        # This would be replaced with actual fine-tuning
        # For now, we'll use a mock implementation
        start_time = datetime.utcnow()
        
        # Mock fine-tuning
        await asyncio.sleep(2)  # Simulate fine-tuning time
        
        end_time = datetime.utcnow()
        fine_tuning_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Update config with new adapter
        lora_adapters = config.get("lora_adapters", {})
        lora_adapters[request.adapter_name] = {
            "path": f"./models/lora/{request.adapter_name}",
            "description": f"Fine-tuned for {request.task}"
        }
        config["lora_adapters"] = lora_adapters
        save_config(BIOMEDLM_CONFIG_PATH, config)
        
        return {
            "success": True,
            "message": f"Model '{request.model_id}' fine-tuned successfully with adapter '{request.adapter_name}'",
            "model_id": request.model_id,
            "adapter_name": request.adapter_name,
            "task": request.task,
            "fine_tuning_time_ms": fine_tuning_time_ms,
            "created_at": end_time.isoformat(),
            "parameters": fine_tuning_params
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fine-tuning BiomedLM model '{request.model_id}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fine-tune BiomedLM model '{request.model_id}': {str(e)}"
        )

@router.get("/config", response_model=Dict[str, Any])
async def get_biomedlm_config(current_user: User = Depends(get_current_user)):
    """
    Get the current BiomedLM configuration.
    
    This endpoint returns the current configuration of BiomedLM.
    """
    if not BIOMEDLM_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="BiomedLM is not available. Please check your installation."
        )
    
    try:
        # Load config from file
        config = load_config(BIOMEDLM_CONFIG_PATH)
        return config
    except Exception as e:
        logger.error(f"Error getting BiomedLM config: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get BiomedLM config: {str(e)}"
        )

@router.put("/config", response_model=Dict[str, Any])
async def update_biomedlm_config(
    config: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_user)
):
    """
    Update the BiomedLM configuration.
    
    This endpoint updates the configuration of BiomedLM.
    """
    if not BIOMEDLM_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="BiomedLM is not available. Please check your installation."
        )
    
    try:
        # Save config to file
        save_config(BIOMEDLM_CONFIG_PATH, config)
        return config
    except Exception as e:
        logger.error(f"Error updating BiomedLM config: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update BiomedLM config: {str(e)}"
        )
