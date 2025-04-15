"""
Unified LLM service for BO backend.

This module provides a unified service for interacting with all LLM components,
including LLM Gateway, DSPy, and BiomedLM.
"""

import os
import yaml
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from fastapi import Depends, HTTPException, status

from ...utils import handle_api_error
from .gateway_service import LLMGatewayService, get_llm_gateway_service
from .dspy_service import DSPyService, get_dspy_service
from .biomedlm_service import BiomedLMService, get_biomedlm_service

logger = logging.getLogger(__name__)

class LLMService:
    """
    Unified service for interacting with all LLM components.
    """
    
    def __init__(
        self,
        gateway_service: LLMGatewayService = None,
        dspy_service: DSPyService = None,
        biomedlm_service: BiomedLMService = None
    ):
        """
        Initialize the LLM service.
        
        Args:
            gateway_service: LLM Gateway service
            dspy_service: DSPy service
            biomedlm_service: BiomedLM service
        """
        self.gateway_service = gateway_service or get_llm_gateway_service()
        self.dspy_service = dspy_service or get_dspy_service()
        self.biomedlm_service = biomedlm_service or get_biomedlm_service()
        
        logger.info("LLM service initialized")
    
    async def get_status(self) -> Dict[str, Any]:
        """
        Get the status of all LLM components.
        
        Returns:
            Status information for all LLM components
        """
        status = {
            "timestamp": datetime.utcnow().isoformat(),
            "components": {}
        }
        
        # Get LLM Gateway status
        try:
            gateway_status = await self.gateway_service.get_status()
            status["components"]["gateway"] = {
                "status": "available",
                "details": gateway_status
            }
        except Exception as e:
            status["components"]["gateway"] = {
                "status": "unavailable",
                "error": str(e)
            }
        
        # Get DSPy status
        try:
            dspy_modules = await self.dspy_service.get_modules()
            status["components"]["dspy"] = {
                "status": "available",
                "modules_count": len(dspy_modules),
                "modules": [module["name"] for module in dspy_modules]
            }
        except Exception as e:
            status["components"]["dspy"] = {
                "status": "unavailable",
                "error": str(e)
            }
        
        # Get BiomedLM status
        try:
            biomedlm_models = await self.biomedlm_service.get_models()
            status["components"]["biomedlm"] = {
                "status": "available",
                "models_count": len(biomedlm_models),
                "models": [model["model_id"] for model in biomedlm_models]
            }
        except Exception as e:
            status["components"]["biomedlm"] = {
                "status": "unavailable",
                "error": str(e)
            }
        
        # Overall status
        component_statuses = [comp["status"] for comp in status["components"].values()]
        if all(s == "available" for s in component_statuses):
            status["overall_status"] = "operational"
        elif any(s == "available" for s in component_statuses):
            status["overall_status"] = "degraded"
        else:
            status["overall_status"] = "unavailable"
        
        return status
    
    async def generate_text(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate text using the appropriate LLM component.
        
        Args:
            request_data: Request data including:
                - component: LLM component to use ("gateway", "dspy", or "biomedlm")
                - model: Model ID
                - prompt: Input prompt
                - Additional parameters specific to each component
            
        Returns:
            Generated text and metadata
        """
        component = request_data.get("component", "gateway")
        
        if component == "gateway":
            return await self.gateway_service.generate(request_data)
        elif component == "dspy":
            module_name = request_data.get("module_name")
            if not module_name:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Module name is required for DSPy generation"
                )
            
            inputs = {
                "question": request_data.get("prompt")
            }
            if "inputs" in request_data:
                inputs.update(request_data["inputs"])
            
            return await self.dspy_service.execute_module(
                module_name=module_name,
                inputs=inputs,
                config=request_data.get("config")
            )
        elif component == "biomedlm":
            return await self.biomedlm_service.generate_text(
                model_id=request_data.get("model"),
                prompt=request_data.get("prompt"),
                max_tokens=request_data.get("max_tokens"),
                temperature=request_data.get("temperature"),
                top_p=request_data.get("top_p"),
                top_k=request_data.get("top_k"),
                repetition_penalty=request_data.get("repetition_penalty"),
                do_sample=request_data.get("do_sample")
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported component: {component}"
            )
    
    async def get_available_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all available models from all LLM components.
        
        Returns:
            Dictionary of models grouped by component
        """
        models = {
            "gateway": [],
            "dspy": [],
            "biomedlm": []
        }
        
        # Get LLM Gateway models
        try:
            providers = await self.gateway_service.get_providers()
            for provider in providers:
                for model_name in provider.get("models", []):
                    models["gateway"].append({
                        "model_id": model_name,
                        "provider_id": provider["provider_id"],
                        "display_name": f"{provider.get('display_name', provider['provider_id'])} - {model_name}",
                        "type": "gateway"
                    })
        except Exception as e:
            logger.error(f"Error getting LLM Gateway models: {str(e)}")
        
        # Get DSPy modules
        try:
            modules = await self.dspy_service.get_modules()
            for module in modules:
                models["dspy"].append({
                    "model_id": module["name"],
                    "display_name": module.get("description", module["name"]),
                    "signature": module.get("signature", ""),
                    "type": "dspy"
                })
        except Exception as e:
            logger.error(f"Error getting DSPy modules: {str(e)}")
        
        # Get BiomedLM models
        try:
            biomedlm_models = await self.biomedlm_service.get_models()
            for model in biomedlm_models:
                models["biomedlm"].append({
                    "model_id": model["model_id"],
                    "display_name": model["display_name"],
                    "description": model.get("description", ""),
                    "adapter_type": model.get("adapter_type"),
                    "type": "biomedlm"
                })
        except Exception as e:
            logger.error(f"Error getting BiomedLM models: {str(e)}")
        
        return models
    
    async def get_usage_statistics(self) -> Dict[str, Any]:
        """
        Get usage statistics for all LLM components.
        
        Returns:
            Usage statistics
        """
        usage = {
            "timestamp": datetime.utcnow().isoformat(),
            "components": {}
        }
        
        # Get LLM Gateway usage
        try:
            gateway_usage = await self.gateway_service.get_usage()
            usage["components"]["gateway"] = gateway_usage
        except Exception as e:
            usage["components"]["gateway"] = {
                "error": str(e)
            }
        
        # DSPy and BiomedLM usage would be added here
        # For now, we'll use placeholder data
        usage["components"]["dspy"] = {
            "total_requests": 50,
            "modules": {
                "medical_rag": 20,
                "contradiction_detection": 15,
                "evidence_extraction": 10,
                "medical_summarization": 5
            }
        }
        
        usage["components"]["biomedlm"] = {
            "total_requests": 30,
            "models": {
                "biomedlm-2-7b": 20,
                "biomedlm-2-7b-contradiction_detection": 10
            }
        }
        
        # Calculate totals
        total_requests = (
            usage["components"]["gateway"].get("total_requests", 0) +
            usage["components"]["dspy"].get("total_requests", 0) +
            usage["components"]["biomedlm"].get("total_requests", 0)
        )
        
        usage["total_requests"] = total_requests
        
        return usage

# Singleton instance
_llm_service = None

def get_llm_service():
    """
    Get the LLM service instance.
    
    Returns:
        LLM service instance
    """
    global _llm_service
    
    if _llm_service is None:
        _llm_service = LLMService()
    
    return _llm_service
