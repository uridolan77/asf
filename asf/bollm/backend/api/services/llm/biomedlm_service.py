"""
BiomedLM service for BO backend.

This module provides a service for interacting with BiomedLM.
"""

import os
import yaml
import logging
import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from fastapi import Depends, HTTPException, status

from ...utils import handle_api_error

# Import BiomedLM components if available
try:
    from asf.medical.ml.models.biomedlm_adapter import BiomedLMAdapter
    BIOMEDLM_AVAILABLE = True
except ImportError:
    BIOMEDLM_AVAILABLE = False
    logging.warning("BiomedLM is not available. Some functionality will be limited.")

logger = logging.getLogger(__name__)

# Constants
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 
                          "config", "llm", "biomedlm_config.yaml")

class BiomedLMService:
    """
    Service for interacting with BiomedLM.
    """
    
    def __init__(self):
        """
        Initialize the BiomedLM service.
        """
        self._adapter = None
        self._config = None
        
        if not BIOMEDLM_AVAILABLE:
            logger.warning("BiomedLM is not available. Some functionality will be limited.")
            return
        
        try:
            # Load config from file
            with open(CONFIG_PATH, 'r') as f:
                self._config = yaml.safe_load(f)
            
            # Create BiomedLM adapter
            self._adapter = BiomedLMAdapter(
                model_name=self._config.get("model_name", "biomedlm-2-7b"),
                model_path=os.path.expandvars(self._config.get("model_path", "")),
                use_gpu=self._config.get("use_gpu", True),
                precision=self._config.get("precision", "fp16")
            )
            logger.info(f"BiomedLM adapter initialized with model: {self._config.get('model_name')}")
        except Exception as e:
            logger.error(f"Failed to initialize BiomedLM adapter: {str(e)}")
    
    async def get_models(self) -> List[Dict[str, Any]]:
        """
        Get all available BiomedLM models.
        
        Returns:
            List of model information dictionaries
        """
        if not BIOMEDLM_AVAILABLE or not self._adapter:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="BiomedLM is not available. Please check your installation."
            )
        
        try:
            # Get base model info
            base_model = {
                "model_id": self._config.get("model_name", "biomedlm-2-7b"),
                "display_name": "BiomedLM Base Model",
                "description": "Base BiomedLM model for medical text generation",
                "base_model": self._config.get("model_name", "biomedlm-2-7b"),
                "parameters": 2700000000,  # 2.7B parameters
                "adapter_type": None,
                "fine_tuned_for": None,
                "created_at": "2023-01-01T00:00:00Z",  # Placeholder
                "last_used": datetime.utcnow().isoformat(),
                "usage_count": 100  # Placeholder
            }
            
            # Get adapter models
            adapter_models = []
            for adapter_name, adapter_config in self._config.get("lora_adapters", {}).items():
                adapter_models.append({
                    "model_id": f"{self._config.get('model_name', 'biomedlm-2-7b')}-{adapter_name}",
                    "display_name": f"BiomedLM {adapter_name.replace('_', ' ').title()}",
                    "description": adapter_config.get("description", ""),
                    "base_model": self._config.get("model_name", "biomedlm-2-7b"),
                    "parameters": 2700000000,  # 2.7B parameters
                    "adapter_type": "lora",
                    "fine_tuned_for": [adapter_name.replace("_", " ")],
                    "created_at": "2023-01-01T00:00:00Z",  # Placeholder
                    "last_used": datetime.utcnow().isoformat(),
                    "usage_count": 50  # Placeholder
                })
            
            return [base_model] + adapter_models
        except Exception as e:
            logger.error(f"Error getting BiomedLM models: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get BiomedLM models: {str(e)}"
            )
    
    async def get_model(self, model_id: str) -> Dict[str, Any]:
        """
        Get information about a specific BiomedLM model.
        
        Args:
            model_id: Model ID
            
        Returns:
            Model information dictionary
        """
        if not BIOMEDLM_AVAILABLE or not self._adapter:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="BiomedLM is not available. Please check your installation."
            )
        
        try:
            # Check if model is the base model
            if model_id == self._config.get("model_name", "biomedlm-2-7b"):
                return {
                    "model_id": model_id,
                    "display_name": "BiomedLM Base Model",
                    "description": "Base BiomedLM model for medical text generation",
                    "base_model": model_id,
                    "parameters": 2700000000,  # 2.7B parameters
                    "adapter_type": None,
                    "fine_tuned_for": None,
                    "created_at": "2023-01-01T00:00:00Z",  # Placeholder
                    "last_used": datetime.utcnow().isoformat(),
                    "usage_count": 100  # Placeholder
                }
            
            # Check if model is an adapter
            for adapter_name, adapter_config in self._config.get("lora_adapters", {}).items():
                adapter_id = f"{self._config.get('model_name', 'biomedlm-2-7b')}-{adapter_name}"
                if model_id == adapter_id:
                    return {
                        "model_id": adapter_id,
                        "display_name": f"BiomedLM {adapter_name.replace('_', ' ').title()}",
                        "description": adapter_config.get("description", ""),
                        "base_model": self._config.get("model_name", "biomedlm-2-7b"),
                        "parameters": 2700000000,  # 2.7B parameters
                        "adapter_type": "lora",
                        "fine_tuned_for": [adapter_name.replace("_", " ")],
                        "created_at": "2023-01-01T00:00:00Z",  # Placeholder
                        "last_used": datetime.utcnow().isoformat(),
                        "usage_count": 50  # Placeholder
                    }
            
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
    
    async def generate_text(self, model_id: str, prompt: str, max_tokens: Optional[int] = None,
                           temperature: Optional[float] = None, top_p: Optional[float] = None,
                           top_k: Optional[int] = None, repetition_penalty: Optional[float] = None,
                           do_sample: Optional[bool] = None) -> Dict[str, Any]:
        """
        Generate text using a BiomedLM model.
        
        Args:
            model_id: Model ID
            prompt: Input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty
            do_sample: Whether to use sampling
            
        Returns:
            Generated text and metadata
        """
        if not BIOMEDLM_AVAILABLE or not self._adapter:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="BiomedLM is not available. Please check your installation."
            )
        
        try:
            # Check if model exists
            models = await self.get_models()
            model_ids = [model["model_id"] for model in models]
            if model_id not in model_ids:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model '{model_id}' not found"
                )
            
            # Check if model is an adapter
            adapter_name = None
            if model_id != self._config.get("model_name", "biomedlm-2-7b"):
                # Extract adapter name from model_id
                adapter_name = model_id.replace(f"{self._config.get('model_name', 'biomedlm-2-7b')}-", "")
            
            # Prepare generation parameters
            generation_params = {
                "max_new_tokens": max_tokens or self._config.get("max_new_tokens", 512),
                "temperature": temperature or self._config.get("temperature", 0.2),
                "top_p": top_p or self._config.get("top_p", 0.95),
                "top_k": top_k or self._config.get("top_k", 50),
                "repetition_penalty": repetition_penalty or self._config.get("repetition_penalty", 1.1),
                "do_sample": do_sample if do_sample is not None else self._config.get("do_sample", True)
            }
            
            # Generate text
            start_time = datetime.utcnow()
            
            # This would be replaced with actual generation
            # For now, we'll use a mock implementation
            if adapter_name:
                # Load adapter
                # self._adapter.load_adapter(adapter_name)
                pass
            
            # Generate text
            generated_text = f"This is a mock response from BiomedLM model {model_id} for prompt: {prompt}"
            
            end_time = datetime.utcnow()
            generation_time_ms = (end_time - start_time).total_seconds() * 1000
            
            return {
                "model_id": model_id,
                "prompt": prompt,
                "generated_text": generated_text,
                "generation_time_ms": generation_time_ms,
                "tokens_generated": len(generated_text.split()),
                "created_at": end_time.isoformat()
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error generating text with BiomedLM model '{model_id}': {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate text with BiomedLM model '{model_id}': {str(e)}"
            )
    
    async def finetune_model(self, model_id: str, adapter_name: str, task: str,
                            dataset: Union[str, List[Dict[str, str]]],
                            learning_rate: Optional[float] = None, batch_size: Optional[int] = None,
                            num_epochs: Optional[int] = None, lora_r: Optional[int] = None,
                            lora_alpha: Optional[int] = None, lora_dropout: Optional[float] = None) -> Dict[str, Any]:
        """
        Fine-tune a BiomedLM model.
        
        Args:
            model_id: Model ID
            adapter_name: Adapter name
            task: Task description
            dataset: Training dataset
            learning_rate: Learning rate
            batch_size: Batch size
            num_epochs: Number of epochs
            lora_r: LoRA r parameter
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout parameter
            
        Returns:
            Fine-tuning results
        """
        if not BIOMEDLM_AVAILABLE or not self._adapter:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="BiomedLM is not available. Please check your installation."
            )
        
        try:
            # Check if model exists
            models = await self.get_models()
            model_ids = [model["model_id"] for model in models]
            if model_id not in model_ids:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model '{model_id}' not found"
                )
            
            # Prepare fine-tuning parameters
            fine_tuning_params = {
                "learning_rate": learning_rate or self._config.get("fine_tuning", {}).get("learning_rate", 3e-4),
                "batch_size": batch_size or self._config.get("fine_tuning", {}).get("batch_size", 8),
                "num_epochs": num_epochs or self._config.get("fine_tuning", {}).get("num_epochs", 3),
                "lora_r": lora_r or self._config.get("fine_tuning", {}).get("lora_r", 16),
                "lora_alpha": lora_alpha or self._config.get("fine_tuning", {}).get("lora_alpha", 32),
                "lora_dropout": lora_dropout or self._config.get("fine_tuning", {}).get("lora_dropout", 0.05)
            }
            
            # This would be replaced with actual fine-tuning
            # For now, we'll use a mock implementation
            start_time = datetime.utcnow()
            
            # Mock fine-tuning
            await asyncio.sleep(2)  # Simulate fine-tuning time
            
            end_time = datetime.utcnow()
            fine_tuning_time_ms = (end_time - start_time).total_seconds() * 1000
            
            # Update config with new adapter
            lora_adapters = self._config.get("lora_adapters", {})
            lora_adapters[adapter_name] = {
                "path": f"./models/lora/{adapter_name}",
                "description": f"Fine-tuned for {task}"
            }
            self._config["lora_adapters"] = lora_adapters
            
            # Save config to file
            with open(CONFIG_PATH, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False)
            
            return {
                "success": True,
                "message": f"Model '{model_id}' fine-tuned successfully with adapter '{adapter_name}'",
                "model_id": model_id,
                "adapter_name": adapter_name,
                "task": task,
                "fine_tuning_time_ms": fine_tuning_time_ms,
                "created_at": end_time.isoformat(),
                "parameters": fine_tuning_params
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error fine-tuning BiomedLM model '{model_id}': {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to fine-tune BiomedLM model '{model_id}': {str(e)}"
            )
    
    async def get_config(self) -> Dict[str, Any]:
        """
        Get the current BiomedLM configuration.
        
        Returns:
            BiomedLM configuration
        """
        if not BIOMEDLM_AVAILABLE:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="BiomedLM is not available. Please check your installation."
            )
        
        return self._config
    
    async def update_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the BiomedLM configuration.
        
        Args:
            config: New configuration
            
        Returns:
            Updated configuration
        """
        if not BIOMEDLM_AVAILABLE:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="BiomedLM is not available. Please check your installation."
            )
        
        try:
            # Save config to file
            with open(CONFIG_PATH, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            # Update instance config
            self._config = config
            
            return config
        except Exception as e:
            logger.error(f"Error updating BiomedLM config: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to update BiomedLM config: {str(e)}"
            )

# Singleton instance
_biomedlm_service = None

def get_biomedlm_service():
    """
    Get the BiomedLM service instance.
    
    Returns:
        BiomedLM service instance
    """
    global _biomedlm_service
    
    if _biomedlm_service is None:
        _biomedlm_service = BiomedLMService()
    
    return _biomedlm_service
