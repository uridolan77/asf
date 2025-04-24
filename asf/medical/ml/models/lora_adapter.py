"""
LoRA (Low-Rank Adaptation) Adapter Module for Medical Research Synthesizer.

This module provides a framework for applying LoRA adapters to large language models,
with specific optimizations for GPT-4o-Mini. LoRA allows efficient fine-tuning of
large language models by only training a small number of adapter parameters
while keeping the base model frozen.
"""

import os
import json
import time
import torch
import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field
from transformers import (
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel
)

# Import PEFT for LoRA
try:
    from peft import (
        LoraConfig, 
        TaskType, 
        get_peft_model, 
        PeftModel,
        PeftConfig,
        prepare_model_for_kbit_training
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

from asf.medical.core.logging_config import get_logger
from asf.medical.ml.models.model_registry import (
    ModelRegistry, ModelStatus, ModelMetrics, ModelFramework, get_model_registry
)

logger = get_logger(__name__)

class QuantizationMode(str, Enum):
    """Quantization modes for model loading."""
    NONE = "none"  # No quantization
    INT8 = "int8"  # 8-bit quantization
    INT4 = "int4"  # 4-bit quantization (recommended for GPT-4o-Mini)
    GPTQ = "gptq"  # GPTQ quantization

class LoraAdapterConfig(BaseModel):
    """Configuration for LoRA adapter."""
    base_model_name: str = Field("gpt-4o-mini", description="Base model identifier.")
    adapter_name: str = Field(None, description="Name of the adapter.")
    adapter_version: str = Field("1.0.0", description="Version of the adapter.")
    
    # LoRA configuration
    r: int = Field(16, description="LoRA attention dimension.")
    lora_alpha: int = Field(32, description="Alpha parameter for LoRA scaling.")
    lora_dropout: float = Field(0.05, description="Dropout probability for LoRA layers.")
    
    # Target modules to apply LoRA to
    target_modules: List[str] = Field(
        ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        description="List of modules to apply LoRA to."
    )
    
    # Quantization settings
    quantization_mode: QuantizationMode = Field(
        QuantizationMode.INT4, 
        description="Quantization mode for model loading."
    )
    
    # Performance settings
    use_gradient_checkpointing: bool = Field(
        True, 
        description="Whether to use gradient checkpointing to save memory."
    )
    
    # Directories
    base_model_dir: Optional[str] = Field(
        None, 
        description="Directory for base model cache."
    )
    adapter_dir: Optional[str] = Field(
        None, 
        description="Directory for adapter storage."
    )
    
    # Advanced settings
    bias: str = Field("none", description="Bias type for LoRA.")
    task_type: Optional[str] = Field(
        "CAUSAL_LM",
        description="Task type for the model (CAUSAL_LM, SEQ_CLS, etc.)."
    )
    fan_in_fan_out: bool = Field(
        False,
        description="Set this to True if the layer to replace stores weight like (fan_in, fan_out)."
    )
    
    # Memory optimization
    max_memory_per_gpu: Optional[Dict[int, str]] = Field(
        None,
        description="Max memory per GPU device for model loading. E.g. {0: '6GiB', 1: '6GiB'}"
    )


class LoraAdapter:
    """
    LoRA adapter for efficient fine-tuning and inference with LLMs.
    
    This class manages LoRA adapters for large language models, providing
    functionality for loading, merging, and inference with adapters.
    """
    
    def __init__(self, config: Union[LoraAdapterConfig, Dict[str, Any]]):
        """
        Initialize the LoRA adapter.
        
        Args:
            config: Configuration for the adapter.
        """
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT library is required for LoRA adapters. "
                             "Install it with: pip install peft")
        
        # Convert dict to LoraAdapterConfig if needed
        if isinstance(config, dict):
            self.config = LoraAdapterConfig(**config)
        else:
            self.config = config
            
        # Set default directories if not provided
        if self.config.base_model_dir is None:
            self.config.base_model_dir = os.path.join(
                os.path.expanduser("~"), 
                ".cache", 
                "asf_medical", 
                "models"
            )
            
        if self.config.adapter_dir is None:
            self.config.adapter_dir = os.path.join(
                os.path.expanduser("~"),
                ".cache",
                "asf_medical",
                "adapters"
            )
            
        # Ensure directories exist
        os.makedirs(self.config.base_model_dir, exist_ok=True)
        os.makedirs(self.config.adapter_dir, exist_ok=True)
        
        # Initialize model components
        self.tokenizer = None
        self.base_model = None
        self.adapter_model = None
        self.task_type = self._get_task_type()
        
        logger.info(f"Initialized LoraAdapter with config: {self.config.model_dump()}")

    def _get_task_type(self) -> Any:
        """
        Convert string task type to PEFT TaskType enum.
        
        Returns:
            PEFT TaskType enum value.
        """
        if not self.config.task_type:
            return TaskType.CAUSAL_LM
            
        task_map = {
            "CAUSAL_LM": TaskType.CAUSAL_LM,
            "SEQ_CLS": TaskType.SEQ_CLS,
            "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
            "QUESTION_ANS": TaskType.QUESTION_ANS,
            "FEATURE_EXTRACTION": TaskType.FEATURE_EXTRACTION
        }
        
        return task_map.get(self.config.task_type, TaskType.CAUSAL_LM)
    
    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """
        Get quantization configuration based on the specified mode.
        
        Returns:
            BitsAndBytesConfig for model quantization, or None if no quantization.
        """
        if self.config.quantization_mode == QuantizationMode.NONE:
            return None
            
        elif self.config.quantization_mode == QuantizationMode.INT8:
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )
            
        elif self.config.quantization_mode == QuantizationMode.INT4:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            
        elif self.config.quantization_mode == QuantizationMode.GPTQ:
            # GPTQ is handled differently, through model_kwargs
            return None
            
        return None
    
    def _get_device_map(self) -> Union[str, Dict[str, int]]:
        """
        Get device map for model loading.
        
        Returns:
            Device map configuration.
        """
        # If max memory per GPU is specified, use it
        if self.config.max_memory_per_gpu:
            return "auto"
            
        # Otherwise, determine device map based on available hardware
        if torch.cuda.is_available():
            return "auto"
        else:
            return "cpu"
    
    def _get_model_loading_kwargs(self) -> Dict[str, Any]:
        """
        Get kwargs for model loading.
        
        Returns:
            Dictionary of kwargs for model loading.
        """
        kwargs = {
            "device_map": self._get_device_map(),
            "torch_dtype": torch.bfloat16
        }
        
        # Add quantization config if applicable
        quantization_config = self._get_quantization_config()
        if quantization_config:
            kwargs["quantization_config"] = quantization_config
        
        # Add max memory if specified
        if self.config.max_memory_per_gpu:
            kwargs["max_memory"] = self.config.max_memory_per_gpu
        
        # Add GPTQ-specific settings if needed
        if self.config.quantization_mode == QuantizationMode.GPTQ:
            kwargs["revision"] = "gptq"
            
        return kwargs
    
    def load_base_model(self) -> None:
        """
        Load the base model.
        
        This loads the base model (e.g., GPT-4o-Mini) without any adapters.
        """
        model_kwargs = self._get_model_loading_kwargs()
        
        logger.info(f"Loading base model {self.config.base_model_name} with kwargs: {model_kwargs}")
        start_time = time.time()
        
        try:
            # Load tokenizer first
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model_name,
                cache_dir=self.config.base_model_dir,
                use_fast=True
            )
            
            # Set padding token if needed
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load the base model
            if self.task_type == TaskType.CAUSAL_LM:
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    self.config.base_model_name,
                    cache_dir=self.config.base_model_dir,
                    **model_kwargs
                )
            elif self.task_type == TaskType.SEQ_CLS:
                self.base_model = AutoModelForSequenceClassification.from_pretrained(
                    self.config.base_model_name,
                    cache_dir=self.config.base_model_dir,
                    **model_kwargs
                )
            else:
                # Default to causal LM
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    self.config.base_model_name,
                    cache_dir=self.config.base_model_dir,
                    **model_kwargs
                )
            
            # Prepare model for training if quantized
            if self.config.quantization_mode in [QuantizationMode.INT8, QuantizationMode.INT4]:
                self.base_model = prepare_model_for_kbit_training(
                    self.base_model, 
                    use_gradient_checkpointing=self.config.use_gradient_checkpointing
                )
            
            logger.info(f"Base model loaded in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error loading base model: {str(e)}")
            raise
    
    def create_lora_config(self) -> LoraConfig:
        """
        Create LoRA configuration.
        
        Returns:
            LoraConfig for the model.
        """
        return LoraConfig(
            r=self.config.r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias=self.config.bias,
            task_type=self.task_type,
            fan_in_fan_out=self.config.fan_in_fan_out
        )
    
    def create_adapter_model(self) -> None:
        """
        Create a LoRA adapter model from the base model.
        
        This prepares the model for fine-tuning with LoRA.
        """
        if self.base_model is None:
            self.load_base_model()
        
        # Create LoRA configuration
        lora_config = self.create_lora_config()
        
        logger.info(f"Creating LoRA adapter model with config: {lora_config}")
        
        # Apply LoRA adapters to the model
        self.adapter_model = get_peft_model(self.base_model, lora_config)
        self.adapter_model.print_trainable_parameters()
    
    def load_adapter(self, adapter_path: Optional[str] = None) -> bool:
        """
        Load a LoRA adapter from disk.
        
        Args:
            adapter_path: Path to the adapter. If None, uses config.adapter_name.
            
        Returns:
            True if adapter was successfully loaded, False otherwise.
        """
        if self.base_model is None:
            self.load_base_model()
        
        # Determine adapter path
        if adapter_path is None:
            if self.config.adapter_name is None:
                raise ValueError("Either adapter_path or config.adapter_name must be provided")
            
            adapter_path = os.path.join(
                self.config.adapter_dir,
                self.config.adapter_name,
                self.config.adapter_version
            )
        
        try:
            logger.info(f"Loading adapter from {adapter_path}")
            
            # Check if adapter exists
            if not os.path.exists(adapter_path):
                logger.warning(f"Adapter not found at {adapter_path}")
                return False
            
            # Load the adapter
            self.adapter_model = PeftModel.from_pretrained(
                self.base_model,
                adapter_path
            )
            
            # Update config with actual adapter name and version
            adapter_config = PeftConfig.from_pretrained(adapter_path)
            self.config.adapter_name = os.path.basename(os.path.dirname(adapter_path))
            
            # Try to extract version from directory name
            dir_name = os.path.basename(adapter_path)
            if dir_name.replace('.', '').isdigit():  # Check if looks like a version
                self.config.adapter_version = dir_name
            
            logger.info(f"Successfully loaded adapter {self.config.adapter_name} v{self.config.adapter_version}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading adapter: {str(e)}")
            # Fallback to base model
            self.adapter_model = self.base_model
            return False
    
    def save_adapter(self, output_dir: Optional[str] = None) -> str:
        """
        Save the LoRA adapter to disk.
        
        Args:
            output_dir: Directory to save the adapter. If None, uses config values.
            
        Returns:
            Path to the saved adapter.
        """
        if self.adapter_model is None:
            raise ValueError("No adapter model to save")
        
        # Determine output directory
        if output_dir is None:
            if self.config.adapter_name is None:
                raise ValueError("Either output_dir or config.adapter_name must be provided")
            
            output_dir = os.path.join(
                self.config.adapter_dir,
                self.config.adapter_name,
                self.config.adapter_version
            )
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the adapter
        self.adapter_model.save_pretrained(output_dir)
        
        # Also save the tokenizer
        if self.tokenizer:
            self.tokenizer.save_pretrained(output_dir)
        
        # Save configuration
        with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
            json.dump(self.config.model_dump(), f, indent=2)
        
        logger.info(f"Saved adapter to {output_dir}")
        return output_dir
    
    def generate(
        self,
        inputs: Union[str, List[str]],
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 0.95,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Generate text using the model with the adapter.
        
        Args:
            inputs: Input text or list of texts.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature.
            do_sample: Whether to use sampling for generation.
            top_p: Top-p sampling parameter.
            **kwargs: Additional generation parameters.
            
        Returns:
            Generated text or list of generated texts.
        """
        if self.adapter_model is None:
            if self.base_model is None:
                self.load_base_model()
            model = self.base_model
        else:
            model = self.adapter_model
        
        # Process inputs
        single_input = isinstance(inputs, str)
        if single_input:
            inputs = [inputs]
        
        # Tokenize inputs
        input_ids = self.tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).input_ids.to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p,
                **kwargs
            )
        
        # Decode outputs
        decoded_outputs = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )
        
        # Return single output if single input
        if single_input:
            return decoded_outputs[0]
        else:
            return decoded_outputs
    
    def classify(
        self,
        inputs: Union[str, List[str]],
        **kwargs
    ) -> Union[Dict[str, float], List[Dict[str, float]]]:
        """
        Classify text using the model with the adapter.
        
        Args:
            inputs: Input text or list of texts.
            **kwargs: Additional classification parameters.
            
        Returns:
            Dictionary mapping labels to probabilities or list of such dictionaries.
        """
        if self.task_type != TaskType.SEQ_CLS:
            raise ValueError("classify() can only be used with a sequence classification model")
        
        if self.adapter_model is None:
            if self.base_model is None:
                self.load_base_model()
            model = self.base_model
        else:
            model = self.adapter_model
        
        # Process inputs
        single_input = isinstance(inputs, str)
        if single_input:
            inputs = [inputs]
        
        # Tokenize inputs
        encoded_inputs = self.tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(model.device)
        
        # Get class predictions
        with torch.no_grad():
            logits = model(**encoded_inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # Get label names if available
        if hasattr(self.tokenizer, "id2label"):
            labels = [self.tokenizer.id2label[i] for i in range(probs.shape[1])]
        else:
            labels = [str(i) for i in range(probs.shape[1])]
        
        # Format results
        results = []
        for prob in probs:
            results.append({
                labels[i]: prob[i].item()
                for i in range(len(labels))
            })
        
        # Return single result if single input
        if single_input:
            return results[0]
        else:
            return results
    
    def merge_adapter_with_base_model(self) -> PreTrainedModel:
        """
        Merge the adapter with the base model.
        
        This creates a standalone model with the adapter weights
        merged into the base model weights.
        
        Returns:
            Merged model.
        """
        if self.adapter_model is None:
            raise ValueError("No adapter model to merge")
        
        # Merge adapter with base model
        merged_model = self.adapter_model.merge_and_unload()
        
        logger.info("Successfully merged adapter with base model")
        return merged_model
    
    def get_trainable_parameters(self) -> int:
        """
        Get the number of trainable parameters in the adapter model.
        
        Returns:
            Number of trainable parameters.
        """
        if self.adapter_model is None:
            return 0
        
        return sum(p.numel() for p in self.adapter_model.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model and adapter.
        
        Returns:
            Dictionary with model and adapter information.
        """
        info = {
            "base_model_name": self.config.base_model_name,
            "adapter_name": self.config.adapter_name,
            "adapter_version": self.config.adapter_version,
            "task_type": self.config.task_type,
            "quantization_mode": self.config.quantization_mode,
        }
        
        # Add parameter counts if models are loaded
        if self.base_model is not None:
            info["base_model_parameters"] = sum(p.numel() for p in self.base_model.parameters())
            info["base_model_loaded"] = True
        else:
            info["base_model_loaded"] = False
        
        if self.adapter_model is not None:
            info["adapter_model_parameters"] = sum(p.numel() for p in self.adapter_model.parameters())
            info["trainable_parameters"] = self.get_trainable_parameters()
            info["trainable_parameters_percent"] = (
                self.get_trainable_parameters() / info.get("base_model_parameters", 1) * 100 
                if "base_model_parameters" in info else None
            )
            info["adapter_loaded"] = True
        else:
            info["adapter_loaded"] = False
        
        return info

# Registry to store adapters
class LoraAdapterRegistry:
    """
    Registry for managing LoRA adapters.
    This provides a global singleton registry for LoRA adapters.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LoraAdapterRegistry, cls).__new__(cls)
            cls._instance.adapters = {}
        return cls._instance
    
    def register_adapter(
        self, 
        adapter_id: str, 
        adapter: LoraAdapter
    ) -> None:
        """
        Register a LoRA adapter.
        
        Args:
            adapter_id: ID for the adapter.
            adapter: LoraAdapter instance.
        """
        self.adapters[adapter_id] = adapter
        logger.info(f"Registered adapter with ID: {adapter_id}")
    
    def get_adapter(self, adapter_id: str) -> Optional[LoraAdapter]:
        """
        Get a LoRA adapter by ID.
        
        Args:
            adapter_id: ID of the adapter.
            
        Returns:
            LoraAdapter if found, None otherwise.
        """
        return self.adapters.get(adapter_id)
    
    def remove_adapter(self, adapter_id: str) -> bool:
        """
        Remove a LoRA adapter from the registry.
        
        Args:
            adapter_id: ID of the adapter.
            
        Returns:
            True if adapter was removed, False otherwise.
        """
        if adapter_id in self.adapters:
            del self.adapters[adapter_id]
            logger.info(f"Removed adapter with ID: {adapter_id}")
            return True
        return False
    
    def list_adapters(self) -> Dict[str, Dict[str, Any]]:
        """
        List all registered adapters.
        
        Returns:
            Dictionary mapping adapter IDs to adapter info.
        """
        return {
            adapter_id: adapter.get_model_info()
            for adapter_id, adapter in self.adapters.items()
        }
    
    def clear(self) -> None:
        """Clear all registered adapters."""
        self.adapters = {}
        logger.info("Cleared all adapters from registry")

# Global singleton instance
_adapter_registry = LoraAdapterRegistry()

def get_adapter_registry() -> LoraAdapterRegistry:
    """
    Get the global LoRA adapter registry.
    
    Returns:
        LoraAdapterRegistry instance.
    """
    return _adapter_registry