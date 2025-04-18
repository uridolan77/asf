"""
LoRA adapter implementation for CL-PEFT.

This module provides the LoRA adapter implementation for CL-PEFT.
"""

import torch
from typing import Dict, List, Optional, Any, Union, Tuple

from transformers import (
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification,
    AutoTokenizer
)

from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training
)

from asf.medical.core.logging_config import get_logger
from asf.medical.ml.cl_peft.config import CLPEFTAdapterConfig, QuantizationMode
from asf.medical.ml.cl_peft.registry import CLPEFTAdapterStatus
from .base import CLPEFTAdapter

logger = get_logger(__name__)

class LoRAAdapter(CLPEFTAdapter):
    """
    LoRA adapter implementation for CL-PEFT.
    
    This class implements the LoRA adapter for CL-PEFT, which uses
    Low-Rank Adaptation for efficient fine-tuning.
    """
    
    def load_base_model(self):
        """
        Load the base model.
        
        This method loads the base model for LoRA fine-tuning.
        """
        logger.info(f"Loading base model {self.config.base_model_name}")
        
        # Update status
        self.registry.update_adapter_status(
            self.config.adapter_id, 
            CLPEFTAdapterStatus.INITIALIZING
        )
        
        try:
            # Load the appropriate model class based on task type
            if self.task_type.name == "CAUSAL_LM":
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    self.config.base_model_name,
                    device_map=self.device_map
                )
            elif self.task_type.name == "SEQ_CLS":
                self.base_model = AutoModelForSequenceClassification.from_pretrained(
                    self.config.base_model_name,
                    device_map=self.device_map
                )
            else:
                # Default to causal LM
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    self.config.base_model_name,
                    device_map=self.device_map
                )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name)
            
            # Update status
            self.registry.update_adapter_status(
                self.config.adapter_id, 
                CLPEFTAdapterStatus.READY
            )
            
            logger.info(f"Successfully loaded base model {self.config.base_model_name}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading base model: {str(e)}")
            self.registry.update_adapter_status(
                self.config.adapter_id, 
                CLPEFTAdapterStatus.ERROR
            )
            raise
    
    def create_lora_config(self) -> LoraConfig:
        """
        Create LoRA configuration.
        
        Returns:
            LoraConfig for the model.
        """
        return LoraConfig(
            r=self.config.lora_r,
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
        
        # Print trainable parameters
        self._print_trainable_parameters()
        
        # Update status
        self.registry.update_adapter_status(
            self.config.adapter_id, 
            CLPEFTAdapterStatus.READY
        )
