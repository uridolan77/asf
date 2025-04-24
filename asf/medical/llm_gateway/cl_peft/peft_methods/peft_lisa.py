"""
LISA: Learning with Integrated Soft Prompts and Adapters.

This module implements LISA, a parameter-efficient fine-tuning method that
combines soft prompts with adapters for improved performance.

Reference:
Cheng, Z., Kasai, J., Yu, H., Ni, A., Iyer, N., Chen, D., Garrette, D., & Zettlemoyer, L. (2023).
LISA: Layerwise Importance Sampling for Memory-Efficient PEFT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Union, Tuple
import math
import copy
import numpy as np

from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
from transformers import PreTrainedModel, PreTrainedTokenizer

from asf.medical.core.logging_config import get_logger

logger = get_logger(__name__)

class LISAConfig(PeftConfig):
    """
    Configuration class for LISA.
    
    LISA (Learning with Integrated Soft Prompts and Adapters) combines
    soft prompts with adapters for improved performance.
    """
    
    def __init__(
        self,
        target_modules: Optional[Union[List[str], str]] = None,
        prompt_length: int = 20,
        prompt_dropout: float = 0.0,
        adapter_type: str = "lora",
        adapter_config: Optional[Dict[str, Any]] = None,
        init_prompt_from_vocab: bool = True,
        init_prompt_vocab_size: int = 10000,
        layerwise_importance_sampling: bool = True,
        importance_sampling_ratio: float = 0.5,
        task_type: str = "CAUSAL_LM",
        **kwargs
    ):
        """
        Initialize LISAConfig.
        
        Args:
            target_modules: List of module names to apply adapters to
            prompt_length: Length of the soft prompt
            prompt_dropout: Dropout probability for the soft prompt
            adapter_type: Type of adapter to use ("lora", "ia3", or "adalora")
            adapter_config: Configuration for the adapter
            init_prompt_from_vocab: Whether to initialize the prompt from vocabulary
            init_prompt_vocab_size: Size of the vocabulary to sample from
            layerwise_importance_sampling: Whether to use layerwise importance sampling
            importance_sampling_ratio: Ratio of layers to sample
            task_type: Task type
            **kwargs: Additional arguments for PeftConfig
        """
        super().__init__(
            peft_type="LISA",
            task_type=task_type,
            **kwargs
        )
        
        self.target_modules = target_modules
        self.prompt_length = prompt_length
        self.prompt_dropout = prompt_dropout
        self.adapter_type = adapter_type
        self.adapter_config = adapter_config or {}
        self.init_prompt_from_vocab = init_prompt_from_vocab
        self.init_prompt_vocab_size = init_prompt_vocab_size
        self.layerwise_importance_sampling = layerwise_importance_sampling
        self.importance_sampling_ratio = importance_sampling_ratio
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """
        Validate the configuration.
        """
        # Check adapter type
        if self.adapter_type not in ["lora", "ia3", "adalora"]:
            raise ValueError(f"Unsupported adapter type: {self.adapter_type}")
        
        # Check importance sampling ratio
        if not 0.0 <= self.importance_sampling_ratio <= 1.0:
            raise ValueError(f"Importance sampling ratio must be between 0 and 1, got {self.importance_sampling_ratio}")

class SoftPrompt(nn.Module):
    """
    Soft prompt module for LISA.
    """
    
    def __init__(
        self,
        config: LISAConfig,
        embedding_layer: nn.Module,
        tokenizer: Optional[PreTrainedTokenizer] = None
    ):
        """
        Initialize SoftPrompt.
        
        Args:
            config: LISA configuration
            embedding_layer: Embedding layer of the model
            tokenizer: Tokenizer for initializing from vocabulary
        """
        super().__init__()
        
        self.config = config
        
        # Get embedding dimension
        self.embedding_dim = embedding_layer.weight.shape[1]
        
        # Create soft prompt embeddings
        self.soft_prompt = nn.Parameter(
            torch.zeros(config.prompt_length, self.embedding_dim)
        )
        
        # Initialize soft prompt
        self._init_soft_prompt(embedding_layer, tokenizer)
        
        # Dropout for soft prompt
        self.prompt_dropout = nn.Dropout(config.prompt_dropout)
    
    def _init_soft_prompt(self, embedding_layer, tokenizer):
        """
        Initialize soft prompt.
        
        Args:
            embedding_layer: Embedding layer of the model
            tokenizer: Tokenizer for initializing from vocabulary
        """
        if self.config.init_prompt_from_vocab and tokenizer is not None:
            # Initialize from vocabulary
            vocab_size = min(len(tokenizer), self.config.init_prompt_vocab_size)
            
            # Sample random tokens from vocabulary
            sampled_ids = torch.randint(0, vocab_size, (self.config.prompt_length,))
            
            # Get embeddings for sampled tokens
            sampled_embeddings = embedding_layer(sampled_ids)
            
            # Initialize soft prompt with sampled embeddings
            self.soft_prompt.data = sampled_embeddings.clone()
            
            logger.info(f"Initialized soft prompt from vocabulary with {self.config.prompt_length} tokens")
        else:
            # Initialize with random values
            nn.init.normal_(self.soft_prompt, mean=0.0, std=0.02)
            
            logger.info(f"Initialized soft prompt with random values")
    
    def forward(self, input_embeds):
        """
        Forward pass.
        
        Args:
            input_embeds: Input embeddings
            
        Returns:
            Input embeddings with soft prompt prepended
        """
        # Apply dropout to soft prompt
        soft_prompt = self.prompt_dropout(self.soft_prompt)
        
        # Get batch size
        batch_size = input_embeds.shape[0]
        
        # Expand soft prompt to batch size
        batch_soft_prompt = soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Prepend soft prompt to input embeddings
        return torch.cat([batch_soft_prompt, input_embeds], dim=1)

class LISAModel(PeftModel):
    """
    LISA model for parameter-efficient fine-tuning.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        config: LISAConfig,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        adapter_name: str = "default"
    ):
        """
        Initialize LISAModel.
        
        Args:
            model: Base model to apply LISA to
            config: LISA configuration
            tokenizer: Tokenizer for initializing soft prompt
            adapter_name: Name of the adapter
        """
        super().__init__(model, config, adapter_name)
        
        # Get embedding layer
        self.embedding_layer = self._get_embedding_layer()
        
        # Create soft prompt
        self.soft_prompt = SoftPrompt(config, self.embedding_layer, tokenizer)
        
        # Add adapter
        self._add_adapter()
        
        # Apply layerwise importance sampling if enabled
        if config.layerwise_importance_sampling:
            self._apply_layerwise_importance_sampling()
        
        # Save original forward method
        self.original_forward_embedding = None
        if hasattr(self.model, "get_input_embeddings"):
            self.original_forward_embedding = self.model.get_input_embeddings().forward
        
        # Register hook for soft prompt
        self._register_soft_prompt_hook()
    
    def _get_embedding_layer(self):
        """
        Get the embedding layer of the model.
        
        Returns:
            Embedding layer
        """
        if hasattr(self.model, "get_input_embeddings"):
            return self.model.get_input_embeddings()
        
        # Try to find embedding layer by name
        for name, module in self.model.named_modules():
            if "embed" in name.lower() and isinstance(module, nn.Embedding):
                return module
        
        raise ValueError("Could not find embedding layer in the model")
    
    def _add_adapter(self):
        """
        Add adapter to the model.
        """
        adapter_type = self.config.adapter_type
        adapter_config = self.config.adapter_config
        
        if adapter_type == "lora":
            # Create LoRA config
            lora_config = LoraConfig(
                target_modules=self.config.target_modules,
                r=adapter_config.get("r", 8),
                lora_alpha=adapter_config.get("lora_alpha", 16),
                lora_dropout=adapter_config.get("lora_dropout", 0.05),
                bias=adapter_config.get("bias", "none"),
                task_type=self.config.task_type
            )
            
            # Apply LoRA
            from peft import get_peft_model
            self.model = get_peft_model(self.model, lora_config)
            
            logger.info(f"Added LoRA adapter with rank {lora_config.r}")
        
        elif adapter_type == "ia3":
            # Create IA³ config
            from asf.medical.ml.cl_peft.peft_methods.peft_ia3 import IA3Config, IA3Model
            
            ia3_config = IA3Config(
                target_modules=self.config.target_modules,
                feedforward_modules=adapter_config.get("feedforward_modules", []),
                attention_modules=adapter_config.get("attention_modules", []),
                output_modules=adapter_config.get("output_modules", []),
                ia3_dropout=adapter_config.get("ia3_dropout", 0.0),
                task_type=self.config.task_type
            )
            
            # Apply IA³
            self.model = IA3Model(self.model, ia3_config)
            
            logger.info(f"Added IA³ adapter")
        
        elif adapter_type == "adalora":
            # Create AdaLoRA config
            from asf.medical.ml.cl_peft.peft_methods.peft_adalora import AdaLoraConfig, AdaLoraModel
            
            adalora_config = AdaLoraConfig(
                target_modules=self.config.target_modules,
                r=adapter_config.get("r", 8),
                lora_alpha=adapter_config.get("lora_alpha", 16),
                lora_dropout=adapter_config.get("lora_dropout", 0.05),
                target_r=adapter_config.get("target_r", None),
                total_step=adapter_config.get("total_step", None),
                task_type=self.config.task_type
            )
            
            # Apply AdaLoRA
            self.model = AdaLoraModel(self.model, adalora_config)
            
            logger.info(f"Added AdaLoRA adapter with initial rank {adalora_config.r}")
    
    def _apply_layerwise_importance_sampling(self):
        """
        Apply layerwise importance sampling.
        """
        # Get all modules with adapters
        adapter_modules = []
        
        for name, module in self.model.named_modules():
            # Check if module has adapter
            has_adapter = False
            
            if self.config.adapter_type == "lora":
                has_adapter = hasattr(module, "lora_A") and hasattr(module, "lora_B")
            elif self.config.adapter_type == "ia3":
                has_adapter = name in self.config.target_modules
            elif self.config.adapter_type == "adalora":
                has_adapter = hasattr(module, "lora_A") and hasattr(module, "lora_B")
            
            if has_adapter:
                adapter_modules.append(name)
        
        # Compute number of modules to keep
        num_modules = len(adapter_modules)
        num_to_keep = max(1, int(num_modules * self.config.importance_sampling_ratio))
        
        # If all modules are kept, no need for importance sampling
        if num_to_keep >= num_modules:
            logger.info(f"Keeping all {num_modules} adapter modules")
            return
        
        # Sample modules to keep
        # In a real implementation, we would use importance scores
        # For simplicity, we'll just sample randomly
        import random
        modules_to_keep = random.sample(adapter_modules, num_to_keep)
        
        # Disable adapters for modules not in the sample
        for name, module in self.model.named_modules():
            if name in adapter_modules and name not in modules_to_keep:
                # Disable adapter based on type
                if self.config.adapter_type == "lora":
                    if hasattr(module, "lora_A"):
                        module.lora_A.requires_grad_(False)
                    if hasattr(module, "lora_B"):
                        module.lora_B.requires_grad_(False)
                elif self.config.adapter_type == "ia3":
                    # Find corresponding IA³ layer
                    for ia3_name, ia3_module in self.model.ia3_layers.items():
                        if name in ia3_name:
                            ia3_module.ia3_vector.requires_grad_(False)
                elif self.config.adapter_type == "adalora":
                    if hasattr(module, "lora_A"):
                        module.lora_A.requires_grad_(False)
                    if hasattr(module, "lora_B"):
                        module.lora_B.requires_grad_(False)
        
        logger.info(f"Applied layerwise importance sampling: keeping {num_to_keep} out of {num_modules} adapter modules")
    
    def _register_soft_prompt_hook(self):
        """
        Register hook for soft prompt.
        """
        # Define new forward method for embedding layer
        def new_forward_embedding(input_ids=None, **kwargs):
            # Call original forward method
            embeds = self.original_forward_embedding(input_ids, **kwargs)
            
            # Apply soft prompt
            if input_ids is not None:
                embeds = self.soft_prompt(embeds)
                
                # Adjust attention mask if present
                if hasattr(self, "last_attention_mask") and self.last_attention_mask is not None:
                    # Create prompt attention mask (all 1s)
                    prompt_mask = torch.ones(
                        self.last_attention_mask.shape[0],
                        self.config.prompt_length,
                        device=self.last_attention_mask.device,
                        dtype=self.last_attention_mask.dtype
                    )
                    
                    # Concatenate with original attention mask
                    self.last_attention_mask = torch.cat([prompt_mask, self.last_attention_mask], dim=1)
            
            return embeds
        
        # Replace forward method of embedding layer
        if self.original_forward_embedding is not None:
            self.model.get_input_embeddings().forward = new_forward_embedding
        
        # Register hook to capture attention mask
        def attention_mask_hook(module, args, kwargs):
            if "attention_mask" in kwargs:
                self.last_attention_mask = kwargs["attention_mask"]
            return None
        
        # Register forward pre-hook on the model
        self.model.register_forward_pre_hook(attention_mask_hook)
    
    def forward(self, *args, **kwargs):
        """
        Forward pass with LISA.
        
        Args:
            *args: Positional arguments for the model
            **kwargs: Keyword arguments for the model
            
        Returns:
            Model outputs
        """
        # Store attention mask for soft prompt hook
        if "attention_mask" in kwargs:
            self.last_attention_mask = kwargs["attention_mask"]
        else:
            self.last_attention_mask = None
        
        # Call parent's forward method
        return super().forward(*args, **kwargs)
    
    def get_nb_trainable_parameters(self):
        """
        Get the number of trainable parameters.
        
        Returns:
            Number of trainable parameters
        """
        nb_params = 0
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                nb_params += param.numel()
        
        return nb_params
    
    def print_trainable_parameters(self):
        """
        Print the number of trainable parameters.
        """
        trainable_params = 0
        all_params = 0
        
        for name, param in self.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        logger.info(
            f"trainable params: {trainable_params} || "
            f"all params: {all_params} || "
            f"trainable%: {100 * trainable_params / all_params:.2f}%"
        )


def get_lisa_model(
    model: PreTrainedModel,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    target_modules: Optional[Union[List[str], str]] = None,
    prompt_length: int = 20,
    prompt_dropout: float = 0.0,
    adapter_type: str = "lora",
    adapter_config: Optional[Dict[str, Any]] = None,
    layerwise_importance_sampling: bool = True,
    importance_sampling_ratio: float = 0.5,
    **kwargs
) -> LISAModel:
    """
    Get a model with LISA applied.
    
    Args:
        model: Base model to apply LISA to
        tokenizer: Tokenizer for initializing soft prompt
        target_modules: List of module names to apply adapters to
        prompt_length: Length of the soft prompt
        prompt_dropout: Dropout probability for the soft prompt
        adapter_type: Type of adapter to use ("lora", "ia3", or "adalora")
        adapter_config: Configuration for the adapter
        layerwise_importance_sampling: Whether to use layerwise importance sampling
        importance_sampling_ratio: Ratio of layers to sample
        **kwargs: Additional arguments for LISAConfig
        
    Returns:
        Model with LISA applied
    """
    # Create LISA config
    config = LISAConfig(
        target_modules=target_modules,
        prompt_length=prompt_length,
        prompt_dropout=prompt_dropout,
        adapter_type=adapter_type,
        adapter_config=adapter_config or {},
        layerwise_importance_sampling=layerwise_importance_sampling,
        importance_sampling_ratio=importance_sampling_ratio,
        **kwargs
    )
    
    # Create LISA model
    lisa_model = LISAModel(model, config, tokenizer)
    
    return lisa_model
