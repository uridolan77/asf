"""
IA³: Infused Adapter by Inhibiting and Amplifying Inner Activations.

This module implements IA³, a parameter-efficient fine-tuning method that
modifies inner activations by applying learned vectors that inhibit or amplify
specific dimensions.

Reference:
Liu, H., Tam, D., Muqeeth, M., Mohta, J., Huang, T., Bansal, M., & Raffel, C. (2022).
Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Union, Tuple
import math
import copy
import numpy as np

from peft import PeftModel, PeftConfig, get_peft_model
from transformers import PreTrainedModel

from asf.medical.core.logging_config import get_logger

logger = get_logger(__name__)

class IA3Config(PeftConfig):
    """
    Configuration class for IA³.
    
    IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations) is a
    parameter-efficient fine-tuning method that modifies inner activations by
    applying learned vectors that inhibit or amplify specific dimensions.
    """
    
    def __init__(
        self,
        target_modules: Optional[Union[List[str], str]] = None,
        feedforward_modules: Optional[Union[List[str], str]] = None,
        attention_modules: Optional[Union[List[str], str]] = None,
        output_modules: Optional[Union[List[str], str]] = None,
        init_ia3_weights: bool = True,
        ia3_dropout: float = 0.0,
        module_mapping: Optional[Dict[str, str]] = None,
        task_type: str = "CAUSAL_LM",
        **kwargs
    ):
        """
        Initialize IA3Config.
        
        Args:
            target_modules: List of module names to apply IA³ to
            feedforward_modules: List of feedforward module names
            attention_modules: List of attention module names
            output_modules: List of output module names
            init_ia3_weights: Whether to initialize IA³ weights
            ia3_dropout: Dropout probability for IA³ layers
            module_mapping: Dictionary mapping module names to their types
            task_type: Task type
            **kwargs: Additional arguments for PeftConfig
        """
        super().__init__(
            peft_type="IA3",
            task_type=task_type,
            **kwargs
        )
        
        self.target_modules = target_modules
        self.feedforward_modules = feedforward_modules or []
        self.attention_modules = attention_modules or []
        self.output_modules = output_modules or []
        self.init_ia3_weights = init_ia3_weights
        self.ia3_dropout = ia3_dropout
        self.module_mapping = module_mapping or {}
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """
        Validate the configuration.
        """
        # Check if at least one module type is specified
        if not self.feedforward_modules and not self.attention_modules and not self.output_modules:
            if not self.target_modules:
                raise ValueError(
                    "At least one of feedforward_modules, attention_modules, output_modules, "
                    "or target_modules must be specified"
                )
            
            # If only target_modules is specified, use it for all module types
            logger.warning(
                "Only target_modules is specified. Using it for all module types. "
                "For more fine-grained control, specify feedforward_modules, "
                "attention_modules, and output_modules separately."
            )
            
            if isinstance(self.target_modules, str):
                self.target_modules = [self.target_modules]
            
            self.feedforward_modules = self.target_modules
            self.attention_modules = self.target_modules
            self.output_modules = self.target_modules

class IA3Layer(nn.Module):
    """
    IA³ layer for inhibiting and amplifying inner activations.
    """
    
    def __init__(
        self,
        hidden_size: int,
        ia3_type: str = "feedforward",
        init_ia3_weights: bool = True,
        ia3_dropout: float = 0.0
    ):
        """
        Initialize IA3Layer.
        
        Args:
            hidden_size: Size of the hidden dimension
            ia3_type: Type of IA³ layer ("feedforward", "attention", or "output")
            init_ia3_weights: Whether to initialize IA³ weights
            ia3_dropout: Dropout probability
        """
        super().__init__()
        
        self.ia3_type = ia3_type
        
        # Create IA³ vector
        if ia3_type == "attention":
            # For attention, we need a vector for each head
            self.ia3_vector = nn.Parameter(torch.ones(hidden_size))
        else:
            # For feedforward and output, we need a single vector
            self.ia3_vector = nn.Parameter(torch.ones(hidden_size))
        
        # Initialize weights
        if init_ia3_weights:
            self._init_weights()
        
        # Dropout
        self.dropout = nn.Dropout(ia3_dropout) if ia3_dropout > 0 else nn.Identity()
    
    def _init_weights(self):
        """
        Initialize IA³ weights.
        """
        # Initialize close to 1 for minimal initial impact
        nn.init.normal_(self.ia3_vector, mean=1.0, std=0.01)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with IA³ applied
        """
        # Apply dropout
        ia3_vector = self.dropout(self.ia3_vector)
        
        # Apply IA³ vector based on type
        if self.ia3_type == "attention":
            # For attention, apply to the query vector
            return x * ia3_vector.unsqueeze(0).unsqueeze(0)
        elif self.ia3_type == "feedforward":
            # For feedforward, apply to the intermediate activations
            return x * ia3_vector.unsqueeze(0)
        else:  # output
            # For output, apply to the output activations
            return x * ia3_vector.unsqueeze(0)

class IA3Model(PeftModel):
    """
    IA³ model for parameter-efficient fine-tuning.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        config: IA3Config,
        adapter_name: str = "default"
    ):
        """
        Initialize IA3Model.
        
        Args:
            model: Base model to apply IA³ to
            config: IA³ configuration
            adapter_name: Name of the adapter
        """
        super().__init__(model, config, adapter_name)
        
        # Add IA³ layers to the model
        self._add_ia3_layers()
        
        # Register forward hooks
        self._register_hooks()
    
    def _add_ia3_layers(self):
        """
        Add IA³ layers to the model.
        """
        # Get hidden size from the model config
        hidden_size = self.model.config.hidden_size
        
        # Create IA³ layers
        self.ia3_layers = nn.ModuleDict()
        
        # Add layers for feedforward modules
        for module_name in self.config.feedforward_modules:
            self.ia3_layers[f"{module_name}_feedforward"] = IA3Layer(
                hidden_size=hidden_size,
                ia3_type="feedforward",
                init_ia3_weights=self.config.init_ia3_weights,
                ia3_dropout=self.config.ia3_dropout
            )
        
        # Add layers for attention modules
        for module_name in self.config.attention_modules:
            # For attention, we need to know the head size
            head_size = hidden_size // self.model.config.num_attention_heads
            
            self.ia3_layers[f"{module_name}_attention"] = IA3Layer(
                hidden_size=head_size,
                ia3_type="attention",
                init_ia3_weights=self.config.init_ia3_weights,
                ia3_dropout=self.config.ia3_dropout
            )
        
        # Add layers for output modules
        for module_name in self.config.output_modules:
            self.ia3_layers[f"{module_name}_output"] = IA3Layer(
                hidden_size=hidden_size,
                ia3_type="output",
                init_ia3_weights=self.config.init_ia3_weights,
                ia3_dropout=self.config.ia3_dropout
            )
    
    def _register_hooks(self):
        """
        Register forward hooks for IA³.
        """
        # Register hooks for each module
        for name, module in self.model.named_modules():
            # Check if this module should have IA³ applied
            module_type = self._get_module_type(name)
            
            if module_type:
                # Get the corresponding IA³ layer
                ia3_layer = self.ia3_layers.get(f"{name}_{module_type}")
                
                if ia3_layer:
                    # Register forward hook
                    module.register_forward_hook(self._create_forward_hook(name, module_type, ia3_layer))
    
    def _get_module_type(self, module_name):
        """
        Get the type of a module.
        
        Args:
            module_name: Name of the module
            
        Returns:
            Type of the module ("feedforward", "attention", "output", or None)
        """
        # Check if module is in the mapping
        if self.config.module_mapping and module_name in self.config.module_mapping:
            return self.config.module_mapping[module_name]
        
        # Check if module is in any of the target lists
        if module_name in self.config.feedforward_modules:
            return "feedforward"
        elif module_name in self.config.attention_modules:
            return "attention"
        elif module_name in self.config.output_modules:
            return "output"
        
        return None
    
    def _create_forward_hook(self, name, module_type, ia3_layer):
        """
        Create a forward hook for a module.
        
        Args:
            name: Name of the module
            module_type: Type of the module
            ia3_layer: IA³ layer to apply
            
        Returns:
            Forward hook function
        """
        def hook(module, input, output):
            # Apply IA³ to the output
            return ia3_layer(output)
        
        return hook
    
    def forward(self, *args, **kwargs):
        """
        Forward pass with IA³.
        
        Args:
            *args: Positional arguments for the model
            **kwargs: Keyword arguments for the model
            
        Returns:
            Model outputs
        """
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


def get_ia3_model(
    model: PreTrainedModel,
    target_modules: Optional[Union[List[str], str]] = None,
    feedforward_modules: Optional[Union[List[str], str]] = None,
    attention_modules: Optional[Union[List[str], str]] = None,
    output_modules: Optional[Union[List[str], str]] = None,
    ia3_dropout: float = 0.0,
    **kwargs
) -> IA3Model:
    """
    Get a model with IA³ applied.
    
    Args:
        model: Base model to apply IA³ to
        target_modules: List of module names to apply IA³ to
        feedforward_modules: List of feedforward module names
        attention_modules: List of attention module names
        output_modules: List of output module names
        ia3_dropout: Dropout probability for IA³ layers
        **kwargs: Additional arguments for IA3Config
        
    Returns:
        Model with IA³ applied
    """
    # Create IA³ config
    config = IA3Config(
        target_modules=target_modules,
        feedforward_modules=feedforward_modules,
        attention_modules=attention_modules,
        output_modules=output_modules,
        ia3_dropout=ia3_dropout,
        **kwargs
    )
    
    # Create IA³ model
    ia3_model = IA3Model(model, config)
    
    return ia3_model
