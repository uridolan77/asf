"""
Mask-based strategies for CL-PEFT.

This module implements mask-based strategies for mitigating catastrophic forgetting
in sequential fine-tuning of LLMs with PEFT, including:
- Mask-based CL: Uses binary masks to activate specific parameters for each task
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Union, Tuple
import copy
import numpy as np

from peft import PeftModel
from transformers import Trainer

from asf.medical.core.logging_config import get_logger
from .base import CLStrategy

logger = get_logger(__name__)

class MaskBasedCL(CLStrategy):
    """
    Mask-based Continual Learning strategy for CL-PEFT.
    
    This strategy learns binary masks to activate specific parameters for each task,
    preventing interference between tasks.
    
    Reference:
    Serra, J., Suris, D., Miron, M., & Karatzoglou, A. (2018).
    Overcoming catastrophic forgetting with hard attention to the task.
    International Conference on Machine Learning (ICML).
    """
    
    def __init__(
        self,
        model: PeftModel,
        sparsity: float = 0.5,
        temperature: float = 2.0,
        **kwargs
    ):
        """
        Initialize the Mask-based CL strategy.
        
        Args:
            model: The PEFT model to apply the strategy to
            sparsity: Target sparsity for the masks (0.0 to 1.0)
            temperature: Temperature for the sigmoid function
            **kwargs: Additional parameters
        """
        super().__init__(model, **kwargs)
        self.sparsity = sparsity
        self.temperature = temperature
        
        # Store masks for each task
        self.task_masks = {}  # task_id -> {param_name -> mask}
        
        # Current task mask parameters (trainable)
        self.mask_parameters = {}  # param_name -> mask_param
    
    def before_training(self, task_id: str, **kwargs):
        """
        Prepare for training on a new task.
        
        This method initializes trainable mask parameters for the current task.
        
        Args:
            task_id: Unique identifier for the task
            **kwargs: Additional arguments
        """
        self.current_task_id = task_id
        logger.info(f"Starting training on task {task_id} with Mask-based CL")
        
        # Initialize mask parameters for the current task
        self._initialize_mask_parameters()
    
    def _initialize_mask_parameters(self):
        """
        Initialize trainable mask parameters for the current task.
        """
        # Clear previous mask parameters
        self.mask_parameters = {}
        
        # Initialize mask parameters for each trainable parameter
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Skip non-LoRA parameters
            if 'lora_A' not in name and 'lora_B' not in name:
                continue
            
            # Initialize mask parameters with small random values
            mask_param = torch.randn_like(param) * 0.01
            mask_param.requires_grad = True
            
            # Store mask parameter
            self.mask_parameters[name] = mask_param
            
            # Register mask parameter as a buffer
            self.model.register_buffer(f"mask_param_{name.replace('.', '_')}", mask_param)
    
    def _compute_masks(self):
        """
        Compute binary masks from the trainable mask parameters.
        
        Returns:
            Dictionary mapping parameter names to binary masks
        """
        masks = {}
        
        for name, mask_param in self.mask_parameters.items():
            # Apply sigmoid with temperature
            mask_prob = torch.sigmoid(mask_param / self.temperature)
            
            # Compute threshold for desired sparsity
            if self.sparsity > 0:
                # Sort mask probabilities
                flat_mask = mask_prob.flatten()
                sorted_mask, _ = torch.sort(flat_mask)
                
                # Compute threshold index
                threshold_idx = int(self.sparsity * len(sorted_mask))
                
                # Get threshold value
                threshold = sorted_mask[threshold_idx]
                
                # Apply threshold to get binary mask
                mask = (mask_prob > threshold).float()
            else:
                # No sparsity constraint, use 0.5 as threshold
                mask = (mask_prob > 0.5).float()
            
            masks[name] = mask
        
        return masks
    
    def modify_loss(
        self,
        loss: torch.Tensor,
        model: Optional[PeftModel] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Modify the loss function to incorporate sparsity regularization.
        
        Args:
            loss: Original loss value
            model: The model (optional, uses self.model if not provided)
            **kwargs: Additional arguments
            
        Returns:
            Modified loss with sparsity regularization
        """
        model = model or self.model
        
        # Compute sparsity regularization
        sparsity_loss = torch.tensor(0.0, device=loss.device)
        
        for name, mask_param in self.mask_parameters.items():
            # Apply sigmoid with temperature
            mask_prob = torch.sigmoid(mask_param / self.temperature)
            
            # Compute L1 regularization to encourage sparsity
            sparsity_loss += torch.mean(mask_prob)
        
        # Add sparsity regularization to the loss
        # The factor 0.1 is a hyperparameter that controls the strength of the regularization
        total_loss = loss + 0.1 * (sparsity_loss - self.sparsity)
        
        return total_loss
    
    def after_training(self, task_id: str, **kwargs):
        """
        Perform post-training operations.
        
        This method computes and stores the final binary masks for the current task.
        
        Args:
            task_id: Unique identifier for the task
            **kwargs: Additional arguments
        """
        logger.info(f"Computing and storing masks for task {task_id}")
        
        # Compute final masks
        masks = self._compute_masks()
        
        # Store masks for the current task
        self.task_masks[task_id] = masks
        
        # Log mask statistics
        for name, mask in masks.items():
            sparsity = 1.0 - (torch.sum(mask) / mask.numel()).item()
            logger.info(f"Mask for {name}: sparsity = {sparsity:.4f}")
    
    def modify_gradients(self, **kwargs):
        """
        Modify gradients during training.
        
        This method applies the current task's masks to the gradients and
        ensures that gradients for parameters masked by previous tasks are zero.
        
        Args:
            **kwargs: Additional arguments
        """
        # Compute current masks
        current_masks = self._compute_masks()
        
        # Apply masks to gradients
        for name, param in self.model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            
            # Skip non-LoRA parameters
            if 'lora_A' not in name and 'lora_B' not in name:
                continue
            
            # Apply current mask
            if name in current_masks:
                param.grad *= current_masks[name]
            
            # Ensure gradients for parameters masked by previous tasks are zero
            for task_id, task_masks in self.task_masks.items():
                if task_id != self.current_task_id and name in task_masks:
                    # Zero out gradients for parameters used by previous tasks
                    param.grad *= (1 - task_masks[name])
    
    def get_task_specific_model(self, task_id: str):
        """
        Get a model with only the parameters for a specific task activated.
        
        Args:
            task_id: Unique identifier for the task
            
        Returns:
            Model with task-specific parameters
        """
        if task_id not in self.task_masks:
            raise ValueError(f"No masks found for task {task_id}")
        
        # Create a copy of the model
        task_model = copy.deepcopy(self.model)
        
        # Apply masks for the specified task
        for name, param in task_model.named_parameters():
            if name in self.task_masks[task_id]:
                # Apply mask
                param.data *= self.task_masks[task_id][name]
        
        return task_model
