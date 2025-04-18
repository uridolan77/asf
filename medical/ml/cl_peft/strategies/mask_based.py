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
        mask_lr: float = 0.01,
        regularization_strength: float = 0.1,
        **kwargs
    ):
        """
        Initialize the Mask-based CL strategy.

        Args:
            model: The PEFT model to apply the strategy to
            sparsity: Target sparsity for the masks (0.0 to 1.0)
            temperature: Temperature for the sigmoid function
            mask_lr: Learning rate for mask parameters
            regularization_strength: Strength of sparsity regularization
            **kwargs: Additional parameters
        """
        super().__init__(model, **kwargs)
        self.sparsity = sparsity
        self.temperature = temperature
        self.mask_lr = mask_lr
        self.regularization_strength = regularization_strength

        # Store masks for each task
        self.task_masks = {}  # task_id -> {param_name -> mask}

        # Current task mask parameters (trainable)
        self.mask_parameters = {}  # param_name -> mask_param

        # Optimizer for mask parameters
        self.mask_optimizer = None

        # Binary mask cache during training
        self.binary_masks = {}  # param_name -> binary_mask

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

        # Create optimizer for mask parameters
        self._create_mask_optimizer()

    def _initialize_mask_parameters(self):
        """
        Initialize trainable mask parameters for the current task.
        """
        # Clear previous mask parameters
        self.mask_parameters = {}
        self.binary_masks = {}

        # Get a list of the previous tasks' masks
        previous_masks = {}
        for task_id, task_masks in self.task_masks.items():
            for name, mask in task_masks.items():
                if name not in previous_masks:
                    previous_masks[name] = []
                previous_masks[name].append(mask)

        # Initialize mask parameters for each trainable LoRA parameter
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # Focus on LoRA parameters
            if 'lora_A' not in name and 'lora_B' not in name:
                continue

            # Get combined previous mask if any
            combined_prev_mask = None
            if name in previous_masks and previous_masks[name]:
                # Combine masks from previous tasks (union)
                combined_prev_mask = torch.zeros_like(param.data, dtype=torch.float32)
                for mask in previous_masks[name]:
                    combined_prev_mask = torch.max(combined_prev_mask, mask.to(combined_prev_mask.device))

            # Initialize mask parameters with small random values
            if combined_prev_mask is not None:
                # Initialize to avoid overlapping with previous tasks
                # Use negative values for areas already masked by previous tasks
                mask_param = torch.randn_like(param.data) * 0.01
                mask_param[combined_prev_mask > 0.5] = -10.0  # Strongly negative to ensure sigmoid ≈ 0
            else:
                # No previous masks, initialize with small random values
                mask_param = torch.randn_like(param.data) * 0.01

            # Ensure mask parameters are trainable
            mask_param = nn.Parameter(mask_param)

            # Store mask parameter
            self.mask_parameters[name] = mask_param

            # Register mask parameter as buffer in model for checkpointing
            buffer_name = f"mask_param_{name.replace('.', '_')}"
            if hasattr(self.model, buffer_name):
                # Update existing buffer
                getattr(self.model, buffer_name).data = mask_param.data
            else:
                # Register new buffer
                self.model.register_buffer(buffer_name, mask_param.data, persistent=True)

    def _create_mask_optimizer(self):
        """
        Create optimizer for mask parameters.
        """
        # Collect mask parameters
        mask_params = list(self.mask_parameters.values())

        # Create optimizer
        self.mask_optimizer = torch.optim.Adam(mask_params, lr=self.mask_lr)

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

            # Compute L1 regularization to encourage target sparsity
            sparsity_loss += torch.abs(mask_prob.mean() - (1.0 - self.sparsity))

        # Scale regularization loss
        sparsity_loss *= self.regularization_strength

        # Add sparsity regularization to the loss
        total_loss = loss + sparsity_loss

        # Update masks after each loss computation
        # This ensures masks are updated throughout training
        self._update_masks()

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

    def _update_masks(self):
        """
        Update mask parameters with gradient descent.
        """
        # Zero gradients
        self.mask_optimizer.zero_grad()

        # Compute sparsity regularization
        sparsity_loss = torch.tensor(0.0, device=self.model.device)

        for name, mask_param in self.mask_parameters.items():
            # Apply sigmoid with temperature
            mask_prob = torch.sigmoid(mask_param / self.temperature)

            # Compute L1 regularization to encourage sparsity
            # Target sparsity rate but allow some flexibility
            sparsity_loss += torch.abs(mask_prob.mean() - (1.0 - self.sparsity))

        # Scale regularization loss
        sparsity_loss *= self.regularization_strength

        # Backward pass for sparsity loss
        sparsity_loss.backward()

        # Optimizer step
        self.mask_optimizer.step()

    def modify_gradients(self, **kwargs):
        """
        Modify gradients during training.

        This method applies the current task's masks to the gradients and
        ensures that gradients for parameters masked by previous tasks are zero.

        Args:
            **kwargs: Additional arguments
        """
        # Compute current masks if needed
        if not self.binary_masks:
            self.binary_masks = self._compute_masks()

        # Apply masks to gradients
        for name, param in self.model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue

            # Skip non-LoRA parameters
            if 'lora_A' not in name and 'lora_B' not in name:
                continue

            # Apply current mask
            if name in self.binary_masks:
                param.grad *= self.binary_masks[name]

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
