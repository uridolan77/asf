"""
Orthogonal LoRA (O-LoRA) strategy for CL-PEFT.

This module implements the Orthogonal LoRA strategy for mitigating catastrophic forgetting
in sequential fine-tuning of LLMs with PEFT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Union, Tuple
import copy
import numpy as np

from peft import PeftModel, LoraConfig, get_peft_model
from transformers import Trainer

from asf.medical.core.logging_config import get_logger
from .base import CLStrategy

logger = get_logger(__name__)

class OrthogonalLoRA(CLStrategy):
    """
    Orthogonal LoRA (O-LoRA) strategy for CL-PEFT.

    This strategy enforces orthogonality between LoRA updates for different tasks
    to reduce interference and mitigate catastrophic forgetting.
    """

    def __init__(
        self,
        model: PeftModel,
        orthogonal_constraint_lambda: float = 1.0,
        orthogonal_projection_alpha: float = 1.0,
        use_gradient_projection: bool = True,
        use_loss_regularization: bool = True,
        **kwargs
    ):
        """
        Initialize the Orthogonal LoRA strategy.

        Args:
            model: The PEFT model to apply the strategy to
            orthogonal_constraint_lambda: Strength of the orthogonality constraint in loss
            orthogonal_projection_alpha: Strength of the orthogonal projection (0 to 1)
            use_gradient_projection: Whether to use gradient projection
            use_loss_regularization: Whether to use loss regularization
            **kwargs: Additional parameters
        """
        super().__init__(model, **kwargs)
        self.orthogonal_constraint_lambda = orthogonal_constraint_lambda
        self.orthogonal_projection_alpha = orthogonal_projection_alpha
        self.use_gradient_projection = use_gradient_projection
        self.use_loss_regularization = use_loss_regularization

        # Store LoRA matrices for previous tasks
        self.previous_lora_matrices = {}  # task_id -> {param_name -> (A, B)}
        
        # Cache for orthogonal projections
        self.projection_cache = {}  # param_name -> projection_matrix
        
        # For visualization and analysis
        self.orthogonality_metrics = {}  # task_id -> {param_name -> orthogonality_score}

    def before_training(self, task_id: str, **kwargs):
        """
        Prepare for training on a new task.

        For O-LoRA, this initializes the LoRA matrices for the new task
        to be orthogonal to previous tasks' matrices.

        Args:
            task_id: Unique identifier for the task
            **kwargs: Additional arguments
        """
        self.current_task_id = task_id
        logger.info(f"Starting training on task {task_id} with Orthogonal LoRA")

        # If there are no previous tasks, no orthogonalization needed
        if not self.previous_lora_matrices:
            logger.info("No previous tasks, skipping orthogonalization")
            return

        # Initialize LoRA matrices to be orthogonal to previous tasks
        self._initialize_orthogonal_lora()
        
        # Clear projection cache
        self.projection_cache = {}

    def _initialize_orthogonal_lora(self):
        """
        Initialize LoRA matrices to be orthogonal to previous tasks.
        """
        # Get all LoRA modules
        lora_modules = {}
        for name, module in self.model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                lora_modules[name] = module

        # For each LoRA module, initialize A and B to be orthogonal to previous tasks
        for name, module in lora_modules.items():
            # Get previous A and B matrices for this module
            prev_A_matrices = []
            prev_B_matrices = []

            for task_matrices in self.previous_lora_matrices.values():
                if name in task_matrices:
                    prev_A, prev_B = task_matrices[name]
                    prev_A_matrices.append(prev_A)
                    prev_B_matrices.append(prev_B)

            # If no previous matrices for this module, skip
            if not prev_A_matrices:
                continue

            # Initialize A to be orthogonal to previous A matrices
            if prev_A_matrices:
                # Stack previous A matrices
                prev_A_stacked = torch.cat(prev_A_matrices, dim=0)

                # Compute orthogonal basis to previous A matrices
                Q, _ = torch.linalg.qr(prev_A_stacked.T)

                # Initialize A to be in the null space of previous A matrices
                null_space_dim = Q.shape[1] - prev_A_stacked.shape[0]
                if null_space_dim > 0:
                    # Use the last columns of Q as the null space basis
                    null_space_basis = Q[:, -null_space_dim:]

                    # Initialize A using the null space basis
                    with torch.no_grad():
                        module.lora_A.weight.copy_(
                            torch.randn(module.lora_A.weight.shape[0], null_space_dim) @
                            null_space_basis.T
                        )

            # Initialize B to be orthogonal to previous B matrices
            if prev_B_matrices:
                # Stack previous B matrices
                prev_B_stacked = torch.cat(prev_B_matrices, dim=0)

                # Compute orthogonal basis to previous B matrices
                Q, _ = torch.linalg.qr(prev_B_stacked.T)

                # Initialize B to be in the null space of previous B matrices
                null_space_dim = Q.shape[1] - prev_B_stacked.shape[0]
                if null_space_dim > 0:
                    # Use the last columns of Q as the null space basis
                    null_space_basis = Q[:, -null_space_dim:]

                    # Initialize B using the null space basis
                    with torch.no_grad():
                        module.lora_B.weight.copy_(
                            torch.randn(module.lora_B.weight.shape[0], null_space_dim) @
                            null_space_basis.T
                        )

    def modify_loss(
        self,
        loss: torch.Tensor,
        model: Optional[PeftModel] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Modify the loss function to incorporate orthogonality constraints.

        Args:
            loss: Original loss value
            model: The model (optional, uses self.model if not provided)
            **kwargs: Additional arguments

        Returns:
            Modified loss with orthogonality constraints
        """
        if not self.previous_lora_matrices or not self.use_loss_regularization:
            # No previous tasks or loss regularization disabled
            return loss

        model = model or self.model

        # Compute orthogonality constraint loss
        ortho_loss = torch.tensor(0.0, device=loss.device)

        # Get all LoRA modules
        lora_modules = {}
        for name, module in model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                lora_modules[name] = module

        # For each LoRA module, compute orthogonality constraints
        for name, module in lora_modules.items():
            # Get current A and B matrices
            current_A = module.lora_A.weight
            current_B = module.lora_B.weight

            # Compute orthogonality constraints with previous tasks
            for task_matrices in self.previous_lora_matrices.values():
                if name in task_matrices:
                    prev_A, prev_B = task_matrices[name]

                    # Compute dot products between current and previous matrices
                    A_dot = torch.sum(current_A * prev_A)
                    B_dot = torch.sum(current_B * prev_B)

                    # Add to orthogonality loss
                    ortho_loss += A_dot.pow(2) + B_dot.pow(2)

        # Add orthogonality constraint to the loss
        total_loss = loss + (self.orthogonal_constraint_lambda * ortho_loss)

        return total_loss

    def after_training(self, task_id: str, **kwargs):
        """
        Perform post-training operations.

        This method stores the LoRA matrices for the current task.

        Args:
            task_id: Unique identifier for the task
            **kwargs: Additional arguments
        """
        logger.info(f"Storing LoRA matrices for task {task_id}")

        # Store LoRA matrices for the current task
        task_matrices = {}

        # Get all LoRA modules
        for name, module in self.model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                # Store A and B matrices
                task_matrices[name] = (
                    module.lora_A.weight.detach().clone(),
                    module.lora_B.weight.detach().clone()
                )

        # Store matrices for this task
        self.previous_lora_matrices[task_id] = task_matrices
        
        # Compute and store orthogonality metrics
        self._compute_orthogonality_metrics(task_id)

        logger.info(f"Stored LoRA matrices for {len(task_matrices)} modules for task {task_id}")

    def modify_gradients(self, **kwargs):
        """
        Modify gradients during training.

        For O-LoRA, this projects gradients to be orthogonal to previous tasks.

        Args:
            **kwargs: Additional arguments
        """
        if not self.previous_lora_matrices or not self.use_gradient_projection:
            # No previous tasks or gradient projection disabled
            return

        # Get all LoRA modules
        lora_modules = {}
        for name, module in self.model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                lora_modules[name] = module

        # For each LoRA module, project gradients to be orthogonal to previous tasks
        for name, module in lora_modules.items():
            # Get gradients for A and B
            if module.lora_A.weight.grad is None or module.lora_B.weight.grad is None:
                continue

            A_grad = module.lora_A.weight.grad
            B_grad = module.lora_B.weight.grad

            # Project gradients for each previous task
            for task_matrices in self.previous_lora_matrices.values():
                if name in task_matrices:
                    prev_A, prev_B = task_matrices[name]

                    # Project A gradient to be orthogonal to previous A
                    A_proj = torch.sum(A_grad * prev_A) / torch.sum(prev_A * prev_A)
                    A_grad = A_grad - (self.orthogonal_projection_alpha * A_proj * prev_A)

                    # Project B gradient to be orthogonal to previous B
                    B_proj = torch.sum(B_grad * prev_B) / torch.sum(prev_B * prev_B)
                    B_grad = B_grad - (self.orthogonal_projection_alpha * B_proj * prev_B)

            # Update gradients
            module.lora_A.weight.grad = A_grad
            module.lora_B.weight.grad = B_grad
    
    def _compute_orthogonality_metrics(self, task_id):
        """
        Compute orthogonality metrics for visualization and analysis.
        
        Args:
            task_id: Task identifier
        """
        # Initialize orthogonality metrics for this task
        self.orthogonality_metrics[task_id] = {}
        
        # Skip if this is the first task
        if len(self.previous_lora_matrices) <= 1:
            return
        
        # Get matrices for the current task
        current_matrices = self.previous_lora_matrices[task_id]
        
        # Compute orthogonality with previous tasks
        for prev_task_id, prev_matrices in self.previous_lora_matrices.items():
            # Skip current task
            if prev_task_id == task_id:
                continue
            
            # Compute orthogonality for each module
            for name, (current_A, current_B) in current_matrices.items():
                if name in prev_matrices:
                    prev_A, prev_B = prev_matrices[name]
                    
                    # Compute cosine similarity between A matrices
                    A_sim = torch.sum(current_A * prev_A) / (
                        torch.norm(current_A) * torch.norm(prev_A)
                    )
                    
                    # Compute cosine similarity between B matrices
                    B_sim = torch.sum(current_B * prev_B) / (
                        torch.norm(current_B) * torch.norm(prev_B)
                    )
                    
                    # Compute overall orthogonality (1 - similarity)
                    orthogonality = 1.0 - (0.5 * (A_sim.abs() + B_sim.abs())).item()
                    
                    # Store orthogonality metric
                    if name not in self.orthogonality_metrics[task_id]:
                        self.orthogonality_metrics[task_id][name] = []
                    
                    self.orthogonality_metrics[task_id][name].append(orthogonality)
        
        # Compute average orthogonality for each module
        for name, values in self.orthogonality_metrics[task_id].items():
            self.orthogonality_metrics[task_id][name] = sum(values) / len(values)
    
    def get_orthogonality_visualization(self):
        """
        Get visualization data for orthogonality metrics.
        
        Returns:
            Dictionary with visualization data
        """
        # Collect orthogonality data across tasks
        viz_data = {
            "tasks": list(self.orthogonality_metrics.keys()),
            "modules": [],
            "orthogonality_values": []
        }
        
        # Get all module names
        all_modules = set()
        for task_data in self.orthogonality_metrics.values():
            all_modules.update(task_data.keys())
        
        # Collect data for visualization
        for module_name in all_modules:
            viz_data["modules"].append(module_name)
            
            # Get orthogonality values across tasks
            values = [task_data.get(module_name, 0) for task_data in self.orthogonality_metrics.values()]
            viz_data["orthogonality_values"].append(values)
        
        return viz_data
