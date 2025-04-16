"""
Orthogonal PEFT strategies for CL-PEFT.

This module implements orthogonality-based strategies for mitigating catastrophic forgetting
in sequential fine-tuning of LLMs with PEFT, including:
- Orthogonal LoRA (O-LoRA): Enforces orthogonality between LoRA updates for different tasks
- Adaptive SVD: Projects gradient updates onto orthogonal subspaces
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

    Reference:
    Mallya, A., & Lazebnik, S. (2018). PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning.
    In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 7765-7773).
    """

    def __init__(
        self,
        model: PeftModel,
        orthogonal_constraint_lambda: float = 1.0,
        **kwargs
    ):
        """
        Initialize the Orthogonal LoRA strategy.

        Args:
            model: The PEFT model to apply the strategy to
            orthogonal_constraint_lambda: Strength of the orthogonality constraint
            **kwargs: Additional parameters
        """
        super().__init__(model, **kwargs)
        self.orthogonal_constraint_lambda = orthogonal_constraint_lambda

        # Store LoRA matrices for previous tasks
        self.previous_lora_matrices = {}  # task_id -> {param_name -> (A, B)}

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
        if not self.previous_lora_matrices:
            # No previous tasks, no orthogonality constraints needed
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

        logger.info(f"Stored LoRA matrices for {len(task_matrices)} modules for task {task_id}")

    def modify_gradients(self, **kwargs):
        """
        Modify gradients during training.

        For O-LoRA, this projects gradients to be orthogonal to previous tasks.

        Args:
            **kwargs: Additional arguments
        """
        if not self.previous_lora_matrices:
            # No previous tasks, no gradient projection needed
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
                    A_grad -= A_proj * prev_A

                    # Project B gradient to be orthogonal to previous B
                    B_proj = torch.sum(B_grad * prev_B) / torch.sum(prev_B * prev_B)
                    B_grad -= B_proj * prev_B

            # Update gradients
            module.lora_A.weight.grad = A_grad
            module.lora_B.weight.grad = B_grad

class AdaptiveSVD(CLStrategy):
    """
    Adaptive SVD strategy for CL-PEFT.

    This strategy projects gradient updates onto low-rank subspaces orthogonal to
    directions important for previous tasks, identified via Singular Value Decomposition.

    Reference:
    Deng, Y., Shen, Y., Jin, H., & Jiang, X. (2021).
    Adaptive Singular Value Decomposition for Continual Learning with Gradient Projection.
    """

    def __init__(
        self,
        model: PeftModel,
        svd_rank: int = 10,
        importance_threshold: float = 0.01,
        projection_strength: float = 1.0,
        update_frequency: int = 50,  # Update directions every N steps
        **kwargs
    ):
        """
        Initialize the Adaptive SVD strategy.

        Args:
            model: The PEFT model to apply the strategy to
            svd_rank: Rank for SVD decomposition
            importance_threshold: Threshold for identifying important directions
                (relative to largest singular value)
            projection_strength: Strength of gradient projection (0.0 to 1.0)
            update_frequency: Update directions every N steps
            **kwargs: Additional parameters
        """
        super().__init__(model, **kwargs)
        self.svd_rank = svd_rank
        self.importance_threshold = importance_threshold
        self.projection_strength = projection_strength
        self.update_frequency = update_frequency

        # Store important directions for each task
        self.important_directions = {}  # task_id -> {param_name -> list of directions}

        # Singular values for each parameter
        self.singular_values = {}  # task_id -> {param_name -> list of values}

        # Step counter for updating directions
        self.steps = 0

        # Store parameter importance for visualization
        self.parameter_importance = {}  # task_id -> {param_name -> importance}

        # Cache for projection matrices to speed up computation
        self.projection_cache = {}  # param_name -> projection_matrix

    def before_training(self, task_id: str, **kwargs):
        """
        Prepare for training on a new task.

        Initialize tracking structures for the new task.

        Args:
            task_id: Unique identifier for the task
            **kwargs: Additional arguments
        """
        self.current_task_id = task_id
        logger.info(f"Starting training on task {task_id} with Adaptive SVD")

        # Initialize task-specific structures
        if task_id not in self.important_directions:
            self.important_directions[task_id] = {}

        if task_id not in self.singular_values:
            self.singular_values[task_id] = {}

        if task_id not in self.parameter_importance:
            self.parameter_importance[task_id] = {}

        # Clear projection cache
        self.projection_cache = {}

        # Reset step counter
        self.steps = 0

    def modify_loss(self, loss: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Modify the loss function.

        For Adaptive SVD, the loss is not modified as the strategy works by
        projecting gradients.

        Args:
            loss: Original loss value
            **kwargs: Additional arguments

        Returns:
            Modified loss value (same as original for Adaptive SVD)
        """
        # No loss modification needed for Adaptive SVD
        return loss

    def _compute_svd(self, param_name: str, param: torch.Tensor, task_id: str):
        """
        Compute SVD for a parameter and identify important directions.

        Args:
            param_name: Name of the parameter
            param: Parameter tensor
            task_id: Task identifier
        """
        # Skip if this is not a task we're tracking
        if task_id not in self.important_directions:
            return

        # Reset stored directions for this parameter
        self.important_directions[task_id][param_name] = []
        self.singular_values[task_id][param_name] = []

        # Reshape parameter for SVD
        original_shape = param.shape
        if len(original_shape) > 2:
            param_2d = param.reshape(original_shape[0], -1)
        else:
            param_2d = param

        try:
            # Compute SVD
            U, S, V = torch.svd(param_2d)

            # Determine importance threshold
            threshold = self.importance_threshold * S[0].item()

            # Identify important directions based on singular values
            important_indices = torch.where(S > threshold)[0]
            important_indices = important_indices[:min(len(important_indices), self.svd_rank)]

            # Store important directions and their singular values
            for idx in important_indices:
                # Get the left singular vector
                direction = U[:, idx].detach().clone()

                # Store the direction
                self.important_directions[task_id][param_name].append(direction)

                # Store the singular value
                self.singular_values[task_id][param_name].append(S[idx].item())

            # Calculate parameter importance based on singular values
            total_variance = S.sum().item()
            explained_variance = S[important_indices].sum().item()
            importance = explained_variance / total_variance if total_variance > 0 else 0
            self.parameter_importance[task_id][param_name] = importance

            logger.info(f"Task {task_id}: Parameter {param_name} has {len(important_indices)} important directions")

        except Exception as e:
            logger.warning(f"SVD failed for {param_name}: {str(e)}")

    def _build_projection_matrix(self, param_name: str, param_shape: torch.Size):
        """
        Build a projection matrix for efficient gradient projection.

        Args:
            param_name: Name of the parameter
            param_shape: Shape of the parameter

        Returns:
            Projection matrix
        """
        # Check if already cached
        if param_name in self.projection_cache:
            return self.projection_cache[param_name]

        # Reshape if needed
        if len(param_shape) > 2:
            dim_0 = param_shape[0]
            dim_1 = np.prod(param_shape[1:]).astype(int)
        else:
            dim_0, dim_1 = param_shape

        # Start with identity matrix
        projection = torch.eye(dim_0, device=self.model.device)

        # Apply projection for each task's important directions
        for task_id, param_dirs in self.important_directions.items():
            if param_name in param_dirs:
                for direction in param_dirs[param_name]:
                    # Ensure direction is on the correct device
                    direction = direction.to(self.model.device)

                    # Compute outer product for projection
                    projection_component = torch.outer(direction, direction)

                    # Subtract from identity (with strength factor)
                    projection -= self.projection_strength * projection_component

        # Cache for future use
        self.projection_cache[param_name] = projection

        return projection

    def after_training(self, task_id: str, **kwargs):
        """
        Perform post-training operations.

        Compute SVD for all parameters to identify important directions.

        Args:
            task_id: Unique identifier for the task
            **kwargs: Additional arguments
        """
        logger.info(f"Computing SVD for task {task_id} parameters")

        # Compute SVD for each relevant parameter
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # Focus on LoRA parameters
            if 'lora_A' not in name and 'lora_B' not in name:
                continue

            self._compute_svd(name, param.data, task_id)

        logger.info(f"Completed SVD computation for task {task_id}")

    def modify_gradients(self, **kwargs):
        """
        Modify gradients during training.

        Project gradients to be orthogonal to important directions from previous tasks.

        Args:
            **kwargs: Additional arguments
        """
        # Increment step counter
        self.steps += 1

        # Periodically update SVD for the current task
        if self.steps % self.update_frequency == 0:
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue

                # Focus on LoRA parameters
                if 'lora_A' not in name and 'lora_B' not in name:
                    continue

                # Update SVD for current task parameters
                self._compute_svd(name, param.data, self.current_task_id)

            # Clear projection cache as directions have changed
            self.projection_cache = {}

        # Skip if no previous tasks have important directions
        if not any(self.important_directions.get(task_id, {})
                  for task_id in self.task_history if task_id != self.current_task_id):
            return

        # Project gradients for each parameter
        for name, param in self.model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue

            # Focus on LoRA parameters
            if 'lora_A' not in name and 'lora_B' not in name:
                continue

            # Check if this parameter has important directions
            has_directions = False
            for task_id, param_dirs in self.important_directions.items():
                if task_id != self.current_task_id and name in param_dirs and param_dirs[name]:
                    has_directions = True
                    break

            if not has_directions:
                continue

            # Get original gradient
            original_shape = param.grad.shape

            # Reshape gradient to 2D if needed
            if len(original_shape) > 2:
                grad_2d = param.grad.reshape(original_shape[0], -1)
            else:
                grad_2d = param.grad

            # Build or get projection matrix
            projection_matrix = self._build_projection_matrix(name, original_shape)

            # Apply projection
            projected_grad = torch.mm(projection_matrix, grad_2d)

            # Reshape back to original shape if needed
            if len(original_shape) > 2:
                param.grad = projected_grad.reshape(original_shape)
            else:
                param.grad = projected_grad
