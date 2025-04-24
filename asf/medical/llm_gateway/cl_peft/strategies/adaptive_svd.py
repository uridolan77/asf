"""
Adaptive SVD strategy for CL-PEFT.

This module implements the Adaptive SVD strategy for mitigating catastrophic forgetting
in sequential fine-tuning of LLMs with PEFT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Union, Tuple
import copy
import numpy as np
from tqdm import tqdm

from peft import PeftModel
from transformers import Trainer

from asf.medical.core.logging_config import get_logger
from .base import CLStrategy

logger = get_logger(__name__)

class AdaptiveSVD(CLStrategy):
    """
    Adaptive SVD strategy for CL-PEFT.

    This strategy projects gradient updates onto low-rank subspaces orthogonal to
    directions important for previous tasks, identified via Singular Value Decomposition.
    """

    def __init__(
        self,
        model: PeftModel,
        svd_rank: int = 10,
        importance_threshold: float = 0.01,
        projection_strength: float = 1.0,
        update_frequency: int = 50,  # Update directions every N steps
        use_incremental_svd: bool = False,
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
            use_incremental_svd: Whether to use incremental SVD updates
            **kwargs: Additional parameters
        """
        super().__init__(model, **kwargs)
        self.svd_rank = svd_rank
        self.importance_threshold = importance_threshold
        self.projection_strength = projection_strength
        self.update_frequency = update_frequency
        self.use_incremental_svd = use_incremental_svd

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
        
        # For incremental SVD
        self.svd_state = {}  # param_name -> (U, S, V)

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
            # Check if we should use incremental SVD
            if self.use_incremental_svd and param_name in self.svd_state:
                # Get previous SVD state
                U_prev, S_prev, V_prev = self.svd_state[param_name]
                
                # Compute incremental SVD update
                U, S, V = self._incremental_svd_update(U_prev, S_prev, V_prev, param_2d)
            else:
                # Compute full SVD
                U, S, V = torch.svd(param_2d)
            
            # Store SVD state for incremental updates
            if self.use_incremental_svd:
                self.svd_state[param_name] = (U.clone(), S.clone(), V.clone())

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
    
    def _incremental_svd_update(self, U, S, V, new_data):
        """
        Update SVD incrementally with new data.
        
        Args:
            U: Left singular vectors
            S: Singular values
            V: Right singular vectors
            new_data: New data matrix
            
        Returns:
            Updated (U, S, V) tuple
        """
        # This is a simplified implementation of incremental SVD
        # A full implementation would use more sophisticated algorithms
        
        # Compute residual
        residual = new_data - U @ torch.diag(S) @ V.T
        
        # Compute QR decomposition of residual
        Q, R = torch.linalg.qr(residual)
        
        # Augment U and V
        U_aug = torch.cat([U, Q], dim=1)
        V_aug = torch.cat([V, torch.zeros((V.shape[0], Q.shape[1]), device=V.device)], dim=1)
        
        # Compute SVD of augmented matrix
        U_new, S_new, V_new = torch.svd(torch.cat([torch.diag(S), R], dim=0))
        
        # Update U and V
        U = U_aug @ U_new
        V = V_aug @ V_new
        
        # Truncate to keep only top-k singular values
        k = min(self.svd_rank * 2, U.shape[1])
        U = U[:, :k]
        S_new = S_new[:k]
        V = V[:, :k]
        
        return U, S_new, V

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

        # Project gradients for each parameter
        for name, param in self.model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue

            # Focus on LoRA parameters
            if 'lora_A' not in name and 'lora_B' not in name:
                continue

            # Check if this parameter has important directions
            has_directions = False
            for task_dirs in self.important_directions.values():
                if name in task_dirs and task_dirs[name]:
                    has_directions = True
                    break

            if not has_directions:
                continue

            # Get projection matrix
            projection = self._build_projection_matrix(name, param.shape)

            # Apply projection to gradient
            if len(param.shape) > 2:
                # Reshape gradient for projection
                original_shape = param.grad.shape
                grad_2d = param.grad.reshape(original_shape[0], -1)

                # Apply projection
                projected_grad = projection @ grad_2d

                # Reshape back
                param.grad = projected_grad.reshape(original_shape)
            else:
                # Apply projection directly
                param.grad = projection @ param.grad
    
    def get_importance_visualization(self, top_k=10):
        """
        Get visualization data for parameter importance.
        
        Args:
            top_k: Number of top parameters to include
            
        Returns:
            Dictionary with visualization data
        """
        # Collect importance data across tasks
        viz_data = {
            "tasks": list(self.parameter_importance.keys()),
            "parameters": [],
            "importance_values": [],
            "singular_values": []
        }
        
        # Get all parameter names
        all_params = set()
        for task_data in self.parameter_importance.values():
            all_params.update(task_data.keys())
        
        # Compute average importance across tasks for each parameter
        avg_importance = {}
        for param_name in all_params:
            values = [task_data.get(param_name, 0) for task_data in self.parameter_importance.values()]
            avg_importance[param_name] = sum(values) / len(values) if values else 0
        
        # Get top-k parameters by average importance
        top_params = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Collect data for visualization
        for param_name, _ in top_params:
            viz_data["parameters"].append(param_name)
            
            # Get importance values across tasks
            importance_values = [task_data.get(param_name, 0) for task_data in self.parameter_importance.values()]
            viz_data["importance_values"].append(importance_values)
            
            # Get singular values for each task
            singular_values = []
            for task_id in viz_data["tasks"]:
                if task_id in self.singular_values and param_name in self.singular_values[task_id]:
                    singular_values.append(self.singular_values[task_id][param_name])
                else:
                    singular_values.append([])
            
            viz_data["singular_values"].append(singular_values)
        
        return viz_data
