"""
Enhanced Mask-based strategies for CL-PEFT.

This module implements enhanced mask-based strategies for mitigating catastrophic forgetting
in sequential fine-tuning of LLMs with PEFT, including:
- Enhanced Mask-based CL: Uses binary masks with improved initialization and visualization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Union, Tuple
import copy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import io
from PIL import Image

from peft import PeftModel
from transformers import Trainer

from asf.medical.core.logging_config import get_logger
from .mask_based import MaskBasedCL

logger = get_logger(__name__)

class EnhancedMaskBasedCL(MaskBasedCL):
    """
    Enhanced Mask-based Continual Learning strategy for CL-PEFT.

    This strategy extends the base MaskBasedCL with improved initialization,
    visualization, and analysis capabilities.
    """

    def __init__(
        self,
        model: PeftModel,
        sparsity: float = 0.5,
        temperature: float = 2.0,
        mask_lr: float = 0.01,
        regularization_strength: float = 0.1,
        mask_init_strategy: str = "random",
        use_supermask: bool = False,
        mask_overlap_penalty: float = 0.1,
        **kwargs
    ):
        """
        Initialize the Enhanced Mask-based CL strategy.

        Args:
            model: The PEFT model to apply the strategy to
            sparsity: Target sparsity for the masks (0.0 to 1.0)
            temperature: Temperature for the sigmoid function
            mask_lr: Learning rate for mask parameters
            regularization_strength: Strength of sparsity regularization
            mask_init_strategy: Strategy for initializing masks ("random", "magnitude", "gradient")
            use_supermask: Whether to use supermask technique for faster training
            mask_overlap_penalty: Penalty for mask overlap with previous tasks
            **kwargs: Additional parameters
        """
        super().__init__(
            model=model,
            sparsity=sparsity,
            temperature=temperature,
            mask_lr=mask_lr,
            regularization_strength=regularization_strength,
            **kwargs
        )
        self.mask_init_strategy = mask_init_strategy
        self.use_supermask = use_supermask
        self.mask_overlap_penalty = mask_overlap_penalty
        
        # For visualization and analysis
        self.mask_sparsity = {}  # task_id -> {param_name -> sparsity}
        self.mask_overlap = {}  # (task_id1, task_id2) -> overlap_ratio
        self.mask_evolution = {}  # task_id -> {step -> {param_name -> sparsity}}
        self.current_step = 0

    def _initialize_mask_parameters(self):
        """
        Initialize mask parameters for the current task.
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

        # Initialize mask parameters for each trainable parameter
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # Skip non-LoRA parameters
            if 'lora_A' not in name and 'lora_B' not in name:
                continue

            # Get combined previous mask if any
            combined_prev_mask = None
            if name in previous_masks and previous_masks[name]:
                # Combine masks from previous tasks (union)
                combined_prev_mask = torch.zeros_like(param.data, dtype=torch.float32)
                for mask in previous_masks[name]:
                    combined_prev_mask = torch.max(combined_prev_mask, mask.to(combined_prev_mask.device))

            # Initialize mask parameters based on the chosen strategy
            if self.mask_init_strategy == "magnitude":
                # Initialize based on parameter magnitudes
                # Higher values for parameters with larger magnitudes
                magnitudes = param.data.abs()
                normalized_magnitudes = magnitudes / (magnitudes.max() + 1e-8)
                
                # Convert to mask parameters (logits)
                # Higher values -> more likely to be selected
                mask_param = torch.log(normalized_magnitudes / (1 - normalized_magnitudes + 1e-8))
                
                # Add noise for exploration
                mask_param = mask_param + torch.randn_like(mask_param) * 0.01
                
            elif self.mask_init_strategy == "gradient":
                # Initialize based on gradient information if available
                if hasattr(param, 'grad') and param.grad is not None:
                    # Higher values for parameters with larger gradients
                    grad_magnitudes = param.grad.abs()
                    normalized_grads = grad_magnitudes / (grad_magnitudes.max() + 1e-8)
                    
                    # Convert to mask parameters (logits)
                    mask_param = torch.log(normalized_grads / (1 - normalized_grads + 1e-8))
                    
                    # Add noise for exploration
                    mask_param = mask_param + torch.randn_like(mask_param) * 0.01
                else:
                    # Fallback to random initialization
                    mask_param = torch.randn_like(param.data)
            else:
                # Default: random initialization
                mask_param = torch.randn_like(param.data)

            # Apply constraints from previous tasks
            if combined_prev_mask is not None:
                # Initialize to avoid overlapping with previous tasks
                # Use negative values for areas already masked by previous tasks
                mask_param[combined_prev_mask > 0.5] = -10.0  # Strongly negative to ensure sigmoid ≈ 0

            # Create trainable mask parameter
            self.mask_parameters[name] = nn.Parameter(mask_param, requires_grad=True)

    def modify_loss(
        self,
        loss: torch.Tensor,
        model: Optional[PeftModel] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Modify the loss function to incorporate sparsity regularization and overlap penalty.

        Args:
            loss: Original loss value
            model: The model (optional, uses self.model if not provided)
            **kwargs: Additional arguments

        Returns:
            Modified loss with sparsity regularization and overlap penalty
        """
        model = model or self.model

        # Compute sparsity regularization
        sparsity_loss = torch.tensor(0.0, device=loss.device)

        for name, mask_param in self.mask_parameters.items():
            # Apply sigmoid with temperature
            mask_prob = torch.sigmoid(mask_param / self.temperature)

            # Compute L1 regularization to encourage target sparsity
            sparsity_loss += torch.abs(mask_prob.mean() - (1.0 - self.sparsity))

        # Compute overlap penalty
        overlap_loss = torch.tensor(0.0, device=loss.device)

        if self.mask_overlap_penalty > 0 and self.task_masks:
            # Get current masks
            current_masks = self._compute_masks()

            # Compute overlap with previous tasks' masks
            for task_id, task_masks in self.task_masks.items():
                for name, mask in current_masks.items():
                    if name in task_masks:
                        # Compute overlap (dot product)
                        overlap = torch.sum(mask * task_masks[name])
                        
                        # Add to overlap loss
                        overlap_loss += overlap

        # Add regularization terms to the loss
        total_loss = loss + (self.regularization_strength * sparsity_loss) + (self.mask_overlap_penalty * overlap_loss)

        # Increment step counter
        self.current_step += 1

        # Store mask evolution data periodically
        if self.current_step % 100 == 0:
            self._store_mask_evolution()

        return total_loss

    def _store_mask_evolution(self):
        """
        Store mask evolution data for visualization.
        """
        # Initialize task entry if needed
        if self.current_task_id not in self.mask_evolution:
            self.mask_evolution[self.current_task_id] = {}

        # Compute current masks
        masks = self._compute_masks()

        # Compute sparsity for each mask
        sparsity_data = {}
        for name, mask in masks.items():
            sparsity = 1.0 - (torch.sum(mask) / mask.numel()).item()
            sparsity_data[name] = sparsity

        # Store sparsity data for current step
        self.mask_evolution[self.current_task_id][self.current_step] = sparsity_data

    def after_training(self, task_id: str, **kwargs):
        """
        Perform post-training operations.

        This method computes and stores the final binary masks for the current task,
        and computes mask overlap with previous tasks.

        Args:
            task_id: Unique identifier for the task
            **kwargs: Additional arguments
        """
        logger.info(f"Computing and storing masks for task {task_id}")

        # Compute final masks
        masks = self._compute_masks()

        # Store masks for the current task
        self.task_masks[task_id] = masks

        # Compute and store mask sparsity
        self.mask_sparsity[task_id] = {}
        for name, mask in masks.items():
            sparsity = 1.0 - (torch.sum(mask) / mask.numel()).item()
            self.mask_sparsity[task_id][name] = sparsity
            logger.info(f"Mask for {name}: sparsity = {sparsity:.4f}")
        
        # Compute mask overlap with previous tasks
        self._compute_mask_overlap(task_id)

    def _compute_mask_overlap(self, task_id):
        """
        Compute overlap between masks for different tasks.
        
        Args:
            task_id: Current task identifier
        """
        # Skip if this is the first task
        if len(self.task_masks) <= 1:
            return
        
        # Compute overlap with previous tasks
        for prev_task_id in self.task_masks:
            # Skip current task
            if prev_task_id == task_id:
                continue
            
            # Get masks for both tasks
            current_masks = self.task_masks[task_id]
            prev_masks = self.task_masks[prev_task_id]
            
            # Compute overlap for each parameter
            overlap_ratios = []
            
            for name in current_masks:
                if name in prev_masks:
                    current_mask = current_masks[name]
                    prev_mask = prev_masks[name]
                    
                    # Compute intersection (parameters active in both tasks)
                    intersection = torch.sum(current_mask * prev_mask).item()
                    
                    # Compute union (parameters active in either task)
                    union = torch.sum(torch.clamp(current_mask + prev_mask, 0, 1)).item()
                    
                    # Compute overlap ratio (Jaccard index)
                    if union > 0:
                        overlap_ratio = intersection / union
                        overlap_ratios.append(overlap_ratio)
            
            # Compute average overlap ratio
            if overlap_ratios:
                avg_overlap = sum(overlap_ratios) / len(overlap_ratios)
                
                # Store overlap
                self.mask_overlap[(task_id, prev_task_id)] = avg_overlap
                self.mask_overlap[(prev_task_id, task_id)] = avg_overlap
                
                logger.info(f"Mask overlap between tasks {task_id} and {prev_task_id}: {avg_overlap:.4f}")

    def get_mask_visualization(self):
        """
        Get visualization data for masks.
        
        Returns:
            Dictionary with visualization data
        """
        # Collect mask data across tasks
        viz_data = {
            "tasks": list(self.mask_sparsity.keys()),
            "parameters": [],
            "sparsity_values": [],
            "overlap_matrix": [],
            "evolution_data": self.mask_evolution
        }
        
        # Get all parameter names
        all_params = set()
        for task_data in self.mask_sparsity.values():
            all_params.update(task_data.keys())
        
        # Collect sparsity data for visualization
        for param_name in all_params:
            viz_data["parameters"].append(param_name)
            
            # Get sparsity values across tasks
            values = [task_data.get(param_name, 0) for task_data in self.mask_sparsity.values()]
            viz_data["sparsity_values"].append(values)
        
        # Create overlap matrix
        tasks = viz_data["tasks"]
        overlap_matrix = np.zeros((len(tasks), len(tasks)))
        
        for i, task1 in enumerate(tasks):
            for j, task2 in enumerate(tasks):
                if i == j:
                    # Diagonal: perfect overlap with self
                    overlap_matrix[i, j] = 1.0
                else:
                    # Off-diagonal: overlap between different tasks
                    overlap_matrix[i, j] = self.mask_overlap.get((task1, task2), 0.0)
        
        viz_data["overlap_matrix"] = overlap_matrix.tolist()
        
        return viz_data
    
    def visualize_masks(self, task_id=None, top_k=5):
        """
        Generate visualizations of masks.
        
        Args:
            task_id: Task identifier (if None, visualize all tasks)
            top_k: Number of top parameters to visualize
            
        Returns:
            Dictionary with visualization images
        """
        visualizations = {}
        
        # Determine tasks to visualize
        if task_id is not None:
            tasks = [task_id]
        else:
            tasks = list(self.task_masks.keys())
        
        # Generate mask heatmaps
        for task in tasks:
            if task not in self.task_masks:
                continue
            
            # Get masks for this task
            masks = self.task_masks[task]
            
            # Sort parameters by size
            param_sizes = {name: mask.numel() for name, mask in masks.items()}
            top_params = sorted(param_sizes.items(), key=lambda x: x[1], reverse=True)[:top_k]
            
            # Generate heatmap for each top parameter
            for name, _ in top_params:
                if name in masks:
                    # Get mask
                    mask = masks[name]
                    
                    # Convert to numpy array
                    mask_np = mask.cpu().numpy()
                    
                    # Reshape for visualization if needed
                    if len(mask_np.shape) > 2:
                        mask_np = mask_np.reshape(mask_np.shape[0], -1)
                    
                    # Generate heatmap
                    plt.figure(figsize=(8, 6))
                    plt.imshow(mask_np, cmap='viridis', interpolation='nearest')
                    plt.colorbar(label='Mask Value')
                    plt.title(f"Task {task}: Mask for {name}")
                    plt.tight_layout()
                    
                    # Save figure to buffer
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    plt.close()
                    
                    # Convert buffer to image
                    buf.seek(0)
                    img = Image.open(buf)
                    
                    # Store visualization
                    viz_key = f"{task}_{name}"
                    visualizations[viz_key] = img
        
        # Generate overlap heatmap
        if len(tasks) > 1:
            # Create overlap matrix
            overlap_matrix = np.zeros((len(tasks), len(tasks)))
            
            for i, task1 in enumerate(tasks):
                for j, task2 in enumerate(tasks):
                    if i == j:
                        # Diagonal: perfect overlap with self
                        overlap_matrix[i, j] = 1.0
                    else:
                        # Off-diagonal: overlap between different tasks
                        overlap_matrix[i, j] = self.mask_overlap.get((task1, task2), 0.0)
            
            # Generate heatmap
            plt.figure(figsize=(8, 6))
            plt.imshow(overlap_matrix, cmap='coolwarm', interpolation='nearest', vmin=0, vmax=1)
            plt.colorbar(label='Overlap Ratio')
            plt.title("Mask Overlap Between Tasks")
            plt.xticks(range(len(tasks)), tasks)
            plt.yticks(range(len(tasks)), tasks)
            plt.tight_layout()
            
            # Save figure to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            
            # Convert buffer to image
            buf.seek(0)
            img = Image.open(buf)
            
            # Store visualization
            visualizations["overlap_heatmap"] = img
        
        return visualizations
    
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
    
    def get_multi_task_model(self, task_ids: List[str]):
        """
        Get a model that can handle multiple tasks.

        Args:
            task_ids: List of task identifiers

        Returns:
            Model with parameters for multiple tasks
        """
        # Validate task IDs
        for task_id in task_ids:
            if task_id not in self.task_masks:
                raise ValueError(f"No masks found for task {task_id}")

        # Create a copy of the model
        multi_task_model = copy.deepcopy(self.model)

        # Create combined masks
        combined_masks = {}
        for name in self.task_masks[task_ids[0]]:
            # Initialize with zeros
            combined_masks[name] = torch.zeros_like(self.task_masks[task_ids[0]][name])
            
            # Combine masks from all specified tasks
            for task_id in task_ids:
                if name in self.task_masks[task_id]:
                    combined_masks[name] = torch.max(combined_masks[name], self.task_masks[task_id][name])

        # Apply combined masks
        for name, param in multi_task_model.named_parameters():
            if name in combined_masks:
                # Apply mask
                param.data *= combined_masks[name]

        return multi_task_model
