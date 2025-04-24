"""
AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning.

This module implements AdaLoRA, which adaptively allocates the parameter budget
across weight matrices based on their importance to the task.

Reference:
Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., & Chen, W. (2022).
LoRA: Low-Rank Adaptation of Large Language Models.

Zhang, H., Li, Y., Jiang, P., Zhang, P., Hu, E. J., & Xie, C. (2023).
AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Union, Tuple
import math
import copy
import numpy as np

from peft import PeftModel, LoraConfig, get_peft_model
from transformers import PreTrainedModel

from asf.medical.core.logging_config import get_logger

logger = get_logger(__name__)

class AdaLoraConfig(LoraConfig):
    """
    Configuration class for AdaLoRA.
    
    AdaLoRA extends LoRA by adaptively allocating the parameter budget
    across weight matrices based on their importance to the task.
    """
    
    def __init__(
        self,
        target_modules: Optional[Union[List[str], str]] = None,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        bias: str = "none",
        task_type: str = "CAUSAL_LM",
        target_r: Optional[int] = None,
        init_r: Optional[int] = None,
        tinit: int = 0,
        tfinal: int = 0,
        deltaT: int = 1,
        beta1: float = 0.85,
        beta2: float = 0.85,
        orth_reg_weight: float = 0.5,
        total_step: Optional[int] = None,
        rank_pattern: Optional[Dict[str, int]] = None,
        **kwargs
    ):
        """
        Initialize AdaLoraConfig.
        
        Args:
            target_modules: List of module names to apply AdaLoRA to
            r: Initial rank for all target modules
            lora_alpha: LoRA alpha parameter
            lora_dropout: Dropout probability for LoRA layers
            bias: Bias type ("none", "all", "lora_only")
            task_type: Task type
            target_r: Target rank after adaptation
            init_r: Initial rank (if different from r)
            tinit: Initial step for rank adaptation
            tfinal: Final step for rank adaptation
            deltaT: Step interval for rank adaptation
            beta1: Exponential moving average factor for importance
            beta2: Exponential moving average factor for update
            orth_reg_weight: Weight for orthogonal regularization
            total_step: Total number of training steps
            rank_pattern: Dictionary mapping module names to their ranks
            **kwargs: Additional arguments for LoraConfig
        """
        super().__init__(
            target_modules=target_modules,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            task_type=task_type,
            **kwargs
        )
        
        self.target_r = target_r if target_r is not None else r // 2
        self.init_r = init_r if init_r is not None else r
        self.tinit = tinit
        self.tfinal = tfinal if tfinal > 0 else (total_step if total_step is not None else 10000)
        self.deltaT = deltaT
        self.beta1 = beta1
        self.beta2 = beta2
        self.orth_reg_weight = orth_reg_weight
        self.total_step = total_step
        self.rank_pattern = rank_pattern or {}
        
        # AdaLoRA specific attributes
        self.peft_type = "ADALORA"

class AdaLoraModel(PeftModel):
    """
    AdaLoRA model for adaptive budget allocation in parameter-efficient fine-tuning.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        config: AdaLoraConfig,
        adapter_name: str = "default"
    ):
        """
        Initialize AdaLoraModel.
        
        Args:
            model: Base model to apply AdaLoRA to
            config: AdaLoRA configuration
            adapter_name: Name of the adapter
        """
        super().__init__(model, config, adapter_name)
        
        # Initialize AdaLoRA-specific attributes
        self.step = 0
        self.target_modules = config.target_modules
        
        # Initialize importance and update metrics
        self.importance_scores = {}
        self.update_metrics = {}
        
        # Initialize SVD components for each module
        self.svd_components = {}
        
        # Initialize rank pattern
        self.rank_pattern = config.rank_pattern.copy() if config.rank_pattern else {}
        
        # Register hooks for importance estimation
        self._register_hooks()
    
    def _register_hooks(self):
        """
        Register hooks for importance estimation.
        """
        for name, module in self.model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                # Initialize importance and update metrics
                self.importance_scores[name] = torch.zeros_like(module.lora_A.weight)
                self.update_metrics[name] = torch.zeros_like(module.lora_A.weight)
                
                # Register backward hook
                module.register_full_backward_hook(self._backward_hook(name))
    
    def _backward_hook(self, name):
        """
        Create a backward hook for a module.
        
        Args:
            name: Name of the module
            
        Returns:
            Backward hook function
        """
        def hook(module, grad_input, grad_output):
            # Skip if not in training mode
            if not self.training:
                return
            
            # Update importance scores
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                # Compute importance based on gradients
                with torch.no_grad():
                    # Get gradients
                    grad_A = module.lora_A.weight.grad
                    grad_B = module.lora_B.weight.grad
                    
                    if grad_A is not None and grad_B is not None:
                        # Compute importance as product of gradient magnitudes
                        importance = torch.abs(grad_A).mean(dim=0) * torch.abs(grad_B).mean(dim=1)
                        
                        # Update importance scores with exponential moving average
                        self.importance_scores[name] = (
                            self.config.beta1 * self.importance_scores[name] +
                            (1 - self.config.beta1) * importance
                        )
                        
                        # Compute update metric
                        update = torch.abs(grad_A @ grad_B)
                        
                        # Update update metrics with exponential moving average
                        self.update_metrics[name] = (
                            self.config.beta2 * self.update_metrics[name] +
                            (1 - self.config.beta2) * update
                        )
        
        return hook
    
    def forward(self, *args, **kwargs):
        """
        Forward pass with AdaLoRA.
        
        Args:
            *args: Positional arguments for the model
            **kwargs: Keyword arguments for the model
            
        Returns:
            Model outputs
        """
        # Increment step counter
        self.step += 1
        
        # Check if it's time to adapt ranks
        if (self.step >= self.config.tinit and 
            self.step <= self.config.tfinal and 
            self.step % self.config.deltaT == 0):
            self._adapt_ranks()
        
        # Call parent's forward method
        return super().forward(*args, **kwargs)
    
    def _adapt_ranks(self):
        """
        Adapt ranks based on importance scores and update metrics.
        """
        logger.info(f"Adapting ranks at step {self.step}")
        
        # Collect all importance scores and update metrics
        all_importances = []
        all_updates = []
        module_map = {}
        
        for name, module in self.model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                if name in self.importance_scores and name in self.update_metrics:
                    # Compute combined metric
                    metric = self.importance_scores[name] / (self.update_metrics[name] + 1e-8)
                    
                    # Flatten and add to list
                    flat_metric = metric.flatten()
                    all_importances.extend(flat_metric.tolist())
                    
                    # Map indices to module and position
                    for i in range(len(flat_metric)):
                        module_map[len(all_importances) - i - 1] = (name, i)
                    
                    # Add update metric
                    all_updates.extend(self.update_metrics[name].flatten().tolist())
        
        # Sort indices by importance
        sorted_indices = sorted(range(len(all_importances)), key=lambda i: all_importances[i], reverse=True)
        
        # Compute target budget
        total_params = len(all_importances)
        target_params = int(total_params * (self.config.target_r / self.config.init_r))
        
        # Select top parameters
        selected_indices = sorted_indices[:target_params]
        
        # Create mask for each module
        masks = {}
        for idx in selected_indices:
            name, pos = module_map[idx]
            if name not in masks:
                masks[name] = torch.zeros_like(self.importance_scores[name])
            
            # Set mask value
            flat_mask = masks[name].flatten()
            flat_mask[pos] = 1.0
            masks[name] = flat_mask.reshape(masks[name].shape)
        
        # Apply masks to modules
        for name, module in self.model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                if name in masks:
                    # Apply SVD to compress the module
                    self._apply_svd_to_module(name, module, masks[name])
    
    def _apply_svd_to_module(self, name, module, mask):
        """
        Apply SVD to compress a module based on the mask.
        
        Args:
            name: Name of the module
            module: Module to compress
            mask: Mask indicating which parameters to keep
        """
        # Get current weights
        A = module.lora_A.weight.data
        B = module.lora_B.weight.data
        
        # Compute effective weight
        W = A @ B
        
        # Apply mask
        W_masked = W * mask
        
        # Compute SVD
        try:
            U, S, V = torch.svd(W_masked)
            
            # Determine new rank based on non-zero singular values
            new_rank = (S > 1e-6).sum().item()
            new_rank = max(1, min(new_rank, self.config.init_r))
            
            # Update rank pattern
            self.rank_pattern[name] = new_rank
            
            # Truncate to new rank
            U = U[:, :new_rank]
            S = S[:new_rank]
            V = V[:, :new_rank]
            
            # Compute new A and B
            new_A = U @ torch.diag(torch.sqrt(S))
            new_B = torch.diag(torch.sqrt(S)) @ V.T
            
            # Apply scaling
            scaling = self.config.lora_alpha / new_rank
            new_A = new_A * math.sqrt(scaling)
            new_B = new_B * math.sqrt(scaling)
            
            # Update module weights
            if new_A.shape[1] != A.shape[1] or new_B.shape[0] != B.shape[0]:
                # Need to resize the module
                logger.info(f"Resizing module {name} from rank {A.shape[1]} to {new_rank}")
                
                # Create new linear layers
                new_lora_A = nn.Linear(A.shape[1], new_rank, bias=False)
                new_lora_B = nn.Linear(new_rank, B.shape[1], bias=False)
                
                # Initialize with new weights
                new_lora_A.weight.data = new_A
                new_lora_B.weight.data = new_B
                
                # Replace old layers
                module.lora_A = new_lora_A
                module.lora_B = new_lora_B
            else:
                # Just update weights
                module.lora_A.weight.data = new_A
                module.lora_B.weight.data = new_B
            
            # Store SVD components
            self.svd_components[name] = (U, S, V)
            
            logger.info(f"Adapted rank for {name}: {new_rank}")
        
        except Exception as e:
            logger.warning(f"SVD failed for {name}: {str(e)}")
    
    def compute_orthogonal_regularization(self):
        """
        Compute orthogonal regularization loss.
        
        Returns:
            Orthogonal regularization loss
        """
        orth_reg_loss = torch.tensor(0.0, device=self.device)
        
        for name, module in self.model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                # Compute orthogonality of A
                A = module.lora_A.weight
                if A.shape[1] > 1:  # Only if rank > 1
                    A_normalized = F.normalize(A, dim=0)
                    A_gram = A_normalized.T @ A_normalized
                    A_orth_loss = torch.norm(A_gram - torch.eye(A_gram.shape[0], device=A_gram.device))
                    orth_reg_loss += A_orth_loss
                
                # Compute orthogonality of B
                B = module.lora_B.weight
                if B.shape[0] > 1:  # Only if rank > 1
                    B_normalized = F.normalize(B, dim=1)
                    B_gram = B_normalized @ B_normalized.T
                    B_orth_loss = torch.norm(B_gram - torch.eye(B_gram.shape[0], device=B_gram.device))
                    orth_reg_loss += B_orth_loss
        
        return self.config.orth_reg_weight * orth_reg_loss
    
    def get_rank_pattern(self):
        """
        Get the current rank pattern.
        
        Returns:
            Dictionary mapping module names to their ranks
        """
        return self.rank_pattern
    
    def get_total_parameters(self):
        """
        Get the total number of parameters in the model.
        
        Returns:
            Total number of parameters
        """
        total_params = 0
        
        for name, module in self.model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                # Count parameters in A and B
                total_params += module.lora_A.weight.numel()
                total_params += module.lora_B.weight.numel()
        
        return total_params
    
    def get_compression_statistics(self):
        """
        Get compression statistics.
        
        Returns:
            Dictionary with compression statistics
        """
        stats = {
            "total_parameters": self.get_total_parameters(),
            "rank_pattern": self.get_rank_pattern(),
            "compression_ratio": {},
            "average_rank": 0.0
        }
        
        # Compute compression ratio for each module
        total_rank = 0
        num_modules = 0
        
        for name, module in self.model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                # Get original and current rank
                original_rank = self.config.init_r
                current_rank = module.lora_A.weight.shape[1]
                
                # Compute compression ratio
                compression_ratio = original_rank / current_rank if current_rank > 0 else float('inf')
                stats["compression_ratio"][name] = compression_ratio
                
                # Update average rank
                total_rank += current_rank
                num_modules += 1
        
        # Compute average rank
        if num_modules > 0:
            stats["average_rank"] = total_rank / num_modules
        
        return stats


def get_adalora_model(
    model: PreTrainedModel,
    target_modules: Optional[Union[List[str], str]] = None,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_r: Optional[int] = None,
    total_step: Optional[int] = None,
    **kwargs
) -> AdaLoraModel:
    """
    Get a model with AdaLoRA applied.
    
    Args:
        model: Base model to apply AdaLoRA to
        target_modules: List of module names to apply AdaLoRA to
        r: Initial rank for all target modules
        lora_alpha: LoRA alpha parameter
        lora_dropout: Dropout probability for LoRA layers
        target_r: Target rank after adaptation
        total_step: Total number of training steps
        **kwargs: Additional arguments for AdaLoraConfig
        
    Returns:
        Model with AdaLoRA applied
    """
    # Create AdaLoRA config
    config = AdaLoraConfig(
        target_modules=target_modules,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_r=target_r,
        total_step=total_step,
        **kwargs
    )
    
    # Create AdaLoRA model
    adalora_model = AdaLoraModel(model, config)
    
    return adalora_model
