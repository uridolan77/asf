"""
Online Elastic Weight Consolidation (EWC) for CL-PEFT.

This module implements the Online EWC strategy for mitigating catastrophic forgetting
in sequential fine-tuning of LLMs with PEFT.

Reference:
Schwarz, J., Czarnecki, W., Luketina, J., Grabska-Barwinska, A., Teh, Y. W., Pascanu, R., & Hadsell, R. (2018).
Progress & compress: A scalable framework for continual learning. In International Conference on Machine Learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Union, Tuple
import copy
from tqdm import tqdm

from peft import PeftModel
from transformers import Trainer
from torch.utils.data import DataLoader, Subset

from asf.medical.core.logging_config import get_logger
from .ewc_base import BaseEWC

logger = get_logger(__name__)

class OnlineEWC(BaseEWC):
    """
    Online Elastic Weight Consolidation (EWC) strategy for CL-PEFT.
    
    Online EWC is a memory-efficient version of EWC that maintains a single
    Fisher information matrix that is updated with a decay factor after each task.
    """
    
    def __init__(
        self,
        model: PeftModel,
        ewc_lambda: float = 5000.0,
        fisher_sample_size: int = 200,
        fisher_estimation_method: str = "empirical",
        normalize_fisher: bool = True,
        online_gamma: float = 0.95,
        **kwargs
    ):
        """
        Initialize the Online EWC strategy.
        
        Args:
            model: The PEFT model to apply EWC to
            ewc_lambda: Regularization strength
            fisher_sample_size: Number of samples to use for Fisher computation
            fisher_estimation_method: Method to estimate Fisher information
                ("empirical", "exact", or "diagonal")
            normalize_fisher: Whether to normalize Fisher matrices
            online_gamma: Decay factor for Online EWC (0 to 1)
            **kwargs: Additional parameters
        """
        super().__init__(
            model=model,
            ewc_lambda=ewc_lambda,
            fisher_sample_size=fisher_sample_size,
            fisher_estimation_method=fisher_estimation_method,
            normalize_fisher=normalize_fisher,
            **kwargs
        )
        self.online_gamma = online_gamma
        
        # For Online EWC
        self.online_fisher = {}         # {param_name -> importance}
        self.online_optimal = {}        # {param_name -> param_value}
    
    def before_training(self, task_id: str, **kwargs):
        """
        Prepare for training on a new task.
        
        For Online EWC, this is a no-op as the important work happens after training.
        
        Args:
            task_id: Unique identifier for the task
            **kwargs: Additional arguments
        """
        self.current_task_id = task_id
        logger.info(f"Starting training on task {task_id} with Online EWC (gamma={self.online_gamma})")
    
    def modify_loss(
        self,
        loss: torch.Tensor,
        model: Optional[PeftModel] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Modify the loss function to incorporate Online EWC regularization.
        
        Args:
            loss: Original loss value
            model: The model (optional, uses self.model if not provided)
            **kwargs: Additional arguments
            
        Returns:
            Modified loss with Online EWC regularization
        """
        if not self.online_fisher:
            # No previous tasks, no regularization needed
            return loss
        
        model = model or self.model
        
        # Compute EWC regularization loss
        ewc_loss = torch.tensor(0.0, device=loss.device)
        
        # Only apply EWC to trainable parameters (LoRA parameters)
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Skip parameters that weren't in previous tasks
            if name not in self.online_fisher or name not in self.online_optimal:
                continue
            
            # Get importance and optimal value for this parameter
            importance = self.online_fisher[name]
            optimal_value = self.online_optimal[name]
            
            # Compute squared difference from optimal value
            ewc_loss += (importance * (param - optimal_value).pow(2)).sum()
        
        # Add regularization term to the loss
        total_loss = loss + (self.ewc_lambda * ewc_loss)
        
        return total_loss
    
    def after_training(self, task_id: str, train_dataset=None, **kwargs):
        """
        Perform post-training operations for Online EWC.
        
        This computes the Fisher information matrix for the current task
        and updates the online Fisher matrix with a decay factor.
        
        Args:
            task_id: Unique identifier for the task
            train_dataset: Training dataset (needed for Fisher computation)
            **kwargs: Additional arguments
        """
        logger.info(f"Computing Fisher information for task {task_id}")
        
        # Store current parameter values as optimal
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.online_optimal[name] = param.data.clone()
        
        # Compute Fisher information if dataset is provided
        if train_dataset is not None:
            new_fisher = self._compute_fisher_information(train_dataset)
            
            if not self.online_fisher:
                # First task, initialize online Fisher
                self.online_fisher = new_fisher
            else:
                # Update online Fisher with decay
                for name in new_fisher:
                    if name in self.online_fisher:
                        self.online_fisher[name] = self.online_gamma * self.online_fisher[name] + \
                                                  (1 - self.online_gamma) * new_fisher[name]
                    else:
                        self.online_fisher[name] = new_fisher[name]
        else:
            logger.warning("No dataset provided for Fisher computation, using uniform importance")
            # Use uniform importance if no dataset is provided
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if name not in self.online_fisher:
                        self.online_fisher[name] = torch.ones_like(param.data)
        
        # Store importance metrics for visualization
        self._store_importance_metrics(task_id)
        
        logger.info(f"Completed Online EWC setup for task {task_id}")
    
    def _store_importance_metrics(self, task_id):
        """
        Store importance metrics for visualization and analysis.
        
        Args:
            task_id: Unique identifier for the task
        """
        # Initialize importance history for this task
        self.importance_history[task_id] = {}
        
        # Compute and store importance metrics from online Fisher
        for name, importance in self.online_fisher.items():
            # Compute a scalar importance score (mean of absolute values)
            importance_score = importance.abs().mean().item()
            self.importance_history[task_id][name] = importance_score
