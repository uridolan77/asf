"""
Elastic Weight Consolidation (EWC) for CL-PEFT.

This module implements the EWC strategy for mitigating catastrophic forgetting
in sequential fine-tuning of LLMs with PEFT.

Reference:
Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., ... & Hadsell, R. (2017).
Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy of Sciences, 114(13), 3521-3526.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Union, Tuple
import copy

from peft import PeftModel
from transformers import Trainer

from asf.medical.core.logging_config import get_logger
from .base import CLStrategy

logger = get_logger(__name__)

class ElasticWeightConsolidation(CLStrategy):
    """
    Elastic Weight Consolidation (EWC) strategy for CL-PEFT.
    
    EWC mitigates catastrophic forgetting by adding a regularization term to the loss
    that penalizes changes to parameters that were important for previous tasks.
    """
    
    def __init__(
        self,
        model: PeftModel,
        ewc_lambda: float = 5000.0,
        fisher_sample_size: int = 200,
        **kwargs
    ):
        """
        Initialize the EWC strategy.
        
        Args:
            model: The PEFT model to apply EWC to
            ewc_lambda: Regularization strength
            fisher_sample_size: Number of samples to use for Fisher computation
            **kwargs: Additional parameters
        """
        super().__init__(model, **kwargs)
        self.ewc_lambda = ewc_lambda
        self.fisher_sample_size = fisher_sample_size
        
        # Dictionary to store parameter importance (Fisher information)
        # and optimal parameter values for each task
        self.parameter_importance = {}  # task_id -> {param_name -> importance}
        self.optimal_parameters = {}    # task_id -> {param_name -> param_value}
    
    def before_training(self, task_id: str, **kwargs):
        """
        Prepare for training on a new task.
        
        For EWC, this is a no-op as the important work happens after training.
        
        Args:
            task_id: Unique identifier for the task
            **kwargs: Additional arguments
        """
        self.current_task_id = task_id
        logger.info(f"Starting training on task {task_id} with EWC")
    
    def modify_loss(
        self,
        loss: torch.Tensor,
        model: Optional[PeftModel] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Modify the loss function to incorporate EWC regularization.
        
        Args:
            loss: Original loss value
            model: The model (optional, uses self.model if not provided)
            **kwargs: Additional arguments
            
        Returns:
            Modified loss with EWC regularization
        """
        if not self.parameter_importance:
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
            if not any(name in task_importance for task_importance in self.parameter_importance.values()):
                continue
            
            # Compute regularization for each previous task
            for task_id, task_importance in self.parameter_importance.items():
                if name in task_importance and name in self.optimal_parameters[task_id]:
                    # Get importance and optimal value for this parameter
                    importance = task_importance[name]
                    optimal_value = self.optimal_parameters[task_id][name]
                    
                    # Compute squared difference from optimal value
                    ewc_loss += (importance * (param - optimal_value).pow(2)).sum()
        
        # Add regularization term to the loss
        total_loss = loss + (self.ewc_lambda * ewc_loss)
        
        return total_loss
    
    def after_training(self, task_id: str, train_dataset=None, **kwargs):
        """
        Perform post-training operations for EWC.
        
        This computes the Fisher information matrix for the current task
        and stores the optimal parameter values.
        
        Args:
            task_id: Unique identifier for the task
            train_dataset: Training dataset (needed for Fisher computation)
            **kwargs: Additional arguments
        """
        logger.info(f"Computing Fisher information for task {task_id}")
        
        # Store current parameter values as optimal for this task
        self.optimal_parameters[task_id] = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.optimal_parameters[task_id][name] = param.data.clone()
        
        # Compute Fisher information if dataset is provided
        if train_dataset is not None:
            self.parameter_importance[task_id] = self._compute_fisher_information(train_dataset)
        else:
            logger.warning("No dataset provided for Fisher computation, using uniform importance")
            # Use uniform importance if no dataset is provided
            self.parameter_importance[task_id] = {}
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.parameter_importance[task_id][name] = torch.ones_like(param.data)
        
        logger.info(f"Completed EWC setup for task {task_id}")
    
    def _compute_fisher_information(self, dataset) -> Dict[str, torch.Tensor]:
        """
        Compute the Fisher information matrix for the current task.
        
        Args:
            dataset: Dataset to compute Fisher information from
            
        Returns:
            Dictionary mapping parameter names to their importance
        """
        # Create a dataloader with a subset of the dataset
        from torch.utils.data import DataLoader, Subset
        import random
        
        # Sample a subset of the dataset
        sample_size = min(self.fisher_sample_size, len(dataset))
        indices = random.sample(range(len(dataset)), sample_size)
        subset = Subset(dataset, indices)
        
        # Create a dataloader
        dataloader = DataLoader(subset, batch_size=1, shuffle=True)
        
        # Initialize Fisher information
        fisher_information = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher_information[name] = torch.zeros_like(param.data)
        
        # Compute Fisher information
        self.model.eval()
        for batch in dataloader:
            # Zero gradients
            self.model.zero_grad()
            
            # Forward pass
            outputs = self.model(**batch)
            
            # Compute log likelihood
            log_likelihood = F.log_softmax(outputs.logits, dim=-1)
            
            # Sample from the output distribution
            samples = torch.multinomial(torch.exp(log_likelihood), 1).squeeze()
            
            # Compute gradients of log likelihood with respect to parameters
            selected_log_likelihood = log_likelihood.gather(-1, samples.unsqueeze(-1)).squeeze(-1)
            selected_log_likelihood.sum().backward()
            
            # Accumulate squared gradients
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_information[name] += param.grad.data.pow(2)
        
        # Normalize by sample size
        for name in fisher_information:
            fisher_information[name] /= sample_size
        
        return fisher_information
    
    def modify_gradients(self, **kwargs):
        """
        Modify gradients during training.
        
        For EWC, this is a no-op as the regularization is applied in the loss.
        
        Args:
            **kwargs: Additional arguments
        """
        # No gradient modification needed for EWC
        pass
