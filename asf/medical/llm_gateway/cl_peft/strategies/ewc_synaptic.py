"""
Synaptic Intelligence for CL-PEFT.

This module implements the Synaptic Intelligence strategy for mitigating catastrophic forgetting
in sequential fine-tuning of LLMs with PEFT.

Reference:
Zenke, F., Poole, B., & Ganguli, S. (2017).
Continual learning through synaptic intelligence. In International Conference on Machine Learning.
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
from .base import CLStrategy

logger = get_logger(__name__)

class SynapticIntelligence(CLStrategy):
    """
    Synaptic Intelligence strategy for CL-PEFT.
    
    Synaptic Intelligence is similar to EWC but computes parameter importance
    through the path integral of gradients during training, rather than using
    the Fisher information matrix.
    """
    
    def __init__(
        self,
        model: PeftModel,
        si_lambda: float = 1.0,
        xi: float = 0.1,
        normalize_importance: bool = True,
        **kwargs
    ):
        """
        Initialize the Synaptic Intelligence strategy.
        
        Args:
            model: The PEFT model to apply the strategy to
            si_lambda: Regularization strength
            xi: Damping parameter to avoid division by zero
            normalize_importance: Whether to normalize importance values
            **kwargs: Additional parameters
        """
        super().__init__(model, **kwargs)
        self.si_lambda = si_lambda
        self.xi = xi
        self.normalize_importance = normalize_importance
        
        # For Synaptic Intelligence
        self.param_prev = {}            # {param_name -> prev_value}
        self.param_delta = {}           # {param_name -> delta}
        self.omega = {}                 # {param_name -> omega}
        
        # For visualization and analysis
        self.importance_history = {}    # task_id -> {param_name -> importance_score}
        self.forgetting_metrics = {}    # task_id -> forgetting_score
    
    def before_training(self, task_id: str, **kwargs):
        """
        Prepare for training on a new task.
        
        For Synaptic Intelligence, this stores initial parameter values and
        initializes delta accumulators.
        
        Args:
            task_id: Unique identifier for the task
            **kwargs: Additional arguments
        """
        self.current_task_id = task_id
        logger.info(f"Starting training on task {task_id} with Synaptic Intelligence")
        
        # Store initial parameter values
        self.param_prev = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.param_prev[name] = param.data.clone()
                
                # Initialize delta accumulator if not exists
                if name not in self.param_delta:
                    self.param_delta[name] = torch.zeros_like(param.data)
    
    def modify_loss(
        self,
        loss: torch.Tensor,
        model: Optional[PeftModel] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Modify the loss function to incorporate Synaptic Intelligence regularization.
        
        Args:
            loss: Original loss value
            model: The model (optional, uses self.model if not provided)
            **kwargs: Additional arguments
            
        Returns:
            Modified loss with Synaptic Intelligence regularization
        """
        if not self.omega:
            # No previous tasks, no regularization needed
            return loss
        
        model = model or self.model
        
        # Compute regularization loss
        si_loss = torch.tensor(0.0, device=loss.device)
        
        # Only apply regularization to trainable parameters (LoRA parameters)
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Skip parameters that weren't in previous tasks
            if name not in self.omega or name not in self.param_prev:
                continue
            
            # Get omega values and previous parameter values
            omega = self.omega[name]
            prev_value = self.param_prev[name]
            
            # Compute squared difference from previous value
            si_loss += (omega * (param - prev_value).pow(2)).sum()
        
        # Add regularization term to the loss
        total_loss = loss + (self.si_lambda * si_loss)
        
        return total_loss
    
    def after_training(self, task_id: str, **kwargs):
        """
        Perform post-training operations for Synaptic Intelligence.
        
        This updates omega values based on parameter changes and accumulated deltas.
        
        Args:
            task_id: Unique identifier for the task
            **kwargs: Additional arguments
        """
        logger.info(f"Computing parameter importance for task {task_id}")
        
        # Update omega based on parameter changes
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.param_prev:
                # Compute parameter change
                delta = param.data - self.param_prev[name]
                
                # Update omega based on accumulated delta and parameter change
                if name in self.param_delta and torch.any(delta != 0):
                    # Normalize by parameter change to avoid division by zero
                    omega_update = self.param_delta[name] / (delta.pow(2) + self.xi)
                    
                    # Update omega
                    if name in self.omega:
                        self.omega[name] += omega_update
                    else:
                        self.omega[name] = omega_update
                
                # Store current parameters as previous for next task
                self.param_prev[name] = param.data.clone()
                
                # Reset delta accumulator
                self.param_delta[name] = torch.zeros_like(param.data)
        
        # Normalize omega values if requested
        if self.normalize_importance and self.omega:
            # Find maximum omega value across all parameters
            max_omega = 0.0
            for name, omega in self.omega.items():
                max_val = omega.max().item()
                if max_val > max_omega:
                    max_omega = max_val
            
            # Normalize all omega values
            if max_omega > 0:
                for name in self.omega:
                    self.omega[name] /= max_omega
        
        # Store importance metrics for visualization
        self._store_importance_metrics(task_id)
        
        logger.info(f"Completed Synaptic Intelligence setup for task {task_id}")
    
    def modify_gradients(self, **kwargs):
        """
        Modify gradients during training.
        
        For Synaptic Intelligence, this accumulates parameter-gradient products
        to estimate parameter importance.
        
        Args:
            **kwargs: Additional arguments
        """
        # Accumulate parameter-gradient products
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # Accumulate parameter-gradient product
                if name in self.param_delta:
                    self.param_delta[name] -= param.grad.data * param.data.detach()
    
    def _store_importance_metrics(self, task_id):
        """
        Store importance metrics for visualization and analysis.
        
        Args:
            task_id: Unique identifier for the task
        """
        # Initialize importance history for this task
        self.importance_history[task_id] = {}
        
        # Compute and store importance metrics from omega
        for name, omega in self.omega.items():
            # Compute a scalar importance score (mean of absolute values)
            importance_score = omega.abs().mean().item()
            self.importance_history[task_id][name] = importance_score
    
    def compute_forgetting(self, task_id, eval_dataset, eval_fn=None):
        """
        Compute forgetting for a specific task.
        
        Args:
            task_id: Task identifier
            eval_dataset: Evaluation dataset for the task
            eval_fn: Function to evaluate the model (optional)
            
        Returns:
            Forgetting metric
        """
        # If no evaluation function provided, use a default one
        if eval_fn is None:
            def default_eval_fn(model, dataset):
                # Create a dataloader
                dataloader = DataLoader(dataset, batch_size=8)
                
                # Evaluate
                model.eval()
                total_loss = 0
                with torch.no_grad():
                    for batch in dataloader:
                        outputs = model(**batch)
                        total_loss += outputs.loss.item()
                
                return total_loss / len(dataloader)
            
            eval_fn = default_eval_fn
        
        # Compute current performance
        current_perf = eval_fn(self.model, eval_dataset)
        
        # Store forgetting metric
        self.forgetting_metrics[task_id] = current_perf
        
        return current_perf
    
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
            "tasks": list(self.importance_history.keys()),
            "parameters": [],
            "importance_values": []
        }
        
        # Get all parameter names
        all_params = set()
        for task_data in self.importance_history.values():
            all_params.update(task_data.keys())
        
        # Compute average importance across tasks for each parameter
        avg_importance = {}
        for param_name in all_params:
            values = [task_data.get(param_name, 0) for task_data in self.importance_history.values()]
            avg_importance[param_name] = sum(values) / len(values) if values else 0
        
        # Get top-k parameters by average importance
        top_params = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Collect data for visualization
        for param_name, _ in top_params:
            viz_data["parameters"].append(param_name)
            
            # Get importance values across tasks
            values = [task_data.get(param_name, 0) for task_data in self.importance_history.values()]
            viz_data["importance_values"].append(values)
        
        return viz_data
