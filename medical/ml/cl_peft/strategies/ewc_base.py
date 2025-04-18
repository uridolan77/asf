"""
Base Elastic Weight Consolidation (EWC) for CL-PEFT.

This module implements the standard EWC strategy for mitigating catastrophic forgetting
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
from tqdm import tqdm

from peft import PeftModel
from transformers import Trainer
from torch.utils.data import DataLoader, Subset

from asf.medical.core.logging_config import get_logger
from .base import CLStrategy

logger = get_logger(__name__)

class BaseEWC(CLStrategy):
    """
    Base Elastic Weight Consolidation (EWC) strategy for CL-PEFT.
    
    EWC mitigates catastrophic forgetting by adding a regularization term to the loss
    that penalizes changes to parameters that were important for previous tasks.
    """
    
    def __init__(
        self,
        model: PeftModel,
        ewc_lambda: float = 5000.0,
        fisher_sample_size: int = 200,
        fisher_estimation_method: str = "empirical",
        normalize_fisher: bool = True,
        **kwargs
    ):
        """
        Initialize the EWC strategy.
        
        Args:
            model: The PEFT model to apply EWC to
            ewc_lambda: Regularization strength
            fisher_sample_size: Number of samples to use for Fisher computation
            fisher_estimation_method: Method to estimate Fisher information
                ("empirical", "exact", or "diagonal")
            normalize_fisher: Whether to normalize Fisher matrices
            **kwargs: Additional parameters
        """
        super().__init__(model, **kwargs)
        self.ewc_lambda = ewc_lambda
        self.fisher_sample_size = fisher_sample_size
        self.fisher_estimation_method = fisher_estimation_method
        self.normalize_fisher = normalize_fisher
        
        # Dictionary to store parameter importance (Fisher information)
        # and optimal parameter values for each task
        self.parameter_importance = {}  # task_id -> {param_name -> importance}
        self.optimal_parameters = {}    # task_id -> {param_name -> param_value}
        
        # For visualization and analysis
        self.importance_history = {}    # task_id -> {param_name -> importance_score}
        self.forgetting_metrics = {}    # task_id -> forgetting_score
    
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
        
        # Store importance metrics for visualization
        self._store_importance_metrics(task_id)
        
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
        
        # Choose Fisher estimation method
        if self.fisher_estimation_method == "empirical":
            fisher_information = self._compute_empirical_fisher(dataloader, fisher_information)
        elif self.fisher_estimation_method == "exact":
            fisher_information = self._compute_exact_fisher(dataloader, fisher_information)
        elif self.fisher_estimation_method == "diagonal":
            fisher_information = self._compute_diagonal_fisher(dataloader, fisher_information)
        else:
            logger.warning(f"Unknown Fisher estimation method: {self.fisher_estimation_method}, using empirical")
            fisher_information = self._compute_empirical_fisher(dataloader, fisher_information)
        
        # Normalize Fisher information if requested
        if self.normalize_fisher:
            # Find maximum Fisher value across all parameters
            max_fisher = 0.0
            for name, fisher in fisher_information.items():
                max_val = fisher.max().item()
                if max_val > max_fisher:
                    max_fisher = max_val
            
            # Normalize all Fisher values
            if max_fisher > 0:
                for name in fisher_information:
                    fisher_information[name] /= max_fisher
        
        return fisher_information
    
    def _compute_empirical_fisher(self, dataloader, fisher_information):
        """
        Compute empirical Fisher information using gradients from model outputs.
        
        Args:
            dataloader: DataLoader for the dataset
            fisher_information: Dictionary to store Fisher information
            
        Returns:
            Updated Fisher information dictionary
        """
        # Process each batch
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing Fisher")):
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
                    fisher_information[name] += param.grad.data.pow(2).detach()
        
        # Normalize by sample size
        sample_size = len(dataloader)
        for name in fisher_information:
            fisher_information[name] /= sample_size
        
        return fisher_information
    
    def _compute_exact_fisher(self, dataloader, fisher_information):
        """
        Compute exact Fisher information using Hessian-vector products.
        
        Args:
            dataloader: DataLoader for the dataset
            fisher_information: Dictionary to store Fisher information
            
        Returns:
            Updated Fisher information dictionary
        """
        # This is a simplified implementation of exact Fisher
        # A full implementation would use Hessian-vector products
        
        # Process each batch
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing Exact Fisher")):
            # Zero gradients
            self.model.zero_grad()
            
            # Forward pass
            outputs = self.model(**batch)
            
            # Compute log likelihood
            log_probs = F.log_softmax(outputs.logits, dim=-1)
            
            # Compute Fisher for each output dimension
            for i in range(log_probs.size(-1)):
                # Zero gradients
                self.model.zero_grad()
                
                # Backward for this output dimension
                log_probs[:, i].sum().backward(retain_graph=(i < log_probs.size(-1) - 1))
                
                # Accumulate squared gradients weighted by probability
                probs = torch.exp(log_probs.detach())
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        fisher_information[name] += probs[:, i].mean() * param.grad.data.pow(2)
        
        # Normalize by sample size
        sample_size = len(dataloader)
        for name in fisher_information:
            fisher_information[name] /= sample_size
        
        return fisher_information
    
    def _compute_diagonal_fisher(self, dataloader, fisher_information):
        """
        Compute diagonal Fisher information using the expected Hessian.
        
        Args:
            dataloader: DataLoader for the dataset
            fisher_information: Dictionary to store Fisher information
            
        Returns:
            Updated Fisher information dictionary
        """
        # Process each batch
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing Diagonal Fisher")):
            # Zero gradients
            self.model.zero_grad()
            
            # Forward pass
            outputs = self.model(**batch)
            
            # Compute loss
            if hasattr(outputs, "loss"):
                loss = outputs.loss
            else:
                # If no loss in outputs, use a simple loss function
                logits = outputs.logits
                labels = batch.get("labels", torch.argmax(logits, dim=-1))
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            # Backward pass
            loss.backward()
            
            # Accumulate squared gradients
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_information[name] += param.grad.data.pow(2).detach()
        
        # Normalize by sample size
        sample_size = len(dataloader)
        for name in fisher_information:
            fisher_information[name] /= sample_size
        
        return fisher_information
    
    def _store_importance_metrics(self, task_id):
        """
        Store importance metrics for visualization and analysis.
        
        Args:
            task_id: Unique identifier for the task
        """
        # Initialize importance history for this task
        self.importance_history[task_id] = {}
        
        # Compute and store importance metrics
        if task_id in self.parameter_importance:
            for name, importance in self.parameter_importance[task_id].items():
                # Compute a scalar importance score (mean of absolute values)
                importance_score = importance.abs().mean().item()
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
    
    def modify_gradients(self, **kwargs):
        """
        Modify gradients during training.
        
        For EWC, this is a no-op as the regularization is applied in the loss.
        
        Args:
            **kwargs: Additional arguments
        """
        # No gradient modification needed for EWC
        pass
