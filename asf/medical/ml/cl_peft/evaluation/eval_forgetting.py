"""
Forgetting metrics for CL-PEFT.

This module provides metrics for measuring catastrophic forgetting in
continual learning with parameter-efficient fine-tuning.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
import copy
import matplotlib.pyplot as plt
import io
from PIL import Image

from peft import PeftModel
from transformers import Trainer, TrainingArguments, PreTrainedModel

from asf.medical.core.logging_config import get_logger

logger = get_logger(__name__)

class ForgettingMetrics:
    """
    Class for measuring catastrophic forgetting in continual learning.
    """
    
    def __init__(
        self,
        model: PeftModel,
        task_datasets: Dict[str, Dict[str, Any]],
        metric_fn: Callable,
        device: Optional[torch.device] = None,
        compute_backward_transfer: bool = True,
        compute_forward_transfer: bool = False,
        compute_forgetting_curve: bool = True,
        **kwargs
    ):
        """
        Initialize ForgettingMetrics.
        
        Args:
            model: The model to evaluate
            task_datasets: Dictionary mapping task IDs to dictionaries with 'train' and 'eval' datasets
            metric_fn: Function to compute metrics (should return a dictionary)
            device: Device to use for evaluation
            compute_backward_transfer: Whether to compute backward transfer
            compute_forward_transfer: Whether to compute forward transfer
            compute_forgetting_curve: Whether to compute forgetting curve
            **kwargs: Additional arguments
        """
        self.model = model
        self.task_datasets = task_datasets
        self.metric_fn = metric_fn
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_backward_transfer = compute_backward_transfer
        self.compute_forward_transfer = compute_forward_transfer
        self.compute_forgetting_curve = compute_forgetting_curve
        
        # Store task order
        self.task_order = list(task_datasets.keys())
        
        # Store metrics for each task
        self.task_metrics = {}
        
        # Store metrics after each task
        self.metrics_after_task = {}
        
        # Store forgetting curve
        self.forgetting_curve = {}
        
        # Store model snapshots if needed
        self.model_snapshots = {}
        
        # Store initial performance if computing forward transfer
        if self.compute_forward_transfer:
            self._compute_initial_performance()
    
    def _compute_initial_performance(self):
        """
        Compute initial performance on all tasks before training.
        """
        logger.info("Computing initial performance for forward transfer")
        
        # Store initial model
        initial_model = copy.deepcopy(self.model)
        
        # Evaluate on all tasks
        self.initial_performance = {}
        
        for task_id in self.task_order:
            # Get evaluation dataset
            eval_dataset = self.task_datasets[task_id].get("eval")
            
            if eval_dataset is None:
                logger.warning(f"No evaluation dataset for task {task_id}, skipping initial performance")
                continue
            
            # Evaluate on task
            metrics = self._evaluate_model(initial_model, eval_dataset)
            
            # Store metrics
            self.initial_performance[task_id] = metrics
            
            logger.info(f"Initial performance on task {task_id}: {metrics}")
    
    def _evaluate_model(self, model, dataset):
        """
        Evaluate a model on a dataset.
        
        Args:
            model: Model to evaluate
            dataset: Dataset to evaluate on
            
        Returns:
            Dictionary with metrics
        """
        # Create trainer for evaluation
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir="./eval_output",
                per_device_eval_batch_size=8,
                remove_unused_columns=False
            ),
            eval_dataset=dataset
        )
        
        # Evaluate
        metrics = trainer.evaluate()
        
        # Apply custom metric function if provided
        if self.metric_fn is not None:
            custom_metrics = self.metric_fn(model, dataset)
            metrics.update(custom_metrics)
        
        return metrics
    
    def after_task(self, task_id: str):
        """
        Compute metrics after training on a task.
        
        Args:
            task_id: ID of the task that was just trained
        """
        logger.info(f"Computing metrics after task {task_id}")
        
        # Store model snapshot if needed
        if self.compute_backward_transfer or self.compute_forgetting_curve:
            self.model_snapshots[task_id] = copy.deepcopy(self.model)
        
        # Evaluate on all previous tasks
        task_metrics = {}
        
        for prev_task_id in self.task_order:
            # Skip tasks that haven't been seen yet
            if self.task_order.index(prev_task_id) > self.task_order.index(task_id):
                continue
            
            # Get evaluation dataset
            eval_dataset = self.task_datasets[prev_task_id].get("eval")
            
            if eval_dataset is None:
                logger.warning(f"No evaluation dataset for task {prev_task_id}, skipping")
                continue
            
            # Evaluate on task
            metrics = self._evaluate_model(self.model, eval_dataset)
            
            # Store metrics
            task_metrics[prev_task_id] = metrics
            
            logger.info(f"Performance on task {prev_task_id} after training on task {task_id}: {metrics}")
        
        # Store metrics
        self.metrics_after_task[task_id] = task_metrics
    
    def compute_forgetting(self):
        """
        Compute forgetting metrics.
        
        Returns:
            Dictionary with forgetting metrics
        """
        logger.info("Computing forgetting metrics")
        
        # Initialize metrics
        forgetting = {}
        backward_transfer = {}
        forward_transfer = {}
        
        # Compute forgetting and backward transfer
        for i, task_id in enumerate(self.task_order):
            # Skip first task for backward transfer
            if i > 0 and self.compute_backward_transfer:
                # Compute backward transfer for all previous tasks
                for j in range(i):
                    prev_task_id = self.task_order[j]
                    
                    # Get metrics after training on previous task and current task
                    prev_metrics = self.metrics_after_task[prev_task_id].get(prev_task_id)
                    curr_metrics = self.metrics_after_task[task_id].get(prev_task_id)
                    
                    if prev_metrics is None or curr_metrics is None:
                        continue
                    
                    # Compute backward transfer
                    bt = {}
                    for metric_name in prev_metrics:
                        if metric_name in curr_metrics:
                            # Compute difference (positive is improvement, negative is forgetting)
                            bt[metric_name] = curr_metrics[metric_name] - prev_metrics[metric_name]
                    
                    # Store backward transfer
                    if prev_task_id not in backward_transfer:
                        backward_transfer[prev_task_id] = {}
                    
                    backward_transfer[prev_task_id][task_id] = bt
            
            # Compute forward transfer if enabled
            if self.compute_forward_transfer and i < len(self.task_order) - 1:
                # Compute forward transfer for all future tasks
                for j in range(i + 1, len(self.task_order)):
                    next_task_id = self.task_order[j]
                    
                    # Get initial metrics and metrics after training on current task
                    initial_metrics = self.initial_performance.get(next_task_id)
                    curr_metrics = self.metrics_after_task[task_id].get(next_task_id)
                    
                    if initial_metrics is None or curr_metrics is None:
                        continue
                    
                    # Compute forward transfer
                    ft = {}
                    for metric_name in initial_metrics:
                        if metric_name in curr_metrics:
                            # Compute difference (positive is improvement)
                            ft[metric_name] = curr_metrics[metric_name] - initial_metrics[metric_name]
                    
                    # Store forward transfer
                    if next_task_id not in forward_transfer:
                        forward_transfer[next_task_id] = {}
                    
                    forward_transfer[next_task_id][task_id] = ft
        
        # Compute forgetting
        for i, task_id in enumerate(self.task_order[:-1]):  # Skip last task
            # Get best performance on this task
            best_metrics = self.metrics_after_task[task_id].get(task_id)
            
            if best_metrics is None:
                continue
            
            # Get final performance on this task
            final_metrics = self.metrics_after_task[self.task_order[-1]].get(task_id)
            
            if final_metrics is None:
                continue
            
            # Compute forgetting
            f = {}
            for metric_name in best_metrics:
                if metric_name in final_metrics:
                    # Compute difference (positive is forgetting)
                    f[metric_name] = best_metrics[metric_name] - final_metrics[metric_name]
            
            # Store forgetting
            forgetting[task_id] = f
        
        # Compute average forgetting
        avg_forgetting = {}
        if forgetting:
            # Get all metric names
            metric_names = set()
            for task_metrics in forgetting.values():
                metric_names.update(task_metrics.keys())
            
            # Compute average for each metric
            for metric_name in metric_names:
                values = [task_metrics.get(metric_name, 0) for task_metrics in forgetting.values()]
                avg_forgetting[metric_name] = sum(values) / len(values)
        
        # Compute average backward transfer
        avg_backward_transfer = {}
        if backward_transfer:
            # Get all metric names
            metric_names = set()
            for task_dict in backward_transfer.values():
                for task_metrics in task_dict.values():
                    metric_names.update(task_metrics.keys())
            
            # Compute average for each metric
            for metric_name in metric_names:
                values = []
                for task_dict in backward_transfer.values():
                    for task_metrics in task_dict.values():
                        if metric_name in task_metrics:
                            values.append(task_metrics[metric_name])
                
                if values:
                    avg_backward_transfer[metric_name] = sum(values) / len(values)
        
        # Compute average forward transfer
        avg_forward_transfer = {}
        if forward_transfer:
            # Get all metric names
            metric_names = set()
            for task_dict in forward_transfer.values():
                for task_metrics in task_dict.values():
                    metric_names.update(task_metrics.keys())
            
            # Compute average for each metric
            for metric_name in metric_names:
                values = []
                for task_dict in forward_transfer.values():
                    for task_metrics in task_dict.values():
                        if metric_name in task_metrics:
                            values.append(task_metrics[metric_name])
                
                if values:
                    avg_forward_transfer[metric_name] = sum(values) / len(values)
        
        # Return metrics
        return {
            "forgetting": forgetting,
            "avg_forgetting": avg_forgetting,
            "backward_transfer": backward_transfer,
            "avg_backward_transfer": avg_backward_transfer,
            "forward_transfer": forward_transfer,
            "avg_forward_transfer": avg_forward_transfer
        }
    
    def compute_forgetting_curve(self):
        """
        Compute forgetting curve.
        
        Returns:
            Dictionary with forgetting curve data
        """
        if not self.compute_forgetting_curve:
            logger.warning("Forgetting curve computation is disabled")
            return {}
        
        logger.info("Computing forgetting curve")
        
        # Initialize forgetting curve
        forgetting_curve = {}
        
        # Compute forgetting curve for each task
        for i, task_id in enumerate(self.task_order[:-1]):  # Skip last task
            # Get evaluation dataset
            eval_dataset = self.task_datasets[task_id].get("eval")
            
            if eval_dataset is None:
                logger.warning(f"No evaluation dataset for task {task_id}, skipping forgetting curve")
                continue
            
            # Get performance after training on this task
            best_metrics = self.metrics_after_task[task_id].get(task_id)
            
            if best_metrics is None:
                continue
            
            # Initialize curve for this task
            forgetting_curve[task_id] = {}
            
            # Add initial point
            for metric_name, value in best_metrics.items():
                forgetting_curve[task_id][metric_name] = [value]
            
            # Compute performance after each subsequent task
            for j in range(i + 1, len(self.task_order)):
                next_task_id = self.task_order[j]
                
                # Get metrics after training on next task
                next_metrics = self.metrics_after_task[next_task_id].get(task_id)
                
                if next_metrics is None:
                    continue
                
                # Add point to curve
                for metric_name, value in next_metrics.items():
                    if metric_name in forgetting_curve[task_id]:
                        forgetting_curve[task_id][metric_name].append(value)
        
        return forgetting_curve
    
    def visualize_forgetting(self, metric_name: str = "eval_loss"):
        """
        Visualize forgetting.
        
        Args:
            metric_name: Name of the metric to visualize
            
        Returns:
            PIL Image with visualization
        """
        # Compute forgetting curve if not already computed
        if not hasattr(self, "forgetting_curve") or not self.forgetting_curve:
            self.forgetting_curve = self.compute_forgetting_curve()
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot forgetting curve for each task
        for task_id, metrics in self.forgetting_curve.items():
            if metric_name in metrics:
                values = metrics[metric_name]
                x = list(range(len(values)))
                plt.plot(x, values, marker='o', label=f"Task {task_id}")
        
        # Add labels and title
        plt.xlabel("Tasks")
        plt.ylabel(metric_name)
        plt.title(f"Forgetting Curve ({metric_name})")
        plt.xticks(list(range(len(self.task_order))), self.task_order)
        plt.legend()
        plt.grid(True)
        
        # Save figure to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        
        # Convert buffer to image
        buf.seek(0)
        img = Image.open(buf)
        
        return img
    
    def visualize_backward_transfer(self, metric_name: str = "eval_loss"):
        """
        Visualize backward transfer.
        
        Args:
            metric_name: Name of the metric to visualize
            
        Returns:
            PIL Image with visualization
        """
        # Compute forgetting metrics if not already computed
        if not hasattr(self, "backward_transfer"):
            metrics = self.compute_forgetting()
            self.backward_transfer = metrics["backward_transfer"]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create matrix for heatmap
        task_ids = self.task_order
        matrix = np.zeros((len(task_ids), len(task_ids)))
        
        # Fill matrix with backward transfer values
        for i, task_id in enumerate(task_ids):
            for j, other_task_id in enumerate(task_ids):
                if j <= i:  # Only consider previous tasks
                    continue
                
                # Get backward transfer
                if task_id in self.backward_transfer and other_task_id in self.backward_transfer[task_id]:
                    bt = self.backward_transfer[task_id].get(other_task_id, {})
                    value = bt.get(metric_name, 0)
                    matrix[i, j] = value
        
        # Plot heatmap
        plt.imshow(matrix, cmap='coolwarm', interpolation='nearest')
        plt.colorbar(label=f"Backward Transfer ({metric_name})")
        
        # Add labels and title
        plt.xlabel("Task trained on")
        plt.ylabel("Previous task")
        plt.title(f"Backward Transfer ({metric_name})")
        plt.xticks(list(range(len(task_ids))), task_ids)
        plt.yticks(list(range(len(task_ids))), task_ids)
        
        # Save figure to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        
        # Convert buffer to image
        buf.seek(0)
        img = Image.open(buf)
        
        return img


def compute_forgetting_metrics(
    model: PeftModel,
    task_datasets: Dict[str, Dict[str, Any]],
    metric_fn: Optional[Callable] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Compute forgetting metrics for a model.
    
    Args:
        model: The model to evaluate
        task_datasets: Dictionary mapping task IDs to dictionaries with 'train' and 'eval' datasets
        metric_fn: Function to compute metrics (should return a dictionary)
        **kwargs: Additional arguments for ForgettingMetrics
        
    Returns:
        Dictionary with forgetting metrics
    """
    # Create forgetting metrics
    metrics = ForgettingMetrics(model, task_datasets, metric_fn, **kwargs)
    
    # Compute forgetting
    forgetting_metrics = metrics.compute_forgetting()
    
    # Compute forgetting curve
    forgetting_curve = metrics.compute_forgetting_curve()
    
    # Add forgetting curve to metrics
    forgetting_metrics["forgetting_curve"] = forgetting_curve
    
    return forgetting_metrics
