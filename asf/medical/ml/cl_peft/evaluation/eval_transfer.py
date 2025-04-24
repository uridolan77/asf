"""
Transfer metrics for CL-PEFT.

This module provides metrics for measuring transfer learning in
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

class TransferMetrics:
    """
    Class for measuring transfer learning in continual learning.
    """
    
    def __init__(
        self,
        model: PeftModel,
        task_datasets: Dict[str, Dict[str, Any]],
        metric_fn: Callable,
        device: Optional[torch.device] = None,
        compute_task_similarity: bool = True,
        compute_adapter_similarity: bool = True,
        compute_learning_curves: bool = True,
        **kwargs
    ):
        """
        Initialize TransferMetrics.
        
        Args:
            model: The model to evaluate
            task_datasets: Dictionary mapping task IDs to dictionaries with 'train' and 'eval' datasets
            metric_fn: Function to compute metrics (should return a dictionary)
            device: Device to use for evaluation
            compute_task_similarity: Whether to compute task similarity
            compute_adapter_similarity: Whether to compute adapter similarity
            compute_learning_curves: Whether to compute learning curves
            **kwargs: Additional arguments
        """
        self.model = model
        self.task_datasets = task_datasets
        self.metric_fn = metric_fn
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_task_similarity = compute_task_similarity
        self.compute_adapter_similarity = compute_adapter_similarity
        self.compute_learning_curves = compute_learning_curves
        
        # Store task order
        self.task_order = list(task_datasets.keys())
        
        # Store metrics for each task
        self.task_metrics = {}
        
        # Store task similarity
        self.task_similarity = {}
        
        # Store adapter similarity
        self.adapter_similarity = {}
        
        # Store learning curves
        self.learning_curves = {}
        
        # Store model snapshots if needed
        self.model_snapshots = {}
    
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
        logger.info(f"Computing transfer metrics after task {task_id}")
        
        # Store model snapshot
        self.model_snapshots[task_id] = copy.deepcopy(self.model)
        
        # Compute task similarity if enabled
        if self.compute_task_similarity:
            self._compute_task_similarity(task_id)
        
        # Compute adapter similarity if enabled
        if self.compute_adapter_similarity:
            self._compute_adapter_similarity(task_id)
    
    def _compute_task_similarity(self, task_id: str):
        """
        Compute similarity between tasks.
        
        Args:
            task_id: ID of the current task
        """
        logger.info(f"Computing task similarity for task {task_id}")
        
        # Skip if this is the first task
        if len(self.model_snapshots) <= 1:
            return
        
        # Get current model
        current_model = self.model_snapshots[task_id]
        
        # Compute similarity with previous tasks
        for prev_task_id, prev_model in self.model_snapshots.items():
            if prev_task_id == task_id:
                continue
            
            # Compute similarity between tasks
            similarity = self._compute_model_similarity(prev_model, current_model)
            
            # Store similarity
            if task_id not in self.task_similarity:
                self.task_similarity[task_id] = {}
            
            self.task_similarity[task_id][prev_task_id] = similarity
            
            logger.info(f"Task similarity between {task_id} and {prev_task_id}: {similarity:.4f}")
    
    def _compute_model_similarity(self, model1, model2):
        """
        Compute similarity between two models.
        
        Args:
            model1: First model
            model2: Second model
            
        Returns:
            Similarity score
        """
        # Compute cosine similarity between model parameters
        similarity = 0.0
        count = 0
        
        # Get parameters from both models
        for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
            if name1 != name2:
                continue
            
            # Skip non-adapter parameters
            if not param1.requires_grad or not param2.requires_grad:
                continue
            
            # Compute cosine similarity
            param1_flat = param1.data.view(-1)
            param2_flat = param2.data.view(-1)
            
            # Compute cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(param1_flat, param2_flat, dim=0)
            
            similarity += cos_sim.item()
            count += 1
        
        # Return average similarity
        return similarity / count if count > 0 else 0.0
    
    def _compute_adapter_similarity(self, task_id: str):
        """
        Compute similarity between adapters.
        
        Args:
            task_id: ID of the current task
        """
        logger.info(f"Computing adapter similarity for task {task_id}")
        
        # Skip if this is the first task
        if len(self.model_snapshots) <= 1:
            return
        
        # Get current model
        current_model = self.model_snapshots[task_id]
        
        # Compute similarity with previous tasks
        for prev_task_id, prev_model in self.model_snapshots.items():
            if prev_task_id == task_id:
                continue
            
            # Compute similarity between adapters
            adapter_similarity = {}
            
            # Iterate over modules with adapters
            for name, module in current_model.named_modules():
                # Check if module has adapter
                if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                    # Get adapter parameters
                    current_A = module.lora_A.weight.data
                    current_B = module.lora_B.weight.data
                    
                    # Get corresponding module in previous model
                    prev_module = dict(prev_model.named_modules()).get(name)
                    
                    if prev_module is None or not hasattr(prev_module, "lora_A") or not hasattr(prev_module, "lora_B"):
                        continue
                    
                    prev_A = prev_module.lora_A.weight.data
                    prev_B = prev_module.lora_B.weight.data
                    
                    # Compute similarity between adapters
                    A_sim = torch.nn.functional.cosine_similarity(
                        current_A.view(-1), prev_A.view(-1), dim=0
                    ).item()
                    
                    B_sim = torch.nn.functional.cosine_similarity(
                        current_B.view(-1), prev_B.view(-1), dim=0
                    ).item()
                    
                    # Store similarity
                    adapter_similarity[name] = (A_sim + B_sim) / 2
            
            # Store adapter similarity
            if task_id not in self.adapter_similarity:
                self.adapter_similarity[task_id] = {}
            
            self.adapter_similarity[task_id][prev_task_id] = adapter_similarity
            
            # Compute average similarity
            avg_similarity = sum(adapter_similarity.values()) / len(adapter_similarity) if adapter_similarity else 0.0
            
            logger.info(f"Adapter similarity between {task_id} and {prev_task_id}: {avg_similarity:.4f}")
    
    def compute_learning_curves(self, num_points: int = 10):
        """
        Compute learning curves for each task.
        
        Args:
            num_points: Number of points in the learning curve
            
        Returns:
            Dictionary with learning curves
        """
        if not self.compute_learning_curves:
            logger.warning("Learning curve computation is disabled")
            return {}
        
        logger.info("Computing learning curves")
        
        # Initialize learning curves
        learning_curves = {}
        
        # Compute learning curve for each task
        for task_id in self.task_order:
            # Get training dataset
            train_dataset = self.task_datasets[task_id].get("train")
            eval_dataset = self.task_datasets[task_id].get("eval")
            
            if train_dataset is None or eval_dataset is None:
                logger.warning(f"No training or evaluation dataset for task {task_id}, skipping learning curve")
                continue
            
            # Create a fresh model for this task
            model = copy.deepcopy(self.model_snapshots[self.task_order[0]])
            
            # Reset adapter parameters
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.data.normal_(mean=0.0, std=0.02)
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=f"./learning_curve_{task_id}",
                    num_train_epochs=1,
                    per_device_train_batch_size=8,
                    per_device_eval_batch_size=8,
                    logging_steps=len(train_dataset) // (num_points * 8),
                    save_steps=len(train_dataset) // (num_points * 8),
                    evaluation_strategy="steps",
                    eval_steps=len(train_dataset) // (num_points * 8),
                    remove_unused_columns=False
                ),
                train_dataset=train_dataset,
                eval_dataset=eval_dataset
            )
            
            # Train and collect metrics
            train_output = trainer.train()
            
            # Get learning curve from logs
            logs = train_output.metrics
            
            # Store learning curve
            learning_curves[task_id] = logs
        
        return learning_curves
    
    def compute_transfer_metrics(self):
        """
        Compute transfer metrics.
        
        Returns:
            Dictionary with transfer metrics
        """
        logger.info("Computing transfer metrics")
        
        # Initialize metrics
        transfer_metrics = {
            "task_similarity": self.task_similarity,
            "adapter_similarity": self.adapter_similarity
        }
        
        # Compute learning curves if enabled
        if self.compute_learning_curves:
            learning_curves = self.compute_learning_curves()
            transfer_metrics["learning_curves"] = learning_curves
        
        # Compute average task similarity
        avg_task_similarity = {}
        for task_id, similarities in self.task_similarity.items():
            avg_task_similarity[task_id] = sum(similarities.values()) / len(similarities) if similarities else 0.0
        
        transfer_metrics["avg_task_similarity"] = avg_task_similarity
        
        # Compute average adapter similarity
        avg_adapter_similarity = {}
        for task_id, task_similarities in self.adapter_similarity.items():
            avg_adapter_similarity[task_id] = {}
            
            for other_task_id, adapter_similarities in task_similarities.items():
                avg_adapter_similarity[task_id][other_task_id] = (
                    sum(adapter_similarities.values()) / len(adapter_similarities)
                    if adapter_similarities else 0.0
                )
        
        transfer_metrics["avg_adapter_similarity"] = avg_adapter_similarity
        
        return transfer_metrics
    
    def visualize_task_similarity(self):
        """
        Visualize task similarity.
        
        Returns:
            PIL Image with visualization
        """
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Create matrix for heatmap
        task_ids = self.task_order
        matrix = np.zeros((len(task_ids), len(task_ids)))
        
        # Fill matrix with task similarity values
        for i, task_id in enumerate(task_ids):
            for j, other_task_id in enumerate(task_ids):
                if i == j:
                    # Diagonal: perfect similarity
                    matrix[i, j] = 1.0
                elif task_id in self.task_similarity and other_task_id in self.task_similarity[task_id]:
                    # Get similarity
                    matrix[i, j] = self.task_similarity[task_id][other_task_id]
                elif other_task_id in self.task_similarity and task_id in self.task_similarity[other_task_id]:
                    # Get similarity (symmetric)
                    matrix[i, j] = self.task_similarity[other_task_id][task_id]
        
        # Plot heatmap
        plt.imshow(matrix, cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
        plt.colorbar(label="Task Similarity")
        
        # Add labels and title
        plt.xlabel("Task")
        plt.ylabel("Task")
        plt.title("Task Similarity Matrix")
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
    
    def visualize_adapter_similarity(self, layer_name: Optional[str] = None):
        """
        Visualize adapter similarity.
        
        Args:
            layer_name: Name of the layer to visualize (if None, use average)
            
        Returns:
            PIL Image with visualization
        """
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Create matrix for heatmap
        task_ids = self.task_order
        matrix = np.zeros((len(task_ids), len(task_ids)))
        
        # Fill matrix with adapter similarity values
        for i, task_id in enumerate(task_ids):
            for j, other_task_id in enumerate(task_ids):
                if i == j:
                    # Diagonal: perfect similarity
                    matrix[i, j] = 1.0
                elif task_id in self.adapter_similarity and other_task_id in self.adapter_similarity[task_id]:
                    # Get similarity
                    if layer_name is not None:
                        # Get similarity for specific layer
                        similarities = self.adapter_similarity[task_id][other_task_id]
                        matrix[i, j] = similarities.get(layer_name, 0.0)
                    else:
                        # Get average similarity
                        similarities = self.adapter_similarity[task_id][other_task_id]
                        matrix[i, j] = sum(similarities.values()) / len(similarities) if similarities else 0.0
                elif other_task_id in self.adapter_similarity and task_id in self.adapter_similarity[other_task_id]:
                    # Get similarity (symmetric)
                    if layer_name is not None:
                        # Get similarity for specific layer
                        similarities = self.adapter_similarity[other_task_id][task_id]
                        matrix[i, j] = similarities.get(layer_name, 0.0)
                    else:
                        # Get average similarity
                        similarities = self.adapter_similarity[other_task_id][task_id]
                        matrix[i, j] = sum(similarities.values()) / len(similarities) if similarities else 0.0
        
        # Plot heatmap
        plt.imshow(matrix, cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
        plt.colorbar(label="Adapter Similarity")
        
        # Add labels and title
        plt.xlabel("Task")
        plt.ylabel("Task")
        title = "Adapter Similarity Matrix"
        if layer_name is not None:
            title += f" ({layer_name})"
        plt.title(title)
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
    
    def visualize_learning_curves(self, metric_name: str = "eval_loss"):
        """
        Visualize learning curves.
        
        Args:
            metric_name: Name of the metric to visualize
            
        Returns:
            PIL Image with visualization
        """
        # Check if learning curves are available
        if not hasattr(self, "learning_curves") or not self.learning_curves:
            logger.warning("Learning curves not available")
            return None
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot learning curve for each task
        for task_id, logs in self.learning_curves.items():
            # Extract metric values
            steps = []
            values = []
            
            for step, metrics in logs.items():
                if metric_name in metrics:
                    steps.append(step)
                    values.append(metrics[metric_name])
            
            # Plot learning curve
            if steps and values:
                plt.plot(steps, values, marker='o', label=f"Task {task_id}")
        
        # Add labels and title
        plt.xlabel("Training Steps")
        plt.ylabel(metric_name)
        plt.title(f"Learning Curves ({metric_name})")
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


def compute_transfer_metrics(
    model: PeftModel,
    task_datasets: Dict[str, Dict[str, Any]],
    metric_fn: Optional[Callable] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Compute transfer metrics for a model.
    
    Args:
        model: The model to evaluate
        task_datasets: Dictionary mapping task IDs to dictionaries with 'train' and 'eval' datasets
        metric_fn: Function to compute metrics (should return a dictionary)
        **kwargs: Additional arguments for TransferMetrics
        
    Returns:
        Dictionary with transfer metrics
    """
    # Create transfer metrics
    metrics = TransferMetrics(model, task_datasets, metric_fn, **kwargs)
    
    # Compute transfer metrics
    transfer_metrics = metrics.compute_transfer_metrics()
    
    return transfer_metrics
