"""
Visualization tools for CL-PEFT.

This module provides visualization tools for analyzing continual learning
with parameter-efficient fine-tuning.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
import copy
import matplotlib.pyplot as plt
import io
from PIL import Image
import seaborn as sns
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA

from peft import PeftModel
from transformers import PreTrainedModel

from asf.medical.core.logging_config import get_logger

logger = get_logger(__name__)

class VisualizationTools:
    """
    Class for visualizing continual learning with parameter-efficient fine-tuning.
    """
    
    def __init__(
        self,
        model: PeftModel,
        task_order: List[str],
        device: Optional[torch.device] = None,
        **kwargs
    ):
        """
        Initialize VisualizationTools.
        
        Args:
            model: The model to visualize
            task_order: List of task IDs in the order they were trained
            device: Device to use for visualization
            **kwargs: Additional arguments
        """
        self.model = model
        self.task_order = task_order
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Store model snapshots
        self.model_snapshots = {}
        
        # Store adapter embeddings
        self.adapter_embeddings = {}
    
    def add_model_snapshot(self, task_id: str, model: PeftModel):
        """
        Add a model snapshot.
        
        Args:
            task_id: ID of the task
            model: Model snapshot
        """
        self.model_snapshots[task_id] = copy.deepcopy(model)
    
    def compute_adapter_embeddings(self, method: str = "pca", **kwargs):
        """
        Compute embeddings for adapter parameters.
        
        Args:
            method: Method to use for dimensionality reduction ("pca", "tsne", or "mds")
            **kwargs: Additional arguments for the dimensionality reduction method
            
        Returns:
            Dictionary mapping task IDs to adapter embeddings
        """
        logger.info(f"Computing adapter embeddings using {method}")
        
        # Check if we have model snapshots
        if not self.model_snapshots:
            logger.warning("No model snapshots available")
            return {}
        
        # Extract adapter parameters for each task
        adapter_params = {}
        
        for task_id, model in self.model_snapshots.items():
            # Extract adapter parameters
            params = []
            
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                
                # Check if this is an adapter parameter
                if "lora_A" in name or "lora_B" in name:
                    # Flatten parameter
                    params.append(param.data.cpu().view(-1).numpy())
            
            # Concatenate parameters
            if params:
                adapter_params[task_id] = np.concatenate(params)
        
        # Check if we have adapter parameters
        if not adapter_params:
            logger.warning("No adapter parameters found")
            return {}
        
        # Stack parameters for dimensionality reduction
        task_ids = list(adapter_params.keys())
        param_matrix = np.stack([adapter_params[task_id] for task_id in task_ids])
        
        # Apply dimensionality reduction
        if method == "pca":
            # Apply PCA
            pca = PCA(n_components=2, **kwargs)
            embeddings = pca.fit_transform(param_matrix)
        elif method == "tsne":
            # Apply t-SNE
            tsne = TSNE(n_components=2, **kwargs)
            embeddings = tsne.fit_transform(param_matrix)
        elif method == "mds":
            # Apply MDS
            mds = MDS(n_components=2, **kwargs)
            embeddings = mds.fit_transform(param_matrix)
        else:
            logger.warning(f"Unknown method: {method}, using PCA")
            pca = PCA(n_components=2)
            embeddings = pca.fit_transform(param_matrix)
        
        # Create dictionary mapping task IDs to embeddings
        adapter_embeddings = {task_id: embeddings[i] for i, task_id in enumerate(task_ids)}
        
        # Store embeddings
        self.adapter_embeddings = adapter_embeddings
        
        return adapter_embeddings
    
    def visualize_adapter_embeddings(self, method: str = "pca", **kwargs):
        """
        Visualize adapter embeddings.
        
        Args:
            method: Method to use for dimensionality reduction ("pca", "tsne", or "mds")
            **kwargs: Additional arguments for the dimensionality reduction method
            
        Returns:
            PIL Image with visualization
        """
        # Compute embeddings if not already computed
        if not self.adapter_embeddings:
            self.compute_adapter_embeddings(method, **kwargs)
        
        # Check if we have embeddings
        if not self.adapter_embeddings:
            logger.warning("No adapter embeddings available")
            return None
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot embeddings
        for task_id, embedding in self.adapter_embeddings.items():
            plt.scatter(embedding[0], embedding[1], label=task_id, s=100)
        
        # Add labels and title
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.title(f"Adapter Embeddings ({method.upper()})")
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
    
    def visualize_parameter_changes(self, layer_name: str, top_k: int = 10):
        """
        Visualize parameter changes across tasks.
        
        Args:
            layer_name: Name of the layer to visualize
            top_k: Number of top parameters to visualize
            
        Returns:
            PIL Image with visualization
        """
        # Check if we have model snapshots
        if not self.model_snapshots:
            logger.warning("No model snapshots available")
            return None
        
        # Extract parameters for the specified layer
        layer_params = {}
        
        for task_id, model in self.model_snapshots.items():
            # Find the layer
            for name, param in model.named_parameters():
                if layer_name in name and param.requires_grad:
                    # Store parameter
                    layer_params[task_id] = param.data.cpu().numpy()
                    break
        
        # Check if we found the layer
        if not layer_params:
            logger.warning(f"Layer {layer_name} not found or has no trainable parameters")
            return None
        
        # Compute parameter changes
        param_changes = {}
        
        for i in range(1, len(self.task_order)):
            prev_task_id = self.task_order[i-1]
            curr_task_id = self.task_order[i]
            
            if prev_task_id in layer_params and curr_task_id in layer_params:
                # Compute change
                change = np.abs(layer_params[curr_task_id] - layer_params[prev_task_id])
                
                # Flatten and get top-k indices
                flat_change = change.flatten()
                top_indices = np.argsort(flat_change)[-top_k:]
                
                # Store changes
                param_changes[f"{prev_task_id} -> {curr_task_id}"] = {
                    "indices": top_indices,
                    "values": flat_change[top_indices]
                }
        
        # Check if we have parameter changes
        if not param_changes:
            logger.warning("No parameter changes found")
            return None
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot parameter changes
        x = np.arange(top_k)
        width = 0.8 / len(param_changes)
        
        for i, (transition, changes) in enumerate(param_changes.items()):
            plt.bar(x + i * width, changes["values"], width, label=transition)
        
        # Add labels and title
        plt.xlabel("Parameter Index")
        plt.ylabel("Absolute Change")
        plt.title(f"Top-{top_k} Parameter Changes for {layer_name}")
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
    
    def visualize_adapter_ranks(self):
        """
        Visualize adapter ranks across tasks.
        
        Returns:
            PIL Image with visualization
        """
        # Check if we have model snapshots
        if not self.model_snapshots:
            logger.warning("No model snapshots available")
            return None
        
        # Extract adapter ranks for each task
        adapter_ranks = {}
        
        for task_id, model in self.model_snapshots.items():
            # Extract adapter ranks
            ranks = {}
            
            for name, module in model.named_modules():
                if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                    # Get rank
                    rank = module.lora_A.weight.shape[1]
                    ranks[name] = rank
            
            # Store ranks
            adapter_ranks[task_id] = ranks
        
        # Check if we have adapter ranks
        if not adapter_ranks:
            logger.warning("No adapter ranks found")
            return None
        
        # Get all module names
        all_modules = set()
        for ranks in adapter_ranks.values():
            all_modules.update(ranks.keys())
        
        # Sort module names
        module_names = sorted(all_modules)
        
        # Create matrix for heatmap
        matrix = np.zeros((len(self.task_order), len(module_names)))
        
        # Fill matrix with ranks
        for i, task_id in enumerate(self.task_order):
            if task_id in adapter_ranks:
                for j, module_name in enumerate(module_names):
                    if module_name in adapter_ranks[task_id]:
                        matrix[i, j] = adapter_ranks[task_id][module_name]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot heatmap
        sns.heatmap(matrix, annot=True, fmt=".0f", cmap="viridis",
                   xticklabels=[name.split(".")[-1] for name in module_names],
                   yticklabels=self.task_order)
        
        # Add labels and title
        plt.xlabel("Module")
        plt.ylabel("Task")
        plt.title("Adapter Ranks Across Tasks")
        plt.tight_layout()
        
        # Save figure to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        
        # Convert buffer to image
        buf.seek(0)
        img = Image.open(buf)
        
        return img
    
    def visualize_parameter_importance(self, importance_scores: Dict[str, Dict[str, np.ndarray]]):
        """
        Visualize parameter importance across tasks.
        
        Args:
            importance_scores: Dictionary mapping task IDs to dictionaries mapping
                               parameter names to importance scores
            
        Returns:
            PIL Image with visualization
        """
        # Check if we have importance scores
        if not importance_scores:
            logger.warning("No importance scores provided")
            return None
        
        # Get all parameter names
        all_params = set()
        for scores in importance_scores.values():
            all_params.update(scores.keys())
        
        # Sort parameter names
        param_names = sorted(all_params)
        
        # Create matrix for heatmap
        matrix = np.zeros((len(self.task_order), len(param_names)))
        
        # Fill matrix with importance scores
        for i, task_id in enumerate(self.task_order):
            if task_id in importance_scores:
                for j, param_name in enumerate(param_names):
                    if param_name in importance_scores[task_id]:
                        # Compute average importance
                        score = importance_scores[task_id][param_name]
                        if isinstance(score, np.ndarray):
                            score = np.mean(score)
                        matrix[i, j] = score
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot heatmap
        sns.heatmap(matrix, cmap="viridis",
                   xticklabels=[name.split(".")[-1] for name in param_names],
                   yticklabels=self.task_order)
        
        # Add labels and title
        plt.xlabel("Parameter")
        plt.ylabel("Task")
        plt.title("Parameter Importance Across Tasks")
        plt.tight_layout()
        
        # Save figure to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        
        # Convert buffer to image
        buf.seek(0)
        img = Image.open(buf)
        
        return img
    
    def visualize_task_trajectory(self, method: str = "pca", **kwargs):
        """
        Visualize task trajectory in parameter space.
        
        Args:
            method: Method to use for dimensionality reduction ("pca", "tsne", or "mds")
            **kwargs: Additional arguments for the dimensionality reduction method
            
        Returns:
            PIL Image with visualization
        """
        # Check if we have model snapshots
        if not self.model_snapshots:
            logger.warning("No model snapshots available")
            return None
        
        # Extract adapter parameters for each task
        adapter_params = {}
        
        for task_id, model in self.model_snapshots.items():
            # Extract adapter parameters
            params = []
            
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                
                # Check if this is an adapter parameter
                if "lora_A" in name or "lora_B" in name:
                    # Flatten parameter
                    params.append(param.data.cpu().view(-1).numpy())
            
            # Concatenate parameters
            if params:
                adapter_params[task_id] = np.concatenate(params)
        
        # Check if we have adapter parameters
        if not adapter_params:
            logger.warning("No adapter parameters found")
            return None
        
        # Stack parameters for dimensionality reduction
        task_ids = self.task_order
        param_matrix = np.stack([adapter_params[task_id] for task_id in task_ids if task_id in adapter_params])
        
        # Apply dimensionality reduction
        if method == "pca":
            # Apply PCA
            pca = PCA(n_components=2, **kwargs)
            embeddings = pca.fit_transform(param_matrix)
        elif method == "tsne":
            # Apply t-SNE
            tsne = TSNE(n_components=2, **kwargs)
            embeddings = tsne.fit_transform(param_matrix)
        elif method == "mds":
            # Apply MDS
            mds = MDS(n_components=2, **kwargs)
            embeddings = mds.fit_transform(param_matrix)
        else:
            logger.warning(f"Unknown method: {method}, using PCA")
            pca = PCA(n_components=2)
            embeddings = pca.fit_transform(param_matrix)
        
        # Create dictionary mapping task IDs to embeddings
        task_embeddings = {}
        i = 0
        for task_id in task_ids:
            if task_id in adapter_params:
                task_embeddings[task_id] = embeddings[i]
                i += 1
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot embeddings
        x = [task_embeddings[task_id][0] for task_id in task_ids if task_id in task_embeddings]
        y = [task_embeddings[task_id][1] for task_id in task_ids if task_id in task_embeddings]
        
        plt.plot(x, y, 'o-', linewidth=2, markersize=10)
        
        # Add task labels
        for task_id in task_ids:
            if task_id in task_embeddings:
                plt.annotate(task_id, task_embeddings[task_id], fontsize=12)
        
        # Add labels and title
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.title(f"Task Trajectory in Parameter Space ({method.upper()})")
        plt.grid(True)
        
        # Save figure to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        
        # Convert buffer to image
        buf.seek(0)
        img = Image.open(buf)
        
        return img


def create_visualization_dashboard(
    model: PeftModel,
    task_order: List[str],
    model_snapshots: Optional[Dict[str, PeftModel]] = None,
    importance_scores: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    **kwargs
) -> Dict[str, Image.Image]:
    """
    Create a visualization dashboard for a model.
    
    Args:
        model: The model to visualize
        task_order: List of task IDs in the order they were trained
        model_snapshots: Dictionary mapping task IDs to model snapshots
        importance_scores: Dictionary mapping task IDs to dictionaries mapping
                           parameter names to importance scores
        **kwargs: Additional arguments for VisualizationTools
        
    Returns:
        Dictionary mapping visualization names to PIL Images
    """
    # Create visualization tools
    viz_tools = VisualizationTools(model, task_order, **kwargs)
    
    # Add model snapshots
    if model_snapshots:
        for task_id, snapshot in model_snapshots.items():
            viz_tools.add_model_snapshot(task_id, snapshot)
    
    # Create visualizations
    visualizations = {}
    
    # Adapter embeddings
    try:
        adapter_viz = viz_tools.visualize_adapter_embeddings(method="pca")
        if adapter_viz:
            visualizations["adapter_embeddings_pca"] = adapter_viz
        
        adapter_viz = viz_tools.visualize_adapter_embeddings(method="tsne")
        if adapter_viz:
            visualizations["adapter_embeddings_tsne"] = adapter_viz
    except Exception as e:
        logger.warning(f"Error creating adapter embeddings visualization: {str(e)}")
    
    # Parameter changes
    try:
        # Find a LoRA layer
        lora_layer = None
        for name, module in model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                lora_layer = name
                break
        
        if lora_layer:
            param_viz = viz_tools.visualize_parameter_changes(lora_layer)
            if param_viz:
                visualizations["parameter_changes"] = param_viz
    except Exception as e:
        logger.warning(f"Error creating parameter changes visualization: {str(e)}")
    
    # Adapter ranks
    try:
        rank_viz = viz_tools.visualize_adapter_ranks()
        if rank_viz:
            visualizations["adapter_ranks"] = rank_viz
    except Exception as e:
        logger.warning(f"Error creating adapter ranks visualization: {str(e)}")
    
    # Parameter importance
    if importance_scores:
        try:
            importance_viz = viz_tools.visualize_parameter_importance(importance_scores)
            if importance_viz:
                visualizations["parameter_importance"] = importance_viz
        except Exception as e:
            logger.warning(f"Error creating parameter importance visualization: {str(e)}")
    
    # Task trajectory
    try:
        trajectory_viz = viz_tools.visualize_task_trajectory(method="pca")
        if trajectory_viz:
            visualizations["task_trajectory"] = trajectory_viz
    except Exception as e:
        logger.warning(f"Error creating task trajectory visualization: {str(e)}")
    
    return visualizations
