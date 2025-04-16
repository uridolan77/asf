"""
Metrics for evaluating CL-PEFT adapters.

This module provides metrics for evaluating CL-PEFT adapters, including:
- Forgetting: Measure of how much performance on previous tasks is lost
- Forward transfer: Measure of how much learning on previous tasks helps on new tasks
"""

from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np

def compute_forgetting(
    original_performance: Dict[str, float],
    current_performance: Dict[str, float],
    metric_keys: List[str] = None,
    higher_is_better: Dict[str, bool] = None
) -> Dict[str, float]:
    """
    Compute forgetting for a set of metrics.
    
    Args:
        original_performance: Original performance metrics
        current_performance: Current performance metrics
        metric_keys: Keys for the metrics to use (optional)
        higher_is_better: Whether higher values are better for each metric
            
    Returns:
        Dictionary mapping metric keys to forgetting values
    """
    # Use all common keys if metric_keys is not provided
    if metric_keys is None:
        metric_keys = [k for k in original_performance if k in current_performance]
    
    # Default higher_is_better based on metric name
    if higher_is_better is None:
        higher_is_better = {}
        for key in metric_keys:
            # Assume higher is better unless the metric contains "loss" or "error"
            higher_is_better[key] = not ("loss" in key.lower() or "error" in key.lower())
    
    # Compute forgetting for each metric
    forgetting = {}
    for key in metric_keys:
        if key not in original_performance or key not in current_performance:
            continue
        
        # Compute forgetting (sign depends on whether higher is better)
        if higher_is_better.get(key, True):
            # For metrics where higher is better (e.g., accuracy)
            # Forgetting is positive when performance decreases
            forgetting[key] = original_performance[key] - current_performance[key]
        else:
            # For metrics where lower is better (e.g., loss)
            # Forgetting is positive when performance increases (loss increases)
            forgetting[key] = current_performance[key] - original_performance[key]
    
    return forgetting

def compute_forward_transfer(
    baseline_performance: Dict[str, float],
    current_performance: Dict[str, float],
    metric_keys: List[str] = None,
    higher_is_better: Dict[str, bool] = None
) -> Dict[str, float]:
    """
    Compute forward transfer for a set of metrics.
    
    Forward transfer measures how much learning on previous tasks helps on new tasks.
    
    Args:
        baseline_performance: Performance metrics for a model trained from scratch
        current_performance: Performance metrics for a model trained on previous tasks
        metric_keys: Keys for the metrics to use (optional)
        higher_is_better: Whether higher values are better for each metric
            
    Returns:
        Dictionary mapping metric keys to forward transfer values
    """
    # Use all common keys if metric_keys is not provided
    if metric_keys is None:
        metric_keys = [k for k in baseline_performance if k in current_performance]
    
    # Default higher_is_better based on metric name
    if higher_is_better is None:
        higher_is_better = {}
        for key in metric_keys:
            # Assume higher is better unless the metric contains "loss" or "error"
            higher_is_better[key] = not ("loss" in key.lower() or "error" in key.lower())
    
    # Compute forward transfer for each metric
    forward_transfer = {}
    for key in metric_keys:
        if key not in baseline_performance or key not in current_performance:
            continue
        
        # Compute forward transfer (sign depends on whether higher is better)
        if higher_is_better.get(key, True):
            # For metrics where higher is better (e.g., accuracy)
            # Forward transfer is positive when performance increases
            forward_transfer[key] = current_performance[key] - baseline_performance[key]
        else:
            # For metrics where lower is better (e.g., loss)
            # Forward transfer is positive when performance decreases (loss decreases)
            forward_transfer[key] = baseline_performance[key] - current_performance[key]
    
    return forward_transfer

def compute_average_forgetting(
    task_performances: Dict[str, Dict[str, Dict[str, float]]],
    metric_key: str = "accuracy",
    higher_is_better: bool = True
) -> float:
    """
    Compute average forgetting across all tasks.
    
    Args:
        task_performances: Dictionary mapping task IDs to dictionaries mapping
            time points to performance metrics
        metric_key: Key for the metric to use
        higher_is_better: Whether higher values are better for the metric
            
    Returns:
        Average forgetting across all tasks
    """
    forgetting_values = []
    
    for task_id, performances in task_performances.items():
        # Get original performance (first time point)
        time_points = sorted(performances.keys())
        if len(time_points) < 2:
            continue
        
        original_perf = performances[time_points[0]].get(metric_key)
        final_perf = performances[time_points[-1]].get(metric_key)
        
        if original_perf is None or final_perf is None:
            continue
        
        # Compute forgetting
        if higher_is_better:
            forgetting = original_perf - final_perf
        else:
            forgetting = final_perf - original_perf
        
        forgetting_values.append(forgetting)
    
    # Compute average forgetting
    if not forgetting_values:
        return 0.0
    
    return np.mean(forgetting_values)
