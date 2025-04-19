"""
Evaluation framework for CL-PEFT.

This package provides tools for evaluating continual learning with
parameter-efficient fine-tuning.
"""

from .eval_forgetting import ForgettingMetrics, compute_forgetting_metrics
from .eval_transfer import TransferMetrics, compute_transfer_metrics
from .eval_visualization import VisualizationTools, create_visualization_dashboard

__all__ = [
    "ForgettingMetrics",
    "compute_forgetting_metrics",
    "TransferMetrics",
    "compute_transfer_metrics",
    "VisualizationTools",
    "create_visualization_dashboard",
]
