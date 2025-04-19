"""
Utility functions for CL-PEFT.

This module provides utility functions for CL-PEFT.
"""

from .model_utils import get_target_modules_for_model
from .metrics import compute_forgetting, compute_forward_transfer

__all__ = [
    'get_target_modules_for_model',
    'compute_forgetting',
    'compute_forward_transfer'
]
