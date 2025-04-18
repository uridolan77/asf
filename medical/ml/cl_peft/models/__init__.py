"""
Model-specific implementations for CL-PEFT.

This module provides model-specific implementations for CL-PEFT.
"""

from .causal_lm import CLPEFTCausalLM

__all__ = [
    'CLPEFTCausalLM'
]
