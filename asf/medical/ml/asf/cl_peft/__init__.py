# CL-PEFT module
"""
This module provides Continual Learning strategies for Parameter-Efficient Fine-Tuning (CL-PEFT).
"""

from .base import CLStrategy
from .generative_replay import GenerativeReplay
from .adaptive_svd import AdaptiveSVD
from .mask_based_cl import MaskBasedCL
from .ewc import EWC
from .experience_replay import ExperienceReplay
from .orthogonal_lora import OrthogonalLoRA

__all__ = [
    'CLStrategy',
    'GenerativeReplay',
    'AdaptiveSVD',
    'MaskBasedCL',
    'EWC',
    'ExperienceReplay',
    'OrthogonalLoRA'
]
