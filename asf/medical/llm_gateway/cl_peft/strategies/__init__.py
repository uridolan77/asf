"""
Continual Learning Strategies for CL-PEFT

This module provides implementations of various Continual Learning strategies
for mitigating catastrophic forgetting in sequential fine-tuning of LLMs.
"""

from .ewc import ElasticWeightConsolidation
from .replay import ExperienceReplay, GenerativeReplay, MixedDataset, QualityController
from .orthogonal import OrthogonalLoRA, AdaptiveSVD
from .mask_based import MaskBasedCL

__all__ = [
    'ElasticWeightConsolidation',
    'ExperienceReplay',
    'GenerativeReplay',
    'MixedDataset',
    'QualityController',
    'OrthogonalLoRA',
    'AdaptiveSVD',
    'MaskBasedCL'
]
