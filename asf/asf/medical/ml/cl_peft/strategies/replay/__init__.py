"""
Replay-based strategies for CL-PEFT.

This module provides replay-based strategies for mitigating catastrophic forgetting
in sequential fine-tuning of LLMs with PEFT, including:
- Experience Replay: Stores and replays examples from previous tasks
- Generative Replay: Uses the model to generate synthetic examples from previous tasks
"""

from .replay_base import ReplayStrategy
from .experience_replay import ExperienceReplay
from .generative_replay import GenerativeReplay
from .mixed_dataset import MixedDataset
from .quality_control import QualityController
from .replay_utils import (
    GenerationCache,
    get_generation_parameters,
    create_generation_key,
    batch_generate_examples,
    adaptive_temperature_sampling
)

__all__ = [
    'ReplayStrategy',
    'ExperienceReplay',
    'GenerativeReplay',
    'MixedDataset',
    'QualityController',
    'GenerationCache',
    'get_generation_parameters',
    'create_generation_key',
    'batch_generate_examples',
    'adaptive_temperature_sampling'
]
