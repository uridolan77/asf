"""
Enums for CL-PEFT configuration.

This module provides enum classes for CL-PEFT configuration options.
"""

from enum import Enum

class CLStrategy(str, Enum):
    """Continual Learning strategy."""
    NAIVE = "naive"  # Sequential fine-tuning without CL mechanisms
    EWC = "ewc"  # Elastic Weight Consolidation
    REPLAY = "replay"  # Experience Replay
    GENERATIVE_REPLAY = "generative_replay"  # Generative Replay
    ORTHOGONAL_LORA = "orthogonal_lora"  # Orthogonal LoRA (O-LoRA)
    ADAPTIVE_SVD = "adaptive_svd"  # Adaptive SVD
    MASK_BASED = "mask_based"  # Mask-based methods

class QuantizationMode(str, Enum):
    """Quantization mode for the base model."""
    NONE = "none"  # No quantization
    INT8 = "int8"  # 8-bit quantization
    INT4 = "int4"  # 4-bit quantization (QLoRA)
