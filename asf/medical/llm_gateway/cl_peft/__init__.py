"""
Continual Learning with Parameter-Efficient Fine-Tuning (CL-PEFT) Module

This module provides a comprehensive framework for combining Continual Learning (CL)
with Parameter-Efficient Fine-Tuning (PEFT) techniques like LoRA and QLoRA for
efficient sequential adaptation of Large Language Models (LLMs).

Key components:
- CL-PEFT Adapter: Main class for creating and managing CL-PEFT adapters
- CL Strategies: Various strategies for mitigating catastrophic forgetting
- PEFT Integration: Support for LoRA, QLoRA, and other PEFT methods
- Evaluation Metrics: Tools for measuring performance and forgetting

This module is designed to work with the ASF (Autopoietic Semantic Fields) framework
and integrates with the LLM Gateway and BO management interface.
"""

# Import from config module
from .config import CLPEFTAdapterConfig, CLStrategy, QuantizationMode

# Import from registry module
from .registry import CLPEFTAdapterRegistry, CLPEFTAdapterStatus, get_cl_peft_registry

# Import from adapter module
from .adapter import CLPEFTAdapter, LoRAAdapter, QLoRAAdapter

# Import from models module
from .models import CLPEFTCausalLM

# Import from utils module
from .utils import get_target_modules_for_model, compute_forgetting, compute_forward_transfer

__all__ = [
    # Config
    'CLPEFTAdapterConfig',
    'CLStrategy',
    'QuantizationMode',

    # Registry
    'CLPEFTAdapterRegistry',
    'CLPEFTAdapterStatus',
    'get_cl_peft_registry',

    # Adapter
    'CLPEFTAdapter',
    'LoRAAdapter',
    'QLoRAAdapter',

    # Models
    'CLPEFTCausalLM',

    # Utils
    'get_target_modules_for_model',
    'compute_forgetting',
    'compute_forward_transfer'
]
