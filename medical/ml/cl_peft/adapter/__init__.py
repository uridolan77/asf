"""
Adapter module for CL-PEFT.

This module provides adapter classes for combining Continual Learning (CL)
with Parameter-Efficient Fine-Tuning (PEFT) techniques.
"""

from .base import CLPEFTAdapter
from .lora import LoRAAdapter
from .qlora import QLoRAAdapter

__all__ = [
    'CLPEFTAdapter',
    'LoRAAdapter',
    'QLoRAAdapter'
]
