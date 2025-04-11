"""
DSPy Integration for Medical Research

This package provides a comprehensive integration of DSPy for medical research applications.
"""

from .dspy_client import DSPyClient, get_dspy_client
from .dspy_settings import DSPySettings, get_dspy_settings, LLMProvider, CacheBackend
from .dspy_module_base import DSPyModuleProtocol, DSPyModuleBase, MedicalDSPyModule, RAGModule, ChainOfThoughtModule
from .dspy_signatures import *
from .dspy_metrics import *

__version__ = "1.0.0"
__all__ = [
    'DSPyClient',
    'get_dspy_client',
    'DSPySettings',
    'get_dspy_settings',
    'LLMProvider',
    'CacheBackend',
    'DSPyModuleProtocol',
    'DSPyModuleBase',
    'MedicalDSPyModule',
    'RAGModule',
    'ChainOfThoughtModule'
]
