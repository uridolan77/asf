"""
Domain-agnostic DSPy Integration for LLM Gateway

This package provides enhanced DSPy integration for the LLM Gateway,
with a focus on reliability, flexibility, and domain-agnostic applications.

Key components:
- Enhanced client with circuit breaker pattern and improved error handling
- Comprehensive logging and tracing capabilities
- General-purpose DSPy modules for common NLP tasks
- Enhanced caching with better error handling and connection pooling
- Enhanced optimization with better metrics tracking and validation
"""

import logging

# Set up package-level logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler if no handlers exist
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Import core components
from .client import DSPyClient, get_enhanced_client
from .config import DSPySettings, get_dspy_settings, LLMProvider, CacheBackend

# Import module base classes
from .modules.base import DSPyModuleProtocol, DSPyModuleBase, RAGModule, ChainOfThoughtModule
from .modules.enhanced_base import EnhancedDSPyModuleProtocol, EnhancedDSPyModuleBase

# Import module implementations
from .modules.qa import QAModule, AdvancedQAModule
from .modules.consistency import ConsistencyChecker, TemporalConsistencyChecker
from .modules.reasoning import ReasoningModule, ExpertReasoningModule
from .modules.extraction import DataExtractor, StructuredDataExtractor
from .modules.retrieval import RAGImplementation, EnhancedRAG
from .modules.summarization import Summarizer, StructuredSummarizer

# Import utilities
from .utils.circuit_breaker import (
    CircuitState, 
    CircuitOpenError,
    CircuitBreaker,
    AsyncCircuitBreaker
)

from .utils.registry import (
    CircuitBreakerRegistry,
    AsyncCircuitBreakerRegistry,
    get_circuit_breaker_registry,
    get_async_circuit_breaker_registry
)

from .utils.decorators import (
    circuit_breaker,
    async_circuit_breaker,
    with_fallback,
    async_with_fallback
)

from .utils.caching import (
    CacheInterface,
    DiskCache,
    RedisCache,
    NullCache,
    create_cache
)

from .utils.logging import (
    DSPyLogger,
    get_dspy_logger,
    configure_logger
)

# Version
__version__ = "1.0.0"

# Export all
__all__ = [
    # Client
    'DSPyClient',
    'get_enhanced_client',
    
    # Settings
    'DSPySettings',
    'get_dspy_settings',
    'LLMProvider',
    'CacheBackend',

    # Module Base
    'DSPyModuleProtocol',
    'DSPyModuleBase',
    'RAGModule',
    'ChainOfThoughtModule',
    'EnhancedDSPyModuleProtocol',
    'EnhancedDSPyModuleBase',

    # Module Implementations
    'QAModule',
    'AdvancedQAModule',
    'ConsistencyChecker',
    'TemporalConsistencyChecker',
    'ReasoningModule',
    'ExpertReasoningModule',
    'DataExtractor',
    'StructuredDataExtractor',
    'RAGImplementation',
    'EnhancedRAG',
    'Summarizer',
    'StructuredSummarizer',

    # Circuit Breaker
    'CircuitState',
    'CircuitOpenError',
    'CircuitBreaker',
    'AsyncCircuitBreaker',
    'CircuitBreakerRegistry',
    'AsyncCircuitBreakerRegistry',
    'get_circuit_breaker_registry',
    'get_async_circuit_breaker_registry',
    'circuit_breaker',
    'async_circuit_breaker',
    'with_fallback',
    'async_with_fallback',

    # Cache
    'CacheInterface',
    'DiskCache',
    'RedisCache',
    'NullCache',
    'create_cache',

    # Logging
    'DSPyLogger',
    'get_dspy_logger',
    'configure_logger',
]