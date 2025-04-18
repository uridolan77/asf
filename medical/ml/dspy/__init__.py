"""
Enhanced DSPy Integration for Medical Research

This package provides enhanced DSPy integration for medical research applications,
with a focus on reliability, security, and compliance with medical research requirements.

Key components:
- Enhanced client with circuit breaker pattern and improved error handling
- Comprehensive audit logging with PHI/PII detection and redaction
- Medical-specific DSPy modules for common medical NLP tasks
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

# Legacy imports for backward compatibility
from .dspy_client import DSPyClient, get_dspy_client
from .dspy_settings import DSPySettings, get_dspy_settings, LLMProvider as LegacyLLMProvider, CacheBackend as LegacyCacheBackend
from .dspy_module_base import DSPyModuleProtocol, DSPyModuleBase, MedicalDSPyModule, RAGModule, ChainOfThoughtModule
from .dspy_signatures import *
from .dspy_metrics import *

# Import enhanced components
from .enhanced_settings import (
    LLMProvider,
    CacheBackend,
    EnhancedDSPySettings,
    get_enhanced_settings
)

from .enhanced_cache import (
    CacheInterface,
    EnhancedDiskCache,
    EnhancedRedisCache,
    NullCache,
    create_cache
)

from .circuit_breaker import (
    CircuitState,
    CircuitOpenError,
    CircuitBreaker,
    AsyncCircuitBreaker
)

from .circuit_breaker_registry import (
    CircuitBreakerRegistry,
    AsyncCircuitBreakerRegistry,
    get_circuit_breaker_registry,
    get_async_circuit_breaker_registry
)

from .circuit_breaker_decorators import (
    circuit_breaker,
    async_circuit_breaker,
    with_fallback,
    async_with_fallback
)

from .audit_logging import (
    AuditLogger,
    get_audit_logger,
    configure_audit_logger
)

from .enhanced_module_base import (
    EnhancedDSPyModuleProtocol,
    EnhancedDSPyModuleBase,
    EnhancedRAGModule,
    EnhancedChainOfThoughtModule
)

from .enhanced_signatures import (
    EnhancedMedicalRAG,
    MedicalEvidenceRAG,
    EnhancedContradictionDetection,
    EnhancedTemporalContradictionDetection,
    EnhancedEvidenceExtraction,
    MultiEvidenceExtraction,
    EnhancedMedicalSummarization,
    StructuredMedicalSummarization,
    EnhancedClinicalQA,
    DiagnosticReasoning,
    MedicalQAAccuracy,
    MedicalRAGRelevance,
    MedicalSummarizationQuality
)

from .enhanced_client import (
    EnhancedDSPyClient,
    get_enhanced_client
)

from .enhanced_optimization import (
    EnhancedOptimizer,
    get_enhanced_optimizer
)

from .medical_modules import (
    MedicalRAG,
    MedicalContradictionDetector,
    TemporalContradictionDetector,
    MedicalEvidenceExtractor,
    MedicalSummarizer,
    ClinicalQA,
    DiagnosticReasoningModule
)

# Version
__version__ = "1.1.0"

# Export all
__all__ = [
    # Legacy components
    'DSPyClient',
    'get_dspy_client',
    'DSPySettings',
    'get_dspy_settings',
    'LegacyLLMProvider',
    'LegacyCacheBackend',
    'DSPyModuleProtocol',
    'DSPyModuleBase',
    'MedicalDSPyModule',
    'RAGModule',
    'ChainOfThoughtModule',

    # Enhanced Settings
    'LLMProvider',
    'CacheBackend',
    'EnhancedDSPySettings',
    'get_enhanced_settings',

    # Cache
    'CacheInterface',
    'EnhancedDiskCache',
    'EnhancedRedisCache',
    'NullCache',
    'create_cache',

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

    # Audit Logging
    'AuditLogger',
    'get_audit_logger',
    'configure_audit_logger',

    # Module Base
    'EnhancedDSPyModuleProtocol',
    'EnhancedDSPyModuleBase',
    'EnhancedRAGModule',
    'EnhancedChainOfThoughtModule',

    # Signatures
    'EnhancedMedicalRAG',
    'MedicalEvidenceRAG',
    'EnhancedContradictionDetection',
    'EnhancedTemporalContradictionDetection',
    'EnhancedEvidenceExtraction',
    'MultiEvidenceExtraction',
    'EnhancedMedicalSummarization',
    'StructuredMedicalSummarization',
    'EnhancedClinicalQA',
    'DiagnosticReasoning',
    'MedicalQAAccuracy',
    'MedicalRAGRelevance',
    'MedicalSummarizationQuality',

    # Client
    'EnhancedDSPyClient',
    'get_enhanced_client',

    # Optimization
    'EnhancedOptimizer',
    'get_enhanced_optimizer',

    # Medical Modules
    'MedicalRAG',
    'MedicalContradictionDetector',
    'TemporalContradictionDetector',
    'MedicalEvidenceExtractor',
    'MedicalSummarizer',
    'ClinicalQA',
    'DiagnosticReasoningModule'
]
