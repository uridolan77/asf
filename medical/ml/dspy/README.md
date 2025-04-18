# Enhanced DSPy Integration for Medical Research

This package provides an enhanced integration of DSPy for medical research applications, with a focus on reliability, security, and compliance with medical research requirements. It includes specialized modules, metrics, and utilities for working with medical data using DSPy.

## Overview

DSPy is a framework for programming foundation models using techniques like prompting, few-shot learning, and chain-of-thought reasoning. This integration extends DSPy with medical-specific functionality, including:

- Medical RAG (Retrieval-Augmented Generation) modules
- Contradiction detection for medical statements
- Evidence extraction from medical text
- Medical summarization
- Clinical question answering
- Specialized metrics for medical tasks

## Enhanced Features

This enhanced version adds several critical features for production-grade medical applications:

- **Circuit Breaker Pattern**: Prevents cascading failures by temporarily disabling endpoints that are experiencing high failure rates
- **Comprehensive Audit Logging**: PHI/PII detection and redaction for compliance with medical privacy requirements
- **Enhanced Caching**: Better error handling, connection pooling, and retry logic
- **Enhanced Optimization**: Better metrics tracking, timeout handling, and validation set evaluation
- **Medical-Specific Modules**: Specialized modules with PHI detection and sanitization

## Architecture

The DSPy integration follows a layered architecture:

1. **Core Layer**: DSPy client, settings, and base classes
2. **Module Layer**: Specialized DSPy modules for medical tasks
3. **API Layer**: FastAPI endpoints for DSPy functionality
4. **Utility Layer**: Metrics, signatures, and helpers

## Components

### DSPy Client

The `DSPyClient` is a singleton class that provides a unified interface to DSPy functionality. It handles:

- LLM provider management (OpenAI, Anthropic, Azure, etc.)
- Caching (disk, Redis, null)
- Module registration and execution
- Optimization workflows
- Telemetry and logging

### Enhanced DSPy Client

The `EnhancedDSPyClient` extends the base client with production-grade features:

- Circuit breaker pattern to prevent cascading failures
- Comprehensive error handling and logging
- Asynchronous operations with proper thread safety
- Audit logging with PHI/PII detection and redaction
- Enhanced caching with better error handling and connection pooling

### DSPy Settings

The `DSPySettings` class manages configuration for the DSPy integration using Pydantic. It handles:

- LLM provider settings
- Model parameters
- Caching settings
- Telemetry settings
- Thread and retry settings
- Medical-specific settings

### Enhanced DSPy Settings

The `EnhancedDSPySettings` extends the base settings with additional configuration options:

- Circuit breaker settings
- Audit logging settings
- Security settings for input validation and output filtering
- Enhanced optimization settings
- Azure-specific settings

### DSPy Module Base

The `DSPyModuleBase` class provides a standard interface for DSPy modules. It includes:

- Protocol definition for DSPy modules
- Base implementation with common functionality
- Medical-specific extensions
- Audit logging and PHI detection

### Enhanced DSPy Module Base

The `EnhancedDSPyModuleBase` extends the base module with additional features:

- Comprehensive PHI/PII detection and redaction
- Input validation and sanitization
- Output filtering and sanitization
- Detailed audit logging
- Enhanced error handling and recovery

### Circuit Breaker

The `CircuitBreaker` provides protection against cascading failures:

- Prevents cascading failures by temporarily disabling endpoints that are experiencing high failure rates
- Configurable failure thresholds and reset timeouts
- Half-open state for testing recovery
- Registry for managing multiple circuit breakers

### Audit Logger

The `AuditLogger` provides comprehensive audit logging with PHI/PII detection and redaction:

- Automatic detection and redaction of PHI/PII
- Immutable logging for compliance requirements
- Rotation and retention policies
- Structured logging for easy analysis

### Specialized Modules

The integration includes specialized modules for medical tasks:

- `MedicalRAGModule`: RAG for medical questions
- `EnhancedMedicalRAGModule`: Advanced RAG with query expansion and fact checking
- `ContradictionDetectionModule`: Detect contradictions between medical statements
- `TemporalContradictionModule`: Detect contradictions with temporal context
- `EvidenceExtractionModule`: Extract evidence from medical text
- `MedicalSummarizationModule`: Summarize medical content
- `ClinicalQAModule`: Answer clinical questions

### Enhanced Medical Modules

The enhanced integration adds improved versions of medical modules:

- `MedicalRAG`: Medical RAG with citations, confidence scores, and limitations
- `MedicalContradictionDetector`: Contradiction detection with detailed explanations
- `TemporalContradictionDetector`: Temporal contradiction detection with relation analysis
- `MedicalEvidenceExtractor`: Evidence extraction with relation classification
- `MedicalSummarizer`: Medical summarization with structured output options
- `ClinicalQA`: Clinical QA with evidence grading and implications
- `DiagnosticReasoningModule`: Step-by-step diagnostic reasoning

### DSPy Signatures

The integration includes predefined DSPy signatures for medical tasks:

- `MedicalQA`: Question answering with medical context
- `MedicalRAG`: RAG with medical passages
- `EvidenceExtraction`: Evidence extraction with relation classification
- `MedicalSummarization`: Summarization with key findings and implications
- `ContradictionDetection`: Contradiction detection with explanation
- `TemporalContradictionDetection`: Temporal contradiction detection
- And many more...

### Enhanced Signatures

The enhanced integration adds improved signatures with more detailed fields:

- `EnhancedMedicalRAG`: RAG with citations, confidence, and limitations
- `MedicalEvidenceRAG`: RAG with evidence grading
- `EnhancedContradictionDetection`: Contradiction detection with confidence and context
- `EnhancedTemporalContradictionDetection`: Temporal contradiction with relation analysis
- `EnhancedEvidenceExtraction`: Evidence extraction with confidence and reasoning
- `MultiEvidenceExtraction`: Extraction of multiple evidence pieces
- `EnhancedMedicalSummarization`: Summarization with audience targeting
- `StructuredMedicalSummarization`: Structured summarization with sections
- `EnhancedClinicalQA`: Clinical QA with evidence grading and implications
- `DiagnosticReasoning`: Step-by-step diagnostic reasoning

### DSPy Metrics

The integration includes specialized metrics for evaluating medical tasks:

- `medical_qa_accuracy`: Evaluate medical QA accuracy
- `medical_rag_relevance`: Evaluate relevance of retrieved passages
- `contradiction_detection_accuracy`: Evaluate contradiction detection
- `medical_summarization_quality`: Evaluate medical summarization
- `evidence_extraction_accuracy`: Evaluate evidence extraction
- `clinical_qa_correctness`: Evaluate clinical QA correctness
- `combined_medical_metric`: Combined metric for medical tasks
- `create_llm_judge_metric`: Create custom LLM-as-judge metrics

### Enhanced Optimization

The `EnhancedOptimizer` provides enhanced optimization functionality:

- Better metrics tracking and validation
- Timeout handling for long-running optimizations
- Validation set evaluation
- Detailed prompt diff statistics
- Comprehensive logging and artifact management

### API Integration

The integration includes FastAPI endpoints for DSPy functionality:

- `/dspy/modules`: List and register modules
- `/dspy/modules/call`: Call registered modules
- `/dspy/medical-rag`: Perform medical RAG
- `/dspy/contradiction-detection`: Detect contradictions
- `/dspy/temporal-contradiction`: Detect temporal contradictions
- `/dspy/evidence-extraction`: Extract evidence
- `/dspy/medical-summarization`: Summarize medical content
- `/dspy/clinical-qa`: Answer clinical questions

## Usage

### Basic Usage

```python
import asyncio
from asf.medical.ml.dspy.dspy_client import get_dspy_client
from asf.medical.ml.dspy.modules.medical_rag import MedicalRAGModule

async def main():
    # Initialize DSPy client
    dspy_client = await get_dspy_client()

    # Create a Medical RAG module
    medical_rag = MedicalRAGModule(k=3)

    # Register the module
    await dspy_client.register_module(
        name="medical_rag",
        module=medical_rag,
        description="Medical RAG module"
    )

    # Call the module
    result = await dspy_client.call_module(
        module_name="medical_rag",
        question="What are the common side effects of statins?"
    )

    print(f"Answer: {result['answer']}")

    # Shut down the client
    await dspy_client.shutdown()

# Run the example
asyncio.run(main())
```

### Enhanced Usage

```python
import asyncio
from asf.medical.ml.dspy import (
    get_enhanced_client,
    MedicalRAG,
    configure_audit_logger
)

async def main():
    # Configure audit logging
    configure_audit_logger(log_dir="audit_logs")

    # Get enhanced client
    client = await get_enhanced_client()

    # Create a medical RAG module
    medical_rag = MedicalRAG(retriever=retriever)

    # Register the module
    await client.register_module(
        name="medical_rag",
        module=medical_rag,
        description="Medical RAG module"
    )

    # Call the module
    result = await client.call_module(
        "medical_rag",
        question="What are the side effects of aspirin?"
    )

    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Citations: {result['citations']}")
    print(f"Medical context: {result['medical_context']}")
    print(f"Limitations: {result['limitations']}")

    # Clean shutdown
    await client.shutdown()

# Run the main function
asyncio.run(main())
```

### Using the API

```python
from fastapi import FastAPI
from asf.medical.ml.dspy.dspy_api import router as dspy_router

app = FastAPI()

# Include the DSPy router
app.include_router(dspy_router)
```

## Examples

The integration includes examples for various medical tasks:

- `medical_rag_example.py`: Example of medical RAG
- `contradiction_detection_example.py`: Example of contradiction detection
- `evidence_extraction_example.py`: Example of evidence extraction
- `medical_summarization_example.py`: Example of medical summarization
- `clinical_qa_example.py`: Example of clinical QA
- `optimization_example.py`: Example of optimization workflows

### Enhanced Examples

The enhanced integration adds several new examples:

- `enhanced_medical_rag_example.py`: Demonstrates enhanced medical RAG with circuit breaker pattern
- `enhanced_contradiction_detection_example.py`: Demonstrates enhanced contradiction detection
- `enhanced_medical_summarization_example.py`: Demonstrates enhanced medical summarization

## Testing

The integration includes unit tests for all components:

- `test_dspy_client.py`: Tests for DSPy client
- `test_dspy_module_base.py`: Tests for DSPy module base
- `test_medical_rag.py`: Tests for medical RAG
- And more...

### Enhanced Testing

The enhanced integration adds comprehensive tests for new components:

- `test_enhanced_client.py`: Tests for enhanced client with circuit breaker
- `test_audit_logging.py`: Tests for audit logging with PHI detection
- `test_enhanced_cache.py`: Tests for enhanced caching
- `test_medical_modules.py`: Tests for enhanced medical modules
- `test_enhanced_optimization.py`: Tests for enhanced optimization

## Requirements

- Python 3.8+
- DSPy
- FastAPI
- Pydantic
- Redis (optional, for enhanced caching)
- MLflow (optional, for telemetry and optimization)
- Diskcache (for local caching)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
