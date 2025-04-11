# DSPy Integration for Medical Research

This package provides a comprehensive integration of DSPy for medical research applications. It includes specialized modules, metrics, and utilities for working with medical data using DSPy.

## Overview

DSPy is a framework for programming foundation models using techniques like prompting, few-shot learning, and chain-of-thought reasoning. This integration extends DSPy with medical-specific functionality, including:

- Medical RAG (Retrieval-Augmented Generation) modules
- Contradiction detection for medical statements
- Evidence extraction from medical text
- Medical summarization
- Clinical question answering
- Specialized metrics for medical tasks

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

### DSPy Settings

The `DSPySettings` class manages configuration for the DSPy integration using Pydantic. It handles:

- LLM provider settings
- Model parameters
- Caching settings
- Telemetry settings
- Thread and retry settings
- Medical-specific settings

### DSPy Module Base

The `DSPyModuleBase` class provides a standard interface for DSPy modules. It includes:

- Protocol definition for DSPy modules
- Base implementation with common functionality
- Medical-specific extensions
- Audit logging and PHI detection

### Specialized Modules

The integration includes specialized modules for medical tasks:

- `MedicalRAGModule`: RAG for medical questions
- `EnhancedMedicalRAGModule`: Advanced RAG with query expansion and fact checking
- `ContradictionDetectionModule`: Detect contradictions between medical statements
- `TemporalContradictionModule`: Detect contradictions with temporal context
- `EvidenceExtractionModule`: Extract evidence from medical text
- `MedicalSummarizationModule`: Summarize medical content
- `ClinicalQAModule`: Answer clinical questions

### DSPy Signatures

The integration includes predefined DSPy signatures for medical tasks:

- `MedicalQA`: Question answering with medical context
- `MedicalRAG`: RAG with medical passages
- `EvidenceExtraction`: Evidence extraction with relation classification
- `MedicalSummarization`: Summarization with key findings and implications
- `ContradictionDetection`: Contradiction detection with explanation
- `TemporalContradictionDetection`: Temporal contradiction detection
- And many more...

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
- And more...

## Testing

The integration includes unit tests for all components:

- `test_dspy_client.py`: Tests for DSPy client
- `test_dspy_module_base.py`: Tests for DSPy module base
- `test_medical_rag.py`: Tests for medical RAG
- And more...

## Requirements

- Python 3.8+
- DSPy
- FastAPI
- Pydantic
- Redis (optional, for Redis cache)
- MLflow (optional, for telemetry)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
