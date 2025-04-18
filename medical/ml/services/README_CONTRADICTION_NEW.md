# Contradiction Detection and Resolution Architecture

This document explains the architecture and relationships between the contradiction detection and resolution components in the ASF Medical Research Synthesizer.

## Overview

The contradiction detection and resolution system consists of several components that work together to identify, classify, and resolve contradictions in medical literature. The system follows a layered architecture with clear separation of concerns.

## Components

### 1. Contradiction Service (`contradiction_service.py`)

This is the main entry point for contradiction detection. It integrates multiple methods and models for accurate contradiction detection, including:

- BioMedLM for semantic contradiction detection
- TSMixer for temporal contradiction detection
- Lorentz embeddings for semantic similarity
- SHAP for explainability

**Status**: Active and preferred for use in the application.

### 2. Contradiction Classifier Service (`contradiction_classifier_service.py`)

This component provides multi-dimensional classification of medical contradictions, integrating:

- Clinical significance assessment
- Evidence quality assessment
- Temporal factor detection
- Population difference detection
- Methodological difference detection

It works closely with the Contradiction Service to provide detailed classification of detected contradictions.

**Status**: Active and used by the Contradiction Service.

### 3. Contradiction Resolution Service (`contradiction_resolution_service.py`)

This service provides strategies for resolving contradictions in medical literature based on evidence-based medicine principles. It includes:

- Resolution by evidence hierarchy
- Resolution by sample size
- Resolution by recency
- Resolution by population specificity
- Resolution by methodological quality
- Resolution by statistical significance
- Combined evidence resolution

**Status**: Active and used for resolving contradictions detected by the Contradiction Service.

### 4. Legacy Services

The following services are deprecated and should not be used in new code:

- `enhanced_contradiction_classifier.py` - Replaced by `contradiction_classifier_service.py`
- `enhanced_contradiction_service.py` - Replaced by `contradiction_service.py`
- `unified_contradiction_service.py` - Replaced by `contradiction_service.py`

**Status**: Deprecated. New code should use the Contradiction Service instead.

## Relationships

1. The Contradiction Service (`contradiction_service.py`) is the main entry point and should be used by API endpoints and other services.

2. The Contradiction Service uses the Contradiction Classifier Service (`contradiction_classifier_service.py`) to classify contradictions.

3. The Contradiction Resolution Service (`contradiction_resolution_service.py`) is used to resolve contradictions detected by the Contradiction Service.

4. For detailed classification of contradictions, use the Contradiction Classifier Service (`contradiction_classifier_service.py`) through the Contradiction Service.

## Usage Examples

### Detecting Contradictions

```python
from asf.medical.ml.services.contradiction_service import ContradictionService

# Initialize service
contradiction_service = ContradictionService()

# Detect contradiction between two claims
result = await contradiction_service.detect_contradiction(
    claim1="Statin therapy reduces cardiovascular risk.",
    claim2="Statin therapy does not reduce cardiovascular risk.",
    use_all_methods=True
)

# Check if contradiction was detected
if result["contradiction_detected"]:
    print(f"Contradiction detected: {result['contradiction_type']}")
    print(f"Confidence: {result['contradiction_confidence']}")
    print(f"Explanation: {result['explanation']}")
```

### Resolving Contradictions

```python
from asf.medical.ml.services.contradiction_service import ContradictionService
from asf.medical.ml.services.resolution.contradiction_resolution_service import MedicalContradictionResolutionService
from asf.medical.ml.services.resolution.resolution_models import ResolutionStrategy

# Initialize services
contradiction_service = ContradictionService()
resolution_service = MedicalContradictionResolutionService()

# Detect contradiction
contradiction = await contradiction_service.detect_contradiction(
    claim1="Statin therapy reduces cardiovascular risk.",
    claim2="Statin therapy does not reduce cardiovascular risk.",
    use_all_methods=True
)

# Resolve contradiction
resolution = await resolution_service.resolve_contradiction(
    contradiction=contradiction,
    strategy=ResolutionStrategy.EVIDENCE_HIERARCHY
)

print(f"Resolution: {resolution['recommendation']}")
print(f"Confidence: {resolution['confidence']}")
print(f"Explanation: {resolution['explanation']}")
```
