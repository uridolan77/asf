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

**Status**: Active and used for resolving contradictions detected by the Enhanced Contradiction Service.

### 4. Legacy Services

The following services are deprecated and should not be used in new code:

- `enhanced_contradiction_classifier.py` - Replaced by `contradiction_classifier_service.py`
- `enhanced_contradiction_service.py` - Replaced by `contradiction_service.py`
- `unified_contradiction_service.py` - Replaced by `contradiction_service.py`

**Status**: Deprecated. New code should use the Contradiction Service instead.

## Relationships

1. The Enhanced Contradiction Service (`enhanced_contradiction_service.py`) is the main entry point and should be used by API endpoints and other services.

2. The Enhanced Contradiction Service uses the Enhanced Contradiction Classifier (`enhanced_contradiction_classifier.py`) to classify contradictions.

3. The Contradiction Resolution Service (`contradiction_resolution_service.py`) is used to resolve contradictions detected by the Enhanced Contradiction Service.

4. The Legacy Contradiction Service (`contradiction_service.py`) is deprecated and should not be used in new code.

## Usage Guidelines

1. For new code, always use the Enhanced Contradiction Service (`enhanced_contradiction_service.py`).

2. Do not use the Legacy Contradiction Service (`contradiction_service.py`) in new code.

3. When resolving contradictions, use the Contradiction Resolution Service (`contradiction_resolution_service.py`).

4. For detailed classification of contradictions, use the Enhanced Contradiction Classifier (`enhanced_contradiction_classifier.py`) through the Enhanced Contradiction Service.
