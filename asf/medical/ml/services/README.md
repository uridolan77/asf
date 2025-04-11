# ASF Medical Research Synthesizer ML Services

This directory contains the machine learning services for the ASF Medical Research Synthesizer.

## Service Architecture

The ML services follow a layered architecture:

1. **API Layer**: Exposes the services to the outside world through FastAPI endpoints
2. **Service Layer**: Contains the business logic and orchestrates the ML models
3. **Repository Layer**: Handles data persistence and retrieval
4. **Client Layer**: Interfaces with external APIs and services
5. **ML Layer**: Contains the machine learning models and algorithms

## Contradiction Detection

The contradiction detection system is composed of the following services:

- **ContradictionService**: Main service for detecting contradictions between medical claims
  - Uses BioMedLM for direct contradiction detection
  - Uses TSMixer for temporal contradiction analysis
  - Uses Lorentz embeddings for hierarchical contradiction detection
  - Uses SHAP for explainability

- **ContradictionClassifierService**: Service for classifying contradictions
  - Provides multi-dimensional classification of contradictions
  - Analyzes temporal factors, population differences, methodological differences
  - Assesses clinical significance and evidence quality

## Contradiction Resolution

The contradiction resolution system is composed of the following services:

- **MedicalContradictionResolutionService**: Main service for resolving contradictions
  - Implements various resolution strategies based on evidence-based medicine principles
  - Generates explanations for resolution decisions

- **ExplanationGenerator**: Service for generating explanations for contradiction resolutions
  - Provides detailed explanations for different resolution strategies
  - Includes clinical implications, limitations, and references

## Temporal Analysis

- **TemporalService**: Service for temporal analysis of medical claims
  - Analyzes temporal aspects of medical claims
  - Calculates temporal confidence
  - Detects temporal contradictions
  - Analyzes temporal sequences

## Bias Assessment

- **BiasAssessmentService**: Service for assessing bias in medical literature
  - Implements various bias assessment tools
  - Provides automated bias assessment of medical literature
  - Categorizes risk as low, moderate, high, or unclear
  - Generates human-readable summaries

## PRISMA-guided Screening

The PRISMA-guided screening service (`prisma_screening_service.py`) implements the PRISMA (Preferred Reporting Items for Systematic Reviews and Meta-Analyses) guidelines for screening medical literature. It provides methods for screening articles at different stages of the review process.

### PRISMA Screening Features

- **PRISMA-compliant**: Follows the PRISMA guidelines for systematic reviews and meta-analyses
- **Multi-stage screening**: Supports identification, screening, eligibility, and inclusion stages
- **Customizable criteria**: Set inclusion and exclusion criteria for each stage
- **Flow diagram**: Generate PRISMA flow diagram data
- **BioMedLM integration**: Use BioMedLM for semantic matching of criteria

### PRISMA Screening Usage

```python
from asf.medical.ml.services.prisma_screening_service import (
    PRISMAScreeningService, ScreeningStage, ScreeningDecision
)

# Initialize service
screening_service = PRISMAScreeningService()

# Set criteria
screening_service.set_criteria(
    stage=ScreeningStage.SCREENING,
    include_criteria=["randomized controlled trial", "condition X"],
    exclude_criteria=["animal study", "in vitro"]
)

# Screen articles
results = await screening_service.screen_articles(
    articles=articles,
    stage=ScreeningStage.SCREENING
)

# Get PRISMA flow data
flow_data = screening_service.get_flow_data()

# Generate flow diagram
diagram_data = screening_service.generate_flow_diagram()
```

## Bias Assessment Implementation

The bias assessment service (`bias_assessment_service.py`) implements methods for assessing risk of bias in medical studies according to standard frameworks like the Cochrane Risk of Bias Tool.

### Bias Assessment Features

- **Comprehensive domains**: Assess bias in randomization, blinding, allocation concealment, sample size, attrition, and selective reporting
- **Evidence-based**: Provide evidence for each bias assessment
- **Risk levels**: Categorize risk as low, moderate, high, or unclear
- **Overall assessment**: Calculate overall risk of bias
- **Human-readable summaries**: Generate summaries of bias assessments

### Bias Assessment Usage

```python
from asf.medical.ml.services.bias_assessment_service import (
    BiasAssessmentService, BiasRisk, BiasDomain
)

# Initialize service
bias_service = BiasAssessmentService()

# Assess bias in a study
assessment = await bias_service.assess_study(study_text)

# Assess bias in multiple studies
results = await bias_service.assess_studies(studies)

# Generate summary
summary = bias_service.generate_summary(assessment)
```

## Legacy Contradiction Detection

The legacy contradiction detection service (`enhanced_contradiction_service.py`) has been replaced by the new `contradiction_service.py` and `contradiction_classifier_service.py`. This section is kept for reference purposes only.

### Legacy Contradiction Features

- **Multiple detection methods**: Direct, negation, temporal, methodological, and statistical contradiction detection
- **BioMedLM integration**: Use BioMedLM for semantic analysis
- **Temporal analysis**: Detect temporal contradictions using TSMixer
- **SHAP explainability**: Explain contradiction detection results
- **Confidence levels**: Categorize confidence as high, medium, or low
- **Detailed results**: Provide detailed results for each detection method

### Legacy Contradiction Usage

```python
from asf.medical.ml.services.enhanced_contradiction_service import (
    EnhancedContradictionService, ContradictionType, ContradictionConfidence
)

# Initialize service
contradiction_service = EnhancedContradictionService()

# Detect contradiction between two claims
result = await contradiction_service.detect_contradiction(
    claim1="Statin therapy reduces cardiovascular risk.",
    claim2="Statin therapy does not reduce cardiovascular risk.",
    use_all_methods=True
)

# Detect contradictions in articles
contradictions = await contradiction_service.detect_contradictions_in_articles(
    articles=articles,
    threshold=0.7,
    use_all_methods=True
)
```

## Temporal Service Implementation

The temporal service (`temporal_service.py`) implements methods for analyzing temporal sequences of medical claims, using TSMixer for temporal modeling.

### Temporal Service Features

- **Temporal modeling**: Model temporal sequences of medical claims
- **Contradiction detection**: Detect contradictions in temporal sequences
- **Trend analysis**: Analyze trends in medical claims over time
- **TSMixer integration**: Use TSMixer for temporal modeling

### Temporal Analysis Usage

```python
from asf.medical.ml.services.temporal_service import TemporalService

# Initialize service
temporal_service = TemporalService()

# Analyze temporal sequence
analysis = await temporal_service.analyze_temporal_sequence(
    sequence=[
        {"claim": "Statin therapy reduces cardiovascular risk.", "timestamp": "2020-01-01"},
        {"claim": "Statin therapy does not reduce cardiovascular risk.", "timestamp": "2021-06-15"}
    ]
)
```

## Testing and Usage

### Testing

Each service has corresponding unit tests in the `asf/medical/tests/unit` directory. Tests can be run using pytest:

```bash
pytest asf/medical/tests/unit/test_contradiction_service.py
pytest asf/medical/tests/unit/test_contradiction_classifier_service.py
pytest asf/medical/tests/unit/test_temporal_service.py
```

### General Usage

Each service is designed to be used independently or as part of a larger system. Services can be instantiated and used directly, or accessed through the API layer.

Example:

```python
from asf.medical.ml.services.contradiction_service import ContradictionService

# Create a contradiction service
contradiction_service = ContradictionService()

# Detect contradiction between two claims
result = await contradiction_service.detect_contradiction(
    claim1="Aspirin reduces the risk of heart attack.",
    claim2="Aspirin does not significantly affect heart attack risk."
)

# Print the result
print(result)
```
