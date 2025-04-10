# ASF Medical Research Synthesizer ML Services

This directory contains the machine learning services for the ASF Medical Research Synthesizer.

## PRISMA-guided Screening

The PRISMA-guided screening service (`prisma_screening_service.py`) implements the PRISMA (Preferred Reporting Items for Systematic Reviews and Meta-Analyses) guidelines for screening medical literature. It provides methods for screening articles at different stages of the review process.

### Features

- **PRISMA-compliant**: Follows the PRISMA guidelines for systematic reviews and meta-analyses
- **Multi-stage screening**: Supports identification, screening, eligibility, and inclusion stages
- **Customizable criteria**: Set inclusion and exclusion criteria for each stage
- **Flow diagram**: Generate PRISMA flow diagram data
- **BioMedLM integration**: Use BioMedLM for semantic matching of criteria

### Usage

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

## Bias Assessment

The bias assessment service (`bias_assessment_service.py`) implements methods for assessing risk of bias in medical studies according to standard frameworks like the Cochrane Risk of Bias Tool.

### Features

- **Comprehensive domains**: Assess bias in randomization, blinding, allocation concealment, sample size, attrition, and selective reporting
- **Evidence-based**: Provide evidence for each bias assessment
- **Risk levels**: Categorize risk as low, moderate, high, or unclear
- **Overall assessment**: Calculate overall risk of bias
- **Human-readable summaries**: Generate summaries of bias assessments

### Usage

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

## Enhanced Contradiction Detection

The enhanced contradiction detection service (`enhanced_contradiction_service.py`) implements methods for detecting contradictions in medical literature, integrating multiple methods and models for more accurate contradiction detection.

### Features

- **Multiple detection methods**: Direct, negation, temporal, methodological, and statistical contradiction detection
- **BioMedLM integration**: Use BioMedLM for semantic analysis
- **Temporal analysis**: Detect temporal contradictions using TSMixer
- **SHAP explainability**: Explain contradiction detection results
- **Confidence levels**: Categorize confidence as high, medium, or low
- **Detailed results**: Provide detailed results for each detection method

### Usage

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

## Temporal Service

The temporal service (`temporal_service.py`) implements methods for analyzing temporal sequences of medical claims, using TSMixer for temporal modeling.

### Features

- **Temporal modeling**: Model temporal sequences of medical claims
- **Contradiction detection**: Detect contradictions in temporal sequences
- **Trend analysis**: Analyze trends in medical claims over time
- **TSMixer integration**: Use TSMixer for temporal modeling

### Usage

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
