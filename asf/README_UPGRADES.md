# ASF Framework Upgrades

This document describes the upgrades implemented in the ASF framework to enhance its capabilities for medical research and clinical applications.

## Overview

The following upgrades have been implemented:

1. **TSMixer for Temporal Models**: Replaced TCN with TSMixer for improved temporal modeling
2. **Lorentz Manifold Embeddings**: Enhanced embedding quality with Lorentz manifolds
3. **Ray-based Orchestration**: Added distributed processing capabilities with Ray
4. **SHAP-based Explainability**: Implemented SHAP for explaining contradiction analysis results

## 1. TSMixer for Temporal Models

TSMixer is a state-of-the-art architecture for time series forecasting that uses MLPs to mix information across both time and feature dimensions. It offers several advantages over traditional TCNs:

- Better handling of long-range dependencies
- More efficient computation through parallel processing
- Improved performance on multivariate time series data
- Reversible Instance Normalization (RevIN) for improved stability

### Implementation

- Created a complete TSMixer implementation in `asf/layer1_knowledge_substrate/temporal/tsmixer.py`
- Updated the TemporalEncoder to use TSMixer instead of TCN
- Added configuration options to switch between TSMixer and legacy TCN
- Integrated TSMixer with the contradiction detection system

### Usage

```python
from asf.layer1_knowledge_substrate.temporal.tsmixer import TSMixer

# Create TSMixer model
model = TSMixer(
    seq_len=24,
    num_features=5,
    num_blocks=3,
    forecast_horizon=12
)

# Forward pass
output = model(input_sequence)
```

## 2. Lorentz Manifold Embeddings

Lorentz manifold embeddings provide improved numerical stability and better representation of hierarchical relationships compared to Poincaré ball embeddings. Benefits include:

- Better numerical stability
- More accurate distance calculations
- Improved representation of hierarchical relationships
- Hybrid Lorentz-Euclidean embeddings for capturing both hierarchical and similarity-based relationships

### Implementation

- Implemented Lorentz embeddings in `asf/layer1_knowledge_substrate/embeddings/lorentz_embeddings.py`
- Updated the HybridSpaceTransformer to use Lorentz manifolds
- Added configuration options to switch between Lorentz and Poincaré embeddings
- Integrated Lorentz embeddings with the contradiction detection system

### Usage

```python
from asf.layer1_knowledge_substrate.embeddings.lorentz_embeddings import LorentzEmbedding

# Create Lorentz embedding
embedding = LorentzEmbedding(
    num_embeddings=1000,
    embedding_dim=32,
    k=-1.0
)

# Get embeddings
embeddings = embedding(indices)
```

## 3. Ray-based Orchestration

Ray is a distributed computing framework that enables parallel execution of tasks, fault tolerance, and scalability across multiple machines. Benefits include:

- Parallel execution of tasks
- Dependency management for complex workflows
- Asynchronous task execution
- Fault tolerance with automatic retries
- Resource management for efficient utilization

### Implementation

- Created a complete Ray orchestration framework in `asf/orchestration/ray_orchestrator.py`
- Implemented task scheduling, dependency management, and fault tolerance
- Added support for both synchronous and asynchronous execution
- Integrated Ray with the medical API for distributed processing

### Usage

```python
from asf.orchestration.ray_orchestrator import RayOrchestrator, RayConfig

# Create orchestrator
config = RayConfig(use_ray=True)
orchestrator = RayOrchestrator(config=config)

# Register functions
orchestrator.register_function(process_data, "process_data")
orchestrator.register_function(analyze_results, "analyze_results")

# Create tasks
task1_id = orchestrator.create_task(
    name="process_data_task",
    function_name="process_data",
    args=[data]
)

task2_id = orchestrator.create_task(
    name="analyze_results_task",
    function_name="analyze_results",
    dependencies=[task1_id]
)

# Execute workflow
results = orchestrator.execute_workflow([task1_id, task2_id])
```

## 4. SHAP-based Explainability

SHAP (SHapley Additive exPlanations) provides a unified approach to explaining the output of any machine learning model. Benefits include:

- Identification of influential words and phrases
- Visualization of feature importance
- Detection of negation patterns
- Analysis of multimodal factors
- HTML report generation for interpretable results

### Implementation

- Implemented SHAP-based explainability in `asf/medical/models/shap_explainer.py`
- Enhanced the BioMedLM wrapper with SHAP explainability
- Added support for negation detection and multimodal factor analysis
- Created visualization and reporting capabilities

### Usage

```python
from asf.medical.models.shap_explainer import ContradictionExplainer, ContradictionVisualizer

# Create explainer
explainer = ContradictionExplainer(
    model=model,
    tokenizer=tokenizer
)

# Explain contradiction
explanation = explainer.explain_contradiction(
    claim1="The treatment significantly reduced mortality rates.",
    claim2="The treatment did not show any significant effect on mortality.",
    contradiction_score=0.85,
    use_shap=True,
    use_negation_detection=True
)

# Generate HTML report
visualizer = ContradictionVisualizer()
visualizer.generate_html_report(explanation, "contradiction_report.html")
```

## Integration with Existing Components

The upgrades have been integrated with the existing ASF framework components:

- TSMixer has been integrated with the TemporalEncoder in the ChronoGnosisLayer
- Lorentz embeddings have been integrated with the HybridSpaceTransformer
- Ray orchestration has been integrated with the medical API
- SHAP explainability has been integrated with the BioMedLM wrapper

## Testing

Comprehensive test suites have been created for all implemented components:

- `asf/tests/test_tsmixer_lorentz.py`: Tests for TSMixer and Lorentz embeddings
- `asf/tests/test_ray_orchestrator.py`: Tests for Ray orchestration
- `asf/tests/test_shap_explainer.py`: Tests for SHAP explainability
- `asf/tests/test_integration.py`: Integration tests for all components

To run the tests:

```bash
cd asf
python -m unittest discover -s tests
```

## Future Work

1. **Performance Optimization**: Optimize the performance of TSMixer and Lorentz embeddings for large-scale datasets
2. **Advanced Explainability**: Enhance the SHAP explainability with more advanced visualization techniques
3. **Distributed Training**: Implement distributed training with Ray for large models
4. **Integration with External Systems**: Enhance integration with external systems like BioMedLM and other medical databases
