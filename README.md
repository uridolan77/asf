# ASF Framework Enhancements

This repository contains enhancements to the ASF (Advanced Scientific Framework) for medical research and clinical applications. The enhancements focus on improving temporal modeling, embedding quality, and scalability.

## Key Enhancements

### 1. TSMixer for Temporal Modeling

The TSMixer architecture has been implemented to replace the traditional TCN (Temporal Convolutional Network) for temporal sequence processing. TSMixer offers several advantages:

- Improved performance on multivariate time series data
- Better handling of long-range dependencies
- More efficient computation through parallel processing
- Reversible Instance Normalization (RevIN) for improved stability

### 2. Lorentz Manifold Embeddings

The embedding system has been upgraded to use Lorentz manifolds instead of Poincaré ball embeddings for hyperbolic space representation. Benefits include:

- Improved numerical stability
- Better representation of hierarchical relationships
- More accurate distance calculations
- Hybrid Lorentz-Euclidean embeddings for capturing both hierarchical and similarity-based relationships

### 3. Ray-based Orchestration Framework

A distributed processing framework based on Ray has been implemented for improved scalability and fault tolerance:

- Parallel execution of tasks
- Dependency management for complex workflows
- Asynchronous task execution
- Fault tolerance with automatic retries
- Resource management for efficient utilization

### 4. SHAP-based Explainability

SHAP (SHapley Additive exPlanations) has been integrated for explaining contradiction analysis results:

- Identification of influential words and phrases
- Visualization of feature importance
- Detection of negation patterns
- Analysis of multimodal factors
- HTML report generation for interpretable results

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- geoopt
- torch_geometric
- Ray (optional, for distributed processing)
- SHAP (optional, for explainability)

### Installation

```bash
pip install -r requirements.txt
```

### Running Tests

```bash
python -m unittest discover -s asf/tests
```

## Usage Examples

### TSMixer for Temporal Modeling

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

### Lorentz Manifold Embeddings

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

### Ray-based Orchestration

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

### SHAP-based Explainability

```python
from asf.medical.models.shap_explainer import ContradictionExplainer

# Create explainer
explainer = ContradictionExplainer(model, tokenizer)

# Explain contradiction
explanation = explainer.explain_contradiction(
    claim1="The treatment significantly reduced mortality rates.",
    claim2="The treatment did not show any significant effect on mortality.",
    contradiction_score=0.85,
    use_shap=True,
    use_negation_detection=True
)

# Generate HTML report
from asf.medical.models.shap_explainer import ContradictionVisualizer
visualizer = ContradictionVisualizer()
visualizer.generate_html_report(explanation, "contradiction_report.html")
```

## Architecture

The enhancements are integrated into the existing ASF framework architecture:

- **Layer 1: Knowledge Substrate**
  - Temporal modeling with TSMixer
  - Graph embeddings with Lorentz manifolds
  - Hybrid space transformations

- **Orchestration Layer**
  - Ray-based distributed processing
  - Task scheduling and dependency management
  - Asynchronous execution

- **Medical Layer**
  - SHAP-based explainability for contradiction analysis
  - Visualization and reporting

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
