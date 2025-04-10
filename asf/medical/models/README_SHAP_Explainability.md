# SHAP-based Explainability for Contradiction Analysis

This document provides an overview of the SHAP-based explainability feature for contradiction analysis in the ASF framework.

## Overview

The SHAP-based explainability feature helps users understand why the system detected contradictions between medical claims. It uses SHAP (SHapley Additive exPlanations) values to identify the most influential words and phrases that contributed to the contradiction detection.

## Components

The SHAP-based explainability feature consists of the following components:

1. **ContradictionExplainer**: A class that provides methods for explaining why two medical claims were determined to be contradictory.
2. **BioMedLMScorer Integration**: Integration with the BioMedLM wrapper to provide explanations for contradiction detection results.
3. **ContradictionVisualizer**: A class that provides methods for visualizing contradiction explanations.

## Explanation Types

The system supports three types of explanations:

1. **SHAP-based Explanation**: Uses SHAP values to identify the most influential words and phrases.
2. **Negation-based Explanation**: Explains contradictions based on negation patterns.
3. **Multimodal Explanation**: Explains contradictions based on study design and sample size differences.

## Usage

### Basic Usage

```python
from asf.medical.models.biomedlm_wrapper import BioMedLMScorer

# Initialize BioMedLMScorer with SHAP explainer
biomedlm_scorer = BioMedLMScorer(
    model_name="microsoft/BioMedLM",
    use_negation_detection=True,
    use_multimodal_fusion=True,
    use_shap_explainer=True
)

# Generate explanation
claim1 = "Aspirin is effective for treating headaches."
claim2 = "Aspirin has no effect on headache symptoms."

explanation = biomedlm_scorer.explain_contradiction(claim1, claim2)

# Print explanation
print(explanation["summary"])
```

### Visualization

```python
from asf.medical.visualization.contradiction_visualizer import ContradictionVisualizer

# Initialize visualizer
visualizer = ContradictionVisualizer(output_dir="./visualizations")

# Generate explanation
explanation = biomedlm_scorer.explain_contradiction(claim1, claim2)

# Visualize explanation
vis_path = visualizer.visualize_explanation(explanation)
print(f"Visualization saved to: {vis_path}")
```

## Testing

You can test the SHAP-based explainability feature using the provided test scripts:

```bash
# Test contradiction explainer
python -m asf.medical.models.test_contradiction_explainer --output-dir ./explanations

# Test visualizer
python -m asf.medical.visualization.test_visualizer --output-dir ./visualizations
```

## Integration with Metadata Extraction

The SHAP-based explainability feature is integrated with the metadata extraction module to provide explanations for contradictions found during the metadata extraction process.

```python
from asf.medical.data_ingestion_layer.metadata_extraction import MetadataExtractor

# Initialize metadata extractor
extractor = MetadataExtractor()

# Find contradictions with explanations
contradictions = extractor.find_contradictions(publications, use_biomedlm=True)

# Access explanations
for contradiction in contradictions:
    if "explanation" in contradiction:
        print(contradiction["explanation"]["summary"])
```

## Visualization Examples

### SHAP-based Explanation

The SHAP-based explanation visualizes the most influential words and phrases that contributed to the contradiction detection. The color indicates the direction of influence (red for positive, blue for negative), and the length of the bar indicates the magnitude of influence.

### Negation-based Explanation

The negation-based explanation visualizes the negated elements in each claim. It shows which elements are negated in one claim but affirmed in the other.

### Multimodal Explanation

The multimodal explanation visualizes the study design and sample size differences between the claims. It helps users understand how these factors contributed to the contradiction detection.

## Requirements

- SHAP: `pip install shap`
- Matplotlib: `pip install matplotlib`
- BioMedLM: See `setup_biomedlm.py` for installation instructions

## References

- [SHAP: A Unified Approach to Interpreting Model Predictions](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions)
- [BioMedLM: A Domain-Specific Large Language Model for Biomedical Text](https://www.microsoft.com/en-us/research/project/biomedlm/)
