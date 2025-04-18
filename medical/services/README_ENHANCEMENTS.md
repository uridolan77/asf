# ASF Medical Research Synthesizer Enhancements

This document describes the enhancements made to various services in the ASF Medical Research Synthesizer.

## Common Enhancements

All services have been enhanced with the following features:

1. **Better Error Handling**
   - Custom exception classes for each service
   - Detailed error messages with context
   - Proper error propagation
   - Comprehensive logging

2. **Input Validation**
   - Decorator-based validation for all service methods
   - Type checking and range validation
   - Consistent validation across services

3. **Caching**
   - Decorator-based caching for expensive operations
   - Configurable TTL (Time-To-Live)
   - Cache key generation based on method name and parameters
   - Cache hit/miss logging

4. **Progress Tracking**
   - Real-time progress tracking for long-running operations
   - Detailed progress information
   - Persistent progress storage in cache
   - Progress visualization support

## Analysis Service Enhancements

The Analysis Service has been enhanced with:

- **Enhanced Error Handling**: Better error handling for analysis errors with detailed error messages and proper error propagation
- **Input Validation**: Comprehensive validation of input data for all analysis methods
- **Caching**: Efficient caching of analysis results with configurable TTL
- **Progress Tracking**: Real-time progress tracking for long-running analyses

### Usage Example

```python
from asf.medical.services.analysis_service_enhancements import (
    validate_analysis_input, track_analysis_progress, enhanced_error_handling, cached_analysis
)

class EnhancedAnalysisService:
    @validate_analysis_input
    @track_analysis_progress("contradiction_analysis", total_steps=5)
    @enhanced_error_handling
    @cached_analysis(ttl=3600, prefix="analyze_contradictions", data_type="analysis")
    async def analyze_contradictions(
        self,
        query: str,
        max_results: int = 20,
        threshold: float = 0.7,
        use_biomedlm: bool = True,
        use_tsmixer: bool = False,
        use_lorentz: bool = False,
        user_id: Optional[int] = None,
        progress_tracker: Optional[AnalysisProgressTracker] = None
    ) -> Dict[str, Any]:
        # Implementation with progress tracking
        # ...
```

## Export Service Enhancements

The Export Service has been enhanced with:

- **Support for More Export Formats**: Added support for XML, HTML, Markdown, BibTeX, RIS, and DOCX formats
- **Enhanced Error Handling**: Better error handling for export errors with detailed error messages
- **Input Validation**: Comprehensive validation of input data for all export methods
- **Progress Tracking**: Real-time progress tracking for large exports

### New Export Formats

- **XML**: Export data to XML format
- **HTML**: Export data to HTML format with customizable templates
- **Markdown**: Export data to Markdown format
- **BibTeX**: Export data to BibTeX format for citation management
- **RIS**: Export data to RIS (Research Information Systems) format
- **DOCX**: Export data to DOCX format using python-docx

### Usage Example

```python
from asf.medical.services.export_service_enhancements import (
    validate_export_input, track_export_progress, enhanced_export_error_handling, ExportFormat
)

class EnhancedExportService:
    @validate_export_input
    @track_export_progress("json", total_steps=3)
    @enhanced_export_error_handling
    async def export_to_json(
        self,
        data: Dict[str, Any],
        include_abstracts: bool = True,
        include_metadata: bool = True,
        progress_tracker: Optional[ExportProgressTracker] = None
    ) -> str:
        # Implementation with progress tracking
        # ...
```

## Knowledge Base Service Enhancements

The Knowledge Base Service has been enhanced with:

- **Enhanced Error Handling**: Better error handling for knowledge base operations
- **Input Validation**: Comprehensive validation of input data for all knowledge base methods
- **Caching**: Efficient caching of knowledge base queries with configurable TTL
- **Progress Tracking**: Real-time progress tracking for long-running operations

### New Utility Functions

- **calculate_kb_statistics**: Calculate statistics for a knowledge base
- **detect_kb_updates_needed**: Detect knowledge bases that need updates
- **merge_knowledge_bases**: Merge multiple knowledge bases into a new one

### Usage Example

```python
from asf.medical.services.knowledge_base_service_enhancements import (
    validate_kb_input, track_kb_progress, enhanced_kb_error_handling, cached_kb_query
)

class EnhancedKnowledgeBaseService:
    @validate_kb_input
    @track_kb_progress("knowledge base creation", total_steps=5)
    @enhanced_kb_error_handling
    async def create_knowledge_base(
        self,
        name: str,
        query: str,
        update_schedule: str = "weekly",
        user_id: Optional[int] = None,
        progress_tracker: Optional[KnowledgeBaseProgressTracker] = None
    ) -> Dict[str, Any]:
        # Implementation with progress tracking
        # ...
```

## ML Services Enhancements

The ML Services have been enhanced with:

- **Enhanced Error Handling**: Better error handling for ML model errors
- **Input Validation**: Comprehensive validation of input data for all ML methods
- **Caching**: Efficient caching of ML model predictions with configurable TTL
- **Progress Tracking**: Real-time progress tracking for long-running ML operations

### Usage Example

```python
from asf.medical.ml.services.ml_service_enhancements import (
    validate_ml_input, track_ml_progress, enhanced_ml_error_handling, cached_ml_prediction
)

class EnhancedContradictionService:
    @validate_ml_input
    @track_ml_progress("contradiction_detection", "contradiction_detection", total_steps=4)
    @enhanced_ml_error_handling
    @cached_ml_prediction(ttl=3600, prefix="contradiction", data_type="prediction")
    async def detect_contradictions_in_articles(
        self,
        articles: List[Dict[str, Any]],
        threshold: float = 0.7,
        use_all_methods: bool = True,
        progress_tracker: Optional[MLProgressTracker] = None
    ) -> List[Dict[str, Any]]:
        # Implementation with progress tracking
        # ...
```

## BioMedLM Service Enhancements

The BioMedLM Service has been enhanced with:

- **Enhanced Error Handling**: Better error handling for model errors, including CUDA out of memory errors
- **Input Validation**: Comprehensive validation of input data for all BioMedLM methods
- **Caching**: Efficient caching of model predictions with configurable TTL
- **Progress Tracking**: Real-time progress tracking for long-running model operations

### Usage Example

```python
from asf.medical.ml.models.biomedlm_enhancements import (
    validate_biomedlm_input, track_biomedlm_progress, enhanced_biomedlm_error_handling, cached_biomedlm_prediction
)

class EnhancedBioMedLMService:
    @validate_biomedlm_input
    @track_biomedlm_progress("encoding", total_steps=3)
    @enhanced_biomedlm_error_handling
    @cached_biomedlm_prediction(ttl=3600, prefix="biomedlm_encode", data_type="embedding")
    async def encode_async(
        self,
        text: str,
        progress_tracker: Optional[BioMedLMProgressTracker] = None
    ) -> np.ndarray:
        # Implementation with progress tracking
        # ...
```

## SHAP Explainer Enhancements

The SHAP Explainer has been enhanced with:

- **Enhanced Error Handling**: Better error handling for explanation errors
- **Input Validation**: Comprehensive validation of input data for all SHAP methods
- **Caching**: Efficient caching of explanations with configurable TTL
- **Progress Tracking**: Real-time progress tracking for long-running explanations

### New Utility Functions

- **generate_shap_report**: Generate a detailed HTML report from a SHAP explanation
- **batch_explain_contradictions**: Generate SHAP explanations for a batch of contradictions

### Usage Example

```python
from asf.medical.ml.models.shap_explainer_enhancements import (
    validate_shap_input, track_shap_progress, enhanced_shap_error_handling, cached_shap_explanation
)

class EnhancedSHAPExplainer:
    @validate_shap_input
    @track_shap_progress("text_explanation", total_steps=5)
    @enhanced_shap_error_handling
    @cached_shap_explanation(ttl=3600, prefix="shap_text", data_type="explanation")
    async def explain_text(
        self,
        text: str,
        background_data: Optional[List[str]] = None,
        num_samples: int = 100,
        visualization_type: str = "bar",
        progress_tracker: Optional[SHAPProgressTracker] = None
    ) -> Dict[str, Any]:
        # Implementation with progress tracking
        # ...
```

## Integration

To integrate these enhancements into the existing services, you can:

1. **Inherit from the existing service classes** and override the methods with the enhanced versions
2. **Use the decorators directly** on the existing service methods
3. **Replace the existing service implementations** with the enhanced versions

### Example: Enhancing Existing Services

```python
from asf.medical.services.analysis_service import AnalysisService
from asf.medical.services.analysis_service_enhancements import (
    validate_analysis_input, track_analysis_progress, enhanced_error_handling, cached_analysis
)

class EnhancedAnalysisService(AnalysisService):
    @validate_analysis_input
    @track_analysis_progress("contradiction_analysis", total_steps=5)
    @enhanced_error_handling
    @cached_analysis(ttl=3600, prefix="analyze_contradictions", data_type="analysis")
    async def analyze_contradictions(self, *args, **kwargs):
        return await super().analyze_contradictions(*args, **kwargs)
```

## Conclusion

These enhancements provide a more robust, efficient, and user-friendly experience for the ASF Medical Research Synthesizer. The use of decorators makes it easy to apply these enhancements to existing services without major code changes.
