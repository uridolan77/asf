# Medical Research Synthesizer

The Medical Research Synthesizer is a comprehensive pipeline for processing, analyzing, and synthesizing information from medical research papers. It integrates document processing, entity extraction, relation extraction, and research summarization components into a cohesive workflow.

## Features

- **Document Processing**: Extract structured information from medical research papers, including sections, references, and metadata.
- **Entity Extraction**: Identify biomedical entities (diseases, drugs, genes, etc.) with UMLS linking.
- **Relation Extraction**: Extract relationships between biomedical entities using advanced NLP techniques.
- **Knowledge Graph Construction**: Build a knowledge graph from extracted entities and relations.
- **Research Summarization**: Generate concise summaries of research papers with key findings and clinical implications.

## Recent Enhancements

The module has been enhanced with the following features:

### 1. Advanced Subsection Detection

The document processor now includes a comprehensive subsection detection algorithm that can identify hierarchical subsections based on:
- Decimal numbering patterns (e.g., 1.1, 1.2)
- Outline numbering patterns (e.g., I.A, I.B)
- Indentation and font size
- Semantic relationships between sections

### 2. Advanced Reference Parsing

A new reference parser has been added that provides:
- Support for multiple citation styles
- Integration with specialized libraries like Anystyle and Grobid
- Extraction of detailed bibliographic information (authors, title, journal, year, DOI, etc.)
- Robust fallback mechanisms for when specialized libraries are not available

### 3. Sentence Segmentation for Relation Extraction

A new sentence segmenter has been added that provides:
- Advanced sentence boundary detection for scientific text
- Special handling for biomedical abbreviations and citations
- Integration with spaCy for improved accuracy
- Support for cross-sentence relation extraction

### 4. Interactive Knowledge Graph Visualization

A new frontend component has been added for interactive visualization of knowledge graphs:
- 2D and 3D visualization options
- Filtering by entity and relation types
- Searching for specific entities
- Zooming and panning
- Exporting to various formats

### 5. Pipeline Optimization

A new pipeline optimizer has been added that provides:
- Parallel processing of multiple documents
- Caching of intermediate results
- Incremental processing for faster updates
- Streaming results for better user experience

## Components

The Medical Research Synthesizer consists of the following components:

### Core Components

1. **Document Structure Classes** (`document_structure.py`):
   - `Entity`: Represents a biomedical entity with UMLS linking
   - `SectionInfo`: Represents a section in a scientific document
   - `DocumentStructure`: Represents the complete structure of a scientific document
   - `RelationInstance`: Represents a relation between two entities
   - `ResearchSummary`: Represents a structured summary of a research paper

2. **Document Processor** (`document_processor.py`):
   - `BiomedicalDocumentProcessor`: Processes medical research papers to extract structured information

3. **Entity Extractor** (`entity_extractor.py`):
   - `BiomedicalEntityExtractor`: Extracts biomedical entities with UMLS linking using SciSpacy
   - `GLiNERBiomedExtractor`: Neural entity extractor using GLiNER

4. **Relation Extractor** (`relation_extractor.py`):
   - `MedicalRelationExtractor`: Extracts relations between biomedical entities
   - `HGTRelationExtractor`: Heterogeneous Graph Transformer for relation extraction

5. **Research Summarizer** (`research_summarizer.py`):
   - `SciFiveResearchSummarizer`: Generates summaries of research papers using SciFive
   - `EnhancedResearchSummarizer`: Enhanced summarizer with factual consistency checking

### Enhanced Components

6. **Reference Parser** (`reference_parser.py`):
   - `ReferenceParser`: Advanced reference parsing with multiple strategies
   - `ParsedReference`: Structured representation of a bibliographic reference

7. **Sentence Segmenter** (`sentence_segmenter.py`):
   - `SentenceSegmenter`: Advanced sentence segmentation for scientific text
   - `Sentence`: Structured representation of a sentence with metadata

8. **Pipeline Optimizer** (`pipeline_optimizer.py`):
   - `PipelineOptimizer`: Optimization capabilities for the document processing pipeline
   - `StreamingPipeline`: Streaming pipeline for incremental document processing

9. **Batch Processor** (`batch_processor.py`):
   - `BatchProcessor`: Batch processing of multiple documents with parallelism

10. **Result Manager** (`result_manager.py`):
    - `ResultManager`: Management of processing results with various export formats

11. **Model Manager** (`model_manager.py`):
    - `ModelManager`: Management of model lifecycle, versioning, and updates

### Main Pipeline

12. **Main Synthesizer**:
    - `MedicalResearchSynthesizer` (`medical_research_synthesizer.py`): Original synthesizer
    - `EnhancedMedicalResearchSynthesizer` (`medical_research_synthesizer_new.py`): Enhanced synthesizer with improved performance and features

## Integration with Model Lifecycle Manager

The Medical Research Synthesizer integrates with the Model Lifecycle Manager to enable:

- Model versioning and tracking
- Drift detection and monitoring
- Automated retraining
- A/B testing

## Usage

### Basic Usage (Original Synthesizer)

```python
from asf.medical.ml.document_processing import MedicalResearchSynthesizer

# Initialize the synthesizer
synthesizer = MedicalResearchSynthesizer()

# Process a medical research paper
result = synthesizer.process("paper.pdf", is_pdf=True)

# Access extracted information
print(f"Title: {result.title}")
print(f"Found {len(result.entities)} biomedical entities")
print(f"Extracted {len(result.relations)} relations")

# Get research summary
if result.summary:
    print(f"Abstract summary: {result.summary['abstract']}")
    print(f"Key findings: {result.summary['key_findings']}")
    print(f"Clinical implications: {result.summary['clinical_implications']}")

# Save all results
synthesizer.save_results(result, "output_folder")
```

### Enhanced Usage

```python
from asf.medical.ml.document_processing import EnhancedMedicalResearchSynthesizer

# Initialize the enhanced synthesizer
synthesizer = EnhancedMedicalResearchSynthesizer(
    use_cache=True,
    cache_dir="cache"
)

try:
    # Process with progress tracking
    doc_structure, metrics = synthesizer.process_with_progress(
        "paper.pdf",
        is_pdf=True,
        progress_callback=lambda stage, progress: print(f"{stage}: {progress*100:.1f}%")
    )

    # Save results in multiple formats
    saved_files = synthesizer.save_results(
        doc_structure,
        "output_folder",
        formats=["json", "csv", "txt", "html"]
    )

    # Print performance metrics
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
finally:
    # Close synthesizer to release resources
    synthesizer.close()
```

### Streaming Processing

```python
# Process with streaming results
doc_structure, metrics = synthesizer.process_streaming(
    "paper.pdf",
    is_pdf=True,
    streaming_callback=lambda stage, result: print(f"Stage {stage} complete"),
    progress_callback=lambda stage, progress: print(f"Stage {stage}: {progress*100:.1f}%")
)
```

### Batch Processing

```python
# Process batch of documents
metrics = synthesizer.process_batch(
    file_list=["doc1.pdf", "doc2.pdf", "doc3.pdf"],
    output_dir="output_folder",
    batch_size=4,
    all_pdfs=True,
    save_results=True,
    progress_callback=lambda progress: print(f"Progress: {progress}")
)
```

## Advanced Configuration

### Original Synthesizer

```python
# Initialize with custom configuration
custom_synthesizer = MedicalResearchSynthesizer(
    document_processor_args={
        "spacy_model": "en_core_sci_lg",  # Use larger SciSpacy model
        "use_umls": True  # Enable UMLS entity linking
    },
    entity_extractor_args={
        "use_umls": True  # Enable UMLS linking
    },
    relation_extractor_args={
        "use_pretrained": True  # Use pretrained weights
    },
    summarizer_args={
        "model_name": "razent/SciFive-large-Pubmed-paper_summary",
        "max_length": 512
    },
    device="cuda:0"  # Use specific GPU if available
)
```

### Enhanced Synthesizer

```python
# Initialize enhanced synthesizer with custom configuration
enhanced_synthesizer = EnhancedMedicalResearchSynthesizer(
    document_processor_args={
        "spacy_model": "en_core_sci_lg",
        "use_umls": True,
        "use_enhanced_section_classifier": True
    },
    entity_extractor_args={
        "use_gliner": True,  # Use GLiNER for entity extraction
        "use_umls": True
    },
    relation_extractor_args={
        "use_hgt": True,  # Use HGT for relation extraction
        "encoder_model": "microsoft/biogpt",
        "use_sentence_segmentation": True  # Enable sentence segmentation
    },
    summarizer_args={
        "use_enhanced": True,  # Use enhanced summarizer
        "check_factual_consistency": True
    },
    device="cuda:0",
    use_cache=True,
    cache_dir="cache",
    model_dir="models"  # Enable model management
)
```

## Migration

To migrate from the old synthesizer to the enhanced version, use the `migrate_to_enhanced.py` script:

```bash
python migrate_to_enhanced.py --old-cache-dir cache --new-cache-dir enhanced_cache --config-path config.json
```

Or update your imports manually:

```python
# Old import
from asf.medical.ml.document_processing import MedicalResearchSynthesizer

# New import
from asf.medical.ml.document_processing import EnhancedMedicalResearchSynthesizer as MedicalResearchSynthesizer
```

## Testing

To test the enhanced synthesizer, use the `test_synthesizer.py` script:

```bash
python test_synthesizer.py --input path/to/document.pdf --output output_dir --streaming
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.12+
- SciSpacy with UMLS linking
- NetworkX for knowledge graph operations
- PyMuPDF for PDF processing
- React and Material UI for frontend components
- Optional: Anystyle or Grobid for advanced reference parsing

## Future Enhancements

Planned future enhancements include:

1. **Neural PDF Parsing**: Integration with neural PDF parsers like Nougat for better handling of complex layouts
2. **Multi-Modal Entity Recognition**: Extraction of entities from figures, charts, and tables
3. **Document-Level Relations**: Enhanced cross-section relation extraction
4. **Knowledge Graph Integration**: Connection with external knowledge bases like DrugBank or SNOMED CT
5. **Active Learning**: Implementation of active learning for model improvement

## License

This project is licensed under the MIT License - see the LICENSE file for details.
