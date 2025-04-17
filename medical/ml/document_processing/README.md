# Medical Research Synthesizer

The Medical Research Synthesizer is a comprehensive pipeline for processing, analyzing, and synthesizing information from medical research papers. It integrates document processing, entity extraction, relation extraction, and research summarization components into a cohesive workflow.

## Features

- **Document Processing**: Extract structured information from medical research papers, including sections, references, and metadata.
- **Entity Extraction**: Identify biomedical entities (diseases, drugs, genes, etc.) with UMLS linking.
- **Relation Extraction**: Extract relationships between biomedical entities using advanced NLP techniques.
- **Knowledge Graph Construction**: Build a knowledge graph from extracted entities and relations.
- **Research Summarization**: Generate concise summaries of research papers with key findings and clinical implications.

## Components

The Medical Research Synthesizer consists of the following components:

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

4. **Relation Extractor** (`relation_extractor.py`):
   - `MedicalRelationExtractor`: Extracts relations between biomedical entities
   - `BioGPTRelationExtractor`: Neural network model for relation extraction

5. **Research Summarizer** (`research_summarizer.py`):
   - `SciFiveResearchSummarizer`: Generates summaries of research papers using SciFive

6. **Main Pipeline** (`medical_research_synthesizer.py`):
   - `MedicalResearchSynthesizer`: Integrates all components into a cohesive workflow

## Integration with Model Lifecycle Manager

The Medical Research Synthesizer integrates with the Model Lifecycle Manager to enable:

- Model versioning and tracking
- Drift detection and monitoring
- Automated retraining
- A/B testing

## Usage

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

## Advanced Configuration

The Medical Research Synthesizer can be configured with custom settings for each component:

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

## Requirements

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.12+
- SciSpacy with UMLS linking
- NetworkX for knowledge graph operations
- PyMuPDF for PDF processing

## License

This project is licensed under the MIT License - see the LICENSE file for details.
