"""
Document processing module for medical research papers and clinical documents.

This module provides components for processing, analyzing, and synthesizing
information from medical research papers, clinical notes, and other medical documents.
"""

# Import document structure classes
from .document_structure import (
    Entity,
    SectionInfo,
    DocumentStructure,
    RelationInstance,
    ResearchSummary
)

# Import processing components
from .document_processor import BiomedicalDocumentProcessor
from .entity_extractor import BiomedicalEntityExtractor
from .relation_extractor import MedicalRelationExtractor
from .research_summarizer import SciFiveResearchSummarizer

# Import enhanced components
from .reference_parser import ReferenceParser, ParsedReference
from .sentence_segmenter import SentenceSegmenter, Sentence
from .pipeline_optimizer import PipelineOptimizer, StreamingPipeline
from .batch_processor import BatchProcessor
from .result_manager import ResultManager
from .model_manager import ModelManager

# Import main synthesizer
from .medical_research_synthesizer import MedicalResearchSynthesizer

# Import new synthesizer (will eventually replace the old one)
from .medical_research_synthesizer_new import MedicalResearchSynthesizer as EnhancedMedicalResearchSynthesizer

__all__ = [
    # Document structure classes
    'Entity',
    'SectionInfo',
    'DocumentStructure',
    'RelationInstance',
    'ResearchSummary',

    # Processing components
    'BiomedicalDocumentProcessor',
    'BiomedicalEntityExtractor',
    'MedicalRelationExtractor',
    'SciFiveResearchSummarizer',

    # Enhanced components
    'ReferenceParser',
    'ParsedReference',
    'SentenceSegmenter',
    'Sentence',
    'PipelineOptimizer',
    'StreamingPipeline',
    'BatchProcessor',
    'ResultManager',
    'ModelManager',

    # Main synthesizer
    'MedicalResearchSynthesizer',
    'EnhancedMedicalResearchSynthesizer',
]