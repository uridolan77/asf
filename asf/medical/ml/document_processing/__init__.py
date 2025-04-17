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

# Import main synthesizer
from .medical_research_synthesizer import MedicalResearchSynthesizer

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

    # Main synthesizer
    'MedicalResearchSynthesizer',
]