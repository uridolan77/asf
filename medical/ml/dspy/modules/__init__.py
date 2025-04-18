"""
DSPy Modules

This package provides domain-agnostic DSPy modules for various NLP tasks.
"""

from .enhanced_rag import AdvancedRAGModule, MultiStageRAGModule
from .contradiction_detection import ContradictionDetectionModule, TemporalContradictionModule
from .evidence_extraction import EvidenceExtractionModule
from .content_summarization import ContentSummarizationModule, StructuredContentSummarizationModule, MultiDocumentSummarizationModule
from .advanced_qa import AdvancedQAModule, StructuredReasoningModule, GuidelineQAModule
from .reasoning import ReasoningModule, ExpertConsultModule

__all__ = [
    'AdvancedRAGModule',
    'MultiStageRAGModule',
    'ContradictionDetectionModule',
    'TemporalContradictionModule',
    'EvidenceExtractionModule',
    'ContentSummarizationModule',
    'StructuredContentSummarizationModule',
    'MultiDocumentSummarizationModule',
    'AdvancedQAModule',
    'StructuredReasoningModule',
    'GuidelineQAModule',
    'ReasoningModule',
    'ExpertConsultModule'
]
