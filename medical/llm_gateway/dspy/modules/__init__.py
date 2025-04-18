"""DSPy Modules

This package provides domain-agnostic DSPy modules for various NLP tasks.
"""

# Base modules
from .base import DSPyModuleBase, DSPyModuleProtocol, RAGModule, ChainOfThoughtModule
from .enhanced_base import EnhancedDSPyModuleBase, EnhancedDSPyModuleProtocol, EnhancedRAGModule, EnhancedChainOfThoughtModule

# Domain-agnostic modules
from .qa import QAModule, AdvancedQAModule, MultiQuestionModule
from .advanced_qa import AdvancedQAModule as EnhancedQAModule, StructuredReasoningModule, GuidelineQAModule
from .reasoning import ReasoningModule, ExpertReasoningModule
from .retrieval import RAGImplementation, EnhancedRAG
from .enhanced_rag import AdvancedRAGModule, MultiStageRAGModule
from .content_summarization import ContentSummarizationModule, StructuredContentSummarizationModule, MultiDocumentSummarizationModule
from .extraction import DataExtractor, StructuredDataExtractor
from .consistency import ConsistencyChecker, TemporalConsistencyChecker
from .contradiction_detection import ContradictionDetector, TemporalContradictionDetector
from .evidence_extraction import EvidenceExtractor, MultiEvidenceExtractor

# Export all
__all__ = [
    # Base modules
    'DSPyModuleBase',
    'DSPyModuleProtocol',
    'RAGModule',
    'ChainOfThoughtModule',
    'EnhancedDSPyModuleBase',
    'EnhancedDSPyModuleProtocol',
    'EnhancedRAGModule',
    'EnhancedChainOfThoughtModule',

    # Domain-agnostic modules
    'QAModule',
    'AdvancedQAModule',
    'MultiQuestionModule',
    'EnhancedQAModule',
    'StructuredReasoningModule',
    'GuidelineQAModule',
    'ReasoningModule',
    'ExpertReasoningModule',
    'RAGImplementation',
    'EnhancedRAG',
    'AdvancedRAGModule',
    'MultiStageRAGModule',
    'ContentSummarizationModule',
    'StructuredContentSummarizationModule',
    'MultiDocumentSummarizationModule',
    'DataExtractor',
    'StructuredDataExtractor',
    'ConsistencyChecker',
    'TemporalConsistencyChecker',
    'ContradictionDetector',
    'TemporalContradictionDetector',
    'EvidenceExtractor',
    'MultiEvidenceExtractor'
]
