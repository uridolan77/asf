"""
DSPy Modules for Medical Research

This package contains specialized DSPy modules for medical research tasks.
"""

from .medical_rag import MedicalRAGModule, EnhancedMedicalRAGModule
from .contradiction_detection import ContradictionDetectionModule, TemporalContradictionModule
from .evidence_extraction import EvidenceExtractionModule
from .medical_summarization import MedicalSummarizationModule
from .clinical_qa import ClinicalQAModule
from .diagnostic_reasoning import DiagnosticReasoningModule, SpecialistConsultModule

__all__ = [
    'MedicalRAGModule',
    'EnhancedMedicalRAGModule',
    'ContradictionDetectionModule',
    'TemporalContradictionModule',
    'EvidenceExtractionModule',
    'MedicalSummarizationModule',
    'ClinicalQAModule',
    'DiagnosticReasoningModule',
    'SpecialistConsultModule'
]
