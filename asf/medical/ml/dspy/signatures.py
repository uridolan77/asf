Enhanced DSPy Signatures

This module provides enhanced DSPy signatures with more medical-specific fields
and constraints for medical research applications.

import dspy
from typing import Dict, Any, List, Optional, Union


# Medical RAG Signatures
class EnhancedMedicalRAG(dspy.Signature):
    Enhanced signature for medical RAG (Retrieval-Augmented Generation).
    
    question = dspy.InputField(desc="Medical question to answer")
    passages = dspy.InputField(desc="Retrieved medical passages")
    answer = dspy.OutputField(desc="Comprehensive medical answer")
    citations = dspy.OutputField(desc="Citations for medical claims")
    confidence = dspy.OutputField(desc="Confidence score (0-1)")
    medical_context = dspy.OutputField(desc="Relevant medical context for understanding the answer")
    limitations = dspy.OutputField(desc="Limitations of the provided answer")


class MedicalEvidenceRAG(dspy.Signature):
    Signature for medical RAG with evidence grading.
    
    question = dspy.InputField(desc="Medical question to answer")
    passages = dspy.InputField(desc="Retrieved medical passages")
    answer = dspy.OutputField(desc="Comprehensive medical answer")
    citations = dspy.OutputField(desc="Citations for medical claims")
    evidence_grade = dspy.OutputField(desc="Grade of evidence (A, B, C, D, or I)")
    evidence_assessment = dspy.OutputField(desc="Assessment of the evidence quality")


# Contradiction Detection Signatures
class EnhancedContradictionDetection(dspy.Signature):
    Enhanced signature for contradiction detection.
    
    statement1 = dspy.InputField(desc="First medical statement")
    statement2 = dspy.InputField(desc="Second medical statement")
    contradiction = dspy.OutputField(desc="Whether the statements contradict each other (true/false)")
    explanation = dspy.OutputField(desc="Detailed medical explanation of the contradiction or lack thereof")
    confidence = dspy.OutputField(desc="Confidence score between 0 and 1")
    medical_context = dspy.OutputField(desc="Relevant medical context for understanding the statements")


class EnhancedTemporalContradictionDetection(dspy.Signature):
    Enhanced signature for temporal contradiction detection.
    
    statement1 = dspy.InputField(desc="First medical statement")
    timestamp1 = dspy.InputField(desc="Timestamp of the first statement")
    statement2 = dspy.InputField(desc="Second medical statement")
    timestamp2 = dspy.InputField(desc="Timestamp of the second statement")
    contradiction = dspy.OutputField(desc="Whether the statements contradict each other (true/false)")
    temporal_relation = dspy.OutputField(desc="Temporal relation between the statements (e.g., 'evolution', 'update', 'contradiction')")
    explanation = dspy.OutputField(desc="Detailed medical explanation of the temporal relation")
    confidence = dspy.OutputField(desc="Confidence score between 0 and 1")


# Evidence Extraction Signatures
class EnhancedEvidenceExtraction(dspy.Signature):
    Enhanced signature for evidence extraction.
    
    text = dspy.InputField(desc="Medical text to analyze")
    claim = dspy.InputField(desc="The claim to find evidence for")
    evidence = dspy.OutputField(desc="Extracted evidence from the text")
    relation = dspy.OutputField(desc="Relation between evidence and claim (supports, refutes, neutral)")
    confidence = dspy.OutputField(desc="Confidence score between 0 and 1")
    reasoning = dspy.OutputField(desc="Reasoning process for determining the relation")


class MultiEvidenceExtraction(dspy.Signature):
    Signature for extracting multiple pieces of evidence.
    
    text = dspy.InputField(desc="Medical text to analyze")
    claim = dspy.InputField(desc="The claim to find evidence for")
    evidence_pieces = dspy.OutputField(desc="List of extracted evidence pieces")
    overall_relation = dspy.OutputField(desc="Overall relation between evidence and claim (supports, refutes, neutral, mixed)")
    confidence = dspy.OutputField(desc="Overall confidence score between 0 and 1")


# Medical Summarization Signatures
class EnhancedMedicalSummarization(dspy.Signature):
    Enhanced signature for medical summarization.
    
    text = dspy.InputField(desc="Medical text to summarize")
    audience = dspy.InputField(desc="Target audience (e.g., 'clinician', 'researcher', 'patient')")
    summary = dspy.OutputField(desc="Concise summary of the medical text")
    key_findings = dspy.OutputField(desc="Key findings from the medical text")
    implications = dspy.OutputField(desc="Clinical or research implications")
    limitations = dspy.OutputField(desc="Limitations of the source material")


class StructuredMedicalSummarization(dspy.Signature):
    Signature for structured medical summarization.
    
    text = dspy.InputField(desc="Medical text to summarize")
    content_type = dspy.InputField(desc="Type of content (e.g., 'research_paper', 'clinical_note', 'guideline')")
    audience = dspy.InputField(desc="Target audience (e.g., 'clinician', 'researcher', 'patient')")
    background = dspy.OutputField(desc="Background or context")
    methods = dspy.OutputField(desc="Methods or approach")
    results = dspy.OutputField(desc="Key results or findings")
    conclusions = dspy.OutputField(desc="Conclusions and implications")
    limitations = dspy.OutputField(desc="Limitations or caveats")


# Clinical QA Signatures
class EnhancedClinicalQA(dspy.Signature):
    Enhanced signature for clinical question answering.
    
    question = dspy.InputField(desc="Clinical question")
    context = dspy.InputField(desc="Medical context or literature")
    answer = dspy.OutputField(desc="Comprehensive clinical answer")
    confidence = dspy.OutputField(desc="Confidence score between 0 and 1")
    evidence_grade = dspy.OutputField(desc="Grade of evidence (A, B, C, D, or I)")
    clinical_implications = dspy.OutputField(desc="Clinical implications of the answer")
    limitations = dspy.OutputField(desc="Limitations of the answer")


class DiagnosticReasoning(dspy.Signature):
    Signature for diagnostic reasoning.
    
    case_description = dspy.InputField(desc="Clinical case description")
    differential_diagnosis = dspy.OutputField(desc="Differential diagnosis with likelihood estimates")
    recommended_tests = dspy.OutputField(desc="Recommended diagnostic tests")
    reasoning = dspy.OutputField(desc="Step-by-step diagnostic reasoning process")
    confidence = dspy.OutputField(desc="Overall confidence in the differential diagnosis")


# Medical Metrics Signatures
class MedicalQAAccuracy(dspy.Signature):
    Signature for evaluating medical QA accuracy.
    
    prediction = dspy.InputField(desc="The model's predicted answer")
    reference = dspy.InputField(desc="The reference answer")
    score = dspy.OutputField(desc="Accuracy score between 0 and 1")
    explanation = dspy.OutputField(desc="Explanation of the score")


class MedicalRAGRelevance(dspy.Signature):
    Signature for evaluating medical RAG relevance.
    
    question = dspy.InputField(desc="The medical question")
    passages = dspy.InputField(desc="The retrieved passages")
    relevance_score = dspy.OutputField(desc="Relevance score between 0 and 1")
    explanation = dspy.OutputField(desc="Explanation of the relevance score")


class MedicalSummarizationQuality(dspy.Signature):
    Signature for evaluating medical summarization quality.
    
    text = dspy.InputField(desc="The original medical text")
    summary = dspy.InputField(desc="The generated summary")
    completeness_score = dspy.OutputField(desc="Completeness score between 0 and 1")
    accuracy_score = dspy.OutputField(desc="Accuracy score between 0 and 1")
    conciseness_score = dspy.OutputField(desc="Conciseness score between 0 and 1")
    overall_score = dspy.OutputField(desc="Overall quality score between 0 and 1")
    explanation = dspy.OutputField(desc="Explanation of the scores")


# Export all signatures
__all__ = [
    'EnhancedMedicalRAG',
    'MedicalEvidenceRAG',
    'EnhancedContradictionDetection',
    'EnhancedTemporalContradictionDetection',
    'EnhancedEvidenceExtraction',
    'MultiEvidenceExtraction',
    'EnhancedMedicalSummarization',
    'StructuredMedicalSummarization',
    'EnhancedClinicalQA',
    'DiagnosticReasoning',
    'MedicalQAAccuracy',
    'MedicalRAGRelevance',
    'MedicalSummarizationQuality'
]
