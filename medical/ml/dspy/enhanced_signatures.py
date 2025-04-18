"""Enhanced DSPy Signatures for Medical Research

This module provides enhanced DSPy signatures for medical research applications,
with specialized fields and constraints for medical tasks.
"""

import dspy


# Medical RAG Signatures

class EnhancedMedicalRAG(dspy.Signature):
    """Enhanced signature for medical RAG with citations and confidence."""
    
    question = dspy.InputField(desc="Medical question to answer")
    passages = dspy.InputField(desc="Retrieved medical passages")
    answer = dspy.OutputField(desc="Comprehensive medical answer")
    citations = dspy.OutputField(desc="Citations for medical claims")
    confidence = dspy.OutputField(desc="Confidence score (0-1)")
    limitations = dspy.OutputField(desc="Limitations of the provided answer")


class MedicalEvidenceRAG(dspy.Signature):
    """Signature for medical RAG with evidence grading."""
    
    question = dspy.InputField(desc="Medical question to answer")
    passages = dspy.InputField(desc="Retrieved medical passages")
    answer = dspy.OutputField(desc="Comprehensive medical answer")
    evidence_quality = dspy.OutputField(
        desc="Assessment of evidence quality (A: High, B: Moderate, C: Low, D: Very Low)"
    )
    recommendation_strength = dspy.OutputField(
        desc="Strength of recommendation (Strong, Moderate, Weak, None)"
    )
    citations = dspy.OutputField(desc="Citations with evidence levels")


# Contradiction Detection Signatures

class EnhancedContradictionDetection(dspy.Signature):
    """Enhanced signature for medical contradiction detection."""
    
    statement1 = dspy.InputField(desc="First medical statement")
    statement2 = dspy.InputField(desc="Second medical statement")
    contradiction = dspy.OutputField(desc="Whether the statements contradict (True/False)")
    explanation = dspy.OutputField(desc="Detailed explanation of the contradiction or lack thereof")
    confidence = dspy.OutputField(desc="Confidence score (0-1)")


class EnhancedTemporalContradictionDetection(dspy.Signature):
    """Enhanced signature for temporal contradiction detection."""
    
    statement1 = dspy.InputField(desc="First medical statement")
    timestamp1 = dspy.InputField(desc="Timestamp of first statement")
    statement2 = dspy.InputField(desc="Second medical statement")
    timestamp2 = dspy.InputField(desc="Timestamp of second statement")
    contradiction = dspy.OutputField(desc="Whether the statements contradict (True/False)")
    temporal_relation = dspy.OutputField(
        desc="Temporal relation between statements (Evolution, Replacement, Contradiction, Consistent)"
    )
    explanation = dspy.OutputField(desc="Explanation considering temporal context")
    confidence = dspy.OutputField(desc="Confidence score (0-1)")


# Evidence Extraction Signatures

class EnhancedEvidenceExtraction(dspy.Signature):
    """Enhanced signature for medical evidence extraction."""
    
    text = dspy.InputField(desc="Medical text to analyze")
    claim = dspy.InputField(desc="The claim to evaluate")
    evidence = dspy.OutputField(desc="Evidence extracted from the text")
    relation = dspy.OutputField(
        desc="Relation between evidence and claim (Supports, Refutes, Neutral)"
    )
    confidence = dspy.OutputField(desc="Confidence score (0-1)")


class MultiEvidenceExtraction(dspy.Signature):
    """Signature for extracting multiple pieces of evidence."""
    
    text = dspy.InputField(desc="Medical text to analyze")
    claim = dspy.InputField(desc="The claim to evaluate")
    evidence_pieces = dspy.OutputField(
        desc="List of evidence pieces extracted from the text"
    )
    overall_relation = dspy.OutputField(
        desc="Overall relation between evidence and claim (Supports, Refutes, Neutral)"
    )
    confidence = dspy.OutputField(desc="Confidence score (0-1)")


# Medical Summarization Signatures

class EnhancedMedicalSummarization(dspy.Signature):
    """Enhanced signature for medical text summarization."""
    
    text = dspy.InputField(desc="Medical text to summarize")
    audience = dspy.InputField(
        desc="Target audience (Physician, Researcher, Patient, General)"
    )
    summary = dspy.OutputField(desc="Concise summary of the medical text")
    key_points = dspy.OutputField(desc="Key medical points from the text")
    medical_terms = dspy.OutputField(desc="Important medical terms with brief definitions")


class StructuredMedicalSummarization(dspy.Signature):
    """Signature for structured medical summarization."""
    
    text = dspy.InputField(desc="Medical text to summarize")
    background = dspy.OutputField(desc="Background information")
    methods = dspy.OutputField(desc="Methods used")
    results = dspy.OutputField(desc="Key results")
    conclusions = dspy.OutputField(desc="Main conclusions")
    implications = dspy.OutputField(desc="Clinical implications")


# Clinical QA Signatures

class EnhancedClinicalQA(dspy.Signature):
    """Enhanced signature for clinical question answering."""
    
    question = dspy.InputField(desc="Clinical question")
    context = dspy.InputField(desc="Clinical context")
    answer = dspy.OutputField(desc="Comprehensive clinical answer")
    evidence_grade = dspy.OutputField(
        desc="Grade of evidence (A: High, B: Moderate, C: Low, D: Very Low)"
    )
    clinical_implications = dspy.OutputField(desc="Implications for clinical practice")
    confidence = dspy.OutputField(desc="Confidence score (0-1)")


class DiagnosticReasoning(dspy.Signature):
    """Signature for diagnostic reasoning."""
    
    case_description = dspy.InputField(desc="Description of the medical case")
    differential_diagnosis = dspy.OutputField(
        desc="Prioritized list of potential diagnoses"
    )
    recommended_tests = dspy.OutputField(
        desc="Recommended diagnostic tests and procedures"
    )
    reasoning = dspy.OutputField(
        desc="Step-by-step reasoning process"
    )
    confidence = dspy.OutputField(desc="Confidence score (0-1)")


# Evaluation Metrics Signatures

class MedicalQAAccuracy(dspy.Signature):
    """Signature for evaluating medical QA accuracy."""
    
    question = dspy.InputField(desc="Medical question")
    reference_answer = dspy.InputField(desc="Reference answer")
    generated_answer = dspy.InputField(desc="Generated answer")
    accuracy_score = dspy.OutputField(desc="Accuracy score (0-1)")
    error_analysis = dspy.OutputField(desc="Analysis of any errors or omissions")


class MedicalRAGRelevance(dspy.Signature):
    """Signature for evaluating medical RAG passage relevance."""
    
    question = dspy.InputField(desc="Medical question")
    retrieved_passages = dspy.InputField(desc="Retrieved passages")
    relevance_scores = dspy.OutputField(desc="Relevance score for each passage (0-1)")
    overall_relevance = dspy.OutputField(desc="Overall relevance score (0-1)")
    missing_information = dspy.OutputField(desc="Important information that is missing")


class MedicalSummarizationQuality(dspy.Signature):
    """Signature for evaluating medical summarization quality."""
    
    original_text = dspy.InputField(desc="Original medical text")
    summary = dspy.InputField(desc="Generated summary")
    completeness = dspy.OutputField(desc="Completeness score (0-1)")
    accuracy = dspy.OutputField(desc="Factual accuracy score (0-1)")
    conciseness = dspy.OutputField(desc="Conciseness score (0-1)")
    overall_quality = dspy.OutputField(desc="Overall quality score (0-1)")
