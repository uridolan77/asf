"""Enhanced DSPy Signatures

This module provides enhanced DSPy signatures with specialized fields and constraints
for various tasks.
"""

import dspy


# RAG Signatures

class EnhancedRAG(dspy.Signature):
    """Enhanced signature for RAG with citations and confidence."""

    question = dspy.InputField(desc="Question to answer")
    passages = dspy.InputField(desc="Retrieved passages")
    answer = dspy.OutputField(desc="Comprehensive answer")
    citations = dspy.OutputField(desc="Citations for claims")
    confidence = dspy.OutputField(desc="Confidence score (0-1)")
    limitations = dspy.OutputField(desc="Limitations of the provided answer")


class EvidenceRAG(dspy.Signature):
    """Signature for RAG with evidence grading."""

    question = dspy.InputField(desc="Question to answer")
    passages = dspy.InputField(desc="Retrieved passages")
    answer = dspy.OutputField(desc="Comprehensive answer")
    evidence_quality = dspy.OutputField(
        desc="Assessment of evidence quality (A: High, B: Moderate, C: Low, D: Very Low)"
    )
    recommendation_strength = dspy.OutputField(
        desc="Strength of recommendation (Strong, Moderate, Weak, None)"
    )
    citations = dspy.OutputField(desc="Citations with evidence levels")


# Contradiction Detection Signatures

class EnhancedContradictionDetection(dspy.Signature):
    """Enhanced signature for contradiction detection."""

    statement1 = dspy.InputField(desc="First statement")
    statement2 = dspy.InputField(desc="Second statement")
    contradiction = dspy.OutputField(desc="Whether the statements contradict (True/False)")
    explanation = dspy.OutputField(desc="Detailed explanation of the contradiction or lack thereof")
    confidence = dspy.OutputField(desc="Confidence score (0-1)")


class EnhancedTemporalContradictionDetection(dspy.Signature):
    """Enhanced signature for temporal contradiction detection."""

    statement1 = dspy.InputField(desc="First statement")
    timestamp1 = dspy.InputField(desc="Timestamp of first statement")
    statement2 = dspy.InputField(desc="Second statement")
    timestamp2 = dspy.InputField(desc="Timestamp of second statement")
    contradiction = dspy.OutputField(desc="Whether the statements contradict (True/False)")
    temporal_relation = dspy.OutputField(
        desc="Temporal relation between statements (Evolution, Replacement, Contradiction, Consistent)"
    )
    explanation = dspy.OutputField(desc="Explanation considering temporal context")
    confidence = dspy.OutputField(desc="Confidence score (0-1)")


# Evidence Extraction Signatures

class EnhancedEvidenceExtraction(dspy.Signature):
    """Enhanced signature for evidence extraction."""

    text = dspy.InputField(desc="Text to analyze")
    claim = dspy.InputField(desc="The claim to evaluate")
    evidence = dspy.OutputField(desc="Evidence extracted from the text")
    relation = dspy.OutputField(
        desc="Relation between evidence and claim (Supports, Refutes, Neutral)"
    )
    confidence = dspy.OutputField(desc="Confidence score (0-1)")


class MultiEvidenceExtraction(dspy.Signature):
    """Signature for extracting multiple pieces of evidence."""

    text = dspy.InputField(desc="Text to analyze")
    claim = dspy.InputField(desc="The claim to evaluate")
    evidence_pieces = dspy.OutputField(
        desc="List of evidence pieces extracted from the text"
    )
    overall_relation = dspy.OutputField(
        desc="Overall relation between evidence and claim (Supports, Refutes, Neutral)"
    )
    confidence = dspy.OutputField(desc="Confidence score (0-1)")


# Content Summarization Signatures

class ContentSummarization(dspy.Signature):
    """Enhanced signature for text summarization."""

    text = dspy.InputField(desc="Text to summarize")
    audience = dspy.InputField(
        desc="Target audience (Expert, Researcher, General)"
    )
    summary = dspy.OutputField(desc="Concise summary of the text")
    key_findings = dspy.OutputField(desc="Key points from the text")
    implications = dspy.OutputField(desc="Implications or applications")


class StructuredSummarization(dspy.Signature):
    """Signature for structured summarization."""

    text = dspy.InputField(desc="Text to summarize")
    background = dspy.OutputField(desc="Background information")
    methods = dspy.OutputField(desc="Methods used")
    results = dspy.OutputField(desc="Key results")
    conclusions = dspy.OutputField(desc="Main conclusions")
    implications = dspy.OutputField(desc="Practical implications")


# QA Signatures

class EnhancedQA(dspy.Signature):
    """Enhanced signature for question answering."""

    question = dspy.InputField(desc="Question")
    context = dspy.InputField(desc="Context")
    answer = dspy.OutputField(desc="Comprehensive answer")
    evidence_grade = dspy.OutputField(
        desc="Grade of evidence (A: High, B: Moderate, C: Low, D: Very Low)"
    )
    practical_implications = dspy.OutputField(desc="Practical implications")
    confidence = dspy.OutputField(desc="Confidence score (0-1)")


class StructuredReasoning(dspy.Signature):
    """Signature for structured reasoning."""

    case_description = dspy.InputField(desc="Description of the case")
    analysis = dspy.OutputField(
        desc="Prioritized list of potential analyses"
    )
    recommendations = dspy.OutputField(
        desc="Recommended actions"
    )
    reasoning = dspy.OutputField(
        desc="Step-by-step reasoning process"
    )
    confidence = dspy.OutputField(desc="Confidence score (0-1)")


# Evaluation Metrics Signatures

class QAAccuracy(dspy.Signature):
    """Signature for evaluating QA accuracy."""

    question = dspy.InputField(desc="Question")
    reference_answer = dspy.InputField(desc="Reference answer")
    generated_answer = dspy.InputField(desc="Generated answer")
    accuracy_score = dspy.OutputField(desc="Accuracy score (0-1)")
    error_analysis = dspy.OutputField(desc="Analysis of any errors or omissions")


class RAGRelevance(dspy.Signature):
    """Signature for evaluating RAG passage relevance."""

    question = dspy.InputField(desc="Question")
    retrieved_passages = dspy.InputField(desc="Retrieved passages")
    relevance_scores = dspy.OutputField(desc="Relevance score for each passage (0-1)")
    overall_relevance = dspy.OutputField(desc="Overall relevance score (0-1)")
    missing_information = dspy.OutputField(desc="Important information that is missing")


class SummarizationQuality(dspy.Signature):
    """Signature for evaluating summarization quality."""

    original_text = dspy.InputField(desc="Original text")
    summary = dspy.InputField(desc="Generated summary")
    completeness = dspy.OutputField(desc="Completeness score (0-1)")
    accuracy = dspy.OutputField(desc="Factual accuracy score (0-1)")
    conciseness = dspy.OutputField(desc="Conciseness score (0-1)")
    overall_quality = dspy.OutputField(desc="Overall quality score (0-1)")
