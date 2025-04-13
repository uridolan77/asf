Medical DSPy Modules

This module provides medical-specific DSPy modules for medical research applications.

import logging
import time
from typing import Dict, Any, List, Optional, Union, Tuple

import dspy

from .enhanced_module_base import EnhancedDSPyModuleBase, EnhancedRAGModule, EnhancedChainOfThoughtModule
from .enhanced_signatures import (
    EnhancedMedicalRAG,
    MedicalEvidenceRAG,
    EnhancedContradictionDetection,
    EnhancedTemporalContradictionDetection,
    EnhancedEvidenceExtraction,
    MultiEvidenceExtraction,
    EnhancedMedicalSummarization,
    StructuredMedicalSummarization,
    EnhancedClinicalQA,
    DiagnosticReasoning
)
from .modules.diagnostic_reasoning import SpecialistConsultModule

# Set up logging
logger = logging.getLogger(__name__)


class MedicalRAG(EnhancedRAGModule):
    """Medical RAG module for answering medical questions with citations.

    This module combines retrieval and generation to answer medical questions
    with proper citations and confidence scores.
    """

    def __init__(
        self,
        retriever: Optional[dspy.Module] = None,
        max_passages: int = 5,
        confidence_threshold: float = 0.7,
        **kwargs
    ):
        """
        Initialize the Medical RAG module.

        Args:
            retriever: Retriever module
            max_passages: Maximum number of passages to retrieve
            confidence_threshold: Threshold for confidence scores
            **kwargs: Additional arguments
        """
        super().__init__(retriever=retriever, **kwargs)
        self.max_passages = max_passages
        self.confidence_threshold = confidence_threshold

        # Create predictor
        self.predictor = dspy.Predict(EnhancedMedicalRAG)

    async def forward(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        Answer a medical question with citations.

        Args:
            question: Medical question to answer
            **kwargs: Additional arguments

        Returns:
            Dict[str, Any]: Answer with citations and confidence
        """
        # Sanitize input
        question = self.sanitize_input(question)

        # Retrieve passages
        if self.retriever:
            retrieval_result = await self.retriever(question=question, k=self.max_passages)
            passages = retrieval_result.passages if hasattr(retrieval_result, 'passages') else []
        else:
            passages = kwargs.get('passages', [])

        # Generate answer
        result = self.predictor(question=question, passages=passages)

        # Sanitize output
        result.answer = self.sanitize_output(result.answer)
        result.medical_context = self.sanitize_output(result.medical_context)
        result.limitations = self.sanitize_output(result.limitations)

        # Convert to dictionary
        output = {
            "question": question,
            "answer": result.answer,
            "citations": result.citations,
            "confidence": float(result.confidence),
            "medical_context": result.medical_context,
            "limitations": result.limitations,
            "passages": passages
        }

        # Log audit event
        self.log_audit(
            event_type="medical_rag_query",
            inputs={"question": question},
            outputs=output
        )

        return output


class MedicalContradictionDetector(EnhancedDSPyModuleBase):
    """Medical contradiction detector for identifying contradictions in medical statements.

    This module identifies contradictions between medical statements and provides
    detailed explanations with confidence scores.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        **kwargs
    ):
        """
        Initialize the Medical Contradiction Detector.

        Args:
            confidence_threshold: Threshold for confidence scores
            **kwargs: Additional arguments
        """
        super().__init__()
        self.confidence_threshold = confidence_threshold

        # Create predictor
        self.predictor = dspy.Predict(EnhancedContradictionDetection)

    def forward(self, statement1: str, statement2: str, **kwargs) -> Dict[str, Any]:
        """
        Detect contradictions between medical statements.

        Args:
            statement1: First medical statement
            statement2: Second medical statement
            **kwargs: Additional arguments

        Returns:
            Dict[str, Any]: Contradiction detection result
        """
        # Sanitize input
        statement1 = self.sanitize_input(statement1)
        statement2 = self.sanitize_input(statement2)

        # Detect contradiction
        result = self.predictor(statement1=statement1, statement2=statement2)

        # Parse contradiction result (convert string to boolean)
        contradiction = False
        if isinstance(result.contradiction, str):
            contradiction = result.contradiction.lower() in ['true', 'yes', '1']
        else:
            contradiction = bool(result.contradiction)

        # Sanitize output
        result.explanation = self.sanitize_output(result.explanation)
        result.medical_context = self.sanitize_output(result.medical_context)

        # Convert to dictionary
        output = {
            "statement1": statement1,
            "statement2": statement2,
            "contradiction": contradiction,
            "explanation": result.explanation,
            "confidence": float(result.confidence),
            "medical_context": result.medical_context
        }

        # Log audit event
        self.log_audit(
            event_type="contradiction_detection",
            inputs={"statement1": statement1, "statement2": statement2},
            outputs=output
        )

        return output


class TemporalContradictionDetector(EnhancedDSPyModuleBase):
    """Temporal contradiction detector for identifying contradictions in medical statements over time.

    This module identifies contradictions between medical statements with timestamps
    and provides detailed explanations of the temporal relationships.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        **kwargs
    ):
        """
        Initialize the Temporal Contradiction Detector.

        Args:
            confidence_threshold: Threshold for confidence scores
            **kwargs: Additional arguments
        """
        super().__init__()
        self.confidence_threshold = confidence_threshold

        # Create predictor
        self.predictor = dspy.Predict(EnhancedTemporalContradictionDetection)

    def forward(
        self,
        statement1: str,
        timestamp1: str,
        statement2: str,
        timestamp2: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Detect temporal contradictions between medical statements.

        Args:
            statement1: First medical statement
            timestamp1: Timestamp of the first statement
            statement2: Second medical statement
            timestamp2: Timestamp of the second statement
            **kwargs: Additional arguments

        Returns:
            Dict[str, Any]: Temporal contradiction detection result
        """
        # Sanitize input
        statement1 = self.sanitize_input(statement1)
        statement2 = self.sanitize_input(statement2)
        timestamp1 = self.sanitize_input(timestamp1)
        timestamp2 = self.sanitize_input(timestamp2)

        # Detect contradiction
        result = self.predictor(
            statement1=statement1,
            timestamp1=timestamp1,
            statement2=statement2,
            timestamp2=timestamp2
        )

        # Parse contradiction result (convert string to boolean)
        contradiction = False
        if isinstance(result.contradiction, str):
            contradiction = result.contradiction.lower() in ['true', 'yes', '1']
        else:
            contradiction = bool(result.contradiction)

        # Sanitize output
        result.explanation = self.sanitize_output(result.explanation)
        result.temporal_relation = self.sanitize_output(result.temporal_relation)

        # Convert to dictionary
        output = {
            "statement1": statement1,
            "timestamp1": timestamp1,
            "statement2": statement2,
            "timestamp2": timestamp2,
            "contradiction": contradiction,
            "temporal_relation": result.temporal_relation,
            "explanation": result.explanation,
            "confidence": float(result.confidence)
        }

        # Log audit event
        self.log_audit(
            event_type="temporal_contradiction_detection",
            inputs={
                "statement1": statement1,
                "timestamp1": timestamp1,
                "statement2": statement2,
                "timestamp2": timestamp2
            },
            outputs=output
        )

        return output


class MedicalEvidenceExtractor(EnhancedDSPyModuleBase):
    """Medical evidence extractor for extracting evidence from medical texts.

    This module extracts evidence from medical texts and determines the relationship
    between the evidence and a given claim.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        multi_evidence: bool = False,
        **kwargs
    ):
        """
        Initialize the Medical Evidence Extractor.

        Args:
            confidence_threshold: Threshold for confidence scores
            multi_evidence: Whether to extract multiple pieces of evidence
            **kwargs: Additional arguments
        """
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.multi_evidence = multi_evidence

        # Create predictor based on mode
        if multi_evidence:
            self.predictor = dspy.Predict(MultiEvidenceExtraction)
        else:
            self.predictor = dspy.Predict(EnhancedEvidenceExtraction)

    def forward(self, text: str, claim: str, **kwargs) -> Dict[str, Any]:
        """
        Extract evidence from medical text.

        Args:
            text: Medical text to analyze
            claim: The claim to find evidence for
            **kwargs: Additional arguments

        Returns:
            Dict[str, Any]: Evidence extraction result
        """
        # Sanitize input
        text = self.sanitize_input(text)
        claim = self.sanitize_input(claim)

        # Extract evidence
        result = self.predictor(text=text, claim=claim)

        # Process result based on mode
        if self.multi_evidence:
            # Sanitize output
            evidence_pieces = []
            if isinstance(result.evidence_pieces, str):
                # Try to parse as list if it's a string
                import json
                try:
                    evidence_pieces = json.loads(result.evidence_pieces)
                except json.JSONDecodeError:
                    evidence_pieces = [result.evidence_pieces]
            else:
                evidence_pieces = result.evidence_pieces

            # Sanitize each piece
            evidence_pieces = [self.sanitize_output(piece) for piece in evidence_pieces]

            # Convert to dictionary
            output = {
                "text": text[:100] + "..." if len(text) > 100 else text,  # Truncate for logging
                "claim": claim,
                "evidence_pieces": evidence_pieces,
                "overall_relation": result.overall_relation,
                "confidence": float(result.confidence)
            }
        else:
            # Sanitize output
            result.evidence = self.sanitize_output(result.evidence)
            result.reasoning = self.sanitize_output(result.reasoning)

            # Convert to dictionary
            output = {
                "text": text[:100] + "..." if len(text) > 100 else text,  # Truncate for logging
                "claim": claim,
                "evidence": result.evidence,
                "relation": result.relation,
                "confidence": float(result.confidence),
                "reasoning": result.reasoning
            }

        # Log audit event
        self.log_audit(
            event_type="evidence_extraction",
            inputs={"text": text[:100] + "..." if len(text) > 100 else text, "claim": claim},
            outputs=output
        )

        return output


class MedicalSummarizer(EnhancedDSPyModuleBase):
    """Medical summarizer for summarizing medical texts.

    This module summarizes medical texts with key findings and implications,
    tailored to different audiences.
    """

    def __init__(
        self,
        structured: bool = False,
        **kwargs
    ):
        """
        Initialize the Medical Summarizer.

        Args:
            structured: Whether to use structured summarization
            **kwargs: Additional arguments
        """
        super().__init__()
        self.structured = structured

        # Create predictor based on mode
        if structured:
            self.predictor = dspy.Predict(StructuredMedicalSummarization)
        else:
            self.predictor = dspy.Predict(EnhancedMedicalSummarization)

    def forward(
        self,
        text: str,
        audience: str = "clinician",
        content_type: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Summarize medical text.

        Args:
            text: Medical text to summarize
            audience: Target audience (e.g., 'clinician', 'researcher', 'patient')
            content_type: Type of content (required for structured mode)
            **kwargs: Additional arguments

        Returns:
            Dict[str, Any]: Summarization result
        """
        # Sanitize input
        text = self.sanitize_input(text)
        audience = self.sanitize_input(audience)

        # Process based on mode
        if self.structured:
            if not content_type:
                content_type = "research_paper"  # Default
            content_type = self.sanitize_input(content_type)

            # Generate structured summary
            result = self.predictor(text=text, audience=audience, content_type=content_type)

            # Sanitize output
            result.background = self.sanitize_output(result.background)
            result.methods = self.sanitize_output(result.methods)
            result.results = self.sanitize_output(result.results)
            result.conclusions = self.sanitize_output(result.conclusions)
            result.limitations = self.sanitize_output(result.limitations)

            # Convert to dictionary
            output = {
                "text": text[:100] + "..." if len(text) > 100 else text,  # Truncate for logging
                "audience": audience,
                "content_type": content_type,
                "background": result.background,
                "methods": result.methods,
                "results": result.results,
                "conclusions": result.conclusions,
                "limitations": result.limitations
            }
        else:
            # Generate summary
            result = self.predictor(text=text, audience=audience)

            # Sanitize output
            result.summary = self.sanitize_output(result.summary)
            result.key_findings = self.sanitize_output(result.key_findings)
            result.implications = self.sanitize_output(result.implications)
            result.limitations = self.sanitize_output(result.limitations)

            # Convert to dictionary
            output = {
                "text": text[:100] + "..." if len(text) > 100 else text,  # Truncate for logging
                "audience": audience,
                "summary": result.summary,
                "key_findings": result.key_findings,
                "implications": result.implications,
                "limitations": result.limitations
            }

        # Log audit event
        self.log_audit(
            event_type="medical_summarization",
            inputs={"text": text[:100] + "..." if len(text) > 100 else text, "audience": audience},
            outputs=output
        )

        return output


class ClinicalQA(EnhancedDSPyModuleBase):
    """Clinical QA module for answering clinical questions.

    This module answers clinical questions with evidence grading and clinical implications.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        **kwargs
    ):
        """
        Initialize the Clinical QA module.

        Args:
            confidence_threshold: Threshold for confidence scores
            **kwargs: Additional arguments
        """
        super().__init__()
        self.confidence_threshold = confidence_threshold

        # Create predictor
        self.predictor = dspy.Predict(EnhancedClinicalQA)

    def forward(self, question: str, context: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Answer a clinical question.

        Args:
            question: Clinical question
            context: Medical context or literature
            **kwargs: Additional arguments

        Returns:
            Dict[str, Any]: Clinical answer with evidence grading
        """
        # Sanitize input
        question = self.sanitize_input(question)
        context = self.sanitize_input(context) if context else ""

        # Generate answer
        result = self.predictor(question=question, context=context)

        # Sanitize output
        result.answer = self.sanitize_output(result.answer)
        result.clinical_implications = self.sanitize_output(result.clinical_implications)
        result.limitations = self.sanitize_output(result.limitations)

        # Convert to dictionary
        output = {
            "question": question,
            "answer": result.answer,
            "confidence": float(result.confidence),
            "evidence_grade": result.evidence_grade,
            "clinical_implications": result.clinical_implications,
            "limitations": result.limitations
        }

        # Log audit event
        self.log_audit(
            event_type="clinical_qa",
            inputs={"question": question, "context": context[:100] + "..." if len(context) > 100 else context},
            outputs=output
        )

        return output


class DiagnosticReasoningModule(EnhancedChainOfThoughtModule):
    """Diagnostic reasoning module for clinical case analysis.

    This module performs step-by-step diagnostic reasoning on clinical cases.
    """

    def __init__(self, **kwargs):
        """
        Initialize the Diagnostic Reasoning module.

        Args:
            **kwargs: Additional arguments
        """
        super().__init__(signature=DiagnosticReasoning, **kwargs)

    def forward(self, case_description: str, **kwargs) -> Dict[str, Any]:
        """
        Perform diagnostic reasoning on a clinical case.

        Args:
            case_description: Clinical case description
            **kwargs: Additional arguments

        Returns:
            Dict[str, Any]: Diagnostic reasoning result
        """
        # Sanitize input
        case_description = self.sanitize_input(case_description)

        # Perform diagnostic reasoning
        result = self.cot(case_description=case_description)

        # Sanitize output
        result.differential_diagnosis = self.sanitize_output(result.differential_diagnosis)
        result.recommended_tests = self.sanitize_output(result.recommended_tests)
        result.reasoning = self.sanitize_output(result.reasoning)

        # Convert to dictionary
        output = {
            "case_description": case_description[:100] + "..." if len(case_description) > 100 else case_description,
            "differential_diagnosis": result.differential_diagnosis,
            "recommended_tests": result.recommended_tests,
            "reasoning": result.reasoning,
            "confidence": float(result.confidence)
        }

        # Log audit event
        self.log_audit(
            event_type="diagnostic_reasoning",
            inputs={"case_description": case_description[:100] + "..." if len(case_description) > 100 else case_description},
            outputs=output
        )

        return output


# Export all classes
__all__ = [
    'MedicalRAG',
    'MedicalContradictionDetector',
    'TemporalContradictionDetector',
    'MedicalEvidenceExtractor',
    'MedicalSummarizer',
    'ClinicalQA',
    'DiagnosticReasoningModule',
    'SpecialistConsultModule'
]
