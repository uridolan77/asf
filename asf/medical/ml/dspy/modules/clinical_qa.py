"""Clinical QA Modules

This module provides specialized DSPy modules for clinical question answering.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple

import dspy

from ..dspy_module_base import MedicalDSPyModule, RAGModule
from ..dspy_signatures import MedicalQA, DiagnosticReasoning

# Set up logging
logger = logging.getLogger(__name__)


class ClinicalQAModule(RAGModule):
    """Module for answering clinical questions based on medical literature.

    This module extends the RAG module with specialized handling for clinical questions,
    including confidence assessment and evidence grading.
    """

    def __init__(
        self,
        retriever: Optional[dspy.Module] = None,
        generator: Optional[dspy.Module] = None,
        evidence_grader: Optional[dspy.Module] = None,
        k: int = 5,
        **kwargs
    ):
        """
        Initialize the Clinical QA module.

        Args:
            retriever: Custom retriever module (optional)
            generator: Custom generator module (optional)
            evidence_grader: Module for grading evidence quality (optional)
            k: Number of passages to retrieve
            **kwargs: Additional arguments for the parent class
        """
        # Create default generator if not provided
        if generator is None:
            generator = dspy.ChainOfThought(MedicalQA)

        super().__init__(retriever=retriever, generator=generator, **kwargs)

        # Create evidence grader signature
        EvidenceGraderSignature = dspy.Signature(
            question=dspy.InputField(desc="Clinical question"),
            answer=dspy.InputField(desc="Generated answer"),
            context=dspy.InputField(desc="Retrieved medical literature"),
            evidence_grade=dspy.OutputField(desc="Grade of evidence (A, B, C, D, or I)"),
            evidence_assessment=dspy.OutputField(desc="Assessment of the evidence quality")
        )

        # Create evidence grader if not provided
        self.evidence_grader = evidence_grader or dspy.ChainOfThought(EvidenceGraderSignature)

    def forward(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        Answer a clinical question.

        Args:
            question: The clinical question to answer
            **kwargs: Additional arguments

        Returns:
            Dict[str, Any]: The clinical QA result
        """
        # Log audit
        self.log_audit("CLINICAL_QA_FORWARD", {"question": question, **kwargs}, {})

        # Sanitize input
        sanitized_question = self.sanitize_input(question)

        # Retrieve relevant passages
        retrieval_result = self.retriever(sanitized_question)

        # Extract passages from retrieval result
        if hasattr(retrieval_result, 'passages'):
            passages = retrieval_result.passages
        else:
            passages = retrieval_result

        # Generate answer using the generator
        try:
            generation_result = self.generator(context=passages, question=sanitized_question)

            # Extract answer and confidence
            if hasattr(generation_result, '__dict__'):
                answer = getattr(generation_result, 'answer', "No answer generated")
                confidence = getattr(generation_result, 'confidence', 0.0)
            elif isinstance(generation_result, dict):
                answer = generation_result.get('answer', "No answer generated")
                confidence = generation_result.get('confidence', 0.0)
            else:
                answer = str(generation_result)
                confidence = 0.0

        except Exception as e:
            logger.error(f"Answer generation failed: {str(e)}")
            answer = f"Error in answer generation: {str(e)}"
            confidence = 0.0

        # Grade the evidence
        try:
            grading_result = self.evidence_grader(
                question=sanitized_question,
                answer=answer,
                context=passages
            )

            if hasattr(grading_result, '__dict__'):
                evidence_grade = getattr(grading_result, 'evidence_grade', "I")
                evidence_assessment = getattr(grading_result, 'evidence_assessment', "Insufficient evidence")
            elif isinstance(grading_result, dict):
                evidence_grade = grading_result.get('evidence_grade', "I")
                evidence_assessment = grading_result.get('evidence_assessment', "Insufficient evidence")
            else:
                evidence_grade = "I"
                evidence_assessment = "Could not assess evidence"

        except Exception as e:
            logger.error(f"Evidence grading failed: {str(e)}")
            evidence_grade = "I"
            evidence_assessment = f"Error in evidence grading: {str(e)}"

        # Sanitize outputs
        sanitized_answer = self.sanitize_output(answer)
        sanitized_evidence_assessment = self.sanitize_output(evidence_assessment)

        # Prepare result
        result = {
            'question': question,
            'answer': sanitized_answer,
            'confidence': confidence,
            'evidence_grade': evidence_grade,
            'evidence_assessment': sanitized_evidence_assessment
        }

        # Log audit
        self.log_audit("CLINICAL_QA_FORWARD", {"question": question, **kwargs}, result)

        return result


class DiagnosticReasoningModule(MedicalDSPyModule):
    """Module for diagnostic reasoning based on clinical cases.

    This module implements a step-by-step diagnostic reasoning process,
    generating differential diagnoses and recommended tests.
    """

    def __init__(
        self,
        model: Optional[dspy.Module] = None,
        knowledge_retriever: Optional[dspy.Module] = None,
        **kwargs
    ):
        """
        Initialize the diagnostic reasoning module.

        Args:
            model: Custom diagnostic reasoning model (optional)
            knowledge_retriever: Module for retrieving medical knowledge (optional)
            **kwargs: Additional arguments for the parent class
        """
        super().__init__(**kwargs)

        # Create default model if not provided
        self.model = model or dspy.ChainOfThought(DiagnosticReasoning)

        # Create knowledge retriever if not provided
        self.knowledge_retriever = knowledge_retriever or dspy.Retrieve(k=5)

    def forward(
        self,
        case_description: str,
        retrieve_knowledge: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform diagnostic reasoning on a clinical case.

        Args:
            case_description: Description of the clinical case
            retrieve_knowledge: Whether to retrieve additional medical knowledge
            **kwargs: Additional arguments

        Returns:
            Dict[str, Any]: Diagnostic reasoning result
        """
        # Log audit
        self.log_audit(
            "DIAGNOSTIC_REASONING_FORWARD",
            {"case_description": case_description, "retrieve_knowledge": retrieve_knowledge, **kwargs},
            {}
        )

        # Sanitize input
        sanitized_case = self.sanitize_input(case_description)

        # Retrieve additional knowledge if requested
        knowledge_context = ""
        if retrieve_knowledge:
            try:
                # Extract key terms for retrieval
                key_terms_signature = dspy.Signature(
                    case=dspy.InputField(desc="Clinical case description"),
                    key_terms=dspy.OutputField(desc="Key medical terms for knowledge retrieval")
                )

                key_terms_extractor = dspy.Predict(key_terms_signature)
                key_terms_result = key_terms_extractor(case=sanitized_case)

                if hasattr(key_terms_result, 'key_terms'):
                    key_terms = key_terms_result.key_terms
                elif isinstance(key_terms_result, dict) and 'key_terms' in key_terms_result:
                    key_terms = key_terms_result['key_terms']
                else:
                    key_terms = sanitized_case

                # Retrieve knowledge
                retrieval_result = self.knowledge_retriever(key_terms)

                if hasattr(retrieval_result, 'passages'):
                    passages = retrieval_result.passages
                else:
                    passages = retrieval_result

                # Format passages
                if isinstance(passages, list):
                    knowledge_context = "\n\n".join([
                        f"Knowledge {i+1}: {p}" if isinstance(p, str) else
                        f"Knowledge {i+1}: {p.get('content', str(p)) if isinstance(p, dict) else str(p)}"
                        for i, p in enumerate(passages)
                    ])
                else:
                    knowledge_context = str(passages)

            except Exception as e:
                logger.error(f"Knowledge retrieval failed: {str(e)}")
                knowledge_context = ""

        # Perform diagnostic reasoning
        try:
            # Combine case with knowledge if available
            if knowledge_context:
                full_case = f"{sanitized_case}\n\nRelevant Medical Knowledge:\n{knowledge_context}"
            else:
                full_case = sanitized_case

            reasoning_result = self.model(case_description=full_case)

            # Extract fields
            if hasattr(reasoning_result, '__dict__'):
                differential_diagnosis = getattr(reasoning_result, 'differential_diagnosis', "No differential diagnosis generated")
                recommended_tests = getattr(reasoning_result, 'recommended_tests', "No tests recommended")
                reasoning = getattr(reasoning_result, 'reasoning', "No reasoning provided")
            elif isinstance(reasoning_result, dict):
                differential_diagnosis = reasoning_result.get('differential_diagnosis', "No differential diagnosis generated")
                recommended_tests = reasoning_result.get('recommended_tests', "No tests recommended")
                reasoning = reasoning_result.get('reasoning', "No reasoning provided")
            else:
                differential_diagnosis = "Error in diagnostic reasoning"
                recommended_tests = "Error in test recommendations"
                reasoning = "Error in reasoning process"

        except Exception as e:
            logger.error(f"Diagnostic reasoning failed: {str(e)}")
            differential_diagnosis = f"Error in diagnostic reasoning: {str(e)}"
            recommended_tests = "Could not generate test recommendations due to an error"
            reasoning = "Could not generate reasoning process due to an error"

        # Sanitize outputs
        sanitized_differential = self.sanitize_output(differential_diagnosis)
        sanitized_tests = self.sanitize_output(recommended_tests)
        sanitized_reasoning = self.sanitize_output(reasoning)

        # Prepare result
        result = {
            'case_description': case_description,
            'differential_diagnosis': sanitized_differential,
            'recommended_tests': sanitized_tests,
            'reasoning': sanitized_reasoning,
            'knowledge_retrieved': bool(knowledge_context)
        }

        # Log audit
        self.log_audit(
            "DIAGNOSTIC_REASONING_FORWARD",
            {"case_description": case_description, "retrieve_knowledge": retrieve_knowledge, **kwargs},
            result
        )

        return result


class ClinicalGuidelineQAModule(MedicalDSPyModule):
    """Module for answering questions based on clinical guidelines.

    This module specializes in retrieving and interpreting information from
    clinical practice guidelines to answer specific clinical questions.
    """

    def __init__(
        self,
        guideline_retriever: Optional[dspy.Module] = None,
        guideline_interpreter: Optional[dspy.Module] = None,
        **kwargs
    ):
        """
        Initialize the clinical guideline QA module.

        Args:
            guideline_retriever: Module for retrieving guideline information (optional)
            guideline_interpreter: Module for interpreting guidelines (optional)
            **kwargs: Additional arguments for the parent class
        """
        super().__init__(**kwargs)

        # Create default retriever if not provided
        self.guideline_retriever = guideline_retriever or dspy.Retrieve(k=3)

        # Create guideline interpreter signature
        GuidelineInterpreterSignature = dspy.Signature(
            question=dspy.InputField(desc="Clinical question about guidelines"),
            guidelines=dspy.InputField(desc="Retrieved guideline information"),
            answer=dspy.OutputField(desc="Answer based on guidelines"),
            recommendation_strength=dspy.OutputField(desc="Strength of recommendation (Strong, Moderate, Weak, None)"),
            evidence_level=dspy.OutputField(desc="Level of evidence (I, II, III, IV)"),
            guideline_source=dspy.OutputField(desc="Source of the guideline information")
        )

        # Create default interpreter if not provided
        self.guideline_interpreter = guideline_interpreter or dspy.ChainOfThought(GuidelineInterpreterSignature)

    def forward(
        self,
        question: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Answer a question based on clinical guidelines.

        Args:
            question: Clinical question about guidelines
            **kwargs: Additional arguments

        Returns:
            Dict[str, Any]: Guideline-based answer
        """
        # Log audit
        self.log_audit("CLINICAL_GUIDELINE_QA_FORWARD", {"question": question, **kwargs}, {})

        # Sanitize input
        sanitized_question = self.sanitize_input(question)

        # Retrieve guideline information
        try:
            retrieval_result = self.guideline_retriever(sanitized_question)

            if hasattr(retrieval_result, 'passages'):
                guidelines = retrieval_result.passages
            else:
                guidelines = retrieval_result

            # Format guidelines
            if isinstance(guidelines, list):
                formatted_guidelines = "\n\n".join([
                    f"Guideline {i+1}: {g}" if isinstance(g, str) else
                    f"Guideline {i+1}: {g.get('content', str(g)) if isinstance(g, dict) else str(g)}"
                    for i, g in enumerate(guidelines)
                ])
            else:
                formatted_guidelines = str(guidelines)

        except Exception as e:
            logger.error(f"Guideline retrieval failed: {str(e)}")
            formatted_guidelines = f"Error in guideline retrieval: {str(e)}"

        # Interpret guidelines
        try:
            interpretation_result = self.guideline_interpreter(
                question=sanitized_question,
                guidelines=formatted_guidelines
            )

            # Extract fields
            if hasattr(interpretation_result, '__dict__'):
                answer = getattr(interpretation_result, 'answer', "No answer generated")
                recommendation_strength = getattr(interpretation_result, 'recommendation_strength', "None")
                evidence_level = getattr(interpretation_result, 'evidence_level', "IV")
                guideline_source = getattr(interpretation_result, 'guideline_source', "Unknown")
            elif isinstance(interpretation_result, dict):
                answer = interpretation_result.get('answer', "No answer generated")
                recommendation_strength = interpretation_result.get('recommendation_strength', "None")
                evidence_level = interpretation_result.get('evidence_level', "IV")
                guideline_source = interpretation_result.get('guideline_source', "Unknown")
            else:
                answer = str(interpretation_result)
                recommendation_strength = "None"
                evidence_level = "IV"
                guideline_source = "Unknown"

        except Exception as e:
            logger.error(f"Guideline interpretation failed: {str(e)}")
            answer = f"Error in guideline interpretation: {str(e)}"
            recommendation_strength = "None"
            evidence_level = "IV"
            guideline_source = "Unknown"

        # Sanitize outputs
        sanitized_answer = self.sanitize_output(answer)
        sanitized_guideline_source = self.sanitize_output(guideline_source)

        # Prepare result
        result = {
            'question': question,
            'answer': sanitized_answer,
            'recommendation_strength': recommendation_strength,
            'evidence_level': evidence_level,
            'guideline_source': sanitized_guideline_source
        }

        # Log audit
        self.log_audit("CLINICAL_GUIDELINE_QA_FORWARD", {"question": question, **kwargs}, result)

        return result
