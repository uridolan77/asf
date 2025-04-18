"""Advanced QA Modules

This module provides specialized DSPy modules for advanced question answering.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple

import dspy

from .base import DSPyModuleBase, RAGModule
from ..templates.enhanced import EnhancedQA, StructuredReasoning

# Set up logging
logger = logging.getLogger(__name__)


class AdvancedQAModule(RAGModule):
    """Module for answering complex questions with evidence assessment.

    This module extends the RAG module with specialized handling for complex questions,
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
        Initialize the Advanced QA module.

        Args:
            retriever: Custom retriever module (optional)
            generator: Custom generator module (optional)
            evidence_grader: Module for grading evidence quality (optional)
            k: Number of passages to retrieve
            **kwargs: Additional arguments for the parent class
        """
        # Create default generator if not provided
        if generator is None:
            generator = dspy.ChainOfThought(EnhancedQA)

        super().__init__(retriever=retriever, generator=generator, **kwargs)

        # Create evidence grader signature
        EvidenceGraderSignature = dspy.Signature(
            question=dspy.InputField(desc="Question"),
            answer=dspy.InputField(desc="Generated answer"),
            context=dspy.InputField(desc="Retrieved context"),
            evidence_grade=dspy.OutputField(desc="Grade of evidence (A, B, C, D, or I)"),
            evidence_assessment=dspy.OutputField(desc="Assessment of the evidence quality")
        )

        # Create evidence grader if not provided
        self.evidence_grader = evidence_grader or dspy.ChainOfThought(EvidenceGraderSignature)

    def forward(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        Answer a complex question.

        Args:
            question: The question to answer
            **kwargs: Additional arguments

        Returns:
            Dict[str, Any]: The QA result
        """
        # Log audit
        self.log_audit("ADVANCED_QA_FORWARD", {"question": question, **kwargs}, {})

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
        self.log_audit("ADVANCED_QA_FORWARD", {"question": question, **kwargs}, result)

        return result


class StructuredReasoningModule(DSPyModuleBase):
    """Module for structured reasoning based on complex cases.

    This module implements a step-by-step reasoning process,
    generating structured analysis and recommendations.
    """

    def __init__(
        self,
        model: Optional[dspy.Module] = None,
        knowledge_retriever: Optional[dspy.Module] = None,
        **kwargs
    ):
        """
        Initialize the structured reasoning module.

        Args:
            model: Custom reasoning model (optional)
            knowledge_retriever: Module for retrieving knowledge (optional)
            **kwargs: Additional arguments for the parent class
        """
        super().__init__(**kwargs)

        # Create default model if not provided
        self.model = model or dspy.ChainOfThought(StructuredReasoning)

        # Create knowledge retriever if not provided
        self.knowledge_retriever = knowledge_retriever or dspy.Retrieve(k=5)

    def forward(
        self,
        case_description: str,
        retrieve_knowledge: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform structured reasoning on a complex case.

        Args:
            case_description: Description of the case
            retrieve_knowledge: Whether to retrieve additional knowledge
            **kwargs: Additional arguments

        Returns:
            Dict[str, Any]: Structured reasoning result
        """
        # Log audit
        self.log_audit(
            "STRUCTURED_REASONING_FORWARD",
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
                    case=dspy.InputField(desc="Case description"),
                    key_terms=dspy.OutputField(desc="Key terms for knowledge retrieval")
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

        # Perform structured reasoning
        try:
            # Combine case with knowledge if available
            if knowledge_context:
                full_case = f"{sanitized_case}\n\nRelevant Knowledge:\n{knowledge_context}"
            else:
                full_case = sanitized_case

            reasoning_result = self.model(case_description=full_case)

            # Extract fields
            if hasattr(reasoning_result, '__dict__'):
                analysis = getattr(reasoning_result, 'analysis', "No analysis generated")
                recommendations = getattr(reasoning_result, 'recommendations', "No recommendations provided")
                reasoning = getattr(reasoning_result, 'reasoning', "No reasoning provided")
            elif isinstance(reasoning_result, dict):
                analysis = reasoning_result.get('analysis', "No analysis generated")
                recommendations = reasoning_result.get('recommendations', "No recommendations provided")
                reasoning = reasoning_result.get('reasoning', "No reasoning provided")
            else:
                analysis = "Error in structured reasoning"
                recommendations = "Error in recommendations"
                reasoning = "Error in reasoning process"

        except Exception as e:
            logger.error(f"Structured reasoning failed: {str(e)}")
            analysis = f"Error in structured reasoning: {str(e)}"
            recommendations = "Could not generate recommendations due to an error"
            reasoning = "Could not generate reasoning process due to an error"

        # Sanitize outputs
        sanitized_analysis = self.sanitize_output(analysis)
        sanitized_recommendations = self.sanitize_output(recommendations)
        sanitized_reasoning = self.sanitize_output(reasoning)

        # Prepare result
        result = {
            'case_description': case_description,
            'analysis': sanitized_analysis,
            'recommendations': sanitized_recommendations,
            'reasoning': sanitized_reasoning,
            'knowledge_retrieved': bool(knowledge_context)
        }

        # Log audit
        self.log_audit(
            "STRUCTURED_REASONING_FORWARD",
            {"case_description": case_description, "retrieve_knowledge": retrieve_knowledge, **kwargs},
            result
        )

        return result


class GuidelineQAModule(DSPyModuleBase):
    """Module for answering questions based on guidelines.

    This module specializes in retrieving and interpreting information from
    guidelines to answer specific questions.
    """

    def __init__(
        self,
        guideline_retriever: Optional[dspy.Module] = None,
        guideline_interpreter: Optional[dspy.Module] = None,
        **kwargs
    ):
        """
        Initialize the guideline QA module.

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
            question=dspy.InputField(desc="Question about guidelines"),
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
        Answer a question based on guidelines.

        Args:
            question: Question about guidelines
            **kwargs: Additional arguments

        Returns:
            Dict[str, Any]: Guideline-based answer
        """
        # Log audit
        self.log_audit("GUIDELINE_QA_FORWARD", {"question": question, **kwargs}, {})

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
        self.log_audit("GUIDELINE_QA_FORWARD", {"question": question, **kwargs}, result)

        return result
