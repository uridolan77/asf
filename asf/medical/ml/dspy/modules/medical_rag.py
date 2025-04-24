"""Medical RAG Modules

This module provides specialized RAG (Retrieval-Augmented Generation) modules for medical research.
"""

import logging
from typing import Dict, Any, List, Optional, Union

import dspy
from dspy.retrieve import Retrieve

from ..dspy_module_base import RAGModule, MedicalDSPyModule
from ..dspy_signatures import MedicalRAG

# Set up logging
logger = logging.getLogger(__name__)


class MedicalRAGModule(RAGModule):
    """Medical RAG module for retrieving and generating medical information.

    This module implements a standard RAG pipeline for medical questions,
    with specialized handling for medical context and citations.
    """

    def __init__(
        self,
        retriever: Optional[dspy.Module] = None,
        generator: Optional[dspy.Module] = None,
        k: int = 5,
        **kwargs
    ):
        """
        Initialize the Medical RAG module.

        Args:
            retriever: Custom retriever module (optional)
            generator: Custom generator module (optional)
            k: Number of passages to retrieve
            **kwargs: Additional arguments for the parent class
        """
        # Create default retriever and generator if not provided
        if retriever is None:
            retriever = Retrieve(k=k)

        if generator is None:
            generator = dspy.ChainOfThought(MedicalRAG)

        super().__init__(retriever=retriever, generator=generator, **kwargs)

    def forward(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        Execute the Medical RAG pipeline.

        Args:
            question: The medical question to answer
            **kwargs: Additional arguments

        Returns:
            Dict[str, Any]: The RAG output with answer and citations
        """
        # Log audit
        self.log_audit("MEDICAL_RAG_FORWARD", {"question": question, **kwargs}, {})

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
            generation_result = self.generator(question=sanitized_question, passages=passages)
        except Exception as e:
            logger.error(f"Error in generator: {str(e)}")
            # Fallback to a simpler generation approach
            generation_result = dspy.Predict(MedicalRAG)(question=sanitized_question, passages=passages)

        # Extract answer and citations
        if hasattr(generation_result, 'answer'):
            answer = generation_result.answer
        elif isinstance(generation_result, dict) and 'answer' in generation_result:
            answer = generation_result['answer']
        else:
            answer = str(generation_result)

        if hasattr(generation_result, 'citations'):
            citations = generation_result.citations
        elif isinstance(generation_result, dict) and 'citations' in generation_result:
            citations = generation_result['citations']
        else:
            citations = "No citations provided"

        # Sanitize output
        sanitized_answer = self.sanitize_output(answer)
        sanitized_citations = self.sanitize_output(citations)

        # Prepare result
        result = {
            'question': question,
            'passages': passages,
            'answer': sanitized_answer,
            'citations': sanitized_citations
        }

        # Log audit
        self.log_audit("MEDICAL_RAG_FORWARD", {"question": question, **kwargs}, result)

        return result


class EnhancedMedicalRAGModule(MedicalDSPyModule):
    """Enhanced Medical RAG module with multi-stage processing.

    This module implements a more sophisticated RAG pipeline with:
    1. Initial retrieval
    2. Query expansion
    3. Secondary retrieval
    4. Answer generation with citations
    5. Fact checking
    """

    def __init__(
        self,
        initial_retriever: Optional[dspy.Module] = None,
        query_expander: Optional[dspy.Module] = None,
        secondary_retriever: Optional[dspy.Module] = None,
        answer_generator: Optional[dspy.Module] = None,
        fact_checker: Optional[dspy.Module] = None,
        initial_k: int = 3,
        secondary_k: int = 5,
        **kwargs
    ):
        """
        Initialize the Enhanced Medical RAG module.

        Args:
            initial_retriever: Initial retrieval module
            query_expander: Query expansion module
            secondary_retriever: Secondary retrieval module
            answer_generator: Answer generation module
            fact_checker: Fact checking module
            initial_k: Number of passages for initial retrieval
            secondary_k: Number of passages for secondary retrieval
            **kwargs: Additional arguments for the parent class
        """
        super().__init__(**kwargs)

        # Initialize components with defaults if not provided
        self.initial_retriever = initial_retriever or Retrieve(k=initial_k)

        self.query_expander = query_expander or dspy.ChainOfThought(
            dspy.Signature(
                question=dspy.InputField(desc="Original medical question"),
                context=dspy.InputField(desc="Initial retrieved passages"),
                expanded_query=dspy.OutputField(desc="Expanded query with additional medical terms and context")
            )
        )

        self.secondary_retriever = secondary_retriever or Retrieve(k=secondary_k)

        self.answer_generator = answer_generator or dspy.ChainOfThought(
            dspy.Signature(
                question=dspy.InputField(desc="Medical question"),
                passages=dspy.InputField(desc="Retrieved medical passages"),
                answer=dspy.OutputField(desc="Comprehensive medical answer"),
                citations=dspy.OutputField(desc="Citations for medical claims"),
                confidence=dspy.OutputField(desc="Confidence score (0-1)")
            )
        )

        self.fact_checker = fact_checker or dspy.ChainOfThought(
            dspy.Signature(
                answer=dspy.InputField(desc="Generated medical answer"),
                passages=dspy.InputField(desc="Source passages"),
                verified_answer=dspy.OutputField(desc="Verified answer with corrections if needed"),
                verification_notes=dspy.OutputField(desc="Notes on the verification process")
            )
        )

    def forward(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        Execute the Enhanced Medical RAG pipeline.

        Args:
            question: The medical question to answer
            **kwargs: Additional arguments

        Returns:
            Dict[str, Any]: The enhanced RAG output
        """
        # Log audit
        self.log_audit("ENHANCED_MEDICAL_RAG_FORWARD", {"question": question, **kwargs}, {})

        # Sanitize input
        sanitized_question = self.sanitize_input(question)

        # Step 1: Initial retrieval
        initial_result = self.initial_retriever(sanitized_question)
        initial_passages = initial_result.passages if hasattr(initial_result, 'passages') else initial_result

        # Step 2: Query expansion
        try:
            expansion_result = self.query_expander(question=sanitized_question, context=initial_passages)
            if hasattr(expansion_result, 'expanded_query'):
                expanded_query = expansion_result.expanded_query
            elif isinstance(expansion_result, dict) and 'expanded_query' in expansion_result:
                expanded_query = expansion_result['expanded_query']
            else:
                expanded_query = sanitized_question
        except Exception as e:
            logger.warning(f"Query expansion failed: {str(e)}. Using original query.")
            expanded_query = sanitized_question

        # Step 3: Secondary retrieval with expanded query
        secondary_result = self.secondary_retriever(expanded_query)
        secondary_passages = secondary_result.passages if hasattr(secondary_result, 'passages') else secondary_result

        # Combine passages, removing duplicates
        all_passages = self._combine_passages(initial_passages, secondary_passages)

        # Step 4: Generate answer
        try:
            generation_result = self.answer_generator(question=sanitized_question, passages=all_passages)

            # Extract fields
            if hasattr(generation_result, '__dict__'):
                answer = getattr(generation_result, 'answer', "No answer generated")
                citations = getattr(generation_result, 'citations', "No citations provided")
                confidence = getattr(generation_result, 'confidence', 0.0)
            elif isinstance(generation_result, dict):
                answer = generation_result.get('answer', "No answer generated")
                citations = generation_result.get('citations', "No citations provided")
                confidence = generation_result.get('confidence', 0.0)
            else:
                answer = str(generation_result)
                citations = "No citations provided"
                confidence = 0.0
        except Exception as e:
            logger.error(f"Answer generation failed: {str(e)}. Using fallback.")
            answer = "Unable to generate a complete answer due to a processing error."
            citations = "No citations available due to processing error."
            confidence = 0.0

        # Step 5: Fact checking
        try:
            fact_check_result = self.fact_checker(answer=answer, passages=all_passages)

            if hasattr(fact_check_result, 'verified_answer'):
                verified_answer = fact_check_result.verified_answer
                verification_notes = getattr(fact_check_result, 'verification_notes', "")
            elif isinstance(fact_check_result, dict):
                verified_answer = fact_check_result.get('verified_answer', answer)
                verification_notes = fact_check_result.get('verification_notes', "")
            else:
                verified_answer = answer
                verification_notes = "Fact checking did not return a structured result."
        except Exception as e:
            logger.error(f"Fact checking failed: {str(e)}. Using unverified answer.")
            verified_answer = answer
            verification_notes = "Fact checking could not be completed due to an error."

        # Sanitize outputs
        sanitized_answer = self.sanitize_output(verified_answer)
        sanitized_citations = self.sanitize_output(citations)
        sanitized_verification_notes = self.sanitize_output(verification_notes)

        # Prepare result
        result = {
            'question': question,
            'expanded_query': expanded_query,
            'passages': all_passages,
            'answer': sanitized_answer,
            'citations': sanitized_citations,
            'confidence': confidence,
            'verification_notes': sanitized_verification_notes
        }

        # Log audit
        self.log_audit("ENHANCED_MEDICAL_RAG_FORWARD", {"question": question, **kwargs}, result)

        return result

    def _combine_passages(self, passages1, passages2) -> List[Any]:
        """
        Combine passages from two retrievals, removing duplicates.

        Args:
            passages1: First set of passages
            passages2: Second set of passages

        Returns:
            List[Any]: Combined unique passages
        """
        # Convert to list if needed
        if not isinstance(passages1, list):
            passages1 = [passages1]
        if not isinstance(passages2, list):
            passages2 = [passages2]

        # Track seen passages by content
        seen = set()
        combined = []

        for passage in passages1 + passages2:
            # Get passage content
            if hasattr(passage, 'content'):
                content = passage.content
            elif isinstance(passage, dict) and 'content' in passage:
                content = passage['content']
            elif isinstance(passage, str):
                content = passage
            else:
                content = str(passage)

            # Add if not seen
            content_hash = hash(content)
            if content_hash not in seen:
                seen.add(content_hash)
                combined.append(passage)

        return combined
