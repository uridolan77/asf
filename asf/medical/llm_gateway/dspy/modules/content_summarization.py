"""Content Summarization Modules

This module provides specialized DSPy modules for summarizing content.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple

import dspy

from .base import DSPyModuleBase
from ..templates.enhanced import ContentSummarization, StructuredSummarization

# Set up logging
logger = logging.getLogger(__name__)


class ContentSummarizationModule(DSPyModuleBase):
    """Module for summarizing content with key findings and implications.

    This module generates concise summaries of content, highlighting
    key findings and implications for different audiences.
    """

    def __init__(
        self,
        model: Optional[dspy.Module] = None,
        **kwargs
    ):
        """
        Initialize the content summarization module.

        Args:
            model: Custom summarization model (optional)
            **kwargs: Additional arguments for the parent class
        """
        super().__init__(**kwargs)

        # Create default model if not provided
        self.model = model or dspy.ChainOfThought(ContentSummarization)

    def forward(
        self,
        text: str,
        audience: str = "expert",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Summarize content.

        Args:
            text: Text to summarize
            audience: Target audience (e.g., 'expert', 'researcher', 'general')
            **kwargs: Additional arguments

        Returns:
            Dict[str, Any]: Summarization result
        """
        # Log audit
        self.log_audit(
            "CONTENT_SUMMARIZATION_FORWARD",
            {"text": text, "audience": audience, **kwargs},
            {}
        )

        # Sanitize inputs
        sanitized_text = self.sanitize_input(text)

        # Summarize content
        try:
            summarization_result = self.model(
                text=sanitized_text,
                audience=audience
            )

            # Extract fields
            if hasattr(summarization_result, '__dict__'):
                summary = getattr(summarization_result, 'summary', "No summary generated")
                key_findings = getattr(summarization_result, 'key_findings', "No key findings identified")
                implications = getattr(summarization_result, 'implications', "No implications identified")
            elif isinstance(summarization_result, dict):
                summary = summarization_result.get('summary', "No summary generated")
                key_findings = summarization_result.get('key_findings', "No key findings identified")
                implications = summarization_result.get('implications', "No implications identified")
            else:
                summary = str(summarization_result)
                key_findings = "No key findings identified"
                implications = "No implications identified"

        except Exception as e:
            logger.error(f"Content summarization failed: {str(e)}")
            summary = f"Error in summarization: {str(e)}"
            key_findings = "Error in key findings extraction"
            implications = "Error in implications extraction"

        # Sanitize outputs
        sanitized_summary = self.sanitize_output(summary)
        sanitized_key_findings = self.sanitize_output(key_findings)
        sanitized_implications = self.sanitize_output(implications)

        # Prepare result
        result = {
            'text': text[:100] + "..." if len(text) > 100 else text,  # Truncate for logging
            'audience': audience,
            'summary': sanitized_summary,
            'key_findings': sanitized_key_findings,
            'implications': sanitized_implications
        }

        # Log audit
        self.log_audit(
            "CONTENT_SUMMARIZATION_FORWARD",
            {"text": text[:100] + "..." if len(text) > 100 else text, "audience": audience, **kwargs},
            result
        )

        return result


class StructuredContentSummarizationModule(DSPyModuleBase):
    """Module for generating structured summaries of content.

    This module produces summaries with specific sections like background,
    methods, results, and conclusions, tailored to the content type.
    """

    def __init__(
        self,
        model: Optional[dspy.Module] = None,
        **kwargs
    ):
        """
        Initialize the structured content summarization module.

        Args:
            model: Custom structured summarization model (optional)
            **kwargs: Additional arguments for the parent class
        """
        super().__init__(**kwargs)

        # Create a specialized signature for structured summarization
        StructuredContentSummarizationSignature = dspy.Signature(
            text=dspy.InputField(desc="Text to summarize"),
            content_type=dspy.InputField(desc="Type of content (e.g., 'research_paper', 'article', 'report')"),
            audience=dspy.InputField(desc="Target audience (e.g., 'expert', 'researcher', 'general')"),
            background=dspy.OutputField(desc="Background or context"),
            methods=dspy.OutputField(desc="Methods or approach"),
            results=dspy.OutputField(desc="Key results or findings"),
            conclusions=dspy.OutputField(desc="Conclusions and implications"),
            limitations=dspy.OutputField(desc="Limitations or caveats")
        )

        # Create default model if not provided
        self.model = model or dspy.ChainOfThought(StructuredContentSummarizationSignature)

    def forward(
        self,
        text: str,
        content_type: str = "research_paper",
        audience: str = "expert",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a structured summary of content.

        Args:
            text: Text to summarize
            content_type: Type of content (e.g., 'research_paper', 'article', 'report')
            audience: Target audience (e.g., 'expert', 'researcher', 'general')
            **kwargs: Additional arguments

        Returns:
            Dict[str, Any]: Structured summarization result
        """
        # Log audit
        self.log_audit(
            "STRUCTURED_CONTENT_SUMMARIZATION_FORWARD",
            {"text": text[:100] + "..." if len(text) > 100 else text, "content_type": content_type, "audience": audience, **kwargs},
            {}
        )

        # Sanitize inputs
        sanitized_text = self.sanitize_input(text)

        # Generate structured summary
        try:
            summarization_result = self.model(
                text=sanitized_text,
                content_type=content_type,
                audience=audience
            )

            # Extract fields
            if hasattr(summarization_result, '__dict__'):
                background = getattr(summarization_result, 'background', "No background provided")
                methods = getattr(summarization_result, 'methods', "No methods described")
                results = getattr(summarization_result, 'results', "No results provided")
                conclusions = getattr(summarization_result, 'conclusions', "No conclusions provided")
                limitations = getattr(summarization_result, 'limitations', "No limitations discussed")
            elif isinstance(summarization_result, dict):
                background = summarization_result.get('background', "No background provided")
                methods = summarization_result.get('methods', "No methods described")
                results = summarization_result.get('results', "No results provided")
                conclusions = summarization_result.get('conclusions', "No conclusions provided")
                limitations = summarization_result.get('limitations', "No limitations discussed")
            else:
                background = "Error in structured summarization"
                methods = "Error in structured summarization"
                results = "Error in structured summarization"
                conclusions = "Error in structured summarization"
                limitations = "Error in structured summarization"

        except Exception as e:
            logger.error(f"Structured content summarization failed: {str(e)}")
            background = f"Error in summarization: {str(e)}"
            methods = "Error in summarization"
            results = "Error in summarization"
            conclusions = "Error in summarization"
            limitations = "Error in summarization"

        # Sanitize outputs
        sanitized_background = self.sanitize_output(background)
        sanitized_methods = self.sanitize_output(methods)
        sanitized_results = self.sanitize_output(results)
        sanitized_conclusions = self.sanitize_output(conclusions)
        sanitized_limitations = self.sanitize_output(limitations)

        # Prepare result
        result = {
            'text': text[:100] + "..." if len(text) > 100 else text,  # Truncate for logging
            'content_type': content_type,
            'audience': audience,
            'background': sanitized_background,
            'methods': sanitized_methods,
            'results': sanitized_results,
            'conclusions': sanitized_conclusions,
            'limitations': sanitized_limitations
        }

        # Log audit
        self.log_audit(
            "STRUCTURED_CONTENT_SUMMARIZATION_FORWARD",
            {"text": text[:100] + "..." if len(text) > 100 else text, "content_type": content_type, "audience": audience, **kwargs},
            result
        )

        return result


class MultiDocumentSummarizationModule(DSPyModuleBase):
    """Module for summarizing multiple documents.

    This module generates a comprehensive summary across multiple
    documents, identifying common themes, agreements, and disagreements.
    """

    def __init__(
        self,
        single_doc_summarizer: Optional[dspy.Module] = None,
        multi_doc_synthesizer: Optional[dspy.Module] = None,
        **kwargs
    ):
        """
        Initialize the multi-document summarization module.

        Args:
            single_doc_summarizer: Summarizer for individual documents (optional)
            multi_doc_synthesizer: Synthesizer for multiple summaries (optional)
            **kwargs: Additional arguments for the parent class
        """
        super().__init__(**kwargs)

        # Create default single document summarizer if not provided
        self.single_doc_summarizer = single_doc_summarizer or ContentSummarizationModule()

        # Create a specialized signature for multi-document synthesis
        MultiDocSynthesisSignature = dspy.Signature(
            summaries=dspy.InputField(desc="List of individual document summaries"),
            audience=dspy.InputField(desc="Target audience (e.g., 'expert', 'researcher', 'general')"),
            integrated_summary=dspy.OutputField(desc="Integrated summary across all documents"),
            common_themes=dspy.OutputField(desc="Common themes across documents"),
            agreements=dspy.OutputField(desc="Points of agreement across documents"),
            disagreements=dspy.OutputField(desc="Points of disagreement or contradiction"),
            knowledge_gaps=dspy.OutputField(desc="Identified knowledge gaps")
        )

        # Create default multi-document synthesizer if not provided
        self.multi_doc_synthesizer = multi_doc_synthesizer or dspy.ChainOfThought(MultiDocSynthesisSignature)

    def forward(
        self,
        documents: List[str],
        audience: str = "expert",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Summarize multiple documents.

        Args:
            documents: List of documents to summarize
            audience: Target audience (e.g., 'expert', 'researcher', 'general')
            **kwargs: Additional arguments

        Returns:
            Dict[str, Any]: Multi-document summarization result
        """
        # Log audit
        self.log_audit(
            "MULTI_DOCUMENT_SUMMARIZATION_FORWARD",
            {"document_count": len(documents), "audience": audience, **kwargs},
            {}
        )

        # Step 1: Summarize each document individually
        individual_summaries = []

        for i, doc in enumerate(documents):
            try:
                summary_result = self.single_doc_summarizer(text=doc, audience=audience)
                individual_summaries.append({
                    'document_index': i,
                    'summary': summary_result.get('summary', "No summary generated"),
                    'key_findings': summary_result.get('key_findings', "No key findings identified")
                })
            except Exception as e:
                logger.error(f"Summarization of document {i} failed: {str(e)}")
                individual_summaries.append({
                    'document_index': i,
                    'summary': f"Error in summarization: {str(e)}",
                    'key_findings': "Error in key findings extraction"
                })

        # Step 2: Synthesize the individual summaries
        try:
            # Format summaries for the synthesizer
            formatted_summaries = "\n\n".join([
                f"Document {s['document_index']+1}:\nSummary: {s['summary']}\nKey Findings: {s['key_findings']}"
                for s in individual_summaries
            ])

            synthesis_result = self.multi_doc_synthesizer(
                summaries=formatted_summaries,
                audience=audience
            )

            # Extract fields
            if hasattr(synthesis_result, '__dict__'):
                integrated_summary = getattr(synthesis_result, 'integrated_summary', "No integrated summary generated")
                common_themes = getattr(synthesis_result, 'common_themes', "No common themes identified")
                agreements = getattr(synthesis_result, 'agreements', "No agreements identified")
                disagreements = getattr(synthesis_result, 'disagreements', "No disagreements identified")
                knowledge_gaps = getattr(synthesis_result, 'knowledge_gaps', "No knowledge gaps identified")
            elif isinstance(synthesis_result, dict):
                integrated_summary = synthesis_result.get('integrated_summary', "No integrated summary generated")
                common_themes = synthesis_result.get('common_themes', "No common themes identified")
                agreements = synthesis_result.get('agreements', "No agreements identified")
                disagreements = synthesis_result.get('disagreements', "No disagreements identified")
                knowledge_gaps = synthesis_result.get('knowledge_gaps', "No knowledge gaps identified")
            else:
                integrated_summary = str(synthesis_result)
                common_themes = "No common themes identified"
                agreements = "No agreements identified"
                disagreements = "No disagreements identified"
                knowledge_gaps = "No knowledge gaps identified"

        except Exception as e:
            logger.error(f"Multi-document synthesis failed: {str(e)}")
            integrated_summary = f"Error in multi-document synthesis: {str(e)}"
            common_themes = "Error in common themes extraction"
            agreements = "Error in agreements extraction"
            disagreements = "Error in disagreements extraction"
            knowledge_gaps = "Error in knowledge gaps identification"

        # Sanitize outputs
        sanitized_integrated_summary = self.sanitize_output(integrated_summary)
        sanitized_common_themes = self.sanitize_output(common_themes)
        sanitized_agreements = self.sanitize_output(agreements)
        sanitized_disagreements = self.sanitize_output(disagreements)
        sanitized_knowledge_gaps = self.sanitize_output(knowledge_gaps)

        # Prepare result
        result = {
            'document_count': len(documents),
            'audience': audience,
            'individual_summaries': individual_summaries,
            'integrated_summary': sanitized_integrated_summary,
            'common_themes': sanitized_common_themes,
            'agreements': sanitized_agreements,
            'disagreements': sanitized_disagreements,
            'knowledge_gaps': sanitized_knowledge_gaps
        }

        # Log audit
        self.log_audit(
            "MULTI_DOCUMENT_SUMMARIZATION_FORWARD",
            {"document_count": len(documents), "audience": audience, **kwargs},
            result
        )

        return result
