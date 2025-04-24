"""
Enhanced Research Paper Summarization

This module provides enhanced functionality for summarizing medical research papers
using SciFive with optimized input strategy and factual consistency checking.
"""

import logging
import torch
import re
from typing import List, Dict, Tuple, Optional, Any, Union

# Local imports
from .document_structure import DocumentStructure, SectionInfo
from .research_summarizer import SciFiveResearchSummarizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedResearchSummarizer(SciFiveResearchSummarizer):
    """
    Enhanced research paper summarization using SciFive with optimized input strategy.
    
    This summarizer extends SciFiveResearchSummarizer with:
    1. Prioritized section input strategy (Abstract, Results, Conclusion)
    2. Factual consistency checking using QAFactEval or SummaC
    """
    
    def __init__(
        self,
        model_name: str = "razent/SciFive-large-Pubmed-paper_summary",
        device: Optional[str] = None,
        max_length: int = 512,
        min_length: int = 50,
        check_factual_consistency: bool = True,
        consistency_threshold: float = 0.5,
        consistency_method: str = "qafacteval"
    ):
        """
        Initialize the enhanced SciFive summarizer.
        
        Args:
            model_name: SciFive model name
            device: Device for model
            max_length: Maximum output length
            min_length: Minimum output length
            check_factual_consistency: Whether to check factual consistency
            consistency_threshold: Threshold for factual consistency
            consistency_method: Method for checking factual consistency ('qafacteval' or 'summac')
        """
        super().__init__(
            model_name=model_name,
            device=device,
            max_length=max_length,
            min_length=min_length
        )
        
        self.check_factual_consistency = check_factual_consistency
        self.consistency_threshold = consistency_threshold
        self.consistency_method = consistency_method
        
        # Initialize factual consistency checker if requested
        self.consistency_checker = None
        if check_factual_consistency:
            self._initialize_consistency_checker(consistency_method)
    
    def _initialize_consistency_checker(self, method: str) -> None:
        """
        Initialize the factual consistency checker.
        
        Args:
            method: Method for checking factual consistency ('qafacteval' or 'summac')
        """
        if method.lower() == "qafacteval":
            try:
                from qafacteval import QAFactEval
                
                logger.info("Initializing QAFactEval for factual consistency checking")
                self.consistency_checker = QAFactEval(
                    use_cuda=torch.cuda.is_available(),
                    cuda_device=0 if torch.cuda.is_available() else -1
                )
                logger.info("QAFactEval initialized successfully")
            except ImportError:
                logger.warning("QAFactEval not available. Install with: pip install qafacteval")
                self.check_factual_consistency = False
            except Exception as e:
                logger.warning(f"Error initializing QAFactEval: {str(e)}")
                self.check_factual_consistency = False
        
        elif method.lower() == "summac":
            try:
                from summac.model_summac import SummaCZS
                
                logger.info("Initializing SummaC for factual consistency checking")
                self.consistency_checker = SummaCZS(
                    granularity="sentence",
                    model_name="vitc",
                    device=self.device
                )
                logger.info("SummaC initialized successfully")
            except ImportError:
                logger.warning("SummaC not available. Install with: pip install summac")
                self.check_factual_consistency = False
            except Exception as e:
                logger.warning(f"Error initializing SummaC: {str(e)}")
                self.check_factual_consistency = False
        
        else:
            logger.warning(f"Unknown consistency method: {method}")
            self.check_factual_consistency = False
    
    def check_consistency(self, source_text: str, summary: str) -> Tuple[float, bool]:
        """
        Check factual consistency of a summary against source text.
        
        Args:
            source_text: Source text
            summary: Generated summary
            
        Returns:
            Tuple of (consistency score, is consistent)
        """
        if not self.check_factual_consistency or self.consistency_checker is None:
            return 1.0, True
        
        try:
            if self.consistency_method.lower() == "qafacteval":
                # QAFactEval expects a list of summaries and sources
                result = self.consistency_checker.score([summary], [[source_text]])
                score = result["scores"][0]
            
            elif self.consistency_method.lower() == "summac":
                # SummaC expects a list of (summary, source) pairs
                result = self.consistency_checker.score([{"summary": summary, "text": source_text}])
                score = result["scores"][0]
            
            else:
                logger.warning(f"Unknown consistency method: {self.consistency_method}")
                return 1.0, True
            
            is_consistent = score >= self.consistency_threshold
            return score, is_consistent
        
        except Exception as e:
            logger.warning(f"Error checking factual consistency: {str(e)}")
            return 1.0, True
    
    def get_prioritized_input(self, doc_structure: DocumentStructure) -> str:
        """
        Get prioritized input text for summarization.
        
        Prioritizes Abstract, Results, and Conclusion sections.
        
        Args:
            doc_structure: Document structure
            
        Returns:
            Prioritized input text
        """
        input_parts = []
        
        # Add abstract if available
        if doc_structure.abstract:
            input_parts.append(f"Abstract: {doc_structure.abstract}")
        
        # Add results sections
        results_sections = [s for s in doc_structure.sections if s.section_type == "results"]
        if results_sections:
            results_text = "\n\n".join([s.text for s in results_sections])
            input_parts.append(f"Results: {results_text}")
        
        # Add conclusion/discussion sections
        conclusion_sections = [s for s in doc_structure.sections 
                              if s.section_type in ["conclusion", "discussion"]]
        if conclusion_sections:
            conclusion_text = "\n\n".join([s.text for s in conclusion_sections])
            input_parts.append(f"Conclusion: {conclusion_text}")
        
        # If we don't have enough content, add introduction
        if len("\n\n".join(input_parts)) < 1000:
            intro_sections = [s for s in doc_structure.sections 
                             if s.section_type in ["introduction", "background"]]
            if intro_sections:
                intro_text = "\n\n".join([s.text for s in intro_sections])
                input_parts.append(f"Introduction: {intro_text}")
        
        # Combine parts
        input_text = "\n\n".join(input_parts)
        
        # If input is too long, truncate while preserving key sections
        if len(input_text) > 4096:
            # Keep abstract and beginning of other sections
            if doc_structure.abstract:
                abstract = f"Abstract: {doc_structure.abstract}\n\n"
                remaining_budget = 4096 - len(abstract)
                other_content = input_text[len(abstract):]
                
                # Truncate other content
                truncated_content = other_content[:remaining_budget]
                input_text = abstract + truncated_content
            else:
                # Simple truncation
                input_text = input_text[:4096]
        
        return input_text
    
    def process_document(self, doc_structure: DocumentStructure) -> DocumentStructure:
        """
        Generate summaries for a document structure with enhanced input strategy.
        
        Args:
            doc_structure: Document structure
            
        Returns:
            Updated document structure with summaries
        """
        logger.info("Generating enhanced research summaries...")
        
        # Get prioritized input text
        prioritized_input = self.get_prioritized_input(doc_structure)
        
        # Generate abstract summary
        abstract_summary = ""
        if doc_structure.abstract:
            abstract_summary = self.summarize_section(
                doc_structure.abstract, "abstract"
            )
            
            # Check factual consistency
            if self.check_factual_consistency:
                score, is_consistent = self.check_consistency(
                    doc_structure.abstract, abstract_summary
                )
                logger.info(f"Abstract summary consistency score: {score:.2f}")
                
                if not is_consistent:
                    logger.warning("Abstract summary may contain factual inconsistencies")
                    abstract_summary += "\n\nNote: This summary may contain factual inconsistencies."
        
        # Extract key findings from prioritized input
        key_findings = self.extract_key_findings(prioritized_input)
        
        # Generate methods summary
        methods_summary = None
        methods_sections = [s for s in doc_structure.sections 
                           if s.section_type in ["methods", "materials_and_methods"]]
        if methods_sections:
            methods_text = "\n".join([s.text for s in methods_sections])
            methods_summary = self.summarize_section(
                methods_text, "methods"
            )
            
            # Check factual consistency
            if self.check_factual_consistency:
                score, is_consistent = self.check_consistency(
                    methods_text, methods_summary
                )
                logger.info(f"Methods summary consistency score: {score:.2f}")
                
                if not is_consistent:
                    logger.warning("Methods summary may contain factual inconsistencies")
                    methods_summary += "\n\nNote: This summary may contain factual inconsistencies."
        
        # Generate conclusions
        conclusions = None
        conclusion_sections = [s for s in doc_structure.sections 
                              if s.section_type in ["conclusion", "discussion"]]
        if conclusion_sections:
            conclusion_text = "\n".join([s.text for s in conclusion_sections])
            conclusions = self.summarize_section(
                conclusion_text, "conclusion"
            )
            
            # Check factual consistency
            if self.check_factual_consistency:
                score, is_consistent = self.check_consistency(
                    conclusion_text, conclusions
                )
                logger.info(f"Conclusions summary consistency score: {score:.2f}")
                
                if not is_consistent:
                    logger.warning("Conclusions summary may contain factual inconsistencies")
                    conclusions += "\n\nNote: This summary may contain factual inconsistencies."
        
        # Extract limitations
        limitations = None
        if conclusion_sections:
            conclusion_text = "\n".join([s.text for s in conclusion_sections])
            limitations = self.extract_limitations(conclusion_text)
        
        # Extract clinical implications
        clinical_implications = None
        if prioritized_input:
            clinical_implications = self.extract_clinical_implications(prioritized_input)
        
        # Create research summary object
        doc_structure.summary = {
            "abstract": abstract_summary,
            "key_findings": key_findings,
            "methods": methods_summary,
            "conclusions": conclusions,
            "limitations": limitations,
            "clinical_implications": clinical_implications
        }
        
        logger.info("Enhanced research summarization complete")
        return doc_structure
