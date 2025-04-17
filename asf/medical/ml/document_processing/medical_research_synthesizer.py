"""
Medical Research Synthesizer

This module provides a complete pipeline for processing and synthesizing medical research papers.
It integrates document processing, entity extraction, relation extraction, and research summarization
components into a cohesive workflow.

The synthesizer orchestrates the following components:
1. Document Processing: Extracts text from PDFs and identifies document structure
2. Entity Extraction: Identifies biomedical entities using GLiNER-biomed or SciSpacy
3. Relation Extraction: Extracts relationships between entities
4. Research Summarization: Generates concise summaries of the research paper
"""

import logging
import time
import torch
from typing import Dict, List, Optional, Any, Tuple

# Local imports
from .document_structure import DocumentStructure
from .document_processor import BiomedicalDocumentProcessor
from .entity_extractor import BiomedicalEntityExtractor
from .gliner_entity_extractor import GLiNERBiomedExtractor
from .relation_extractor import MedicalRelationExtractor
from .hgt_relation_extractor import HGTRelationExtractor
from .research_summarizer import SciFiveResearchSummarizer
from .enhanced_summarizer import EnhancedResearchSummarizer
from .result_exporter import ResultExporter
from .lifecycle_manager import LifecycleManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MedicalResearchSynthesizer:
    """
    Complete pipeline for processing and synthesizing medical research papers.

    Integrates document processing, entity extraction, relation extraction, and
    research summarization components into a cohesive workflow.
    """

    def __init__(
        self,
        document_processor_args: Dict = None,
        entity_extractor_args: Dict = None,
        relation_extractor_args: Dict = None,
        summarizer_args: Dict = None,
        device: Optional[str] = None
    ):
        """
        Initialize the medical research synthesizer.

        Args:
            document_processor_args: Arguments for document processor
            entity_extractor_args: Arguments for entity extractor
            relation_extractor_args: Arguments for relation extractor
            summarizer_args: Arguments for summarizer
            device: Device for PyTorch models
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing Medical Research Synthesizer on {self.device}")

        # Initialize document processor
        doc_args = document_processor_args or {}
        self.document_processor = BiomedicalDocumentProcessor(
            device=self.device, **doc_args
        )

        # Initialize entity extractor
        entity_args = entity_extractor_args or {}

        # Use GLiNER-biomed if specified, otherwise use traditional BiomedicalEntityExtractor
        use_gliner = entity_args.pop('use_gliner', True)
        if use_gliner:
            try:
                self.entity_extractor = GLiNERBiomedExtractor(
                    device=self.device,
                    **entity_args
                )
                logger.info("Using GLiNER-biomed for entity extraction")
            except Exception as e:
                logger.warning(f"Failed to initialize GLiNER-biomed: {str(e)}")
                logger.info("Falling back to traditional BiomedicalEntityExtractor")
                self.entity_extractor = BiomedicalEntityExtractor(**entity_args)
        else:
            self.entity_extractor = BiomedicalEntityExtractor(**entity_args)
            logger.info("Using traditional BiomedicalEntityExtractor")

        # Initialize relation extractor
        rel_args = relation_extractor_args or {}

        # Use HGT relation extractor if specified, otherwise use traditional MedicalRelationExtractor
        use_hgt = rel_args.pop('use_hgt', True)
        if use_hgt:
            try:
                self.relation_extractor = HGTRelationExtractor(
                    device=self.device,
                    **rel_args
                )
                logger.info("Using HGT for relation extraction")
            except Exception as e:
                logger.warning(f"Failed to initialize HGT relation extractor: {str(e)}")
                logger.info("Falling back to traditional MedicalRelationExtractor")
                self.relation_extractor = MedicalRelationExtractor(
                    device=self.device, **rel_args
                )
        else:
            self.relation_extractor = MedicalRelationExtractor(
                device=self.device, **rel_args
            )
            logger.info("Using traditional MedicalRelationExtractor")

        # Initialize summarizer
        sum_args = summarizer_args or {}

        # Use enhanced summarizer if specified, otherwise use traditional SciFiveResearchSummarizer
        use_enhanced = sum_args.pop('use_enhanced', True)
        if use_enhanced:
            try:
                self.summarizer = EnhancedResearchSummarizer(
                    device=self.device,
                    **sum_args
                )
                logger.info("Using enhanced summarizer with factual consistency checking")
            except Exception as e:
                logger.warning(f"Failed to initialize enhanced summarizer: {str(e)}")
                logger.info("Falling back to traditional SciFiveResearchSummarizer")
                self.summarizer = SciFiveResearchSummarizer(
                    device=self.device, **sum_args
                )
        else:
            self.summarizer = SciFiveResearchSummarizer(
                device=self.device, **sum_args
            )
            logger.info("Using traditional SciFiveResearchSummarizer")

        logger.info("Medical Research Synthesizer initialized successfully")

    def process(self, text_or_path: str, is_pdf: bool = False, track_performance: bool = True) -> Tuple[DocumentStructure, Dict[str, Any]]:
        """
        Process a medical research paper end-to-end.

        Args:
            text_or_path: Text or path to PDF
            is_pdf: Whether the input is a PDF path
            track_performance: Whether to track performance metrics

        Returns:
            Tuple of (processed document structure, performance metrics)
        """
        start_time = time.time()
        performance_metrics = {}

        # Step 1: Document processing
        logger.info("Step 1: Document processing")
        step_start = time.time()
        doc_structure = self.document_processor.process_document(text_or_path, is_pdf)
        performance_metrics["document_processing_time"] = time.time() - step_start

        # Step 2: Entity extraction
        logger.info("Step 2: Biomedical entity extraction")
        step_start = time.time()
        doc_structure = self.entity_extractor.process_document(doc_structure)
        performance_metrics["entity_extraction_time"] = time.time() - step_start
        performance_metrics["entity_count"] = len(doc_structure.entities)

        # Step 3: Relation extraction
        logger.info("Step 3: Relation extraction")
        step_start = time.time()
        doc_structure = self.relation_extractor.process_document(doc_structure)
        performance_metrics["relation_extraction_time"] = time.time() - step_start
        performance_metrics["relation_count"] = len(doc_structure.relations)

        # Step 4: Research summarization
        logger.info("Step 4: Research summarization")
        step_start = time.time()
        doc_structure = self.summarizer.process_document(doc_structure)
        performance_metrics["summarization_time"] = time.time() - step_start

        # Calculate total processing time
        total_time = time.time() - start_time
        performance_metrics["total_processing_time"] = total_time

        # Log performance if requested
        if track_performance and hasattr(doc_structure, 'title'):
            document_id = doc_structure.title.replace(" ", "_")[:50]
            LifecycleManager.log_performance(
                document_id=document_id,
                entity_count=len(doc_structure.entities),
                relation_count=len(doc_structure.relations),
                processing_time=total_time,
                additional_metrics=performance_metrics
            )

        logger.info(f"Processing complete in {total_time:.2f} seconds")
        return doc_structure, performance_metrics

    def save_results(self, doc_structure: DocumentStructure, output_dir: str) -> None:
        """
        Save processing results to files.

        Args:
            doc_structure: Processed document structure
            output_dir: Output directory
        """
        ResultExporter.save_results(doc_structure, output_dir)

    def register_with_lifecycle_manager(self, version: str = "1.0.0") -> bool:
        """
        Register the synthesizer with the model lifecycle manager.

        This enables model versioning, monitoring, and automated retraining.

        Args:
            version: Model version

        Returns:
            True if registration was successful, False otherwise
        """
        component_names = {
            "document_processor": self.document_processor.__class__.__name__,
            "entity_extractor": self.entity_extractor.__class__.__name__,
            "relation_extractor": self.relation_extractor.__class__.__name__,
            "summarizer": self.summarizer.__class__.__name__
        }

        return LifecycleManager.register_synthesizer(
            component_names=component_names,
            version=version
        )
