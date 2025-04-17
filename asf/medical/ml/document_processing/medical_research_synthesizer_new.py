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
import os
from typing import Dict, List, Optional, Any, Tuple, Union, Callable

# Local imports
from .document_structure import DocumentStructure
from .document_processor import BiomedicalDocumentProcessor
from .entity_extractor import BiomedicalEntityExtractor
from .gliner_entity_extractor import GLiNERBiomedExtractor
from .relation_extractor import MedicalRelationExtractor
from .hgt_relation_extractor import HGTRelationExtractor
from .research_summarizer import SciFiveResearchSummarizer
from .enhanced_summarizer import EnhancedResearchSummarizer

# Import new modules
from .pipeline_optimizer import (
    PipelineOptimizer, StreamingPipeline,
    optimize_document_processor, optimize_entity_extractor,
    optimize_relation_extractor, optimize_summarizer
)
from .batch_processor import BatchProcessor
from .result_manager import ResultManager
from .model_manager import ModelManager

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
        device: Optional[str] = None,
        use_cache: bool = True,
        cache_dir: str = "cache",
        cache_size_mb: int = 1000,
        model_dir: Optional[str] = None
    ):
        """
        Initialize the medical research synthesizer.

        Args:
            document_processor_args: Arguments for document processor
            entity_extractor_args: Arguments for entity extractor
            relation_extractor_args: Arguments for relation extractor
            summarizer_args: Arguments for summarizer
            device: Device for PyTorch models
            use_cache: Whether to use caching for model outputs
            cache_dir: Directory for cache files
            cache_size_mb: Maximum cache size in MB
            model_dir: Directory for model storage
        """
        # Set device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # Initialize pipeline optimizer
        self.optimizer = PipelineOptimizer(
            cache_dir=cache_dir,
            use_cache=use_cache,
            max_workers=4
        )
        logger.info(f"Pipeline optimizer initialized with cache_dir={cache_dir}, use_cache={use_cache}")

        # Initialize model manager if model_dir is provided
        if model_dir:
            self.model_manager = ModelManager(model_dir=model_dir)
            logger.info(f"Model manager initialized with model_dir={model_dir}")
        else:
            self.model_manager = None

        # Initialize document processor
        doc_args = document_processor_args or {}
        self.document_processor = BiomedicalDocumentProcessor(**doc_args)
        logger.info("Document processor initialized")

        # Initialize entity extractor
        ent_args = entity_extractor_args or {}

        # Use GLiNER if specified, otherwise use traditional BiomedicalEntityExtractor
        use_gliner = ent_args.pop('use_gliner', True)
        if use_gliner:
            try:
                self.entity_extractor = GLiNERBiomedExtractor(
                    device=self.device, **ent_args
                )
                logger.info("Using GLiNER biomedical entity extractor")
            except Exception as e:
                logger.warning(f"Failed to initialize GLiNER: {str(e)}")
                logger.info("Falling back to traditional BiomedicalEntityExtractor")
                self.entity_extractor = BiomedicalEntityExtractor(
                    device=self.device, **ent_args
                )
        else:
            self.entity_extractor = BiomedicalEntityExtractor(
                device=self.device, **ent_args
            )
            logger.info("Using traditional BiomedicalEntityExtractor")

        # Initialize relation extractor
        rel_args = relation_extractor_args or {}

        # Use HGT if specified, otherwise use traditional MedicalRelationExtractor
        use_hgt = rel_args.pop('use_hgt', True)
        if use_hgt:
            try:
                self.relation_extractor = HGTRelationExtractor(
                    device=self.device, **rel_args
                )
                logger.info("Using HGT relation extractor")
            except Exception as e:
                logger.warning(f"Failed to initialize HGT: {str(e)}")
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

        # Initialize batch processor
        self.batch_processor = BatchProcessor(
            synthesizer=self,
            max_workers=4,
            batch_size=4
        )
        logger.info("Batch processor initialized")

        # Optimize components with pipeline optimizer
        if use_cache:
            # Optimize document processor
            self.document_processor = optimize_document_processor(
                self.document_processor, self.optimizer
            )
            logger.info("Document processor optimized with pipeline optimizer")

            # Optimize entity extractor
            self.entity_extractor = optimize_entity_extractor(
                self.entity_extractor, self.optimizer
            )
            logger.info("Entity extractor optimized with pipeline optimizer")

            # Optimize relation extractor
            self.relation_extractor = optimize_relation_extractor(
                self.relation_extractor, self.optimizer
            )
            logger.info("Relation extractor optimized with pipeline optimizer")

            # Optimize summarizer
            self.summarizer = optimize_summarizer(
                self.summarizer, self.optimizer
            )
            logger.info("Summarizer optimized with pipeline optimizer")

        logger.info("Medical Research Synthesizer initialized successfully")

    def process(
        self,
        text_or_path: str,
        is_pdf: bool = False,
        track_performance: bool = True
    ) -> Tuple[DocumentStructure, Dict[str, Any]]:
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

        # Use optimizer for document processing
        doc_structure = self.optimizer.process_with_cache(
            self.document_processor.process_document,
            text_or_path,
            cache_key=None,  # Auto-compute from input
            stage="document_processing",
            force_recompute=False
        )

        performance_metrics["document_processing_time"] = time.time() - step_start

        # Step 2: Entity extraction
        logger.info("Step 2: Biomedical entity extraction")
        step_start = time.time()

        # Use optimizer for entity extraction
        doc_structure = self.optimizer.process_with_cache(
            self.entity_extractor.process_document,
            doc_structure,
            cache_key=f"entity_extraction:{doc_structure.title}" if hasattr(doc_structure, 'title') else None,
            stage="entity_extraction",
            force_recompute=False
        )

        performance_metrics["entity_extraction_time"] = time.time() - step_start
        performance_metrics["entity_count"] = len(doc_structure.entities)

        # Step 3: Relation extraction
        logger.info("Step 3: Relation extraction")
        step_start = time.time()

        # Use optimizer for relation extraction
        doc_structure = self.optimizer.process_with_cache(
            self.relation_extractor.process_document,
            doc_structure,
            cache_key=f"relation_extraction:{doc_structure.title}:{len(doc_structure.entities)}" if hasattr(doc_structure, 'title') else None,
            stage="relation_extraction",
            force_recompute=False
        )

        performance_metrics["relation_extraction_time"] = time.time() - step_start
        performance_metrics["relation_count"] = len(doc_structure.relations)

        # Step 4: Research summarization
        logger.info("Step 4: Research summarization")
        step_start = time.time()

        # Use optimizer for summarization
        doc_structure = self.optimizer.process_with_cache(
            self.summarizer.process_document,
            doc_structure,
            cache_key=f"summarization:{doc_structure.title}:{len(doc_structure.entities)}:{len(doc_structure.relations)}" if hasattr(doc_structure, 'title') else None,
            stage="summarization",
            force_recompute=False
        )

        performance_metrics["summarization_time"] = time.time() - step_start

        # Calculate total processing time
        performance_metrics["total_processing_time"] = time.time() - start_time
        logger.info(f"Processing complete in {performance_metrics['total_processing_time']:.2f}s")

        return doc_structure, performance_metrics

    def process_with_progress(
        self,
        text_or_path: str,
        is_pdf: bool = False,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Tuple[DocumentStructure, Dict[str, Any]]:
        """
        Process a medical research paper with progress tracking.

        Args:
            text_or_path: Text or path to PDF
            is_pdf: Whether the input is a PDF path
            progress_callback: Callback function for progress updates

        Returns:
            Tuple of (processed document structure, performance metrics)
        """
        start_time = time.time()
        performance_metrics = {}

        # Update progress
        if progress_callback:
            progress_callback("start", 0.0)

        # Step 1: Document processing
        logger.info("Step 1: Document processing")
        step_start = time.time()

        # Update progress
        if progress_callback:
            progress_callback("document_processing", 0.0)

        # Process document
        doc_structure = self.document_processor.process_document(text_or_path, is_pdf)
        performance_metrics["document_processing_time"] = time.time() - step_start

        # Update progress
        if progress_callback:
            progress_callback("document_processing", 1.0)
            progress_callback("overall", 0.25)

        # Step 2: Entity extraction
        logger.info("Step 2: Biomedical entity extraction")
        step_start = time.time()

        # Update progress
        if progress_callback:
            progress_callback("entity_extraction", 0.0)

        # Process entities
        doc_structure = self.entity_extractor.process_document(doc_structure)
        performance_metrics["entity_extraction_time"] = time.time() - step_start
        performance_metrics["entity_count"] = len(doc_structure.entities)

        # Update progress
        if progress_callback:
            progress_callback("entity_extraction", 1.0)
            progress_callback("overall", 0.5)

        # Step 3: Relation extraction
        logger.info("Step 3: Relation extraction")
        step_start = time.time()

        # Update progress
        if progress_callback:
            progress_callback("relation_extraction", 0.0)

        # Process relations
        doc_structure = self.relation_extractor.process_document(doc_structure)
        performance_metrics["relation_extraction_time"] = time.time() - step_start
        performance_metrics["relation_count"] = len(doc_structure.relations)

        # Update progress
        if progress_callback:
            progress_callback("relation_extraction", 1.0)
            progress_callback("overall", 0.75)

        # Step 4: Research summarization
        logger.info("Step 4: Research summarization")
        step_start = time.time()

        # Update progress
        if progress_callback:
            progress_callback("summarization", 0.0)

        # Process summary
        doc_structure = self.summarizer.process_document(doc_structure)
        performance_metrics["summarization_time"] = time.time() - step_start

        # Update progress
        if progress_callback:
            progress_callback("summarization", 1.0)
            progress_callback("overall", 1.0)

        # Calculate total processing time
        performance_metrics["total_processing_time"] = time.time() - start_time
        logger.info(f"Processing complete in {performance_metrics['total_processing_time']:.2f}s")

        return doc_structure, performance_metrics

    def process_streaming(
        self,
        text_or_path: str,
        is_pdf: bool = False,
        streaming_callback: Optional[Callable[[str, Any], None]] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Tuple[DocumentStructure, Dict[str, Any]]:
        """
        Process a medical research paper with streaming results.

        Args:
            text_or_path: Text or path to PDF
            is_pdf: Whether the input is a PDF path
            streaming_callback: Callback function for streaming results
            progress_callback: Callback function for progress updates

        Returns:
            Tuple of (processed document structure, performance metrics)
        """
        # Define processing stages
        stages = [
            ("document_processing", lambda x: self.document_processor.process_document(x, is_pdf)),
            ("entity_extraction", self.entity_extractor.process_document),
            ("relation_extraction", self.relation_extractor.process_document),
            ("summarization", self.summarizer.process_document)
        ]

        # Create streaming pipeline
        pipeline = StreamingPipeline(
            optimizer=self.optimizer,
            stages=stages,
            callback=streaming_callback
        )

        # Process with streaming pipeline
        start_time = time.time()

        # Generate a task ID for progress tracking
        task_id = f"task_{int(time.time())}"

        # Process document
        stage_results = pipeline.process(
            text_or_path,
            task_id=task_id,
            force_recompute=False
        )

        # Get final document structure
        doc_structure = stage_results.get("summarization")
        if not doc_structure:
            # If summarization failed, use the latest available stage
            for stage in ["relation_extraction", "entity_extraction", "document_processing"]:
                if stage in stage_results:
                    doc_structure = stage_results[stage]
                    break

        # Calculate performance metrics
        performance_metrics = {
            "total_processing_time": time.time() - start_time
        }

        # Add stage-specific metrics
        for stage, _ in stages:
            if stage in stage_results:
                performance_metrics[f"{stage}_time"] = self.optimizer.get_progress(f"{task_id}_{stage}")

        # Add entity and relation counts
        if hasattr(doc_structure, 'entities'):
            performance_metrics["entity_count"] = len(doc_structure.entities)

        if hasattr(doc_structure, 'relations'):
            performance_metrics["relation_count"] = len(doc_structure.relations)

        logger.info(f"Streaming processing complete in {performance_metrics['total_processing_time']:.2f}s")

        return doc_structure, performance_metrics

    def process_batch(
        self,
        file_list: List[str],
        output_dir: str,
        batch_size: int = 4,
        all_pdfs: bool = True,
        save_results: bool = True,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Process a batch of documents with configurable parallelism.

        Args:
            file_list: List of file paths
            output_dir: Base output directory
            batch_size: Number of documents to process in parallel
            all_pdfs: Whether all files are PDFs
            save_results: Whether to save results to disk
            progress_callback: Callback function for progress updates

        Returns:
            Dictionary with aggregate performance metrics
        """
        # Update batch processor settings
        self.batch_processor.batch_size = batch_size
        self.batch_processor.progress_callback = progress_callback

        # Process batch
        return self.batch_processor.process_batch(
            file_list=file_list,
            output_dir=output_dir,
            all_pdfs=all_pdfs,
            save_results=save_results
        )

    def save_results(self, doc_structure: DocumentStructure, output_dir: str, formats: List[str] = None) -> Dict[str, str]:
        """
        Save processing results to files in various formats.

        Args:
            doc_structure: Processed document structure
            output_dir: Output directory
            formats: List of formats to save (json, csv, pickle, graphml)

        Returns:
            Dictionary mapping format to file path
        """
        return ResultManager.save_results(doc_structure, output_dir, formats)

    def update_models(
        self,
        labeled_data: Dict[str, List[Dict]],
        learning_rate: float = 1e-5,
        batch_size: int = 4,
        epochs: int = 1
    ) -> Dict[str, Any]:
        """
        Update models with new labeled data using online learning.

        Args:
            labeled_data: Dictionary with labeled data for different components
            learning_rate: Learning rate for updates
            batch_size: Batch size for updates
            epochs: Number of epochs for updates

        Returns:
            Dictionary with update results
        """
        if not self.model_manager:
            logger.warning("Model manager not initialized. Cannot update models.")
            return {"success": False, "error": "Model manager not initialized"}

        results = {}

        # Update entity extractor if labeled data is provided
        if "entities" in labeled_data and hasattr(self.entity_extractor, "update_model"):
            try:
                entity_result = self.entity_extractor.update_model(
                    labeled_data["entities"],
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    epochs=epochs
                )

                # Register updated model with model manager
                if entity_result.get("success", False):
                    model_path = entity_result.get("model_path")
                    performance = entity_result.get("performance", {})

                    self.model_manager.register_model(
                        model_name="entity_extractor",
                        model_type=self.entity_extractor.__class__.__name__,
                        version=entity_result.get("version", "latest"),
                        model_path=model_path,
                        performance_metrics=performance
                    )

                results["entity_extractor"] = entity_result
            except Exception as e:
                logger.error(f"Error updating entity extractor: {str(e)}")
                results["entity_extractor"] = {"success": False, "error": str(e)}

        # Update relation extractor if labeled data is provided
        if "relations" in labeled_data and hasattr(self.relation_extractor, "update_model"):
            try:
                relation_result = self.relation_extractor.update_model(
                    labeled_data["relations"],
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    epochs=epochs
                )

                # Register updated model with model manager
                if relation_result.get("success", False):
                    model_path = relation_result.get("model_path")
                    performance = relation_result.get("performance", {})

                    self.model_manager.register_model(
                        model_name="relation_extractor",
                        model_type=self.relation_extractor.__class__.__name__,
                        version=relation_result.get("version", "latest"),
                        model_path=model_path,
                        performance_metrics=performance
                    )

                results["relation_extractor"] = relation_result
            except Exception as e:
                logger.error(f"Error updating relation extractor: {str(e)}")
                results["relation_extractor"] = {"success": False, "error": str(e)}

        # Update summarizer if labeled data is provided
        if "summaries" in labeled_data and hasattr(self.summarizer, "update_model"):
            try:
                summary_result = self.summarizer.update_model(
                    labeled_data["summaries"],
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    epochs=epochs
                )

                # Register updated model with model manager
                if summary_result.get("success", False):
                    model_path = summary_result.get("model_path")
                    performance = summary_result.get("performance", {})

                    self.model_manager.register_model(
                        model_name="summarizer",
                        model_type=self.summarizer.__class__.__name__,
                        version=summary_result.get("version", "latest"),
                        model_path=model_path,
                        performance_metrics=performance
                    )

                results["summarizer"] = summary_result
            except Exception as e:
                logger.error(f"Error updating summarizer: {str(e)}")
                results["summarizer"] = {"success": False, "error": str(e)}

        return results

    def close(self):
        """
        Close the synthesizer and release resources.
        """
        # Close optimizer
        if hasattr(self.optimizer, 'close'):
            self.optimizer.close()

        # Close batch processor
        if hasattr(self.batch_processor, 'close'):
            self.batch_processor.close()

        logger.info("Medical Research Synthesizer closed")
