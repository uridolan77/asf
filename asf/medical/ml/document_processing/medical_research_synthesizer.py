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
import json
import hashlib
import sqlite3
import concurrent.futures
from typing import Dict, List, Optional, Any, Tuple, Set, Union

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


class ModelCache:
    """
    Cache for model outputs to avoid redundant processing.
    """

    def __init__(self, cache_dir: str, max_size_mb: int = 1000, max_entry_size_mb: int = 100):
        """
        Initialize the model cache.

        Args:
            cache_dir: Directory for cache files
            max_size_mb: Maximum cache size in MB
            max_entry_size_mb: Maximum size of a single cache entry in MB
        """
        self.cache_dir = cache_dir
        self.max_size_mb = max_size_mb
        self.max_entry_size_mb = max_entry_size_mb
        os.makedirs(cache_dir, exist_ok=True)

        # Initialize cache database
        self.db_path = os.path.join(cache_dir, "cache.db")
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

        # Create cache tables if they don't exist
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS cache_entries (
            hash TEXT PRIMARY KEY,
            component TEXT,
            timestamp REAL,
            size_bytes INTEGER,
            filename TEXT
        )
        ''')
        self.conn.commit()
        
        # Don't cleanup at initialization - only do it on explicit request
        # or when adding new items that exceed the limit
        
        # Create an executor for background operations
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.pending_tasks = []

    def _get_hash(self, text: str, component: str) -> str:
        """Get hash for cache key."""
        return hashlib.md5(f"{component}:{text}".encode()).hexdigest()

    def _cleanup_if_needed(self):
        """Cleanup old cache entries if cache is too large."""
        # Get total cache size
        self.cursor.execute("SELECT SUM(size_bytes) FROM cache_entries")
        result = self.cursor.fetchone()
        total_size = result[0] if result[0] else 0

        # If cache is too large, remove oldest entries
        if total_size > self.max_size_mb * 1024 * 1024:
            logger.info(f"Cache size {total_size/1024/1024:.2f}MB exceeds limit, cleaning up")

            # Get entries to remove - remove more entries at once to avoid frequent cleanups
            self.cursor.execute(
                "SELECT hash, filename FROM cache_entries ORDER BY timestamp ASC LIMIT 200"
            )
            entries_to_remove = self.cursor.fetchall()

            # Remove entries
            for hash_val, filename in entries_to_remove:
                try:
                    file_path = os.path.join(self.cache_dir, filename)
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    self.cursor.execute("DELETE FROM cache_entries WHERE hash = ?", (hash_val,))
                except Exception as e:
                    logger.warning(f"Error removing cache entry: {str(e)}")

            self.conn.commit()
            logger.info(f"Cleaned up {len(entries_to_remove)} cache entries")

    def cleanup(self, force: bool = False):
        """
        Explicitly run cache cleanup.
        
        Args:
            force: If True, force cleanup regardless of current cache size
        """
        if force:
            # Force cleanup by setting a temporary max size of 0
            original_max_size = self.max_size_mb
            self.max_size_mb = 0
            self._cleanup_if_needed()
            self.max_size_mb = original_max_size
        else:
            # Normal cleanup based on current cache size
            self._cleanup_if_needed()
            
        return True

    def get(self, text: str, component: str) -> Optional[Any]:
        """
        Get cached result for text and component.

        Args:
            text: Input text
            component: Component name

        Returns:
            Cached result or None if not in cache
        """
        import pickle

        hash_val = self._get_hash(text, component)

        try:
            self.cursor.execute(
                "SELECT filename FROM cache_entries WHERE hash = ?",
                (hash_val,)
            )
            result = self.cursor.fetchone()

            if result:
                filename = result[0]
                file_path = os.path.join(self.cache_dir, filename)

                try:
                    if os.path.exists(file_path):
                        with open(file_path, 'rb') as f:
                            cached_result = pickle.load(f)

                        # Update timestamp in background
                        def update_timestamp():
                            try:
                                conn = sqlite3.connect(self.db_path)
                                cursor = conn.cursor()
                                cursor.execute(
                                    "UPDATE cache_entries SET timestamp = ? WHERE hash = ?",
                                    (time.time(), hash_val)
                                )
                                conn.commit()
                                conn.close()
                            except Exception as e:
                                logger.warning(f"Error updating timestamp: {str(e)}")
                        
                        self.executor.submit(update_timestamp)
                        return cached_result
                    else:
                        # File doesn't exist, remove the entry
                        self.cursor.execute("DELETE FROM cache_entries WHERE hash = ?", (hash_val,))
                        self.conn.commit()
                except Exception as e:
                    logger.warning(f"Error reading cache entry: {str(e)}")
                    return None
        except Exception as e:
            logger.warning(f"Error querying cache: {str(e)}")
            
        return None

    def set(self, text: str, component: str, result: Any) -> None:
        """
        Cache result for text and component.

        Args:
            text: Input text
            component: Component name
            result: Result to cache
        """
        # Submit caching operation to background executor to avoid blocking UI
        future = self.executor.submit(self._set_async, text, component, result)
        self.pending_tasks.append(future)
        
        # Clean up completed tasks
        self.pending_tasks = [task for task in self.pending_tasks if not task.done()]

    def _set_async(self, text: str, component: str, result: Any) -> None:
        """Asynchronous implementation of cache set operation."""
        import pickle
        import tempfile

        hash_val = self._get_hash(text, component)
        filename = f"{hash_val}.pkl"
        file_path = os.path.join(self.cache_dir, filename)

        try:
            # First save to a temporary file to check the size
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                pickle.dump(result, temp_file)
                temp_path = temp_file.name
            
            # Check file size
            size_bytes = os.path.getsize(temp_path)
            max_entry_bytes = self.max_entry_size_mb * 1024 * 1024
            
            if size_bytes > max_entry_bytes:
                logger.warning(f"Cache entry too large ({size_bytes/1024/1024:.2f}MB > {self.max_entry_size_mb}MB), skipping")
                os.unlink(temp_path)
                return
                
            # If size is acceptable, move to final location
            import shutil
            shutil.move(temp_path, file_path)

            # Run a connection in this thread to avoid locking the main connection
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Update cache database
            cursor.execute(
                "INSERT OR REPLACE INTO cache_entries VALUES (?, ?, ?, ?, ?)",
                (hash_val, component, time.time(), size_bytes, filename)
            )
            conn.commit()
            conn.close()

            # Cleanup in background if needed
            self._cleanup_if_needed()
        except Exception as e:
            logger.warning(f"Error caching result: {str(e)}")
            # Clean up temp file if it exists
            if 'temp_path' in locals() and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass

    def close(self):
        """Close database connection and executor."""
        if self.conn:
            self.conn.close()
            
        # Shutdown executor after completing pending tasks
        self.executor.shutdown(wait=True)

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
        cache_size_mb: int = 1000
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
        """
        # Set device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize cache if needed
        self.use_cache = use_cache
        if use_cache:
            self.cache = ModelCache(cache_dir=cache_dir, max_size_mb=cache_size_mb)
            logger.info(f"Model cache initialized at {cache_dir} with max size {cache_size_mb} MB")
        else:
            self.cache = None
            logger.info("Model caching disabled")
            
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
        cache_hits = 0

        # Check if we have a cached result for the entire document
        if self.use_cache and not is_pdf:
            # Only cache text inputs, not PDF paths
            cached_result = self.cache.get(text_or_path, "full_document")
            if cached_result:
                logger.info("Using cached result for full document")
                doc_structure, cached_metrics = cached_result
                # Update metrics with cache information
                performance_metrics = cached_metrics.copy()
                performance_metrics["cache_hit"] = True
                performance_metrics["retrieval_time"] = time.time() - start_time
                return doc_structure, performance_metrics

        # Step 1: Document processing
        logger.info("Step 1: Document processing")
        step_start = time.time()
        doc_structure = self.document_processor.process_document(text_or_path, is_pdf)
        performance_metrics["document_processing_time"] = time.time() - step_start

        # Step 2: Entity extraction
        logger.info("Step 2: Biomedical entity extraction")
        step_start = time.time()

        # Check cache for entity extraction
        if self.use_cache and hasattr(doc_structure, 'title'):
            cache_key = f"entity_extraction:{doc_structure.title}"
            cached_entities = self.cache.get(cache_key, "entity_extraction")
            if cached_entities:
                logger.info("Using cached entities")
                doc_structure.entities = cached_entities
                cache_hits += 1
            else:
                # Process and cache
                doc_structure = self.entity_extractor.process_document(doc_structure)
                if doc_structure.entities:
                    self.cache.set(cache_key, "entity_extraction", doc_structure.entities)
        else:
            # Process without caching
            doc_structure = self.entity_extractor.process_document(doc_structure)

        performance_metrics["entity_extraction_time"] = time.time() - step_start
        performance_metrics["entity_count"] = len(doc_structure.entities)

        # Step 3: Relation extraction
        logger.info("Step 3: Relation extraction")
        step_start = time.time()

        # Check cache for relation extraction
        if self.use_cache and hasattr(doc_structure, 'title') and doc_structure.entities:
            cache_key = f"relation_extraction:{doc_structure.title}:{len(doc_structure.entities)}"
            cached_relations = self.cache.get(cache_key, "relation_extraction")
            if cached_relations:
                logger.info("Using cached relations")
                doc_structure.relations = cached_relations
                cache_hits += 1
            else:
                # Process and cache
                doc_structure = self.relation_extractor.process_document(doc_structure)
                if doc_structure.relations:
                    self.cache.set(cache_key, "relation_extraction", doc_structure.relations)
        else:
            # Process without caching
            doc_structure = self.relation_extractor.process_document(doc_structure)

        performance_metrics["relation_extraction_time"] = time.time() - step_start
        performance_metrics["relation_count"] = len(doc_structure.relations)

        # Step 4: Research summarization
        logger.info("Step 4: Research summarization")
        step_start = time.time()

        # Check cache for summarization
        if self.use_cache and hasattr(doc_structure, 'title'):
            cache_key = f"summarization:{doc_structure.title}:{len(doc_structure.entities)}:{len(doc_structure.relations)}"
            cached_summary = self.cache.get(cache_key, "summarization")
            if cached_summary:
                logger.info("Using cached summary")
                doc_structure.summary = cached_summary
                cache_hits += 1
            else:
                # Process and cache
                doc_structure = self.summarizer.process_document(doc_structure)
                if doc_structure.summary:
                    self.cache.set(cache_key, "summarization", doc_structure.summary)
        else:
            # Process without caching
            doc_structure = self.summarizer.process_document(doc_structure)

        performance_metrics["summarization_time"] = time.time() - step_start

        # Calculate total processing time
        total_time = time.time() - start_time
        performance_metrics["total_processing_time"] = total_time
        performance_metrics["cache_hits"] = cache_hits

        # Cache the full result if appropriate
        if self.use_cache and not is_pdf and hasattr(doc_structure, 'title'):
            self.cache.set(text_or_path, "full_document", (doc_structure, performance_metrics))

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

        logger.info(f"Processing complete in {total_time:.2f} seconds with {cache_hits} cache hits")
        return doc_structure, performance_metrics

    def process_parallel(self, text_or_path: str, is_pdf: bool = False, track_performance: bool = True) -> Tuple[DocumentStructure, Dict[str, Any]]:
        """
        Process a medical research paper with parallel execution of components.

        Args:
            text_or_path: Text or path to PDF
            is_pdf: Whether the input is a PDF path
            track_performance: Whether to track performance metrics

        Returns:
            Tuple of (processed document structure, performance metrics)
        """
        start_time = time.time()
        performance_metrics = {}

        # Step 1: Document processing (must be done first)
        logger.info("Step 1: Document processing")
        step_start = time.time()
        doc_structure = self.document_processor.process_document(text_or_path, is_pdf)
        performance_metrics["document_processing_time"] = time.time() - step_start

        # Steps 2 and 3 can be run in parallel
        logger.info("Steps 2 & 3: Running entity extraction and relation extraction in parallel")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Submit entity extraction task
            entity_start = time.time()
            entity_future = executor.submit(
                self.entity_extractor.process_document,
                doc_structure
            )

            # Wait for entity extraction to finish
            doc_structure_with_entities = entity_future.result()
            performance_metrics["entity_extraction_time"] = time.time() - entity_start
            performance_metrics["entity_count"] = len(doc_structure_with_entities.entities)

            # Submit relation extraction task with the updated document structure
            relation_start = time.time()
            relation_future = executor.submit(
                self.relation_extractor.process_document,
                doc_structure_with_entities
            )

            # Wait for relation extraction to finish
            doc_structure_with_relations = relation_future.result()
            performance_metrics["relation_extraction_time"] = time.time() - relation_start
            performance_metrics["relation_count"] = len(doc_structure_with_relations.relations)

        # Step 4: Research summarization
        logger.info("Step 4: Research summarization")
        step_start = time.time()
        final_doc_structure = self.summarizer.process_document(doc_structure_with_relations)
        performance_metrics["summarization_time"] = time.time() - step_start

        # Calculate total processing time
        total_time = time.time() - start_time
        performance_metrics["total_processing_time"] = total_time

        # Log performance if requested
        if track_performance and hasattr(final_doc_structure, 'title'):
            document_id = final_doc_structure.title.replace(" ", "_")[:50]
            LifecycleManager.log_performance(
                document_id=document_id,
                entity_count=len(final_doc_structure.entities),
                relation_count=len(final_doc_structure.relations),
                processing_time=total_time,
                additional_metrics=performance_metrics
            )

        logger.info(f"Parallel processing complete in {total_time:.2f} seconds")
        return final_doc_structure, performance_metrics

    def process_batch(self, file_list: List[str], output_dir: str, batch_size: int = 4, all_pdfs: bool = True) -> Dict[str, Any]:
        """
        Process a batch of documents with configurable parallelism.

        Args:
            file_list: List of file paths
            output_dir: Base output directory
            batch_size: Number of documents to process in parallel
            all_pdfs: Whether all files are PDFs

        Returns:
            Dictionary with aggregate performance metrics
        """
        overall_start = time.time()
        aggregate_metrics = {
            "total_documents": len(file_list),
            "successful": 0,
            "failed": 0,
            "entities_total": 0,
            "relations_total": 0,
            "processing_times": []
        }

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        def process_single_file(file_path):
            try:
                doc_name = os.path.basename(file_path).split('.')[0]
                doc_output_dir = os.path.join(output_dir, doc_name)

                # Process document
                doc_structure, metrics = self.process(file_path, is_pdf=all_pdfs)

                # Save results
                self.save_results(doc_structure, doc_output_dir)

                return {
                    "file": file_path,
                    "status": "success",
                    "metrics": metrics,
                    "entity_count": len(doc_structure.entities),
                    "relation_count": len(doc_structure.relations)
                }
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                return {
                    "file": file_path,
                    "status": "failed",
                    "error": str(e)
                }

        # Process files in batches
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=batch_size) as executor:
            # Submit all processing tasks
            future_to_file = {executor.submit(process_single_file, file_path): file_path
                             for file_path in file_list}

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)

                    if result["status"] == "success":
                        aggregate_metrics["successful"] += 1
                        aggregate_metrics["entities_total"] += result["entity_count"]
                        aggregate_metrics["relations_total"] += result["relation_count"]
                        aggregate_metrics["processing_times"].append(result["metrics"]["total_processing_time"])
                    else:
                        aggregate_metrics["failed"] += 1

                    logger.info(f"Processed {len(results)}/{len(file_list)}: {file_path}")
                except Exception as e:
                    logger.error(f"Error handling result for {file_path}: {str(e)}")
                    aggregate_metrics["failed"] += 1

        # Calculate aggregate metrics
        aggregate_metrics["total_processing_time"] = time.time() - overall_start
        if aggregate_metrics["processing_times"]:
            aggregate_metrics["avg_document_time"] = sum(aggregate_metrics["processing_times"]) / len(aggregate_metrics["processing_times"])
            aggregate_metrics["min_document_time"] = min(aggregate_metrics["processing_times"])
            aggregate_metrics["max_document_time"] = max(aggregate_metrics["processing_times"])

        # Save aggregate metrics
        with open(os.path.join(output_dir, "batch_metrics.json"), "w") as f:
            json.dump(aggregate_metrics, f, indent=2)

        return aggregate_metrics

    def save_results(self, doc_structure: DocumentStructure, output_dir: str) -> None:
        """
        Save processing results to files.

        Args:
            doc_structure: Processed document structure
            output_dir: Output directory
        """
        ResultExporter.save_results(doc_structure, output_dir)

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
            Dictionary with update metrics
        """
        update_metrics = {}

        # Update entity extractor if data provided
        if "entities" in labeled_data and hasattr(self.entity_extractor, "update_model"):
            try:
                logger.info(f"Updating entity extractor with {len(labeled_data['entities'])} examples")
                entity_metrics = self.entity_extractor.update_model(
                    labeled_data["entities"],
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    epochs=epochs
                )
                update_metrics["entity_extractor"] = entity_metrics
            except Exception as e:
                logger.error(f"Error updating entity extractor: {str(e)}")

        # Update relation extractor if data provided
        if "relations" in labeled_data and hasattr(self.relation_extractor, "update_model"):
            try:
                logger.info(f"Updating relation extractor with {len(labeled_data['relations'])} examples")
                relation_metrics = self.relation_extractor.update_model(
                    labeled_data["relations"],
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    epochs=epochs
                )
                update_metrics["relation_extractor"] = relation_metrics
            except Exception as e:
                logger.error(f"Error updating relation extractor: {str(e)}")

        # Update summarizer if data provided
        if "summaries" in labeled_data and hasattr(self.summarizer, "update_model"):
            try:
                logger.info(f"Updating summarizer with {len(labeled_data['summaries'])} examples")
                summary_metrics = self.summarizer.update_model(
                    labeled_data["summaries"],
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    epochs=epochs
                )
                update_metrics["summarizer"] = summary_metrics
            except Exception as e:
                logger.error(f"Error updating summarizer: {str(e)}")

        # Register updated model with lifecycle manager
        if update_metrics:
            self.register_with_lifecycle_manager(version="1.0.0-updated")

            # Clear cache if we have one
            if self.use_cache and self.cache:
                try:
                    # Close and reinitialize cache to clear it
                    cache_dir = self.cache.cache_dir
                    max_size = self.cache.max_size_mb
                    self.cache.close()
                    self.cache = ModelCache(cache_dir=cache_dir, max_size_mb=max_size)
                    logger.info("Cache cleared after model update")
                except Exception as e:
                    logger.error(f"Error clearing cache: {str(e)}")

        return update_metrics

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
