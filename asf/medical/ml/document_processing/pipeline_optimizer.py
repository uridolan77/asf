"""
Pipeline Optimizer Module

This module provides optimization capabilities for the document processing pipeline,
including parallel processing, caching, and incremental processing.
"""

import os
import time
import logging
import hashlib
import json
import pickle
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache, wraps
import threading
from dataclasses import asdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PipelineOptimizer:
    """
    Optimizer for document processing pipeline.

    This class provides optimization capabilities including:
    1. Parallel processing of multiple documents
    2. Caching of intermediate results
    3. Incremental processing for faster updates
    4. Progress tracking and reporting
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        max_workers: int = None,
        use_processes: bool = False,
        chunk_size: int = 10,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ):
        """
        Initialize the pipeline optimizer.

        Args:
            cache_dir: Directory to store cache files
            use_cache: Whether to use caching
            max_workers: Maximum number of workers for parallel processing
            use_processes: Whether to use processes instead of threads
            chunk_size: Size of chunks for batch processing
            progress_callback: Callback function for progress updates
        """
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.max_workers = max_workers
        self.use_processes = use_processes
        self.chunk_size = chunk_size
        self.progress_callback = progress_callback

        # Create cache directory if it doesn't exist
        if use_cache and cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        # Initialize locks for thread safety
        self.cache_lock = threading.Lock()

        # Initialize progress tracking
        self.progress = {}
        self.progress_lock = threading.Lock()

    def compute_hash(self, data: Any) -> str:
        """
        Compute a hash for the input data.

        Args:
            data: Input data

        Returns:
            Hash string
        """
        if isinstance(data, str):
            # If data is a string (e.g., file path), use it directly
            content = data.encode('utf-8')
        elif isinstance(data, bytes):
            # If data is already bytes, use it directly
            content = data
        else:
            # Otherwise, serialize to JSON
            try:
                content = json.dumps(data, sort_keys=True).encode('utf-8')
            except (TypeError, ValueError):
                # If JSON serialization fails, use pickle
                content = pickle.dumps(data)

        # Compute hash
        return hashlib.md5(content).hexdigest()

    def get_cache_path(self, cache_key: str, stage: str) -> str:
        """
        Get the cache file path for a given key and stage.

        Args:
            cache_key: Cache key
            stage: Processing stage

        Returns:
            Cache file path
        """
        if not self.cache_dir:
            raise ValueError("Cache directory not specified")

        return os.path.join(self.cache_dir, f"{cache_key}_{stage}.pkl")

    def cache_result(self, cache_key: str, stage: str, result: Any) -> None:
        """
        Cache a result for a given key and stage.

        Args:
            cache_key: Cache key
            stage: Processing stage
            result: Result to cache
        """
        if not self.use_cache or not self.cache_dir:
            return

        cache_path = self.get_cache_path(cache_key, stage)

        with self.cache_lock:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(result, f)
                logger.debug(f"Cached result for {stage} at {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to cache result: {str(e)}")

    def get_cached_result(self, cache_key: str, stage: str) -> Optional[Any]:
        """
        Get a cached result for a given key and stage.

        Args:
            cache_key: Cache key
            stage: Processing stage

        Returns:
            Cached result or None if not found
        """
        if not self.use_cache or not self.cache_dir:
            return None

        cache_path = self.get_cache_path(cache_key, stage)

        with self.cache_lock:
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'rb') as f:
                        result = pickle.load(f)
                    logger.debug(f"Retrieved cached result for {stage} from {cache_path}")
                    return result
                except Exception as e:
                    logger.warning(f"Failed to load cached result: {str(e)}")

        return None

    def update_progress(self, task_id: str, progress: float) -> None:
        """
        Update progress for a task.

        Args:
            task_id: Task ID
            progress: Progress value (0-1)
        """
        with self.progress_lock:
            self.progress[task_id] = progress

            if self.progress_callback:
                self.progress_callback(task_id, progress)

    def get_progress(self, task_id: str) -> float:
        """
        Get progress for a task.

        Args:
            task_id: Task ID

        Returns:
            Progress value (0-1)
        """
        with self.progress_lock:
            return self.progress.get(task_id, 0.0)

    def process_with_cache(
        self,
        func: Callable,
        input_data: Any,
        cache_key: Optional[str] = None,
        stage: str = "default",
        force_recompute: bool = False,
        task_id: Optional[str] = None
    ) -> Any:
        """
        Process input data with caching.

        Args:
            func: Processing function
            input_data: Input data
            cache_key: Cache key (computed from input_data if not provided)
            stage: Processing stage
            force_recompute: Whether to force recomputation
            task_id: Task ID for progress tracking

        Returns:
            Processing result
        """
        # Compute cache key if not provided
        if cache_key is None:
            cache_key = self.compute_hash(input_data)

        # Check cache if not forcing recomputation
        if not force_recompute:
            cached_result = self.get_cached_result(cache_key, stage)
            if cached_result is not None:
                if task_id:
                    self.update_progress(task_id, 1.0)
                return cached_result

        # Process input data
        start_time = time.time()
        result = func(input_data)
        end_time = time.time()

        logger.debug(f"Processed {stage} in {end_time - start_time:.2f}s")

        # Cache result
        self.cache_result(cache_key, stage, result)

        # Update progress
        if task_id:
            self.update_progress(task_id, 1.0)

        return result

    def process_batch(
        self,
        func: Callable,
        batch: List[Any],
        task_ids: Optional[List[str]] = None,
        **kwargs
    ) -> List[Any]:
        """
        Process a batch of inputs in parallel.

        Args:
            func: Processing function
            batch: List of inputs
            task_ids: List of task IDs for progress tracking
            **kwargs: Additional arguments for process_with_cache

        Returns:
            List of results
        """
        if not batch:
            return []

        # Create executor
        executor_cls = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor

        results = []
        with executor_cls(max_workers=self.max_workers) as executor:
            # Submit tasks
            futures = []
            for i, input_data in enumerate(batch):
                task_id = task_ids[i] if task_ids and i < len(task_ids) else None

                # Wrap function to use caching
                def process_item(item):
                    return self.process_with_cache(func, item, task_id=task_id, **kwargs)

                futures.append(executor.submit(process_item, input_data))

            # Collect results
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing batch item: {str(e)}")
                    results.append(None)

        return results

    def process_large_batch(
        self,
        func: Callable,
        batch: List[Any],
        task_ids: Optional[List[str]] = None,
        **kwargs
    ) -> List[Any]:
        """
        Process a large batch of inputs by splitting into chunks.

        Args:
            func: Processing function
            batch: List of inputs
            task_ids: List of task IDs for progress tracking
            **kwargs: Additional arguments for process_batch

        Returns:
            List of results
        """
        if not batch:
            return []

        # Split batch into chunks
        chunks = [batch[i:i + self.chunk_size] for i in range(0, len(batch), self.chunk_size)]

        # Split task IDs into chunks if provided
        task_id_chunks = None
        if task_ids:
            task_id_chunks = [task_ids[i:i + self.chunk_size] for i in range(0, len(task_ids), self.chunk_size)]

        # Process chunks
        results = []
        for i, chunk in enumerate(chunks):
            chunk_task_ids = task_id_chunks[i] if task_id_chunks else None
            chunk_results = self.process_batch(func, chunk, task_ids=chunk_task_ids, **kwargs)
            results.extend(chunk_results)

        return results

    def incremental_process(
        self,
        func: Callable,
        input_data: Any,
        previous_result: Any,
        diff_func: Callable[[Any, Any], bool],
        task_id: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Process input data incrementally if it has changed.

        Args:
            func: Processing function
            input_data: Input data
            previous_result: Previous result
            diff_func: Function to check if input has changed
            task_id: Task ID for progress tracking
            **kwargs: Additional arguments for process_with_cache

        Returns:
            Processing result
        """
        # Check if input has changed
        if previous_result is not None and not diff_func(input_data, previous_result):
            logger.debug("Input has not changed, reusing previous result")
            if task_id:
                self.update_progress(task_id, 1.0)
            return previous_result

        # Process input data
        return self.process_with_cache(func, input_data, task_id=task_id, **kwargs)

    def cached(self, stage: str = "default", key_func: Optional[Callable] = None):
        """
        Decorator for caching function results.

        Args:
            stage: Processing stage
            key_func: Function to compute cache key from arguments

        Returns:
            Decorated function
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Compute cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    # Use first argument as basis for cache key
                    cache_key = self.compute_hash(args[0] if args else kwargs)

                # Force recomputation if specified
                force_recompute = kwargs.pop('force_recompute', False)

                # Get task ID if provided
                task_id = kwargs.pop('task_id', None)

                # Process with cache
                return self.process_with_cache(
                    lambda x: func(*args, **kwargs),
                    args[0] if args else kwargs,
                    cache_key=cache_key,
                    stage=stage,
                    force_recompute=force_recompute,
                    task_id=task_id
                )
            return wrapper
        return decorator


class StreamingPipeline:
    """
    Streaming pipeline for incremental document processing.

    This class enables streaming results as they become available,
    rather than waiting for the entire pipeline to complete.
    """

    def __init__(
        self,
        optimizer: PipelineOptimizer,
        stages: List[Tuple[str, Callable]],
        callback: Optional[Callable[[str, Any], None]] = None
    ):
        """
        Initialize the streaming pipeline.

        Args:
            optimizer: Pipeline optimizer
            stages: List of (stage_name, stage_func) tuples
            callback: Callback function for stage results
        """
        self.optimizer = optimizer
        self.stages = stages
        self.callback = callback

        # Initialize stage results
        self.stage_results = {}
        self.stage_lock = threading.Lock()

    def process(
        self,
        input_data: Any,
        task_id: Optional[str] = None,
        force_recompute: bool = False
    ) -> Dict[str, Any]:
        """
        Process input data through the pipeline stages.

        Args:
            input_data: Input data
            task_id: Task ID for progress tracking
            force_recompute: Whether to force recomputation

        Returns:
            Dictionary of stage results
        """
        # Compute cache key
        cache_key = self.optimizer.compute_hash(input_data)

        # Initialize stage results
        with self.stage_lock:
            self.stage_results[cache_key] = {}

        # Process each stage
        current_input = input_data
        for i, (stage_name, stage_func) in enumerate(self.stages):
            # Update progress
            if task_id:
                progress = (i / len(self.stages)) * 0.9  # Reserve 10% for final stage
                self.optimizer.update_progress(task_id, progress)

            # Process stage
            try:
                stage_result = self.optimizer.process_with_cache(
                    stage_func,
                    current_input,
                    cache_key=f"{cache_key}_{i}",
                    stage=stage_name,
                    force_recompute=force_recompute,
                    task_id=None  # Don't update progress for individual stages
                )

                # Store stage result
                with self.stage_lock:
                    self.stage_results[cache_key][stage_name] = stage_result

                # Call callback if provided
                if self.callback:
                    self.callback(stage_name, stage_result)

                # Use stage result as input for next stage
                current_input = stage_result
            except Exception as e:
                logger.error(f"Error in stage {stage_name}: {str(e)}")
                # Continue with next stage using previous input

        # Update final progress
        if task_id:
            self.optimizer.update_progress(task_id, 1.0)

        # Return all stage results
        with self.stage_lock:
            return self.stage_results.get(cache_key, {})

    def get_stage_result(
        self,
        cache_key: str,
        stage_name: str
    ) -> Optional[Any]:
        """
        Get result for a specific stage.

        Args:
            cache_key: Cache key
            stage_name: Stage name

        Returns:
            Stage result or None if not available
        """
        with self.stage_lock:
            stage_results = self.stage_results.get(cache_key, {})
            return stage_results.get(stage_name)

    def get_all_results(self, cache_key: str) -> Dict[str, Any]:
        """
        Get all stage results.

        Args:
            cache_key: Cache key

        Returns:
            Dictionary of stage results
        """
        with self.stage_lock:
            return self.stage_results.get(cache_key, {})


def optimize_document_processor(processor, optimizer):
    """
    Optimize a document processor using the pipeline optimizer.

    Args:
        processor: Document processor
        optimizer: Pipeline optimizer

    Returns:
        Optimized document processor
    """
    # Cache document processing
    processor.process_document = optimizer.cached(stage="document_processing")(processor.process_document)

    # Cache PDF extraction
    if hasattr(processor, 'extract_pdf_text'):
        original_extract_pdf_text = processor.extract_pdf_text

        @optimizer.cached(stage="pdf_extraction")
        def optimized_extract_pdf_text(pdf_path):
            return original_extract_pdf_text(pdf_path)

        processor.extract_pdf_text = optimized_extract_pdf_text

    # Cache section classification if available
    if hasattr(processor, 'classify_sections'):
        original_classify_sections = processor.classify_sections

        @optimizer.cached(stage="section_classification")
        def optimized_classify_sections(sections):
            return original_classify_sections(sections)

        processor.classify_sections = optimized_classify_sections

    # Cache subsection identification if available
    if hasattr(processor, 'identify_subsections'):
        original_identify_subsections = processor.identify_subsections

        @optimizer.cached(stage="subsection_identification")
        def optimized_identify_subsections(sections):
            return original_identify_subsections(sections)

        processor.identify_subsections = optimized_identify_subsections

    return processor


def optimize_entity_extractor(extractor, optimizer):
    """
    Optimize an entity extractor using the pipeline optimizer.

    Args:
        extractor: Entity extractor
        optimizer: Pipeline optimizer

    Returns:
        Optimized entity extractor
    """
    # Cache entity extraction
    extractor.process_document = optimizer.cached(stage="entity_extraction")(extractor.process_document)

    # Cache entity recognition if available
    if hasattr(extractor, 'recognize_entities'):
        original_recognize_entities = extractor.recognize_entities

        @optimizer.cached(stage="entity_recognition")
        def optimized_recognize_entities(text):
            return original_recognize_entities(text)

        extractor.recognize_entities = optimized_recognize_entities

    # Cache entity linking if available
    if hasattr(extractor, 'link_entities'):
        original_link_entities = extractor.link_entities

        @optimizer.cached(stage="entity_linking")
        def optimized_link_entities(entities):
            return original_link_entities(entities)

        extractor.link_entities = optimized_link_entities

    return extractor


def optimize_relation_extractor(extractor, optimizer):
    """
    Optimize a relation extractor using the pipeline optimizer.

    Args:
        extractor: Relation extractor
        optimizer: Pipeline optimizer

    Returns:
        Optimized relation extractor
    """
    # Cache relation extraction
    extractor.process_document = optimizer.cached(stage="relation_extraction")(extractor.process_document)

    # Cache relation extraction from text if available
    if hasattr(extractor, 'extract_relations_from_text'):
        original_extract_relations = extractor.extract_relations_from_text

        @optimizer.cached(stage="relation_extraction_from_text")
        def optimized_extract_relations(text, entities, threshold=0.5, sentences=None):
            return original_extract_relations(text, entities, threshold, sentences)

        extractor.extract_relations_from_text = optimized_extract_relations

    # Cache knowledge graph construction if available
    if hasattr(extractor, 'build_knowledge_graph'):
        original_build_kg = extractor.build_knowledge_graph

        @optimizer.cached(stage="knowledge_graph_construction")
        def optimized_build_kg(relations):
            return original_build_kg(relations)

        extractor.build_knowledge_graph = optimized_build_kg

    return extractor


def optimize_summarizer(summarizer, optimizer):
    """
    Optimize a research summarizer using the pipeline optimizer.

    Args:
        summarizer: Research summarizer
        optimizer: Pipeline optimizer

    Returns:
        Optimized summarizer
    """
    # Cache summarization
    summarizer.process_document = optimizer.cached(stage="summarization")(summarizer.process_document)

    # Cache abstract summarization if available
    if hasattr(summarizer, 'summarize_abstract'):
        original_summarize_abstract = summarizer.summarize_abstract

        @optimizer.cached(stage="abstract_summarization")
        def optimized_summarize_abstract(abstract):
            return original_summarize_abstract(abstract)

        summarizer.summarize_abstract = optimized_summarize_abstract

    # Cache section summarization if available
    if hasattr(summarizer, 'summarize_section'):
        original_summarize_section = summarizer.summarize_section

        @optimizer.cached(stage="section_summarization")
        def optimized_summarize_section(section):
            return original_summarize_section(section)

        summarizer.summarize_section = optimized_summarize_section

    # Cache factual consistency checking if available
    if hasattr(summarizer, 'check_factual_consistency'):
        original_check_consistency = summarizer.check_factual_consistency

        @optimizer.cached(stage="factual_consistency_checking")
        def optimized_check_consistency(summary, source):
            return original_check_consistency(summary, source)

        summarizer.check_factual_consistency = optimized_check_consistency

    return summarizer
