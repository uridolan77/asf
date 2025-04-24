"""
Batch Processor Module

This module provides functionality for batch processing of medical research papers,
with configurable parallelism, progress tracking, and error handling.
"""

import os
import time
import logging
import concurrent.futures
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
import threading
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BatchProcessor:
    """
    Batch processor for medical research papers.
    
    This class provides functionality for processing multiple documents in parallel,
    with configurable batch size, progress tracking, and error handling.
    """
    
    def __init__(
        self,
        synthesizer,
        max_workers: int = None,
        batch_size: int = 4,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """
        Initialize the batch processor.
        
        Args:
            synthesizer: Medical research synthesizer
            max_workers: Maximum number of workers for parallel processing
            batch_size: Size of batches for processing
            progress_callback: Callback function for progress updates
        """
        self.synthesizer = synthesizer
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.progress_callback = progress_callback
        
        # Initialize progress tracking
        self.progress = {}
        self.progress_lock = threading.Lock()
    
    def update_progress(self, file_path: str, status: str, progress: float, metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        Update progress for a file.
        
        Args:
            file_path: File path
            status: Status (queued, processing, completed, failed)
            progress: Progress value (0-1)
            metrics: Performance metrics
        """
        with self.progress_lock:
            self.progress[file_path] = {
                "status": status,
                "progress": progress,
                "metrics": metrics or {}
            }
            
            if self.progress_callback:
                self.progress_callback(self.progress)
    
    def get_progress(self) -> Dict[str, Dict[str, Any]]:
        """
        Get progress for all files.
        
        Returns:
            Dictionary mapping file paths to progress information
        """
        with self.progress_lock:
            return self.progress.copy()
    
    def process_batch(
        self,
        file_list: List[str],
        output_dir: str,
        all_pdfs: bool = True,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Process a batch of documents with configurable parallelism.
        
        Args:
            file_list: List of file paths
            output_dir: Base output directory
            all_pdfs: Whether all files are PDFs
            save_results: Whether to save results to disk
            
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
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
        
        # Initialize progress tracking
        for file_path in file_list:
            self.update_progress(file_path, "queued", 0.0)
        
        # Process files in batches
        for i in range(0, len(file_list), self.batch_size):
            batch = file_list[i:i + self.batch_size]
            batch_results = self._process_batch_parallel(batch, output_dir, all_pdfs, save_results)
            
            # Update aggregate metrics
            for result in batch_results:
                if result["status"] == "success":
                    aggregate_metrics["successful"] += 1
                    aggregate_metrics["entities_total"] += result.get("entity_count", 0)
                    aggregate_metrics["relations_total"] += result.get("relation_count", 0)
                    aggregate_metrics["processing_times"].append(result["metrics"].get("total_processing_time", 0))
                else:
                    aggregate_metrics["failed"] += 1
        
        # Calculate aggregate statistics
        aggregate_metrics["total_processing_time"] = time.time() - overall_start
        
        if aggregate_metrics["processing_times"]:
            aggregate_metrics["avg_processing_time"] = sum(aggregate_metrics["processing_times"]) / len(aggregate_metrics["processing_times"])
            aggregate_metrics["min_processing_time"] = min(aggregate_metrics["processing_times"])
            aggregate_metrics["max_processing_time"] = max(aggregate_metrics["processing_times"])
        
        # Save aggregate metrics
        if save_results:
            metrics_path = os.path.join(output_dir, "batch_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(aggregate_metrics, f, indent=2)
        
        return aggregate_metrics
    
    def _process_batch_parallel(
        self,
        batch: List[str],
        output_dir: str,
        all_pdfs: bool,
        save_results: bool
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of documents in parallel.
        
        Args:
            batch: List of file paths
            output_dir: Base output directory
            all_pdfs: Whether all files are PDFs
            save_results: Whether to save results to disk
            
        Returns:
            List of processing results
        """
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks
            future_to_file = {}
            for file_path in batch:
                future = executor.submit(
                    self._process_single_file,
                    file_path,
                    output_dir,
                    all_pdfs,
                    save_results
                )
                future_to_file[future] = file_path
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
                    results.append({
                        "file": file_path,
                        "status": "failed",
                        "error": str(e)
                    })
                    self.update_progress(file_path, "failed", 1.0, {"error": str(e)})
        
        return results
    
    def _process_single_file(
        self,
        file_path: str,
        output_dir: str,
        is_pdf: bool,
        save_results: bool
    ) -> Dict[str, Any]:
        """
        Process a single file.
        
        Args:
            file_path: File path
            output_dir: Base output directory
            is_pdf: Whether the file is a PDF
            save_results: Whether to save results to disk
            
        Returns:
            Processing result
        """
        try:
            # Update progress
            self.update_progress(file_path, "processing", 0.1)
            
            doc_name = os.path.basename(file_path).split('.')[0]
            doc_output_dir = os.path.join(output_dir, doc_name)
            
            # Create progress callback for this file
            def file_progress_callback(stage: str, progress: float):
                # Map stage progress to overall progress (10% to 90%)
                overall_progress = 0.1 + (progress * 0.8)
                self.update_progress(file_path, "processing", overall_progress)
            
            # Process document
            doc_structure, metrics = self.synthesizer.process_with_progress(
                file_path,
                is_pdf=is_pdf,
                progress_callback=file_progress_callback
            )
            
            # Save results
            if save_results:
                os.makedirs(doc_output_dir, exist_ok=True)
                self.synthesizer.save_results(doc_structure, doc_output_dir)
            
            # Update progress
            self.update_progress(file_path, "completed", 1.0, metrics)
            
            return {
                "file": file_path,
                "status": "success",
                "metrics": metrics,
                "entity_count": len(doc_structure.entities),
                "relation_count": len(doc_structure.relations)
            }
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            self.update_progress(file_path, "failed", 1.0, {"error": str(e)})
            return {
                "file": file_path,
                "status": "failed",
                "error": str(e)
            }
    
    def process_streaming(
        self,
        file_list: List[str],
        output_dir: str,
        all_pdfs: bool = True,
        save_intermediate: bool = True,
        callback: Optional[Callable[[str, str, Any], None]] = None
    ) -> Dict[str, Any]:
        """
        Process a batch of documents with streaming results.
        
        Args:
            file_list: List of file paths
            output_dir: Base output directory
            all_pdfs: Whether all files are PDFs
            save_intermediate: Whether to save intermediate results
            callback: Callback function for intermediate results
            
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
        
        # Initialize progress tracking
        for file_path in file_list:
            self.update_progress(file_path, "queued", 0.0)
        
        # Process files in batches
        for i in range(0, len(file_list), self.batch_size):
            batch = file_list[i:i + self.batch_size]
            
            # Process batch with streaming results
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit tasks
                future_to_file = {}
                for file_path in batch:
                    future = executor.submit(
                        self._process_streaming_file,
                        file_path,
                        output_dir,
                        all_pdfs,
                        save_intermediate,
                        callback
                    )
                    future_to_file[future] = file_path
                
                # Collect results
                for future in concurrent.futures.as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        result = future.result()
                        if result["status"] == "success":
                            aggregate_metrics["successful"] += 1
                            aggregate_metrics["entities_total"] += result.get("entity_count", 0)
                            aggregate_metrics["relations_total"] += result.get("relation_count", 0)
                            aggregate_metrics["processing_times"].append(result["metrics"].get("total_processing_time", 0))
                        else:
                            aggregate_metrics["failed"] += 1
                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {str(e)}")
                        aggregate_metrics["failed"] += 1
        
        # Calculate aggregate statistics
        aggregate_metrics["total_processing_time"] = time.time() - overall_start
        
        if aggregate_metrics["processing_times"]:
            aggregate_metrics["avg_processing_time"] = sum(aggregate_metrics["processing_times"]) / len(aggregate_metrics["processing_times"])
            aggregate_metrics["min_processing_time"] = min(aggregate_metrics["processing_times"])
            aggregate_metrics["max_processing_time"] = max(aggregate_metrics["processing_times"])
        
        # Save aggregate metrics
        metrics_path = os.path.join(output_dir, "batch_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(aggregate_metrics, f, indent=2)
        
        return aggregate_metrics
    
    def _process_streaming_file(
        self,
        file_path: str,
        output_dir: str,
        is_pdf: bool,
        save_intermediate: bool,
        callback: Optional[Callable[[str, str, Any], None]]
    ) -> Dict[str, Any]:
        """
        Process a single file with streaming results.
        
        Args:
            file_path: File path
            output_dir: Base output directory
            is_pdf: Whether the file is a PDF
            save_intermediate: Whether to save intermediate results
            callback: Callback function for intermediate results
            
        Returns:
            Processing result
        """
        try:
            # Update progress
            self.update_progress(file_path, "processing", 0.1)
            
            doc_name = os.path.basename(file_path).split('.')[0]
            doc_output_dir = os.path.join(output_dir, doc_name)
            
            if save_intermediate:
                os.makedirs(doc_output_dir, exist_ok=True)
            
            # Create streaming callback
            def streaming_callback(stage: str, result: Any):
                # Call user callback if provided
                if callback:
                    callback(file_path, stage, result)
                
                # Save intermediate result if requested
                if save_intermediate:
                    stage_path = os.path.join(doc_output_dir, f"{stage}.json")
                    try:
                        with open(stage_path, 'w') as f:
                            if hasattr(result, 'to_dict'):
                                json.dump(result.to_dict(), f, indent=2)
                            elif hasattr(result, '__dict__'):
                                json.dump(result.__dict__, f, indent=2)
                            else:
                                json.dump(str(result), f, indent=2)
                    except Exception as e:
                        logger.warning(f"Failed to save intermediate result: {str(e)}")
            
            # Create progress callback
            def progress_callback(stage: str, progress: float):
                # Map stage progress to overall progress (10% to 90%)
                stage_weights = {
                    "document_processing": 0.3,
                    "entity_extraction": 0.3,
                    "relation_extraction": 0.3,
                    "summarization": 0.1
                }
                
                # Calculate overall progress
                stage_weight = stage_weights.get(stage, 0.25)
                stage_start = sum([w for s, w in stage_weights.items() if s < stage])
                overall_progress = 0.1 + (stage_start + (progress * stage_weight)) * 0.8
                
                self.update_progress(file_path, "processing", overall_progress)
            
            # Process document with streaming
            doc_structure, metrics = self.synthesizer.process_streaming(
                file_path,
                is_pdf=is_pdf,
                streaming_callback=streaming_callback,
                progress_callback=progress_callback
            )
            
            # Save final results
            if save_intermediate:
                self.synthesizer.save_results(doc_structure, doc_output_dir)
            
            # Update progress
            self.update_progress(file_path, "completed", 1.0, metrics)
            
            return {
                "file": file_path,
                "status": "success",
                "metrics": metrics,
                "entity_count": len(doc_structure.entities),
                "relation_count": len(doc_structure.relations)
            }
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            self.update_progress(file_path, "failed", 1.0, {"error": str(e)})
            return {
                "file": file_path,
                "status": "failed",
                "error": str(e)
            }
