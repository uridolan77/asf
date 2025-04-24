"""
Test script for the Medical Research Synthesizer.

This script demonstrates the usage of the new Medical Research Synthesizer
with its enhanced capabilities.
"""

import os
import sys
import logging
import argparse
import json
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Import synthesizer
from medical.ml.document_processing.medical_research_synthesizer_new import MedicalResearchSynthesizer


def print_progress(stage: str, progress: float):
    """Print progress updates."""
    if stage == "overall":
        print(f"Overall progress: {progress * 100:.1f}%")
    else:
        print(f"  {stage}: {progress * 100:.1f}%")


def print_streaming_result(stage: str, result: Any):
    """Print streaming results."""
    print(f"\nIntermediate result from {stage}:")
    
    if hasattr(result, 'title'):
        print(f"Title: {result.title}")
    
    if stage == "document_processing":
        print(f"Sections: {len(result.sections)}")
        for i, section in enumerate(result.sections[:3]):
            print(f"  Section {i+1}: {section.heading}")
        if len(result.sections) > 3:
            print(f"  ... and {len(result.sections) - 3} more sections")
    
    elif stage == "entity_extraction":
        print(f"Entities: {len(result.entities)}")
        for i, entity in enumerate(result.entities[:5]):
            print(f"  Entity {i+1}: {entity.text} ({entity.label})")
        if len(result.entities) > 5:
            print(f"  ... and {len(result.entities) - 5} more entities")
    
    elif stage == "relation_extraction":
        print(f"Relations: {len(result.relations)}")
        for i, relation in enumerate(result.relations[:5]):
            print(f"  Relation {i+1}: {relation.head_entity} → {relation.relation_type} → {relation.tail_entity}")
        if len(result.relations) > 5:
            print(f"  ... and {len(result.relations) - 5} more relations")
    
    elif stage == "summarization":
        print("Summary:")
        if isinstance(result.summary, dict):
            for key, value in result.summary.items():
                print(f"  {key.replace('_', ' ').title()}:")
                print(f"    {value[:100]}...")
        else:
            print(f"  {result.summary[:100]}...")


def print_batch_progress(progress: Dict[str, Dict[str, Any]]):
    """Print batch processing progress."""
    print("\nBatch processing progress:")
    
    # Count files by status
    status_counts = {"queued": 0, "processing": 0, "completed": 0, "failed": 0}
    for file_path, info in progress.items():
        status = info.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    
    # Print status counts
    for status, count in status_counts.items():
        if count > 0:
            print(f"  {status.capitalize()}: {count}")
    
    # Print details for in-progress files
    for file_path, info in progress.items():
        if info.get("status") == "processing":
            file_name = os.path.basename(file_path)
            progress_val = info.get("progress", 0)
            print(f"  {file_name}: {progress_val * 100:.1f}%")


def process_single_document(synthesizer, input_path: str, output_dir: str, streaming: bool = False):
    """Process a single document."""
    print(f"\nProcessing document: {input_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine if input is a PDF
    is_pdf = input_path.lower().endswith('.pdf')
    
    # Process document
    if streaming:
        print("Using streaming processing mode")
        doc_structure, metrics = synthesizer.process_streaming(
            input_path,
            is_pdf=is_pdf,
            streaming_callback=print_streaming_result,
            progress_callback=print_progress
        )
    else:
        print("Using standard processing mode with progress tracking")
        doc_structure, metrics = synthesizer.process_with_progress(
            input_path,
            is_pdf=is_pdf,
            progress_callback=print_progress
        )
    
    # Save results
    saved_files = synthesizer.save_results(
        doc_structure,
        output_dir,
        formats=["json", "csv", "txt", "html"]
    )
    
    print("\nProcessing complete!")
    print(f"Results saved to: {output_dir}")
    
    # Print performance metrics
    print("\nPerformance metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Print saved files
    print("\nSaved files:")
    for fmt, path in saved_files.items():
        print(f"  {fmt}: {path}")
    
    return doc_structure, metrics


def process_batch(synthesizer, input_dir: str, output_dir: str, batch_size: int = 4):
    """Process a batch of documents."""
    print(f"\nProcessing documents from: {input_dir}")
    
    # Get list of PDF files
    file_list = []
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.pdf'):
            file_list.append(os.path.join(input_dir, filename))
    
    if not file_list:
        print("No PDF files found in the input directory")
        return
    
    print(f"Found {len(file_list)} PDF files")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process batch
    metrics = synthesizer.process_batch(
        file_list=file_list,
        output_dir=output_dir,
        batch_size=batch_size,
        all_pdfs=True,
        save_results=True,
        progress_callback=print_batch_progress
    )
    
    print("\nBatch processing complete!")
    print(f"Results saved to: {output_dir}")
    
    # Print aggregate metrics
    print("\nAggregate metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    return metrics


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test the Medical Research Synthesizer")
    parser.add_argument("--input", required=True, help="Input file or directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--batch", action="store_true", help="Process in batch mode")
    parser.add_argument("--streaming", action="store_true", help="Use streaming processing")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for parallel processing")
    parser.add_argument("--cache-dir", default="cache", help="Cache directory")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    
    args = parser.parse_args()
    
    # Initialize synthesizer
    print("Initializing Medical Research Synthesizer...")
    synthesizer = MedicalResearchSynthesizer(
        use_cache=not args.no_cache,
        cache_dir=args.cache_dir
    )
    
    try:
        if args.batch:
            # Process batch
            process_batch(
                synthesizer,
                input_dir=args.input,
                output_dir=args.output,
                batch_size=args.batch_size
            )
        else:
            # Process single document
            process_single_document(
                synthesizer,
                input_path=args.input,
                output_dir=args.output,
                streaming=args.streaming
            )
    finally:
        # Close synthesizer
        synthesizer.close()


if __name__ == "__main__":
    main()
