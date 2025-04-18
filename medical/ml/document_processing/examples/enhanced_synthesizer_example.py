"""
Enhanced Medical Research Synthesizer Example

This example demonstrates the enhanced features of the Medical Research Synthesizer:
1. Parallel processing
2. Batch processing
3. Caching
4. Online learning
"""

import os
import time
import logging
from typing import List, Dict, Any

from asf.medical.ml.document_processing import MedicalResearchSynthesizer
from asf.medical.ml.document_processing.document_structure import DocumentStructure

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Sample text for demonstration
SAMPLE_TEXT = """
# Efficacy of Novel Antiviral Treatment for COVID-19

## Abstract
This study evaluates the efficacy of a novel antiviral treatment for COVID-19. 
In a randomized controlled trial with 500 patients, we found that the treatment 
reduced hospitalization rates by 45% and decreased viral load significantly 
faster than the control group. Side effects were minimal and transient.

## Introduction
The COVID-19 pandemic has necessitated rapid development of effective treatments.
Antiviral medications that can reduce viral replication present a promising approach.

## Methods
We conducted a double-blind, placebo-controlled trial across 12 medical centers.
Patients were randomized to receive either the novel antiviral or placebo within
48 hours of symptom onset. Primary endpoints were hospitalization rate and time to
viral clearance.

## Results
Treatment with the novel antiviral reduced hospitalization rates from 9.8% to 5.4%
(p<0.001). Median time to viral clearance was 5.8 days in the treatment group versus
9.1 days in the control group. Adverse events were reported in 12% of treatment recipients
versus 10.8% of placebo recipients.

## Discussion
These results indicate that early treatment with the novel antiviral significantly
reduces disease severity and viral shedding duration. The safety profile appears
favorable for widespread use.

## Conclusion
The novel antiviral treatment shows promise as an early intervention for COVID-19,
potentially reducing healthcare burden and transmission rates.
"""


def demonstrate_basic_processing():
    """Demonstrate basic processing with the enhanced synthesizer."""
    logger.info("Initializing Medical Research Synthesizer with enhanced features...")
    
    # Initialize the synthesizer with caching enabled
    synthesizer = MedicalResearchSynthesizer(
        use_cache=True,
        cache_dir="demo_cache",
        cache_size_mb=500
    )
    
    # Process the sample text
    logger.info("Processing sample text (first run, no cache)...")
    start_time = time.time()
    result, metrics = synthesizer.process(SAMPLE_TEXT)
    
    # Print basic information and metrics
    print("\n" + "="*80)
    print(f"Title: {result.title}")
    print(f"Entities: {len(result.entities)}")
    print(f"Relations: {len(result.relations)}")
    print(f"Processing time: {metrics['total_processing_time']:.2f} seconds")
    print(f"Cache hits: {metrics.get('cache_hits', 0)}")
    print("="*80 + "\n")
    
    # Process the same text again to demonstrate caching
    logger.info("Processing sample text again (should use cache)...")
    start_time = time.time()
    cached_result, cached_metrics = synthesizer.process(SAMPLE_TEXT)
    
    # Print cached processing information
    print("\n" + "="*80)
    print(f"Title: {cached_result.title}")
    print(f"Entities: {len(cached_result.entities)}")
    print(f"Relations: {len(cached_result.relations)}")
    print(f"Processing time: {cached_metrics['total_processing_time']:.2f} seconds")
    print(f"Cache hits: {cached_metrics.get('cache_hits', 0)}")
    print(f"Cache hit: {cached_metrics.get('cache_hit', False)}")
    print("="*80 + "\n")
    
    return result, metrics


def demonstrate_parallel_processing():
    """Demonstrate parallel processing capabilities."""
    logger.info("Initializing Medical Research Synthesizer for parallel processing...")
    
    # Initialize the synthesizer
    synthesizer = MedicalResearchSynthesizer()
    
    # Process the sample text with standard sequential processing
    logger.info("Processing with standard sequential approach...")
    start_time = time.time()
    _, standard_metrics = synthesizer.process(SAMPLE_TEXT)
    standard_time = standard_metrics["total_processing_time"]
    
    # Process the sample text with parallel processing
    logger.info("Processing with parallel approach...")
    start_time = time.time()
    _, parallel_metrics = synthesizer.process_parallel(SAMPLE_TEXT)
    parallel_time = parallel_metrics["total_processing_time"]
    
    # Print comparison
    print("\n" + "="*80)
    print("PARALLEL PROCESSING COMPARISON")
    print(f"Standard processing time: {standard_time:.2f} seconds")
    print(f"Parallel processing time: {parallel_time:.2f} seconds")
    print(f"Speedup: {standard_time/parallel_time:.2f}x")
    print("="*80 + "\n")


def demonstrate_batch_processing():
    """Demonstrate batch processing capabilities."""
    logger.info("Initializing Medical Research Synthesizer for batch processing...")
    
    # Initialize the synthesizer
    synthesizer = MedicalResearchSynthesizer()
    
    # Create a batch of sample texts with slight variations
    batch_texts = []
    for i in range(5):
        # Add a unique identifier to each sample to make them different
        modified_text = SAMPLE_TEXT + f"\n\nSample ID: {i+1}"
        
        # Save to temporary file
        file_path = f"temp_sample_{i+1}.txt"
        with open(file_path, "w") as f:
            f.write(modified_text)
        
        batch_texts.append(file_path)
    
    # Process the batch
    logger.info(f"Batch processing {len(batch_texts)} documents...")
    output_dir = "batch_output"
    batch_metrics = synthesizer.process_batch(
        file_list=batch_texts,
        output_dir=output_dir,
        batch_size=2,  # Process 2 documents in parallel
        all_pdfs=False  # These are text files, not PDFs
    )
    
    # Print batch processing results
    print("\n" + "="*80)
    print("BATCH PROCESSING RESULTS")
    print(f"Total documents: {batch_metrics['total_documents']}")
    print(f"Successfully processed: {batch_metrics['successful']}")
    print(f"Failed: {batch_metrics['failed']}")
    print(f"Total entities extracted: {batch_metrics['entities_total']}")
    print(f"Total relations extracted: {batch_metrics['relations_total']}")
    print(f"Average processing time: {batch_metrics.get('avg_document_time', 0):.2f} seconds")
    print(f"Total batch time: {batch_metrics['total_processing_time']:.2f} seconds")
    print("="*80 + "\n")
    
    # Clean up temporary files
    for file_path in batch_texts:
        if os.path.exists(file_path):
            os.remove(file_path)


def demonstrate_online_learning():
    """Demonstrate online learning capabilities."""
    logger.info("Initializing Medical Research Synthesizer for online learning...")
    
    # Initialize the synthesizer
    synthesizer = MedicalResearchSynthesizer()
    
    # Process a document to get initial results
    result, _ = synthesizer.process(SAMPLE_TEXT)
    
    # Print initial entities
    print("\n" + "="*80)
    print("INITIAL ENTITY EXTRACTION")
    for i, entity in enumerate(result.entities[:5]):  # Show first 5 entities
        print(f"{i+1}. {entity.text} ({entity.label})")
    print("="*80 + "\n")
    
    # Create labeled data for online learning
    labeled_data = {
        "entities": [
            {
                "text": "COVID-19",
                "label": "DISEASE",
                "start": 50,
                "end": 58,
                "context": "This study evaluates the efficacy of a novel antiviral treatment for COVID-19."
            },
            {
                "text": "novel antiviral",
                "label": "MEDICATION",
                "start": 35,
                "end": 50,
                "context": "This study evaluates the efficacy of a novel antiviral treatment for COVID-19."
            },
            {
                "text": "viral clearance",
                "label": "CLINICAL_FINDING",
                "start": 300,
                "end": 314,
                "context": "Primary endpoints were hospitalization rate and time to viral clearance."
            }
        ],
        "relations": [
            {
                "head": "novel antiviral",
                "tail": "COVID-19",
                "relation": "TREATS",
                "context": "This study evaluates the efficacy of a novel antiviral treatment for COVID-19."
            }
        ]
    }
    
    # Update models with labeled data
    logger.info("Updating models with labeled data...")
    update_metrics = synthesizer.update_models(
        labeled_data=labeled_data,
        learning_rate=2e-5,
        batch_size=2,
        epochs=3
    )
    
    # Print update metrics
    print("\n" + "="*80)
    print("MODEL UPDATE RESULTS")
    for component, metrics in update_metrics.items():
        print(f"{component}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
    print("="*80 + "\n")
    
    # Process the same document again to see if results improved
    updated_result, _ = synthesizer.process(SAMPLE_TEXT)
    
    # Print updated entities
    print("\n" + "="*80)
    print("UPDATED ENTITY EXTRACTION")
    for i, entity in enumerate(updated_result.entities[:5]):  # Show first 5 entities
        print(f"{i+1}. {entity.text} ({entity.label})")
    print("="*80 + "\n")


def main():
    """Run the enhanced Medical Research Synthesizer example."""
    print("\n" + "="*80)
    print("ENHANCED MEDICAL RESEARCH SYNTHESIZER EXAMPLE")
    print("="*80 + "\n")
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Demonstrate basic processing with caching
    demonstrate_basic_processing()
    
    # Demonstrate parallel processing
    demonstrate_parallel_processing()
    
    # Demonstrate batch processing
    demonstrate_batch_processing()
    
    # Demonstrate online learning
    demonstrate_online_learning()
    
    print("\nAll demonstrations completed successfully!")


if __name__ == "__main__":
    main()
