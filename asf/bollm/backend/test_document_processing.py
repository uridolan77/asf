"""
Test script for document processing.

This script tests the document processing functionality by directly calling
the MedicalResearchSynthesizer class.
"""

import os
import sys
import logging
import json
from datetime import datetime

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the document processing module
try:
    from asf.medical.ml.document_processing import MedicalResearchSynthesizer, EnhancedMedicalResearchSynthesizer
    from asf.medical.ml.document_processing.document_structure import DocumentStructure
except ImportError as e:
    logger.error(f"Error importing document processing module: {e}")
    sys.exit(1)

def process_document(file_path, use_enhanced=True, use_streaming=True):
    """
    Process a document using the MedicalResearchSynthesizer.
    
    Args:
        file_path: Path to the document to process
        use_enhanced: Whether to use the enhanced synthesizer
        use_streaming: Whether to use streaming processing
        
    Returns:
        The processed document structure
    """
    logger.info(f"Processing document: {file_path}")
    logger.info(f"Using enhanced synthesizer: {use_enhanced}")
    logger.info(f"Using streaming: {use_streaming}")
    
    # Create result directory
    result_dir = os.path.join(os.path.dirname(file_path), "results")
    os.makedirs(result_dir, exist_ok=True)
    
    # Initialize synthesizer
    if use_enhanced:
        logger.info("Initializing EnhancedMedicalResearchSynthesizer")
        synthesizer = EnhancedMedicalResearchSynthesizer()
    else:
        logger.info("Initializing MedicalResearchSynthesizer")
        synthesizer = MedicalResearchSynthesizer()
    
    # Define callback functions
    def progress_callback(stage, progress):
        logger.info(f"Progress: {stage} - {progress:.2f}")
    
    def streaming_callback(stage, result):
        logger.info(f"Streaming result: {stage}")
        
        # Try to get entity and relation counts
        if hasattr(result, 'entities'):
            logger.info(f"Entities: {len(result.entities)}")
        if hasattr(result, 'relations'):
            logger.info(f"Relations: {len(result.relations)}")
    
    # Process the document
    try:
        start_time = datetime.now()
        
        # Check if the file is a PDF
        is_pdf = file_path.lower().endswith('.pdf')
        
        if use_enhanced and use_streaming:
            logger.info("Using streaming processing")
            doc_structure, metrics = synthesizer.process_streaming(
                file_path,
                is_pdf=is_pdf,
                streaming_callback=streaming_callback,
                progress_callback=progress_callback
            )
        elif use_enhanced:
            logger.info("Using enhanced processing with progress tracking")
            doc_structure, metrics = synthesizer.process_with_progress(
                file_path,
                is_pdf=is_pdf,
                progress_callback=progress_callback
            )
        else:
            logger.info("Using standard processing")
            doc_structure, metrics = synthesizer.process(file_path, is_pdf=is_pdf)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Log results
        logger.info(f"Processing completed in {processing_time:.2f} seconds")
        
        # Check if we have any entities or relations
        entity_count = len(doc_structure.entities) if hasattr(doc_structure, 'entities') else 0
        relation_count = len(doc_structure.relations) if hasattr(doc_structure, 'relations') else 0
        
        logger.info(f"Entities: {entity_count}")
        logger.info(f"Relations: {relation_count}")
        
        # Save results
        logger.info(f"Saving results to {result_dir}")
        synthesizer.save_results(doc_structure, result_dir)
        
        # Save document structure as JSON
        result_file_path = os.path.join(result_dir, "document_structure.json")
        with open(result_file_path, "w") as f:
            # Convert document structure to dict if possible
            if hasattr(doc_structure, 'to_dict') and callable(getattr(doc_structure, 'to_dict')):
                json.dump(doc_structure.to_dict(), f, indent=2, default=str)
            else:
                # Use default JSON serialization
                json.dump(doc_structure.__dict__, f, indent=2, default=str)
        
        logger.info(f"Results saved to {result_file_path}")
        
        return doc_structure
        
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise
    finally:
        # Close the synthesizer
        synthesizer.close()

def main():
    """
    Main function.
    """
    # Check if a file path was provided
    if len(sys.argv) < 2:
        logger.error("Please provide a file path to process")
        logger.info("Usage: python test_document_processing.py <file_path> [--no-enhanced] [--no-streaming]")
        sys.exit(1)
    
    # Get file path
    file_path = sys.argv[1]
    
    # Check if the file exists
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        sys.exit(1)
    
    # Get options
    use_enhanced = "--no-enhanced" not in sys.argv
    use_streaming = "--no-streaming" not in sys.argv
    
    # Process the document
    try:
        doc_structure = process_document(file_path, use_enhanced, use_streaming)
        logger.info("Document processing completed successfully")
    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
