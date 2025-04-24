"""
Minimal document processor script.

This script demonstrates the document processing functionality without requiring
the full backend server. It can be used to test the document processing pipeline
with minimal dependencies.
"""

import os
import sys
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def process_document(file_path):
    """
    Process a document using PyMuPDF or PDFMiner.Six.
    
    This function demonstrates the minimal document processing functionality
    without requiring the full medical research synthesizer pipeline.
    
    Args:
        file_path: Path to the document to process
        
    Returns:
        Extracted text and metadata
    """
    logger.info(f"Processing document: {file_path}")
    
    # Check if the file exists
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None
    
    # Check if the file is a PDF
    is_pdf = file_path.lower().endswith('.pdf')
    if not is_pdf:
        logger.error(f"Only PDF files are supported: {file_path}")
        return None
    
    # Try to extract text using PyMuPDF (fitz)
    try:
        import fitz
        logger.info("Using PyMuPDF (fitz) for text extraction")
        
        doc = fitz.open(file_path)
        text = ""
        metadata = {
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "subject": doc.metadata.get("subject", ""),
            "keywords": doc.metadata.get("keywords", ""),
            "creator": doc.metadata.get("creator", ""),
            "producer": doc.metadata.get("producer", ""),
            "page_count": len(doc),
            "file_size": os.path.getsize(file_path)
        }
        
        for page_num, page in enumerate(doc):
            text += f"--- Page {page_num + 1} ---\n"
            text += page.get_text()
            text += "\n\n"
        
        doc.close()
        
        logger.info(f"Successfully extracted text using PyMuPDF: {len(text)} characters")
        return {
            "text": text,
            "metadata": metadata,
            "extractor": "pymupdf"
        }
    except ImportError:
        logger.warning("PyMuPDF (fitz) not available. Trying PDFMiner.Six...")
    except Exception as e:
        logger.error(f"Error extracting text with PyMuPDF: {str(e)}")
        logger.warning("Trying PDFMiner.Six...")
    
    # Try to extract text using PDFMiner.Six
    try:
        from pdfminer.high_level import extract_text
        from pdfminer.pdfparser import PDFParser
        from pdfminer.pdfdocument import PDFDocument
        
        logger.info("Using PDFMiner.Six for text extraction")
        
        # Extract text
        text = extract_text(file_path)
        
        # Extract metadata
        with open(file_path, 'rb') as f:
            parser = PDFParser(f)
            doc = PDFDocument(parser)
            metadata = {
                "title": "",
                "author": "",
                "subject": "",
                "keywords": "",
                "creator": "",
                "producer": "",
                "page_count": 0,
                "file_size": os.path.getsize(file_path)
            }
            
            if doc.info:
                for key, value in doc.info[0].items():
                    if key in metadata and value:
                        try:
                            if isinstance(value, bytes):
                                metadata[key] = value.decode('utf-8', errors='ignore')
                            else:
                                metadata[key] = str(value)
                        except:
                            pass
        
        logger.info(f"Successfully extracted text using PDFMiner.Six: {len(text)} characters")
        return {
            "text": text,
            "metadata": metadata,
            "extractor": "pdfminer"
        }
    except ImportError:
        logger.error("PDFMiner.Six not available. Please install with: pip install pdfminer.six")
        return None
    except Exception as e:
        logger.error(f"Error extracting text with PDFMiner.Six: {str(e)}")
        return None

def save_results(results, output_dir):
    """
    Save the processing results to files.
    
    Args:
        results: Processing results
        output_dir: Directory to save results
    """
    if not results:
        logger.error("No results to save")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save text to file
    text_file_path = os.path.join(output_dir, "extracted_text.txt")
    with open(text_file_path, "w", encoding="utf-8") as f:
        f.write(results["text"])
    
    # Save metadata to file
    metadata_file_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_file_path, "w", encoding="utf-8") as f:
        json.dump(results["metadata"], f, indent=2)
    
    logger.info(f"Results saved to {output_dir}")
    logger.info(f"Text file: {text_file_path}")
    logger.info(f"Metadata file: {metadata_file_path}")

def main():
    """
    Main function.
    """
    # Check if a file path was provided
    if len(sys.argv) < 2:
        logger.error("Please provide a file path to process")
        logger.info("Usage: python minimal_document_processor.py <file_path> [output_dir]")
        sys.exit(1)
    
    # Get file path
    file_path = sys.argv[1]
    
    # Get output directory
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "document_processing_results"
    
    # Process the document
    start_time = datetime.now()
    results = process_document(file_path)
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    if results:
        logger.info(f"Document processing completed in {processing_time:.2f} seconds")
        
        # Print some information about the document
        logger.info(f"Title: {results['metadata'].get('title', 'Unknown')}")
        logger.info(f"Author: {results['metadata'].get('author', 'Unknown')}")
        logger.info(f"Pages: {results['metadata'].get('page_count', 0)}")
        logger.info(f"Text length: {len(results['text'])} characters")
        logger.info(f"Extractor: {results['extractor']}")
        
        # Save the results
        save_results(results, output_dir)
    else:
        logger.error("Document processing failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
