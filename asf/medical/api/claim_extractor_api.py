"""
API endpoints for scientific claim extraction functionality.

This module provides REST API endpoints for:
- Extracting claims from provided text
- Processing documents to identify and visualize claims
- Analyzing claim relationships and patterns
"""

from flask import Blueprint, request, jsonify
from ..ml.models.claim_extractor import ClaimExtractor, ExtractedClaim
import logging

# Initialize logger
logger = logging.getLogger(__name__)

# Create Flask Blueprint
claim_api = Blueprint('claim_api', __name__)

# Initialize claim extractor
claim_extractor = None

def get_claim_extractor():
    """Get or initialize the claim extractor."""
    global claim_extractor
    if claim_extractor is None:
        try:
            # Initialize with default settings
            claim_extractor = ClaimExtractor(
                use_registry=True,  # Use model registry if available
                sci_spacy=True      # Use scientific spaCy model if available
            )
            logger.info("Claim extractor initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing claim extractor: {e}")
            raise
    return claim_extractor

@claim_api.route('/extract-claims', methods=['POST'])
def extract_claims():
    """
    Extract scientific claims from provided text.
    
    Expects JSON with:
    - text: The text to extract claims from
    - doc_id: Optional document ID
    
    Returns:
    - A list of extracted claims with metadata
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        doc_id = data.get('doc_id')
        
        # Get the claim extractor instance
        extractor = get_claim_extractor()
        
        # Extract claims
        extracted_claims = extractor.extract_claims(text, doc_id)
        
        # Convert claims to dictionaries
        claims_data = [claim.to_dict() for claim in extracted_claims]
        
        # Return the results
        return jsonify({
            'claims': claims_data,
            'count': len(claims_data)
        }), 200
        
    except Exception as e:
        logger.error(f"Error extracting claims: {e}")
        return jsonify({'error': str(e)}), 500

@claim_api.route('/extract-claims-from-abstract', methods=['POST'])
def extract_claims_from_abstract():
    """
    Extract claims specifically from an abstract.
    
    Expects JSON with:
    - abstract: The abstract text
    - doc_id: Optional document ID
    
    Returns:
    - A list of extracted claims with metadata
    """
    try:
        data = request.get_json()
        
        if not data or 'abstract' not in data:
            return jsonify({'error': 'No abstract provided'}), 400
        
        abstract = data['abstract']
        doc_id = data.get('doc_id')
        
        # Get the claim extractor instance
        extractor = get_claim_extractor()
        
        # Extract claims from abstract
        extracted_claims = extractor.extract_claims_from_abstract(abstract, doc_id)
        
        # Convert claims to dictionaries
        claims_data = [claim.to_dict() for claim in extracted_claims]
        
        # Return the results
        return jsonify({
            'claims': claims_data,
            'count': len(claims_data)
        }), 200
        
    except Exception as e:
        logger.error(f"Error extracting claims from abstract: {e}")
        return jsonify({'error': str(e)}), 500

@claim_api.route('/extract-claims-from-full-text', methods=['POST'])
def extract_claims_from_full_text():
    """
    Extract claims from a full text document, optionally with sections.
    
    Expects JSON with:
    - full_text: The full text content
    - doc_id: Optional document ID
    - sections: Optional dictionary of section names to text
    
    Returns:
    - A dictionary mapping section names to lists of extracted claims
    """
    try:
        data = request.get_json()
        
        if not data or 'full_text' not in data:
            return jsonify({'error': 'No full text provided'}), 400
        
        full_text = data['full_text']
        doc_id = data.get('doc_id')
        sections = data.get('sections')
        
        # Get the claim extractor instance
        extractor = get_claim_extractor()
        
        # Extract claims from full text
        result = extractor.extract_claims_from_full_text(full_text, doc_id, sections)
        
        # Convert claims to dictionaries for each section
        processed_results = {}
        for section_name, claims in result.items():
            processed_results[section_name] = [claim.to_dict() for claim in claims]
        
        # Return the results
        return jsonify({
            'sections': processed_results,
            'total_claims': sum(len(claims) for claims in processed_results.values())
        }), 200
        
    except Exception as e:
        logger.error(f"Error extracting claims from full text: {e}")
        return jsonify({'error': str(e)}), 500

@claim_api.route('/batch-extract-claims', methods=['POST'])
def batch_extract_claims():
    """
    Extract claims from multiple texts in batch mode.
    
    Expects JSON with:
    - texts: List of text strings
    - doc_ids: Optional list of document IDs
    
    Returns:
    - A dictionary mapping document IDs to lists of extracted claims
    """
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({'error': 'No texts provided'}), 400
        
        texts = data['texts']
        doc_ids = data.get('doc_ids')
        
        # Get the claim extractor instance
        extractor = get_claim_extractor()
        
        # Extract claims from batch of texts
        results = extractor.batch_extract_claims(texts, doc_ids)
        
        # Convert claims to dictionaries for each document
        processed_results = {}
        for doc_id, claims in results.items():
            processed_results[doc_id] = [claim.to_dict() for claim in claims]
        
        # Return the results
        return jsonify({
            'results': processed_results,
            'total_claims': sum(len(claims) for claims in processed_results.values())
        }), 200
        
    except Exception as e:
        logger.error(f"Error batch extracting claims: {e}")
        return jsonify({'error': str(e)}), 500