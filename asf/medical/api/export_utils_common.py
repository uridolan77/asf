"""
Common Export Utilities for Medical Research Synthesizer.

This module provides common utility functions for data cleaning, validation,
and formatting that are used across different export formats (JSON, CSV, Excel, PDF).
These utilities help ensure consistent behavior and reduce code duplication.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

from asf.medical.core.exceptions import ValidationError, ExportError

logger = logging.getLogger(__name__)


def validate_export_data(data: Any) -> List[Dict[str, Any]]:
    """
    Validate export data to ensure it's a list of dictionaries.
    
    Args:
        data: Data to validate
    
    Returns:
        Validated list of dictionaries
    
    Raises:
        ValidationError: If data is not a list or contains invalid items
    """
    if not isinstance(data, list):
        raise ValidationError("Data must be a list of dictionaries")
    
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValidationError(f"Item at index {i} is not a dictionary")
    
    return data


def clean_export_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Clean export data by handling non-serializable values.
    
    Args:
        data: List of dictionaries to clean
    
    Returns:
        Cleaned list of dictionaries
    """
    cleaned_data = []
    
    for item in data:
        cleaned_item = {}
        for key, value in item.items():
            if value is not None:
                if isinstance(value, (dict, list, str, int, float, bool, type(None))):
                    cleaned_item[key] = value
                else:
                    cleaned_item[key] = str(value)
        cleaned_data.append(cleaned_item)
    
    return cleaned_data


def filter_export_data(
    data: List[Dict[str, Any]],
    include_fields: Optional[List[str]] = None,
    exclude_fields: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Filter export data to include or exclude specific fields.
    
    Args:
        data: List of dictionaries to filter
        include_fields: Fields to include (if None, include all)
        exclude_fields: Fields to exclude
    
    Returns:
        Filtered list of dictionaries
    """
    exclude_fields = exclude_fields or []
    filtered_data = []
    
    for item in data:
        filtered_item = {}
        
        if include_fields:
            # Include only specified fields
            for field in include_fields:
                if field in item and field not in exclude_fields:
                    filtered_item[field] = item[field]
        else:
            # Include all fields except excluded ones
            for key, value in item.items():
                if key not in exclude_fields:
                    filtered_item[key] = value
        
        filtered_data.append(filtered_item)
    
    return filtered_data


def get_export_metadata(query_text: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate metadata for export files.
    
    Args:
        query_text: Optional query text that produced the results
    
    Returns:
        Dictionary containing export metadata
    """
    metadata = {
        "exported_at": datetime.now().isoformat(),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    if query_text:
        metadata["query"] = query_text
    
    return metadata


def get_common_fields() -> Dict[str, List[str]]:
    """
    Get common field lists for different export types.
    
    Returns:
        Dictionary containing field lists for different export types
    """
    return {
        "basic": [
            'pmid', 'title', 'journal', 'publication_date', 'authors'
        ],
        "standard": [
            'pmid', 'title', 'journal', 'publication_date', 'iso_date',
            'authors', 'abstract', 'doi'
        ],
        "detailed": [
            'pmid', 'title', 'journal', 'publication_date', 'iso_date',
            'authors', 'abstract', 'impact_factor', 'journal_quartile',
            'citation_count', 'authority_score', 'publication_types',
            'doi', 'mesh_terms'
        ],
        "analysis": [
            'pmid', 'title', 'journal', 'publication_date', 'authors',
            'abstract', 'contradiction_score', 'contradiction_type',
            'evidence_level'
        ]
    }


def handle_export_error(e: Exception, operation: str) -> None:
    """
    Handle export errors consistently.
    
    Args:
        e: The exception that occurred
        operation: The export operation that failed
    
    Raises:
        ExportError: Always raised with consistent formatting
    """
    error_message = f"Failed to export data to {operation}: {str(e)}"
    logger.error(error_message)
    raise ExportError(error_message) from e
