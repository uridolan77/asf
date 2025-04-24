"""
Simple tests for export utilities.

This module contains simplified tests for the export utilities in the Medical Research Synthesizer.
"""

import json
import pytest
from fastapi.responses import JSONResponse

from asf.medical.api.export_utils import export_to_json
from asf.medical.core.exceptions import ValidationError


def test_export_to_json():
    """Test exporting data to JSON."""
    # Sample data
    sample_data = [
        {
            "pmid": "12345678",
            "title": "Test Article 1",
            "journal": "Test Journal",
            "authors": ["Smith, J."]
        },
        {
            "pmid": "87654321",
            "title": "Test Article 2",
            "journal": "Another Journal",
            "authors": ["Johnson, A."]
        }
    ]
    
    # Test with valid data
    response = export_to_json(sample_data, query_text="test query")
    
    assert isinstance(response, JSONResponse)
    content = json.loads(response.body)
    assert content["count"] == 2
    assert len(content["results"]) == 2
    assert "query" in content
    assert content["query"] == "test query"
    
    # Test with invalid data
    with pytest.raises(ValidationError):
        export_to_json("not a list")
