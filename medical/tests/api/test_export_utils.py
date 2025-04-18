"""
Tests for export utilities.

This module contains tests for the export utilities in the Medical Research Synthesizer.
"""

import json
import pytest
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.testclient import TestClient

from asf.medical.api.export_utils import (
    export_to_json,
    export_to_csv,
    export_to_excel,
    export_to_pdf,
    export_contradiction_analysis_to_pdf
)
from asf.medical.core.exceptions import ValidationError


@pytest.fixture
def sample_data():
    """Sample data for testing export utilities."""
    return [
        {
            "pmid": "12345678",
            "title": "Test Article 1",
            "journal": "Test Journal",
            "publication_date": "2023-01-01",
            "authors": ["Smith, J.", "Doe, J."],
            "abstract": "This is a test abstract for article 1.",
            "doi": "10.1234/test.1",
            "citation_count": 10,
            "impact_factor": 3.5
        },
        {
            "pmid": "87654321",
            "title": "Test Article 2",
            "journal": "Another Journal",
            "publication_date": "2023-02-01",
            "authors": ["Johnson, A.", "Williams, B."],
            "abstract": "This is a test abstract for article 2.",
            "doi": "10.1234/test.2",
            "citation_count": 5,
            "impact_factor": 2.8
        }
    ]


@pytest.fixture
def sample_contradiction_analysis():
    """Sample contradiction analysis for testing export utilities."""
    return {
        "query": "test query",
        "contradictions": [
            {
                "statement1": "Treatment A is effective for condition X.",
                "statement2": "Treatment A shows no benefit for condition X.",
                "contradiction_score": 0.85,
                "explanation": "These statements directly contradict each other regarding the efficacy of Treatment A.",
                "sources": [
                    {"pmid": "12345678", "title": "Study on Treatment A"},
                    {"pmid": "87654321", "title": "Review of Treatment A"}
                ]
            }
        ]
    }


def test_export_to_json(sample_data):
    """Test exporting data to JSON."""
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


def test_export_to_csv(sample_data):
    """Test exporting data to CSV."""
    # Test with valid data
    response = export_to_csv(sample_data, query_text="test query")
    
    assert isinstance(response, StreamingResponse)
    assert response.headers["Content-Disposition"] == "attachment; filename=search_results.csv"
    assert response.media_type == "text/csv"
    
    # Test with invalid data
    with pytest.raises(ValidationError):
        export_to_csv("not a list")


def test_export_to_excel(sample_data):
    """Test exporting data to Excel."""
    # Test with valid data
    response = export_to_excel(sample_data, query_text="test query")
    
    assert isinstance(response, StreamingResponse)
    assert response.headers["Content-Disposition"] == "attachment; filename=search_results.xlsx"
    assert response.media_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    
    # Test with invalid data
    with pytest.raises(ValidationError):
        export_to_excel("not a list")


def test_export_to_pdf(sample_data):
    """Test exporting data to PDF."""
    # Test with valid data
    response = export_to_pdf(sample_data, query_text="test query")
    
    assert isinstance(response, FileResponse)
    assert response.headers["Content-Disposition"] == "attachment; filename=search_results.pdf"
    assert response.media_type == "application/pdf"
    
    # Test with invalid data
    with pytest.raises(ValidationError):
        export_to_pdf("not a list")


def test_export_contradiction_analysis_to_pdf(sample_contradiction_analysis):
    """Test exporting contradiction analysis to PDF."""
    # Test with valid data
    response = export_contradiction_analysis_to_pdf(
        sample_contradiction_analysis, 
        query_text="test query"
    )
    
    assert isinstance(response, FileResponse)
    assert response.headers["Content-Disposition"] == "attachment; filename=contradiction_analysis.pdf"
    assert response.media_type == "application/pdf"
    
    # Test with invalid data
    with pytest.raises(ValidationError):
        export_contradiction_analysis_to_pdf("not a dict", query_text="test query")
