"""
Tests for the GraphRAG service.

This module contains tests for the GraphRAG service.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, patch
import numpy as np

from asf.medical.graph.graph_rag import GraphRAG
from asf.medical.graph.graph_service import GraphService
from asf.medical.ml.models.biomedlm import BioMedLMService


@pytest.fixture
def mock_graph_service():
    """Create a mock graph service."""
    mock = MagicMock(spec=GraphService)
    mock.vector_search.return_value = [
        {
            "pmid": "12345",
            "title": "Test Article 1",
            "abstract": "This is a test abstract.",
            "authors": ["Author 1", "Author 2"],
            "publication_date": "2021-01-01",
            "journal": "Test Journal",
            "similarity": 0.9
        },
        {
            "pmid": "67890",
            "title": "Test Article 2",
            "abstract": "This is another test abstract.",
            "authors": ["Author 3", "Author 4"],
            "publication_date": "2021-02-01",
            "journal": "Test Journal 2",
            "similarity": 0.8
        }
    ]
    return mock


@pytest.fixture
def mock_biomedlm_service():
    """Create a mock BioMedLM service."""
    mock = MagicMock(spec=BioMedLMService)
    mock.encode.return_value = np.array([0.1, 0.2, 0.3])
    mock.calculate_similarity.return_value = 0.85
    return mock


@pytest.fixture
def graph_rag(mock_graph_service, mock_biomedlm_service):
    """Create a GraphRAG instance with mock dependencies."""
    return GraphRAG(
        graph_service=mock_graph_service,
        biomedlm_service=mock_biomedlm_service
    )


@pytest.mark.asyncio
async def test_search(graph_rag, mock_graph_service, mock_biomedlm_service):
    """Test the search method."""
    # Set up mock for retrieve_articles_by_concept
    graph_rag.retrieve_articles_by_concept = MagicMock(return_value=[
        {
            "pmid": "12345",
            "title": "Test Article 1",
            "abstract": "This is a test abstract.",
            "authors": ["Author 1", "Author 2"],
            "publication_date": "2021-01-01",
            "journal": "Test Journal"
        }
    ])
    
    # Set up mock for retrieve_related_concepts
    graph_rag.retrieve_related_concepts = MagicMock(return_value=[
        {
            "cui": "C0123456",
            "name": "Test Concept",
            "semantic_types": ["Disease"]
        }
    ])
    
    # Call the search method
    results = await graph_rag.search("test query", max_results=5)
    
    # Verify the results
    assert results is not None
    assert "query" in results
    assert results["query"] == "test query"
    assert "results" in results
    assert len(results["results"]) > 0
    assert "result_count" in results
    assert results["result_count"] > 0
    assert "search_time" in results
    assert "source" in results
    assert results["source"] == "graphrag"
    assert "search_methods" in results
    assert results["search_methods"]["vector_search"] is True
    assert results["search_methods"]["graph_search"] is True
    
    # Verify that the mock methods were called
    mock_graph_service.connect.assert_called_once()
    mock_biomedlm_service.encode.assert_called_once_with("test query")
    mock_graph_service.vector_search.assert_called_once()
    graph_rag.retrieve_articles_by_concept.assert_called()
    graph_rag.retrieve_related_concepts.assert_called()


@pytest.mark.asyncio
async def test_search_vector_only(graph_rag, mock_graph_service, mock_biomedlm_service):
    """Test the search method with vector search only."""
    # Call the search method with graph_search=False
    results = await graph_rag.search("test query", max_results=5, use_graph_search=False)
    
    # Verify the results
    assert results is not None
    assert "search_methods" in results
    assert results["search_methods"]["vector_search"] is True
    assert results["search_methods"]["graph_search"] is False
    
    # Verify that the mock methods were called
    mock_graph_service.connect.assert_called_once()
    mock_biomedlm_service.encode.assert_called_once_with("test query")
    mock_graph_service.vector_search.assert_called_once()


@pytest.mark.asyncio
async def test_search_graph_only(graph_rag, mock_graph_service, mock_biomedlm_service):
    """Test the search method with graph search only."""
    # Set up mock for retrieve_articles_by_concept
    graph_rag.retrieve_articles_by_concept = MagicMock(return_value=[
        {
            "pmid": "12345",
            "title": "Test Article 1",
            "abstract": "This is a test abstract.",
            "authors": ["Author 1", "Author 2"],
            "publication_date": "2021-01-01",
            "journal": "Test Journal"
        }
    ])
    
    # Set up mock for retrieve_related_concepts
    graph_rag.retrieve_related_concepts = MagicMock(return_value=[
        {
            "cui": "C0123456",
            "name": "Test Concept",
            "semantic_types": ["Disease"]
        }
    ])
    
    # Call the search method with vector_search=False
    results = await graph_rag.search("test query", max_results=5, use_vector_search=False)
    
    # Verify the results
    assert results is not None
    assert "search_methods" in results
    assert results["search_methods"]["vector_search"] is False
    assert results["search_methods"]["graph_search"] is True
    
    # Verify that the mock methods were called
    mock_graph_service.connect.assert_called_once()
    mock_graph_service.vector_search.assert_not_called()
    graph_rag.retrieve_articles_by_concept.assert_called()
    graph_rag.retrieve_related_concepts.assert_called()


@pytest.mark.asyncio
async def test_generate_summary(graph_rag, mock_graph_service, mock_biomedlm_service):
    """Test the generate_summary method."""
    # Set up mock for retrieve_articles_by_concept
    graph_rag.retrieve_articles_by_concept = MagicMock(return_value=[
        {
            "pmid": "12345",
            "title": "Test Article 1",
            "abstract": "This is a test abstract.",
            "authors": ["Author 1", "Author 2"],
            "publication_date": "2021-01-01",
            "journal": "Test Journal"
        }
    ])
    
    # Call the generate_summary method
    summary = await graph_rag.generate_summary("test query", max_articles=3)
    
    # Verify the summary
    assert summary is not None
    assert "query" in summary
    assert summary["query"] == "test query"
    assert "articles" in summary
    assert len(summary["articles"]) > 0
    assert "article_count" in summary
    assert summary["article_count"] > 0
    assert "generated_at" in summary
    assert "source" in summary
    assert summary["source"] == "graphrag"
    
    # Verify that the mock methods were called
    mock_graph_service.connect.assert_called_once()
    graph_rag.retrieve_articles_by_concept.assert_called()
