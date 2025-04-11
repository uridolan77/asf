"""
Unit tests for the SearchService.

This module provides unit tests for the SearchService.
"""

import pytest
import logging
import uuid
from typing import Dict, Any, List, Optional
from unittest.mock import MagicMock, AsyncMock, patch

from asf.medical.services.search_service import SearchService, SearchMethod
from asf.medical.core.exceptions import ExternalServiceError, DatabaseError

# Configure logging
logger = logging.getLogger(__name__)

@pytest.fixture
def mock_ncbi_client():
    """Mock NCBI client for testing."""
    mock = AsyncMock()
    
    # Mock search_pubmed method
    mock.search_pubmed.return_value = {
        "query": "test query",
        "results": [
            {
                "pmid": "12345",
                "title": "Test Article 1",
                "abstract": "This is a test abstract for article 1.",
                "authors": ["Author A", "Author B"],
                "journal": "Test Journal",
                "publication_date": "2023-01-01",
                "doi": "10.1234/test.1"
            },
            {
                "pmid": "67890",
                "title": "Test Article 2",
                "abstract": "This is a test abstract for article 2.",
                "authors": ["Author C", "Author D"],
                "journal": "Test Journal",
                "publication_date": "2023-02-01",
                "doi": "10.1234/test.2"
            }
        ],
        "total": 2
    }
    
    # Mock search_clinical_trials method
    mock.search_clinical_trials.return_value = {
        "query": "test query",
        "results": [
            {
                "nct_id": "NCT12345",
                "title": "Test Clinical Trial 1",
                "summary": "This is a test summary for clinical trial 1.",
                "status": "Completed",
                "start_date": "2022-01-01",
                "completion_date": "2023-01-01",
                "phase": "Phase 2"
            },
            {
                "nct_id": "NCT67890",
                "title": "Test Clinical Trial 2",
                "summary": "This is a test summary for clinical trial 2.",
                "status": "Recruiting",
                "start_date": "2023-01-01",
                "completion_date": "2024-01-01",
                "phase": "Phase 3"
            }
        ],
        "total": 2
    }
    
    return mock

@pytest.fixture
def mock_graph_rag():
    """Mock GraphRAG for testing."""
    mock = AsyncMock()
    
    # Mock search method
    mock.search.return_value = {
        "query": "test query",
        "results": [
            {
                "pmid": "12345",
                "title": "Test Article 1",
                "abstract": "This is a test abstract for article 1.",
                "authors": ["Author A", "Author B"],
                "journal": "Test Journal",
                "publication_date": "2023-01-01",
                "doi": "10.1234/test.1",
                "relevance_score": 0.95
            },
            {
                "pmid": "67890",
                "title": "Test Article 2",
                "abstract": "This is a test abstract for article 2.",
                "authors": ["Author C", "Author D"],
                "journal": "Test Journal",
                "publication_date": "2023-02-01",
                "doi": "10.1234/test.2",
                "relevance_score": 0.85
            }
        ],
        "total": 2,
        "search_method": "graph_rag"
    }
    
    return mock

@pytest.fixture
def mock_result_repository():
    """Mock result repository for testing."""
    mock = AsyncMock()
    
    # Mock get_by_result_id_async method
    mock.get_by_result_id_async.return_value = MagicMock(
        query=MagicMock(query_text="test query"),
        result_data={
            "articles": [
                {
                    "pmid": "12345",
                    "title": "Test Article 1",
                    "abstract": "This is a test abstract for article 1.",
                    "authors": ["Author A", "Author B"],
                    "journal": "Test Journal",
                    "publication_date": "2023-01-01",
                    "doi": "10.1234/test.1"
                }
            ]
        },
        created_at=MagicMock(isoformat=lambda: "2023-01-01T12:00:00"),
        user_id=1
    )
    
    # Mock save_async method
    mock.save_async.return_value = MagicMock(id=str(uuid.uuid4()))
    
    return mock

@pytest.fixture
def search_service(mock_ncbi_client, mock_graph_rag, mock_result_repository):
    """SearchService instance for testing."""
    service = SearchService(
        ncbi_client=mock_ncbi_client,
        graph_rag=mock_graph_rag,
        result_repository=mock_result_repository
    )
    
    # Mock is_graph_rag_available method
    service.is_graph_rag_available = lambda: True
    
    return service

@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.service
class TestSearchService:
    """Test cases for SearchService."""
    
    async def test_search_with_pubmed(self, search_service, mock_ncbi_client):
        """Test search with PubMed."""
        # Call the search method with PubMed search method
        result = await search_service.search(
            query="test query",
            max_results=10,
            search_method=SearchMethod.PUBMED
        )
        
        # Check that the NCBI client's search_pubmed method was called
        mock_ncbi_client.search_pubmed.assert_called_once_with("test query", max_results=10)
        
        # Check the result
        assert result["query"] == "test query"
        assert len(result["results"]) == 2
        assert result["results"][0]["pmid"] == "12345"
        assert result["results"][1]["pmid"] == "67890"
    
    async def test_search_with_clinical_trials(self, search_service, mock_ncbi_client):
        """Test search with ClinicalTrials.gov."""
        # Call the search method with ClinicalTrials.gov search method
        result = await search_service.search(
            query="test query",
            max_results=10,
            search_method=SearchMethod.CLINICAL_TRIALS
        )
        
        # Check that the NCBI client's search_clinical_trials method was called
        mock_ncbi_client.search_clinical_trials.assert_called_once_with("test query", max_results=10)
        
        # Check the result
        assert result["query"] == "test query"
        assert len(result["results"]) == 2
        assert result["results"][0]["nct_id"] == "NCT12345"
        assert result["results"][1]["nct_id"] == "NCT67890"
    
    async def test_search_with_graph_rag(self, search_service, mock_graph_rag):
        """Test search with GraphRAG."""
        # Call the search method with GraphRAG search method
        result = await search_service.search(
            query="test query",
            max_results=10,
            search_method=SearchMethod.GRAPH_RAG
        )
        
        # Check that the GraphRAG's search method was called
        mock_graph_rag.search.assert_called_once_with(
            "test query",
            max_results=10,
            use_vector_search=True,
            use_graph_search=True
        )
        
        # Check the result
        assert result["query"] == "test query"
        assert len(result["results"]) == 2
        assert result["results"][0]["pmid"] == "12345"
        assert result["results"][1]["pmid"] == "67890"
        assert result["results"][0]["relevance_score"] == 0.95
    
    async def test_search_with_graph_rag_not_available(self, search_service, mock_ncbi_client):
        """Test search with GraphRAG not available."""
        # Mock is_graph_rag_available to return False
        search_service.is_graph_rag_available = lambda: False
        
        # Call the search method with GraphRAG search method
        result = await search_service.search(
            query="test query",
            max_results=10,
            search_method=SearchMethod.GRAPH_RAG
        )
        
        # Check that the NCBI client's search_pubmed method was called as fallback
        mock_ncbi_client.search_pubmed.assert_called_once_with("test query", max_results=10)
        
        # Check the result
        assert result["query"] == "test query"
        assert len(result["results"]) == 2
        assert "fallback_reason" in result
        assert result["fallback_reason"] == "GraphRAG not available"
    
    async def test_search_with_pagination(self, search_service):
        """Test search with pagination."""
        # Call the search method with pagination
        result = await search_service.search(
            query="test query",
            max_results=10,
            page=2,
            page_size=1
        )
        
        # Check the pagination
        assert result["pagination"]["page"] == 2
        assert result["pagination"]["page_size"] == 1
        assert result["pagination"]["total_pages"] == 2
        assert result["pagination"]["total_results"] == 2
        
        # Check that only one result is returned (page 2, page_size 1)
        assert len(result["results"]) == 1
        assert result["results"][0]["pmid"] == "67890"
    
    async def test_search_with_external_service_error(self, search_service, mock_ncbi_client):
        """Test search with external service error."""
        # Mock the NCBI client to raise an exception
        mock_ncbi_client.search_pubmed.side_effect = Exception("External service error")
        
        # Call the search method and expect an ExternalServiceError
        with pytest.raises(ExternalServiceError) as excinfo:
            await search_service.search(
                query="test query",
                max_results=10
            )
        
        # Check the exception
        assert "Failed to search PubMed" in str(excinfo.value)
    
    async def test_get_result_by_id(self, search_service, mock_result_repository):
        """Test get_result_by_id."""
        # Call the get_result_by_id method
        result = await search_service.get_result_by_id("test-result-id")
        
        # Check that the result repository's get_by_result_id_async method was called
        mock_result_repository.get_by_result_id_async.assert_called_once_with(db=None, result_id="test-result-id")
        
        # Check the result
        assert result["query"] == "test query"
        assert len(result["results"]) == 1
        assert result["results"][0]["pmid"] == "12345"
        assert result["timestamp"] == "2023-01-01T12:00:00"
        assert result["user_id"] == 1
    
    async def test_get_result_by_id_not_found(self, search_service, mock_result_repository):
        """Test get_result_by_id with result not found."""
        # Mock the result repository to return None
        mock_result_repository.get_by_result_id_async.return_value = None
        
        # Call the get_result_by_id method
        result = await search_service.get_result_by_id("test-result-id")
        
        # Check the result
        assert result is None
    
    async def test_get_result_by_id_with_database_error(self, search_service, mock_result_repository):
        """Test get_result_by_id with database error."""
        # Mock the result repository to raise an exception
        mock_result_repository.get_by_result_id_async.side_effect = Exception("Database error")
        
        # Call the get_result_by_id method and expect a DatabaseError
        with pytest.raises(DatabaseError) as excinfo:
            await search_service.get_result_by_id("test-result-id")
        
        # Check the exception
        assert "Failed to retrieve search result" in str(excinfo.value)
    
    async def test_save_result(self, search_service, mock_result_repository):
        """Test save_result."""
        # Call the save_result method
        result_id = await search_service.save_result(
            query="test query",
            results=[{"pmid": "12345", "title": "Test Article"}],
            user_id=1
        )
        
        # Check that the result repository's save_async method was called
        mock_result_repository.save_async.assert_called_once()
        
        # Check that a result ID was returned
        assert result_id is not None
        assert isinstance(result_id, str)
    
    async def test_save_result_with_database_error(self, search_service, mock_result_repository):
        """Test save_result with database error."""
        # Mock the result repository to raise an exception
        mock_result_repository.save_async.side_effect = Exception("Database error")
        
        # Call the save_result method and expect a DatabaseError
        with pytest.raises(DatabaseError) as excinfo:
            await search_service.save_result(
                query="test query",
                results=[{"pmid": "12345", "title": "Test Article"}],
                user_id=1
            )
        
        # Check the exception
        assert "Failed to save search result" in str(excinfo.value)
