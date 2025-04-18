"""
Unit tests for the SearchService.
This module provides unit tests for the SearchService.
"""
import pytest
import logging
import uuid
from unittest.mock import AsyncMock, MagicMock
from ...services.search_service import SearchService
logger = logging.getLogger(__name__)
@pytest.fixture
def mock_ncbi_client():
    """Mock NCBI client for testing.
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
    """
    mock = AsyncMock()
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
    """Mock GraphRAG for testing.
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
    """
    mock = AsyncMock()
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
    """Mock result repository for testing.
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
    """
    mock = AsyncMock()
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
    mock.save_async.return_value = MagicMock(id=str(uuid.uuid4()))
    return mock
@pytest.fixture
def search_service(mock_ncbi_client, mock_graph_rag, mock_result_repository):
    """SearchService instance for testing.
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
    """
    service = SearchService(
        ncbi_client=mock_ncbi_client,
        graph_rag=mock_graph_rag,
        result_repository=mock_result_repository
    )
    service.is_graph_rag_available = lambda: True
    return service
@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.service
class TestSearchService:
    """Test cases for SearchService."""
    async def test_search_with_pubmed(self, search_service, mock_ncbi_client):
        """
        Test search with PubMed.
        Args:
            # TODO: Add parameter descriptions
        Returns:
            # TODO: Add return description
        """