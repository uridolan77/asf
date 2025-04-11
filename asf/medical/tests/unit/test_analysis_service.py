"""
Unit tests for the AnalysisService.

This module provides unit tests for the AnalysisService.
"""

import pytest
import logging
import uuid
from typing import Dict, Any, List, Optional
from unittest.mock import MagicMock, AsyncMock, patch

from asf.medical.services.analysis_service import AnalysisService
from asf.medical.core.exceptions import ExternalServiceError, DatabaseError

# Configure logging
logger = logging.getLogger(__name__)

@pytest.fixture
def mock_search_service():
    """Mock SearchService for testing."""
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
    
    # Mock save_result method
    mock.save_result.return_value = str(uuid.uuid4())
    
    return mock

@pytest.fixture
def mock_contradiction_service():
    """Mock ContradictionService for testing."""
    mock = AsyncMock()
    
    # Mock detect_contradictions_in_articles method
    mock.detect_contradictions_in_articles.return_value = [
        {
            "article1": {
                "pmid": "12345",
                "title": "Test Article 1",
                "abstract": "This is a test abstract for article 1."
            },
            "article2": {
                "pmid": "67890",
                "title": "Test Article 2",
                "abstract": "This is a test abstract for article 2."
            },
            "contradiction_score": 0.85,
            "contradiction_type": "negation",
            "explanation": "The articles contradict each other on the effectiveness of the treatment."
        }
    ]
    
    # Mock detect_contradiction method
    mock.detect_contradiction.return_value = {
        "is_contradiction": True,
        "score": 0.85,
        "type": "negation",
        "explanation": "The claims contradict each other on the effectiveness of the treatment."
    }
    
    return mock

@pytest.fixture
def mock_analysis_repository():
    """Mock AnalysisRepository for testing."""
    mock = AsyncMock()
    
    # Mock get_by_analysis_id_async method
    mock.get_by_analysis_id_async.return_value = MagicMock(
        query=MagicMock(query_text="test query"),
        analysis_data={
            "contradictions": [
                {
                    "article1": {
                        "pmid": "12345",
                        "title": "Test Article 1"
                    },
                    "article2": {
                        "pmid": "67890",
                        "title": "Test Article 2"
                    },
                    "contradiction_score": 0.85,
                    "contradiction_type": "negation"
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
def analysis_service(mock_search_service, mock_contradiction_service, mock_analysis_repository):
    """AnalysisService instance for testing."""
    return AnalysisService(
        search_service=mock_search_service,
        contradiction_service=mock_contradiction_service,
        analysis_repository=mock_analysis_repository
    )

@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.service
class TestAnalysisService:
    """Test cases for AnalysisService."""
    
    async def test_analyze_contradictions(self, analysis_service, mock_search_service, mock_contradiction_service):
        """Test analyze_contradictions."""
        # Call the analyze_contradictions method
        result = await analysis_service.analyze_contradictions(
            query="test query",
            max_results=10,
            threshold=0.7,
            use_biomedlm=True
        )
        
        # Check that the search service's search method was called
        mock_search_service.search.assert_called_once_with("test query", 10, None)
        
        # Check that the contradiction service's detect_contradictions_in_articles method was called
        mock_contradiction_service.detect_contradictions_in_articles.assert_called_once()
        
        # Check the result
        assert result["query"] == "test query"
        assert len(result["contradictions"]) == 1
        assert result["contradictions"][0]["contradiction_score"] == 0.85
        assert result["contradictions"][0]["contradiction_type"] == "negation"
        assert "analysis_id" in result
        assert result["detection_method"] == "biomedlm"
    
    async def test_analyze_contradictions_no_results(self, analysis_service, mock_search_service):
        """Test analyze_contradictions with no search results."""
        # Mock the search service to return no results
        mock_search_service.search.return_value = {
            "query": "test query",
            "results": [],
            "total": 0
        }
        
        # Call the analyze_contradictions method
        result = await analysis_service.analyze_contradictions(
            query="test query",
            max_results=10,
            threshold=0.7
        )
        
        # Check the result
        assert result["query"] == "test query"
        assert result["total_articles"] == 0
        assert len(result["contradictions"]) == 0
        assert "analysis_id" in result
        assert result["detection_method"] == "none"
    
    async def test_analyze_contradictions_search_error(self, analysis_service, mock_search_service):
        """Test analyze_contradictions with search error."""
        # Mock the search service to raise an exception
        mock_search_service.search.side_effect = Exception("Search error")
        
        # Call the analyze_contradictions method and expect an ExternalServiceError
        with pytest.raises(ExternalServiceError) as excinfo:
            await analysis_service.analyze_contradictions(
                query="test query",
                max_results=10,
                threshold=0.7
            )
        
        # Check the exception
        assert "Failed to search for articles" in str(excinfo.value)
    
    async def test_analyze_contradictions_with_multiple_methods(self, analysis_service):
        """Test analyze_contradictions with multiple detection methods."""
        # Call the analyze_contradictions method with multiple detection methods
        result = await analysis_service.analyze_contradictions(
            query="test query",
            max_results=10,
            threshold=0.7,
            use_biomedlm=True,
            use_tsmixer=True,
            use_lorentz=True
        )
        
        # Check the detection method
        assert "biomedlm" in result["detection_method"]
        assert "tsmixer" in result["detection_method"]
        assert "lorentz" in result["detection_method"]
    
    async def test_detect_contradiction(self, analysis_service, mock_contradiction_service):
        """Test detect_contradiction."""
        # Call the detect_contradiction method
        result = await analysis_service.detect_contradiction(
            claim1="Treatment X is effective for condition Y.",
            claim2="Treatment X is not effective for condition Y.",
            threshold=0.7
        )
        
        # Check that the contradiction service's detect_contradiction method was called
        mock_contradiction_service.detect_contradiction.assert_called_once_with(
            "Treatment X is effective for condition Y.",
            "Treatment X is not effective for condition Y.",
            threshold=0.7,
            use_biomedlm=True,
            use_tsmixer=False,
            use_lorentz=False,
            use_temporal=False,
            skip_cache=False
        )
        
        # Check the result
        assert result["is_contradiction"] is True
        assert result["score"] == 0.85
        assert result["type"] == "negation"
        assert "explanation" in result
    
    async def test_detect_contradiction_with_multiple_methods(self, analysis_service, mock_contradiction_service):
        """Test detect_contradiction with multiple detection methods."""
        # Call the detect_contradiction method with multiple detection methods
        result = await analysis_service.detect_contradiction(
            claim1="Treatment X is effective for condition Y.",
            claim2="Treatment X is not effective for condition Y.",
            threshold=0.7,
            use_biomedlm=True,
            use_tsmixer=True,
            use_lorentz=True,
            use_temporal=True
        )
        
        # Check that the contradiction service's detect_contradiction method was called with the right parameters
        mock_contradiction_service.detect_contradiction.assert_called_once_with(
            "Treatment X is effective for condition Y.",
            "Treatment X is not effective for condition Y.",
            threshold=0.7,
            use_biomedlm=True,
            use_tsmixer=True,
            use_lorentz=True,
            use_temporal=True,
            skip_cache=False
        )
    
    async def test_get_analysis_by_id(self, analysis_service, mock_analysis_repository):
        """Test get_analysis_by_id."""
        # Call the get_analysis_by_id method
        result = await analysis_service.get_analysis_by_id("test-analysis-id")
        
        # Check that the analysis repository's get_by_analysis_id_async method was called
        mock_analysis_repository.get_by_analysis_id_async.assert_called_once_with(db=None, analysis_id="test-analysis-id")
        
        # Check the result
        assert result["query"] == "test query"
        assert len(result["contradictions"]) == 1
        assert result["contradictions"][0]["contradiction_score"] == 0.85
        assert result["timestamp"] == "2023-01-01T12:00:00"
        assert result["user_id"] == 1
    
    async def test_get_analysis_by_id_not_found(self, analysis_service, mock_analysis_repository):
        """Test get_analysis_by_id with analysis not found."""
        # Mock the analysis repository to return None
        mock_analysis_repository.get_by_analysis_id_async.return_value = None
        
        # Call the get_analysis_by_id method
        result = await analysis_service.get_analysis_by_id("test-analysis-id")
        
        # Check the result
        assert result is None
    
    async def test_get_analysis_by_id_with_database_error(self, analysis_service, mock_analysis_repository):
        """Test get_analysis_by_id with database error."""
        # Mock the analysis repository to raise an exception
        mock_analysis_repository.get_by_analysis_id_async.side_effect = Exception("Database error")
        
        # Call the get_analysis_by_id method and expect a DatabaseError
        with pytest.raises(DatabaseError) as excinfo:
            await analysis_service.get_analysis_by_id("test-analysis-id")
        
        # Check the exception
        assert "Failed to retrieve analysis" in str(excinfo.value)
    
    async def test_save_analysis(self, analysis_service, mock_analysis_repository):
        """Test save_analysis."""
        # Call the save_analysis method
        analysis_id = await analysis_service.save_analysis(
            query="test query",
            contradictions=[{"contradiction_score": 0.85, "contradiction_type": "negation"}],
            user_id=1
        )
        
        # Check that the analysis repository's save_async method was called
        mock_analysis_repository.save_async.assert_called_once()
        
        # Check that an analysis ID was returned
        assert analysis_id is not None
        assert isinstance(analysis_id, str)
    
    async def test_save_analysis_with_database_error(self, analysis_service, mock_analysis_repository):
        """Test save_analysis with database error."""
        # Mock the analysis repository to raise an exception
        mock_analysis_repository.save_async.side_effect = Exception("Database error")
        
        # Call the save_analysis method and expect a DatabaseError
        with pytest.raises(DatabaseError) as excinfo:
            await analysis_service.save_analysis(
                query="test query",
                contradictions=[{"contradiction_score": 0.85, "contradiction_type": "negation"}],
                user_id=1
            )
        
        # Check the exception
        assert "Failed to save analysis" in str(excinfo.value)
