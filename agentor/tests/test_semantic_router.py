import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
from agentor.llm_gateway.agents.router import SemanticRouter


@pytest.mark.asyncio
async def test_semantic_router():
    """Test the SemanticRouter class."""
    # Create a mock embedding response
    mock_embedding_response = MagicMock()
    mock_embedding_response.data = [MagicMock()]
    
    # Create different embeddings for different intents
    weather_embedding = [1.0, 0.0, 0.0]
    news_embedding = [0.0, 1.0, 0.0]
    sports_embedding = [0.0, 0.0, 1.0]
    
    # Create a mock OpenAI client
    mock_client = AsyncMock()
    mock_client.embeddings.create = AsyncMock()
    
    # Patch the AsyncOpenAI class
    with patch('openai.AsyncOpenAI', return_value=mock_client):
        # Create a router
        router = SemanticRouter(embedding_model="test-model", api_key="test-key")
        
        # Mock the embedding responses
        async def mock_get_embedding(text):
            if "weather" in text.lower():
                return weather_embedding
            elif "news" in text.lower():
                return news_embedding
            elif "sports" in text.lower():
                return sports_embedding
            else:
                # Default to a mix of all three
                return [0.5, 0.3, 0.2]
        
        # Replace the _get_embedding method with our mock
        router._get_embedding = mock_get_embedding
        
        # Add routes
        await router.add_route("weather", "Get weather information for a location")
        await router.add_route("news", "Get the latest news headlines")
        await router.add_route("sports", "Get sports scores and updates")
        
        # Test routing
        assert await router.route("What's the weather like today?") == "weather"
        assert await router.route("Tell me the latest news") == "news"
        assert await router.route("What's the score of the game?") == "sports"
        
        # Test a mixed query (should route to the closest intent)
        assert await router.route("What's happening in the world today?") == "news"


@pytest.mark.asyncio
async def test_cosine_similarity():
    """Test the _cosine_similarity method."""
    router = SemanticRouter(embedding_model="test-model", api_key="test-key")
    
    # Test with orthogonal vectors
    assert router._cosine_similarity([1, 0, 0], [0, 1, 0]) == 0
    
    # Test with identical vectors
    assert router._cosine_similarity([1, 2, 3], [1, 2, 3]) == 1
    
    # Test with opposite vectors
    assert router._cosine_similarity([1, 2, 3], [-1, -2, -3]) == -1
    
    # Test with arbitrary vectors
    similarity = router._cosine_similarity([1, 2, 3], [4, 5, 6])
    expected = np.dot([1, 2, 3], [4, 5, 6]) / (np.linalg.norm([1, 2, 3]) * np.linalg.norm([4, 5, 6]))
    assert abs(similarity - expected) < 1e-10
