import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from agentor.llm_gateway.llm.base import LLMRequest, LLMResponse
from agentor.llm_gateway.llm.cache import LLMCache, CachedLLM


@pytest.mark.asyncio
async def test_llm_cache():
    """Test the LLMCache class."""
    # Create a cache
    cache = LLMCache(ttl=60)
    
    # Create a request and response
    request = LLMRequest(
        prompt="Test prompt",
        model="test-model",
        temperature=0.7
    )
    
    response = LLMResponse(
        text="Test response",
        model="test-model",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    )
    
    # Test cache miss
    cached_response = await cache.get(request)
    assert cached_response is None
    
    # Test cache set
    await cache.set(request, response)
    
    # Test cache hit
    cached_response = await cache.get(request)
    assert cached_response is not None
    assert cached_response.text == "Test response"
    assert cached_response.model == "test-model"
    assert cached_response.usage["prompt_tokens"] == 10
    assert cached_response.usage["completion_tokens"] == 5
    assert cached_response.usage["total_tokens"] == 15


@pytest.mark.asyncio
async def test_cached_llm():
    """Test the CachedLLM class."""
    # Create a mock LLM
    mock_llm = AsyncMock()
    mock_response = LLMResponse(
        text="Test response",
        model="test-model",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    )
    mock_llm.generate.return_value = mock_response
    
    # Create a cache
    cache = LLMCache(ttl=60)
    
    # Create a cached LLM
    cached_llm = CachedLLM(llm=mock_llm, cache=cache)
    
    # Create a request
    request = LLMRequest(
        prompt="Test prompt",
        model="test-model",
        temperature=0.7
    )
    
    # Test cache miss (should call the LLM)
    response1 = await cached_llm.generate(request)
    assert response1.text == "Test response"
    mock_llm.generate.assert_called_once_with(request)
    
    # Reset the mock
    mock_llm.generate.reset_mock()
    
    # Test cache hit (should not call the LLM)
    response2 = await cached_llm.generate(request)
    assert response2.text == "Test response"
    mock_llm.generate.assert_not_called()
    
    # Test different request (should call the LLM)
    request2 = LLMRequest(
        prompt="Different prompt",
        model="test-model",
        temperature=0.7
    )
    
    response3 = await cached_llm.generate(request2)
    assert response3.text == "Test response"
    mock_llm.generate.assert_called_once_with(request2)
