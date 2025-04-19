import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import aiohttp

from agentor.llm_gateway.utils.http import AsyncHTTPClient


@pytest.mark.asyncio
async def test_async_http_client():
    """Test the AsyncHTTPClient class."""
    # Create a mock response
    mock_response = AsyncMock()
    mock_response.raise_for_status = AsyncMock()
    mock_response.json.return_value = {"key": "value"}
    
    # Create a mock session
    mock_session = AsyncMock()
    mock_session.get.return_value.__aenter__.return_value = mock_response
    mock_session.post.return_value.__aenter__.return_value = mock_response
    mock_session.put.return_value.__aenter__.return_value = mock_response
    mock_session.delete.return_value.__aenter__.return_value = mock_response
    
    # Patch the ClientSession
    with patch('aiohttp.ClientSession', return_value=mock_session):
        # Create an HTTP client
        client = AsyncHTTPClient()
        
        # Test GET
        result = await client.get("https://example.com/api")
        assert result == {"key": "value"}
        mock_session.get.assert_called_once_with("https://example.com/api", headers=None)
        
        # Test POST
        result = await client.post("https://example.com/api", data={"data": "value"})
        assert result == {"key": "value"}
        mock_session.post.assert_called_once_with(
            "https://example.com/api",
            json={"data": "value"},
            headers=None
        )
        
        # Test PUT
        result = await client.put("https://example.com/api", data={"data": "value"})
        assert result == {"key": "value"}
        mock_session.put.assert_called_once_with(
            "https://example.com/api",
            json={"data": "value"},
            headers=None
        )
        
        # Test DELETE
        result = await client.delete("https://example.com/api")
        assert result == {"key": "value"}
        mock_session.delete.assert_called_once_with("https://example.com/api", headers=None)
        
        # Test with headers
        headers = {"Authorization": "Bearer token"}
        result = await client.get("https://example.com/api", headers=headers)
        assert result == {"key": "value"}
        mock_session.get.assert_called_with("https://example.com/api", headers=headers)
        
        # Test close
        await client.close()
        mock_session.close.assert_called_once()
