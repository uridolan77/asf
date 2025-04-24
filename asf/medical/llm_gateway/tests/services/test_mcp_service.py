"""
Unit tests for MCP service implementation.

This module contains tests for the MCP service implementation of the LLM Gateway service abstraction layer.
"""

import unittest
import asyncio
from typing import Dict, Any, List
from unittest.mock import MagicMock, patch, AsyncMock

from asf.medical.llm_gateway.services.mcp_service import MCPService
from asf.medical.llm_gateway.interfaces.exceptions import (
    ModelNotAvailableException,
    ServiceUnavailableException,
    InvalidRequestException
)
from asf.medical.llm_gateway.mcp.errors import MCPError
from asf.medical.llm_gateway.core.models import LLMResponse, ContentItem, ChatMessage


class TestMCPService(unittest.TestCase):
    """Test cases for the MCP service implementation."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the MCPProvider
        self.mock_provider = MagicMock()
        
        # Create an MCPService instance with a mock provider
        with patch('asf.medical.llm_gateway.services.mcp_service.MCPProvider', return_value=self.mock_provider):
            self.service = MCPService({'mcp': {}})
        
        # Set up common test data
        self.test_model = "mcp-test-model"
        self.test_prompt = "Test prompt"
        self.test_params = {"temperature": 0.7}
        
    def test_create_llm_request(self):
        """Test the creation of LLM requests."""
        request = self.service._create_llm_request(self.test_prompt, self.test_model, self.test_params)
        
        # Verify the request is correctly formed
        self.assertEqual(request.model, self.test_model)
        self.assertEqual(request.provider_id, "mcp")
        self.assertEqual(len(request.content_items), 1)
        self.assertEqual(request.content_items[0].content, self.test_prompt)
        self.assertEqual(request.provider_parameters.temperature, 0.7)
    
    def test_extract_text(self):
        """Test the extraction of text from LLM responses."""
        # Create a mock response
        mock_response = MagicMock(spec=LLMResponse)
        mock_response.content_items = [ContentItem(content="Test response", content_type="text")]
        
        # Test extraction
        result = self.service._extract_text(mock_response)
        self.assertEqual(result, "Test response")
        
        # Test with empty response
        mock_response.content_items = []
        result = self.service._extract_text(mock_response)
        self.assertEqual(result, "")
    
    def test_format_chat_response(self):
        """Test the formatting of chat responses."""
        # Test with chat_message
        mock_response = MagicMock(spec=LLMResponse)
        mock_response.chat_message = ChatMessage(role="assistant", content="Test chat response")
        mock_response.content_items = []
        
        result = self.service._format_chat_response(mock_response)
        self.assertEqual(result["role"], "assistant")
        self.assertEqual(result["content"], "Test chat response")
        
        # Test with content_items only
        mock_response.chat_message = None
        mock_response.content_items = [ContentItem(content="Test content", content_type="text")]
        
        result = self.service._format_chat_response(mock_response)
        self.assertEqual(result["role"], "assistant")
        self.assertEqual(result["content"], "Test content")
        
        # Test with empty response
        mock_response.chat_message = None
        mock_response.content_items = []
        
        result = self.service._format_chat_response(mock_response)
        self.assertEqual(result["role"], "assistant")
        self.assertEqual(result["content"], "")
    
    def test_translate_error(self):
        """Test the translation of MCP-specific errors to service-level exceptions."""
        # Test model_not_available error
        error = MCPError("Model not available", error_type="model_not_available")
        result = self.service._translate_error(error)
        self.assertIsInstance(result, ModelNotAvailableException)
        
        # Test service_unavailable error
        error = MCPError("Service unavailable", error_type="service_unavailable")
        result = self.service._translate_error(error)
        self.assertIsInstance(result, ServiceUnavailableException)
        
        # Test invalid_request error
        error = MCPError("Invalid request", error_type="invalid_request")
        result = self.service._translate_error(error)
        self.assertIsInstance(result, InvalidRequestException)


class TestMCPServiceAsync(unittest.IsolatedAsyncioTestCase):
    """Test cases for the asynchronous methods of the MCP service."""

    async def asyncSetUp(self):
        """Set up test fixtures."""
        # Mock the MCPProvider
        self.mock_provider = AsyncMock()
        
        # Create an MCPService instance with a mock provider
        with patch('asf.medical.llm_gateway.services.mcp_service.MCPProvider', return_value=self.mock_provider):
            self.service = MCPService({'mcp': {}})
        
        # Set up common test data
        self.test_model = "mcp-test-model"
        self.test_prompt = "Test prompt"
        self.test_params = {"temperature": 0.7}
    
    async def test_generate_text(self):
        """Test the generate_text method."""
        # Set up mock response
        mock_response = MagicMock(spec=LLMResponse)
        mock_response.content_items = [ContentItem(content="Generated text", content_type="text")]
        self.mock_provider.generate = AsyncMock(return_value=mock_response)
        
        # Call the method
        result = await self.service._generate_text_impl(self.test_prompt, self.test_model, self.test_params)
        
        # Verify the result
        self.assertEqual(result, "Generated text")
        
        # Verify the provider was called with the correct request
        self.mock_provider.generate.assert_called_once()
        request = self.mock_provider.generate.call_args[0][0]
        self.assertEqual(request.model, self.test_model)
    
    async def test_get_embeddings(self):
        """Test the get_embeddings method."""
        # Set up mock response
        mock_embeddings = [[0.1, 0.2, 0.3]]
        mock_response = MagicMock(spec=LLMResponse)
        mock_response.embeddings = mock_embeddings
        self.mock_provider.get_embeddings = AsyncMock(return_value=mock_response)
        
        # Call the method
        result = await self.service._get_embeddings_impl(self.test_prompt, self.test_model, self.test_params)
        
        # Verify the result
        self.assertEqual(result, mock_embeddings)
        
        # Verify the provider was called with the correct request
        self.mock_provider.get_embeddings.assert_called_once()
        request = self.mock_provider.get_embeddings.call_args[0][0]
        self.assertEqual(request.model, self.test_model)
    
    async def test_chat(self):
        """Test the chat method."""
        # Set up test messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"}
        ]
        
        # Set up mock response
        mock_response = MagicMock(spec=LLMResponse)
        mock_response.chat_message = ChatMessage(role="assistant", content="Hello, how can I help you?")
        self.mock_provider.chat = AsyncMock(return_value=mock_response)
        
        # Call the method
        result = await self.service._chat_impl(messages, self.test_model, self.test_params)
        
        # Verify the result
        self.assertEqual(result["role"], "assistant")
        self.assertEqual(result["content"], "Hello, how can I help you?")
        
        # Verify the provider was called with the correct request
        self.mock_provider.chat.assert_called_once()
        request = self.mock_provider.chat.call_args[0][0]
        self.assertEqual(request.model, self.test_model)
    
    async def test_error_handling(self):
        """Test error handling in async methods."""
        # Set up the provider to raise an error
        self.mock_provider.generate = AsyncMock(side_effect=MCPError("Test error", error_type="model_not_available"))
        
        # Verify that the error is translated correctly
        with self.assertRaises(ModelNotAvailableException):
            await self.service.generate_text(self.test_prompt, self.test_model, self.test_params)


if __name__ == '__main__':
    unittest.main()