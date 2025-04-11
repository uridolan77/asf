"""
Unit tests for DSPy client.

This module contains unit tests for the DSPy client implementation.
"""

import os
import asyncio
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import tempfile
import shutil

import pytest
import dspy

from asf.medical.ml.dspy.dspy_client import DSPyClient
from asf.medical.ml.dspy.dspy_settings import DSPySettings, LLMProvider, CacheBackend
from asf.medical.ml.dspy.dspy_module_base import DSPyModuleBase


class TestDSPyClient(unittest.TestCase):
    """Test cases for DSPy client."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for cache
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock settings
        self.mock_settings_patcher = patch('asf.medical.ml.dspy.dspy_client.dspy_settings')
        self.mock_settings = self.mock_settings_patcher.start()
        self.mock_settings.LLM_PROVIDER = LLMProvider.OPENAI
        self.mock_settings.LLM_API_KEY.get_secret_value.return_value = "test-api-key"
        self.mock_settings.DEFAULT_MODEL = "gpt-3.5-turbo"
        self.mock_settings.CACHE_BACKEND = CacheBackend.DISK
        self.mock_settings.CACHE_DIRECTORY = self.temp_dir
        self.mock_settings.ENABLE_TELEMETRY = False
        self.mock_settings.THREAD_LIMIT = 2
        
        # Mock dspy
        self.mock_dspy_patcher = patch('asf.medical.ml.dspy.dspy_client.dspy')
        self.mock_dspy = self.mock_dspy_patcher.start()
        
        # Mock OpenAI LM
        self.mock_lm = MagicMock()
        self.mock_dspy.OpenAI.return_value = self.mock_lm
    
    def tearDown(self):
        """Clean up after tests."""
        # Stop patches
        self.mock_settings_patcher.stop()
        self.mock_dspy_patcher.stop()
        
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_singleton_pattern(self):
        """Test that DSPyClient follows the singleton pattern."""
        # Reset the singleton instance
        DSPyClient._instance = None
        
        # Create two instances
        client1 = DSPyClient()
        client2 = DSPyClient()
        
        # Check that they are the same object
        self.assertIs(client1, client2)
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test client initialization."""
        # Reset the singleton instance
        DSPyClient._instance = None
        
        # Create client
        client = DSPyClient()
        await client.initialize()
        
        # Check that client is initialized
        self.assertTrue(client._initialized)
        self.assertIsNotNone(client.lm)
        self.assertIsNotNone(client.cache)
        self.assertIsNotNone(client._executor)
        
        # Clean up
        await client.shutdown()
    
    @pytest.mark.asyncio
    async def test_call_lm(self):
        """Test calling the language model."""
        # Reset the singleton instance
        DSPyClient._instance = None
        
        # Set up mock response
        mock_response = {"choices": [{"message": {"content": "Test response"}}]}
        self.mock_lm.call.return_value = mock_response
        
        # Create client
        client = DSPyClient()
        await client.initialize()
        
        # Call LM
        response = await client.call_lm("Test prompt")
        
        # Check response
        self.assertEqual(response, mock_response)
        self.mock_lm.call.assert_called_once()
        
        # Clean up
        await client.shutdown()
    
    @pytest.mark.asyncio
    async def test_caching(self):
        """Test caching functionality."""
        # Reset the singleton instance
        DSPyClient._instance = None
        
        # Set up mock response
        mock_response = {"choices": [{"message": {"content": "Test response"}}]}
        self.mock_lm.call.return_value = mock_response
        
        # Create client
        client = DSPyClient()
        await client.initialize()
        
        # First call should hit the API
        await client.call_lm("Test prompt")
        self.assertEqual(self.mock_lm.call.call_count, 1)
        
        # Second call with same prompt should use cache
        await client.call_lm("Test prompt")
        self.assertEqual(self.mock_lm.call.call_count, 1)  # Still 1
        
        # Different prompt should hit the API again
        await client.call_lm("Different prompt")
        self.assertEqual(self.mock_lm.call.call_count, 2)
        
        # Clear cache
        await client.clear_cache()
        
        # Same prompt should hit the API again after cache clear
        await client.call_lm("Test prompt")
        self.assertEqual(self.mock_lm.call.call_count, 3)
        
        # Clean up
        await client.shutdown()
    
    @pytest.mark.asyncio
    async def test_module_registration(self):
        """Test module registration and retrieval."""
        # Reset the singleton instance
        DSPyClient._instance = None
        
        # Create client
        client = DSPyClient()
        await client.initialize()
        
        # Create a test module
        class TestModule(DSPyModuleBase):
            def forward(self, input_text):
                return {"output": f"Processed: {input_text}"}
        
        test_module = TestModule()
        
        # Register module
        await client.register_module(
            name="test_module",
            module=test_module,
            description="Test module"
        )
        
        # List modules
        modules = client.list_modules()
        self.assertEqual(len(modules), 1)
        self.assertEqual(modules[0]["name"], "test_module")
        self.assertEqual(modules[0]["description"], "Test module")
        
        # Get module
        retrieved_module = client.get_module("test_module")
        self.assertIs(retrieved_module, test_module)
        
        # Clean up
        await client.shutdown()
    
    @pytest.mark.asyncio
    async def test_module_call(self):
        """Test calling a registered module."""
        # Reset the singleton instance
        DSPyClient._instance = None
        
        # Create client
        client = DSPyClient()
        await client.initialize()
        
        # Create a test module
        class TestModule(DSPyModuleBase):
            def forward(self, input_text):
                return {"output": f"Processed: {input_text}"}
        
        test_module = TestModule()
        
        # Register module
        await client.register_module(
            name="test_module",
            module=test_module,
            description="Test module"
        )
        
        # Call module
        result = await client.call_module(
            module_name="test_module",
            input_text="test input"
        )
        
        # Check result
        self.assertEqual(result["output"], "Processed: test input")
        
        # Clean up
        await client.shutdown()
    
    @pytest.mark.asyncio
    async def test_shutdown(self):
        """Test client shutdown."""
        # Reset the singleton instance
        DSPyClient._instance = None
        
        # Create client
        client = DSPyClient()
        await client.initialize()
        
        # Shutdown
        await client.shutdown()
        
        # Check that client is shut down
        self.assertFalse(client._initialized)
        
        # Reinitialize to ensure it works after shutdown
        await client.initialize()
        self.assertTrue(client._initialized)
        
        # Clean up
        await client.shutdown()


if __name__ == '__main__':
    unittest.main()
