Unit tests for DSPy client.

This module contains unit tests for the DSPy client implementation.

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
    Test cases for DSPy client.
        # Reset the singleton instance
        DSPyClient._instance = None
        
        # Create client
        client = DSPyClient()
        await client.initialize()
        
        # Create a test module
        class TestModule(DSPyModuleBase):
            """
            TestModule class.
            
            This class provides functionality for...
            """
            def forward(self, input_text):
                """
                forward function.
                
                This function provides functionality for...
                Args:
                    input_text: Description of input_text
                """
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
            """
            TestModule class.
            
            This class provides functionality for...
            """
            def forward(self, input_text):
                """
                forward function.
                
                This function provides functionality for...
                Args:
                    input_text: Description of input_text
                """
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
