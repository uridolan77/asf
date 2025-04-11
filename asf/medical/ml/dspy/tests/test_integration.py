"""
Integration tests for DSPy integration.

This module contains integration tests for the DSPy integration components.
"""

import os
import asyncio
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import tempfile
import shutil

import pytest
import dspy

from asf.medical.ml.dspy.dspy_client import get_dspy_client, DSPyClient
from asf.medical.ml.dspy.dspy_settings import DSPySettings, LLMProvider, CacheBackend
from asf.medical.ml.dspy.modules.medical_rag import MedicalRAGModule
from asf.medical.ml.dspy.modules.contradiction_detection import ContradictionDetectionModule
from asf.medical.ml.dspy.dspy_metrics import medical_qa_accuracy


class TestDSPyIntegration(unittest.TestCase):
    """Integration tests for DSPy integration."""
    
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
        
        # Mock dspy modules
        self.mock_dspy_modules_patcher = patch('asf.medical.ml.dspy.modules.medical_rag.dspy')
        self.mock_dspy_modules = self.mock_dspy_modules_patcher.start()
        
        # Mock Retrieve
        self.mock_retrieve = MagicMock()
        self.mock_dspy_modules.Retrieve.return_value = self.mock_retrieve
        
        # Mock ChainOfThought
        self.mock_cot = MagicMock()
        self.mock_dspy_modules.ChainOfThought.return_value = self.mock_cot
        
        # Reset the singleton instance
        DSPyClient._instance = None
    
    def tearDown(self):
        """Clean up after tests."""
        # Stop patches
        self.mock_settings_patcher.stop()
        self.mock_dspy_patcher.stop()
        self.mock_dspy_modules_patcher.stop()
        
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    @pytest.mark.asyncio
    async def test_medical_rag_integration(self):
        """Test integration of Medical RAG module with DSPy client."""
        # Set up mock responses
        self.mock_retrieve.return_value = ["Passage 1", "Passage 2", "Passage 3"]
        self.mock_cot.return_value = MagicMock(
            answer="Test answer",
            citations="Test citations"
        )
        
        # Create client
        client = await get_dspy_client()
        
        # Create Medical RAG module
        medical_rag = MedicalRAGModule(k=3)
        
        # Register module
        await client.register_module(
            name="medical_rag",
            module=medical_rag,
            description="Medical RAG module"
        )
        
        # Call module
        result = await client.call_module(
            module_name="medical_rag",
            question="What are the common side effects of statins?"
        )
        
        # Check result
        self.assertIn("answer", result)
        self.assertIn("citations", result)
        self.assertEqual(result["answer"], "Test answer")
        self.assertEqual(result["citations"], "Test citations")
        
        # Clean up
        await client.shutdown()
    
    @pytest.mark.asyncio
    async def test_contradiction_detection_integration(self):
        """Test integration of Contradiction Detection module with DSPy client."""
        # Set up mock response
        self.mock_dspy.ChainOfThought.return_value = MagicMock(
            contradiction=True,
            explanation="Test explanation",
            confidence=0.8
        )
        
        # Create client
        client = await get_dspy_client()
        
        # Create Contradiction Detection module
        contradiction_module = ContradictionDetectionModule()
        
        # Register module
        await client.register_module(
            name="contradiction_detection",
            module=contradiction_module,
            description="Contradiction Detection module"
        )
        
        # Call module
        result = await client.call_module(
            module_name="contradiction_detection",
            statement1="Aspirin is recommended for primary prevention.",
            statement2="Aspirin should not be used for primary prevention."
        )
        
        # Check result
        self.assertIn("contradiction", result)
        self.assertIn("explanation", result)
        self.assertIn("confidence", result)
        self.assertTrue(result["contradiction"])
        self.assertEqual(result["explanation"], "Test explanation")
        self.assertEqual(result["confidence"], 0.8)
        
        # Clean up
        await client.shutdown()
    
    @pytest.mark.asyncio
    async def test_metrics_integration(self):
        """Test integration of metrics with DSPy modules."""
        # Set up mock response for judge
        self.mock_dspy.Predict.return_value = MagicMock(
            score=0.75,
            explanation="Test explanation"
        )
        
        # Create prediction and example
        prediction = MagicMock(answer="Test answer")
        example = {"_output": {"answer": "Reference answer"}}
        
        # Calculate accuracy
        with patch('asf.medical.ml.dspy.dspy_metrics.dspy') as mock_dspy_metrics:
            mock_dspy_metrics.Predict.return_value = MagicMock(
                return_value=MagicMock(score=0.75)
            )
            
            accuracy = medical_qa_accuracy(prediction, example)
        
        # Check accuracy
        self.assertEqual(accuracy, 0.75)
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """Test end-to-end workflow with multiple components."""
        # Set up mock responses
        self.mock_retrieve.return_value = ["Passage 1", "Passage 2", "Passage 3"]
        self.mock_cot.return_value = MagicMock(
            answer="Test answer",
            citations="Test citations"
        )
        self.mock_dspy.Predict.return_value = MagicMock(
            return_value=MagicMock(score=0.75)
        )
        
        # Create client
        client = await get_dspy_client()
        
        # Create Medical RAG module
        medical_rag = MedicalRAGModule(k=3)
        
        # Register module
        await client.register_module(
            name="medical_rag",
            module=medical_rag,
            description="Medical RAG module"
        )
        
        # Call module
        result = await client.call_module(
            module_name="medical_rag",
            question="What are the common side effects of statins?"
        )
        
        # Check result
        self.assertIn("answer", result)
        self.assertEqual(result["answer"], "Test answer")
        
        # Create example for evaluation
        example = {"_output": {"answer": "Reference answer"}}
        
        # Evaluate result
        with patch('asf.medical.ml.dspy.dspy_metrics.dspy') as mock_dspy_metrics:
            mock_dspy_metrics.Predict.return_value = MagicMock(
                return_value=MagicMock(score=0.75)
            )
            
            # Convert result to prediction-like object
            prediction = MagicMock(answer=result["answer"])
            
            accuracy = medical_qa_accuracy(prediction, example)
        
        # Check accuracy
        self.assertEqual(accuracy, 0.75)
        
        # Clean up
        await client.shutdown()


if __name__ == '__main__':
    unittest.main()
