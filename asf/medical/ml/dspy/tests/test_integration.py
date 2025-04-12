Integration tests for DSPy integration.

This module contains integration tests for the DSPy integration components.

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
    Integration tests for DSPy integration.
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
