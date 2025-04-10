"""
Integration Test for ASF Framework Enhancements

This module provides integration tests for the ASF framework enhancements,
including TSMixer, Lorentz embeddings, Ray orchestration, and SHAP explainability.
"""

import os
import sys
import unittest
import logging
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components
from asf.medical.layer1_knowledge_substrate.temporal.tsmixer import TSMixer, TSMixerEncoder
from asf.medical.layer1_knowledge_substrate.embeddings.lorentz_embeddings import LorentzEmbedding
from asf.medical.orchestration.ray_orchestrator import RayOrchestrator, RayConfig
from asf.medical.models.shap_explainer import ContradictionExplainer, ContradictionVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test-integration")

class TestIntegration(unittest.TestCase):
    """Integration tests for ASF framework enhancements."""

    def setUp(self):
        """Set up test fixtures."""
        # Initialize components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create TSMixer model
        self.tsmixer = TSMixer(
            seq_len=24,
            num_features=5,
            num_blocks=2,
            forecast_horizon=12
        ).to(self.device)

        # Create Lorentz embedding
        self.lorentz_embedding = LorentzEmbedding(
            num_embeddings=1000,
            embedding_dim=32,
            k=-1.0
        ).to(self.device)

        # Create Ray orchestrator
        self.ray_config = RayConfig(use_ray=False)  # Use local execution for testing
        self.orchestrator = RayOrchestrator(config=self.ray_config)

        # Register test functions
        self.orchestrator.register_function(self._add, "add")
        self.orchestrator.register_function(self._multiply, "multiply")

    def _add(self, a, b):
        """Test function for addition."""
        return a + b

    def _multiply(self, a, b):
        """Test function for multiplication."""
        return a * b

    def test_tsmixer_forward(self):
        """Test TSMixer forward pass."""
        # Create random input
        x = torch.randn(16, 24, 5).to(self.device)

        # Forward pass
        output = self.tsmixer(x)

        # Check output shape
        self.assertEqual(output.shape, (16, 12, 5))

        # Check output is not NaN
        self.assertFalse(torch.isnan(output).any())

    def test_lorentz_embedding(self):
        """Test Lorentz embedding."""
        # Create random indices
        indices = torch.randint(0, 1000, (16,)).to(self.device)

        # Get embeddings
        embeddings = self.lorentz_embedding(indices)

        # Check output shape
        self.assertEqual(embeddings.shape, (16, 32))

        # Check output is not NaN
        self.assertFalse(torch.isnan(embeddings).any())

        # Check embeddings are on the manifold
        # For Lorentz manifold with k=-1, we need x_0^2 - sum(x_i^2) = 1
        x_0_squared = embeddings[:, 0] ** 2
        x_i_squared_sum = torch.sum(embeddings[:, 1:] ** 2, dim=1)
        manifold_constraint = x_0_squared - x_i_squared_sum

        # Check constraint is close to 1
        self.assertTrue(torch.allclose(manifold_constraint, torch.ones_like(manifold_constraint), rtol=1e-5, atol=1e-5))

    def test_ray_orchestrator(self):
        """Test Ray orchestrator."""
        # Create task
        task_id = self.orchestrator.create_task(
            name="add_task",
            function_name="add",
            args=[1, 2]
        )

        # Execute task
        result = self.orchestrator.execute_task(task_id)

        # Check result
        self.assertEqual(result, 3)

        # Create workflow
        task1_id = self.orchestrator.create_task(
            name="add_task",
            function_name="add",
            args=[1, 2]
        )

        task2_id = self.orchestrator.create_task(
            name="multiply_task",
            function_name="multiply",
            args=[3, 4],
            dependencies=[task1_id]
        )

        # Execute workflow
        results = self.orchestrator.execute_workflow([task1_id, task2_id])

        # Check results
        self.assertEqual(results[task1_id], 3)
        self.assertEqual(results[task2_id], 12)

    def test_contradiction_explanation(self):
        """Test contradiction explanation."""
        # Create mock model and tokenizer
        class MockModel:
            def __call__(self, **kwargs):
                class MockOutput:
                    def __init__(self):
                        self.logits = torch.tensor([[0.2, 0.8]])
                return MockOutput()

        class MockTokenizer:
            def __call__(self, *args, **kwargs):
                return {
                    "input_ids": torch.tensor([[101, 2054, 2003, 1037, 3231, 102]]),
                    "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]])
                }

        # Create explainer
        explainer = ContradictionExplainer(
            model=MockModel(),
            tokenizer=MockTokenizer(),
            device=self.device
        )

        # Create explanation
        explanation = explainer.explain_contradiction(
            claim1="The treatment significantly reduced mortality rates.",
            claim2="The treatment did not show any significant effect on mortality.",
            contradiction_score=0.8,
            use_shap=False  # Disable SHAP for testing
        )

        # Check explanation
        self.assertEqual(explanation.claim1, "The treatment significantly reduced mortality rates.")
        self.assertEqual(explanation.claim2, "The treatment did not show any significant effect on mortality.")
        self.assertEqual(explanation.contradiction_score, 0.8)

        # Create visualizer
        visualizer = ContradictionVisualizer()

        # Generate HTML report
        output_path = "test_report.html"
        visualizer.generate_html_report(explanation, output_path)

        # Check file exists
        self.assertTrue(os.path.exists(output_path))

        # Clean up
        os.remove(output_path)

if __name__ == "__main__":
    unittest.main()
