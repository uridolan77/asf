"""
Test SHAP Explainer

This module provides tests for the SHAP-based explainability for contradiction analysis
in the ASF framework.
"""

import os
import sys
import unittest
import logging
import torch
import json
from unittest.mock import MagicMock, patch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components
from asf.medical.models.shap_explainer import (
    ContradictionExplanation, ContradictionExplainer, ContradictionVisualizer
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test-shap-explainer")

class TestContradictionExplanation(unittest.TestCase):
    """Test cases for ContradictionExplanation class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.claim1 = "The treatment significantly reduced mortality rates."
        self.claim2 = "The treatment did not show any significant effect on mortality."
        self.contradiction_score = 0.85
        self.explanation_type = "shap"
        self.influential_words = {
            "significantly": 0.45,
            "reduced": 0.38,
            "not": -0.52,
            "any": -0.25,
            "effect": -0.18
        }
        self.negation_patterns = [
            {
                "type": "direct_negation",
                "word": "not",
                "claim": "claim2",
                "context": "The treatment did not show any significant effect"
            }
        ]
        self.multimodal_factors = {
            "study_design_difference": {
                "design1": "RCT",
                "design2": "observational",
                "impact": "high"
            }
        }
        self.visualization_data = {
            "shap_values": {
                "values": [0.1, 0.2, -0.3, 0.4, -0.5],
                "data": ["The", "treatment", "significantly", "reduced", "mortality"]
            }
        }
        
        # Create explanation
        self.explanation = ContradictionExplanation(
            claim1=self.claim1,
            claim2=self.claim2,
            contradiction_score=self.contradiction_score,
            explanation_type=self.explanation_type,
            influential_words=self.influential_words,
            negation_patterns=self.negation_patterns,
            multimodal_factors=self.multimodal_factors,
            visualization_data=self.visualization_data
        )
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        # Convert to dictionary
        explanation_dict = self.explanation.to_dict()
        
        # Check dictionary contains all fields
        self.assertEqual(explanation_dict["claim1"], self.claim1)
        self.assertEqual(explanation_dict["claim2"], self.claim2)
        self.assertEqual(explanation_dict["contradiction_score"], self.contradiction_score)
        self.assertEqual(explanation_dict["explanation_type"], self.explanation_type)
        self.assertEqual(explanation_dict["influential_words"], self.influential_words)
        self.assertEqual(explanation_dict["negation_patterns"], self.negation_patterns)
        self.assertEqual(explanation_dict["multimodal_factors"], self.multimodal_factors)
        self.assertEqual(explanation_dict["visualization_data"], self.visualization_data)
    
    def test_to_json(self):
        """Test conversion to JSON."""
        # Convert to JSON
        explanation_json = self.explanation.to_json()
        
        # Parse JSON
        explanation_dict = json.loads(explanation_json)
        
        # Check dictionary contains all fields
        self.assertEqual(explanation_dict["claim1"], self.claim1)
        self.assertEqual(explanation_dict["claim2"], self.claim2)
        self.assertEqual(explanation_dict["contradiction_score"], self.contradiction_score)
        self.assertEqual(explanation_dict["explanation_type"], self.explanation_type)
        self.assertEqual(explanation_dict["influential_words"], self.influential_words)
        self.assertEqual(explanation_dict["negation_patterns"], self.negation_patterns)
        self.assertEqual(explanation_dict["multimodal_factors"], self.multimodal_factors)
        self.assertEqual(explanation_dict["visualization_data"], self.visualization_data)
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        # Create dictionary
        explanation_dict = {
            "claim1": self.claim1,
            "claim2": self.claim2,
            "contradiction_score": self.contradiction_score,
            "explanation_type": self.explanation_type,
            "influential_words": self.influential_words,
            "negation_patterns": self.negation_patterns,
            "multimodal_factors": self.multimodal_factors,
            "visualization_data": self.visualization_data
        }
        
        # Create explanation from dictionary
        explanation = ContradictionExplanation.from_dict(explanation_dict)
        
        # Check explanation fields
        self.assertEqual(explanation.claim1, self.claim1)
        self.assertEqual(explanation.claim2, self.claim2)
        self.assertEqual(explanation.contradiction_score, self.contradiction_score)
        self.assertEqual(explanation.explanation_type, self.explanation_type)
        self.assertEqual(explanation.influential_words, self.influential_words)
        self.assertEqual(explanation.negation_patterns, self.negation_patterns)
        self.assertEqual(explanation.multimodal_factors, self.multimodal_factors)
        self.assertEqual(explanation.visualization_data, self.visualization_data)
    
    def test_from_json(self):
        """Test creation from JSON."""
        # Create JSON
        explanation_json = json.dumps({
            "claim1": self.claim1,
            "claim2": self.claim2,
            "contradiction_score": self.contradiction_score,
            "explanation_type": self.explanation_type,
            "influential_words": self.influential_words,
            "negation_patterns": self.negation_patterns,
            "multimodal_factors": self.multimodal_factors,
            "visualization_data": self.visualization_data
        })
        
        # Create explanation from JSON
        explanation = ContradictionExplanation.from_json(explanation_json)
        
        # Check explanation fields
        self.assertEqual(explanation.claim1, self.claim1)
        self.assertEqual(explanation.claim2, self.claim2)
        self.assertEqual(explanation.contradiction_score, self.contradiction_score)
        self.assertEqual(explanation.explanation_type, self.explanation_type)
        self.assertEqual(explanation.influential_words, self.influential_words)
        self.assertEqual(explanation.negation_patterns, self.negation_patterns)
        self.assertEqual(explanation.multimodal_factors, self.multimodal_factors)
        self.assertEqual(explanation.visualization_data, self.visualization_data)

class TestContradictionExplainer(unittest.TestCase):
    """Test cases for ContradictionExplainer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock model and tokenizer
        self.model = MagicMock()
        self.tokenizer = MagicMock()
        
        # Mock model output
        self.model.return_value = MagicMock(logits=torch.tensor([[0.2, 0.8]]))
        
        # Mock tokenizer output
        self.tokenizer.return_value = {
            "input_ids": torch.tensor([[101, 2054, 2003, 1037, 3231, 102]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]])
        }
        
        # Create explainer
        self.explainer = ContradictionExplainer(
            model=self.model,
            tokenizer=self.tokenizer,
            device="cpu"
        )
        
        # Test claims
        self.claim1 = "The treatment significantly reduced mortality rates."
        self.claim2 = "The treatment did not show any significant effect on mortality."
        self.contradiction_score = 0.85
    
    @patch("asf.medical.models.shap_explainer.HAS_SHAP", False)
    def test_explain_contradiction_without_shap(self):
        """Test explanation without SHAP."""
        # Explain contradiction
        explanation = self.explainer.explain_contradiction(
            claim1=self.claim1,
            claim2=self.claim2,
            contradiction_score=self.contradiction_score,
            use_shap=True,  # This will be ignored since HAS_SHAP is False
            use_negation_detection=True,
            use_multimodal_factors=False
        )
        
        # Check explanation fields
        self.assertEqual(explanation.claim1, self.claim1)
        self.assertEqual(explanation.claim2, self.claim2)
        self.assertEqual(explanation.contradiction_score, self.contradiction_score)
        self.assertEqual(explanation.explanation_type, "combined")
        
        # Check negation patterns
        self.assertTrue(len(explanation.negation_patterns) > 0)
        
        # Check influential words (should be empty since SHAP is not available)
        self.assertEqual(explanation.influential_words, {})
    
    def test_detect_negation_patterns(self):
        """Test negation pattern detection."""
        # Detect negation patterns
        patterns = self.explainer._detect_negation_patterns(
            claim1=self.claim1,
            claim2=self.claim2
        )
        
        # Check patterns
        self.assertTrue(len(patterns) > 0)
        
        # Check for direct negation
        direct_negations = [p for p in patterns if p["type"] == "direct_negation"]
        self.assertTrue(len(direct_negations) > 0)
        self.assertEqual(direct_negations[0]["word"], "not")
        self.assertEqual(direct_negations[0]["claim"], "claim2")
    
    def test_analyze_multimodal_factors(self):
        """Test multimodal factor analysis."""
        # Create metadata
        metadata = {
            "study_design1": "RCT",
            "study_design2": "observational",
            "sample_size1": 1000,
            "sample_size2": 50,
            "publication_date1": "2020-01-01",
            "publication_date2": "2010-01-01",
            "population1": "adults",
            "population2": "children"
        }
        
        # Analyze multimodal factors
        factors = self.explainer._analyze_multimodal_factors(metadata)
        
        # Check factors
        self.assertTrue(len(factors) > 0)
        
        # Check for study design difference
        self.assertIn("study_design_difference", factors)
        self.assertEqual(factors["study_design_difference"]["design1"], "RCT")
        self.assertEqual(factors["study_design_difference"]["design2"], "observational")
        self.assertEqual(factors["study_design_difference"]["impact"], "high")
        
        # Check for sample size difference
        self.assertIn("sample_size_difference", factors)
        self.assertEqual(factors["sample_size_difference"]["size1"], 1000)
        self.assertEqual(factors["sample_size_difference"]["size2"], 50)
        self.assertEqual(factors["sample_size_difference"]["impact"], "high")
        
        # Check for publication date difference
        self.assertIn("publication_date_difference", factors)
        self.assertEqual(factors["publication_date_difference"]["date1"], "2020-01-01")
        self.assertEqual(factors["publication_date_difference"]["date2"], "2010-01-01")
        self.assertEqual(factors["publication_date_difference"]["year_difference"], 10)
        
        # Check for population difference
        self.assertIn("population_difference", factors)
        self.assertEqual(factors["population_difference"]["population1"], "adults")
        self.assertEqual(factors["population_difference"]["population2"], "children")
        self.assertEqual(factors["population_difference"]["impact"], "high")

class TestContradictionVisualizer(unittest.TestCase):
    """Test cases for ContradictionVisualizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create explanation
        self.explanation = ContradictionExplanation(
            claim1="The treatment significantly reduced mortality rates.",
            claim2="The treatment did not show any significant effect on mortality.",
            contradiction_score=0.85,
            explanation_type="shap",
            influential_words={
                "significantly": 0.45,
                "reduced": 0.38,
                "not": -0.52,
                "any": -0.25,
                "effect": -0.18
            },
            negation_patterns=[
                {
                    "type": "direct_negation",
                    "word": "not",
                    "claim": "claim2",
                    "context": "The treatment did not show any significant effect"
                }
            ],
            multimodal_factors={
                "study_design_difference": {
                    "design1": "RCT",
                    "design2": "observational",
                    "impact": "high"
                }
            },
            visualization_data={
                "shap_values": {
                    "values": [0.1, 0.2, -0.3, 0.4, -0.5],
                    "data": ["The", "treatment", "significantly", "reduced", "mortality"]
                }
            }
        )
        
        # Create visualizer
        self.visualizer = ContradictionVisualizer()
    
    @patch("asf.medical.models.shap_explainer.HAS_SHAP", False)
    def test_visualize_shap_without_shap(self):
        """Test SHAP visualization without SHAP."""
        # This should not raise an exception
        self.visualizer.visualize_shap(self.explanation)
    
    def test_generate_html_report(self):
        """Test HTML report generation."""
        # Generate HTML report
        output_path = "test_report.html"
        self.visualizer.generate_html_report(self.explanation, output_path)
        
        # Check file exists
        self.assertTrue(os.path.exists(output_path))
        
        # Check file content
        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read()
            
            # Check for key elements
            self.assertIn("Contradiction Explanation Report", content)
            self.assertIn("The treatment significantly reduced mortality rates.", content)
            self.assertIn("The treatment did not show any significant effect on mortality.", content)
            self.assertIn("0.8500", content)
            self.assertIn("Influential Words", content)
            self.assertIn("Negation Patterns", content)
            self.assertIn("Direct Negation", content)
            self.assertIn("not", content)
            self.assertIn("Multimodal Factors", content)
            self.assertIn("Study Design Difference", content)
            self.assertIn("RCT", content)
            self.assertIn("observational", content)
        
        # Clean up
        os.remove(output_path)

if __name__ == "__main__":
    unittest.main()
