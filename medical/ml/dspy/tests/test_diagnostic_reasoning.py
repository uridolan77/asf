"""Unit tests for the diagnostic reasoning module.

This module contains unit tests for the DiagnosticReasoningModule implementation.
"""

import os
import asyncio
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import tempfile
import shutil

import pytest
import dspy

from asf.medical.ml.dspy.client import get_enhanced_client
from asf.medical.ml.dspy.modules.diagnostic_reasoning import DiagnosticReasoningModule, SpecialistConsultModule
from asf.medical.ml.dspy.enhanced_signatures import DiagnosticReasoning


class TestDiagnosticReasoningModule(unittest.TestCase):
    """Test cases for the DiagnosticReasoningModule."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test module initialization."""
        # Create module
        module = DiagnosticReasoningModule(
            max_diagnoses=3,
            include_rare_conditions=False
        )
        
        # Check attributes
        self.assertEqual(module.max_diagnoses, 3)
        self.assertEqual(module.include_rare_conditions, False)
        self.assertTrue(hasattr(module, 'reasoning_model'))
    
    @pytest.mark.asyncio
    async def test_preprocess_case(self):
        """Test case preprocessing."""
        # Create module
        module = DiagnosticReasoningModule()
        
        # Test case with age and gender
        case = "A 65-year-old male presents with shortness of breath."
        result = module._preprocess_case(case)
        
        # Check extracted information
        self.assertEqual(result['age'], 65)
        self.assertEqual(result['gender'], 'male')
        self.assertTrue(any('shortness of breath' in s.lower() for s in result['symptoms']))
        
        # Test case without age or gender
        case = "Patient presents with fever and cough."
        result = module._preprocess_case(case)
        
        # Check extracted information
        self.assertIsNone(result['age'])
        self.assertIsNone(result['gender'])
        self.assertTrue(any('fever and cough' in s.lower() for s in result['symptoms']))
    
    @pytest.mark.asyncio
    async def test_validate_reasoning_output(self):
        """Test reasoning output validation."""
        # Create module
        module = DiagnosticReasoningModule()
        
        # Test with object attributes
        class MockResult:
            def __init__(self):
                self.differential_diagnosis = ["Pneumonia", "Bronchitis"]
                self.recommended_tests = ["Chest X-ray", "CBC"]
                self.reasoning = "Patient has respiratory symptoms."
                self.confidence = 0.8
        
        result = MockResult()
        valid, output = module._validate_reasoning_output(result)
        
        # Check validation result
        self.assertTrue(valid)
        self.assertEqual(output['differential_diagnosis'], ["Pneumonia", "Bronchitis"])
        self.assertEqual(output['recommended_tests'], ["Chest X-ray", "CBC"])
        self.assertEqual(output['reasoning'], "Patient has respiratory symptoms.")
        self.assertEqual(output['confidence'], 0.8)
        
        # Test with dictionary
        result = {
            'differential_diagnosis': ["Pneumonia", "Bronchitis"],
            'recommended_tests': ["Chest X-ray", "CBC"],
            'reasoning': "Patient has respiratory symptoms.",
            'confidence': 0.8
        }
        valid, output = module._validate_reasoning_output(result)
        
        # Check validation result
        self.assertTrue(valid)
        self.assertEqual(output['differential_diagnosis'], ["Pneumonia", "Bronchitis"])
        
        # Test with string differential
        result = {
            'differential_diagnosis': "Pneumonia, Bronchitis",
            'recommended_tests': ["Chest X-ray", "CBC"],
            'reasoning': "Patient has respiratory symptoms.",
            'confidence': 0.8
        }
        valid, output = module._validate_reasoning_output(result)
        
        # Check validation result
        self.assertTrue(valid)
        self.assertEqual(len(output['differential_diagnosis']), 2)
        
        # Test with invalid result
        result = "Invalid result"
        valid, output = module._validate_reasoning_output(result)
        
        # Check validation result
        self.assertFalse(valid)
    
    @pytest.mark.asyncio
    async def test_forward(self):
        """Test forward method with mocked reasoning model."""
        # Create module with mocked reasoning model
        mock_model = MagicMock()
        mock_model.return_value = MagicMock(
            differential_diagnosis=["Myocardial Infarction", "Angina"],
            recommended_tests=["ECG", "Cardiac Enzymes"],
            reasoning="Patient has chest pain with risk factors.",
            confidence=0.9
        )
        
        module = DiagnosticReasoningModule(reasoning_model=mock_model)
        
        # Test case
        case = "A 55-year-old male with chest pain."
        result = module.forward(case)
        
        # Check result
        self.assertIn('differential_diagnosis', result)
        self.assertIn('recommended_tests', result)
        self.assertIn('reasoning', result)
        self.assertIn('confidence', result)
        self.assertIn('case_summary', result)
        
        # Check that model was called
        mock_model.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in forward method."""
        # Create module with mocked reasoning model that raises an exception
        mock_model = MagicMock(side_effect=Exception("Test error"))
        module = DiagnosticReasoningModule(reasoning_model=mock_model)
        
        # Test case
        case = "A 55-year-old male with chest pain."
        result = module.forward(case)
        
        # Check result contains error information
        self.assertIn('differential_diagnosis', result)
        self.assertIn('recommended_tests', result)
        self.assertIn('reasoning', result)
        self.assertIn('confidence', result)
        self.assertTrue(result['confidence'] < 0.5)  # Low confidence due to error
        
        # Check that model was called
        mock_model.assert_called_once()


class TestSpecialistConsultModule(unittest.TestCase):
    """Test cases for the SpecialistConsultModule."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test module initialization."""
        # Create module
        module = SpecialistConsultModule(specialty="cardiology")
        
        # Check attributes
        self.assertEqual(module.specialty, "cardiology")
        self.assertTrue(hasattr(module, 'base_reasoning'))
        self.assertTrue(hasattr(module, 'specialist_model'))
    
    @pytest.mark.asyncio
    async def test_forward(self):
        """Test forward method with mocked models."""
        # Create mock base reasoning module
        mock_base = MagicMock()
        mock_base.return_value = {
            'differential_diagnosis': ["Pneumonia", "Bronchitis"],
            'recommended_tests': ["Chest X-ray", "CBC"],
            'reasoning': "Patient has respiratory symptoms.",
            'confidence': 0.8
        }
        
        # Create mock specialist model
        mock_specialist = MagicMock()
        mock_specialist.return_value = MagicMock(
            specialist_assessment="Detailed pulmonology assessment.",
            specialist_diagnosis=["Pneumonia", "COPD"],
            specialist_recommendations=["Chest CT", "Sputum Culture"],
            confidence=0.85
        )
        
        # Create module with mocked models
        module = SpecialistConsultModule(
            specialty="pulmonology",
            base_reasoning_module=mock_base,
            specialist_model=mock_specialist
        )
        
        # Test case
        case = "A 65-year-old with cough and fever."
        result = module.forward(case)
        
        # Check result
        self.assertIn('base_assessment', result)
        self.assertIn('specialty', result)
        self.assertIn('specialist_assessment', result)
        self.assertIn('specialist_diagnosis', result)
        self.assertIn('specialist_recommendations', result)
        self.assertIn('confidence', result)
        
        # Check that models were called
        mock_base.assert_called_once()
        mock_specialist.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in forward method."""
        # Create mock base reasoning module
        mock_base = MagicMock()
        mock_base.return_value = {
            'differential_diagnosis': ["Pneumonia", "Bronchitis"],
            'recommended_tests': ["Chest X-ray", "CBC"],
            'reasoning': "Patient has respiratory symptoms.",
            'confidence': 0.8
        }
        
        # Create mock specialist model that raises an exception
        mock_specialist = MagicMock(side_effect=Exception("Test error"))
        
        # Create module with mocked models
        module = SpecialistConsultModule(
            specialty="pulmonology",
            base_reasoning_module=mock_base,
            specialist_model=mock_specialist
        )
        
        # Test case
        case = "A 65-year-old with cough and fever."
        result = module.forward(case)
        
        # Check result contains error information
        self.assertIn('base_assessment', result)
        self.assertIn('specialty', result)
        self.assertIn('specialist_assessment', result)
        self.assertTrue("Error" in result['specialist_assessment'])
        self.assertEqual(result['specialist_diagnosis'], [])
        self.assertEqual(result['specialist_recommendations'], [])
        self.assertTrue(result['confidence'] < 0.5)  # Low confidence due to error
        
        # Check that models were called
        mock_base.assert_called_once()
        mock_specialist.assert_called_once()


if __name__ == '__main__':
    unittest.main()
