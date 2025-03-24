
import asyncio
import time
import unittest
import logging
import random
import uuid
from unittest.mock import MagicMock, patch

from asf.layer4_environmental_coupling.components.active_inference_controller import ActiveInferenceController
from asf.layer4_environmental_coupling.models import EnvironmentalCoupling, ActiveInferenceTest
from asf.layer4_environmental_coupling.enums import CouplingType, CouplingStrength, CouplingState

# Configure logging
logging.basicConfig(level=logging.INFO)

class TestActiveInference(unittest.TestCase):
    
    def setUp(self):
        # Create mock knowledge substrate
        self.knowledge_substrate = MagicMock()
        
        # Create mock entity
        self.mock_entity = {
            'id': 'test_entity_001',
            'type': 'service',
            'properties': {},
            'created_at': time.time()
        }
        
        # Set up mock for get_entity method
        self.knowledge_substrate.get_entity = MagicMock(return_value=self.mock_entity)
        
        # Create active inference controller with mock substrate
        self.active_inference = ActiveInferenceController(self.knowledge_substrate)
        
        # Create mock coupling registry
        self.mock_registry = MagicMock()
        
        # Set up mock for get_coupling method
        self.mock_coupling = EnvironmentalCoupling(
            id='test_coupling_001',
            internal_entity_id='test_entity_001',
            environmental_entity_id='env_entity_001',
            coupling_type=CouplingType.INFORMATIONAL,
            coupling_strength=0.7,
            coupling_state=CouplingState.ACTIVE,
            bayesian_confidence=0.6,
            interaction_count=5,
            prediction_precision=1.0
        )
        
        self.mock_registry.get_coupling = MagicMock(return_value=self.mock_coupling)
        self.mock_registry.update_coupling = MagicMock(return_value=True)
        
        # Connect registry to controller
        self.active_inference.set_coupling_registry(self.mock_registry)
    
    def test_active_inference_test_generation(self):
        # Run the test in event loop
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self._async_test_inference_generation())
        
    async def _async_test_inference_generation(self):
        """Test the generation of active inference tests."""
        # Generate a test with uncertainty focus
        test = await self.active_inference.generate_test_interaction(
            'test_coupling_001',
            uncertainty_focus=True
        )
        
        # Verify test structure
        self.assertIsNotNone(test)
        self.assertIsInstance(test, ActiveInferenceTest)
        self.assertEqual(test.coupling_id, 'test_coupling_001')
        self.assertIsNotNone(test.test_parameters)
        
        # Verify test parameters
        params = test.test_parameters
        self.assertIn('interaction_type', params)
        self.assertEqual(params['interaction_type'], 'active_inference_test')
        self.assertIn('uncertainty_area', params)
        self.assertIn('test_content', params)
        self.assertIn('prediction', params)
        
        # Check that test info is recorded in history
        self.assertIn('test_coupling_001', self.active_inference.inference_history)
        self.assertEqual(len(self.active_inference.inference_history['test_coupling_001']), 1)
        
        # Generate a standard test
        test2 = await self.active_inference.generate_test_interaction(
            'test_coupling_001',
            uncertainty_focus=False
        )
        
        # Verify standard test
        self.assertIsNotNone(test2)
        self.assertEqual(test2.test_parameters['uncertainty_area'], 'general')
        
        # Verify history is updated
        self.assertEqual(len(self.active_inference.inference_history['test_coupling_001']), 2)
        
        return test
    
    def test_test_result_evaluation(self):
        # Run the test in event loop
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self._async_test_result_evaluation())
    
    async def _async_test_result_evaluation(self):
        """Test the evaluation of active inference test results."""
        # First generate a test
        test = await self.active_inference.generate_test_interaction(
            'test_coupling_001',
            uncertainty_focus=True
        )
        
        # Get the test type and area
        test_content = test.test_parameters.get('test_content', {})
        test_type = test_content.get('test_type', 'unknown')
        uncertainty_area = test.test_parameters.get('uncertainty_area', 'general')
        
        # Create a mock result based on test type
        if test_type == 'timing_test':
            actual_result = {
                'response_time': 1.2,  # Good response time
                'timestamp': time.time(),
                'status': 'success'
            }
        elif test_type == 'pattern_test':
            expected_next = test_content.get('expected_next')
            actual_result = {
                'next_value': expected_next,  # Correct pattern continuation
                'pattern_identified': True,
                'status': 'success'
            }
        elif test_type == 'reliability_test':
            verification_code = test_content.get('verification_code')
            actual_result = {
                'verification_code': verification_code,  # Perfect echo
                'status': 'success'
            }
        elif test_type == 'structure_test':
            # Create result with required fields
            actual_result = {
                'id': 'response_123',
                'timestamp': time.time(),
                'value': 42,
                'metadata': {'source': 'test'},
                'status': 'success'
            }
        elif test_type == 'context_test':
            context = test_content.get('context')
            expected_behavior = test_content.get('expected_behavior', {})
            actual_result = {
                'context_acknowledged': context,
                'behavior': expected_behavior.copy(),  # Perfect behavior match
                'status': 'success'
            }
        else:  # general test
            echo_data = test_content.get('echo_data')
            actual_result = {
                'echo': echo_data,
                'response_time': 1.5,
                'status': 'success'
            }
            
        # Evaluate the test result
        evaluation = await self.active_inference.evaluate_test_result(test.id, actual_result)
        
        # Verify evaluation structure
        self.assertTrue(evaluation.get('success', False))
        self.assertEqual(evaluation.get('test_id'), test.id)
        self.assertIn('information_gain', evaluation)
        self.assertIn('target_area', evaluation)
        self.assertIn('coupling_updates', evaluation)
        
        # Verify high information gain for perfect results
        self.assertGreater(evaluation['information_gain'], 0.7)
        
        # Check that the uncertainty profile was updated
        self.assertIn('test_coupling_001', self.active_inference.uncertainty_profiles)
        profile = self.active_inference.uncertainty_profiles['test_coupling_001']
        self.assertIn(uncertainty_area, profile)
        
        # Verify that coupling was updated
        self.mock_registry.update_coupling.assert_called()
        
        # Test with imperfect results
        if test_type == 'timing_test':
            bad_result = {
                'response_time': 10.0,  # Much slower than expected
                'status': 'delayed'
            }
        elif test_type == 'pattern_test':
            bad_result = {
                'next_value': 999,  # Wrong pattern continuation
                'status': 'error'
            }
        elif test_type == 'reliability_test':
            bad_result = {
                'verification_code': 'wrong_code',  # Incorrect echo
                'status': 'error'
            }
        else:
            bad_result = {
                'status': 'error',
                'message': 'Test failed'
            }
        
        # Evaluate with bad result
        bad_evaluation = await self.active_inference.evaluate_test_result(test.id, bad_result)
        
        # Verify lower information gain for bad results
        self.assertLess(bad_evaluation['information_gain'], evaluation['information_gain'])
        
        return evaluation
    
    def test_uncertainty_profile_creation(self):
        # Run the test in event loop
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self._async_test_uncertainty_profile())
    
    async def _async_test_uncertainty_profile(self):
        """Test the creation and updating of uncertainty profiles."""
        # Create initial uncertainty profile
        await self.active_inference._create_uncertainty_profile('test_coupling_001', self.mock_coupling)
        
        # Verify profile exists
        self.assertIn('test_coupling_001', self.active_inference.uncertainty_profiles)
        
        # Verify profile structure
        profile = self.active_inference.uncertainty_profiles['test_coupling_001']
        self.assertIn('response_timing', profile)
        self.assertIn('interaction_pattern', profile)
        self.assertIn('reliability', profile)
        self.assertIn('content_structure', profile)
        self.assertIn('contextual_behavior', profile)
        
        # Verify initial values are reasonable
        for area in ['response_timing', 'interaction_pattern', 'reliability', 'content_structure', 'contextual_behavior']:
            self.assertGreaterEqual(profile[area], 0.1)
            self.assertLessEqual(profile[area], 1.0)
        
        # Test analyzing uncertainty
        uncertainty_areas = await self.active_inference._analyze_coupling_uncertainty(self.mock_coupling)
        
        # Verify analysis returns a sorted list of areas
        self.assertIsInstance(uncertainty_areas, list)
        self.assertEqual(len(uncertainty_areas), 5)  # All 5 areas should be included
        
        # Verify that uncertainty updates after tests
        original_uncertainty = profile['response_timing']
        
        # Simulate a test that reduced uncertainty
        await self.active_inference._update_uncertainty_profile(
            'test_coupling_001',
            'response_timing',
            0.9,  # High information gain
            {'response_time': 1.5}
        )
        
        # Verify uncertainty decreased
        self.assertLess(profile['response_timing'], original_uncertainty)
        
        # Simulate a test that increased uncertainty
        original_uncertainty = profile['reliability']
        
        await self.active_inference._update_uncertainty_profile(
            'test_coupling_001',
            'reliability',
            0.2,  # Low information gain
            {'verification_failed': True}
        )
        
        # Verify uncertainty increased or stayed high
        self.assertGreaterEqual(profile['reliability'], original_uncertainty)
        
        return profile

if __name__ == "__main__":
    unittest.main()
