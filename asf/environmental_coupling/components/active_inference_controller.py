import asyncio
import time
import uuid
import logging
import numpy as np
import random
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

from asf.environmental_coupling.models import ActiveInferenceTest, EnvironmentalCoupling
from asf.environmental_coupling.enums import CouplingType, CouplingState

class ActiveInferenceController:
    """
    Implements Seth's active inference principle to minimize prediction errors.
    Proactively tests environmental couplings to reduce uncertainty.
    """
    def __init__(self, knowledge_substrate):
        self.knowledge_substrate = knowledge_substrate
        self.coupling_registry = None  # Will be set during initialization
        self.inference_history = defaultdict(list)  # Maps coupling_id to test history
        self.test_results = {}  # Maps test_id to results
        self.uncertainty_profiles = {}  # Maps coupling_id to uncertainty profile
        self.logger = logging.getLogger("ASF.Layer4.ActiveInferenceController")
    
    def set_coupling_registry(self, registry):
        """Set reference to coupling registry."""
        self.coupling_registry = registry
    
    async def generate_test_interaction(self, coupling_id, uncertainty_focus=True):
        """
        Generate an interaction that tests a coupling relationship.
        If uncertainty_focus is True, focuses on areas with highest uncertainty.
        """
        # Get coupling data
        coupling = await self.coupling_registry.get_coupling(coupling_id)
        if not coupling:
            self.logger.warning(f"Cannot generate test for unknown coupling: {coupling_id}")
            return None
            
        # Get entity data
        entity = await self.knowledge_substrate.get_entity(coupling.internal_entity_id)
        if not entity:
            self.logger.warning(f"Cannot generate test for unknown entity: {coupling.internal_entity_id}")
            return None
        
        # Create uncertainty profile if it doesn't exist
        if coupling_id not in self.uncertainty_profiles:
            await self._create_uncertainty_profile(coupling_id, coupling)
        
        # Analyze areas of highest uncertainty
        uncertainty_areas = await self._analyze_coupling_uncertainty(coupling)
        
        # Generate test based on uncertainty focus preference
        if uncertainty_focus and uncertainty_areas:
            # Focus on the area with highest uncertainty
            target_area = uncertainty_areas[0]
            test_interaction = await self._generate_uncertainty_reducing_test(coupling, entity, target_area)
        else:
            # Generate a more general test
            test_interaction = await self._generate_standard_test(coupling, entity)
        
        # Create test record
        test = ActiveInferenceTest(
            id=str(uuid.uuid4()),
            coupling_id=coupling_id,
            prediction_id=test_interaction.get('prediction_id'),
            test_parameters=test_interaction
        )
        
        # Record in history
        self.inference_history[coupling_id].append({
            'test_id': test.id,
            'timestamp': time.time(),
            'test_type': 'uncertainty_focused' if uncertainty_focus else 'standard',
            'target_area': uncertainty_areas[0] if uncertainty_focus and uncertainty_areas else 'general'
        })
        
        # Log the test creation
        self.logger.info(f"Generated active inference test {test.id} for coupling {coupling_id}, " +
                         f"targeting {'uncertainty in ' + uncertainty_areas[0] if uncertainty_focus and uncertainty_areas else 'general properties'}")
        
        return test
    
    async def evaluate_test_result(self, test_id, actual_result):
        """
        Evaluate the result of an active inference test.
        Updates uncertainty profiles based on test results.
        """
        # Find the test
        test = None
        coupling_id = None
        
        # Search for test in inference history
        for cid, history in self.inference_history.items():
            for entry in history:
                if entry.get('test_id') == test_id:
                    coupling_id = cid
                    test_entry = entry
                    break
            if coupling_id:
                break
        
        if not coupling_id:
            self.logger.warning(f"Test {test_id} not found in inference history")
            return {'success': False, 'error': 'Test not found'}
        
        # Get the coupling
        coupling = await self.coupling_registry.get_coupling(coupling_id)
        if not coupling:
            self.logger.warning(f"Coupling {coupling_id} not found for test {test_id}")
            return {'success': False, 'error': 'Coupling not found'}
        
        # Calculate information gain
        test_parameters = test_entry.get('test_parameters', {})
        target_area = test_entry.get('target_area', 'general')
        
        info_gain = self._calculate_information_gain(test_parameters, actual_result, target_area)
        
        # Update uncertainty profile
        await self._update_uncertainty_profile(coupling_id, target_area, info_gain, actual_result)
        
        # Store test result
        self.test_results[test_id] = {
            'coupling_id': coupling_id,
            'actual_result': actual_result,
            'information_gain': info_gain,
            'target_area': target_area,
            'evaluation_time': time.time()
        }
        
        # Update coupling based on test results
        update_result = await self._update_coupling_from_test(coupling, target_area, info_gain, actual_result)
        
        self.logger.info(f"Evaluated test {test_id} with information gain {info_gain:.3f}")
        
        return {
            'success': True,
            'test_id': test_id,
            'information_gain': info_gain,
            'target_area': target_area,
            'coupling_updates': update_result
        }
    
    async def _create_uncertainty_profile(self, coupling_id, coupling):
        """Create initial uncertainty profile for a coupling."""
        # Initialize uncertainty values for different areas
        self.uncertainty_profiles[coupling_id] = {
            'response_timing': 1.0,  # Higher value = higher uncertainty
            'interaction_pattern': 1.0,
            'reliability': 1.0,
            'content_structure': 1.0,
            'contextual_behavior': 1.0,
            'last_updated': time.time(),
            'update_count': 0
        }
        
        # If coupling has some history, refine initial estimates
        if hasattr(coupling, 'interaction_count') and coupling.interaction_count > 0:
            if coupling.interaction_count > 5:
                self.uncertainty_profiles[coupling_id]['interaction_pattern'] *= 0.8
            
            if hasattr(coupling, 'bayesian_confidence') and coupling.bayesian_confidence > 0.6:
                self.uncertainty_profiles[coupling_id]['reliability'] *= 0.8
            
            if hasattr(coupling, 'prediction_precision') and coupling.prediction_precision > 1.0:
                self.uncertainty_profiles[coupling_id]['response_timing'] *= 0.8
    
    async def _analyze_coupling_uncertainty(self, coupling):
        """Identify areas of highest uncertainty for a coupling."""
        coupling_id = coupling.id
        
        # If we don't have an uncertainty profile, create one
        if coupling_id not in self.uncertainty_profiles:
            await self._create_uncertainty_profile(coupling_id, coupling)
        
        profile = self.uncertainty_profiles[coupling_id]
        
        # Sort areas by uncertainty (descending)
        areas = [
            'response_timing',
            'interaction_pattern',
            'reliability',
            'content_structure',
            'contextual_behavior'
        ]
        
        sorted_areas = sorted(areas, key=lambda area: profile[area], reverse=True)
        
        # Additional dynamic factors that may increase uncertainty
        
        # Low interaction count increases pattern uncertainty
        if hasattr(coupling, 'interaction_count') and coupling.interaction_count < 5:
            if 'interaction_pattern' not in sorted_areas[:2]:
                # Move to front of list
                sorted_areas.remove('interaction_pattern')
                sorted_areas.insert(0, 'interaction_pattern')
        
        # Low bayesian confidence increases reliability uncertainty
        if hasattr(coupling, 'bayesian_confidence') and coupling.bayesian_confidence < 0.5:
            if 'reliability' not in sorted_areas[:2]:
                # Move to front or second position
                sorted_areas.remove('reliability')
                sorted_areas.insert(min(1, len(sorted_areas)), 'reliability')
        
        # Low prediction precision increases timing uncertainty
        if hasattr(coupling, 'prediction_precision') and coupling.prediction_precision < 0.8:
            if 'response_timing' not in sorted_areas[:2]:
                # Move to front or second position
                sorted_areas.remove('response_timing')
                sorted_areas.insert(min(1, len(sorted_areas)), 'response_timing')
        
        return sorted_areas
    
    async def _generate_uncertainty_reducing_test(self, coupling, entity, uncertainty_area):
        """Generate a test focused on reducing a specific uncertainty area."""
        # Start with basic test structure
        test_interaction = {
            'interaction_type': 'active_inference_test',
            'target_entity_id': coupling.environmental_entity_id,
            'uncertainty_area': uncertainty_area,
            'source_entity_id': coupling.internal_entity_id,
            'timestamp': time.time(),
            'test_id': str(uuid.uuid4())
        }
        
        # Customize test based on uncertainty area
        if uncertainty_area == 'response_timing':
            test_interaction.update({
                'test_content': {
                    'test_type': 'timing_test',
                    'request_timestamp': time.time(),
                    'expected_response_window': [0.5, 3.0],  # seconds
                    'require_timestamp': True
                },
                'prediction': {
                    'expected_response_time': 1.5,  # seconds
                    'timestamp_precision': 0.1  # seconds
                }
            })
            
        elif uncertainty_area == 'interaction_pattern':
            # Define a sequence pattern test
            test_interaction.update({
                'test_content': {
                    'test_type': 'pattern_test',
                    'sequence': [1, 2, 3],
                    'expected_next': 4,
                    'pattern_description': 'sequential_increment'
                },
                'prediction': {
                    'expected_completion': True,
                    'alternative_patterns': ['fibonacci', 'even_numbers', 'powers_of_two']
                }
            })
            
        elif uncertainty_area == 'reliability':
            # Create an echo test with verification
            test_interaction.update({
                'test_content': {
                    'test_type': 'reliability_test',
                    'verification_code': str(uuid.uuid4())[:8],
                    'verification_instruction': 'return_exact_code',
                    'verification_timestamp': time.time()
                },
                'prediction': {
                    'expected_verification': True,
                    'expected_code_match': True
                }
            })
            
        elif uncertainty_area == 'content_structure':
            # Test ability to follow a structured data format
            test_interaction.update({
                'test_content': {
                    'test_type': 'structure_test',
                    'required_fields': ['id', 'timestamp', 'value', 'metadata'],
                    'expected_types': {
                        'id': 'string',
                        'timestamp': 'number',
                        'value': 'number',
                        'metadata': 'object'
                    }
                },
                'prediction': {
                    'expected_conformance': 0.8,  # 80% of fields correct
                    'critical_fields': ['id', 'timestamp']
                }
            })
            
        elif uncertainty_area == 'contextual_behavior':
            # Test how entity behavior changes based on context
            contexts = ['normal', 'urgent', 'error', 'maintenance']
            selected_context = random.choice(contexts)
            
            test_interaction.update({
                'test_content': {
                    'test_type': 'context_test',
                    'context': selected_context,
                    'context_indicator': f"This is a {selected_context} situation",
                    'expected_behavior': self._get_expected_behavior(selected_context)
                },
                'prediction': {
                    'context_recognition': True,
                    'behavior_adaptation': True
                }
            })
        
        else:  # 'general' or unknown area
            # General test with multiple aspects
            test_interaction.update({
                'test_content': {
                    'test_type': 'general_test',
                    'verification_code': str(uuid.uuid4())[:6],
                    'timestamp': time.time(),
                    'structured_request': {
                        'operation': 'echo',
                        'payload': {
                            'code': test_interaction['test_id'][-8:],
                            'timestamp': time.time()
                        }
                    }
                },
                'prediction': {
                    'response_expected': True,
                    'response_time_range': [0.5, 5.0]
                }
            })
        
        # Add tracking ID for the prediction we're making
        test_interaction['prediction_id'] = str(uuid.uuid4())
        
        return test_interaction
    
    async def _generate_standard_test(self, coupling, entity):
        """Generate a standard test interaction without specific uncertainty focus."""
        # Create a general test that checks basic functionality
        test_interaction = {
            'interaction_type': 'active_inference_test',
            'target_entity_id': coupling.environmental_entity_id,
            'uncertainty_area': 'general',
            'source_entity_id': coupling.internal_entity_id,
            'timestamp': time.time(),
            'test_id': str(uuid.uuid4()),
            'test_content': {
                'test_type': 'general_test',
                'message': 'Standard active inference test',
                'request_id': str(uuid.uuid4()),
                'echo_data': f"test_{int(time.time())}"
            },
            'prediction': {
                'response_expected': True,
                'contains_echo': True,
                'response_time_under': 5.0  # seconds
            }
        }
        
        # Add tracking ID for the prediction we're making
        test_interaction['prediction_id'] = str(uuid.uuid4())
        
        return test_interaction
    
    def _calculate_information_gain(self, test_parameters, actual_result, uncertainty_area):
        """
        Calculate information gain from test results.
        Higher values indicate more valuable information was acquired.
        """
        # Extract test details and prediction
        test_content = test_parameters.get('test_content', {})
        prediction = test_parameters.get('prediction', {})
        test_type = test_content.get('test_type', 'unknown')
        
        # Base information gain starts neutral
        information_gain = 0.5
        
        # Calculate based on test type
        if test_type == 'timing_test':
            # Extract expectations and actual values
            expected_time = prediction.get('expected_response_time', 1.0)
            actual_time = actual_result.get('response_time')
            
            if actual_time is not None:
                # Calculate time prediction error
                time_error = abs(actual_time - expected_time)
                expected_window = test_content.get('expected_response_window', [0, 10])
                max_window = expected_window[1] - expected_window[0]
                
                # Normalize error to 0-1 scale
                normalized_error = min(1.0, time_error / max_window)
                
                # Information gain is higher when error is lower
                information_gain = 1.0 - normalized_error
            else:
                # No timing data means little information gained
                information_gain = 0.1
        
        elif test_type == 'pattern_test':
            # Extract expectations and actual values
            expected_next = test_content.get('expected_next')
            actual_next = actual_result.get('next_value')
            
            if actual_next is not None:
                # Binary result - either correct or not
                if expected_next == actual_next:
                    information_gain = 1.0
                else:
                    # Still gain some information even if wrong
                    information_gain = 0.3
                    
                # Additional information if alternative pattern identified
                if actual_result.get('pattern_identified'):
                    information_gain += 0.2
            else:
                information_gain = 0.1
        
        elif test_type == 'reliability_test':
            # Extract verification expectations
            verification_code = test_content.get('verification_code')
            actual_code = actual_result.get('verification_code')
            
            # Check if verification happened at all
            if actual_code is not None:
                # How accurate was the verification
                if verification_code == actual_code:
                    information_gain = 1.0
                else:
                    # Partial match also provides information
                    similarity = self._calculate_string_similarity(verification_code, actual_code)
                    information_gain = max(0.1, similarity)
            else:
                information_gain = 0.0
        
        elif test_type == 'structure_test':
            # Check conformance to expected structure
            required_fields = test_content.get('required_fields', [])
            expected_types = test_content.get('expected_types', {})
            
            if isinstance(actual_result, dict):
                # Count matching fields
                field_matches = 0
                type_matches = 0
                
                for field in required_fields:
                    if field in actual_result:
                        field_matches += 1
                        
                        # Also check type if expected
                        if field in expected_types:
                            expected_type = expected_types[field]
                            actual_type = self._get_type_name(actual_result[field])
                            
                            if expected_type == actual_type:
                                type_matches += 1
                
                # Calculate information gain based on matches
                if required_fields:
                    field_score = field_matches / len(required_fields)
                    type_score = type_matches / len(expected_types) if expected_types else 1.0
                    
                    # Weight field presence more heavily than type matching
                    information_gain = (field_score * 0.6) + (type_score * 0.4)
                else:
                    information_gain = 0.5
            else:
                # Not even a dict structure
                information_gain = 0.1
        
        elif test_type == 'context_test':
            # Check if context was recognized and behavior adapted
            expected_behavior = test_content.get('expected_behavior', {})
            context = test_content.get('context')
            
            if isinstance(actual_result, dict):
                # Check for context indicators in response
                context_recognized = False
                behavior_appropriate = False
                
                # Look for context acknowledgment
                if actual_result.get('context_acknowledged') == context:
                    context_recognized = True
                
                # Check behavior against expectations
                if expected_behavior and actual_result.get('behavior'):
                    similarity = self._calculate_behavior_match(
                        expected_behavior, 
                        actual_result['behavior']
                    )
                    behavior_appropriate = similarity > 0.7
                
                # Calculate information gain
                if context_recognized and behavior_appropriate:
                    information_gain = 1.0
                elif context_recognized:
                    information_gain = 0.7
                elif behavior_appropriate:
                    information_gain = 0.6
                else:
                    information_gain = 0.3
            else:
                information_gain = 0.1
        
        else:  # general or unknown test type
            # For general tests, check basic response properties
            if isinstance(actual_result, dict) and actual_result:
                # Response exists and has content
                information_gain = 0.6
                
                # Check for echo data
                echo_data = test_content.get('echo_data')
                if echo_data and actual_result.get('echo') == echo_data:
                    information_gain += 0.2
                    
                # Check response timing if available
                if 'response_time' in actual_result:
                    information_gain += 0.1
            else:
                # Empty or invalid response
                information_gain = 0.2
        
        # Ensure result is normalized
        return max(0.0, min(1.0, information_gain))
    
    async def _update_uncertainty_profile(self, coupling_id, uncertainty_area, information_gain, actual_result):
        """Update the uncertainty profile based on test results."""
        if coupling_id not in self.uncertainty_profiles:
            # If profile doesn't exist, we can't update it
            return False
        
        profile = self.uncertainty_profiles[coupling_id]
        
        # Higher information gain means lower uncertainty
        current_uncertainty = profile.get(uncertainty_area, 1.0)
        
        # Calculate new uncertainty with exponential moving average
        # More weight on new observation if current uncertainty is high
        weight = min(0.8, current_uncertainty)  # Weight between 0 and 0.8
        
        # Convert information gain to uncertainty reduction
        # Higher information gain = lower uncertainty
        uncertainty_reduction = 1.0 - information_gain
        
        # Update uncertainty value
        new_uncertainty = (current_uncertainty * (1 - weight)) + (uncertainty_reduction * weight)
        
        # Ensure uncertainty stays within bounds
        new_uncertainty = max(0.1, min(1.0, new_uncertainty))
        
        # Update profile
        profile[uncertainty_area] = new_uncertainty
        profile['last_updated'] = time.time()
        profile['update_count'] = profile.get('update_count', 0) + 1
        
        return True
    
    async def _update_coupling_from_test(self, coupling, uncertainty_area, information_gain, actual_result):
        """Update coupling based on test results."""
        updates = []
        coupling_changed = False
        
        # Update coupling based on uncertainty area
        if uncertainty_area == 'response_timing':
            if hasattr(coupling, 'properties') and 'expected_response_time' not in coupling.properties:
                coupling.properties['expected_response_time'] = actual_result.get('response_time', 1.0)
                updates.append('added expected_response_time property')
                coupling_changed = True
            
            if hasattr(coupling, 'prediction_precision'):
                # Adjust prediction precision based on information gain
                precision_delta = (information_gain - 0.5) * 0.5  # Scale to smaller adjustment
                coupling.prediction_precision = max(0.1, coupling.prediction_precision + precision_delta)
                updates.append(f"adjusted prediction_precision to {coupling.prediction_precision:.3f}")
                coupling_changed = True
        
        elif uncertainty_area == 'reliability':
            if hasattr(coupling, 'bayesian_confidence'):
                # Adjust confidence based on test results
                confidence_delta = (information_gain - 0.5) * 0.1  # Small adjustment
                coupling.bayesian_confidence = max(0.1, min(0.95, coupling.bayesian_confidence + confidence_delta))
                updates.append(f"adjusted bayesian_confidence to {coupling.bayesian_confidence:.3f}")
                coupling_changed = True
        
        elif uncertainty_area == 'interaction_pattern':
            if hasattr(coupling, 'properties') and isinstance(coupling.properties, dict):
                # Record pattern information
                if 'interaction_patterns' not in coupling.properties:
                    coupling.properties['interaction_patterns'] = {}
                
                test_content = actual_result.get('test_content', {})
                if test_content.get('pattern_description'):
                    pattern_type = test_content['pattern_description']
                    coupling.properties['interaction_patterns'][pattern_type] = {
                        'verified': information_gain > 0.7,
                        'last_tested': time.time()
                    }
                    updates.append(f"updated interaction pattern: {pattern_type}")
                    coupling_changed = True
        
        # General updates applicable to all test types
        if hasattr(coupling, 'properties') and isinstance(coupling.properties, dict):
            # Record test history
            if 'active_inference_tests' not in coupling.properties:
                coupling.properties['active_inference_tests'] = []
            
            # Add test result summary
            coupling.properties['active_inference_tests'].append({
                'area': uncertainty_area,
                'information_gain': information_gain,
                'timestamp': time.time()
            })
            
            # Limit test history size
            if len(coupling.properties['active_inference_tests']) > 10:
                coupling.properties['active_inference_tests'] = coupling.properties['active_inference_tests'][-10:]
            
            updates.append("added test to history")
            coupling_changed = True
        
        # Update coupling counts
        if hasattr(coupling, 'interaction_count'):
            coupling.interaction_count += 1
            updates.append(f"incremented interaction_count to {coupling.interaction_count}")
            coupling_changed = True
        
        # Record last interaction time
        coupling.last_interaction = time.time()
        
        # If coupling changed and we have registry, update it
        if coupling_changed and self.coupling_registry:
            await self.coupling_registry.update_coupling(coupling)
        
        return {
            'coupling_id': coupling.id,
            'updates': updates,
            'coupling_changed': coupling_changed
        }
    
    def _get_expected_behavior(self, context):
        """Get expected behavior for a context test."""
        behaviors = {
            'normal': {
                'response_time': 'standard',
                'priority': 'normal',
                'detail_level': 'standard',
                'error_checking': 'standard'
            },
            'urgent': {
                'response_time': 'accelerated',
                'priority': 'high',
                'detail_level': 'minimal',
                'error_checking': 'minimal'
            },
            'error': {
                'response_time': 'standard',
                'priority': 'high',
                'detail_level': 'detailed',
                'error_checking': 'thorough'
            },
            'maintenance': {
                'response_time': 'relaxed',
                'priority': 'low',
                'detail_level': 'comprehensive',
                'error_checking': 'thorough'
            }
        }
        
        return behaviors.get(context, behaviors['normal'])
    
    def _calculate_string_similarity(self, str1, str2):
        """Calculate simple string similarity (0-1 scale)."""
        if not str1 or not str2:
            return 0.0
        
        # Length of longest common substring
        shorter = min(len(str1), len(str2))
        if shorter == 0:
            return 0.0
        
        # Count matching characters
        matches = sum(c1 == c2 for c1, c2 in zip(str1, str2))
        return matches / shorter
    
    def _calculate_behavior_match(self, expected, actual):
        """Calculate how well actual behavior matches expected behavior."""
        if not isinstance(expected, dict) or not isinstance(actual, dict):
            return 0.0
        
        total_attributes = len(expected)
        if total_attributes == 0:
            return 0.0
        
        matching_attributes = 0
        
        for key, exp_value in expected.items():
            if key in actual and actual[key] == exp_value:
                matching_attributes += 1
        
        return matching_attributes / total_attributes
    
    def _get_type_name(self, value):
        """Get type name of a value."""
        if isinstance(value, str):
            return 'string'
        elif isinstance(value, (int, float)):
            return 'number'
        elif isinstance(value, bool):
            return 'boolean'
        elif isinstance(value, dict):
            return 'object'
        elif isinstance(value, list):
            return 'array'
        elif value is None:
            return 'null'
        else:
            return 'unknown'
