import asyncio
import time
import uuid
import logging
import numpy as np
import random
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

from asf.layer4_environmental_coupling.models import ActiveInferenceTest, EnvironmentalCoupling
from asf.layer4_environmental_coupling.enums import CouplingType, CouplingState

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
        Evaluate the result of an active inference test.
        Updates uncertainty profiles based on test results.
        self.uncertainty_profiles[coupling_id] = {
            'response_timing': 1.0,  # Higher value = higher uncertainty
            'interaction_pattern': 1.0,
            'reliability': 1.0,
            'content_structure': 1.0,
            'contextual_behavior': 1.0,
            'last_updated': time.time(),
            'update_count': 0
        }
        
        if hasattr(coupling, 'interaction_count') and coupling.interaction_count > 0:
            if coupling.interaction_count > 5:
                self.uncertainty_profiles[coupling_id]['interaction_pattern'] *= 0.8
            
            if hasattr(coupling, 'bayesian_confidence') and coupling.bayesian_confidence > 0.6:
                self.uncertainty_profiles[coupling_id]['reliability'] *= 0.8
            
            if hasattr(coupling, 'prediction_precision') and coupling.prediction_precision > 1.0:
                self.uncertainty_profiles[coupling_id]['response_timing'] *= 0.8
    
    async def _analyze_coupling_uncertainty(self, coupling):
        test_interaction = {
            'interaction_type': 'active_inference_test',
            'target_entity_id': coupling.environmental_entity_id,
            'uncertainty_area': uncertainty_area,
            'source_entity_id': coupling.internal_entity_id,
            'timestamp': time.time(),
            'test_id': str(uuid.uuid4())
        }
        
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
        
        test_interaction['prediction_id'] = str(uuid.uuid4())
        
        return test_interaction
    
    async def _generate_standard_test(self, coupling, entity):
        Calculate information gain from test results.
        Higher values indicate more valuable information was acquired.
        if coupling_id not in self.uncertainty_profiles:
            return False
        
        profile = self.uncertainty_profiles[coupling_id]
        
        current_uncertainty = profile.get(uncertainty_area, 1.0)
        
        weight = min(0.8, current_uncertainty)  # Weight between 0 and 0.8
        
        uncertainty_reduction = 1.0 - information_gain
        
        new_uncertainty = (current_uncertainty * (1 - weight)) + (uncertainty_reduction * weight)
        
        new_uncertainty = max(0.1, min(1.0, new_uncertainty))
        
        profile[uncertainty_area] = new_uncertainty
        profile['last_updated'] = time.time()
        profile['update_count'] = profile.get('update_count', 0) + 1
        
        return True
    
    async def _update_coupling_from_test(self, coupling, uncertainty_area, information_gain, actual_result):
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
