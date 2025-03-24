
import asyncio
import time
import logging
import json
import uuid
from asf.layer4_environmental_coupling.components.active_inference_controller import ActiveInferenceController
from asf.layer4_environmental_coupling.components.coupling_registry import SparseCouplingRegistry
from asf.layer4_environmental_coupling.models import EnvironmentalCoupling
from asf.layer4_environmental_coupling.enums import CouplingType, CouplingStrength, CouplingState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ActiveInferenceDemo")

class MockKnowledgeSubstrate:
    """Mock knowledge substrate for demo purposes."""
    
    async def get_entity(self, entity_id):
        """Return a mock entity."""
        return {
            'id': entity_id,
            'type': 'service',
            'properties': {
                'name': f'Entity {entity_id}',
                'description': 'Mock entity for demo'
            }
        }

class MockEnvironmentalEntity:
    """Mock environmental entity that responds to tests."""
    
    def __init__(self, entity_id, behavior_profile=None):
        self.entity_id = entity_id
        self.behavior_profile = behavior_profile or {}
        self.interaction_history = []
        
    async def process_test(self, test_parameters):
        """Process an active inference test and generate a response."""
        test_content = test_parameters.get('test_content', {})
        test_type = test_content.get('test_type', 'unknown')
        
        # Record interaction
        self.interaction_history.append({
            'timestamp': time.time(),
            'test_type': test_type,
            'test_id': test_parameters.get('test_id')
        })
        
        # Generate response based on test type
        if test_type == 'timing_test':
            # Simulate response time based on behavior profile
            response_time = self.behavior_profile.get('response_time', 1.0)
            # Add some randomness
            actual_time = response_time * (0.8 + 0.4 * random.random())
            # Simulate processing time
            await asyncio.sleep(actual_time)
            
            return {
                'response_time': actual_time,
                'timestamp': time.time(),
                'entity_id': self.entity_id,
                'status': 'success'
            }
            
        elif test_type == 'pattern_test':
            # Check if we recognize patterns
            pattern_recognition = self.behavior_profile.get('pattern_recognition', True)
            
            if pattern_recognition:
                # Get the sequence and expected next item
                sequence = test_content.get('sequence', [])
                expected_next = test_content.get('expected_next')
                
                # If our pattern recognition is fuzzy, sometimes get it wrong
                pattern_accuracy = self.behavior_profile.get('pattern_accuracy', 0.9)
                
                if random.random() <= pattern_accuracy:
                    next_value = expected_next
                else:
                    # Generate incorrect answer
                    next_value = expected_next + random.randint(1, 10)
                
                return {
                    'next_value': next_value,
                    'pattern_identified': True,
                    'entity_id': self.entity_id,
                    'status': 'success'
                }
            else:
                # Can't recognize patterns
                return {
                    'error': 'pattern_recognition_not_supported',
                    'entity_id': self.entity_id,
                    'status': 'error'
                }
                
        elif test_type == 'reliability_test':
            # Check reliability behavior
            reliability = self.behavior_profile.get('reliability', 0.95)
            
            if random.random() <= reliability:
                # Return the verification code correctly
                verification_code = test_content.get('verification_code')
                return {
                    'verification_code': verification_code,
                    'timestamp': time.time(),
                    'entity_id': self.entity_id,
                    'status': 'success'
                }
            else:
                # Return error or wrong code
                return {
                    'verification_code': 'invalid_code',
                    'error': 'verification_failed',
                    'entity_id': self.entity_id,
                    'status': 'error'
                }
                
        elif test_type == 'structure_test':
            # Check if we support structured data
            structure_support = self.behavior_profile.get('structure_support', True)
            
            if structure_support:
                # Get required fields and create response
                required_fields = test_content.get('required_fields', [])
                response = {
                    'entity_id': self.entity_id,
                    'timestamp': time.time(),
                    'status': 'success'
                }
                
                # Add required fields
                field_accuracy = self.behavior_profile.get('field_accuracy', 0.9)
                
                for field in required_fields:
                    if random.random() <= field_accuracy:
                        # Add field with correct type
                        if field == 'id':
                            response[field] = f"response_{uuid.uuid4().hex[:8]}"
                        elif field == 'timestamp':
                            response[field] = time.time()
                        elif field == 'value':
                            response[field] = random.randint(1, 100)
                        elif field == 'metadata':
                            response[field] = {'source': self.entity_id}
                        else:
                            response[field] = f"value_for_{field}"
                
                return response
            else:
                # Don't support structure tests
                return {
                    'error': 'structure_not_supported',
                    'entity_id': self.entity_id,
                    'status': 'error'
                }
                
        elif test_type == 'context_test':
            # Check if we handle contexts
            context_awareness = self.behavior_profile.get('context_awareness', True)
            
            if context_awareness:
                # Get context information
                context = test_content.get('context')
                expected_behavior = test_content.get('expected_behavior', {})
                
                # Determine our actual behavior based on profile
                context_accuracy = self.behavior_profile.get('context_accuracy', 0.9)
                
                if random.random() <= context_accuracy:
                    # Correct context recognition
                    actual_behavior = expected_behavior.copy()
                else:
                    # Incorrect context behavior
                    actual_behavior = {k: 'incorrect' for k in expected_behavior}
                
                return {
                    'context_acknowledged': context if random.random() <= context_accuracy else 'unknown',
                    'behavior': actual_behavior,
                    'entity_id': self.entity_id,
                    'status': 'success'
                }
            else:
                # Don't support context
                return {
                    'error': 'context_not_supported',
                    'entity_id': self.entity_id,
                    'status': 'error'
                }
                
        else:  # General test
            # Basic response
            echo_data = test_content.get('echo_data')
            return {
                'echo': echo_data,
                'response_time': self.behavior_profile.get('response_time', 1.0),
                'entity_id': self.entity_id,
                'status': 'success'
            }

async def run_active_inference_demo():
    """Run a demonstration of active inference capabilities."""
    logger.info("Starting Active Inference Demonstration")
    
    # Create components
    knowledge_substrate = MockKnowledgeSubstrate()
    coupling_registry = SparseCouplingRegistry()
    active_inference = ActiveInferenceController(knowledge_substrate)
    
    # Initialize components
    await coupling_registry.initialize()
    active_inference.set_coupling_registry(coupling_registry)
    
    # Create mock entities with different behavior profiles
    entities = {
        'reliable_entity': MockEnvironmentalEntity('reliable_entity', {
            'response_time': 0.8,
            'reliability': 0.95,
            'pattern_recognition': True,
            'pattern_accuracy': 0.9,
            'structure_support': True,
            'field_accuracy': 0.95,
            'context_awareness': True,
            'context_accuracy': 0.9
        }),
        'unreliable_entity': MockEnvironmentalEntity('unreliable_entity', {
            'response_time': 2.5,
            'reliability': 0.6,
            'pattern_recognition': True,
            'pattern_accuracy': 0.7,
            'structure_support': True,
            'field_accuracy': 0.7,
            'context_awareness': False
        }),
        'specialized_entity': MockEnvironmentalEntity('specialized_entity', {
            'response_time': 0.5,
            'reliability': 0.99,
            'pattern_recognition': False,
            'structure_support': False,
            'context_awareness': True,
            'context_accuracy': 0.99
        })
    }
    
    # Create couplings between internal entity and environmental entities
    couplings = {}
    
    for entity_id, entity in entities.items():
        coupling = EnvironmentalCoupling(
            id=f"coupling_{entity_id}",
            internal_entity_id='system_entity',
            environmental_entity_id=entity_id,
            coupling_type=CouplingType.INFORMATIONAL,
            coupling_strength=0.7,
            coupling_state=CouplingState.ACTIVE,
            bayesian_confidence=0.5,
            interaction_count=1
        )
        
        # Add to registry
        await coupling_registry.add_coupling(coupling)
        couplings[entity_id] = coupling
    
    logger.info(f"Created {len(couplings)} couplings with environmental entities")
    
    # Run active inference learning cycle for each coupling
    for entity_id, coupling in couplings.items():
        logger.info(f"\n{'=' * 40}\nRunning active inference cycle for {entity_id}\n{'=' * 40}")
        entity = entities[entity_id]
        
        # Run multiple tests to learn about entity behavior
        for i in range(5):
            logger.info(f"\nTest {i+1} for {entity_id}")
            
            # Generate test - focus on uncertainty
            test = await active_inference.generate_test_interaction(
                coupling.id,
                uncertainty_focus=True
            )
            
            # Log test details
            test_content = test.test_parameters.get('test_content', {})
            test_type = test_content.get('test_type', 'unknown')
            uncertainty_area = test.test_parameters.get('uncertainty_area', 'general')
            
            logger.info(f"Generated {test_type} test targeting {uncertainty_area}")
            
            # Process test through mock entity
            actual_result = await entity.process_test(test.test_parameters)
            
            # Evaluate test result
            evaluation = await active_inference.evaluate_test_result(test.id, actual_result)
            
            # Log results
            logger.info(f"Test information gain: {evaluation['information_gain']:.3f}")
            
            # Get updated uncertainty profile
            if coupling.id in active_inference.uncertainty_profiles:
                profile = active_inference.uncertainty_profiles[coupling.id]
                profile_summary = {k: round(v, 3) for k, v in profile.items() 
                                 if k not in ['last_updated', 'update_count']}
                
                logger.info(f"Updated uncertainty profile: {json.dumps(profile_summary, indent=2)}")
            
            # Brief pause between tests
            await asyncio.sleep(0.5)
        
        # Analyze what we learned about the entity
        if coupling.id in active_inference.uncertainty_profiles:
            profile = active_inference.uncertainty_profiles[coupling.id]
            
            # Find lowest and highest uncertainty areas
            areas = [
                'response_timing',
                'interaction_pattern',
                'reliability',
                'content_structure',
                'contextual_behavior'
            ]
            
            sorted_areas = sorted(areas, key=lambda area: profile[area])
            
            logger.info(f"\nAfter testing, learned that {entity_id}:")
            logger.info(f"- Most predictable in: {sorted_areas[0]} (uncertainty: {profile[sorted_areas[0]]:.3f})")
            logger.info(f"- Least predictable in: {sorted_areas[-1]} (uncertainty: {profile[sorted_areas[-1]]:.3f})")
            
            # Compare with actual behavior profile
            actual_behavior = entities[entity_id].behavior_profile
            logger.info("\nComparison with actual behavior profile:")
            
            if 'response_time' in actual_behavior:
                logger.info(f"- Actual response time: {actual_behavior['response_time']:.2f}s")
                
            if 'reliability' in actual_behavior:
                logger.info(f"- Actual reliability: {actual_behavior['reliability']:.2f}")
                
            if 'pattern_recognition' in actual_behavior:
                pattern_support = "Supported" if actual_behavior['pattern_recognition'] else "Not supported"
                logger.info(f"- Pattern recognition: {pattern_support}")
                
            if 'context_awareness' in actual_behavior:
                context_support = "Supported" if actual_behavior['context_awareness'] else "Not supported"
                logger.info(f"- Context awareness: {context_support}")
    
    logger.info("\nActive Inference Demonstration Completed")

if __name__ == "__main__":
    # Import random here to avoid scope issues
    import random
    
    # Run the demo
    asyncio.run(run_active_inference_demo())
