
import asyncio
import time
import logging
import random
import json
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("IntegratedPredictiveDemo")

# Import necessary components
from asf.environmental_coupling.layer import EnvironmentalCouplingLayer
from asf.environmental_coupling.predictive_orchestrator import PredictiveProcessingOrchestrator
from asf.environmental_coupling.models import EnvironmentalCoupling
from asf.environmental_coupling.enums import CouplingType, CouplingStrength, CouplingState

class MockKnowledgeSubstrate:
    """Mock knowledge substrate for the demo."""
    
    async def get_entity(self, entity_id):
        """Get entity information."""
        return {
            'id': entity_id,
            'type': 'service',
            'properties': {
                'name': f'Entity {entity_id}',
                'created_at': time.time() - 3600  # Created an hour ago
            }
        }

class MockEnvironmentalEntity:
    """Mock environmental entity that can be interacted with."""
    
    def __init__(self, entity_id, behavior_profile=None):
        self.entity_id = entity_id
        self.behavior_profile = behavior_profile or {}
        self.state = {
            'last_update': time.time(),
            'interaction_count': 0,
            'data': {}
        }
        self.interaction_history = []
        logger.info(f"Created environmental entity {entity_id}")
    
    async def process_interaction(self, interaction_data, source_id=None):
        """Process an interaction from an internal entity."""
        self.interaction_history.append({
            'timestamp': time.time(),
            'source_id': source_id,
            'data': interaction_data
        })
        
        self.state['interaction_count'] += 1
        self.state['last_update'] = time.time()
        
        # Generate response based on behavior profile
        response_time = self.behavior_profile.get('response_time', 1.0)
        response_variability = self.behavior_profile.get('response_variability', 0.2)
        actual_response_time = response_time * (1 + random.uniform(-response_variability, response_variability))
        
        # Simulate processing time
        await asyncio.sleep(actual_response_time)
        
        # Generate response
        response = {
            'timestamp': time.time(),
            'entity_id': self.entity_id,
            'response_time': actual_response_time,
            'interaction_type': interaction_data.get('interaction_type', 'response'),
            'content_type': 'structured',
            'data': {
                'status': 'success',
                'source_id': source_id,
                'message': f"Processed interaction from {source_id}"
            }
        }
        
        # Add some predictable patterns based on interaction type
        if interaction_data.get('interaction_type') == 'query':
            response['data']['query_result'] = {'value': random.randint(1, 100)}
        elif interaction_data.get('interaction_type') == 'update':
            new_value = interaction_data.get('value', 0)
            self.state['data']['value'] = new_value
            response['data']['update_result'] = {'previous': self.state.get('value', 0), 'new': new_value}
        
        # Sometimes introduce errors based on reliability
        reliability = self.behavior_profile.get('reliability', 0.95)
        if random.random() > reliability:
            response['data']['status'] = 'error'
            response['data']['error'] = 'Random error occurred'
        
        logger.debug(f"Entity {self.entity_id} processed interaction from {source_id}")
        return response
    
    async def process_test(self, test_data, source_id=None):
        """Process an active inference test."""
        # Record test in history
        self.interaction_history.append({
            'timestamp': time.time(),
            'source_id': source_id,
            'test_data': test_data,
            'is_test': True
        })
        
        # Extract test parameters
        test_type = test_data.get('test_type', 'unknown')
        
        # Generate response based on test type and behavior profile
        if test_type == 'timing_test':
            # Respond with actual timing capabilities
            actual_time = self.behavior_profile.get('response_time', 1.0)
            variation = self.behavior_profile.get('response_variability', 0.2)
            
            # Wait the actual response time
            await asyncio.sleep(actual_time * (1 + random.uniform(-variation, variation)))
            
            return {
                'timestamp': time.time(),
                'response_time': actual_time,
                'variation': variation,
                'test_type': test_type,
                'entity_id': self.entity_id
            }
            
        elif test_type == 'reliability_test':
            # Respond with actual reliability information
            reliability = self.behavior_profile.get('reliability', 0.95)
            
            # Sometimes fail based on reliability
            if random.random() > reliability:
                return {
                    'timestamp': time.time(),
                    'test_type': test_type,
                    'status': 'error',
                    'entity_id': self.entity_id,
                    'error': 'Test failed due to reliability issues'
                }
            
            # Return validation code if requested
            validation_code = test_data.get('validation_code')
            return {
                'timestamp': time.time(),
                'test_type': test_type,
                'status': 'success',
                'entity_id': self.entity_id,
                'validation_code': validation_code,
                'reliability': reliability
            }
            
        elif test_type == 'pattern_test':
            # Check pattern recognition capabilities
            pattern_recognition = self.behavior_profile.get('pattern_recognition', True)
            
            if not pattern_recognition:
                return {
                    'timestamp': time.time(),
                    'test_type': test_type,
                    'status': 'error',
                    'entity_id': self.entity_id,
                    'error': 'Pattern recognition not supported'
                }
            
            # Get sequence and expected next value
            sequence = test_data.get('sequence', [])
            expected_next = test_data.get('expected_next')
            
            # Determine accuracy of pattern recognition
            pattern_accuracy = self.behavior_profile.get('pattern_accuracy', 0.9)
            
            if random.random() <= pattern_accuracy:
                # Correctly recognize pattern
                return {
                    'timestamp': time.time(),
                    'test_type': test_type,
                    'status': 'success',
                    'entity_id': self.entity_id,
                    'next_value': expected_next,
                    'pattern_recognized': True
                }
            else:
                # Incorrectly recognize pattern
                return {
                    'timestamp': time.time(),
                    'test_type': test_type,
                    'status': 'success',
                    'entity_id': self.entity_id,
                    'next_value': expected_next + random.randint(1, 5),
                    'pattern_recognized': False
                }
        
        # Default response for unknown test types
        return {
            'timestamp': time.time(),
            'test_type': test_type,
            'status': 'unknown_test_type',
            'entity_id': self.entity_id
        }

class MockDistributionLayer:
    """Mock distribution layer for the demo."""
    
    def __init__(self):
        self.entities = {}  # Maps entity_id to MockEnvironmentalEntity
        logger.info("Created distribution layer")
    
    def register_entity(self, entity):
        """Register an environmental entity."""
        self.entities[entity.entity_id] = entity
        logger.info(f"Registered entity {entity.entity_id} with distribution layer")
        return True
    
    async def distribute_entity(self, source_id, target_id, interaction_data, context=None):
        """Distribute an interaction to a target entity."""
        if target_id not in self.entities:
            logger.warning(f"Target entity {target_id} not found in distribution layer")
            return None
        
        entity = self.entities[target_id]
        
        # Check if this is a test or regular interaction
        if context and context.get('test_interaction'):
            return await entity.process_test(interaction_data.get('test_content', {}), source_id)
        else:
            return await entity.process_interaction(interaction_data, source_id)

async def setup_demo_environment():
    """Set up the demo environment with mock components."""
    
    # Create knowledge substrate
    knowledge_substrate = MockKnowledgeSubstrate()
    
    # Create mock environmental entities with different behavior profiles
    entities = {
        'reliable_entity': MockEnvironmentalEntity('reliable_entity', {
            'response_time': 0.5,
            'response_variability': 0.1,
            'reliability': 0.95,
            'pattern_recognition': True,
            'pattern_accuracy': 0.9
        }),
        'unreliable_entity': MockEnvironmentalEntity('unreliable_entity', {
            'response_time': 1.5,
            'response_variability': 0.4,
            'reliability': 0.6,
            'pattern_recognition': True,
            'pattern_accuracy': 0.7
        }),
        'slow_entity': MockEnvironmentalEntity('slow_entity', {
            'response_time': 2.0,
            'response_variability': 0.3,
            'reliability': 0.9,
            'pattern_recognition': False
        }),
        'fast_entity': MockEnvironmentalEntity('fast_entity', {
            'response_time': 0.2,
            'response_variability': 0.1,
            'reliability': 0.8,
            'pattern_recognition': True,
            'pattern_accuracy': 0.95
        })
    }
    
    # Create distribution layer
    distribution_layer = MockDistributionLayer()
    
    # Register entities with distribution layer
    for entity in entities.values():
        distribution_layer.register_entity(entity)
    
    # Create coupling layer
    coupling_layer = EnvironmentalCouplingLayer(knowledge_substrate)
    
    # Initialize layer
    await coupling_layer.initialize(layer6=distribution_layer)
    
    # Create orchestrator
    orchestrator = PredictiveProcessingOrchestrator(coupling_layer)
    await orchestrator.initialize()
    
    # Create internal entities
    internal_entities = ['system_entity_1', 'system_entity_2']
    
    # Register internal entities with orchestrator
    for entity_id in internal_entities:
        await orchestrator.register_entity(entity_id)
    
    # Create couplings between internal and environmental entities
    couplings = []
    
    # Entity 1 couples with reliable and unreliable entities
    couplings.append(EnvironmentalCoupling(
        id=str(time.time()) + "_coupling_1",
        internal_entity_id='system_entity_1',
        environmental_entity_id='reliable_entity',
        coupling_type=CouplingType.INFORMATIONAL,
        coupling_strength=0.7,
        coupling_state=CouplingState.ACTIVE
    ))
    
    couplings.append(EnvironmentalCoupling(
        id=str(time.time()) + "_coupling_2",
        internal_entity_id='system_entity_1',
        environmental_entity_id='unreliable_entity',
        coupling_type=CouplingType.CONTEXTUAL,
        coupling_strength=0.5,
        coupling_state=CouplingState.ACTIVE
    ))
    
    # Entity 2 couples with slow and fast entities
    couplings.append(EnvironmentalCoupling(
        id=str(time.time()) + "_coupling_3",
        internal_entity_id='system_entity_2',
        environmental_entity_id='slow_entity',
        coupling_type=CouplingType.ADAPTIVE,
        coupling_strength=0.6,
        coupling_state=CouplingState.ACTIVE
    ))
    
    couplings.append(EnvironmentalCoupling(
        id=str(time.time()) + "_coupling_4",
        internal_entity_id='system_entity_2',
        environmental_entity_id='fast_entity',
        coupling_type=CouplingType.PREDICTIVE,
        coupling_strength=0.8,
        coupling_state=CouplingState.ACTIVE
    ))
    
    # Register couplings with coupling layer
    for coupling in couplings:
        await coupling_layer.coupling_registry.add_coupling(coupling)
        logger.info(f"Added coupling {coupling.id} between {coupling.internal_entity_id} and {coupling.environmental_entity_id}")
    
    return {
        'knowledge_substrate': knowledge_substrate,
        'entities': entities,
        'distribution_layer': distribution_layer,
        'coupling_layer': coupling_layer,
        'orchestrator': orchestrator,
        'internal_entities': internal_entities,
        'couplings': couplings
    }

async def run_integrated_demo(duration=300):
    """Run the integrated predictive processing demonstration."""
    
    logger.info("Setting up demonstration environment...")
    env = await setup_demo_environment()
    
    logger.info(f"Running demonstration for {duration} seconds...")
    
    # Start adaptive cycles for entity 1
    entity1_task = asyncio.create_task(
        env['orchestrator'].run_adaptive_cycles(
            'system_entity_1',
            min_interval=10,
            max_interval=60
        )
    )
    
    # Start continuous cycles for entity 2
    entity2_task = asyncio.create_task(
        env['orchestrator'].run_continuous_cycles(
            'system_entity_2',
            interval=30
        )
    )
    
    # Run for specified duration
    try:
        await asyncio.sleep(duration)
    except asyncio.CancelledError:
        logger.info("Demo cancelled")
    finally:
        # Stop cycles
        entity1_task.cancel()
        entity2_task.cancel()
        
        try:
            await asyncio.gather(entity1_task, entity2_task, return_exceptions=True)
        except Exception as e:
            pass
    
    # Unregister entities
    for entity_id in env['internal_entities']:
        await env['orchestrator'].unregister_entity(entity_id)
    
    # Perform final maintenance
    maintenance_results = await env['orchestrator'].perform_maintenance()
    
    logger.info("Demonstration completed")
    logger.info(f"Final maintenance results: {json.dumps(maintenance_results, indent=2)}")
    
    # Return statistics
    return {
        'duration': duration,
        'entities_processed': len(env['internal_entities']),
        'environmental_entities': len(env['entities']),
        'couplings': len(env['couplings']),
        'maintenance': maintenance_results
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Integrated Predictive Processing Demo')
    parser.add_argument('--duration', type=int, default=300, help='Duration in seconds')
    args = parser.parse_args()
    
    logger.info(f"Starting Integrated Predictive Processing Demo ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    
    asyncio.run(run_integrated_demo(args.duration))
