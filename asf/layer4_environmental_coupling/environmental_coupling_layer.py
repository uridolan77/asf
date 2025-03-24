import asyncio
import time
import uuid
import logging
import numpy as np
import random
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

from asf.layer4_environmental_coupling.enums import (
    CouplingType, CouplingStrength, CouplingState, EventPriority, PredictionState
)
from asf.layer4_environmental_coupling.models import (
    CouplingEvent, EnvironmentalCoupling, EnvironmentalPrediction, ActiveInferenceTest
)
from asf.layer4_environmental_coupling.components.coupling_registry import SparseCouplingRegistry
from asf.layer4_environmental_coupling.components.event_processor import EventDrivenProcessor, AsyncEventQueue
from asf.layer4_environmental_coupling.components.enhanced_bayesian_updater import EnhancedBayesianUpdater
from asf.layer4_environmental_coupling.components.rl_optimizer import ReinforcementLearningOptimizer
from asf.layer4_environmental_coupling.components.coherence_boundary import CoherenceBoundaryController
from asf.layer4_environmental_coupling.components.gpu_accelerator import GPUAccelerationManager
from asf.layer4_environmental_coupling.components.context_tracker import AdaptiveContextTracker
from asf.layer4_environmental_coupling.components.distributed_cache import DistributedCouplingCache
from asf.layer4_environmental_coupling.components.metrics_collector import PerformanceMetricsCollector

# Seth's Data Paradox components
from asf.layer4_environmental_coupling.components.predictive_modeler import PredictiveEnvironmentalModeler
from asf.layer4_environmental_coupling.components.active_inference_controller import ActiveInferenceController
from asf.layer4_environmental_coupling.components.counterfactual_simulator import CounterfactualSimulator

class EnvironmentalCouplingLayer:
    """
    Enhanced Layer 4 with complete integration of Seth's predictive processing principles.
    Orchestrates controlled hallucination, precision-weighted prediction errors, active inference,
    and counterfactual simulation as a unified predictive system.
    """
    def __init__(self, knowledge_substrate, async_queue=None, config=None):
        # Original components with enhancements
        self.knowledge_substrate = knowledge_substrate
        self.async_queue = async_queue or AsyncEventQueue()
        self.config = config or {}
        
        # Core components with enhanced versions
        self.coupling_registry = SparseCouplingRegistry(
            initial_capacity=self.config.get('initial_capacity', 10000)
        )
        self.event_processor = EventDrivenProcessor(
            max_concurrency=self.config.get('max_concurrency', 16)
        )
        self.bayesian_updater = EnhancedBayesianUpdater()  # Seth-enhanced version
        self.rl_optimizer = ReinforcementLearningOptimizer(
            learning_rate=self.config.get('learning_rate', 0.01)
        )
        
        # Seth's Data Paradox components
        self.predictive_modeler = PredictiveEnvironmentalModeler()
        self.active_inference = ActiveInferenceController(knowledge_substrate)
        self.counterfactual_simulator = CounterfactualSimulator()
        
        # Original components
        self.coherence_boundary = CoherenceBoundaryController(knowledge_substrate)
        self.gpu_accelerator = GPUAccelerationManager(
            enabled=self.config.get('use_gpu', True)
        )
        self.context_tracker = AdaptiveContextTracker()
        self.distributed_cache = DistributedCouplingCache(
            self.config.get('cache_config', {})
        )
        
        # Seth's principle configuration
        self.prediction_enabled = self.config.get('prediction_enabled', True)
        self.active_inference_enabled = self.config.get('active_inference_enabled', True)
        self.counterfactual_enabled = self.config.get('counterfactual_enabled', True)
        
        # Adjacent layers
        self.layer5 = None
        self.layer6 = None
        
        # Performance metrics
        self.metrics_collector = PerformanceMetricsCollector()
        self.logger = logging.getLogger("ASF.Layer4.EnvironmentalCouplingLayer")
        
    async def initialize(self, layer5=None, layer6=None):
        """Initialize the layer and connect to adjacent layers."""
        self.layer5 = layer5  # AutopoieticMaintenanceLayer
        self.layer6 = layer6  # EnvironmentalDistributionLayer
        
        # Initialize components
        await self.coupling_registry.initialize()
        await self.event_processor.initialize(self.process_coupling_event)
        await self.rl_optimizer.initialize(self.knowledge_substrate)
        await self.gpu_accelerator.initialize()
        
        # Set up component connections
        self.active_inference.set_coupling_registry(self.coupling_registry)
        
        # Start background task for async event processing
        asyncio.create_task(self.event_processor.run_processing_loop())
        
        self.logger.info(f"Layer 4 (Environmental Coupling) initialized with Seth's Data Paradox principles")
        self.logger.info(f"Prediction enabled: {self.prediction_enabled}")
        self.logger.info(f"Active inference enabled: {self.active_inference_enabled}")
        self.logger.info(f"Counterfactual simulation enabled: {self.counterfactual_enabled}")
        
        return {'status': 'initialized'}
        
    async def integrate_prediction_cycle(self, entity_id, context=None):
        """
        Execute a complete predictive processing cycle for an entity.
        This implements the full predictive cycle from Seth's framework:
        1. Generate predictions (controlled hallucination)
        2. Collect sensory evidence
        3. Calculate precision-weighted prediction errors
        4. Update internal models
        5. Perform active inference as needed
        6. Explore counterfactual configurations
        """
        start_time = time.time()
        cycle_results = {}
        
        # 1. Generate predictions using controlled hallucination
        predictions = await self._generate_entity_predictions(entity_id, context)
        cycle_results['predictions'] = {
            'count': len(predictions),
            'prediction_ids': [p.id for p in predictions]
        }
        
        # 2 & 3. Collect evidence and evaluate predictions
        current_state = await self._collect_environmental_state(entity_id)
        evaluations = await self._evaluate_predictions(predictions, current_state)
        cycle_results['evaluations'] = {
            'count': len(evaluations),
            'avg_error': np.mean([e['error'] for e in evaluations]) if evaluations else None,
            'avg_precision': np.mean([e['precision'] for e in evaluations]) if evaluations else None
        }
        
        # 4. Update internal models based on precision-weighted errors
        model_updates = await self._update_internal_models(entity_id, evaluations)
        cycle_results['model_updates'] = model_updates
        
        # 5. Perform active inference if needed
        if self._should_perform_active_inference(evaluations):
            inference_results = await self.perform_active_inference(entity_id)
            cycle_results['active_inference'] = inference_results
        
        # 6. Explore counterfactual configurations
        counterfactuals = await self._explore_counterfactuals(entity_id, evaluations)
        cycle_results['counterfactuals'] = counterfactuals
        
        # Generate new predictions for next cycle
        new_predictions = await self._generate_entity_predictions(entity_id, context, current_state)
        cycle_results['new_predictions'] = {
            'count': len(new_predictions),
            'prediction_ids': [p.id for p in new_predictions]
        }
        
        # Calculate cycle metrics
        cycle_results['elapsed_time'] = time.time() - start_time
        cycle_results['success'] = True
        
        self.logger.info(f"Completed predictive cycle for entity {entity_id} in {cycle_results['elapsed_time']:.3f}s")
        return cycle_results
    
    async def _generate_entity_predictions(self, entity_id, context=None, current_state=None):
        """Generate predictions for an entity across all its environmental connections."""
        predictions = []
        
        # Get all couplings for this entity
        couplings = await self.coupling_registry.get_couplings_by_internal_entity(entity_id)
        
        for coupling in couplings:
            # Skip inactive couplings
            if coupling.coupling_state != CouplingState.ACTIVE:
                continue
            
            # Generate prediction for each coupling
            prediction = await self.predictive_modeler.predict_interaction(
                coupling.environmental_entity_id,
                context={
                    'coupling_id': coupling.id,
                    'coupling_type': coupling.coupling_type.name,
                    'bayesian_confidence': coupling.bayesian_confidence,
                    'current_state': current_state,
                    **(context or {})
                }
            )
            
            predictions.append(prediction)
            
            # Associate prediction with coupling
            if hasattr(coupling, 'expected_interactions'):
                coupling.expected_interactions[prediction.id] = time.time()
                # Cleanup old predictions
                if len(coupling.expected_interactions) > 10:
                    oldest = min(coupling.expected_interactions.items(), key=lambda x: x[1])[0]
                    del coupling.expected_interactions[oldest]
                    
                # Update coupling in registry
                await self.coupling_registry.update_coupling(coupling)
        
        return predictions
    
    async def _collect_environmental_state(self, entity_id):
        """Collect current state information about environmental connections."""
        # This would normally query environmental entities for current state
        # For this implementation, we'll create a simplified representation
        
        state = {
            'timestamp': time.time(),
            'entity_id': entity_id,
            'active_couplings': 0,
            'environments': {}
        }
        
        # Get all couplings for this entity
        couplings = await self.coupling_registry.get_couplings_by_internal_entity(entity_id)
        
        for coupling in couplings:
            if coupling.coupling_state == CouplingState.ACTIVE:
                state['active_couplings'] += 1
                
                # Collect state for each environmental entity
                env_id = coupling.environmental_entity_id
                if env_id not in state['environments']:
                    state['environments'][env_id] = {
                        'last_interaction': coupling.last_interaction,
                        'coupling_strength': coupling.coupling_strength,
                        'bayesian_confidence': coupling.bayesian_confidence,
                        'coupling_type': coupling.coupling_type.name
                    }
                    
                    # Add precision information if available
                    if hasattr(coupling, 'prediction_precision'):
                        state['environments'][env_id]['prediction_precision'] = coupling.prediction_precision
        
        return state
    
    async def _evaluate_predictions(self, predictions, current_state):
        """
        Evaluate predictions against current state.
        This is a simplified implementation since we don't have actual environmental data.
        """
        evaluations = []
        
        for prediction in predictions:
            env_id = prediction.environmental_entity_id
            
            # If we have state data for this environment, evaluate against it
            if env_id in current_state.get('environments', {}):
                env_state = current_state['environments'][env_id]
                
                # Create mock actual data based on state (simplified)
                actual_data = {
                    'interaction_type': prediction.predicted_data.get('predicted_interaction_type', 'unknown'),
                    'timestamp': time.time(),
                    'content_type': 'state_update',
                    'response_time': time.time() - env_state.get('last_interaction', time.time() - 60)
                }
                
                # Evaluate prediction
                evaluation = await self.predictive_modeler.evaluate_prediction(
                    prediction.id, actual_data
                )
                
                if evaluation:
                    evaluations.append(evaluation)
                    self.logger.debug(f"Evaluated prediction {prediction.id} with error {evaluation['error']:.3f}")
        
        return evaluations
    
    async def _update_internal_models(self, entity_id, evaluations):
        """Update internal models based on prediction evaluations."""
        update_results = {
            'precision_updates': 0,
            'model_updates': 0,
            'confidence_changes': []
        }
        
        # Get all couplings for this entity
        couplings = await self.coupling_registry.get_couplings_by_internal_entity(entity_id)
        couplings_by_env = {c.environmental_entity_id: c for c in couplings}
        
        # Process each evaluation
        for evaluation in evaluations:
            prediction_id = evaluation.get('prediction_id')
            if prediction_id and prediction_id in self.predictive_modeler.predictions:
                prediction = self.predictive_modeler.predictions[prediction_id]
                env_id = prediction.environmental_entity_id
                
                # If we have a coupling for this environment, update it
                if env_id in couplings_by_env:
                    coupling = couplings_by_env[env_id]
                    
                    # Update precision for the coupling
                    precision = evaluation.get('precision', 1.0)
                    old_precision = getattr(coupling, 'prediction_precision', 1.0)
                    coupling.prediction_precision = precision
                    update_results['precision_updates'] += 1
                    
                    # Update Bayesian model with precision-weighted confidence
                    # Get base confidence (could be from prediction or default)
                    confidence = prediction.confidence if hasattr(prediction, 'confidence') else 0.5
                    
                    # Update Bayesian model with precision weighting
                    bayesian_result = await self.bayesian_updater.update_precision(
                        coupling.id, evaluation.get('error', 0.5)
                    )
                    
                    # Update confidence tracking
                    update_results['confidence_changes'].append({
                        'coupling_id': coupling.id,
                        'old_precision': old_precision,
                        'new_precision': precision,
                        'error': evaluation.get('error', 0.5)
                    })
                    
                    # Update coupling in registry
                    await self.coupling_registry.update_coupling(coupling)
                    update_results['model_updates'] += 1
        
        return update_results
    
    def _should_perform_active_inference(self, evaluations):
        """Determine if active inference should be performed based on evaluations."""
        if not evaluations:
            return False
            
        # Perform active inference if:
        # 1. Average error is high (above 0.5)
        # 2. OR precision is low (below 1.0)
        avg_error = np.mean([e.get('error', 0) for e in evaluations])
        avg_precision = np.mean([e.get('precision', 1.0) for e in evaluations])
        
        return avg_error > 0.5 or avg_precision < 1.0
    
    async def _explore_counterfactuals(self, entity_id, evaluations):
        """Explore counterfactual configurations for high-error predictions."""
        results = {
            'simulations_run': 0,
            'improvements_found': 0,
            'optimal_configurations': []
        }
        
        # Focus on evaluations with high error
        high_error_evaluations = [e for e in evaluations if e.get('error', 0) > 0.4]
        
        if not high_error_evaluations:
            return results
            
        # Get couplings for entity
        couplings = await self.coupling_registry.get_couplings_by_internal_entity(entity_id)
        couplings_by_env = {c.environmental_entity_id: c for c in couplings}
        
        # For each high-error prediction, explore counterfactuals
        for evaluation in high_error_evaluations:
            prediction_id = evaluation.get('prediction_id')
            if prediction_id and prediction_id in self.predictive_modeler.predictions:
                prediction = self.predictive_modeler.predictions[prediction_id]
                env_id = prediction.environmental_entity_id
                
                # If we have a coupling for this environment, explore counterfactuals
                if env_id in couplings_by_env:
                    coupling = couplings_by_env[env_id]
                    
                    # Run counterfactual simulation
                    simulation_result = await self.simulate_counterfactual_coupling(coupling.id, 5)
                    results['simulations_run'] += 1
                    
                    if simulation_result.get('success', False):
                        optimal_config = simulation_result.get('optimal_configuration')
                        if optimal_config and optimal_config.get('improvement', 0) > 0.1:
                            # Found a significant improvement
                            results['improvements_found'] += 1
                            results['optimal_configurations'].append({
                                'coupling_id': coupling.id,
                                'improvement': optimal_config.get('improvement', 0),
                                'configuration_type': optimal_config.get('optimal_configuration', {}).get('variation_type')
                            })
        
        return results
    
    async def process_environmental_interaction(self, interaction_data, source_id=None, interaction_type=None, confidence=None, context=None):
        """
        Process an interaction from the environment with predictive enhancements.
        Implements Seth's principles of prediction, precision-weighting, and active inference.
        """
        start_time = time.time()
        
        # Create processing context if not provided
        if context is None:
            context = await self.context_tracker.create_context(
                interaction_data, source_id, interaction_type
            )
        
        # Check if we have predictions for this interaction
        predictions = []
        prediction_evaluations = []
        if self.prediction_enabled and source_id:
            predictions = await self._get_relevant_predictions(source_id, interaction_type)
            
            # Evaluate predictions against actual data
            for prediction in predictions:
                evaluation = await self.predictive_modeler.evaluate_prediction(
                    prediction.id, interaction_data
                )
                if evaluation:
                    prediction_evaluations.append(evaluation)
                    self.logger.debug(f"Prediction evaluation: error={evaluation['error']:.3f}, precision={evaluation['precision']:.3f}")
        
        # Check coherence boundary
        coherent, boundary_result = await self.coherence_boundary.check_interaction_coherence(
            interaction_data, source_id, interaction_type, context
        )
        
        if not coherent:
            return {
                'success': False,
                'error': 'Interaction violates coherence boundary',
                'boundary_result': boundary_result,
                'prediction_evaluations': prediction_evaluations,
                'prediction_count': len(predictions),
                'elapsed_time': time.time() - start_time
            }
        
        # Get relevant couplings
        couplings = []
        if source_id:
            couplings = await self.coupling_registry.get_couplings_by_environmental_entity(source_id)
        
        # Process through couplings with precision-weighted updating
        processing_results = []
        
        for coupling in couplings:
            # Get precision from prediction evaluations
            precision = 1.0
            if prediction_evaluations:
                # Use the most relevant prediction's precision
                precision = prediction_evaluations[0].get('precision', 1.0)
            
            # Prepare integrated event data
            event_data = {
                'interaction_data': interaction_data,
                'interaction_type': interaction_type,
                'confidence': confidence,
                'context': context,
                'prediction_evaluations': prediction_evaluations,
                'precision': precision
            }
            
            # Create event for processing
            event = CouplingEvent(
                id=str(uuid.uuid4()),
                event_type='environmental_interaction',
                coupling_id=coupling.id,
                entity_id=coupling.internal_entity_id,
                environmental_id=coupling.environmental_entity_id,
                data=event_data,
                # Add prediction information
                predicted=bool(predictions),
                prediction_id=predictions[0].id if predictions else None,
                precision=precision
            )
            
            # Submit for processing
            await self.event_processor.submit_event(event)
            
            processing_results.append({
                'coupling_id': coupling.id,
                'event_id': event.id,
                'status': 'submitted',
                'precision_applied': precision
            })
        
        # If no couplings processed, still create an event for system-level handling
        if not processing_results:
            event = CouplingEvent(
                id=str(uuid.uuid4()),
                event_type='unassociated_interaction',
                environmental_id=source_id,
                data={
                    'interaction_data': interaction_data,
                    'interaction_type': interaction_type,
                    'confidence': confidence,
                    'context': context
                }
            )
            await self.event_processor.submit_event(event)
            processing_results.append({
                'event_id': event.id,
                'status': 'submitted_unassociated'
            })
        
        # After processing the interaction, generate new predictions
        if self.prediction_enabled and source_id:
            # Context enriched with evaluation results
            prediction_context = {
                'last_interaction_type': interaction_type,
                'evaluation_results': [
                    {'error': e['error'], 'precision': e['precision']}
                    for e in prediction_evaluations
                ] if prediction_evaluations else []
            }
            
            new_prediction = await self.predictive_modeler.predict_interaction(
                source_id, prediction_context
            )
            
            self.logger.debug(f"Generated new prediction {new_prediction.id} for {source_id}")
        
        # Check if active inference is needed based on prediction errors
        active_inference_triggered = False
        if self.active_inference_enabled and prediction_evaluations:
            avg_error = np.mean([e['error'] for e in prediction_evaluations])
            if avg_error > 0.6:  # High error threshold
                # Get the internal entity for the source
                internal_entity_id = None
                if couplings:
                    internal_entity_id = couplings[0].internal_entity_id
                
                if internal_entity_id:
                    # Schedule active inference in background
                    asyncio.create_task(self.perform_active_inference(internal_entity_id))
                    active_inference_triggered = True
                    self.logger.info(f"Triggered active inference due to high prediction error: {avg_error:.3f}")
        
        # Check if counterfactual simulation is needed
        counterfactual_triggered = False
        if self.counterfactual_enabled and couplings and prediction_evaluations:
            # Find the coupling with highest prediction error
            max_error_eval = max(prediction_evaluations, key=lambda e: e['error'])
            if max_error_eval['error'] > 0.7:  # Very high error threshold
                coupling_to_simulate = couplings[0].id  # Default to first
                
                # Try to find the specific coupling for this high-error prediction
                prediction_id = max_error_eval.get('prediction_id')
                if prediction_id and prediction_id in self.predictive_modeler.predictions:
                    pred = self.predictive_modeler.predictions[prediction_id]
                    env_id = pred.environmental_entity_id
                    
                    # Find the coupling for this environmental entity
                    for c in couplings:
                        if c.environmental_entity_id == env_id:
                            coupling_to_simulate = c.id
                            break
                
                # Schedule counterfactual simulation in background
                asyncio.create_task(self.simulate_counterfactual_coupling(coupling_to_simulate))
                counterfactual_triggered = True
                self.logger.info(f"Triggered counterfactual simulation due to very high prediction error: {max_error_eval['error']:.3f}")
        
        return {
            'success': True,
            'couplings_found': len(couplings),
            'processing_results': processing_results,
            'prediction_evaluations': prediction_evaluations,
            'prediction_count': len(predictions),
            'active_inference_triggered': active_inference_triggered,
            'counterfactual_triggered': counterfactual_triggered,
            'elapsed_time': time.time() - start_time
        }
    
    async def process_coupling_event(self, event):
        """
        Process a coupling event with integrated predictive components.
        """
        start_time = time.time()
        
        # Check if this is an active inference test result
        if hasattr(event, 'data') and event.data.get('active_inference_test_result'):
            test_result = event.data.get('active_inference_test_result')
            test_id = test_result.get('test_id')
            
            if test_id:
                test_processing = await self.process_test_result(test_id, test_result)
                result = {
                    'success': True,
                    'event_type': 'active_inference_test_result',
                    'test_processing': test_processing,
                    'elapsed_time': time.time() - start_time
                }
                return result
        
        if not hasattr(event, 'coupling_id') or not event.coupling_id:
            return {'success': False, 'error': 'No coupling ID in event'}
            
        # Get the coupling
        coupling = await self.coupling_registry.get_coupling(event.coupling_id)
        if not coupling:
            return {'success': False, 'error': 'Coupling not found'}
            
        # Extract event data
        event_data = event.data if hasattr(event, 'data') and event.data else {}
        interaction_data = event_data.get('interaction_data', {})
        interaction_type = event_data.get('interaction_type')
        confidence = event_data.get('confidence', 0.5)
        context = event_data.get('context', {})
        
        # Stage 1: Apply precision-weighted Bayesian updates
        # Get precision from event or prediction evaluations
        precision = event_data.get('precision', 1.0)
        prediction_evaluations = event_data.get('prediction_evaluations', [])
        
        if not precision and prediction_evaluations:
            precision = prediction_evaluations[0].get('precision', 1.0)
        
        # Update Bayesian model with prediction-based precision
        bayesian_result = await self.bayesian_updater.update_from_interaction(
            coupling.id, interaction_data, interaction_type, confidence
        )
        
        # Update coupling based on Bayesian result
        coupling.bayesian_confidence = bayesian_result['new_confidence']
        
        # If strength changed significantly, update coupling strength
        if abs(bayesian_result['strength_delta']) > 0.1:
            old_strength = coupling.coupling_strength
            new_strength = max(0.1, min(1.0, old_strength + bayesian_result['strength_delta'] * 0.2))
            coupling.coupling_strength = new_strength
            
        # Update interaction counter and timestamp
        coupling.interaction_count += 1
        coupling.last_interaction = time.time()
        
        # Stage 2: Update prediction precision from evaluations
        if prediction_evaluations:
            # Store prediction error information for future precision calculations
            coupling.prediction_errors = coupling.prediction_errors or []
            for eval_result in prediction_evaluations:
                if 'error' in eval_result:
                    coupling.prediction_errors.append(eval_result['error'])
                    
            # Limit history size
            if len(coupling.prediction_errors) > 20:
                coupling.prediction_errors = coupling.prediction_errors[-20:]
                
            # Update coupling's prediction precision
            coupling.prediction_precision = precision
        
        # Stage 3: Check for contradictions and handle them
        if bayesian_result['contradiction_detected']:
            contradiction = bayesian_result['contradiction']
            await self._handle_contradiction(coupling, contradiction, context)
        
        # Stage 4: Use Reinforcement Learning to optimize coupling parameters
        rl_result = await self.rl_optimizer.update_from_interaction(
            coupling.id, interaction_data, interaction_type, bayesian_result
        )
        
        # Stage 5: Update coupling in registry
        await self.coupling_registry.update_coupling(coupling)
        
        # Return processing results
        return {
            'success': True,
            'coupling_id': coupling.id,
            'bayesian_update': bayesian_result,
            'reinforcement_learning': rl_result,
            'precision_applied': precision,
            'prediction_count': len(prediction_evaluations),
            'elapsed_time': time.time() - start_time
        }
    
    async def _handle_contradiction(self, coupling, contradiction, context):
        """
        Handle a detected contradiction in coupling.
        Contradictions are opportunities for learning and boundary refinement.
        """
        if not contradiction:
            return
            
        self.logger.info(f"Handling contradiction in coupling {coupling.id}: {contradiction['type']}")
        
        # Notify upper layers about contradiction
        if self.layer5:
            asyncio.create_task(self.layer5.process_contradiction(
                coupling.id,
                coupling.internal_entity_id,
                coupling.environmental_entity_id,
                contradiction,
                context
            ))
        
        # Record contradiction in coupling history
        if not hasattr(coupling, 'history'):
            coupling.history = []
            
        coupling.history.append({
            'timestamp': time.time(),
            'event_type': 'contradiction',
            'contradiction_type': contradiction['type'],
            'prior_confidence': contradiction['prior_confidence'],
            'context': context
        })
        
        # For high-confidence contradictions, consider active inference
        if contradiction.get('prior_confidence', 0) > 0.8 and self.active_inference_enabled:
            asyncio.create_task(self.perform_active_inference(coupling.internal_entity_id))
    
    async def perform_active_inference(self, entity_id):
        """
        Actively test and optimize couplings using controlled interactions.
        Implements Seth's active inference principle.
        """
        if not self.active_inference_enabled:
            return {'success': False, 'error': 'Active inference is disabled'}
        
        start_time = time.time()
        
        # Get all current couplings
        couplings = await self.coupling_registry.get_couplings_by_internal_entity(entity_id)
        if not couplings:
            return {
                'success': False,
                'error': 'No couplings found',
                'elapsed_time': time.time() - start_time
            }
        
        inference_results = []
        
        # For each coupling, generate test interactions to minimize uncertainty
        for coupling in couplings:
            # Only test active couplings
            if not hasattr(coupling, 'coupling_state') or coupling.coupling_state != CouplingState.ACTIVE:
                continue
                
            # Generate optimal test interaction based on current uncertainty
            test = await self.active_inference.generate_test_interaction(
                coupling.id,
                uncertainty_focus=True
            )
            
            if test:
                # Execute test interaction to gather information
                if self.layer6:  # EnvironmentalDistributionLayer
                    # Send test interaction through the distribution layer
                    distribution_task = asyncio.create_task(
                        self.layer6.distribute_entity(
                            entity_id,
                            target_id=coupling.environmental_entity_id,
                            context={
                                'test_interaction': True,
                                'active_inference': True,
                                'test_id': test.id
                            },
                            interaction_data=test.test_parameters
                        )
                    )
                    
                    inference_results.append({
                        'coupling_id': coupling.id,
                        'test_id': test.id,
                        'test_parameters': test.test_parameters,
                        'status': 'submitted'
                    })
                else:
                    # No distribution layer available
                    inference_results.append({
                        'coupling_id': coupling.id,
                        'test_id': test.id,
                        'test_parameters': test.test_parameters,
                        'status': 'distribution_layer_unavailable'
                    })
        
        return {
            'success': True,
            'entity_id': entity_id,
            'couplings_tested': len(inference_results),
            'inference_results': inference_results,
            'elapsed_time': time.time() - start_time
        }
    
    async def process_test_result(self, test_id, actual_result):
        """
        Process the result of an active inference test.
        Updates coupling based on test information gain.
        """
        if not self.active_inference_enabled:
            return {'success': False, 'error': 'Active inference is disabled'}
        
        start_time = time.time()
        
        # Evaluate test result
        evaluation = await self.active_inference.evaluate_test_result(test_id, actual_result)
        
        if not evaluation.get('success', False):
            return {
                'success': False,
                'error': evaluation.get('error', 'Test evaluation failed'),
                'elapsed_time': time.time() - start_time
            }
        
        # Log information gain
        self.logger.info(f"Test {test_id} resulted in information gain: {evaluation['information_gain']:.3f}")
        
        # Get the coupling this test was for
        coupling_id = evaluation.get('coupling_updates', {}).get('coupling_id')
        if coupling_id:
            coupling = await self.coupling_registry.get_coupling(coupling_id)
            
            if coupling:
                # Apply any coupling updates from test result
                if 'coupling_updates' in evaluation and evaluation['coupling_updates'].get('coupling_changed', False):
                    # The active inference controller has already updated the coupling
                    self.logger.debug(f"Coupling {coupling_id} updated based on test results")
                
                # If this test significantly reduced uncertainty, consider additional tests
                if evaluation['information_gain'] > 0.8:
                    # Schedule additional tests if high information gain suggests more could be learned
                    asyncio.create_task(self.schedule_follow_up_tests(coupling))
        
        # Check if layer5 (maintenance) needs to be informed of test results
        if self.layer5 and evaluation.get('information_gain', 0) < 0.3:
            # Low information gain might indicate a boundary issue
            asyncio.create_task(
                self.layer5.examine_coupling(
                    coupling_id, 
                    reason="low_active_inference_gain",
                    metadata={
                        'test_id': test_id,
                        'information_gain': evaluation.get('information_gain', 0),
                        'target_area': evaluation.get('target_area', 'unknown')
                    }
                )
            )
        
        return {
            'success': True,
            'test_id': test_id,
            'evaluation': evaluation,
            'elapsed_time': time.time() - start_time
        }
    
    async def schedule_follow_up_tests(self, coupling):
        """
        Schedule follow-up tests when a test yields high information gain.
        This allows the system to progressively refine its understanding.
        """
        # Wait a bit before the follow-up test
        await asyncio.sleep(5)
        
        # Generate a new test focusing on a different uncertainty area
        test = await self.active_inference.generate_test_interaction(
            coupling.id,
            uncertainty_focus=True
        )
        
        if test and self.layer6:
            # Send follow-up test through distribution layer
            await self.layer6.distribute_entity(
                coupling.internal_entity_id,
                target_id=coupling.environmental_entity_id,
                context={
                    'test_interaction': True,
                    'active_inference': True,
                    'test_id': test.id,
                    'follow_up': True
                },
                interaction_data=test.test_parameters
            )
            
            self.logger.info(f"Scheduled follow-up test {test.id} for coupling {coupling.id}")
    
    async def simulate_counterfactual_coupling(self, coupling_id, variations=3):
        """
        Simulate alternative coupling configurations.
        Implements Seth's counterfactual processing principle.
        """
        if not self.counterfactual_enabled:
            return {'success': False, 'error': 'Counterfactual simulation is disabled'}
        
        start_time = time.time()
        
        # Get the coupling
        coupling = await self.coupling_registry.get_coupling(coupling_id)
        if not coupling:
            return {
                'success': False,
                'error': 'Coupling not found',
                'elapsed_time': time.time() - start_time
            }
        
        # Generate counterfactual variations
        counterfactuals = await self.counterfactual_simulator.generate_coupling_variations(
            coupling, variations
        )
        
        # Simulate outcomes
        simulation_results = await self.counterfactual_simulator.simulate_outcomes(
            counterfactuals
        )
        
        # Identify optimal configuration
        optimal_config = await self.counterfactual_simulator.identify_optimal_configuration(
            simulation_results
        )
        
        # Log the counterfactual simulation
        self.logger.info(f"Completed counterfactual simulation for coupling {coupling_id} with {variations} variations")
        
        # Store optimal configuration for future reference
        if not hasattr(coupling, 'properties'):
            coupling.properties = {}
            
        coupling.properties['counterfactual_recommendation'] = {
            'timestamp': time.time(),
            'optimal_config': {
                'variation_type': optimal_config['optimal_configuration']['variation_type'],
                'description': optimal_config['optimal_configuration']['description'],
                'predicted_improvement': optimal_config['improvement']
            }
        }
        
        # Update coupling in registry
        await self.coupling_registry.update_coupling(coupling)
        
        return {
            'success': True,
            'coupling_id': coupling_id,
            'counterfactual_count': len(counterfactuals),
            'simulation_results': simulation_results,
            'optimal_configuration': optimal_config,
            'elapsed_time': time.time() - start_time
        }
    
    async def apply_counterfactual_recommendation(self, coupling_id):
        """
        Apply the recommended counterfactual configuration to an actual coupling.
        """
        if not self.counterfactual_enabled:
            return {'success': False, 'error': 'Counterfactual simulation is disabled'}
        
        # Get the coupling
        coupling = await self.coupling_registry.get_coupling(coupling_id)
        if not coupling:
            return {'success': False, 'error': 'Coupling not found'}
        
        # Check if we have a recommendation
        if (not hasattr(coupling, 'properties') or
            'counterfactual_recommendation' not in coupling.properties):
            return {'success': False, 'error': 'No counterfactual recommendation available'}
        
        recommendation = coupling.properties['counterfactual_recommendation']
        if 'optimal_config' not in recommendation:
            return {'success': False, 'error': 'Invalid recommendation format'}
        
        optimal = recommendation['optimal_config']
        
        # Apply recommended changes
        original_values = {}
        
        # Record original values
        original_values['coupling_type'] = coupling.coupling_type
        original_values['coupling_strength'] = coupling.coupling_strength
        if hasattr(coupling, 'properties'):
            original_values['properties'] = dict(coupling.properties)
        
        # Apply changes based on variation type
        variation_type = optimal.get('variation_type')
        
        if variation_type == 'strength_increase':
            coupling.coupling_strength = min(1.0, coupling.coupling_strength * 1.2)
        
        elif variation_type == 'strength_decrease':
            coupling.coupling_strength = max(0.1, coupling.coupling_strength * 0.8)
        
        elif variation_type == 'type_adaptive':
            coupling.coupling_type = CouplingType.ADAPTIVE
        
        elif variation_type == 'type_predictive':
            coupling.coupling_type = CouplingType.PREDICTIVE
        
        elif variation_type == 'property_responsiveness':
            if not hasattr(coupling, 'properties'):
                coupling.properties = {}
            coupling.properties['response_threshold'] = 0.3
        
        elif variation_type == 'property_reliability':
            if not hasattr(coupling, 'properties'):
                coupling.properties = {}
            coupling.properties['reliability_factor'] = 0.8
        
        elif variation_type == 'property_precision':
            if not hasattr(coupling, 'properties'):
                coupling.properties = {}
            coupling.properties['precision_target'] = 2.0
        
        # Update application metadata
        coupling.properties['counterfactual_application'] = {
            'timestamp': time.time(),
            'applied_variation': variation_type,
            'original_values': original_values,
            'description': optimal.get('description', 'Counterfactual optimization')
        }
        
        # Update coupling in registry
        await self.coupling_registry.update_coupling(coupling)
        
        # Update history
        if hasattr(coupling, 'history'):
            coupling.history.append({
                'timestamp': time.time(),
                'action': 'counterfactual_applied',
                'variation_type': variation_type,
                'predicted_improvement': optimal.get('predicted_improvement', 0)
            })
        
        self.logger.info(f"Applied counterfactual recommendation to coupling {coupling_id}: {variation_type}")
        
        return {
            'success': True,
            'coupling_id': coupling_id,
            'applied_variation': variation_type,
            'description': optimal.get('description'),
            'original_values': original_values
        }
    
    async def _get_relevant_predictions(self, entity_id, interaction_type=None):
        """
        Get relevant predictions for an entity and interaction type.
        This method retrieves predictions that match the incoming interaction.
        """
        if not self.prediction_enabled or not entity_id:
            return []
            
        # Get recent predictions for this entity
        predictions = await self.predictive_modeler.get_predictions_for_entity(
            entity_id, 
            limit=5, 
            future_only=False  # Include recently verified predictions for comparison
        )
        
        # If we have a specific interaction type, prioritize matching predictions
        if interaction_type and predictions:
            # Sort by relevance:
            # 1. Predictions with matching interaction type
            # 2. Unverified predictions (future predictions)
            # 3. Most recent predictions
            
            def prediction_relevance(pred):
                # Calculate a relevance score (higher is better)
                score = 0
                
                # Higher score for matching interaction type
                if 'predicted_interaction_type' in pred.predicted_data:
                    if pred.predicted_data['predicted_interaction_type'] == interaction_type:
                        score += 10
                
                # Higher score for unverified predictions
                if pred.verification_time is None:
                    score += 5
                    
                # Higher score for recent predictions
                time_factor = max(0, 3600 - (time.time() - pred.prediction_time)) / 3600
                score += time_factor * 3
                
                return score
                
            # Sort by relevance score
            predictions.sort(key=prediction_relevance, reverse=True)
            
        # Limit to most relevant predictions
        return predictions[:3] if predictions else []
    
    async def process_predictions_for_entity(self, entity_id):
        """
        Process all predictions for an entity and update precision.
        Useful for periodic maintenance of prediction precision.
        """
        if not self.prediction_enabled:
            return {'success': False, 'status': 'predictions_disabled'}
            
        # Get all predictions for this entity
        predictions = await self.predictive_modeler.get_predictions_for_entity(
            entity_id, limit=10, future_only=False
        )
        
        if not predictions:
            return {'success': True, 'status': 'no_predictions', 'entity_id': entity_id}
            
        # Find verified predictions needing precision updates
        verified_count = 0
        precision_updates = 0
        
        for prediction in predictions:
            if prediction.verification_time is not None and prediction.prediction_error is not None:
                verified_count += 1
                
                # Update precision for verified predictions
                if prediction.environmental_entity_id:
                    await self.predictive_modeler.update_entity_precision(
                        prediction.environmental_entity_id, prediction.prediction_error
                    )
                    precision_updates += 1
                    
        # Generate new predictions if needed
        if verified_count == len(predictions):
            new_prediction = await self.predictive_modeler.predict_interaction(entity_id)
            self.logger.debug(f"Generated new prediction {new_prediction.id} for entity {entity_id}")
        
        return {
            'success': True,
            'entity_id': entity_id,
            'prediction_count': len(predictions),
            'verified_count': verified_count,
            'precision_updates': precision_updates
        }
