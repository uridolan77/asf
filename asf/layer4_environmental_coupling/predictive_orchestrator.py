import asyncio
import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict

class PredictiveProcessingOrchestrator:
    """
    Orchestrates the predictive processing cycle across all components.
    Implements Seth's integrated predictive processing framework with controlled
    hallucination, precision-weighted errors, active inference, and counterfactual simulation.
    """
    def __init__(self, coupling_layer):
        self.coupling_layer = coupling_layer
        self.active_entities = set()
        self.prediction_cycles = defaultdict(int)  # Maps entity_id to cycle count
        self.logger = logging.getLogger("ASF.Layer4.PredictiveOrchestrator")
        
    async def initialize(self):
        """Initialize the orchestrator."""
        self.logger.info("Initializing Predictive Processing Orchestrator")
        return True
        
    async def register_entity(self, entity_id):
        """Register an entity for predictive processing."""
        self.active_entities.add(entity_id)
        self.logger.info(f"Registered entity {entity_id} for predictive processing")
        return True
        
    async def unregister_entity(self, entity_id):
        """Unregister an entity from predictive processing."""
        if entity_id in self.active_entities:
            self.active_entities.remove(entity_id)
            self.logger.info(f"Unregistered entity {entity_id} from predictive processing")
        return True
        
    async def run_predictive_cycle(self, entity_id, context=None):
        """Run a complete predictive processing cycle for an entity."""
        if entity_id not in self.active_entities:
            return {'success': False, 'error': 'Entity not registered'}
            
        # Increment cycle count
        self.prediction_cycles[entity_id] += 1
        cycle_number = self.prediction_cycles[entity_id]
        
        # Enhanced context with cycle information
        enhanced_context = {
            'cycle_number': cycle_number,
            'orchestrator_timestamp': time.time(),
            **(context or {})
        }
        
        # Run the cycle through the coupling layer
        result = await self.coupling_layer.integrate_prediction_cycle(entity_id, enhanced_context)
        
        self.logger.info(f"Completed prediction cycle {cycle_number} for entity {entity_id}")
        
        return {
            'success': True,
            'entity_id': entity_id,
            'cycle_number': cycle_number,
            'cycle_results': result
        }
        
    async def run_continuous_cycles(self, entity_id, interval=60, max_cycles=None):
        """Run continuous predictive cycles for an entity at specified interval."""
        if entity_id not in self.active_entities:
            return {'success': False, 'error': 'Entity not registered'}
            
        cycle_count = 0
        start_time = time.time()
        
        self.logger.info(f"Starting continuous prediction cycles for entity {entity_id} at {interval}s intervals")
        
        while (max_cycles is None or cycle_count < max_cycles) and entity_id in self.active_entities:
            cycle_start = time.time()
            
            # Run a cycle
            await self.run_predictive_cycle(entity_id, {
                'continuous_mode': True,
                'continuous_cycle': cycle_count,
                'elapsed_time': time.time() - start_time
            })
            
            cycle_count += 1
            
            # Wait for next cycle, accounting for processing time
            cycle_duration = time.time() - cycle_start
            wait_time = max(0, interval - cycle_duration)
            
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        
        return {
            'success': True,
            'entity_id': entity_id,
            'cycles_completed': cycle_count,
            'total_runtime': time.time() - start_time
        }
        
    async def run_adaptive_cycles(self, entity_id, min_interval=10, max_interval=300):
        """
        Run adaptive predictive cycles with frequency based on prediction accuracy.
        More accurate predictions = longer intervals, less accurate = shorter intervals.
        """
        if entity_id not in self.active_entities:
            return {'success': False, 'error': 'Entity not registered'}
            
        cycle_count = 0
        current_interval = (min_interval + max_interval) / 2  # Start with middle interval
        start_time = time.time()
        
        self.logger.info(f"Starting adaptive prediction cycles for entity {entity_id}")
        
        while entity_id in self.active_entities:
            cycle_start = time.time()
            
            # Run a cycle
            result = await self.run_predictive_cycle(entity_id, {
                'adaptive_mode': True,
                'current_interval': current_interval,
                'adaptive_cycle': cycle_count
            })
            
            cycle_count += 1
            
            # Adapt interval based on prediction accuracy
            cycle_results = result.get('cycle_results', {})
            evaluations = cycle_results.get('evaluations', {})
            avg_error = evaluations.get('avg_error')
            
            if avg_error is not None:
                # Adjust interval: higher error = shorter interval
                error_factor = 1.0 - min(1.0, avg_error * 2)  # 0 to 1 scale
                new_interval = min_interval + error_factor * (max_interval - min_interval)
                
                # Smooth adjustment
                current_interval = (current_interval * 0.7) + (new_interval * 0.3)
                self.logger.debug(f"Adjusted interval to {current_interval:.1f}s based on error {avg_error:.3f}")
            
            # Wait for next cycle
            cycle_duration = time.time() - cycle_start
            wait_time = max(0, current_interval - cycle_duration)
            
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        
        return {
            'success': True,
            'entity_id': entity_id,
            'cycles_completed': cycle_count,
            'total_runtime': time.time() - start_time
        }
        
    async def perform_maintenance(self):
        """Perform maintenance on the predictive processing system."""
        self.logger.info("Performing maintenance on predictive processing system")
        
        start_time = time.time()
        
        # Collect maintenance metrics
        metrics = {
            'active_entities': len(self.active_entities),
            'total_prediction_cycles': sum(self.prediction_cycles.values()),
            'components_maintained': 0
        }
        
        # Maintain predictive modeler
        if hasattr(self.coupling_layer, 'predictive_modeler'):
            # Clean up old predictions
            await self._cleanup_old_predictions()
            metrics['components_maintained'] += 1
        
        # Maintain active inference controller
        if hasattr(self.coupling_layer, 'active_inference'):
            # Clean up old tests
            await self._cleanup_old_tests()
            metrics['components_maintained'] += 1
        
        # Maintain counterfactual simulator
        if hasattr(self.coupling_layer, 'counterfactual_simulator'):
            # Clean up old simulations
            await self._cleanup_old_simulations()
            metrics['components_maintained'] += 1
        
        metrics['elapsed_time'] = time.time() - start_time
        
        return metrics
        
    async def _cleanup_old_predictions(self):
        """Clean up old predictions to free memory."""
        predictive_modeler = self.coupling_layer.predictive_modeler
        
        # Find old predictions to clean up
        current_time = time.time()
        expired_predictions = []
        
        for pred_id, prediction in predictive_modeler.predictions.items():
            # Clean verified predictions older than 1 hour
            if prediction.verification_time and (current_time - prediction.verification_time > 3600):
                expired_predictions.append(pred_id)
            
            # Clean unverified predictions older than 24 hours
            elif current_time - prediction.prediction_time > 86400:
                expired_predictions.append(pred_id)
        
        # Remove expired predictions
        for pred_id in expired_predictions:
            if pred_id in predictive_modeler.predictions:
                prediction = predictive_modeler.predictions[pred_id]
                # Remove from entity predictions
                if prediction.environmental_entity_id in predictive_modeler.entity_predictions:
                    if pred_id in predictive_modeler.entity_predictions[prediction.environmental_entity_id]:
                        predictive_modeler.entity_predictions[prediction.environmental_entity_id].remove(pred_id)
                # Remove from predictions
                del predictive_modeler.predictions[pred_id]
        
        self.logger.info(f"Cleaned up {len(expired_predictions)} expired predictions")
        
    async def _cleanup_old_tests(self):
        """Clean up old active inference tests."""
        active_inference = self.coupling_layer.active_inference
        
        # Implementation depends on active inference controller implementation
        # This is a placeholder for the cleanup logic
        cleaned_tests = 0
        current_time = time.time()
        
        self.logger.info(f"Cleaned up {cleaned_tests} old active inference tests")
        
    async def _cleanup_old_simulations(self):
        """Clean up old counterfactual simulations."""
        counterfactual_simulator = self.coupling_layer.counterfactual_simulator
        
        # Implementation depends on counterfactual simulator implementation
        # This is a placeholder for the cleanup logic
        cleaned_simulations = 0
        current_time = time.time()
        
        self.logger.info(f"Cleaned up {cleaned_simulations} old counterfactual simulations")
