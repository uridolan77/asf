import asyncio
import time
import uuid
import logging
import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

from asf.layer4_environmental_coupling.models import EnvironmentalPrediction, EnvironmentalCoupling

class PredictiveEnvironmentalModeler:
    """
    Implements Seth's controlled hallucination principle for environmental interactions.
    Generates predictions about environmental entities and updates based on actual observations.
    """
    def __init__(self):
        self.predictions = {}  # Maps prediction_id to EnvironmentalPrediction
        self.entity_predictions = defaultdict(list)  # Maps entity_id to prediction_ids
        self.prediction_errors = defaultdict(list)  # Maps entity_id to historical errors
        self.precision = {}  # Maps entity_id to precision (inverse error variance)
        self.logger = logging.getLogger("ASF.Layer4.PredictiveEnvironmentalModeler")
    
    async def predict_interaction(self, environmental_entity_id, context=None):
        """
        Generate a prediction about an upcoming interaction with an environmental entity.
        Implements Seth's controlled hallucination principle.
        """
        prediction_id = str(uuid.uuid4())
        context = context or {}
        
        # Generate prediction based on historical patterns
        predicted_data = await self._generate_prediction(environmental_entity_id, context)
        
        # Get confidence based on precision
        confidence = self._calculate_confidence(environmental_entity_id)
        
        # Create prediction object
        prediction = EnvironmentalPrediction(
            id=prediction_id,
            environmental_entity_id=environmental_entity_id,
            predicted_data=predicted_data,
            confidence=confidence,
            context=context,
            precision=self.precision.get(environmental_entity_id, 1.0)
        )
        
        # Store prediction
        self.predictions[prediction_id] = prediction
        self.entity_predictions[environmental_entity_id].append(prediction_id)
        
        # Clean up old predictions for this entity
        await self._cleanup_old_predictions(environmental_entity_id)
        
        self.logger.debug(f"Generated prediction {prediction_id} for entity {environmental_entity_id} with confidence {confidence:.2f}")
        return prediction
    
    async def evaluate_prediction(self, prediction_id, actual_data):
        """
        Compare prediction to actual data and update precision.
        Implements Seth's precision-weighted prediction error.
        """
        if prediction_id not in self.predictions:
            return None
            
        prediction = self.predictions[prediction_id]
        prediction.verification_time = time.time()
        
        # Calculate prediction error
        error = self._calculate_prediction_error(prediction.predicted_data, actual_data)
        prediction.prediction_error = error
        
        # Update error history
        entity_id = prediction.environmental_entity_id
        self.prediction_errors[entity_id].append(error)
        
        # Limit history size
        if len(self.prediction_errors[entity_id]) > 20:
            self.prediction_errors[entity_id] = self.prediction_errors[entity_id][-20:]
        
        # Update precision (inverse variance)
        if len(self.prediction_errors[entity_id]) > 1:
            variance = np.var(self.prediction_errors[entity_id])
            new_precision = 1.0 / (variance + 1e-6)  # Avoid division by zero
            self.precision[entity_id] = new_precision
            prediction.precision = new_precision
        
        self.logger.debug(f"Evaluated prediction {prediction_id} with error {error:.4f}, new precision {self.precision.get(entity_id, 1.0):.4f}")
        return {
            'prediction_id': prediction_id,
            'error': error,
            'precision': self.precision.get(entity_id, 1.0)
        }
    
    async def get_predictions_for_entity(self, environmental_entity_id, limit=5, future_only=True):
        """
        Retrieve recent predictions for an entity.
        If future_only is True, only return predictions that haven't been verified yet.
        """
        prediction_ids = self.entity_predictions.get(environmental_entity_id, [])
        
        # Get actual prediction objects
        predictions = []
        current_time = time.time()
        
        for pred_id in reversed(prediction_ids):  # Most recent first
            if pred_id in self.predictions:
                pred = self.predictions[pred_id]
                
                # Filter for future predictions if requested
                if future_only and pred.verification_time is not None:
                    continue
                    
                predictions.append(pred)
                
                if len(predictions) >= limit:
                    break
        
        return predictions
    
    async def _generate_prediction(self, environmental_entity_id, context):
        """
        Internal method to generate prediction based on historical patterns.
        In a real implementation, this would use sophisticated ML models.
        """
        # Extract context information that might be useful for prediction
        interaction_type = context.get('last_interaction_type')
        
        # Generate basic prediction structure
        predicted_data = {
            'predicted_interaction_type': await self._predict_interaction_type(environmental_entity_id, interaction_type),
            'predicted_content': await self._generate_content_prediction(environmental_entity_id),
            'predicted_timing': await self._predict_timing(environmental_entity_id),
            'predicted_response_time': random.uniform(0.1, 2.0),
            'predicted_confidence': random.uniform(0.5, 0.9)
        }
        
        return predicted_data
    
    async def _predict_interaction_type(self, entity_id, last_interaction_type=None):
        """
        Predict the type of interaction expected from this entity.
        Uses historical patterns where available.
        """
        # If we have prediction errors, check for patterns
        if entity_id in self.prediction_errors and len(self.prediction_errors[entity_id]) > 0:
            # This would be more sophisticated in a real implementation
            interaction_types = ['query', 'update', 'notification', 'request']
            
            # If we have last interaction type, use that as a signal
            if last_interaction_type:
                # Simple pattern: if last type was query, next might be update
                if last_interaction_type == 'query':
                    return 'update'
                elif last_interaction_type == 'update':
                    return 'notification'
                else:
                    return random.choice(interaction_types)
            else:
                return random.choice(interaction_types)
        else:
            # No history yet, return a default prediction
            return 'query'
    
    async def _generate_content_prediction(self, entity_id):
        """
        Generate predicted content for an interaction.
        This would use more sophisticated models in a real implementation.
        """
        # Simplified content prediction
        content_types = [
            {'type': 'text', 'expected_length': random.randint(10, 100)},
            {'type': 'structured', 'expected_fields': ['id', 'timestamp', 'value']},
            {'type': 'numeric', 'expected_range': [0, 100]}
        ]
        
        return random.choice(content_types)
    
    async def _predict_timing(self, entity_id):
        """
        Predict when the next interaction will occur.
        Uses timing patterns from history where available.
        """
        # Start with a default timing (30-60 seconds in the future)
        base_timing = time.time() + random.uniform(30, 60)
        
        # If we have prediction history, refine the timing prediction
        if entity_id in self.entity_predictions and len(self.entity_predictions[entity_id]) >= 2:
            # Get the last few predictions to analyze timing patterns
            recent_pred_ids = self.entity_predictions[entity_id][-5:]
            recent_predictions = [self.predictions[pid] for pid in recent_pred_ids if pid in self.predictions]
            
            if len(recent_predictions) >= 2:
                # Calculate average time between predictions
                times = [p.prediction_time for p in recent_predictions]
                intervals = [times[i] - times[i-1] for i in range(1, len(times))]
                avg_interval = sum(intervals) / len(intervals)
                
                # Predict next interaction based on average interval
                last_time = max(times)
                predicted_time = last_time + avg_interval
                
                # Add some randomness (±20%)
                predicted_time += avg_interval * random.uniform(-0.2, 0.2)
                
                return predicted_time
        
        return base_timing
    
    def _calculate_confidence(self, entity_id):
        """
        Calculate confidence level based on prediction history.
        Higher precision leads to higher confidence.
        """
        # Base confidence starts moderate
        base_confidence = 0.5
        
        # Adjust based on precision if available
        precision = self.precision.get(entity_id, 1.0)
        
        # Scale precision to confidence (0.3-0.9 range)
        # Higher precision = higher confidence
        precision_factor = min(0.4, (precision - 1.0) * 0.1) if precision > 1.0 else 0
        confidence = base_confidence + precision_factor
        
        # Cap confidence
        confidence = min(0.9, max(0.3, confidence))
        
        return confidence
    
    def _calculate_prediction_error(self, predicted_data, actual_data):
        """
        Calculate error between prediction and actual data.
        Returns a normalized error value between 0.0 (perfect) and 1.0 (completely wrong).
        """
        # For simplicity, we'll focus on a few key comparisons
        error_components = []
        
        # Check interaction type prediction if available
        if 'predicted_interaction_type' in predicted_data and 'interaction_type' in actual_data:
            type_match = predicted_data['predicted_interaction_type'] == actual_data['interaction_type']
            error_components.append(0.0 if type_match else 1.0)
        
        # Check content type prediction if available
        if 'predicted_content' in predicted_data and 'content_type' in actual_data:
            if 'type' in predicted_data['predicted_content']:
                content_match = predicted_data['predicted_content']['type'] == actual_data['content_type']
                error_components.append(0.0 if content_match else 1.0)
        
        # Check timing prediction if available (with a tolerance)
        if 'predicted_timing' in predicted_data and 'timestamp' in actual_data:
            timing_diff = abs(predicted_data['predicted_timing'] - actual_data['timestamp'])
            # Normalize timing error (0.0 for perfect, 1.0 for >60 seconds off)
            timing_error = min(1.0, timing_diff / 60.0)
            error_components.append(timing_error)
        
        # If we couldn't calculate any components, return a default error
        if not error_components:
            return 0.5
        
        # Return average error across all components
        return sum(error_components) / len(error_components)
    
    async def _cleanup_old_predictions(self, entity_id, max_age=3600, max_count=100):
        """Clean up old predictions to prevent memory issues."""
        if entity_id not in self.entity_predictions:
            return
            
        # Get prediction IDs for this entity
        prediction_ids = self.entity_predictions[entity_id]
        
        # If under max count, no need to clean up
        if len(prediction_ids) <= max_count:
            return
            
        current_time = time.time()
        predictions_to_remove = []
        
        # Identify old predictions to remove
        for pred_id in prediction_ids:
            if pred_id in self.predictions:
                pred = self.predictions[pred_id]
                
                # Remove verified predictions older than max_age
                if pred.verification_time and (current_time - pred.verification_time > max_age):
                    predictions_to_remove.append(pred_id)
                    
                # Remove old unverified predictions
                elif current_time - pred.prediction_time > max_age * 2:
                    predictions_to_remove.append(pred_id)
            else:
                # ID exists but prediction doesn't (should not happen)
                predictions_to_remove.append(pred_id)
                
        # Remove identified predictions
        for pred_id in predictions_to_remove:
            if pred_id in self.predictions:
                del self.predictions[pred_id]
            self.entity_predictions[entity_id].remove(pred_id)
