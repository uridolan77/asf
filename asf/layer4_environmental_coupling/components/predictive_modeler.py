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
        prediction_id = str(uuid.uuid4())
        context = context or {}
        
        predicted_data = await self._generate_prediction(environmental_entity_id, context)
        
        confidence = self._calculate_confidence(environmental_entity_id)
        
        prediction = EnvironmentalPrediction(
            id=prediction_id,
            environmental_entity_id=environmental_entity_id,
            predicted_data=predicted_data,
            confidence=confidence,
            context=context,
            precision=self.precision.get(environmental_entity_id, 1.0)
        )
        
        self.predictions[prediction_id] = prediction
        self.entity_predictions[environmental_entity_id].append(prediction_id)
        
        await self._cleanup_old_predictions(environmental_entity_id)
        
        self.logger.debug(f"Generated prediction {prediction_id} for entity {environmental_entity_id} with confidence {confidence:.2f}")
        return prediction
    
    async def evaluate_prediction(self, prediction_id, actual_data):
        if prediction_id not in self.predictions:
            return None
            
        prediction = self.predictions[prediction_id]
        prediction.verification_time = time.time()
        
        error = self._calculate_prediction_error(prediction.predicted_data, actual_data)
        prediction.prediction_error = error
        
        entity_id = prediction.environmental_entity_id
        self.prediction_errors[entity_id].append(error)
        
        if len(self.prediction_errors[entity_id]) > 20:
            self.prediction_errors[entity_id] = self.prediction_errors[entity_id][-20:]
        
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
        prediction_ids = self.entity_predictions.get(environmental_entity_id, [])
        
        predictions = []
        current_time = time.time()
        
        for pred_id in reversed(prediction_ids):  # Most recent first
            if pred_id in self.predictions:
                pred = self.predictions[pred_id]
                
                if future_only and pred.verification_time is not None:
                    continue
                    
                predictions.append(pred)
                
                if len(predictions) >= limit:
                    break
        
        return predictions
    
    async def _generate_prediction(self, environmental_entity_id, context):
        interaction_type = context.get('last_interaction_type')
        
        predicted_data = {
            'predicted_interaction_type': await self._predict_interaction_type(environmental_entity_id, interaction_type),
            'predicted_content': await self._generate_content_prediction(environmental_entity_id),
            'predicted_timing': await self._predict_timing(environmental_entity_id),
            'predicted_response_time': random.uniform(0.1, 2.0),
            'predicted_confidence': random.uniform(0.5, 0.9)
        }
        
        return predicted_data
    
    async def _predict_interaction_type(self, entity_id, last_interaction_type=None):
        if entity_id in self.prediction_errors and len(self.prediction_errors[entity_id]) > 0:
            interaction_types = ['query', 'update', 'notification', 'request']
            
            if last_interaction_type:
                if last_interaction_type == 'query':
                    return 'update'
                elif last_interaction_type == 'update':
                    return 'notification'
                else:
                    return random.choice(interaction_types)
            else:
                return random.choice(interaction_types)
        else:
            return 'query'
    
    async def _generate_content_prediction(self, entity_id):
        content_types = [
            {'type': 'text', 'expected_length': random.randint(10, 100)},
            {'type': 'structured', 'expected_fields': ['id', 'timestamp', 'value']},
            {'type': 'numeric', 'expected_range': [0, 100]}
        ]
        
        return random.choice(content_types)
    
    async def _predict_timing(self, entity_id):
        base_timing = time.time() + random.uniform(30, 60)
        
        if entity_id in self.entity_predictions and len(self.entity_predictions[entity_id]) >= 2:
            recent_pred_ids = self.entity_predictions[entity_id][-5:]
            recent_predictions = [self.predictions[pid] for pid in recent_pred_ids if pid in self.predictions]
            
            if len(recent_predictions) >= 2:
                times = [p.prediction_time for p in recent_predictions]
                intervals = [times[i] - times[i-1] for i in range(1, len(times))]
                avg_interval = sum(intervals) / len(intervals)
                
                last_time = max(times)
                predicted_time = last_time + avg_interval
                
                predicted_time += avg_interval * random.uniform(-0.2, 0.2)
                
                return predicted_time
        
        return base_timing
    
    def _calculate_confidence(self, entity_id):
        """
        Calculate confidence level based on prediction history.
        Higher precision leads to higher confidence.
        """
        base_confidence = 0.5
        
        precision = self.precision.get(entity_id, 1.0)
        
        precision_factor = min(0.4, (precision - 1.0) * 0.1) if precision > 1.0 else 0
        confidence = base_confidence + precision_factor
        
        confidence = min(0.9, max(0.3, confidence))
        
        return confidence
    
    def _calculate_prediction_error(self, predicted_data, actual_data):
        """
        Calculate error between prediction and actual data.
        Returns a normalized error value between 0.0 (perfect) and 1.0 (completely wrong).
        """
        error_components = []
        
        if 'predicted_interaction_type' in predicted_data and 'interaction_type' in actual_data:
            type_match = predicted_data['predicted_interaction_type'] == actual_data['interaction_type']
            error_components.append(0.0 if type_match else 1.0)
        
        if 'predicted_content' in predicted_data and 'content_type' in actual_data:
            if 'type' in predicted_data['predicted_content']:
                content_match = predicted_data['predicted_content']['type'] == actual_data['content_type']
                error_components.append(0.0 if content_match else 1.0)
        
        if 'predicted_timing' in predicted_data and 'timestamp' in actual_data:
            timing_diff = abs(predicted_data['predicted_timing'] - actual_data['timestamp'])
            timing_error = min(1.0, timing_diff / 60.0)
            error_components.append(timing_error)
        
        if not error_components:
            return 0.5
        
        return sum(error_components) / len(error_components)
    
    async def _cleanup_old_predictions(self, entity_id, max_age=3600, max_count=100):