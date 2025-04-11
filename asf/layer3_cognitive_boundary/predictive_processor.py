import time
import numpy as np
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict

class PredictiveProcessor:
    """
    Central coordinator for predictive processing across Layer 3.
    Implements Seth's core insight that cognition is "controlled hallucination."
    
    This service:
    1. Manages predictions across all components
    2. Tracks prediction errors and precision for uncertainty estimation
    3. Provides adaptive learning rates based on precision
    4. Supports active inference for minimizing prediction errors
    """
    def __init__(self):
        self.predictions = {}  # Prediction ID -> prediction data
        self.prediction_errors = defaultdict(list)  # Entity ID -> errors
        self.precision_values = {}  # Entity ID -> precision value
        self.learning_rates = {}  # Entity ID -> adaptive learning rate
        self.context_history = {}  # Context ID -> previous states
        
    async def register_prediction(self, context_id: str, entity_id: str, 
                                prediction_type: str, prediction_value: Any) -> str:
        if prediction_id not in self.predictions:
            return None
            
        prediction = self.predictions[prediction_id]
        if prediction['evaluated']:
            return None
            
        predicted_value = prediction['value']
        error = self._calculate_error(predicted_value, actual_value)
        
        prediction['evaluated'] = True
        prediction['actual_value'] = actual_value
        prediction['error'] = error
        prediction['evaluation_time'] = time.time()
        
        entity_id = prediction['entity_id']
        self.prediction_errors[entity_id].append(error)
        
        if len(self.prediction_errors[entity_id]) > 20:
            self.prediction_errors[entity_id] = self.prediction_errors[entity_id][-20:]
        
        if len(self.prediction_errors[entity_id]) > 1:
            variance = np.var(self.prediction_errors[entity_id])
            precision = 1.0 / (variance + 1e-6)  # Avoid division by zero
            self.precision_values[entity_id] = min(10.0, precision)  # Cap very high precision
            
        precision = self.get_precision(entity_id)
        base_rate = min(0.8, error * 2)  # Error-proportional component
        precision_factor = max(0.1, min(0.9, 1.0 / (1.0 + precision * 0.2)))
        learning_rate = min(0.9, max(0.1, base_rate * precision_factor))
        self.learning_rates[entity_id] = learning_rate
        
        return {
            'prediction_id': prediction_id,
            'error': error,
            'precision': precision,
            'learning_rate': learning_rate
        }
        
    def get_precision(self, entity_id: str) -> float:
        """Get precision (inverse variance) for a specific entity"""
        return self.precision_values.get(entity_id, 1.0)
        
    def get_learning_rate(self, entity_id: str) -> float:
        """Get adaptive learning rate for an entity"""
        return self.learning_rates.get(entity_id, 0.3)
        
    def _calculate_error(self, predicted: Any, actual: Any) -> float:
        """Calculate normalized error between predicted and actual values"""
        if isinstance(predicted, (int, float, np.number)) and isinstance(actual, (int, float, np.number)):
            # For numeric values, normalized absolute difference
            return abs(predicted - actual) / (1.0 + abs(actual))
            
        elif isinstance(predicted, (list, np.ndarray)) and isinstance(actual, (list, np.ndarray)):
            # For vectors, normalized Euclidean distance
            try:
                predicted_arr = np.array(predicted)
                actual_arr = np.array(actual)
                
                if predicted_arr.shape != actual_arr.shape:
                    return 1.0  # Maximum error for shape mismatch
                    
                if predicted_arr.size == 0 or actual_arr.size == 0:
                    return 1.0  # Maximum error for empty arrays
                    
                # Normalized Euclidean distance
                diff = np.linalg.norm(predicted_arr - actual_arr)
                norm = np.linalg.norm(actual_arr)
                return min(1.0, diff / (1.0 + norm))
            except:
                return 1.0  # Maximum error if calculation fails
            
        elif isinstance(predicted, dict) and isinstance(actual, dict):
            # For dictionaries, calculate average error across shared keys
            shared_keys = set(predicted.keys()) & set(actual.keys())
            
            if not shared_keys:
                return 1.0  # Maximum error if no shared keys
                
            errors = []
            for key in shared_keys:
                errors.append(self._calculate_error(predicted[key], actual[key]))
                
            return sum(errors) / len(errors)
            
        else:
            # Fallback for other types
            return 1.0 if predicted != actual else 0.0
    
    def perform_active_inference(self, entity_id: str, current_state: Dict[str, Any], 
                                possible_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select action that minimizes expected prediction error (Seth's active inference)
        
        Args:
            entity_id: Entity performing the inference
            current_state: Current state of the entity
            possible_actions: List of possible actions to take
            
        Returns:
            Selected action that minimizes expected prediction error