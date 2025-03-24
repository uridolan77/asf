import time
import numpy as np
from collections import defaultdict
import torch
import logging

logger = logging.getLogger(__name__)

class PredictiveProcessor:
    """
    Implements Seth's predictive processing principles across the ASF system.
    
    Manages the balance between predictions and data-driven updates,
    tracking prediction errors and precision weighting.
    
    This is a cross-cutting component that coordinates predictive processing
    across all parts of Layer 1.
    
    Philosophical influences: Seth's Predictive Processing, Friston's Free Energy
    """
    
    def __init__(self):
        # Store predictions for entities and contexts
        self.prediction_models = {}
        
        # Track prediction errors for assessing precision
        self.prediction_errors = defaultdict(list)
        
        # Precision values (inverse variance of prediction errors)
        self.precision_weights = {}
        
        # Learning rates (adaptive based on prediction error and precision)
        self.learning_rates = {}
        
        # Surprise history for monitoring
        self.surprise_history = []
        
        # Default parameters
        self.default_precision = 1.0
        self.min_learning_rate = 0.1
        self.max_learning_rate = 0.9
        
        # Maximum history length to prevent unbounded memory growth
        self.max_history_length = 50
        
        # Statistics
        self.stats = {
            "predictions_made": 0,
            "predictions_evaluated": 0,
            "avg_error": 0.0,
            "avg_precision": 1.0
        }
        
    def register_prediction(self, entity_id, context_id, predicted_value, metadata=None):
        """
        Register a prediction for later evaluation
        
        Args:
            entity_id: Identifier for entity being predicted
            context_id: Identifier for context in which prediction is made
            predicted_value: The predicted value
            metadata: Optional additional information about the prediction
            
        Returns:
            prediction_id: Unique identifier for this prediction
        """
        # Create unique ID for this prediction
        prediction_id = f"{entity_id}_{context_id}_{int(time.time()*1000)}"
        
        # Store prediction
        self.prediction_models[prediction_id] = {
            "entity_id": entity_id,
            "context_id": context_id,
            "value": predicted_value,
            "timestamp": time.time(),
            "evaluated": False,
            "metadata": metadata or {}
        }
        
        # Update statistics
        self.stats["predictions_made"] += 1
        
        return prediction_id
    
    def evaluate_prediction(self, entity_id, context_id, actual_value, prediction_id=None):
        """
        Evaluate prediction against actual value and update precision
        
        Args:
            entity_id: Entity identifier
            context_id: Context identifier
            actual_value: Observed value to compare against prediction
            prediction_id: Optional specific prediction to evaluate
            
        Returns:
            Dictionary with evaluation results
        """
        # Find matching prediction
        if prediction_id is not None and prediction_id in self.prediction_models:
            prediction = self.prediction_models[prediction_id]
        else:
            # Find most recent unevaluated prediction for this entity/context
            matching_predictions = [
                (pid, p) for pid, p in self.prediction_models.items()
                if p["entity_id"] == entity_id and p["context_id"] == context_id and not p["evaluated"]
            ]
            
            if not matching_predictions:
                return None  # No matching prediction found
                
            # Sort by timestamp (most recent first)
            matching_predictions.sort(key=lambda x: x[1]["timestamp"], reverse=True)
            prediction_id, prediction = matching_predictions[0]
        
        # Skip if already evaluated
        if prediction["evaluated"]:
            return None
        
        # Calculate prediction error
        predicted_value = prediction["value"]
        error = self._calculate_prediction_error(predicted_value, actual_value)
        
        # Track error for this entity
        self.prediction_errors[entity_id].append(error)
        
        # Trim history if needed
        if len(self.prediction_errors[entity_id]) > self.max_history_length:
            self.prediction_errors[entity_id] = self.prediction_errors[entity_id][-self.max_history_length:]
        
        # Update precision (inverse variance of prediction errors)
        if len(self.prediction_errors[entity_id]) > 1:
            variance = np.var(self.prediction_errors[entity_id])
            precision = 1.0 / (variance + 1e-6)  # Avoid division by zero
            
            # Cap precision to reasonable range
            precision = min(10.0, precision)
            
            # Store precision
            self.precision_weights[entity_id] = precision
        else:
            # Default precision for first observation
            precision = self.default_precision
            self.precision_weights[entity_id] = precision
        
        # Calculate adaptive learning rate based on prediction error and precision
        # Higher error -> higher learning rate (learn more from surprising events)
        # Higher precision -> lower learning rate (already well-predicted)
        base_learning_rate = min(0.8, error * 2)  # Error-proportional component
        precision_factor = max(0.1, min(0.9, 1.0 / (1.0 + precision * 0.2)))  # Precision-based modifier
        
        learning_rate = min(
            self.max_learning_rate,
            max(self.min_learning_rate, base_learning_rate * precision_factor)
        )
        
        # Store learning rate
        self.learning_rates[entity_id] = learning_rate
        
        # Mark prediction as evaluated
        prediction["evaluated"] = True
        prediction["error"] = error
        prediction["precision"] = precision
        prediction["learning_rate"] = learning_rate
        prediction["evaluation_time"] = time.time()
        
        # Update statistics
        self.stats["predictions_evaluated"] += 1
        total_evaluated = self.stats["predictions_evaluated"]
        self.stats["avg_error"] = (self.stats["avg_error"] * (total_evaluated - 1) + error) / total_evaluated
        self.stats["avg_precision"] = (self.stats["avg_precision"] * (total_evaluated - 1) + precision) / total_evaluated
        
        # Add to surprise history
        self.surprise_history.append({
            "entity_id": entity_id,
            "context_id": context_id,
            "error": error,
            "precision": precision,
            "timestamp": time.time()
        })
        
        # Trim surprise history
        if len(self.surprise_history) > self.max_history_length:
            self.surprise_history = self.surprise_history[-self.max_history_length:]
        
        # Return evaluation results
        return {
            "prediction_id": prediction_id,
            "error": error,
            "precision": precision,
            "learning_rate": learning_rate
        }
    
    def get_precision_weight(self, entity_id):
        """
        Get current precision weight for entity
        
        Higher precision = more predictable entity = higher weight
        """
        return self.precision_weights.get(entity_id, self.default_precision)
    
    def get_learning_rate(self, entity_id):
        """
        Get adaptive learning rate for entity
        
        This balances learning speed based on prediction errors and precision
        """
        return self.learning_rates.get(entity_id, 0.3)  # Default learning rate
    
    def _calculate_prediction_error(self, predicted, actual):
        """
        Calculate normalized prediction error between predicted and actual values
        
        Handles different data types appropriately
        """
        # Handle different value types
        if isinstance(predicted, (int, float, np.number)) and isinstance(actual, (int, float, np.number)):
            # For numeric values, normalized absolute difference
            return abs(predicted - actual) / (1.0 + abs(actual))
        
        elif isinstance(predicted, (list, np.ndarray)) and isinstance(actual, (list, np.ndarray)):
            # For vectors, cosine distance or normalized Euclidean distance
            predicted_arr = np.array(predicted)
            actual_arr = np.array(actual)
            
            if predicted_arr.shape != actual_arr.shape:
                return 1.0  # Maximum error for shape mismatch
                
            if predicted_arr.size == 0 or actual_arr.size == 0:
                return 1.0  # Maximum error for empty arrays
                
            # Try cosine similarity if vectors are non-zero
            pred_norm = np.linalg.norm(predicted_arr)
            actual_norm = np.linalg.norm(actual_arr)
            
            if pred_norm > 0 and actual_norm > 0:
                similarity = np.dot(predicted_arr, actual_arr) / (pred_norm * actual_norm)
                return max(0, 1.0 - similarity)  # Convert to distance
            else:
                # Fall back to normalized Euclidean for zero vectors
                diff = np.linalg.norm(predicted_arr - actual_arr)
                return min(1.0, diff / (1.0 + actual_norm))
        
        elif isinstance(predicted, dict) and isinstance(actual, dict):
            # For dictionaries, calculate average error across shared keys
            shared_keys = set(predicted.keys()) & set(actual.keys())
            
            if not shared_keys:
                return 1.0  # Maximum error if no shared keys
                
            errors = []
            for key in shared_keys:
                errors.append(self._calculate_prediction_error(predicted[key], actual[key]))
                
            return sum(errors) / len(errors)
        
        elif isinstance(predicted, bool) and isinstance(actual, bool):
            # For booleans, 0 for match, 1 for mismatch
            return 0.0 if predicted == actual else 1.0
        
        elif isinstance(predicted, str) and isinstance(actual, str):
            # For strings, Levenshtein distance would be ideal
            # For simplicity, use 0 for exact match, 1 for completely different
            if predicted == actual:
                return 0.0
            
            # Simple heuristic: fraction of length difference
            len_diff = abs(len(predicted) - len(actual))
            max_len = max(len(predicted), len(actual))
            
            if max_len == 0:
                return 0.0 if len_diff == 0 else 1.0
                
            return min(1.0, len_diff / max_len)
        
        else:
            # Fallback for other types
            return 1.0 if predicted != actual else 0.0
    
    def get_prediction_statistics(self, entity_id=None, time_window=None):
        """
        Get prediction statistics overall or for specific entity
        
        Args:
            entity_id: Optional entity to filter statistics for
            time_window: Optional time window in seconds
            
        Returns:
            Dictionary of prediction statistics
        """
        if entity_id is None:
            # Overall statistics
            return dict(self.stats)
        
        # Entity-specific statistics
        errors = self.prediction_errors.get(entity_id, [])
        
        if time_window is not None:
            # Filter to recent history
            cutoff_time = time.time() - time_window
            relevant_history = [
                entry for entry in self.surprise_history
                if entry["entity_id"] == entity_id and entry["timestamp"] >= cutoff_time
            ]
        else:
            # Use all history for this entity
            relevant_history = [
                entry for entry in self.surprise_history
                if entry["entity_id"] == entity_id
            ]
        
        if not relevant_history:
            return {
                "entity_id": entity_id,
                "avg_error": None,
                "precision": self.precision_weights.get(entity_id, None),
                "learning_rate": self.learning_rates.get(entity_id, None),
                "prediction_count": 0
            }
            
        # Calculate statistics
        avg_error = sum(entry["error"] for entry in relevant_history) / len(relevant_history)
        
        return {
            "entity_id": entity_id,
            "avg_error": avg_error,
            "precision": self.precision_weights.get(entity_id, None),
            "learning_rate": self.learning_rates.get(entity_id, None),
            "prediction_count": len(relevant_history)
        }
    
    def predict_with_precision(self, predictions, entity_id=None):
        """
        Adjust predictions based on precision weights
        
        For multiple predictions, weight them by precision and combine
        
        Args:
            predictions: Dictionary of predictions from different sources
                         {source_id: (predicted_value, base_confidence)}
            entity_id: Optional entity ID for precision lookup
            
        Returns:
            Final prediction and confidence
        """
        if not predictions:
            return None, 0.0
            
        if len(predictions) == 1:
            # Only one prediction source, just return it
            source_id, (value, confidence) = next(iter(predictions.items()))
            return value, confidence
            
        # Multiple prediction sources, combine with precision weighting
        weighted_values = []
        total_weight = 0.0
        
        for source_id, (value, confidence) in predictions.items():
            # Get precision for this source
            precision = self.precision_weights.get(source_id, self.default_precision)
            
            # Weight = confidence * precision
            weight = confidence * precision
            weighted_values.append((value, weight))
            total_weight += weight
            
        if total_weight == 0:
            # No weights, return first prediction
            return next(iter(predictions.values()))[0], 0.0
            
        # For numeric values, calculate weighted average
        first_value = next(iter(predictions.values()))[0]
        
        if isinstance(first_value, (int, float, np.number)):
            # Weighted average for numbers
            weighted_sum = sum(value * weight for value, weight in weighted_values)
            combined_value = weighted_sum / total_weight
            combined_confidence = min(1.0, total_weight / len(predictions))
            return combined_value, combined_confidence
            
        elif isinstance(first_value, (list, np.ndarray)) and all(
            isinstance(v[0], (list, np.ndarray)) for v in weighted_values
        ):
            # For vectors, try weighted average if shapes match
            try:
                arrays = [np.array(v) * w for v, w in weighted_values]
                if all(a.shape == arrays[0].shape for a in arrays):
                    combined_array = sum(arrays) / total_weight
                    combined_confidence = min(1.0, total_weight / len(predictions))
                    return combined_array.tolist(), combined_confidence
            except:
                # Fall back to highest weighted prediction
                pass
        
        # For other types or if vector combination fails, return highest weighted prediction
        best_prediction = max(weighted_values, key=lambda x: x[1])
        best_confidence = best_prediction[1] / total_weight
        return best_prediction[0], min(1.0, best_confidence)
