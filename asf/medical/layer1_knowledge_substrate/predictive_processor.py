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
    def integrate_prediction(self, prediction_id, integration_function, *args, **kwargs):
        """
        Integrates a prediction into the system using a provided function.

        This allows for flexible handling of predictions, which could involve
        updating knowledge, triggering actions, etc.

        Args:
            prediction_id: The ID of the prediction to integrate.
            integration_function: A callable that takes the predicted value
                                 and any other necessary arguments and performs
                                 the integration.
            *args, **kwargs:  Arguments to be passed to the integration_function.

        Returns:
            The result of the integration_function.
        """
        if prediction_id not in self.prediction_models:
            raise ValueError(f"Prediction with ID '{prediction_id}' not found.")

        prediction = self.prediction_models[prediction_id]
        predicted_value = prediction["value"]
        return integration_function(predicted_value, *args, **kwargs)

# Example usage
if __name__ == '__main__':
    processor = PredictiveProcessor()

    # Register some predictions
    processor.register_prediction("entity_1", "context_A", 25.5)
    processor.register_prediction("entity_1", "context_B", 18.2)
    processor.register_prediction("entity_2", "context_A", [1, 0, 1])
    processor.register_prediction("entity_2", "context_A", [1,0,0], metadata = {'source': 'sensor_X'})
    p_id = processor.register_prediction("entity_3", "context_C", "Sunny")

    print(f"Prediction id 4 = {p_id}")
    time.sleep(0.5)  # Simulate some time passing

    # Evaluate the predictions
    eval_result1 = processor.evaluate_prediction("entity_1", "context_A", 26.8) #Slightly off
    eval_result2 = processor.evaluate_prediction("entity_1", "context_B", 17.9) #Close
    eval_result3 = processor.evaluate_prediction("entity_2", "context_A", [1, 1, 1]) #Off
    eval_result4 = processor.evaluate_prediction("entity_3", "context_C", "Cloudy", prediction_id=p_id) #Use Prediction ID

    if eval_result1:
        print(f"\nEvaluation for entity_1, context_A: Error = {eval_result1['error']:.2f}, Precision = {eval_result1['precision']:.2f}, Learning Rate = {eval_result1['learning_rate']:.2f}")
    if eval_result2:
        print(f"Evaluation for entity_1, context_B: Error = {eval_result2['error']:.2f}, Precision = {eval_result2['precision']:.2f}, Learning Rate = {eval_result2['learning_rate']:.2f}")
    if eval_result3:
        print(f"Evaluation for entity_2, context_A: Error = {eval_result3['error']:.2f}, Precision = {eval_result3['precision']:.2f}, Learning Rate = {eval_result3['learning_rate']:.2f}")
    if eval_result4:
        print(f"Evaluation for entity_3, context_C: Error = {eval_result4['error']:.2f}, Precision = {eval_result4['precision']:.2f}, Learning Rate = {eval_result4['learning_rate']:.2f}")

    # Get precision weights
    print(f"\nPrecision Weight for entity_1: {processor.get_precision_weight('entity_1'):.2f}")
    print(f"Precision Weight for entity_2: {processor.get_precision_weight('entity_2'):.2f}")
    print(f"Precision Weight for entity_3: {processor.get_precision_weight('entity_3'):.2f}")

    # Get learning rates
    print(f"\nLearning Rate for entity_1: {processor.get_learning_rate('entity_1'):.2f}")
    print(f"Learning Rate for entity_2: {processor.get_learning_rate('entity_2'):.2f}")
    print(f"Learning Rate for entity_3: {processor.get_learning_rate('entity_3'):.2f}")


    #Get Statistics:
    print(f"\nOverall Statistics: {processor.get_prediction_statistics()}")
    print(f"Statistics for entity_1: {processor.get_prediction_statistics(entity_id='entity_1')}")

    # Example of combining predictions with precision weighting
    predictions = {
      "source_A": (25, 0.8),   # Value 25, base confidence 0.8
      "source_B": (28, 0.9),   # Value 28, base confidence 0.9
    }
    processor.precision_weights['source_A'] = 0.5 #Pretend source_A has low precision
    processor.precision_weights['source_B'] = 5.0 #Pretend source_B has high precision.

    combined_value, combined_confidence = processor.predict_with_precision(predictions)
    print(f"\nCombined Prediction: Value = {combined_value}, Confidence = {combined_confidence:.2f}")

    predictions = {
        "source_A": ([1,0,1], 0.8),  # Vector prediction
        "source_B": ([1,1,1], 0.9),
    }
    combined_value, combined_confidence = processor.predict_with_precision(predictions, entity_id="entity_2")
    print(f"\nCombined Vector Prediction: Value = {combined_value}, Confidence = {combined_confidence:.2f}")

    # --- Example of integrating a prediction ---
    #Dummy Integration function:
    def example_integration_function(predicted_value, knowledge_base, entity_id):
        print(f"Integrating prediction: {predicted_value} for entity {entity_id} into knowledge base.")
        # In a real system, this function would update the knowledge base
        # based on the prediction.  This could involve:
        # - Creating new knowledge entities
        # - Updating existing knowledge entities
        # - Adding relationships between entities
        # - Triggering actions or events
        #For this example, we will pretend that it modifies the content of an entity.
        current_knowledge = knowledge_base.get_knowledge(entity_id)
        if current_knowledge:
          new_content = f"Predicted Value: {predicted_value}"
          current_knowledge['content']['predicted'] = new_content
          knowledge_base.update_knowledge(entity_id, current_knowledge)
          return f"Updated knowledge for entity {entity_id}."
        return f"Entity {entity_id} Not Found"

    # Create a mock KnowledgeBase (as defined in previous examples)
    knowledge_base = MockKnowledgeBase()
    knowledge_base.add_knowledge("entity_1", {"temperature": 20})

    #Register the Prediction:
    pred_id = processor.register_prediction('entity_1', 'context_D', 30.2)

    # Later, integrate the prediction
    result = processor.integrate_prediction(pred_id, example_integration_function, knowledge_base, "entity_1")
    print(f"\nIntegration Result: {result}")
    print(knowledge_base.get_knowledge("entity_1"))

    def register_prediction(self, entity_id, context_id, predicted_value, metadata=None):
        """
        Register a prediction.  predicted_value is now expected to be a tuple: (mean, variance)
        for probabilistic predictions.  It can still be a single value for backwards compatibility.
        """
        prediction_id = f"{entity_id}_{context_id}_{int(time.time()*1000)}"

        self.prediction_models[prediction_id] = {
            "entity_id": entity_id,
            "context_id": context_id,
            "value": predicted_value,  # Now stores (mean, variance)
            "timestamp": time.time(),
            "evaluated": False,
            "metadata": metadata or {}
        }
        self.stats["predictions_made"] += 1
        return prediction_id

    def _calculate_prediction_error(self, predicted, actual):
        """
        Calculate prediction error, handling probabilistic predictions.
        """
        if isinstance(predicted, tuple) and len(predicted) == 2:
            # Probabilistic prediction (mean, logvar)
            mean, logvar = predicted
            if isinstance(mean, torch.Tensor):
                mean = mean.detach().cpu().numpy()
            if isinstance(logvar, torch.Tensor):
                logvar = logvar.detach().cpu().numpy()

            variance = np.exp(logvar)
            # Calculate negative log-likelihood (NLL) as the error
            actual = np.array(actual)  # Ensure actual is a numpy array
            error = 0.5 * (np.log(2 * np.pi * variance) + (actual - mean)**2 / variance)
            return np.mean(error)  # Average NLL

        else:
            # Deterministic prediction (use existing logic)
            # ... (rest of the existing _calculate_prediction_error logic) ...
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


    def predict_with_precision(self, predictions, entity_id=None):
      """
      Combine predictions, weighted by precision. Now handles distributions.
      """
      if not predictions:
          return None, 0.0

      if len(predictions) == 1:
          source_id, prediction = next(iter(predictions.items()))
          #If it's a tuple, it is already mean, variance
          if isinstance(prediction, tuple) and len(prediction) == 2:
            mean, variance = prediction
            confidence = 1.0 / (1.0 + np.mean(np.sqrt(variance))) # Use standard deviation
            return mean, confidence
          else: #Handle old style prediction
            value, confidence = prediction
            return value, confidence


      # Multiple predictions: combine, considering uncertainty
      weighted_means = []
      total_weight = 0.0
      variances = []  # Collect variances for combined uncertainty

      for source_id, prediction in predictions.items():
        precision = self.precision_weights.get(source_id, self.default_precision)

        if isinstance(prediction, tuple) and len(prediction) == 2: #mean, variance
          mean, variance = prediction
          if isinstance(mean, (list, np.ndarray)): #If mean is array-like
            mean = np.array(mean) #Convert to numpy array
          confidence = 1.0 / (1.0 + np.mean(np.sqrt(variance))) #Use average standard deviation.
        else:
          # Handle as a point prediction with given confidence
          mean, confidence = prediction
          variance = (1.0 - confidence)**2  # Estimate variance from confidence

        weight = confidence * precision
        weighted_means.append((mean, weight))
        variances.append(variance) #Keep track of variance
        total_weight += weight

      if total_weight == 0.0:
          return None, 0.0

      # --- Combine Predictions ---
      # Check if means are numeric or array-like
      if isinstance(weighted_means[0][0], (int, float, np.number)):
          # Combine scalar predictions using weighted average
          combined_mean = sum(mean * weight for mean, weight in weighted_means) / total_weight

          # Combine variances (simplified - assumes independence)
          combined_variance = sum(var / total_weight for var in variances)
          combined_confidence = 1.0 / (1.0 + np.sqrt(combined_variance))
          return combined_mean, combined_confidence

      elif isinstance(weighted_means[0][0], (list, np.ndarray)):
          # Combine array-like predictions (if shapes are compatible)
          try:
            arrays = [np.array(mean) * weight for mean, weight in weighted_means]
            if all(arr.shape == arrays[0].shape for arr in arrays):  #Check shapes
              combined_mean = sum(arrays) / total_weight

              # Combine variances (simplified - assumes independence)
              combined_variance = sum(np.array(var) / total_weight for var in variances)
              combined_confidence = 1.0 / (1.0 + np.mean(np.sqrt(combined_variance)))
              return combined_mean.tolist(), combined_confidence  # Convert back to list
          except:
              pass  # Fallback if shapes are incompatible

      # Fallback: Return prediction with highest weight if combination fails
      best_prediction = max(weighted_means, key=lambda x: x[1])
      best_mean = best_prediction[0]
      best_confidence = best_prediction[1] / total_weight

      # Extract predicted variance if available, otherwise estimate from confidence.
      if isinstance(best_mean, tuple) and len(best_mean) == 2:
        best_confidence =  1.0 / (1.0 + np.mean(np.sqrt(best_mean[1])))
        best_mean = best_mean[0]

      return best_mean, min(1.0, best_confidence) #Return the best mean.

# --- TemporalKnowledgeManager (Modified) ---
# No changes *needed* here, but you'd likely want to update the
# example usage to use the probabilistic predictions.  The key is
# that predictions are now *tuples* of (mean, variance).