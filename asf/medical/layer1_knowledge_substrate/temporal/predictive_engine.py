import time
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from asf.layer1_knowledge_substrate.temporal.processing_engine import TemporalProcessingEngine


class TemporalPredictor(nn.Module):
    """Neural network model for temporal prediction"""

    def __init__(self, input_size=64, hidden_size=128):
        super().__init__()

        # RNN for sequence modeling
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )

        # Prediction layers
        self.interval_predictor = nn.Linear(hidden_size * 2, 5)  # Predict next 5 intervals
        self.confidence_predictor = nn.Linear(hidden_size * 2, 5)  # Confidence for each prediction

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor [batch_size, seq_length, input_size]

        Returns:
            predicted_intervals: [batch_size, 5]
            confidence: [batch_size, 5]
        """
        # Process sequence
        output, hidden = self.rnn(x)

        # Use last hidden state
        last_hidden = output[:, -1, :]

        # Predict intervals
        intervals = self.interval_predictor(last_hidden)

        # Ensure intervals are positive
        intervals = torch.abs(intervals)

        # Predict confidence
        confidence = torch.sigmoid(self.confidence_predictor(last_hidden))

        return intervals, confidence

class PredictiveTemporalEngine(TemporalProcessingEngine):
    """
    Enhanced temporal processing engine incorporating Seth's predictive processing principles.

    Extends the base TemporalProcessingEngine with:
    - Explicit prediction of future events
    - Prediction error tracking
    - Precision-weighted updates
    - Counterfactual temporal simulations
    - Neural network based prediction (optional)

    Philosophical influences: Seth's predictive processing, Friston's Free Energy
    """

    def __init__(self, input_size=1, hidden_size=64, use_neural_prediction=False):
        super().__init__()
        # Prediction errors
        self.prediction_errors = defaultdict(list)
        # Predictions
        self.predictions = {}
        # Actual prediction outcomes
        self.prediction_outcomes = []
        # Precision (inverse variance of prediction errors)
        self.precision_values = {}
        # Learning rates for adjustments
        self.learning_rates = defaultdict(lambda: 0.3)  # Default
        # Counterfactual simulations
        self.counterfactual_simulations = []

        # Neural network prediction (optional)
        self.use_neural_prediction = use_neural_prediction
        if self.use_neural_prediction:
            self.predictor = TemporalPredictor(input_size=input_size, hidden_size=hidden_size)

    def predict_next_events(self, entity_id, sequence_type, time_horizon=3600):
        """Predicts future events using either statistical or neural methods.

        Args:
            entity_id (str): ID of the entity.
            sequence_type (str): Type of the event sequence.
            time_horizon (int): Prediction horizon in seconds.

        Returns:
            list: Predicted events with timestamps and confidence.
        """
        if entity_id not in self.sequences or sequence_type not in self.sequences[entity_id]:
            return []

        sequence = self.sequences[entity_id][sequence_type]
        recent_events = sequence.get_events_in_window(window_size=time_horizon)

        if len(recent_events) < 3:  # Minimum events for prediction
            return []

        timestamps = list(sequence.timestamps)
        if len(timestamps) < 2:  # Need at least 2 timestamps to calculate intervals
            return []


        # --- Neural Network Prediction (if enabled) ---
        if self.use_neural_prediction and len(recent_events) >= 5 :
          neural_predictions = self._neural_predict(entity_id, sequence_type, time_horizon)
          if neural_predictions:
            # Store the prediction for later evaluation
            prediction_id = f"{entity_id}_{sequence_type}_{int(time.time())}"
            self.predictions[prediction_id] = {
                "entity_id": entity_id,
                "sequence_type": sequence_type,
                "predictions": neural_predictions,
                "timestamp": time.time(),
                "method": "neural"
            }
            return neural_predictions


        # --- Statistical Prediction (fallback) ---

        # Calculate event intervals
        event_intervals = np.diff(timestamps)
        if len(event_intervals) == 0:
          return []
        mean_interval = np.mean(event_intervals)
        interval_variance = np.var(event_intervals)

        # Calculate precision (inverse variance, capped)
        precision = min(10.0, 1.0 / (interval_variance + 1e-6))

        # Store the precision
        self.precision_values.setdefault(entity_id, {})[sequence_type] = precision

        last_time = timestamps[-1]
        current_time = time.time()
        next_time = last_time + mean_interval
        predictions = []

        while next_time <= current_time + time_horizon:
            confidence = min(0.9, np.tanh(precision) * 0.8)  # Confidence based on precision
            predictions.append({
                "predicted_time": next_time,
                "confidence": confidence,
                "time_from_now": next_time-current_time
            })
            next_time += mean_interval

        # Store the prediction for later evaluation
        if predictions:
          prediction_id = f"{entity_id}_{sequence_type}_{int(time.time())}"
          self.predictions[prediction_id] = {
            "entity_id": entity_id,
            "sequence_type": sequence_type,
            "predictions": predictions,
            "timestamp": time.time(),
            "method": "statistical"
          }

        return predictions

    def _neural_predict(self, entity_id, sequence_type, time_horizon):
      """
      Use the neural network for prediction.
      """

      sequence = self.sequences[entity_id][sequence_type]
      recent_events = sequence.get_events_in_window(window_size=time_horizon)

      # Need enough data for the neural network
      if len(recent_events) < 5:
          return []

      # --- Feature Extraction ---
      #  Here you would extract features from your events.  This is *highly*
      #  dependent on the nature of your events.  For this example, I'll
      #  assume each event has a 'value' attribute, which could be a scalar
      #  or a vector.

      # 1. Timestamps and Intervals:
      timestamps = list(sequence.timestamps)[-len(recent_events):]
      intervals = np.diff(timestamps)

      # 2. Event Features (example - adapt to your event data)
      event_features = []
      for event in recent_events:
          if hasattr(event, 'value'):  # Assuming your events have a 'value'
              if isinstance(event.value, (int, float)):
                event_features.append([event.value]) #wrap scalar to be 2d
              elif isinstance(event.value, (list, np.ndarray)):
                event_features.append(event.value) #append list or array
              else: #unsupported type.
                return []
          else: #Event does not have a "value" attribute.
              return []  # Can't use neural net without features

      # Pad intervals to match the number of features
      if len(intervals) < len(event_features):
        intervals = np.append(intervals, [0.0] * (len(event_features)-len(intervals))) #pad

      # Combine intervals and features (example - adjust as needed)
      combined_features = []
      for i in range(len(event_features)):
        combined_features.append(np.concatenate(( [intervals[i]], np.array(event_features[i]).flatten())))

      # --- Input Tensor Creation ---
      # Ensure consistent feature size by padding.
      max_feature_size = max(len(x) for x in combined_features)

      padded_features = []
      for features in combined_features:
        padding_needed = max_feature_size - len(features)
        padded_features.append(np.pad(features, (0, padding_needed), 'constant'))

      # Convert to tensor [batch_size (1), seq_length, feature_size]
      sequence_tensor = torch.tensor(padded_features, dtype=torch.float32).unsqueeze(0)

      # --- Prediction ---
      with torch.no_grad():  # Don't track gradients during prediction
          predicted_intervals, confidence = self.predictor(sequence_tensor)

          # Convert to timestamps from intervals
          last_time = timestamps[-1]
          current_time = time.time()
          predictions = []

          for i, interval in enumerate(predicted_intervals[0]): #we only have a batch size of 1.
              next_time = last_time + interval.item()
              last_time = next_time

              if next_time <= current_time + time_horizon:
                predictions.append({
                    "predicted_time": next_time,
                    "confidence": min(0.95, confidence[0,i].item()),
                    "time_from_now": next_time - current_time
                })

      return predictions

    def evaluate_predictions(self, entity_id, sequence_type, current_time=None):
        """Evaluates past predictions and updates error tracking.

        Args:
            entity_id (str): ID of the entity.
            sequence_type (str): Type of the event sequence.
            current_time (float, optional): Current time. Defaults to now.

        Returns:
            dict: Evaluation metrics (or None if no predictions to evaluate).
        """
        if current_time is None:
            current_time = time.time()

        # Find relevant predictions
        relevant_predictions = []
        for pred_id, prediction in self.predictions.items():
            if (prediction["entity_id"] == entity_id and
                prediction["sequence_type"] == sequence_type and
                prediction["timestamp"] < current_time): #prediction was made in the past
                relevant_predictions.append((pred_id, prediction)) #tuple

        if not relevant_predictions:
            return {"evaluated": False, "reason": "no_predictions"}

        # Get actual events
        if entity_id not in self.sequences or sequence_type not in self.sequences[entity_id]:
            return {"evaluated": False, "reason": "sequence_not_found"}

        sequence = self.sequences[entity_id][sequence_type]

        # Evaluation results for all predictions
        evaluation_results = []

        for pred_id, prediction in relevant_predictions:
          # If already evaluated, skip it
          if "evaluated" in prediction and prediction["evaluated"]:
            continue;

          pred_time = prediction["timestamp"]
          pred_events = prediction["predictions"] #list of predictions

          # Actual timestamps that happened after the prediction was made
          actual_timestamps = [ts for ts in sequence.timestamps if ts > pred_time]

          # If we don't have future data, we can't evaluate.
          if not actual_timestamps:
            continue;

          # For each predicted event, find closest actual event
          prediction_errors = []

          for pred_event in pred_events:
            pred_timestamp = pred_event["predicted_time"]

            # Don't evaluate future predictions.
            if pred_timestamp > current_time:
              continue;

            # Find the closest actual event time
            closest_actual = min(actual_timestamps, key=lambda x: abs(x - pred_timestamp))

            # Calculate the error (time difference)
            error = abs(closest_actual - pred_timestamp)

            # Normalize the error relative to the prediction horizon
            horizon = max(pred_event["time_from_now"], 60) #at least 1 minute
            normalized_error = min(1.0, error/horizon)

            prediction_errors.append(normalized_error)

          if not prediction_errors:
            continue;

          # Average error for this prediction batch
          avg_error = sum(prediction_errors) / len(prediction_errors)

          # Track prediction errors
          error_key = f"{entity_id}_{sequence_type}"
          self.prediction_errors[error_key].append(avg_error)

          # Limit history size
          if len(self.prediction_errors[error_key]) > 20:
              self.prediction_errors[error_key] = self.prediction_errors[error_key][-20:]

          # Update precision value
          if len(self.prediction_errors[error_key]) > 1:
              error_variance = np.var(self.prediction_errors[error_key])
              precision = 1.0 / (error_variance + 1e-6)
              precision = min(10.0, precision) #Cap precision

              # Store the precision
              self.precision_values.setdefault(entity_id, {})[sequence_type] = precision

          # Mark prediction as evaluated, and record outcome
          prediction["evaluated"] = True
          prediction["avg_error"] = avg_error
          prediction["evaluation_time"] = current_time

          evaluation_results.append({
            "prediction_id": pred_id,
            "avg_error": avg_error,
            "num_predictions": len(pred_events),
            "num_evaluated": len(prediction_errors)
          })

          # Record outcome for learning
          self.prediction_outcomes.append({
                "entity_id": entity_id,
                "sequence_type": sequence_type,
                "prediction_method": prediction["method"],
                "avg_error": avg_error,
                "precision": self.precision_values.get(entity_id, {}).get(sequence_type, 1.0), #default
                "timestamp": current_time
          })

        # Return evaluation summary
        if evaluation_results:
            return {
              "evaluated": True,
              "num_predictions": len(evaluation_results),
              "avg_error": sum(r["avg_error"] for r in evaluation_results) / len(evaluation_results),
              "precision": self.precision_values.get(entity_id, {}).get(sequence_type, 1.0),
              "details": evaluation_results
            }
        else:
          return {"evaluated": False, "reason": "no_evaluable_predictions"}

    def add_event(self, entity_id, sequence_type, event_data):
        """Adds an event and evaluates predictions."""
        # Call parent implementation
        result = super().add_event(entity_id, sequence_type, event_data)

        # Evaluate existing predictions
        self.evaluate_predictions(entity_id, sequence_type)
        return result
    
    def generate_counterfactual_sequence(self, entity_id, sequence_type, 
                                           modification, time_horizon=3600):
        
        """Generates a counterfactual event sequence.

        Args:
            entity_id (str): Entity ID.
            sequence_type (str): Sequence type.
            modification (dict): Describes the modification (e.g., {"interval_factor": 0.5}).
            time_horizon (int): Time horizon in seconds.

        Returns:
            list: Counterfactual sequence of (event, timestamp) tuples.
        """
        if entity_id not in self.sequences or sequence_type not in self.sequences[entity_id]:
            return []

        sequence = self.sequences[entity_id][sequence_type]
        recent_events = sequence.get_events_in_window(window_size=time_horizon)

        if len(recent_events) < 3:
            return []  # Not enough data

        timestamps = list(sequence.timestamps)
        if len(timestamps) < 2:
          return []

        # Create copies for the counterfactual sequence
        cf_events = recent_events.copy()
        cf_timestamps = timestamps.copy()

        # Apply modifications
        if "interval_factor" in modification:
            factor = modification["interval_factor"]
            new_timestamps = [timestamps[0]]
            for i in range(1, len(timestamps)):
                interval = timestamps[i] - timestamps[i - 1]
                new_interval = interval * factor
                new_timestamps.append(new_timestamps[-1] + new_interval)
            cf_timestamps = new_timestamps
        elif "remove_pattern" in modification:
            # Remove a specific pattern (implementation depends on pattern definition)
            pattern = modification["remove_pattern"]

            filtered_events = []
            filtered_timestamps = []

            for i, event in enumerate(cf_events):
                if not self._matches_pattern(event, pattern): #helper function needed
                  filtered_events.append(event)
                  filtered_timestamps.append(cf_timestamps[i])

            cf_events = filtered_events
            cf_timestamps = filtered_timestamps

        # Add more modification types as needed (e.g., add specific events,
        # shift events in time, change event attributes).

        # Record the counterfactual simulation
        cf_id = f"cf_{entity_id}_{sequence_type}_{int(time.time())}"  # Unique ID
        self.counterfactual_simulations.append({
            "id": cf_id,
            "entity_id": entity_id,
            "sequence_type": sequence_type,
            "modification": modification,  # What was changed
            "original_count": len(recent_events),  # Number of events originally
            "counterfactual_count": len(cf_events),  # Number of events in counterfactual
            "timestamp": time.time()
        })

        # Return the counterfactual sequence as (event, timestamp) pairs
        return list(zip(cf_events, cf_timestamps))

    def _matches_pattern(self, event, pattern):
        """
        Helper function to check if an event matches a given pattern.

        This is a placeholder.  The actual implementation depends *heavily* on
        how you represent events and patterns.  It needs to be tailored to
        your specific data and use case.

        Args:
           event:  The event to check.
           pattern: The pattern to match against.

        Returns:
            bool: True if the event matches the pattern, False otherwise.
        """

        # --- Example 1:  Simple Attribute Matching (if events are dictionaries) ---
        if isinstance(event, dict) and isinstance(pattern, dict):
            for key, value in pattern.items():
                if key not in event or event[key] != value:
                    return False  # Doesn't match
            return True  # All pattern attributes match

        # --- Example 2:  Feature Vector Similarity (if events have feature vectors) ---
        if hasattr(event, 'get_feature_vector') and hasattr(pattern, 'get_feature_vector'):
            event_vec = event.get_feature_vector()
            pattern_vec = pattern.get_feature_vector()

            # Check if vectors have the same dimensionality
            if len(event_vec) == len(pattern_vec):
                # Calculate cosine similarity
                similarity = np.dot(event_vec, pattern_vec) / (
                    np.linalg.norm(event_vec) * np.linalg.norm(pattern_vec))
                return similarity > 0.8  # Threshold for similarity

        # --- Example 3:  Using a custom matching function ---
        if hasattr(pattern, 'matches') and callable(pattern.matches):
          return pattern.matches(event) #Delegate

        # --- Default: No Match (if we can't determine a match) ---
        return False