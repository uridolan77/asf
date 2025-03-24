import time
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from asf.layer1_knowledge_substrate.temporal.processing_engine import TemporalProcessingEngine

class PredictiveTemporalEngine(TemporalProcessingEngine):
    """
    Enhanced temporal processing engine incorporating Seth's predictive processing principles.
    
    Extends the base TemporalProcessingEngine with:
    - Explicit prediction of future events
    - Prediction error tracking
    - Precision-weighted updates
    - Counterfactual temporal simulations
    
    Philosophical influences: Seth's predictive processing, Friston's Free Energy
    """
    
    def __init__(self):
        super().__init__()
        # Track prediction errors
        self.prediction_errors = defaultdict(list)
        # Track predictions and outcomes
        self.predictions = {}
        self.prediction_outcomes = []
        # Precision values (inverse variance of prediction errors)
        self.precision_values = {}
        # Learning rates (adaptive based on prediction error)
        self.learning_rates = defaultdict(lambda: 0.3)  # Default learning rate
        # Counterfactual simulations
        self.counterfactual_simulations = []
        
        # Pytorch model for temporal prediction (optional, more sophisticated prediction)
        self.use_neural_prediction = False  # Toggle for neural prediction
        if self.use_neural_prediction:
            self.predictor = TemporalPredictor(input_size=64, hidden_size=128)
    
    def predict_next_events(self, entity_id, sequence_type, time_horizon=3600):
        """
        Predict future events based on temporal patterns.
        
        Args:
            entity_id: Entity to predict for
            sequence_type: Type of sequence to predict
            time_horizon: How far into the future to predict (seconds)
            
        Returns:
            List of predicted events with timestamps
        """
        if entity_id not in self.sequences or sequence_type not in self.sequences[entity_id]:
            return []
        
        sequence = self.sequences[entity_id][sequence_type]
        recent_events = sequence.get_events_in_window(window_size=time_horizon)
        
        if len(recent_events) < 3:  # Need minimum data for prediction
            return []
        
        # Extract timestamps for interval analysis
        timestamps = list(sequence.timestamps)
        if len(timestamps) < 2:
            return []
            
        # Analyze temporal patterns
        event_intervals = []
        for i in range(1, len(timestamps)):
            interval = timestamps[i] - timestamps[i-1]
            event_intervals.append(interval)
        
        if not event_intervals:
            return []
        
        # Calculate mean interval and variance (for precision)
        mean_interval = np.mean(event_intervals)
        interval_variance = np.var(event_intervals)
        
        # More regular patterns = higher precision = more confident predictions
        precision = 1.0 / (interval_variance + 1e-6)
        
        # Cap precision to reasonable range
        precision = min(10.0, precision)
        
        # Store precision for this sequence
        self.precision_values.setdefault(entity_id, {})
        self.precision_values[entity_id][sequence_type] = precision
        
        # Last event time
        last_time = timestamps[-1] if timestamps else time.time()
        
        # Predict future events based on pattern
        predictions = []
        current_time = time.time()
        next_time = last_time + mean_interval
        
        # Use neural prediction if enabled and enough data
        if self.use_neural_prediction and len(recent_events) >= 5:
            neural_predictions = self._neural_predict(entity_id, sequence_type, time_horizon)
            if neural_predictions:
                # Store predictions for later evaluation
                prediction_id = f"{entity_id}_{sequence_type}_{int(time.time())}"
                self.predictions[prediction_id] = {
                    "entity_id": entity_id,
                    "sequence_type": sequence_type,
                    "predictions": neural_predictions,
                    "timestamp": time.time(),
                    "method": "neural"
                }
                return neural_predictions
        
        # Fall back to statistical prediction
        while next_time <= current_time + time_horizon:
            # Create prediction with confidence based on precision
            confidence = min(0.9, np.tanh(precision) * 0.8)
            
            predictions.append({
                "predicted_time": next_time,
                "confidence": confidence,
                "time_from_now": next_time - current_time
            })
            
            next_time += mean_interval
        
        # Store predictions for later evaluation
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
    
    def evaluate_predictions(self, entity_id, sequence_type, current_time=None):
        """
        Evaluate accuracy of past predictions for this entity/sequence.
        Updates prediction errors and precision values.
        
        Returns evaluation metrics dictionary.
        """
        if current_time is None:
            current_time = time.time()
            
        # Find predictions for this entity/sequence
        relevant_predictions = []
        for pred_id, prediction in self.predictions.items():
            if (prediction["entity_id"] == entity_id and 
                prediction["sequence_type"] == sequence_type and
                prediction["timestamp"] < current_time):
                relevant_predictions.append((pred_id, prediction))
        
        if not relevant_predictions:
            return {"evaluated": False, "reason": "no_predictions"}
            
        # Get actual events that occurred since predictions were made
        if entity_id not in self.sequences or sequence_type not in self.sequences[entity_id]:
            return {"evaluated": False, "reason": "sequence_not_found"}
            
        sequence = self.sequences[entity_id][sequence_type]
        
        # Results for all evaluated predictions
        evaluation_results = []
        
        for pred_id, prediction in relevant_predictions:
            # Skip if already evaluated
            if "evaluated" in prediction and prediction["evaluated"]:
                continue
                
            pred_time = prediction["timestamp"]
            pred_events = prediction["predictions"]
            
            # Get actual events that occurred after prediction was made
            actual_timestamps = [ts for ts in sequence.timestamps if ts > pred_time]
            
            # Skip if no actual events to compare against
            if not actual_timestamps:
                continue
                
            # For each predicted event, find closest actual event
            prediction_errors = []
            
            for pred_event in pred_events:
                pred_timestamp = pred_event["predicted_time"]
                
                # Skip predicted events in the future
                if pred_timestamp > current_time:
                    continue
                    
                # Find closest actual event
                closest_actual = min(actual_timestamps, key=lambda x: abs(x - pred_timestamp))
                
                # Calculate error (time difference in seconds)
                error = abs(closest_actual - pred_timestamp)
                
                # Normalize error relative to prediction horizon
                horizon = max(pred_event["time_from_now"], 60)  # At least 1 minute
                normalized_error = min(1.0, error / horizon)
                
                prediction_errors.append(normalized_error)
            
            # Skip if no predictions were evaluated
            if not prediction_errors:
                continue
                
            # Calculate average error
            avg_error = sum(prediction_errors) / len(prediction_errors)
            
            # Update prediction error tracking
            error_key = f"{entity_id}_{sequence_type}"
            self.prediction_errors[error_key].append(avg_error)
            
            # Limit history size
            if len(self.prediction_errors[error_key]) > 20:
                self.prediction_errors[error_key] = self.prediction_errors[error_key][-20:]
            
            # Update precision value
            if len(self.prediction_errors[error_key]) > 1:
                error_variance = np.var(self.prediction_errors[error_key])
                precision = 1.0 / (error_variance + 1e-6)
                precision = min(10.0, precision)  # Cap precision
                
                # Store updated precision
                self.precision_values.setdefault(entity_id, {})
                self.precision_values[entity_id][sequence_type] = precision
            
            # Mark prediction as evaluated
            prediction["evaluated"] = True
            prediction["avg_error"] = avg_error
            prediction["evaluation_time"] = current_time
            
            # Add to results
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
                "precision": self.precision_values.get(entity_id, {}).get(sequence_type, 1.0),
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
        """
        Add an event and update predictions. 
        Overrides base method to include prediction evaluation.
        """
        # First call parent implementation
        result = super().add_event(entity_id, sequence_type, event_data)
        
        # Then evaluate existing predictions
        self.evaluate_predictions(entity_id, sequence_type)
        
        # Return original result
        return result
    
    def generate_counterfactual_sequence(self, entity_id, sequence_type, 
                                        modification, time_horizon=3600):
        """
        Generate a counterfactual sequence by modifying existing temporal patterns.
        
        Args:
            entity_id: Entity to generate counterfactual for
            sequence_type: Type of sequence
            modification: Dictionary specifying the modification 
                (e.g., {"interval_factor": 0.5} to make events twice as frequent)
            time_horizon: Time horizon for counterfactual (seconds)
            
        Returns:
            Counterfactual sequence of events
        """
        if entity_id not in self.sequences or sequence_type not in self.sequences[entity_id]:
            return []
            
        sequence = self.sequences[entity_id][sequence_type]
        recent_events = sequence.get_events_in_window(window_size=time_horizon)
        
        if len(recent_events) < 3:  # Need minimum data
            return []
            
        # Extract timestamps
        timestamps = list(sequence.timestamps)
        if len(timestamps) < 2:
            return []
            
        # Create copy of events for counterfactual
        cf_events = recent_events.copy()
        cf_timestamps = timestamps.copy()
        
        # Apply modifications
        if "interval_factor" in modification:
            # Change frequency of events
            factor = modification["interval_factor"]
            
            # Calculate new intervals
            new_timestamps = [timestamps[0]]  # Keep first timestamp
            
            for i in range(1, len(timestamps)):
                interval = timestamps[i] - timestamps[i-1]
                # Apply factor to interval
                new_interval = interval * factor
                new_timestamps.append(new_timestamps[-1] + new_interval)
                
            cf_timestamps = new_timestamps
        
        elif "remove_pattern" in modification:
            # Remove a specific pattern from sequence
            pattern = modification["remove_pattern"]
            
            # Filter events not matching pattern
            filtered_events = []
            filtered_timestamps = []
            
            for i, event in enumerate(cf_events):
                if not self._matches_pattern(event, pattern):
                    filtered_events.append(event)
                    filtered_timestamps.append(cf_timestamps[i])
                    
            cf_events = filtered_events
            cf_timestamps = filtered_timestamps
        
        # Record counterfactual simulation
        cf_id = f"cf_{entity_id}_{sequence_type}_{int(time.time())}"
        self.counterfactual_simulations.append({
            "id": cf_id,
            "entity_id": entity_id,
            "sequence_type": sequence_type,
            "modification": modification,
            "original_count": len(recent_events),
            "counterfactual_count": len(cf_events),
            "timestamp": time.time()
        })
        
        # Return counterfactual sequence
        return list(zip(cf_events, cf_timestamps))
    
    def _neural_predict(self, entity_id, sequence_type, time_horizon):
        """
        Use neural network for more sophisticated temporal prediction.
        
        Returns list of predicted events with timestamps and confidence.
        """
        if not self.use_neural_prediction:
            return []
            
        sequence = self.sequences[entity_id][sequence_type]
        recent_events = sequence.get_events_in_window(time_horizon)
        
        if len(recent_events) < 5:  # Need more data for neural prediction
            return []
            
        # Extract features and timestamps
        timestamps = list(sequence.timestamps)[-len(recent_events):]
        
        # Convert to intervals
        intervals = []
        for i in range(1, len(timestamps)):
            intervals.append(timestamps[i] - timestamps[i-1])
            
        if not intervals:
            return []
            
        # Create input tensor
        # For simplicity, just use intervals as features
        # In real implementation, would include event features
        sequence_tensor = torch.tensor(intervals, dtype=torch.float32).unsqueeze(0)
        
        # Generate predictions
        with torch.no_grad():
            predicted_intervals, confidence = self.predictor(sequence_tensor)
            
        # Convert back to timestamps
        last_time = timestamps[-1]
        current_time = time.time()
        
        predictions = []
        for i, interval in enumerate(predicted_intervals[0]):
            next_time = last_time + interval.item()
            last_time = next_time
            
            if next_time <= current_time + time_horizon:
                predictions.append({
                    "predicted_time": next_time,
                    "confidence": min(0.95, confidence[0, i].item()),
                    "time_from_now": next_time - current_time
                })
                
        return predictions
    
    def _matches_pattern(self, event, pattern):
        """Check if event matches specified pattern"""
        # Implementation depends on event and pattern representation
        # This is a simplified placeholder
        if hasattr(event, 'get_feature_vector') and hasattr(pattern, 'get_feature_vector'):
            event_vec = event.get_feature_vector()
            pattern_vec = pattern.get_feature_vector()
            
            if len(event_vec) == len(pattern_vec):
                similarity = np.dot(event_vec, pattern_vec) / (
                    np.linalg.norm(event_vec) * np.linalg.norm(pattern_vec))
                return similarity > 0.8
        
        return False


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
