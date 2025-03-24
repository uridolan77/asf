import time
import math
import numpy as np
import joblib
from collections import defaultdict
from typing import Dict, Any, List, Optional, Set, Tuple

from asf.layer2_autopoietic_maintanance.symbol import SymbolElement
from asf.layer2_autopoietic_maintanance.potentials import SymbolicPotential
from asf.layer2_autopoietic_maintanance.enums import NonlinearityOrder, SymbolConfidenceState

class PredictiveSymbolElement(SymbolElement):
    """
    Enhanced symbol with predictive capabilities.
    Predicts meaning activations before observing context.
    """
    def __init__(self, symbol_id: str, perceptual_anchors: Dict[str, float] = None):
        super().__init__(symbol_id, perceptual_anchors)
        self.predicted_meanings = {}  # Context hash -> predicted meanings
        self.prediction_errors = defaultdict(list)  # Context -> errors
        self.precision_values = {}  # Context -> precision (1/variance)
        
    def predict_meaning(self, context_hash: str, context: Dict[str, Any]) -> Dict[str, float]:
        """Predict meaning before actualization based on similar contexts"""
        # If we don't have enough historical data, can't predict meaningfully
        if len(self._actual_meanings) < 3:
            return {}
            
        # Find similar contexts based on keys
        similar_contexts = []
        context_keys = set(context.keys())
        
        for past_hash in self._actual_meanings:
            # Simple similarity heuristic - add to similar if we have actual data
            if past_hash in self._actual_meanings:
                similar_contexts.append(past_hash)
                
        if not similar_contexts:
            return {}
            
        # Calculate predicted meaning based on similar contexts
        predicted = {}
        weights = {}
        
        for past_hash in similar_contexts:
            past_meaning = self._actual_meanings[past_hash]
            
            # Weight by recency (more recent contexts get higher weight)
            weight = 1.0
            if past_hash in self._activation_time:
                # Decay weight by time since activation
                time_elapsed = time.time() - self._activation_time[past_hash]
                decay_factor = math.exp(-time_elapsed / 3600)  # 1-hour half-life
                weight *= decay_factor
            
            # Accumulate weighted meanings
            for potential_id, activation in past_meaning.items():
                if potential_id not in predicted:
                    predicted[potential_id] = 0.0
                    weights[potential_id] = 0.0
                
                predicted[potential_id] += activation * weight
                weights[potential_id] += weight
        
        # Normalize by weights
        for potential_id in predicted:
            if weights[potential_id] > 0:
                predicted[potential_id] /= weights[potential_id]
        
        # Store prediction
        self.predicted_meanings[context_hash] = predicted
        return predicted
        
    def actualize_meaning(self, context_hash: str, context: Dict[str, Any]) -> Dict[str, float]:
        """First predict meaning, then actualize and compare"""
        # Make prediction if we have prior contexts
        predicted = {}
        if len(self._actual_meanings) >= 3:
            predicted = self.predict_meaning(context_hash, context)
        
        # Actual actualization from parent class
        actualized = super().actualize_meaning(context_hash, context)
        
        # Calculate prediction error if prediction was made
        if predicted:
            # Calculate error for each potential
            errors = []
            for potential_id in set(list(predicted.keys()) + list(actualized.keys())):
                pred_value = predicted.get(potential_id, 0.0)
                actual_value = actualized.get(potential_id, 0.0)
                error = abs(pred_value - actual_value)
                errors.append(error)
            
            # Average error across all potentials
            if errors:
                avg_error = sum(errors) / len(errors)
                self.prediction_errors[context_hash].append(avg_error)
                
                # Calculate precision (inverse variance)
                if len(self.prediction_errors[context_hash]) > 1:
                    variance = np.var(self.prediction_errors[context_hash])
                    precision = 1.0 / (variance + 1e-6)
                    self.precision_values[context_hash] = precision
        
        return actualized
    
    def get_prediction_precision(self, context_hash=None):
        """Get prediction precision for a context or overall"""
        if context_hash and context_hash in self.precision_values:
            return self.precision_values[context_hash]
            
        # Calculate overall precision across all contexts
        all_errors = []
        for errors in self.prediction_errors.values():
            all_errors.extend(errors)
            
        if len(all_errors) < 2:
            return 1.0  # Default precision
            
        variance = np.var(all_errors)
        precision = 1.0 / (variance + 1e-6)
        return min(10.0, precision)  # Cap very high precision
