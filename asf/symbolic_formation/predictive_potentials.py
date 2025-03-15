import time
import numpy as np
import joblib
from collections import defaultdict
from typing import Dict, Any, List, Optional, Set, Tuple

from asf.symbolic_formation.potentials import SymbolicPotential
from asf.symbolic_formation.enums import NonlinearityOrder

class PredictiveSymbolicPotential(SymbolicPotential):
    """
    Symbolic potential with prediction capabilities.
    Uses precision-weighted activation for better uncertainty handling.
    """
    def __init__(self, id: str, strength: float = 1.0, nonlinearity: NonlinearityOrder = NonlinearityOrder.LINEAR):
        super().__init__(id, strength, nonlinearity)
        self.predicted_activations = {}  # Context hash -> predicted activation
        self.prediction_errors = defaultdict(list)  # Context hash -> list of errors
        self.precision_values = {}  # Context hash -> precision value
        
    def predict_activation(self, context: Dict[str, Any]) -> Optional[float]:
        """Predict activation before context actualization"""
        context_hash = joblib.hash(context)
        
        # If we don't have enough past activations, can't predict
        if len(self._activations) < 3:
            return None
            
        # Find similar contexts (simplified implementation)
        similar_contexts = list(self._activations.keys())[:5]  # Use 5 most recent contexts
        
        if not similar_contexts:
            return None
        
        # Calculate predicted activation as weighted average of similar contexts
        total_weight = 0.0
        weighted_sum = 0.0
        
        for past_hash in similar_contexts:
            weight = 1.0  # Equal weighting for simplicity
            activation = self._activations[past_hash]
            
            weighted_sum += activation * weight
            total_weight += weight
        
        if total_weight > 0:
            predicted = weighted_sum / total_weight
            self.predicted_activations[context_hash] = predicted
            return predicted
            
        return None
        
    def actualize(self, context: Dict[str, Any], potential_network: Optional[Dict[str, 'SymbolicPotential']] = None) -> float:
        """Predict activation, then actualize and calculate error"""
        context_hash = joblib.hash(context)
        
        # First make prediction
        predicted = self.predict_activation(context)
        
        # Then get actual activation
        actual = super().actualize(context, potential_network)
        
        # Calculate prediction error if prediction was made
        if predicted is not None:
            error = abs(predicted - actual)
            self.prediction_errors[context_hash].append(error)
            
            # Limit history size
            if len(self.prediction_errors[context_hash]) > 20:
                self.prediction_errors[context_hash] = self.prediction_errors[context_hash][-20:]
            
            # Update precision (inverse variance)
            if len(self.prediction_errors[context_hash]) > 1:
                variance = np.var(self.prediction_errors[context_hash])
                precision = 1.0 / (variance + 1e-6)  # Avoid division by zero
                self.precision_values[context_hash] = min(10.0, precision)  # Cap very high precision
        
        return actual
    
    def get_precision(self, context_hash=None):
        """Get precision for a context or overall"""
        if context_hash and context_hash in self.precision_values:
            return self.precision_values[context_hash]
        
        # Calculate overall precision
        all_errors = []
        for errors in self.prediction_errors.values():
            all_errors.extend(errors)
            
        if len(all_errors) < 2:
            return 1.0  # Default precision
            
        variance = np.var(all_errors)
        precision = 1.0 / (variance + 1e-6)  # Avoid division by zero
        return min(10.0, precision)  # Cap very high precision
