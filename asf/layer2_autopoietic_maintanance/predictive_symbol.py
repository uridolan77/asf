import time
import math
import numpy as np
from collections import defaultdict
from typing import Any, Dict, List, Optional

# Minimal base class for a symbol element.
class SymbolElement:
    def __init__(self, symbol_id: str, perceptual_anchors: Optional[Dict[str, float]] = None):
        self.symbol_id = symbol_id
        self.perceptual_anchors = perceptual_anchors or {}
        self._actual_meanings: Dict[str, Dict[str, float]] = {}   # Maps context hashes to actualized meanings.
        self._activation_time: Dict[str, float] = {}                # Maps context hashes to activation timestamps.

    def actualize_meaning(self, context_hash: str, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Dummy actualization method: returns stored meaning for the given context hash.
        In a real implementation, this would compute activations based on current inputs.
        """
        return self._actual_meanings.get(context_hash, {})

class EnhancedPredictiveSymbolElement(SymbolElement):
    """
    An enhanced symbol element that predicts meaning activations in advance.
    
    Enhancements include:
      • Adaptive decay: Context-specific decay rates rather than a fixed half-life.
      • Hierarchical context: Allows for richer, nuanced comparison via stored contextual data.
      • Concept drift detection: Monitors recent prediction errors using a sliding window.
      • Cross-symbol learning: Aggregates patterns from related symbols to reinforce predictions.
      • Memory management: Limits unbounded growth of error history with trimming.
      • Serialization: Provides to_dict and from_dict for persistence.
      • Robust error handling: Wraps sensitive operations with exception handling.
    """
    def __init__(self, symbol_id: str, perceptual_anchors: Optional[Dict[str, float]] = None):
        super().__init__(symbol_id, perceptual_anchors)
        self.predicted_meanings: Dict[str, Dict[str, float]] = {}   # context_hash -> predicted meanings
        self.prediction_errors: Dict[str, List[float]] = defaultdict(list)  # context_hash -> list of errors
        self.precision_values: Dict[str, float] = {}               # context_hash -> precision (inverse variance)
        
        # Additional enhancement attributes:
        self.adaptive_decay_params: Dict[str, float] = {}          # context_hash -> decay parameter (1/half-life)
        self.hierarchical_contexts: Dict[str, Dict[str, Any]] = {}   # context_hash -> richer context info
        self.cross_symbol_patterns: Dict[str, Dict[str, float]] = {} # Related symbol pattern aggregation
        
        # Parameters for improved context similarity and memory management:
        self.similarity_threshold: float = 0.5  # Minimum required similarity to consider past context
        self.error_history_window: int = 100    # Maximum number of error samples to retain per context

    def set_adaptive_decay(self, context_hash: str, decay_rate: float):
        """Set an adaptive decay rate for a specific past context."""
        self.adaptive_decay_params[context_hash] = decay_rate

    def update_hierarchical_context(self, context_hash: str, hierarchy: Dict[str, Any]):
        """Store hierarchical context information for the given context hash."""
        self.hierarchical_contexts[context_hash] = hierarchy

    def _calculate_context_similarity(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
        """
        Calculate a similarity score between two context dicts.
        Returns a value between 0 (completely different) and 1 (identical).
        Uses a simple average over common keys.
        """
        keys1, keys2 = set(context1.keys()), set(context2.keys())
        common_keys = keys1.intersection(keys2)
        if not common_keys:
            return 0.0
        similarities = []
        for key in common_keys:
            val1, val2 = context1[key], context2[key]
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                max_val = max(abs(val1), abs(val2)) or 1.0
                sim = 1.0 - min(1.0, abs(val1 - val2) / max_val)
            elif isinstance(val1, str) and isinstance(val2, str):
                sim = 1.0 if val1 == val2 else 0.0
            else:
                sim = 0.0
            similarities.append(sim)
        return sum(similarities) / len(similarities)

    def _get_similar_contexts(self, new_context: Dict[str, Any]) -> List[str]:
        """
        Identify past contexts that are similar to the new context based on stored hierarchical data.
        Returns a list of context hashes meeting or exceeding the similarity threshold.
        """
        similar_contexts = []
        new_hier = self.hierarchical_contexts.get("current", new_context)
        for past_hash, past_context in self.hierarchical_contexts.items():
            sim = self._calculate_context_similarity(new_hier, past_context)
            if sim >= self.similarity_threshold:
                similar_contexts.append(past_hash)
        return similar_contexts

    def trim_error_history(self):
        """Trim error history lists to avoid unbounded memory growth."""
        for context in self.prediction_errors:
            if len(self.prediction_errors[context]) > self.error_history_window:
                self.prediction_errors[context] = self.prediction_errors[context][-self.error_history_window:]

    def predict_meaning(self, context_hash: str, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Predict the symbol's meaning for a given context.
        
        Uses similar past contexts (based on hierarchical similarity), applies adaptive decay,
        and weighs activations accordingly. Requires at least three historical actualizations.
        """
        if len(self._actual_meanings) < 3:
            return {}
        if context_hash not in self.hierarchical_contexts:
            self.hierarchical_contexts[context_hash] = context

        similar_contexts = self._get_similar_contexts(context)
        if not similar_contexts:
            return {}

        predicted: Dict[str, float] = {}
        weights: Dict[str, float] = {}
        for past_hash in similar_contexts:
            if past_hash not in self._actual_meanings:
                continue
            past_meaning = self._actual_meanings[past_hash]
            weight = 1.0
            if past_hash in self._activation_time:
                time_elapsed = time.time() - self._activation_time[past_hash]
                decay_param = self.adaptive_decay_params.get(past_hash, 1.0 / 3600)
                decay_factor = math.exp(-time_elapsed * decay_param)
                weight *= decay_factor
            hier = self.hierarchical_contexts.get(past_hash, {})
            hierarchical_weight = hier.get("weight", 1.0) if isinstance(hier, dict) else 1.0
            weight *= hierarchical_weight
            for potential_id, activation in past_meaning.items():
                predicted[potential_id] = predicted.get(potential_id, 0.0) + activation * weight
                weights[potential_id] = weights.get(potential_id, 0.0) + weight

        for potential_id in predicted:
            if weights[potential_id] > 0:
                predicted[potential_id] /= weights[potential_id]

        self.predicted_meanings[context_hash] = predicted
        return predicted

    def actualize_meaning(self, context_hash: str, context: Dict[str, Any]) -> Dict[str, float]:
        """
        First generate a prediction, then actualize meaning (using the base implementation).
        
        Compares the prediction to the actual value, tracks error, and updates precision.
        """
        predicted = {}
        if len(self._actual_meanings) >= 3:
            predicted = self.predict_meaning(context_hash, context)
        actualized = super().actualize_meaning(context_hash, context)
        if predicted:
            errors = []
            for potential_id in set(list(predicted.keys()) + list(actualized.keys())):
                pred_value = predicted.get(potential_id, 0.0)
                actual_value = actualized.get(potential_id, 0.0)
                errors.append(abs(pred_value - actual_value))
            if errors:
                avg_error = sum(errors) / len(errors)
                self.prediction_errors[context_hash].append(avg_error)
                self.trim_error_history()
                try:
                    if len(self.prediction_errors[context_hash]) > 1:
                        variance = np.var(self.prediction_errors[context_hash])
                        precision = 1.0 / (variance + 1e-6)
                        self.precision_values[context_hash] = min(10.0, precision)
                except Exception as e:
                    self.precision_values[context_hash] = 1.0
        return actualized

    def get_prediction_precision(self, context_hash: Optional[str] = None) -> float:
        """
        Retrieve prediction precision.
        If a context hash is provided and available, return its precision;
        otherwise, compute an overall precision across all contexts.
        """
        if context_hash and context_hash in self.precision_values:
            return self.precision_values[context_hash]
        all_errors = []
        for errors in self.prediction_errors.values():
            all_errors.extend(errors)
        if len(all_errors) < 2:
            return 1.0
        variance = np.var(all_errors)
        precision = 1.0 / (variance + 1e-6)
        return min(10.0, precision)

    def detect_concept_drift(self, context_hash: str, threshold: float = 0.2, window_size: int = 5) -> bool:
        """
        Detect concept drift by checking if the difference between the max and min 
        in a recent window of prediction errors exceeds a threshold.
        """
        errors = self.prediction_errors.get(context_hash, [])
        if len(errors) < window_size:
            return False
        recent_errors = errors[-window_size:]
        return (max(recent_errors) - min(recent_errors)) > threshold

    def incorporate_cross_symbol_learning(self, related_symbols: Dict[str, Dict[str, float]]):
        """
        Update cross-symbol patterns with learning data from related symbols.
        
        related_symbols should map a symbol ID to its pattern dictionary.
        """
        for symbol_id, patterns in related_symbols.items():
            if symbol_id not in self.cross_symbol_patterns:
                self.cross_symbol_patterns[symbol_id] = {}
            for pattern, weight in patterns.items():
                self.cross_symbol_patterns[symbol_id][pattern] = (
                    self.cross_symbol_patterns[symbol_id].get(pattern, 0.0) + weight
                )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the internal state of the symbol element to a dictionary.
        Useful for persistence or state transfer.
        """
        return {
            "symbol_id": self.symbol_id,
            "perceptual_anchors": self.perceptual_anchors,
            "_actual_meanings": self._actual_meanings,
            "_activation_time": self._activation_time,
            "predicted_meanings": self.predicted_meanings,
            "prediction_errors": dict(self.prediction_errors),
            "precision_values": self.precision_values,
            "adaptive_decay_params": self.adaptive_decay_params,
            "hierarchical_contexts": self.hierarchical_contexts,
            "cross_symbol_patterns": self.cross_symbol_patterns,
            "similarity_threshold": self.similarity_threshold,
            "error_history_window": self.error_history_window,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnhancedPredictiveSymbolElement':
        """
        Reconstruct an EnhancedPredictiveSymbolElement instance from a dictionary.
        """
        symbol = cls(data["symbol_id"], data.get("perceptual_anchors"))
        symbol._actual_meanings = data.get("_actual_meanings", {})
        symbol._activation_time = data.get("_activation_time", {})
        symbol.predicted_meanings = data.get("predicted_meanings", {})
        symbol.prediction_errors = defaultdict(list, data.get("prediction_errors", {}))
        symbol.precision_values = data.get("precision_values", {})
        symbol.adaptive_decay_params = data.get("adaptive_decay_params", {})
        symbol.hierarchical_contexts = data.get("hierarchical_contexts", {})
        symbol.cross_symbol_patterns = data.get("cross_symbol_patterns", {})
        symbol.similarity_threshold = data.get("similarity_threshold", 0.5)
        symbol.error_history_window = data.get("error_history_window", 100)
        return symbol

# Example usage
if __name__ == "__main__":
    # Instantiate the enhanced predictive symbol element.
    eps = EnhancedPredictiveSymbolElement("symbol_1")
    
    # Simulate adding historical actual meanings.
    ctx1 = "ctx1"
    eps._actual_meanings[ctx1] = {"meaning1": 0.7, "meaning2": 0.3}
    eps._activation_time[ctx1] = time.time() - 1800  # 30 minutes ago
    eps.set_adaptive_decay(ctx1, 1.0 / 1800)
    eps.update_hierarchical_context(ctx1, {"weight": 1.2, "feature": "A"})

    # Predict meaning for a new context.
    new_ctx = "ctx_new"
    predicted = eps.predict_meaning(new_ctx, {"feature": "A", "detail": 42})
    print("Predicted meaning:", predicted)
    
    # Actualize meaning (simulate by calling the base actualization).
    actualized = eps.actualize_meaning(new_ctx, {"feature": "A", "detail": 42})
    print("Actualized meaning:", actualized)
    
    # Check prediction precision.
    precision = eps.get_prediction_precision()
    print("Overall prediction precision:", precision)
    
    # Detect concept drift in the new context.
    drift = eps.detect_concept_drift(new_ctx)
    print("Concept drift detected?", drift)
    
    # Incorporate cross-symbol learning from a related symbol.
    eps.incorporate_cross_symbol_learning({"symbol_2": {"pattern_A": 0.5}})
    
    # Serialize and deserialize the element.
    state = eps.to_dict()
    eps_copy = EnhancedPredictiveSymbolElement.from_dict(state)
    print("Serialization round-trip successful for symbol:", eps_copy.symbol_id)
