import time
import math
import numpy as np
from collections import defaultdict
from typing import Any, Dict, List, Optional

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
        
        self.adaptive_decay_params: Dict[str, float] = {}          # context_hash -> decay parameter (1/half-life)
        self.hierarchical_contexts: Dict[str, Dict[str, Any]] = {}   # context_hash -> richer context info
        self.cross_symbol_patterns: Dict[str, Dict[str, float]] = {} # Related symbol pattern aggregation
        
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
        First generate a prediction, then actualize meaning (using the base implementation).
        
        Compares the prediction to the actual value, tracks error, and updates precision.
        Retrieve prediction precision.
        If a context hash is provided and available, return its precision;
        otherwise, compute an overall precision across all contexts.
        Detect concept drift by checking if the difference between the max and min 
        in a recent window of prediction errors exceeds a threshold.
        Update cross-symbol patterns with learning data from related symbols.
        
        related_symbols should map a symbol ID to its pattern dictionary.
        Serialize the internal state of the symbol element to a dictionary.
        Useful for persistence or state transfer.
        Reconstruct an EnhancedPredictiveSymbolElement instance from a dictionary.