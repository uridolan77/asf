# Enhancement for semantic_node.py - Add to existing file

import time
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Set, Tuple
from collections import defaultdict

# Keep existing SemanticNode class and add these methods:

class SemanticNode:
    # Existing initialization and properties...
    
    # Add these fields to __init__ or __post_init__:
    # self.anticipated_activations = {}  # Context -> predicted activation
    # self.activation_errors = defaultdict(list)  # Context -> list of errors
    # self.precision_values = {}  # Context -> precision value
    # self.anticipated_properties = {}  # Context -> predicted properties
    
    def anticipate_activation(self, context: Dict[str, Any], stimulus_sources: Optional[List[str]] = None) -> float:
        """
        Anticipate activation level before receiving actual stimulus.
        Implements Seth's "controlled hallucination" principle.
        
        Args:
            context: Current processing context
            stimulus_sources: Optional list of stimulus source IDs
            
        Returns:
            Anticipated activation level (0-1)
        """
        context_key = self._get_context_key(context)
        
        # If we've seen this exact context before, use cached prediction
        if hasattr(self, 'anticipated_activations') and context_key in self.anticipated_activations:
            return self.anticipated_activations[context_key]
        
        # Initialize anticipation fields if not present
        if not hasattr(self, 'anticipated_activations'):
            self.anticipated_activations = {}
        if not hasattr(self, 'activation_errors'):
            self.activation_errors = defaultdict(list)
        if not hasattr(self, 'precision_values'):
            self.precision_values = {}
        
        # Base prediction on similar contexts we've seen before
        similar_contexts = self._find_similar_contexts(context)
        
        if not similar_contexts:
            # No similar contexts, use current activation as prediction
            prediction = self.activation
        else:
            # Calculate prediction as weighted average of activations in similar contexts
            total_weight = 0.0
            weighted_sum = 0.0
            
            for sim_context, similarity in similar_contexts:
                if sim_context in self.precision_values:
                    # Weight by both similarity and precision
                    precision = self.precision_values[sim_context]
                    weight = similarity * precision
                else:
                    weight = similarity
                
                # Retrieve past activation for this context
                past_activation = self._get_past_activation(sim_context)
                
                weighted_sum += past_activation * weight
                total_weight += weight
            
            if total_weight > 0:
                prediction = weighted_sum / total_weight
            else:
                prediction = self.activation
        
        # Store prediction
        self.anticipated_activations[context_key] = prediction
        
        return prediction
    
    def update_activation(self, stimulus: float, decay_rate: float = 0.95) -> float:
        """
        Update activation level and evaluate prediction accuracy.
        Uses leaky integrator model with MGC-inspired maximization.
        
        Args:
            stimulus: Input activation stimulus
            decay_rate: Rate of activation decay
            
        Returns:
            Updated activation level
        """
        # Get current context
        context_key = self._get_current_context_key()
        
        # Check if we made a prediction for this context
        if hasattr(self, 'anticipated_activations') and context_key in self.anticipated_activations:
            predicted = self.anticipated_activations[context_key]
            
            # Calculate prediction error
            prediction_error = abs(predicted - stimulus)
            
            # Initialize if needed
            if not hasattr(self, 'activation_errors'):
                self.activation_errors = defaultdict(list)
            if not hasattr(self, 'precision_values'):
                self.precision_values = {}
            
            # Track error for this context
            self.activation_errors[context_key].append(prediction_error)
            
            # Limit history size
            if len(self.activation_errors[context_key]) > 20:
                self.activation_errors[context_key] = self.activation_errors[context_key][-20:]
            
            # Update precision (inverse variance)
            if len(self.activation_errors[context_key]) > 1:
                variance = np.var(self.activation_errors[context_key])
                precision = 1.0 / (variance + 1e-6)  # Avoid division by zero
                self.precision_values[context_key] = precision
        
        # MGC-inspired maximization-based update
        self.activation = max(
            decay_rate * self.activation,
            stimulus
        )
        
        self.temporal_metadata.update_access()
        return self.activation
    
    def anticipate_property_changes(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Anticipate how properties might change in a given context.
        
        Args:
            context: Processing context
            
        Returns:
            Dictionary of anticipated property values
        """
        context_key = self._get_context_key(context)
        
        # Initialize if needed
        if not hasattr(self, 'anticipated_properties'):
            self.anticipated_properties = {}
        
        # If we already predicted for this context, return cached prediction
        if context_key in self.anticipated_properties:
            return self.anticipated_properties[context_key]
        
        # Find similar contexts where properties changed
        similar_contexts = self._find_similar_contexts(context)
        
        predictions = {}
        
        # Look at property changes in similar contexts
        for prop_name in self.properties:
            prop_predictions = []
            
            for sim_context, similarity in similar_contexts:
                # Check if we have record of this property changing in similar context
                change_key = f"{sim_context}_{prop_name}"
                if 'property_changes' in self.metadata and change_key in self.metadata['property_changes']:
                    change_record = self.metadata['property_changes'][change_key]
                    # Weight by similarity and recency
                    time_factor = np.exp(-(time.time() - change_record['time']) / 86400)  # Decay over a day
                    weight = similarity * time_factor
                    prop_predictions.append((change_record['new_value'], weight))
            
            # If we have predictions for this property, calculate weighted average
            if prop_predictions:
                if all(isinstance(p[0], (int, float)) for p in prop_predictions):
                    # For numeric properties, weighted average
                    total_weight = sum(w for _, w in prop_predictions)
                    weighted_sum = sum(v * w for v, w in prop_predictions)
                    if total_weight > 0:
                        predictions[prop_name] = weighted_sum / total_weight
                else:
                    # For non-numeric, use most strongly weighted value
                    predictions[prop_name] = max(prop_predictions, key=lambda x: x[1])[0]
        
        # Store predictions
        self.anticipated_properties[context_key] = predictions
        
        return predictions
    
    def _get_context_key(self, context: Dict[str, Any]) -> str:
        """Generate a stable hash key for a context."""
        # Simple context key generation - can be enhanced with better hashing
        sorted_items = sorted((str(k), str(v)) for k, v in context.items())
        return '_'.join(f"{k}:{v}" for k, v in sorted_items)
    
    def _get_current_context_key(self) -> str:
        """Get context key for current state."""
        # In real implementation, this would capture current processing context
        return f"ctx_{int(time.time())}"
    
    def _find_similar_contexts(self, context: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Find contexts similar to the given context with similarity scores."""
        # Initialize if needed
        if not hasattr(self, 'anticipated_activations'):
            self.anticipated_activations = {}
        if not hasattr(self, 'precision_values'):
            self.precision_values = {}
            
        context_key = self._get_context_key(context)
        context_keys = set(self.anticipated_activations.keys()) | set(key for key in self.precision_values.keys())
        
        similar_contexts = []
        
        for key in context_keys:
            similarity = self._calculate_context_similarity(context_key, key)
            if similarity > 0.2:  # Threshold for considering contexts similar
                similar_contexts.append((key, similarity))
        
        # Sort by similarity (descending)
        return sorted(similar_contexts, key=lambda x: -x[1])
    
    def _calculate_context_similarity(self, context_key1: str, context_key2: str) -> float:
        """Calculate similarity between two context keys."""
        # Simple implementation - in production, would use semantic similarity
        tokens1 = set(context_key1.split('_'))
        tokens2 = set(context_key2.split('_'))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union)
    
    def _get_past_activation(self, context_key: str) -> float:
        """Retrieve past activation for a context."""
        # In real implementation, would store activation history
        return self.activation  # Default to current activation
