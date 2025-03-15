import numpy as np
import time
from collections import defaultdict
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from asf.semantic_organization.temporal import AdaptiveTemporalMetadata

@dataclass
class SemanticRelation:
    """
    Enhanced edge in the semantic network with flexible weighting and attention.
    Supports hyperbolic geometry for hierarchical relationships and
    predictive processing for relation strength anticipation.
    """
    id: str
    source_id: str
    target_id: str
    relation_type: str
    weight: float = 0.8
    bidirectional: bool = False
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.7
    temporal_metadata: AdaptiveTemporalMetadata = field(default_factory=AdaptiveTemporalMetadata)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Enhanced properties
    attention_weight: float = 1.0  # For attention-weighted graph
    embedding: Optional[np.ndarray] = None  # Relation-specific embedding
    
    # Predictive properties
    anticipated_weights: Dict[str, float] = field(default_factory=dict)
    weight_prediction_errors: Dict[str, list] = field(default_factory=lambda: defaultdict(list))
    weight_precision: Dict[str, float] = field(default_factory=dict)
    
    def compute_effective_weight(self, context='default'):
        """Compute effective weight considering temporal relevance and attention."""
        temporal_factor = self.temporal_metadata.compute_relevance(context=context)
        return self.weight * temporal_factor * self.confidence * self.attention_weight
    
    def update_attention(self, new_attention):
        """Update attention weight for this relation."""
        self.attention_weight = new_attention
        self.temporal_metadata.update_modification()
        
    def anticipate_weight(self, context='default', factors=None):
        """
        Anticipate how this relation's weight might evolve in a given context.
        Implements Seth's "controlled hallucination" principle for relations.
        
        Args:
            context: Context identifier
            factors: Additional factors that might affect the weight
            
        Returns:
            Anticipated weight
        """
        # If we already have a prediction for this context, return it
        if context in self.anticipated_weights:
            return self.anticipated_weights[context]
        
        factors = factors or {}
        current_weight = self.weight
        
        # Base prediction on current weight
        prediction = current_weight
        
        # Adjust prediction based on factors
        if 'temporal_decay' in factors and factors['temporal_decay']:
            # Predict temporal decay effect
            time_factor = np.exp(-0.1 * factors.get('elapsed_time', 0) / 86400)
            prediction *= time_factor
            
        if 'competing_relation' in factors:
            # Predict effect of competing relation
            prediction *= 0.9
            
        if 'reinforcing_relation' in factors:
            # Predict effect of reinforcing relation
            prediction = min(1.0, prediction * 1.1)
            
        # Store prediction
        self.anticipated_weights[context] = prediction
        
        return prediction
    
    def update_weight(self, new_weight, context='default'):
        """
        Update weight and evaluate prediction accuracy.
        Implements Seth's prediction error minimization principle.
        
        Args:
            new_weight: New weight value
            context: Context in which the update occurs
            
        Returns:
            Prediction error
        """
        old_weight = self.weight
        self.weight = new_weight
        
        # If we had a prediction for this context, calculate error
        error = None
        if context in self.anticipated_weights:
            predicted = self.anticipated_weights[context]
            error = abs(predicted - new_weight)
            
            # Track error for precision calculation
            self.weight_prediction_errors[context].append(error)
            
            # Limit history size
            if len(self.weight_prediction_errors[context]) > 20:
                self.weight_prediction_errors[context] = self.weight_prediction_errors[context][-20:]
            
            # Update precision (inverse variance)
            if len(self.weight_prediction_errors[context]) > 1:
                variance = np.var(self.weight_prediction_errors[context])
                precision = 1.0 / (variance + 1e-6)  # Avoid division by zero
                self.weight_precision[context] = precision
        
        self.temporal_metadata.update_modification()
        return error
    
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'source_id': self.source_id,
            'target_id': self.target_id,
            'relation_type': self.relation_type,
            'weight': self.weight,
            'bidirectional': self.bidirectional,
            'properties': self.properties,
            'confidence': self.confidence,
            'attention_weight': self.attention_weight,
            'metadata': self.metadata,
            'prediction_metadata': {
                'contexts': list(self.anticipated_weights.keys()),
                'precision': {k: v for k, v in self.weight_precision.items()}
            }
        }
