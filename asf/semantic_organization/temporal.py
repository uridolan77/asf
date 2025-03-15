import time
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class AdaptiveTemporalMetadata:
    """
    Enhanced temporal metadata with adaptive decay based on context.
    Supports multi-resolution temporal tracking with confidence weighting.
    Implements Seth's principle that perception is shaped by temporal context.
    """
    creation_time: float = field(default_factory=lambda: time.time())
    last_accessed: float = field(default_factory=lambda: time.time())
    last_modified: float = field(default_factory=lambda: time.time())
    access_count: int = 0
    modification_count: int = 0
    contextual_half_lives: Dict[str, float] = field(default_factory=lambda: {
        'default': 86400 * 7,  # 7 days
        'critical': 86400 * 30,  # 30 days
        'ephemeral': 3600  # 1 hour
    })
    
    # Seth's Data Paradox enhancements
    predicted_relevance: Dict[str, float] = field(default_factory=dict)
    relevance_errors: Dict[str, list] = field(default_factory=lambda: {})
    relevance_precision: Dict[str, float] = field(default_factory=dict)
    
    def update_access(self):
        """Update access time and count when entity is accessed."""
        self.last_accessed = time.time()
        self.access_count += 1
        
    def update_modification(self):
        """Update modification time when entity is modified."""
        self.last_modified = time.time()
        self.modification_count += 1
    
    def compute_relevance(self, current_time=None, context='default'):
        """Compute temporal relevance using adaptive weighting."""
        if current_time is None:
            current_time = time.time()
            
        elapsed = current_time - self.last_accessed
        half_life = self.contextual_half_lives.get(context, 
                                                 self.contextual_half_lives['default'])
        
        # Dynamic weight based on access frequency
        dynamic_weight = min(1.0, 0.5 + (self.access_count / 20))
        
        # Logarithmic decay is gentler for frequently accessed items
        base_relevance = 1.0 / (1.0 + np.log1p(elapsed / half_life))
        
        # Apply dynamic weighting
        return base_relevance * dynamic_weight
    
    def compute_freshness(self, current_time=None):
        """Compute how fresh/recent a modification is."""
        if current_time is None:
            current_time = time.time()
            
        elapsed = current_time - self.last_modified
        
        # Exponential decay function for freshness
        decay_rate = 0.1  # Controls decay speed
        freshness = np.exp(-decay_rate * elapsed / 86400)  # Normalized to days
        
        return freshness
    
    def should_retain(self, current_time=None, context='default', threshold=0.2):
        """Determine if this entity should be retained in memory."""
        relevance = self.compute_relevance(current_time, context)
        freshness = self.compute_freshness(current_time)
        
        # Combined score with higher weight on relevance
        retention_score = relevance * 0.7 + freshness * 0.3
        
        return retention_score > threshold
    
    def anticipate_relevance(self, future_time, context='default'):
        """
        Anticipate future relevance of this entity.
        Implements Seth's predictive principle for temporal perception.
        
        Args:
            future_time: Future time point to predict relevance for
            context: Context for prediction
            
        Returns:
            Predicted relevance score
        """
        prediction_key = f"{context}_{int(future_time)}"
        
        # If we already have a prediction, return it
        if prediction_key in self.predicted_relevance:
            return self.predicted_relevance[prediction_key]
        
        # Calculate time delta
        current_time = time.time()
        delta = future_time - current_time
        
        # Base prediction on current relevance
        current_relevance = self.compute_relevance(current_time, context)
        
        # Predict future relevance based on contextual half-life decay
        half_life = self.contextual_half_lives.get(context, 
                                                self.contextual_half_lives['default'])
        
        # Calculate decay factor (exponential decay)
        decay_factor = np.exp(-0.693 * delta / half_life)  # 0.693 is ln(2)
        
        # Adjust for predicted access or modifications
        expected_accesses = (self.access_count / max(1, current_time - self.creation_time)) * delta
        access_factor = min(1.2, 1.0 + (expected_accesses * 0.1))
        
        # Calculate predicted relevance
        predicted_relevance = current_relevance * decay_factor * access_factor
        
        # Store prediction
        self.predicted_relevance[prediction_key] = predicted_relevance
        
        return predicted_relevance
    
    def update_relevance_prediction(self, future_time, actual_relevance, context='default'):
        """
        Update relevance predictions with actual observed relevance.
        Implements Seth's error minimization principle.
        
        Args:
            future_time: The time point that was predicted
            actual_relevance: The actual observed relevance
            context: Context for prediction
            
        Returns:
            Prediction error
        """
        prediction_key = f"{context}_{int(future_time)}"
        
        # If we didn't predict for this time, nothing to update
        if prediction_key not in self.predicted_relevance:
            return None
        
        # Calculate prediction error
        predicted = self.predicted_relevance[prediction_key]
        error = abs(predicted - actual_relevance)
        
        # Initialize error tracking for this context if needed
        if context not in self.relevance_errors:
            self.relevance_errors[context] = []
        
        # Track error for precision calculation
        self.relevance_errors[context].append(error)
        
        # Limit history size
        if len(self.relevance_errors[context]) > 20:
            self.relevance_errors[context] = self.relevance_errors[context][-20:]
        
        # Update precision (inverse variance)
        if len(self.relevance_errors[context]) > 1:
            variance = np.var(self.relevance_errors[context])
            precision = 1.0 / (variance + 1e-6)  # Avoid division by zero
            self.relevance_precision[context] = precision
        
        return error
