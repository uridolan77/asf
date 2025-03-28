import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

class PrecisionWeightedLearningModule:
    """
    Implements precision-weighted learning based on predictive processing principles.
    
    This module dynamically adjusts learning rates based on prediction error and precision.
    It embodies the key idea from Seth's predictive processing framework that learning
    should be more aggressive when prediction errors are surprising (high error, high precision)
    and more conservative when errors are expected (high error, low precision).
    
    Key features:
    - Precision-weighted learning rate adjustment
    - Adaptive confidence updating
    - Temporal context tracking
    - Metalearning of optimal learning parameters
    """
    def __init__(self, config=None):
        self.config = config or {}
        self.learning_windows = {}  # Context -> list of learning windows
        self.precision_history = defaultdict(list)  # Context -> list of precision values
        self.error_history = defaultdict(list)  # Context -> list of error values
        self.adaptive_learning_rates = {}  # Context -> current learning rate
        self.update_counts = defaultdict(int)  # Context -> number of updates
        
        # Configuration parameters
        self.min_learning_rate = self.config.get('min_learning_rate', 0.05)
        self.max_learning_rate = self.config.get('max_learning_rate', 0.9)
        self.base_learning_rate = self.config.get('base_learning_rate', 0.3)
        self.surprise_weight = self.config.get('surprise_weight', 0.6)
        self.precision_trend_threshold = self.config.get('precision_trend_threshold', 0.1)
        self.history_window_size = self.config.get('history_window_size', 20)
        self.learning_rate_smoothing = self.config.get('learning_rate_smoothing', 0.7)  # Smoothing factor
        
        # Metalearning parameters (learn how to learn)
        self.meta_learning_enabled = self.config.get('meta_learning_enabled', True)
        self.meta_learning_rate = self.config.get('meta_learning_rate', 0.01)
        self.parameter_history = {
            'surprise_weight': [],
            'learning_rate_smoothing': []
        }
        
        # Domain-specific adaptation
        self.domain_specific_rates = defaultdict(dict)  # Domain -> parameter mapping
        
    def calculate_learning_rate(self, context_key: str, prediction_error: float, 
                               precision: float, domain: Optional[str] = None) -> float:
        """
        Calculate optimal learning rate based on prediction error and precision.
        Higher precision = more confident updates for surprising inputs.
        Lower precision = conservative updates even for surprising inputs.
        
        Args:
            context_key: Context identifier
            prediction_error: Current prediction error magnitude
            precision: Current precision value (inverse variance)
            domain: Optional domain identifier for domain-specific adaptation
            
        Returns:
            Optimal learning rate for this update
        """
        # Cap precision to avoid extreme values
        capped_precision = min(10.0, max(0.1, precision))
        
        # Get domain-specific parameters if available
        if domain and domain in self.domain_specific_rates:
            domain_params = self.domain_specific_rates[domain]
            surprise_weight = domain_params.get('surprise_weight', self.surprise_weight)
            smoothing = domain_params.get('learning_rate_smoothing', self.learning_rate_smoothing)
        else:
            surprise_weight = self.surprise_weight
            smoothing = self.learning_rate_smoothing
        
        # Calculate surprise (normalized prediction error weighted by precision)
        # Higher precision means errors are more surprising
        surprise = min(1.0, prediction_error * capped_precision)
        
        # Base learning rate depends on surprise level
        base_rate = self.min_learning_rate + ((self.max_learning_rate - self.min_learning_rate) * 
                                            surprise * surprise_weight)
        
        # Modify based on precision trend
        if context_key in self.precision_history and len(self.precision_history[context_key]) > 5:
            recent_precision = self.precision_history[context_key][-5:]
            precision_trend = self._calculate_trend(recent_precision)
            
            # If precision is improving, we can be more aggressive in learning
            if precision_trend > self.precision_trend_threshold:  # Significant improvement
                base_rate *= 1.2
            # If precision is degrading, be more conservative
            elif precision_trend < -self.precision_trend_threshold:  # Significant degradation
                base_rate *= 0.8
        
        # Apply smoothing with previous learning rate if available
        if context_key in self.adaptive_learning_rates:
            previous_rate = self.adaptive_learning_rates[context_key]
            smoothed_rate = (previous_rate * smoothing) + (base_rate * (1 - smoothing))
        else:
            smoothed_rate = base_rate
        
        # Ensure learning rate stays within bounds
        final_rate = max(self.min_learning_rate, min(self.max_learning_rate, smoothed_rate))
        
        # Track history
        self._update_history(context_key, prediction_error, precision, final_rate)
        
        # Update metalearning if enabled
        if self.meta_learning_enabled and self.update_counts[context_key] % 10 == 0:
            self._update_metalearning(context_key)
        
        return final_rate
    
    def update_confidence(self, current_confidence: float, new_evidence: float, 
                         context_key: str, precision: float, domain: Optional[str] = None) -> float:
        """
        Update confidence using precision-weighted learning.
        
        Args:
            current_confidence: Current confidence value
            new_evidence: New evidence (0-1 scale)
            context_key: Context identifier
            precision: Current precision value
            domain: Optional domain identifier
            
        Returns:
            Updated confidence value
        """
        # Calculate prediction error
        prediction_error = abs(current_confidence - new_evidence)
        
        # Get optimal learning rate
        learning_rate = self.calculate_learning_rate(context_key, prediction_error, precision, domain)
        
        # Update confidence using the learning rate
        updated_confidence = current_confidence + (learning_rate * (new_evidence - current_confidence))
        
        # Ensure confidence stays within bounds
        updated_confidence = max(0.0, min(1.0, updated_confidence))
        
        return updated_confidence
    
    def get_confidence_interval(self, confidence: float, precision: float) -> Tuple[float, float]:
        """
        Calculate confidence interval based on confidence and precision.
        
        Args:
            confidence: Current confidence value
            precision: Precision value (inverse variance)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        # Higher precision = narrower interval
        interval_width = 1.0 / (precision + 1.0)
        
        # Calculate bounds
        lower_bound = max(0.0, confidence - interval_width)
        upper_bound = min(1.0, confidence + interval_width)
        
        return (lower_bound, upper_bound)
    
    def adaptive_decay(self, confidence: float, elapsed_time: float, 
                      domain: str, context_key: Optional[str] = None) -> float:
        """
        Apply adaptive temporal decay to confidence based on domain characteristics.
        
        Args:
            confidence: Current confidence value
            elapsed_time: Time since last update (seconds)
            domain: Knowledge domain
            context_key: Optional specific context
            
        Returns:
            Decayed confidence value
        """
        # Get domain-specific decay rate if available
        if domain in self.domain_specific_rates:
            decay_rate = self.domain_specific_rates[domain].get('decay_rate', 0.1)
        else:
            decay_rate = 0.1  # Default decay rate
        
        # Adjust decay based on precision if context_key provided
        if context_key and context_key in self.precision_history and self.precision_history[context_key]:
            precision = self.precision_history[context_key][-1]
            # Higher precision = slower decay (more confident in knowledge)
            decay_modifier = 1.0 / (1.0 + precision * 0.5)
            adjusted_decay = decay_rate * decay_modifier
        else:
            adjusted_decay = decay_rate
        
        # Apply exponential decay
        decay_factor = np.exp(-adjusted_decay * elapsed_time / 86400)  # Normalized to days
        decayed_confidence = confidence * decay_factor
        
        return decayed_confidence
    
    def register_domain(self, domain: str, parameters: Dict[str, Any]) -> None:
        """
        Register domain-specific learning parameters.
        
        Args:
            domain: Domain identifier
            parameters: Parameter mapping for this domain
        """
        self.domain_specific_rates[domain] = parameters
    
    def _update_history(self, context_key: str, error: float, precision: float, learning_rate: float) -> None:
        """Update historical data for a context."""
        # Track precision history
        self.precision_history[context_key].append(precision)
        if len(self.precision_history[context_key]) > self.history_window_size:
            self.precision_history[context_key] = self.precision_history[context_key][-self.history_window_size:]
        
        # Track error history
        self.error_history[context_key].append(error)
        if len(self.error_history[context_key]) > self.history_window_size:
            self.error_history[context_key] = self.error_history[context_key][-self.history_window_size:]
        
        # Store learning rate
        self.adaptive_learning_rates[context_key] = learning_rate
        
        # Increment update count
        self.update_counts[context_key] += 1
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in a series of values."""
        if not values or len(values) < 2:
            return 0.0
        
        # Simple trend calculation: relative change from first to last
        first_value = max(values[0], 0.0001)  # Avoid division by zero
        last_value = values[-1]
        
        return (last_value - first_value) / first_value
    
    def _update_metalearning(self, context_key: str) -> None:
        """Update metalearning parameters based on learning success."""
        if context_key not in self.error_history or len(self.error_history[context_key]) < 10:
            return
        
        # Check if errors are decreasing
        recent_errors = self.error_history[context_key][-10:]
        error_trend = self._calculate_trend(recent_errors)
        
        # If errors are decreasing, current parameters are working well
        if error_trend < -0.05:  # Significant improvement
            return
        
        # If errors are not decreasing or increasing, adjust parameters
        # Experiment with slightly different surprise_weight
        current_surprise_weight = self.surprise_weight
        delta = self.meta_learning_rate * (1.0 if np.random.random() > 0.5 else -1.0)
        new_surprise_weight = max(0.1, min(0.9, current_surprise_weight + delta))
        
        # Store adjustment
        self.parameter_history['surprise_weight'].append(new_surprise_weight)
        if len(self.parameter_history['surprise_weight']) > 100:
            self.parameter_history['surprise_weight'] = self.parameter_history['surprise_weight'][-100:]
        
        # Apply change
        self.surprise_weight = new_surprise_weight
        
        # Similarly adjust learning_rate_smoothing
        current_smoothing = self.learning_rate_smoothing
        delta = self.meta_learning_rate * 0.5 * (1.0 if np.random.random() > 0.5 else -1.0)
        new_smoothing = max(0.3, min(0.95, current_smoothing + delta))
        
        # Store adjustment
        self.parameter_history['learning_rate_smoothing'].append(new_smoothing)
        if len(self.parameter_history['learning_rate_smoothing']) > 100:
            self.parameter_history['learning_rate_smoothing'] = self.parameter_history['learning_rate_smoothing'][-100:]
        
        # Apply change
        self.learning_rate_smoothing = new_smoothing
    
    def get_learning_statistics(self, context_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Get learning statistics for monitoring and debugging.
        
        Args:
            context_key: Optional specific context
            
        Returns:
            Dictionary of learning statistics
        """
        stats = {
            'global_parameters': {
                'surprise_weight': self.surprise_weight,
                'learning_rate_smoothing': self.learning_rate_smoothing,
                'min_learning_rate': self.min_learning_rate,
                'max_learning_rate': self.max_learning_rate
            },
            'domains': {domain: params for domain, params in self.domain_specific_rates.items()},
            'meta_learning': {
                'enabled': self.meta_learning_enabled,
                'parameter_history': {k: v[-5:] if v else [] for k, v in self.parameter_history.items()}
            }
        }
        
        if context_key:
            stats['context'] = {
                'learning_rate': self.adaptive_learning_rates.get(context_key),
                'precision_history': self.precision_history.get(context_key, [])[-5:],
                'error_history': self.error_history.get(context_key, [])[-5:],
                'update_count': self.update_counts.get(context_key, 0)
            }
        else:
            # Summary across all contexts
            all_learning_rates = list(self.adaptive_learning_rates.values())
            all_precisions = [p[-1] for p in self.precision_history.values() if p]
            stats['summary'] = {
                'context_count': len(self.adaptive_learning_rates),
                'avg_learning_rate': np.mean(all_learning_rates) if all_learning_rates else None,
                'avg_precision': np.mean(all_precisions) if all_precisions else None,
                'total_updates': sum(self.update_counts.values())
            }
        
        return stats