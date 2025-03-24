import numpy as np
from asf.__core.enums import EntityConfidenceState
from asf.layer1_knowledge_substrate.confidence.bayesian_updater import BayesianConfidenceUpdater

class PredictiveConfidenceUpdater(BayesianConfidenceUpdater):
    """
    Enhanced Bayesian updater incorporating Seth's predictive processing principles.
    Extends standard Bayesian updating with prediction error tracking and adaptive learning rates.
    
    Philosophical Influence: Seth's Predictive Processing, Friston's Free Energy Principle
    """
    
    def __init__(self, prior_alpha=1.0, prior_beta=1.0, decay_rate=0.95):
        super().__init__(prior_alpha, prior_beta, decay_rate)
        # Entity-specific prediction models
        self.prediction_models = {}
        # Track prediction errors to calculate precision
        self.prediction_errors = {}
        # Adaptive learning rates based on prediction error
        self.adaptive_learning_rates = {}
        # Default adaptive range
        self.min_learning_rate = 0.1
        self.max_learning_rate = 0.9
        
    def update_confidence(self, entity_id, observation_relevant, context_vector=None):
        """
        Updates entity confidence using predictive processing principles
        
        Parameters:
        - entity_id: Identifier for the perceptual entity
        - observation_relevant: Boolean indicating if entity was relevant in current context
        - context_vector: Optional vector representing current processing context
        
        Returns: Updated confidence value
        """
        # First, generate prediction based on context
        predicted_relevance = self.predict_relevance(entity_id, context_vector)
        
        # Calculate prediction error (difference between prediction and actual observation)
        prediction_error = abs(float(observation_relevant) - predicted_relevance)
        
        # Track prediction errors for this entity (for precision calculation)
        if entity_id not in self.prediction_errors:
            self.prediction_errors[entity_id] = []
        self.prediction_errors[entity_id].append(prediction_error)
        
        # Limit history size to prevent unbounded growth
        if len(self.prediction_errors[entity_id]) > 20:
            self.prediction_errors[entity_id] = self.prediction_errors[entity_id][-20:]
        
        # Calculate precision (inverse variance of prediction errors)
        if len(self.prediction_errors[entity_id]) > 1:
            precision = 1.0 / (np.var(self.prediction_errors[entity_id]) + 1e-6)
            # Normalize precision to a reasonable range
            precision = min(10.0, precision)  # Cap very high precision
        else:
            precision = 1.0  # Default precision
        
        # Adjust learning rate based on prediction error
        # Higher error = higher learning rate (to learn faster from surprising events)
        # Higher precision = lower learning rate (already well-predicted)
        learning_rate = min(self.max_learning_rate, 
                           max(self.min_learning_rate, 
                               prediction_error * (1.0 / precision)))
        
        # Store the adaptive learning rate
        self.adaptive_learning_rates[entity_id] = learning_rate
        
        # Get current posterior parameters or initialize with priors
        if entity_id not in self.entity_posteriors:
            self.entity_posteriors[entity_id] = (self.prior_alpha, self.prior_beta)
        
        alpha, beta = self.entity_posteriors[entity_id]
        
        # Apply time decay to previous observations (gradually return to prior)
        # Seth's insight: stronger decay for contexts with high uncertainty (low precision)
        decay_modifier = 1.0 / (1.0 + precision * 0.1)  # More decay with lower precision
        effective_decay = self.decay_rate * (1.0 + decay_modifier)
        effective_decay = min(0.99, effective_decay)  # Ensure decay rate is reasonable
        
        alpha = self.prior_alpha + (alpha - self.prior_alpha) * effective_decay
        beta = self.prior_beta + (beta - self.prior_beta) * effective_decay
        
        # Apply Bayesian update with weighted evidence based on learning rate
        if observation_relevant:
            alpha += learning_rate  # Weight evidence by learning rate
        else:
            beta += learning_rate  # Weight evidence by learning rate
        
        # Store updated posterior parameters
        self.entity_posteriors[entity_id] = (alpha, beta)
        
        # Calculate confidence as mean of Beta distribution
        confidence = alpha / (alpha + beta)
        
        # Update context model if context provided
        if context_vector is not None:
            self._update_context_model(entity_id, context_vector, observation_relevant)
        
        return confidence
    
    def predict_relevance(self, entity_id, context_vector=None):
        """
        Predicts entity relevance in a given context
        Incorporates precision weighting for more accurate predictions
        
        Parameters:
        - entity_id: ID of entity to evaluate
        - context_vector: Context for prediction
        
        Returns: Probability between 0 and 1
        """
        # If no context or no prior data, use standard prediction
        if context_vector is None or entity_id not in self.entity_posteriors:
            return super().predict_relevance(entity_id, context_vector)
        
        # Get precision for this entity
        if entity_id in self.prediction_errors and len(self.prediction_errors[entity_id]) > 1:
            precision = 1.0 / (np.var(self.prediction_errors[entity_id]) + 1e-6)
            precision = min(10.0, precision)  # Cap very high precision
        else:
            precision = 1.0  # Default precision
        
        # Get standard prediction
        standard_prediction = super().predict_relevance(entity_id, context_vector)
        
        # For high precision (well-learned predictions), rely more on the model
        # For low precision (uncertain predictions), regress toward prior
        alpha, beta = self.entity_posteriors[entity_id]
        prior_mean = alpha / (alpha + beta)
        
        # Calculate precision-weighted prediction
        # Higher precision = more weight to model prediction
        # Lower precision = more weight to prior
        precision_weight = precision / (precision + 1.0)  # Normalize to 0-1 range
        prediction = (precision_weight * standard_prediction + 
                     (1 - precision_weight) * prior_mean)
        
        return prediction
    
    def get_prediction_metrics(self, entity_id):
        """
        Get prediction-related metrics for an entity
        
        Returns dictionary with metrics including prediction error, precision,
        and adaptive learning rate
        """
        if entity_id not in self.prediction_errors:
            return {
                "avg_prediction_error": None,
                "prediction_precision": None,
                "adaptive_learning_rate": None
            }
        
        errors = self.prediction_errors.get(entity_id, [])
        avg_error = sum(errors) / len(errors) if errors else None
        
        precision = 1.0 / (np.var(errors) + 1e-6) if len(errors) > 1 else None
        precision = min(10.0, precision) if precision is not None else None
        
        learning_rate = self.adaptive_learning_rates.get(entity_id)
        
        return {
            "avg_prediction_error": avg_error,
            "prediction_precision": precision,
            "adaptive_learning_rate": learning_rate
        }
