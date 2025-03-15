import numpy as np
from asf.core.enums import EntityConfidenceState

class BayesianConfidenceUpdater:
    """
    Updates entity confidence using Bayesian inference.
    Philosophical Influence: Bayesian epistemology, Popper's falsifiability
    """
    def __init__(self, prior_alpha=1.0, prior_beta=1.0, decay_rate=0.95):
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.decay_rate = decay_rate
        self.entity_posteriors = {}  # Maps entity_id to (alpha, beta) parameters
        self.context_models = {}  # Context-specific prediction models
    
    def update_confidence(self, entity_id, observation_relevant, context_vector=None):
        """
        Updates entity confidence using Bayesian inference

        Parameters:
        - entity_id: Identifier for the perceptual entity
        - observation_relevant: Boolean indicating if entity was relevant in current context
        - context_vector: Optional vector representing current processing context
        """
        # Get current posterior parameters or initialize with priors
        if entity_id not in self.entity_posteriors:
            self.entity_posteriors[entity_id] = (self.prior_alpha, self.prior_beta)
        
        alpha, beta = self.entity_posteriors[entity_id]
        
        # Apply time decay to previous observations (gradually return to prior)
        alpha = self.prior_alpha + (alpha - self.prior_alpha) * self.decay_rate
        beta = self.prior_beta + (beta - self.prior_beta) * self.decay_rate
        
        # Update with new observation
        if observation_relevant:
            alpha += 1
        else:
            beta += 1
        
        # Store updated posterior parameters
        self.entity_posteriors[entity_id] = (alpha, beta)
        
        # Calculate confidence as mean of Beta distribution
        confidence = alpha / (alpha + beta)
        
        # Update context model if context provided
        if context_vector is not None:
            self._update_context_model(entity_id, context_vector, observation_relevant)
        
        return confidence
    
    def _update_context_model(self, entity_id, context_vector, was_relevant):
        """Update context-based prediction model for the entity"""
        if entity_id not in self.context_models:
            # Initialize model for this entity
            self.context_models[entity_id] = {
                "contexts": [],
                "relevance": []
            }
        
        # Add current observation to model
        self.context_models[entity_id]["contexts"].append(context_vector)
        self.context_models[entity_id]["relevance"].append(1.0 if was_relevant else 0.0)
        
        # Limit model size to prevent unbounded growth
        max_samples = 100
        if len(self.context_models[entity_id]["contexts"]) > max_samples:
            self.context_models[entity_id]["contexts"] = self.context_models[entity_id]["contexts"][-max_samples:]
            self.context_models[entity_id]["relevance"] = self.context_models[entity_id]["relevance"][-max_samples:]
    
    def predict_relevance(self, entity_id, context_vector):
        """
        Predicts entity relevance in a given context
        Returns a probability between 0 and 1
        """
        # Return posterior mean if no context model exists
        if entity_id not in self.entity_posteriors:
            return 0.5  # Prior mean if no data
        
        if entity_id not in self.context_models or not self.context_models[entity_id]["contexts"]:
            # Fall back to posterior mean if no context data
            alpha, beta = self.entity_posteriors[entity_id]
            return alpha / (alpha + beta)
        
        # Use context similarity to predict relevance
        contexts = np.array(self.context_models[entity_id]["contexts"])
        relevance = np.array(self.context_models[entity_id]["relevance"])
        
        # Convert context vector to numpy array if needed
        if not isinstance(context_vector, np.ndarray):
            context_vector = np.array(context_vector)
        
        # Compute similarities between current context and historical contexts
        similarities = self._compute_similarities(context_vector, contexts)
        
        # Weighted average of historical relevance by similarity
        weighted_relevance = np.sum(similarities * relevance) / np.sum(similarities)
        
        # Combine with base rate from posterior
        alpha, beta = self.entity_posteriors[entity_id]
        base_rate = alpha / (alpha + beta)
        
        # Weighted combination of context-based prediction and base rate
        # Higher similarity gives more weight to context-based prediction
        max_similarity = np.max(similarities) if len(similarities) > 0 else 0
        combined_prediction = max_similarity * weighted_relevance + (1 - max_similarity) * base_rate
        
        return combined_prediction
    
    def _compute_similarities(self, query_vector, context_vectors):
        """Compute cosine similarities between query and context vectors"""
        # Handle edge case of empty context vectors
        if len(context_vectors) == 0:
            return np.array([])
        
        # Normalize vectors
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            return np.zeros(len(context_vectors))
        query_vector = query_vector / query_norm
        
        # Compute similarities
        similarities = []
        for ctx in context_vectors:
            ctx_norm = np.linalg.norm(ctx)
            if ctx_norm == 0:
                similarities.append(0)
            else:
                ctx_normalized = ctx / ctx_norm
                similarity = np.dot(query_vector, ctx_normalized)
                similarities.append(max(0, similarity))  # Ensure non-negative
        
        return np.array(similarities)
    
    def get_confidence_state(self, confidence_score):
        """Convert confidence score to EntityConfidenceState"""
        if confidence_score > 0.8:
            return EntityConfidenceState.CANONICAL
        elif confidence_score > 0.5:
            return EntityConfidenceState.PROVISIONAL
        else:
            return EntityConfidenceState.UNVERIFIED
