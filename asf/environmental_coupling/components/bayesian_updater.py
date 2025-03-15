import asyncio
import time
import uuid
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict

class EnhancedBayesianUpdater:
    """
    Enhanced Bayesian updater with precision-weighted prediction errors.
    Implements Seth's principle of weighting updates by precision.
    """
    def __init__(self):
        self.coupling_models = {}  # Maps coupling_id to Bayesian models
        self.update_history = defaultdict(list)  # Maps coupling_id to update history
        self.precision_values = {}  # Maps coupling_id to precision value
        self.logger = logging.getLogger("ASF.Layer4.EnhancedBayesianUpdater")
        
    async def initialize_coupling_model(self, coupling_id):
        """Initialize Bayesian model for a new coupling."""
        self.coupling_models[coupling_id] = {
            'prior_confidence': 0.5,
            'likelihood_estimates': {},
            'last_updated': time.time(),
            'update_count': 0
        }
        self.precision_values[coupling_id] = 1.0  # Default precision
        return True
        
    async def update_from_interaction(self, coupling_id, interaction_data, interaction_type, confidence):
        """
        Update coupling Bayesian model with precision weighting.
        Implements Seth's precision-weighted error principle.
        """
        start_time = time.time()
        
        if coupling_id not in self.coupling_models:
            await self.initialize_coupling_model(coupling_id)
            
        model = self.coupling_models[coupling_id]
        
        # Extract key features from interaction data
        features = self._extract_features(interaction_data)
        
        # Get current prior confidence
        prior = model['prior_confidence']
        
        # Calculate likelihood based on feature match
        likelihood = self._calculate_likelihood(features, model, interaction_type)
        
        # Get precision for this coupling
        precision = self.precision_values.get(coupling_id, 1.0)
        
        # Weight confidence by precision (Seth's principle)
        weighted_confidence = confidence * precision
        
        # Apply Bayes' rule with weighted confidence
        posterior = self._apply_bayes_rule(prior, likelihood, weighted_confidence)
        
        # Determine effect on coupling strength
        strength_delta = (posterior - prior) * 0.2  # Scale factor
        
        # Update model
        model['prior_confidence'] = posterior
        model['last_updated'] = time.time()
        model['update_count'] += 1
        
        # Update likelihood estimates
        if interaction_type not in model['likelihood_estimates']:
            model['likelihood_estimates'][interaction_type] = {}
            
        for feature, value in features.items():
            feature_key = f"{feature}:{value}"
            if feature_key not in model['likelihood_estimates'][interaction_type]:
                model['likelihood_estimates'][interaction_type][feature_key] = 0.5
                
            # Update likelihood estimate using exponential moving average
            current_estimate = model['likelihood_estimates'][interaction_type][feature_key]
            update_weight = 0.2  # Learning rate
            new_estimate = (1 - update_weight) * current_estimate + update_weight * weighted_confidence
            model['likelihood_estimates'][interaction_type][feature_key] = new_estimate
        
        # Check for contradictions in the data
        contradiction = self._check_for_contradictions(interaction_data, model)
        
        # Add to update history
        self.update_history[coupling_id].append({
            'timestamp': time.time(),
            'interaction_type': interaction_type,
            'prior': prior,
            'likelihood': likelihood,
            'posterior': posterior,
            'features': len(features),
            'weighted_confidence': weighted_confidence,
            'precision': precision,
            'contradiction_detected': contradiction is not None
        })
        
        # Limit history size
        if len(self.update_history[coupling_id]) > 50:
            self.update_history[coupling_id] = self.update_history[coupling_id][-50:]
        
        return {
            'prior_confidence': prior,
            'new_confidence': posterior,
            'likelihood': likelihood,
            'strength_delta': strength_delta,
            'precision_applied': precision,
            'contradiction_detected': contradiction is not None,
            'contradiction': contradiction,
            'features_processed': len(features),
            'elapsed_time': time.time() - start_time
        }
    
    async def update_precision(self, coupling_id, prediction_error):
        """
        Update precision based on prediction error.
        Implements Seth's precision updating mechanism.
        """
        if coupling_id not in self.precision_values:
            self.precision_values[coupling_id] = 1.0
            
        # In a real implementation, this would use a more sophisticated
        # approach like Kalman filtering or variational inference
        # For this example, we use a simple inverse variance calculation
        
        # Track error history
        if coupling_id not in self.update_history:
            self.update_history[coupling_id] = []
            
        # Add error to history
        self.update_history[coupling_id].append({
            'timestamp': time.time(),
            'prediction_error': prediction_error
        })
        
        # Limit history size
        if len(self.update_history[coupling_id]) > 20:
            self.update_history[coupling_id] = self.update_history[coupling_id][-20:]
            
        # Calculate precision as inverse variance of recent errors
        recent_errors = [update.get('prediction_error', 0) 
                         for update in self.update_history[coupling_id]
                         if 'prediction_error' in update]
        
        if len(recent_errors) > 1:
            variance = np.var(recent_errors)
            # Avoid division by zero and excessive weights
            new_precision = min(10.0, 1.0 / (variance + 1e-6))
            self.precision_values[coupling_id] = new_precision
            
        return self.precision_values[coupling_id]
        
    async def get_precision(self, coupling_id):
        """Get current precision for a coupling."""
        return self.precision_values.get(coupling_id, 1.0)
    
    async def cleanup_coupling_model(self, coupling_id):
        """Clean up resources for a terminated coupling."""
        if coupling_id in self.coupling_models:
            del self.coupling_models[coupling_id]
        if coupling_id in self.update_history:
            del self.update_history[coupling_id]
        if coupling_id in self.precision_values:
            del self.precision_values[coupling_id]
        return True
        
    async def perform_maintenance(self):
        """Perform periodic maintenance on Bayesian models."""
        start_time = time.time()
        
        # Count models
        model_count = len(self.coupling_models)
        
        # Identify inactive models
        current_time = time.time()
        inactive_threshold = 86400  # 24 hours
        inactive_models = [
            coupling_id for coupling_id, model in self.coupling_models.items()
            if current_time - model['last_updated'] > inactive_threshold
        ]
        
        # Compact history for all models
        for coupling_id in self.update_history:
            if len(self.update_history[coupling_id]) > 50:
                self.update_history[coupling_id] = self.update_history[coupling_id][-50:]
        
        return {
            'model_count': model_count,
            'inactive_models': len(inactive_models),
            'precision_values': len(self.precision_values),
            'elapsed_time': time.time() - start_time
        }
        
    async def get_metrics(self):
        """Get metrics about the enhanced Bayesian updater."""
        return {
            'model_count': len(self.coupling_models),
            'avg_update_count': np.mean([model['update_count'] for model in self.coupling_models.values()]) if self.coupling_models else 0,
            'avg_precision': np.mean(list(self.precision_values.values())) if self.precision_values else 1.0,
            'max_precision': max(self.precision_values.values()) if self.precision_values else 1.0,
            'history_entries': sum(len(history) for history in self.update_history.values())
        }
    
    def _extract_features(self, interaction_data):
        """Extract features from interaction data."""
        features = {}
        
        if not isinstance(interaction_data, dict):
            return features
            
        # Extract basic features
        for key, value in interaction_data.items():
            if isinstance(value, (str, int, float, bool)):
                features[key] = str(value)
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, (str, int, float, bool)):
                        features[f"{key}_{subkey}"] = str(subvalue)
        
        return features
    
    def _calculate_likelihood(self, features, model, interaction_type):
        """Calculate likelihood based on feature match to existing model."""
        if interaction_type not in model['likelihood_estimates']:
            return 0.5  # Default for unknown interaction type
            
        likelihood_estimates = model['likelihood_estimates'][interaction_type]
        
        # Calculate likelihood from matched features
        matched_likelihoods = []
        
        for feature, value in features.items():
            feature_key = f"{feature}:{value}"
            if feature_key in likelihood_estimates:
                matched_likelihoods.append(likelihood_estimates[feature_key])
                
        # If no matches found, use default
        if not matched_likelihoods:
            return 0.5
            
        # Use mean of matched likelihoods
        return np.mean(matched_likelihoods)
    
    def _apply_bayes_rule(self, prior, likelihood, confidence):
        """Apply Bayes' rule to update belief with confidence weighting."""
        # Confidence-weighted Bayesian update
        if likelihood > 0.5:  # Evidence supports the model
            support = likelihood * confidence
            posterior = prior + (1 - prior) * support * 0.5
        else:  # Evidence against the model
            contradiction = (1 - likelihood) * confidence
            posterior = prior - prior * contradiction * 0.5
            
        # Ensure posterior is in valid range
        return max(0.1, min(0.95, posterior))
    
    def _check_for_contradictions(self, interaction_data, model):
        """Check for contradictions in interaction data compared to model."""
        # Simplified contradiction detection
        if not isinstance(interaction_data, dict):
            return None
            
        # If confidence is high but interaction data contradicts model
        if model['prior_confidence'] > 0.7:
            features = self._extract_features(interaction_data)
            
            for interaction_type, estimates in model['likelihood_estimates'].items():
                matched_likelihoods = []
                
                for feature, value in features.items():
                    feature_key = f"{feature}:{value}"
                    if feature_key in estimates:
                        matched_likelihoods.append(estimates[feature_key])
                
                if matched_likelihoods and np.mean(matched_likelihoods) < 0.3:
                    # This is a potential contradiction
                    return {
                        'type': 'likelihood_contradiction',
                        'interaction_type': interaction_type,
                        'prior_confidence': model['prior_confidence'],
                        'contradiction_likelihood': np.mean(matched_likelihoods),
                        'features': features
                    }
                    
        return None
