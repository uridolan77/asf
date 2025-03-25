import numpy as np
import torch
import time
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

class BayesianUncertaintyEstimator:
    """
    Enhances predictive processing with Bayesian uncertainty estimation.
    Implements more sophisticated uncertainty quantification for Seth's
    "controlled hallucination" principle.
    """
    def __init__(self, precision_prior=1.0, sample_size=10):
        self.precision_priors = {}  # Entity ID -> prior precision
        self.precision_samples = defaultdict(list)  # Entity ID -> precision samples
        self.prediction_posteriors = {}  # Prediction ID -> posterior distribution
        self.default_precision_prior = precision_prior
        self.sample_size = sample_size
        self.calibration_metrics = {}  # Tracks calibration performance
        
    async def estimate_prediction_uncertainty(self, prediction, context_id, entity_id):
        """
        Estimates uncertainty for a prediction using Bayesian methods.
        
        Args:
            prediction: The predicted value or distribution
            context_id: Context identifier
            entity_id: Entity being predicted
            
        Returns:
            Dict containing uncertainty metrics
        """
        # Generate prediction ID
        prediction_id = f"{context_id}_{entity_id}_{int(time.time()*1000)}"
        
        # Get prior precision (inverse variance)
        precision_prior = self.precision_priors.get(entity_id, self.default_precision_prior)
        
        # Calculate uncertainty based on value type
        if isinstance(prediction, (int, float, np.number)):
            # For scalar predictions, use Beta distribution to model uncertainty
            # Higher precision = narrower distribution
            uncertainty = self._estimate_scalar_uncertainty(prediction, precision_prior)
            
        elif isinstance(prediction, (list, np.ndarray)) and len(prediction) > 0:
            # For vector predictions, estimate multivariate uncertainty
            uncertainty = self._estimate_vector_uncertainty(prediction, precision_prior)
            
        elif isinstance(prediction, dict):
            # For structured predictions, estimate uncertainty for each component
            uncertainty = self._estimate_structured_uncertainty(prediction, precision_prior)
            
        else:
            # Default fallback
            uncertainty = {
                'confidence_interval': (0.0, 1.0),
                'entropy': 1.0,
                'precision': precision_prior
            }
        
        # Store posterior for later evaluation
        self.prediction_posteriors[prediction_id] = {
            'prediction': prediction,
            'uncertainty': uncertainty,
            'timestamp': time.time()
        }
        
        return {
            'prediction_id': prediction_id,
            'uncertainty': uncertainty
        }
    
    async def update_uncertainty_model(self, prediction_id, actual_value):
        """
        Update the uncertainty model based on observed outcome.
        Implements Bayesian updating of precision estimates.
        
        Args:
            prediction_id: ID of the prediction to evaluate
            actual_value: The observed actual value
            
        Returns:
            Updated uncertainty metrics
        """
        if prediction_id not in self.prediction_posteriors:
            return None
            
        posterior = self.prediction_posteriors[prediction_id]
        prediction = posterior['prediction']
        entity_id = prediction_id.split('_')[1]  # Extract entity_id from prediction_id
        
        # Calculate prediction error
        error = self._calculate_prediction_error(prediction, actual_value)
        
        # Extract precision from posterior
        prior_precision = posterior['uncertainty'].get('precision', self.default_precision_prior)
        
        # Update precision estimate using Bayesian update
        # The likelihood precision is inversely proportional to squared error
        likelihood_precision = 1.0 / max(1e-6, error**2)  # Avoid division by zero
        
        # Posterior precision is sum of prior and likelihood precisions
        posterior_precision = prior_precision + likelihood_precision
        
        # Store precision sample
        self.precision_samples[entity_id].append(posterior_precision)
        
        # Limit sample size
        if len(self.precision_samples[entity_id]) > self.sample_size:
            self.precision_samples[entity_id] = self.precision_samples[entity_id][-self.sample_size:]
        
        # Update precision prior for future predictions
        self.precision_priors[entity_id] = np.mean(self.precision_samples[entity_id])
        
        # Update calibration metrics
        self._update_calibration_metrics(prediction_id, posterior, actual_value, error)
        
        return {
            'prediction_id': prediction_id,
            'error': error,
            'updated_precision': self.precision_priors[entity_id],
            'calibration': self.calibration_metrics.get(entity_id, {})
        }
    
    def get_precision(self, entity_id):
        """Get current precision estimate for an entity."""
        return self.precision_priors.get(entity_id, self.default_precision_prior)
    
    def _estimate_scalar_uncertainty(self, prediction, precision_prior):
        """Estimate uncertainty for scalar predictions."""
        # Standard deviation = 1/sqrt(precision)
        std_dev = 1.0 / np.sqrt(max(1e-6, precision_prior))
        
        # 95% confidence interval (assuming approximately normal distribution)
        confidence_interval = (
            prediction - 1.96 * std_dev,
            prediction + 1.96 * std_dev
        )
        
        # Calculate entropy (higher = more uncertain)
        # For normal distribution, entropy is related to variance
        entropy = 0.5 * np.log(2 * np.pi * np.e / precision_prior)
        
        return {
            'confidence_interval': confidence_interval,
            'std_dev': std_dev,
            'entropy': entropy,
            'precision': precision_prior
        }
    
    def _estimate_vector_uncertainty(self, prediction, precision_prior):
        """Estimate uncertainty for vector predictions."""
        if isinstance(prediction, list):
            prediction = np.array(prediction)
            
        # For simplicity, assume diagonal covariance matrix
        dim = prediction.shape[0]
        std_dev = 1.0 / np.sqrt(max(1e-6, precision_prior))
        
        # Vector-specific entropy calculation
        entropy = 0.5 * dim * (1.0 + np.log(2 * np.pi / precision_prior))
        
        return {
            'std_dev': std_dev,
            'entropy': entropy,
            'precision': precision_prior,
            'dimension': dim
        }
    
    def _estimate_structured_uncertainty(self, prediction, precision_prior):
        """Estimate uncertainty for structured (dictionary) predictions."""
        component_uncertainties = {}
        overall_entropy = 0.0
        
        for key, value in prediction.items():
            if isinstance(value, (int, float, np.number)):
                component_uncertainties[key] = self._estimate_scalar_uncertainty(
                    value, precision_prior
                )
                overall_entropy += component_uncertainties[key]['entropy']
                
        return {
            'component_uncertainties': component_uncertainties,
            'overall_entropy': overall_entropy,
            'precision': precision_prior
        }
    
    def _calculate_prediction_error(self, prediction, actual):
        """Calculate normalized error between predicted and actual values."""
        if isinstance(prediction, (int, float, np.number)) and isinstance(actual, (int, float, np.number)):
            # For numeric values, normalized absolute difference
            return abs(prediction - actual) / (1.0 + abs(actual))
            
        elif isinstance(prediction, (list, np.ndarray)) and isinstance(actual, (list, np.ndarray)):
            try:
                # For vectors, normalized Euclidean distance
                pred_arr = np.array(prediction)
                actual_arr = np.array(actual)
                
                if pred_arr.shape != actual_arr.shape:
                    return 1.0  # Maximum error for shape mismatch
                    
                diff = np.linalg.norm(pred_arr - actual_arr)
                norm = np.linalg.norm(actual_arr)
                return min(1.0, diff / (1.0 + norm))
            except:
                return 1.0  # Maximum error if calculation fails
                
        elif isinstance(prediction, dict) and isinstance(actual, dict):
            # For dictionaries, average error of shared keys
            shared_keys = set(prediction.keys()) & set(actual.keys())
            if not shared_keys:
                return 1.0
                
            errors = [
                self._calculate_prediction_error(prediction[k], actual[k])
                for k in shared_keys
            ]
            return sum(errors) / len(errors)
            
        else:
            # Default for different types
            return 1.0 if prediction != actual else 0.0
    
    def _update_calibration_metrics(self, prediction_id, posterior, actual_value, error):
        """Update calibration metrics to track uncertainty estimation quality."""
        entity_id = prediction_id.split('_')[1]
        
        if entity_id not in self.calibration_metrics:
            self.calibration_metrics[entity_id] = {
                'error_samples': [],
                'confidence_interval_hits': 0,
                'total_predictions': 0,
                'mean_error': 0.0
            }
            
        metrics = self.calibration_metrics[entity_id]
        metrics['error_samples'].append(error)
        metrics['total_predictions'] += 1
        
        # Check if actual value is within confidence interval (if available)
        if 'confidence_interval' in posterior['uncertainty']:
            interval = posterior['uncertainty']['confidence_interval']
            if isinstance(actual_value, (int, float, np.number)):
                if interval[0] <= actual_value <= interval[1]:
                    metrics['confidence_interval_hits'] += 1
        
        # Limit history size
        if len(metrics['error_samples']) > 50:
            metrics['error_samples'] = metrics['error_samples'][-50:]
        
        # Update mean error
        metrics['mean_error'] = np.mean(metrics['error_samples'])
        
        # Calculate calibration score (ideally close to 0.95 for 95% confidence intervals)
        if metrics['total_predictions'] > 0:
            metrics['calibration_score'] = metrics['confidence_interval_hits'] / metrics['total_predictions']


class MultiModalUncertaintyProcessor:
    """
    Extends uncertainty estimation to handle multiple modalities of information.
    Critical for integrating diverse information types in the cognitive boundary.
    """
    def __init__(self):
        self.uncertainty_estimators = {
            'numeric': BayesianUncertaintyEstimator(precision_prior=1.0),
            'categorical': BayesianUncertaintyEstimator(precision_prior=0.5),
            'temporal': BayesianUncertaintyEstimator(precision_prior=2.0),
            'relational': BayesianUncertaintyEstimator(precision_prior=0.8)
        }
        self.modality_detectors = {
            'numeric': self._is_numeric,
            'categorical': self._is_categorical,
            'temporal': self._is_temporal,
            'relational': self._is_relational
        }
        
    async def process_prediction(self, prediction, context_id, entity_id):
        """Process prediction through appropriate uncertainty estimator."""
        modality = self._detect_modality(prediction)
        estimator = self.uncertainty_estimators.get(modality, self.uncertainty_estimators['numeric'])
        result = await estimator.estimate_prediction_uncertainty(prediction, context_id, entity_id)
        result['modality'] = modality
        return result
        
    async def update_model(self, prediction_id, actual_value):
        """Update appropriate uncertainty model based on prediction ID."""
        # Extract modality from prediction ID if available, otherwise detect
        if '_' in prediction_id and len(prediction_id.split('_')) > 3:
            modality = prediction_id.split('_')[2]
        else:
            modality = self._detect_modality(actual_value)
            
        estimator = self.uncertainty_estimators.get(modality, self.uncertainty_estimators['numeric'])
        return await estimator.update_uncertainty_model(prediction_id, actual_value)
    
    def _detect_modality(self, data):
        """Detect the modality of the data for appropriate uncertainty estimation."""
        for modality, detector in self.modality_detectors.items():
            if detector(data):
                return modality
        return 'numeric'  # Default modality
    
    def _is_numeric(self, data):
        """Check if data is primarily numeric."""
        if isinstance(data, (int, float, np.number)):
            return True
        elif isinstance(data, (list, np.ndarray)) and len(data) > 0:
            return all(isinstance(x, (int, float, np.number)) for x in data)
        return False
    
    def _is_categorical(self, data):
        """Check if data is categorical."""
        if isinstance(data, str) and not self._is_temporal(data):
            return True
        elif isinstance(data, dict) and 'categories' in data:
            return True
        return False
    
    def _is_temporal(self, data):
        """Check if data is temporal."""
        if isinstance(data, str):
            # Simple heuristic for date/time strings
            return any(marker in data.lower() for marker in ['time', 'date', 'day', 'month', 'year'])
        elif isinstance(data, dict) and any(k in data for k in ['timestamp', 'date', 'time']):
            return True
        return False
    
    def _is_relational(self, data):
        """Check if data represents relationships."""
        if isinstance(data, dict) and any(k in data for k in ['source', 'target', 'relation']):
            return True
        elif isinstance(data, (list, tuple)) and len(data) == 2:
            return True
        return False


# Integration with existing PredictiveProcessor
class EnhancedPredictiveProcessor:
    """
    Enhanced version of PredictiveProcessor with advanced uncertainty estimation.
    Seamlessly integrates with existing ASF architecture.
    """
    def __init__(self):
        self.predictions = {}
        self.uncertainty_processor = MultiModalUncertaintyProcessor()
        self.prediction_errors = defaultdict(list)
        self.precision_values = {}
        self.learning_rates = {}
        self.context_history = {}
        
    async def register_prediction(self, context_id, entity_id, prediction_type, prediction_value):
        """Register a prediction with uncertainty estimation."""
        # Generate unique prediction ID
        prediction_id = f"{context_id}_{entity_id}_{prediction_type}_{int(time.time()*1000)}"
        
        # Process prediction through uncertainty estimator
        uncertainty_result = await self.uncertainty_processor.process_prediction(
            prediction_value, context_id, entity_id
        )
        
        # Store prediction with enhanced metadata
        self.predictions[prediction_id] = {
            'context_id': context_id,
            'entity_id': entity_id,
            'type': prediction_type,
            'value': prediction_value,
            'timestamp': time.time(),
            'evaluated': False,
            'uncertainty': uncertainty_result['uncertainty'],
            'modality': uncertainty_result.get('modality', 'unknown')
        }
        
        return prediction_id
        
    async def evaluate_prediction(self, prediction_id, actual_value):
        """Evaluate prediction against actual value with uncertainty analysis."""
        if prediction_id not in self.predictions:
            return None
            
        prediction = self.predictions[prediction_id]
        if prediction['evaluated']:
            return None
            
        # Update uncertainty model
        update_result = await self.uncertainty_processor.update_model(prediction_id, actual_value)
        
        # Calculate conventional error
        error = self._calculate_error(prediction['value'], actual_value)
        
        # Update prediction record
        prediction['evaluated'] = True
        prediction['actual_value'] = actual_value
        prediction['error'] = error
        prediction['evaluation_time'] = time.time()
        
        # Track error for legacy precision calculation
        entity_id = prediction['entity_id']
        self.prediction_errors[entity_id].append(error)
        
        # Limit history size
        if len(self.prediction_errors[entity_id]) > 20:
            self.prediction_errors[entity_id] = self.prediction_errors[entity_id][-20:]
        
        # Update legacy precision
        if len(self.prediction_errors[entity_id]) > 1:
            variance = np.var(self.prediction_errors[entity_id])
            precision = 1.0 / (variance + 1e-6)
            self.precision_values[entity_id] = min(10.0, precision)
            
        # Calculate adaptive learning rate
        precision = update_result.get('updated_precision', self.get_precision(entity_id))
        base_rate = min(0.8, error * 2)
        precision_factor = max(0.1, min(0.9, 1.0 / (1.0 + precision * 0.2)))
        learning_rate = min(0.9, max(0.1, base_rate * precision_factor))
        self.learning_rates[entity_id] = learning_rate
        
        return {
            'prediction_id': prediction_id,
            'error': error,
            'precision': precision,
            'learning_rate': learning_rate,
            'uncertainty_analysis': update_result
        }
        
    def get_precision(self, entity_id):
        """Get precision for a specific entity."""
        return self.precision_values.get(entity_id, 1.0)
        
    def get_learning_rate(self, entity_id):
        """Get adaptive learning rate for an entity."""
        return self.learning_rates.get(entity_id, 0.3)
    
    def _calculate_error(self, predicted, actual):
        """Legacy error calculation function."""
        if isinstance(predicted, (int, float, np.number)) and isinstance(actual, (int, float, np.number)):
            return abs(predicted - actual) / (1.0 + abs(actual))
        elif isinstance(predicted, (list, np.ndarray)) and isinstance(actual, (list, np.ndarray)):
            try:
                predicted_arr = np.array(predicted)
                actual_arr = np.array(actual)
                
                if predicted_arr.shape != actual_arr.shape:
                    return 1.0
                    
                diff = np.linalg.norm(predicted_arr - actual_arr)
                norm = np.linalg.norm(actual_arr)
                return min(1.0, diff / (1.0 + norm))
            except:
                return 1.0
        elif isinstance(predicted, dict) and isinstance(actual, dict):
            shared_keys = set(predicted.keys()) & set(actual.keys())
            
            if not shared_keys:
                return 1.0
                
            errors = []
            for key in shared_keys:
                errors.append(self._calculate_error(predicted[key], actual[key]))
                
            return sum(errors) / len(errors)
        else:
            return 1.0 if predicted != actual else 0.0

# Example usage
async def test_enhanced_predictive_processor():
    processor = EnhancedPredictiveProcessor()
    
    # Register a prediction
    prediction_id = await processor.register_prediction(
        context_id="test_context",
        entity_id="test_entity",
        prediction_type="numeric_property",
        prediction_value=42.5
    )
    
    # Evaluate prediction
    evaluation = await processor.evaluate_prediction(
        prediction_id=prediction_id,
        actual_value=45.2
    )
    
    print(f"Prediction evaluation: {evaluation}")
    
    # Another prediction with vector values
    vector_pred_id = await processor.register_prediction(
        context_id="test_context",
        entity_id="test_entity",
        prediction_type="embedding",
        prediction_value=[0.1, 0.2, 0.3, 0.4]
    )
    
    # Evaluate vector prediction
    vector_eval = await processor.evaluate_prediction(
        prediction_id=vector_pred_id,
        actual_value=[0.15, 0.25, 0.35, 0.45]
    )
    
    print(f"Vector prediction evaluation: {vector_eval}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_enhanced_predictive_processor())