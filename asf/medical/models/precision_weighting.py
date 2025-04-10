"""
Precision Weighting for SHAP Explainability

This module implements precision weighting based on Seth's predictive processing
principles to enhance SHAP explainability for contradiction detection.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("precision-weighting")

@dataclass
class PrecisionWeights:
    """Precision weights for SHAP values."""
    
    feature_weights: Dict[str, float] = field(default_factory=dict)
    global_precision: float = 0.8
    prediction_errors: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_weights": self.feature_weights,
            "global_precision": self.global_precision,
            "prediction_errors": self.prediction_errors
        }

class PrecisionWeighter:
    """
    Precision weighter for SHAP explainability.
    
    This class implements precision weighting based on Seth's predictive processing
    principles to enhance SHAP explainability for contradiction detection.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the precision weighter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Default feature weights
        self.default_feature_weights = {
            "negation": 1.5,  # Negation is highly important
            "temporal": 1.2,  # Temporal information is important
            "numeric": 1.3,  # Numeric values are important
            "medical_term": 1.4,  # Medical terms are important
            "relation": 1.1,  # Relations are somewhat important
            "general": 0.8  # General words are less important
        }
        
        # Initialize precision weights
        self.precision_weights = PrecisionWeights(
            feature_weights=self.default_feature_weights.copy(),
            global_precision=0.8
        )
        
        # Initialize feature classifiers
        self._initialize_feature_classifiers()
    
    def _initialize_feature_classifiers(self):
        """Initialize feature classifiers."""
        # Negation words
        self.negation_words = {
            "not", "no", "none", "neither", "nor", "never", "without",
            "absence", "absent", "negative", "deny", "denies", "denied",
            "exclude", "excludes", "excluded", "rule out", "ruled out",
            "free of", "lack of", "lacking", "lacks", "non", "un", "in"
        }
        
        # Temporal words
        self.temporal_words = {
            "before", "after", "during", "while", "when", "until", "since",
            "day", "week", "month", "year", "hour", "minute", "second",
            "time", "period", "duration", "interval", "frequency", "often",
            "rarely", "sometimes", "always", "never", "occasionally"
        }
        
        # Medical prefixes and suffixes
        self.medical_affixes = {
            "anti", "auto", "bio", "cardi", "cyt", "derm", "endo", "gastro",
            "hemat", "hepat", "immun", "logy", "itis", "osis", "pathy", "ectomy",
            "plasty", "scopy", "tomy", "gram", "graph", "scope", "meter"
        }
    
    def classify_feature(self, feature: str) -> str:
        """
        Classify a feature into a category.
        
        Args:
            feature: Feature to classify
            
        Returns:
            Feature category
        """
        feature_lower = feature.lower()
        
        # Check for negation
        if feature_lower in self.negation_words or any(neg in feature_lower for neg in ["not", "no", "n't"]):
            return "negation"
        
        # Check for temporal
        if feature_lower in self.temporal_words or any(temp in feature_lower for temp in ["time", "day", "week", "month", "year"]):
            return "temporal"
        
        # Check for numeric
        if any(c.isdigit() for c in feature):
            return "numeric"
        
        # Check for medical term
        if any(affix in feature_lower for affix in self.medical_affixes):
            return "medical_term"
        
        # Check for relation
        relation_words = {"cause", "effect", "result", "lead", "associate", "correlate", "link", "relate", "due to", "because"}
        if any(rel in feature_lower for rel in relation_words):
            return "relation"
        
        # Default to general
        return "general"
    
    def weight_shap_values(self, shap_values: Dict[str, float]) -> Dict[str, float]:
        """
        Weight SHAP values based on feature precision.
        
        Args:
            shap_values: Dictionary mapping features to SHAP values
            
        Returns:
            Dictionary with weighted SHAP values
        """
        weighted_values = {}
        
        for feature, value in shap_values.items():
            # Classify feature
            feature_category = self.classify_feature(feature)
            
            # Get weight for feature category
            weight = self.precision_weights.feature_weights.get(feature_category, 1.0)
            
            # Apply weight
            weighted_values[feature] = value * weight
        
        return weighted_values
    
    def update_precision_weights(self, prediction_error: float, features: List[str]):
        """
        Update precision weights based on prediction error.
        
        Args:
            prediction_error: Error in prediction
            features: Features used in prediction
        """
        # Add prediction error to history
        self.precision_weights.prediction_errors.append(prediction_error)
        
        # Limit history length
        max_history = self.config.get("max_history", 100)
        if len(self.precision_weights.prediction_errors) > max_history:
            self.precision_weights.prediction_errors = self.precision_weights.prediction_errors[-max_history:]
        
        # Update global precision
        self.precision_weights.global_precision = 1.0 / (1.0 + np.mean(self.precision_weights.prediction_errors))
        
        # Update feature weights
        for feature in features:
            feature_category = self.classify_feature(feature)
            
            # Get current weight
            current_weight = self.precision_weights.feature_weights.get(feature_category, 1.0)
            
            # Update weight based on prediction error
            # Lower error means higher precision
            new_weight = current_weight * (1.0 - 0.1 * prediction_error)
            
            # Ensure weight is within reasonable bounds
            new_weight = max(0.5, min(2.0, new_weight))
            
            # Update weight
            self.precision_weights.feature_weights[feature_category] = new_weight
    
    def get_precision_weights(self) -> PrecisionWeights:
        """
        Get current precision weights.
        
        Returns:
            Current precision weights
        """
        return self.precision_weights
