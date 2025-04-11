import torch
import numpy as np
from asf.layer1_knowledge_substrate.semiotic.feature_weighter import SemioticFeatureWeighter
class FeatureExtractor:
    """
    Base class for feature extraction from perceptual inputs.
    """
    def __init__(self):
        self.feature_cache = {}  # Cache extracted features
        self.feature_weighter = SemioticFeatureWeighter()
    def extract_features(self, input_data):
        """
        Extract features from input data.
        To be implemented by subclasses for specific modalities.
        """
        raise NotImplementedError("Subclasses must implement extract_features")
    def get_cache_key(self, input_data):
        """
        Generate a cache key for input data.
        Default implementation uses hash of string representation.
        """
        return hash(str(input_data))
    def extract_weighted_features(self, input_data, context=None):
        """
        Extract features and apply semiotic weighting
        """
        raw_features = self.extract_features(input_data)
        feature_keys = list(raw_features.keys())
        if not feature_keys:
            return {}
        try:
            feature_values = torch.stack([
                torch.tensor(raw_features[k], dtype=torch.float32) for k in feature_keys
            ])
            weighted_features, importance_scores = self.feature_weighter.weight_features(
                feature_values,
                context
            )
            result = {}
            for i, key in enumerate(feature_keys):
                result[key] = {
                    'value': weighted_features[i].numpy(),
                    'importance': importance_scores[i].item()
                }
            return result
        except Exception as e:
    logger.error(f\"Error in feature weighting: {str(e)}\")
    raise DatabaseError(f\"Error in feature weighting: {str(e)}\")
            return raw_features  # Fallback to unweighted features
    def get_key_features(self, input_data, top_k=5):
        """
        Identifies most important features for the input
        """
        raw_features = self.extract_features(input_data)
        return self.feature_weighter.extract_key_features(raw_features, top_k)