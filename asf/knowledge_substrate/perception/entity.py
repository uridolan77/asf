import numpy as np
from sortedcontainers import SortedDict
from asf.core.enums import EntityConfidenceState
from asf.knowledge_substrate.temporal.adaptive_metadata import AdaptiveTemporalMetadata
from asf.knowledge_substrate.perception.feature import PerceptualFeature

class PerceptualEntity:
    """
    Represents a structured object derived from perceptual input.
    Philosophical influence: Kant's Transcendental Schema
    """
    def __init__(self, entity_id, input_type, initial_features=None):
        self.id = entity_id
        self.input_type = input_type  # PerceptualInputType
        self.features = SortedDict()  # Using SortedDict for efficient feature access
        self.confidence_state = EntityConfidenceState.UNVERIFIED
        self.confidence_score = 0.5  # Initial confidence (0-1)
        self.temporal_metadata = AdaptiveTemporalMetadata()
        self.cross_modal_links = set()  # IDs of linked entities from other modalities
        
        # Add any initial features
        if initial_features:
            for name, value in initial_features.items():
                self.add_feature(name, value)
    
    def add_feature(self, name, value, confidence=0.5):
        """Add a new feature to this entity"""
        self.features[name] = PerceptualFeature(name, value, confidence)
    
    def update_feature(self, name, value=None, confidence=None):
        """Update an existing feature's value and/or confidence"""
        if name in self.features:
            if value is not None:
                self.features[name].value = value
            if confidence is not None:
                self.features[name].update_confidence(confidence)
            return True
        return False
    
    def get_feature(self, name):
        """Get a specific feature by name"""
        return self.features.get(name)
    
    def get_all_features(self):
        """Get all features as a dictionary"""
        return {name: feature for name, feature in self.features.items()}
    
    def update_confidence_state(self, new_state, new_score=None):
        """Update the confidence state of this entity"""
        self.confidence_state = new_state
        if new_score is not None:
            self.confidence_score = new_score
        self.temporal_metadata.update_access_time()
    
    def add_cross_modal_link(self, entity_id):
        """Link this entity to another entity from a different modality"""
        self.cross_modal_links.add(entity_id)
    
    def get_cross_modal_links(self):
        """Get all cross-modal links for this entity"""
        return self.cross_modal_links
    
    def get_feature_vector(self):
        """
        Get a vector representation of all features for similarity calculation
        """
        # Implementation depends on feature types - this is simplified
        feature_values = []
        for name, feature in self.features.items():
            if isinstance(feature.value, (list, np.ndarray)):
                feature_values.extend(feature.value)
            else:
                feature_values.append(feature.value)
        return np.array(feature_values)
