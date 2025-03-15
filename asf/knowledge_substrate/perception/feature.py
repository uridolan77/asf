from asf.knowledge_substrate.temporal.adaptive_metadata import AdaptiveTemporalMetadata

class PerceptualFeature:
    """
    Represents an individual feature extracted from perceptual input.
    """
    def __init__(self, name, value, confidence=0.5):
        self.name = name
        self.value = value  # Could be a scalar, vector, or more complex structure
        self.confidence = confidence
        self.temporal_metadata = AdaptiveTemporalMetadata()
    
    def update_confidence(self, new_confidence):
        """Update confidence score for this feature"""
        self.confidence = new_confidence
        self.temporal_metadata.update_access_time()
    
    def get_relevance(self):
        """Calculate overall relevance based on confidence and temporal factors"""
        temporal_relevance = self.temporal_metadata.get_temporal_relevance()
        return self.confidence * temporal_relevance
