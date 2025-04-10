import time
from asf.layer1_knowledge_substrate.temporal.sequence import TemporalSequence
from asf.layer1_knowledge_substrate.temporal.pattern_detector import TemporalPatternDetector

class TemporalProcessingEngine:
    """
    Manages temporal processing across different modalities and timescales.
    """
    def __init__(self):
        self.sequences = {}  # Maps entity_ids to their temporal sequences
        self.pattern_detectors = {}  # Different detectors for different sequence types
        self.temporal_context = {}  # Maintains context across processing cycles
    
    def register_sequence(self, entity_id, sequence_type):
        """Register a new temporal sequence for tracking"""
        if entity_id not in self.sequences:
            self.sequences[entity_id] = {}
        self.sequences[entity_id][sequence_type] = TemporalSequence()
    
    def add_event(self, entity_id, sequence_type, event_data):
        """Add an event to an entity's temporal sequence"""
        if entity_id in self.sequences and sequence_type in self.sequences[entity_id]:
            self.sequences[entity_id][sequence_type].add_event(event_data)
            # Check for patterns when new events are added
            return self.check_for_patterns(entity_id, sequence_type)
        return None
    
    def check_for_patterns(self, entity_id, sequence_type):
        """Analyze sequence for temporal patterns"""
        if sequence_type not in self.pattern_detectors:
            # Initialize appropriate detector if needed
            feature_dim = self._get_feature_dimension(sequence_type)
            self.pattern_detectors[sequence_type] = TemporalPatternDetector(feature_dim)
        
        # Get recent events as feature vectors
        sequence = self.sequences[entity_id][sequence_type]
        recent_events = sequence.get_events_in_window(window_size=3600)  # Last hour
        if len(recent_events) < 3:  # Need minimum sequence length
            return None
        
        # Prepare data for pattern detection
        sequence_data = [self._prepare_event_features(e) for e in recent_events]
        
        # Detect patterns
        pattern_score = self.pattern_detectors[sequence_type].detect_patterns(sequence_data)
        if pattern_score > 0.7:  # Threshold for pattern detection
            return {
                "entity_id": entity_id,
                "sequence_type": sequence_type,
                "pattern_score": pattern_score,
                "event_count": len(recent_events)
            }
        return None
    
    def _get_feature_dimension(self, sequence_type):
        """Return feature dimension based on sequence type"""
        # Different sequence types may have different feature dimensions
        feature_dimensions = {
            "text": 768,
            "image": 512,
            "audio": 256,
            "numerical": 64,
            "default": 128
        }
        return feature_dimensions.get(sequence_type, feature_dimensions["default"])
    
    def _prepare_event_features(self, event):
        """Extract feature vector from event for pattern detection"""
        # Implementation depends on event type
        if hasattr(event, 'get_feature_vector'):
            return event.get_feature_vector()
        return event  # Assume event is already a feature vector
    
    def maintain_temporal_context(self, context_id, context_data):
        """Update temporal context for a given context ID"""
        self.temporal_context[context_id] = {
            "data": context_data,
            "timestamp": time.time()
        }
    
    def get_temporal_context(self, context_id, max_age=None):
        """Retrieve temporal context if not expired"""
        if context_id in self.temporal_context:
            context = self.temporal_context[context_id]
            if max_age is None or time.time() - context["timestamp"] <= max_age:
                return context["data"]
        return None
