import enum

class PerceptualInputType(enum.Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    NUMERICAL = "numerical"
    STRUCTURED = "structured"
    SENSOR = "sensor"
    MULTIMODAL = "multimodal"

class EntityConfidenceState(enum.Enum):
    UNVERIFIED = "unverified"  # Initial state, not yet validated
    PROVISIONAL = "provisional"  # Partially validated but requires further confirmation
    CANONICAL = "canonical"  # Fully validated entity

class PerceptualEventType(enum.Enum):
    NEW_INPUT = "new_input"
    PATTERN_DETECTED = "pattern_detected"
    ANOMALY_DETECTED = "anomaly_detected"
    ENTITY_PROMOTION = "entity_promotion"
    CONFIDENCE_UPDATE = "confidence_update"
    TEMPORAL_PATTERN = "temporal_pattern"
    CAUSAL_RELATION_DETECTED = "causal_relation_detected"
