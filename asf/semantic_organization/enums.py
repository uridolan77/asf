from enum import Enum

class SemanticNodeType(Enum):
    """Enumeration of semantic node types"""
    CONCEPT = "concept"
    CATEGORY = "category"
    BLEND = "blend"
    ABSTRACTION = "abstraction"
    RELATION = "relation"

class SemanticConfidenceState(Enum):
    """Confidence states for semantic structures"""
    HYPOTHETICAL = "hypothetical"  # Initial state, speculative
    PROVISIONAL = "provisional"    # Partially validated
    CANONICAL = "canonical"        # Fully validated semantic structure
