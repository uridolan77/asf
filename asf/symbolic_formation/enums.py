from enum import Enum

class NonlinearityOrder(Enum):
    """
    Enumeration for tracking order of nonlinearity in symbolic transformations.
    Higher orders represent more complex conceptual transformations.
    """
    LINEAR = 1      # Direct correspondences and simple relationships
    QUADRATIC = 2   # Second-order transformations involving two interacting elements
    CUBIC = 3       # Third-order transformations with multiple interactions
    EXPONENTIAL = 4 # Transformations with rapidly increasing complexity
    COMPOSITIONAL = 5 # Highest order - complex compositions of multiple transformations

class SymbolConfidenceState(Enum):
    """
    Enumeration for tracking confidence states of symbols.
    Mirrors the confidence states in Layer 1 for consistency.
    """
    HYPOTHESIS = "hypothesis"  # Initial hypothetical state
    PROVISIONAL = "provisional" # Partially validated
    CANONICAL = "canonical"    # Fully validated
