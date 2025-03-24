import enum
from enum import Enum, auto

class CouplingType(Enum):
    """Types of environmental coupling relationships."""
    INFORMATIONAL = auto()  # Information exchange coupling
    OPERATIONAL = auto()    # Operational/functional coupling
    CONTEXTUAL = auto()     # Context-providing coupling
    ADAPTIVE = auto()       # Learning/feedback-based coupling
    PREDICTIVE = auto()     # NEW: Seth's predictive coupling type

class CouplingStrength(Enum):
    """Strength classifications for coupling relationships."""
    WEAK = 0.2      # Minimal influence, easily disrupted
    MODERATE = 0.5  # Standard coupling with moderate influence
    STRONG = 0.8    # Strong influence, resistant to disruption
    CRITICAL = 1.0  # System-defining, essential coupling

class CouplingState(Enum):
    """States of coupling relationships."""
    POTENTIAL = auto()     # Possible coupling not yet established
    FORMING = auto()       # Coupling in formation process
    ACTIVE = auto()        # Active coupling relationship
    DEGRADING = auto()     # Coupling beginning to decay
    DORMANT = auto()       # Inactive but restorable
    TERMINATED = auto()    # Permanently terminated
    ANTICIPATORY = auto()  # NEW: Seth's anticipatory state

class EventPriority(Enum):
    """Priority levels for event processing."""
    LOW = 0.2
    MEDIUM = 0.5
    HIGH = 0.8
    CRITICAL = 1.0

class PredictionState(Enum):
    """NEW: States for predictive processing."""
    ANTICIPATING = auto()   # Generating predictions
    PERCEIVING = auto()     # Receiving actual data
    COMPARING = auto()      # Comparing prediction to reality
    UPDATING = auto()       # Updating model based on prediction error
    TESTING = auto()        # Actively testing a prediction
