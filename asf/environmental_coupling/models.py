import time
import uuid
from typing import Dict, List, Set, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict
from asf.environmental_coupling.enums import (
    CouplingType, CouplingStrength, CouplingState,
    EventPriority, PredictionState
)

@dataclass
class CouplingEvent:
    """Represents an event related to environmental coupling."""
    id: str
    event_type: str
    coupling_id: Optional[str] = None
    entity_id: Optional[str] = None
    environmental_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)
    priority: float = 0.5
    processed: bool = False
    processing_time: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    
    # NEW: Seth's Data Paradox enhancements
    predicted: bool = False
    prediction_id: Optional[str] = None
    prediction_error: Optional[float] = None
    precision: Optional[float] = None

@dataclass
class EnvironmentalCoupling:
    """Represents a coupling relationship between internal and environmental entities."""
    id: str
    internal_entity_id: str
    environmental_entity_id: str
    coupling_type: CouplingType = CouplingType.INFORMATIONAL
    coupling_strength: float = 0.5
    coupling_state: CouplingState = CouplingState.FORMING
    creation_time: float = field(default_factory=time.time)
    last_interaction: float = field(default_factory=time.time)
    interaction_count: int = 0
    bayesian_confidence: float = 0.5
    properties: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    tensor_coordinates: Optional[Tuple[int, int]] = None
    
    # NEW: Seth's Data Paradox enhancements
    prediction_precision: float = 1.0  # Inverse variance of prediction errors
    prediction_errors: List[float] = field(default_factory=list)
    expected_interactions: Dict[str, float] = field(default_factory=dict)
    
@dataclass
class EnvironmentalPrediction:
    """Represents a prediction about an environmental entity or interaction."""
    id: str
    environmental_entity_id: str
    predicted_data: Dict[str, Any]
    confidence: float = 0.5
    precision: float = 1.0
    prediction_time: float = field(default_factory=time.time)
    verification_time: Optional[float] = None
    prediction_error: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ActiveInferenceTest:
    """Represents an active test of a prediction."""
    id: str
    coupling_id: str
    prediction_id: Optional[str] = None
    test_parameters: Dict[str, Any] = field(default_factory=dict)
    created_time: float = field(default_factory=time.time)
    execution_time: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    information_gain: Optional[float] = None
