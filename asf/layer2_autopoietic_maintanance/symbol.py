import time
import numpy as np
from typing import Dict, Any, List
from asf.layer2_autopoietic_maintanance.enums import NonlinearityOrder, SymbolConfidenceState
from asf.layer2_autopoietic_maintanance.potentials import SymbolicPotential

class SymbolElement:
    """
    Represents a symbol with its internal structure and potentials.
    Optimized for efficient operations on potential networks.
    Enhanced with Bayesian confidence updating.
    """
    def __init__(self, symbol_id: str, perceptual_anchors: Dict[str, float] = None):
        self.id = symbol_id
        self.name = f"symbol_{symbol_id[-8:]}"  # Auto-generate a name based on ID
        self.perceptual_anchors = perceptual_anchors or {}
        self.potentials: Dict[str, SymbolicPotential] = {}
        
        self.confidence = 0.5  # Initial confidence
        self.confidence_state = SymbolConfidenceState.HYPOTHESIS
        self.confidence_evidence = {'positive': 1, 'negative': 1}  # Bayesian priors
        
        self._actual_meanings: Dict[str, Dict[str, float]] = {}
        
        self._activation_time: Dict[str, float] = {}
        self._nonlinearity = NonlinearityOrder.LINEAR
        
        self.source_entities = []
        self.created_at = time.time()
        self.last_accessed = time.time()
    
    def add_potential(self, potential: SymbolicPotential) -> None:
        """Add a meaning potential to this symbol."""
        self.potentials[potential.id] = potential
        
        # Update symbol's nonlinearity based on potentials
        if potential.nonlinearity.value > self._nonlinearity.value:
            self._nonlinearity = potential.nonlinearity
    
    def actualize_meaning(self, context_hash: str, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Actualize meaning in a specific context using maximization-based approach.
        Returns dictionary of actualized meaning aspects with strengths.
        Calculate symbolic pregnancy - richness of potential meanings.
        Higher values indicate more meaning potential.
        return self._nonlinearity
    
    def clear_caches(self) -> None:
        """Clear cached calculations."""
        # Clear instance caches
        self._actual_meanings = {}
        self._activation_time = {}
    
    # Phase 2 enhancement: Bayesian confidence updating
    def update_confidence(self, new_evidence: bool, weight: float = 1.0) -> float:
        """
        Update symbol confidence using Bayesian updating.
        
        Args:
            new_evidence: True for positive evidence, False for negative
            weight: Weight of evidence (1.0 = standard weight)
            
        Returns:
            Updated confidence value
        if entity_id not in self.source_entities:
            self.source_entities.append(entity_id)
