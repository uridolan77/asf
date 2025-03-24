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
        
        # Phase 2 enhancement: Bayesian confidence
        self.confidence = 0.5  # Initial confidence
        self.confidence_state = SymbolConfidenceState.HYPOTHESIS
        self.confidence_evidence = {'positive': 1, 'negative': 1}  # Bayesian priors
        
        # Use sparse dictionary for actual meanings to save memory
        self._actual_meanings: Dict[str, Dict[str, float]] = {}
        
        # Performance tracking
        self._activation_time: Dict[str, float] = {}
        self._nonlinearity = NonlinearityOrder.LINEAR
        
        # Phase 1 enhancement: source tracking
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
        """
        start_time = time.time()
        
        # Create potential network for propagation
        potential_network = {}
        for potential_id, potential in self.potentials.items():
            key = f"{self.id}:{potential_id}"
            potential_network[key] = potential
            
        # If already actualized in this context, return cached
        if context_hash in self._actual_meanings:
            return self._actual_meanings[context_hash]
            
        # Actualize potentials in this context using maximization-based approach
        actualized = {}
        for potential_id, potential in self.potentials.items():
            # Pass potential network for MGC-inspired propagation
            activation = potential.actualize(context, potential_network)
            if activation > 0.2:  # Threshold for inclusion
                actualized[potential_id] = activation
                
        # Store for future reference
        self._actual_meanings[context_hash] = actualized
        
        # Track performance
        self._activation_time[context_hash] = time.time() - start_time
        self.last_accessed = time.time()
        
        return actualized
    
    def get_pregnancy(self) -> float:
        """
        Calculate symbolic pregnancy - richness of potential meanings.
        Higher values indicate more meaning potential.
        """
        if not self.potentials:
            return 0.0
            
        # Use maximization-based approach rather than sum
        pregnancy_values = [
            p.strength * (1 + len(p._associations) * 0.1)
            for p in self.potentials.values()
        ]
        
        return max(pregnancy_values) * len(self.potentials)
    
    def get_nonlinearity(self) -> NonlinearityOrder:
        """Get the symbol's nonlinearity order."""
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
        """
        # Apply evidence to Bayesian model
        if new_evidence:
            self.confidence_evidence['positive'] += weight
        else:
            self.confidence_evidence['negative'] += weight
            
        # Calculate confidence from evidence
        p = self.confidence_evidence['positive']
        n = self.confidence_evidence['negative']
        self.confidence = p / (p + n)
        
        # Update confidence state
        if self.confidence >= 0.8:
            self.confidence_state = SymbolConfidenceState.CANONICAL
        elif self.confidence >= 0.5:
            self.confidence_state = SymbolConfidenceState.PROVISIONAL
        else:
            self.confidence_state = SymbolConfidenceState.HYPOTHESIS
            
        return self.confidence
    
    def add_source_entity(self, entity_id: str) -> None:
        """Add a source entity ID reference."""
        if entity_id not in self.source_entities:
            self.source_entities.append(entity_id)
