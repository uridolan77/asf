import time
import joblib
import asyncio
import logging
from typing import Dict, List, Any, Optional, Set, Tuple

from asf.layer2_autopoietic_maintanance.enums import NonlinearityOrder, SymbolConfidenceState
from asf.layer2_autopoietic_maintanance.symbol import SymbolElement
from asf.layer2_autopoietic_maintanance.network import SparseTensorSymbolNetwork
from asf.layer2_autopoietic_maintanance.autocatalytic_network import NonlinearityOrderTracker, AutocatalyticNetwork
from asf.layer2_autopoietic_maintanance.operational_closure import OperationalClosure
from asf.layer2_autopoietic_maintanance.recognition import SymbolRecognizer

class AutopoieticMaintananceLayer:
    """
    Main controller for the Autopoietic Maintanance Layer (Layer 2).
    Coordinates all symbol-related processes and interfaces with other layers.
    """
    def __init__(self, config=None):
        self.config = config or {}
        
        self.symbols: Dict[str, SymbolElement] = {}
        self.recognized_symbol_mapping: Dict[str, str] = {}  # Maps perceptual IDs to symbol IDs
        
        self.nonlinearity_tracker = NonlinearityOrderTracker()
        self.network = SparseTensorSymbolNetwork()
        self.autocatalytic_network = AutocatalyticNetwork(self.nonlinearity_tracker)
        self.operational_closure = OperationalClosure()
        self.recognizer = SymbolRecognizer(
            threshold=self.config.get('recognition_threshold', 0.7)
        )
        
        self.stats = {
            "symbols_created": 0,
            "symbols_recognized": 0,
            "potentials_actualized": 0,
            "system_closure": 0.0
        }
        
        self.context_cache = {}
        
        self.logger = logging.getLogger("ASF.SymbolicFormation")
        
        self._initialize_production_templates()
    
    def _initialize_production_templates(self) -> None:
        """Initialize basic production templates for autocatalytic network."""
        # Simple combination template
        self.autocatalytic_network.add_production_template(
            "combination",
            {"entity1": 0.7, "entity2": 0.7},
            NonlinearityOrder.LINEAR
        )
        
        # Contrast template
        self.autocatalytic_network.add_production_template(
            "contrast",
            {"entity1": 0.8, "opposite1": 0.5},
            NonlinearityOrder.QUADRATIC
        )
        
        # Abstraction template
        self.autocatalytic_network.add_production_template(
            "abstraction",
            {"specific1": 0.6, "specific2": 0.6, "specific3": 0.6},
            NonlinearityOrder.CUBIC
        )
        
        # Analogy template
        self.autocatalytic_network.add_production_template(
            "analogy",
            {"source": 0.8, "target": 0.7, "relation": 0.5},
            NonlinearityOrder.EXPONENTIAL
        )
    
    async def process_perceptual_input(self, perceptual_data: Dict[str, Dict[str, float]],
                                    context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process new perceptual input to form or recognize symbols.
        
        Args:
            perceptual_data: Dictionary mapping entity IDs to features
            context: Optional processing context
            
        Returns:
            Dictionary with processing results
        Form new symbols from perceptual data using autocatalytic network.
        Actualize the meaning of a symbol in a specific context.
        
        Args:
            symbol_id: ID of symbol to actualize
            context: Context for actualization
            
        Returns:
            Dictionary of actualized meaning aspects with strengths
        Propagate activations through the symbol network.
        
        Args:
            initial_activations: Dictionary mapping symbol IDs to initial activations
            context: Optional context for meaning actualization
            iterations: Number of propagation iterations
            
        Returns:
            Dictionary of symbol IDs to their actualized meanings
        Maintain operational closure of the symbol system.
        Adds suggested relations to ensure system coherence.
        return self.symbols.get(symbol_id)
    
    def get_symbols_for_entity(self, entity_id: str) -> List[SymbolElement]:
        """Get all symbols associated with a perceptual entity."""
        if entity_id in self.recognized_symbol_mapping:
            symbol_id = self.recognized_symbol_mapping[entity_id]
            if symbol_id in self.symbols:
                return [self.symbols[symbol_id]]
        
        # Search for entity in source_entities lists
        result = []
        for symbol in self.symbols.values():
            if entity_id in symbol.source_entities:
                result.append(symbol)
                
        return result
    
    def get_layer_statistics(self) -> Dict[str, Any]:
        """Get statistics about layer operations."""
        return {
            **self.stats,
            "total_symbols": len(self.symbols),
            "canonical_symbols": sum(1 for s in self.symbols.values() 
                                  if s.confidence_state == SymbolConfidenceState.CANONICAL),
            "total_potentials": sum(len(s.potentials) for s in self.symbols.values()),
            "avg_pregnancy": sum(s.get_pregnancy() for s in self.symbols.values()) / 
                            max(1, len(self.symbols))
        }
    
    async def run_integration_cycle(self) -> Dict[str, Any]:
        start_time = time.time()
        
        self._maintain_closure()
        
        flat_perceptual = {}  # Empty perceptual input for pure emergent generation
        symbols_before = len(self.symbols)
        
        new_symbols = self.autocatalytic_network.generate_symbols(
            self.symbols,
            flat_perceptual,
            threshold=self.config.get('emergent_symbol_threshold', 0.7)
        )
        
        for symbol_id, symbol in new_symbols.items():
            self.symbols[symbol_id] = symbol
            
            self.nonlinearity_tracker.register_symbol(
                symbol_id, symbol.get_nonlinearity()
            )
            
            self.network.add_symbol(symbol_id)
        
        closure = self.operational_closure.calculate_closure(self.symbols)
        self.stats["system_closure"] = closure
        
        return {
            "processing_time": time.time() - start_time,
            "new_symbols": len(self.symbols) - symbols_before,
            "system_closure": closure,
            "total_symbols": len(self.symbols)
        }
