import time
import joblib
import asyncio
import logging
from typing import Dict, List, Any, Optional, Set, Tuple

from asf.symbolic_formation.enums import NonlinearityOrder, SymbolConfidenceState
from asf.symbolic_formation.potentials import SymbolicPotential
from asf.symbolic_formation.symbol import SymbolElement
from asf.symbolic_formation.network import SparseTensorSymbolNetwork
from asf.symbolic_formation.autocatalytic import NonlinearityOrderTracker, AutocatalyticNetwork
from asf.symbolic_formation.operational_closure import OperationalClosure
from asf.symbolic_formation.recognition import SymbolRecognizer

class SymbolicFormationLayer:
    """
    Main controller for the Symbolic Formation Layer (Layer 2).
    Coordinates all symbol-related processes and interfaces with other layers.
    """
    def __init__(self, config=None):
        self.config = config or {}
        
        # Core data structures
        self.symbols: Dict[str, SymbolElement] = {}
        self.recognized_symbol_mapping: Dict[str, str] = {}  # Maps perceptual IDs to symbol IDs
        
        # Component initialization
        self.nonlinearity_tracker = NonlinearityOrderTracker()
        self.network = SparseTensorSymbolNetwork()
        self.autocatalytic_network = AutocatalyticNetwork(self.nonlinearity_tracker)
        self.operational_closure = OperationalClosure()
        self.recognizer = SymbolRecognizer(
            threshold=self.config.get('recognition_threshold', 0.7)
        )
        
        # Performance tracking
        self.stats = {
            "symbols_created": 0,
            "symbols_recognized": 0,
            "potentials_actualized": 0,
            "system_closure": 0.0
        }
        
        # Cache for context hashing
        self.context_cache = {}
        
        # Initialize logger
        self.logger = logging.getLogger("ASF.SymbolicFormation")
        
        # Load initial templates
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
        """
        start_time = time.time()
        context = context or {}
        context_hash = joblib.hash(context)
        
        # Try to recognize existing symbols first
        recognition_result = await self.recognizer.recognize(
            perceptual_data, self.symbols, context
        )
        
        result = {
            "processing_time": 0,
            "new_symbols": [],
            "recognized_symbols": [],
            "activations": {}
        }
        
        if recognition_result["recognized"]:
            # Symbol recognized - update mapping and confidence
            symbol_id = recognition_result["symbol_id"]
            symbol = self.symbols[symbol_id]
            
            # Update confidence based on successful recognition
            symbol.update_confidence(True, weight=recognition_result["confidence"])
            
            # Update entity mapping
            for entity_id in perceptual_data:
                self.recognized_symbol_mapping[entity_id] = symbol_id
                # Add source tracking
                symbol.add_source_entity(entity_id)
            
            # Actualize symbol in current context
            meaning = symbol.actualize_meaning(context_hash, context)
            
            # Track stats
            self.stats["symbols_recognized"] += 1
            self.stats["potentials_actualized"] += len(meaning)
            
            result["recognized_symbols"].append({
                "symbol_id": symbol_id,
                "confidence": recognition_result["confidence"],
                "strategy": recognition_result.get("strategy")
            })
            result["activations"][symbol_id] = meaning
        else:
            # No recognition - try to form new symbols
            new_symbols = await self._form_new_symbols(perceptual_data, context)
            
            if new_symbols:
                # Add new symbols to main collection
                for symbol_id, symbol in new_symbols.items():
                    self.symbols[symbol_id] = symbol
                    
                    # Register with nonlinearity tracker
                    self.nonlinearity_tracker.register_symbol(
                        symbol_id, symbol.get_nonlinearity()
                    )
                    
                    # Add to network
                    self.network.add_symbol(symbol_id)
                    
                    # Create mappings for source entities
                    for entity_id in perceptual_data:
                        self.recognized_symbol_mapping[entity_id] = symbol_id
                        symbol.add_source_entity(entity_id)
                        
                    # Track formation of new symbol
                    result["new_symbols"].append({
                        "symbol_id": symbol_id,
                        "name": symbol.name,
                        "pregnancy": symbol.get_pregnancy(),
                        "potentials": len(symbol.potentials)
                    })
                
                # Track stats
                self.stats["symbols_created"] += len(new_symbols)
                
                # Maintain operational closure after adding new symbols
                self._maintain_closure()
        
        # Calculate current closure
        self.stats["system_closure"] = self.operational_closure.calculate_closure(self.symbols)
        
        # Calculate processing time
        result["processing_time"] = time.time() - start_time
        
        return result
    
    async def _form_new_symbols(self, perceptual_data: Dict[str, Dict[str, float]],
                              context: Dict[str, Any]) -> Dict[str, SymbolElement]:
        """
        Form new symbols from perceptual data using autocatalytic network.
        """
        # Flatten perceptual data for symbol formation
        flat_perceptual = {}
        for entity_id, features in perceptual_data.items():
            for feature_name, value in features.items():
                key = f"{entity_id}:{feature_name}"
                flat_perceptual[key] = value
                
        # Use autocatalytic network to generate new symbols
        return self.autocatalytic_network.generate_symbols(
            self.symbols,
            flat_perceptual,
            threshold=self.config.get('symbol_formation_threshold', 0.5)
        )
    
    def actualize_symbol(self, symbol_id: str, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Actualize the meaning of a symbol in a specific context.
        
        Args:
            symbol_id: ID of symbol to actualize
            context: Context for actualization
            
        Returns:
            Dictionary of actualized meaning aspects with strengths
        """
        if symbol_id not in self.symbols:
            return {}
            
        symbol = self.symbols[symbol_id]
        context_hash = joblib.hash(context)
        
        # Actualize the symbol in this context
        meaning = symbol.actualize_meaning(context_hash, context)
        
        # Track potentials actualized
        self.stats["potentials_actualized"] += len(meaning)
        
        return meaning
    
    def propagate_activations(self, initial_activations: Dict[str, float],
                            context: Dict[str, Any] = None,
                            iterations: int = 2) -> Dict[str, Dict[str, float]]:
        """
        Propagate activations through the symbol network.
        
        Args:
            initial_activations: Dictionary mapping symbol IDs to initial activations
            context: Optional context for meaning actualization
            iterations: Number of propagation iterations
            
        Returns:
            Dictionary of symbol IDs to their actualized meanings
        """
        # First propagate through network to get activated symbols
        propagated = self.network.propagate_activations(initial_activations, iterations)
        
        # For symbols that are sufficiently activated, actualize their meanings
        result = {}
        context_hash = joblib.hash(context) if context else "default"
        
        for symbol_id, activation in propagated.items():
            if activation > 0.2 and symbol_id in self.symbols:  # Activation threshold
                symbol = self.symbols[symbol_id]
                meaning = symbol.actualize_meaning(context_hash, context or {})
                
                # Scale meaning by activation strength
                scaled_meaning = {k: v * activation for k, v in meaning.items()}
                result[symbol_id] = scaled_meaning
                
                # Track stats
                self.stats["potentials_actualized"] += len(meaning)
                
        return result
    
    def _maintain_closure(self) -> None:
        """
        Maintain operational closure of the symbol system.
        Adds suggested relations to ensure system coherence.
        """
        suggested_relations = self.operational_closure.maintain_closure(
            self.symbols,
            self.nonlinearity_tracker,
            min_closure=self.config.get('min_closure', 0.7)
        )
        
        # Add suggested relations to maintain closure
        for source_key, target_key in suggested_relations:
            # Extract symbol and potential IDs
            if ":" in source_key and ":" in target_key:
                source_symbol_id, source_potential_id = source_key.split(":", 1)
                target_symbol_id, target_potential_id = target_key.split(":", 1)
                
                # Check if symbols and potentials exist
                if (source_symbol_id in self.symbols and
                    source_potential_id in self.symbols[source_symbol_id].potentials and
                    target_symbol_id in self.symbols and
                    target_potential_id in self.symbols[target_symbol_id].potentials):
                    
                    # Add association between potentials
                    source_potential = self.symbols[source_symbol_id].potentials[source_potential_id]
                    source_potential.add_association(target_key, 0.5)  # Default strength
                    
                    # Add to network
                    self.network.add_relation(source_key, target_key, 0, 0.5)
    
    def get_symbol_by_id(self, symbol_id: str) -> Optional[SymbolElement]:
        """Get a symbol by ID."""
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
        """
        Run an integration cycle to enhance system coherence.
        This performs maintenance operations on the symbol system.
        """
        start_time = time.time()
        
        # Maintain operational closure
        self._maintain_closure()
        
        # Generate new symbols through autocatalytic process
        # Use current symbols to generate new emergent symbols
        flat_perceptual = {}  # Empty perceptual input for pure emergent generation
        symbols_before = len(self.symbols)
        
        new_symbols = self.autocatalytic_network.generate_symbols(
            self.symbols,
            flat_perceptual,
            threshold=self.config.get('emergent_symbol_threshold', 0.7)
        )
        
        # Add new symbols to main collection
        for symbol_id, symbol in new_symbols.items():
            self.symbols[symbol_id] = symbol
            
            # Register with nonlinearity tracker
            self.nonlinearity_tracker.register_symbol(
                symbol_id, symbol.get_nonlinearity()
            )
            
            # Add to network
            self.network.add_symbol(symbol_id)
        
        # Update system closure
        closure = self.operational_closure.calculate_closure(self.symbols)
        self.stats["system_closure"] = closure
        
        return {
            "processing_time": time.time() - start_time,
            "new_symbols": len(self.symbols) - symbols_before,
            "system_closure": closure,
            "total_symbols": len(self.symbols)
        }
