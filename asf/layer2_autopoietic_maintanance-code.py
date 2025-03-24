

# === FILE: layer2_autopoietic_maintanance\__init__.py ===

from asf.symbolic_formation.enums import NonlinearityOrder, SymbolConfidenceState
from asf.symbolic_formation.potentials import SymbolicPotential
from asf.symbolic_formation.symbol import SymbolElement
from asf.symbolic_formation.network import SparseTensorSymbolNetwork
from asf.symbolic_formation.autocatalytic import AutocatalyticNetwork, NonlinearityOrderTracker
from asf.symbolic_formation.operational_closure import OperationalClosure
from asf.symbolic_formation.recognition import SymbolRecognizer
from asf.symbolic_formation.symbolic_layer import SymbolicFormationLayer

# Seth's Data Paradox enhancements
from asf.symbolic_formation.predictive_potentials import PredictiveSymbolicPotential
from asf.symbolic_formation.predictive_symbol import PredictiveSymbolElement
from asf.symbolic_formation.predictive_recognition import PredictiveSymbolRecognizer
from asf.symbolic_formation.counterfactual_network import CounterfactualAutocatalyticNetwork
from asf.symbolic_formation.predictive_processor import SymbolicPredictiveProcessor
from asf.symbolic_formation.predictive_layer import PredictiveSymbolicFormationLayer



# === FILE: layer2_autopoietic_maintanance\autocatalytic.py ===

import numpy as np
import time
import uuid
from collections import defaultdict
from typing import Dict, List, Optional
import torch

from asf.symbolic_formation.enums import NonlinearityOrder
from asf.symbolic_formation.symbol import SymbolElement
from asf.symbolic_formation.network import SparseTensorSymbolNetwork

class NonlinearityOrderTracker:
    """
    Tracks and optimizes order of nonlinearity in symbol transformations.
    Enhanced with learning capabilities for adaptive nonlinearity classification.
    """
    def __init__(self):
        self.symbol_nonlinearity: Dict[str, NonlinearityOrder] = {}
        self.potential_nonlinearity: Dict[str, NonlinearityOrder] = {}
        self.transformation_complexity: Dict[tuple, NonlinearityOrder] = {}
        # Phase 2 enhancement: learning statistics
        self.transformation_observations = defaultdict(list)
        self.learning_rate = 0.1
    
    def register_symbol(self, symbol_id: str, nonlinearity: NonlinearityOrder = NonlinearityOrder.LINEAR) -> None:
        """Register a symbol with its nonlinearity order."""
        self.symbol_nonlinearity[symbol_id] = nonlinearity
    
    def register_potential(self, potential_id: str, nonlinearity: NonlinearityOrder = NonlinearityOrder.LINEAR) -> None:
        """Register a potential with its nonlinearity order."""
        self.potential_nonlinearity[potential_id] = nonlinearity
    
    def register_transformation(self, source_id: str, target_id: str, nonlinearity: NonlinearityOrder) -> None:
        """Register a transformation between symbols/potentials with its nonlinearity order."""
        self.transformation_complexity[(source_id, target_id)] = nonlinearity
        # Phase 2 enhancement: record observation for learning
        self.transformation_observations[(source_id, target_id)].append(nonlinearity)
        self._update_transformation_complexity(source_id, target_id)
    
    def _update_transformation_complexity(self, source_id: str, target_id: str) -> None:
        """Update transformation complexity based on observations."""
        observations = self.transformation_observations[(source_id, target_id)]
        if len(observations) < 3:
            return  # Not enough data
            
        # Count occurrences of each nonlinearity order
        order_counts = {}
        for obs in observations:
            if obs.value in order_counts:
                order_counts[obs.value] += 1
            else:
                order_counts[obs.value] = 1
                
        # Find most common nonlinearity order
        most_common_value = max(order_counts.items(), key=lambda x: x[1])[0]
        most_common_order = NonlinearityOrder(most_common_value)
        
        # Update complexity with some inertia (learning rate)
        current_order = self.transformation_complexity.get((source_id, target_id), NonlinearityOrder.LINEAR)
        
        # If most common differs from current, adapt with learning rate
        if most_common_order.value != current_order.value:
            # Move toward most common based on learning rate
            if most_common_order.value > current_order.value:
                # Increase nonlinearity
                new_value = min(
                    NonlinearityOrder.COMPOSITIONAL.value,
                    current_order.value + round(self.learning_rate * (most_common_order.value - current_order.value))
                )
            else:
                # Decrease nonlinearity
                new_value = max(
                    NonlinearityOrder.LINEAR.value,
                    current_order.value - round(self.learning_rate * (current_order.value - most_common_order.value))
                )
                
            self.transformation_complexity[(source_id, target_id)] = NonlinearityOrder(new_value)

class AutocatalyticNetwork:
    """
    Implements autocatalytic networks where symbols help create other symbols.
    Optimized with nonlinearity order tracking for better generalization.
    Enhanced with template adaptation learning.
    """
    def __init__(self, nonlinearity_tracker: NonlinearityOrderTracker):
        self.catalytic_relations: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.production_templates: Dict[str, Dict[str, float]] = {}
        self.template_nonlinearity: Dict[str, NonlinearityOrder] = {}
        self.nonlinearity_tracker = nonlinearity_tracker
        
        # Sparse tensor representation for large networks
        self._tensor_network = SparseTensorSymbolNetwork()
        # Adaptive template selection based on past success
        self._template_success_rate: Dict[str, float] = defaultdict(float)
        self._template_usage_count: Dict[str, int] = defaultdict(int)
        # Phase 2 enhancement: template adaptation
        self._template_adaption_rate = 0.05
        self._template_evolution_history = []
    
    def add_catalytic_relation(self, catalyst_id: str, target_id: str, strength: float) -> None:
        """Add a catalytic relationship where one symbol helps produce another."""
        self.catalytic_relations[catalyst_id][target_id] = strength
        # Add to tensor network
        self._tensor_network.add_relation(
            catalyst_id, target_id, relation_type=1, strength=strength
        )
    
    def add_production_template(self, template_id: str, required_elements: Dict[str, float],
                              nonlinearity: NonlinearityOrder = NonlinearityOrder.LINEAR) -> None:
        """Add a template for producing new symbolic elements."""
        self.production_templates[template_id] = required_elements
        self.template_nonlinearity[template_id] = nonlinearity
        # Register with nonlinearity tracker
        self.nonlinearity_tracker.register_transformation(
            "template", template_id, nonlinearity
        )
    
    def generate_symbols(self, existing_symbols: Dict[str, SymbolElement],
                       perceptual_inputs: Dict[str, float],
                       threshold: float = 0.5) -> Dict[str, SymbolElement]:
        """
        Generate new symbols based on existing symbols and perceptual inputs.
        Uses tensor-based propagation and prioritizes templates with lower nonlinearity.
        """
        new_symbols = {}
        # Set initial activations for tensor propagation
        initial_activations = {}
        for symbol_id, symbol in existing_symbols.items():
            initial_activations[symbol_id] = symbol.get_pregnancy() / 10  # Normalize
            
        # Propagate activations to find potential catalytic interactions
        propagated = self._tensor_network.propagate_activations(initial_activations)
        
        # Sort templates by nonlinearity order (prioritize simpler templates)
        sorted_templates = sorted(
            self.production_templates.keys(),
            key=lambda t: (self.template_nonlinearity[t].value, -self._template_success_rate.get(t, 0.0))
        )
        
        # Check each production template in order of increasing nonlinearity
        for template_id in sorted_templates:
            required = self.production_templates[template_id]
            # Calculate production strength based on propagated activations
            production_strength = self._calculate_production_strength(propagated, required)
            
            if production_strength > threshold:
                # Generate new symbol
                new_id = f"generated_{template_id}_{len(new_symbols)}_{uuid.uuid4().hex[:4]}"
                # Create with subset of perceptual anchors
                anchors = {k: v for k, v in perceptual_inputs.items() if v > 0.3}  # Threshold for inclusion
                new_symbol = SymbolElement(new_id, anchors)
                # Set name based on template
                new_symbol.name = f"{template_id}_{new_id[-4:]}"
                
                # Add initial potentials based on template
                for req_id, req_strength in required.items():
                    if req_id in propagated and propagated[req_id] > 0.2:
                        if req_id in existing_symbols:
                            # Inherit some potentials from catalyst symbols
                            self._transfer_potentials(
                                existing_symbols[req_id],
                                new_symbol,
                                req_strength
                            )
                
                # Only add if symbol has meaningful potentials
                if new_symbol.potentials:
                    new_symbols[new_id] = new_symbol
                    
                    # Update template success statistics
                    self._template_usage_count[template_id] += 1
                    self._template_success_rate[template_id] = (
                        (self._template_success_rate[template_id] * (self._template_usage_count[template_id] - 1)
                         + 1.0) /
                        self._template_usage_count[template_id]
                    )
                    
                    # Phase 2 enhancement: template adaptation
                    self._adapt_template(template_id, True, new_symbol)
            else:
                # Update template failure statistics
                self._template_usage_count[template_id] += 1
                self._template_success_rate[template_id] = (
                    (self._template_success_rate[template_id] * (self._template_usage_count[template_id] - 1)
                     + 0.0) /
                    self._template_usage_count[template_id]
                )
                
                # Phase 2 enhancement: template adaptation
                self._adapt_template(template_id, False, None)
                
        return new_symbols
    
    def _calculate_production_strength(self, propagated_activations: Dict[str, float],
                                     required: Dict[str, float]) -> float:
        """Calculate production strength based on propagated activations."""
        if not required:
            return 0.0
            
        # Use maximization-based approach rather than average (MGC-inspired)
        strengths = []
        for req_id, req_strength in required.items():
            if req_id in propagated_activations:
                # Consider both requirement strength and activation
                activation = propagated_activations[req_id]
                strengths.append(activation * req_strength)
                
        if not strengths:
            return 0.0
            
        # Return maximum strength rather than average
        return max(strengths)
    
    def _transfer_potentials(self, source: SymbolElement, target: SymbolElement, strength: float) -> None:
        """
        Transfer potentials from source to target symbol.
        Optimized to maintain lower nonlinearity order.
        """
        # Select potentials to transfer
        potentials_to_transfer = []
        
        # Select based on both strength and nonlinearity order
        for potential in source.potentials.values():
            # Prioritize potentials with lower nonlinearity
            transfer_score = strength * (1.0 / (1.0 + potential.nonlinearity.value))
            if np.random.random() < transfer_score:
                potentials_to_transfer.append(potential)
                
        # Transfer selected potentials with modified strength
        for potential in potentials_to_transfer:
            # Create new potential with nonlinearity one order higher
            new_nonlinearity = NonlinearityOrder(
                min(NonlinearityOrder.COMPOSITIONAL.value,
                   potential.nonlinearity.value + 1))
            new_potential_id = f"{target.id}_{potential.id}"
            
            from asf.symbolic_formation.potentials import SymbolicPotential
            
            new_potential = SymbolicPotential(
                new_potential_id,
                potential.strength * strength,
                nonlinearity=new_nonlinearity
            )
            
            # Copy subset of associations based on strength
            assoc_items = list(potential._associations.items())
            if len(assoc_items) > 3:
                assoc_items = assoc_items[-3:]  # Top 3
                
            for assoc_id, assoc_strength in assoc_items:
                if np.random.random() < strength:  # Probabilistic transfer
                    new_potential.add_association(assoc_id, assoc_strength * strength)
                    
            target.add_potential(new_potential)
            
            # Register with nonlinearity tracker
            self.nonlinearity_tracker.register_potential(
                new_potential.id, new_nonlinearity
            )
    
    # Phase 2 enhancement: template adaptation
    def _adapt_template(self, template_id: str, success: bool, new_symbol: Optional[SymbolElement] = None) -> None:
        """
        Adapt template based on success or failure in generating symbols.
        
        Args:
            template_id: Template to adapt
            success: Whether the generation was successful
            new_symbol: The newly generated symbol if success is True
        """
        if template_id not in self.production_templates:
            return
            
        current_template = self.production_templates[template_id]
        
        if success and new_symbol:
            # Strengthen requirements that contributed to success
            # Use potentials of new symbol to guide adaptation
            for potential in new_symbol.potentials.values():
                # Find which requirement contributed to this potential
                potential_source = potential.id.split("_")[-1]  # Assuming potential ID format
                for req_id in current_template:
                    if potential_source in req_id:
                        # Strengthen this requirement slightly
                        current_template[req_id] = min(1.0,
                                                    current_template[req_id] + self._template_adaption_rate)
        else:
            # Slight reduction in all requirements to make template more flexible
            for req_id in current_template:
                current_template[req_id] = max(0.1,
                                            current_template[req_id] - (self._template_adaption_rate / 2))
                
        # Record adaptation history
        self._template_evolution_history.append({
            'template_id': template_id,
            'success': success,
            'timestamp': time.time(),
            'new_requirements': dict(current_template)
        })



# === FILE: layer2_autopoietic_maintanance\autopoietic_maintanance_layer.py ===

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



# === FILE: layer2_autopoietic_maintanance\autopoietic_maintenance_engine.py ===




# === FILE: layer2_autopoietic_maintanance\bayesian_confidence.py ===




# === FILE: layer2_autopoietic_maintanance\contradiction_detection.py ===




# === FILE: layer2_autopoietic_maintanance\contradiction_pattern_analysis.py ===




# === FILE: layer2_autopoietic_maintanance\counterfactual_network.py ===

import time
import uuid
import numpy as np
from typing import Dict, List, Any, Optional, Set, Tuple

from asf.symbolic_formation.autocatalytic import AutocatalyticNetwork
from asf.symbolic_formation.autocatalytic import NonlinearityOrderTracker
from asf.symbolic_formation.symbol import SymbolElement

class CounterfactualAutocatalyticNetwork(AutocatalyticNetwork):
    """
    Enhances AutocatalyticNetwork with counterfactual reasoning capabilities.
    Allows testing hypothetical scenarios through virtual interventions.
    """
    def __init__(self, nonlinearity_tracker: NonlinearityOrderTracker):
        super().__init__(nonlinearity_tracker)
        self.counterfactual_history = []
        
    def generate_counterfactual_symbols(self, existing_symbols: Dict[str, SymbolElement], 
                                       perceptual_inputs: Dict[str, float], 
                                       modification_rules: List[Dict[str, Any]]) -> Dict[str, SymbolElement]:
        """
        Generate 'what if' symbols by counterfactually modifying inputs
        
        Args:
            existing_symbols: Current symbols
            perceptual_inputs: Actual perceptual data
            modification_rules: Rules for counterfactual modifications
        
        Returns:
            Dictionary of counterfactual symbols that might have been generated
        """
        # Create counterfactual perceptual inputs
        cf_inputs = self._apply_modifications(perceptual_inputs, modification_rules)
        
        # Generate symbols from counterfactual inputs
        cf_symbols = self.generate_symbols(existing_symbols, cf_inputs)
        
        # Track counterfactual generation
        self.counterfactual_history.append({
            'timestamp': time.time(),
            'modifications': modification_rules,
            'original_input_size': len(perceptual_inputs),
            'modified_input_size': len(cf_inputs),
            'symbols_generated': len(cf_symbols)
        })
        
        return cf_symbols
    
    def _apply_modifications(self, perceptual_inputs: Dict[str, float], 
                            modification_rules: List[Dict[str, Any]]) -> Dict[str, float]:
        """Apply modification rules to perceptual inputs"""
        modified_inputs = perceptual_inputs.copy()
        
        for rule in modification_rules:
            rule_type = rule.get('type')
            
            if rule_type == 'remove':
                # Remove features matching pattern
                pattern = rule.get('pattern', '')
                keys_to_remove = [k for k in modified_inputs if pattern in k]
                for key in keys_to_remove:
                    modified_inputs.pop(key, None)
                    
            elif rule_type == 'add':
                # Add new features
                features = rule.get('features', {})
                for key, value in features.items():
                    modified_inputs[key] = value
                    
            elif rule_type == 'modify':
                # Modify existing features
                pattern = rule.get('pattern', '')
                operation = rule.get('operation', 'multiply')
                factor = rule.get('factor', 1.0)
                
                for key, value in list(modified_inputs.items()):
                    if pattern in key:
                        if operation == 'multiply':
                            modified_inputs[key] = value * factor
                        elif operation == 'add':
                            modified_inputs[key] = value + factor
                        elif operation == 'replace':
                            modified_inputs[key] = factor
        
        return modified_inputs
    
    def compare_counterfactual_outcomes(self, actual_symbols: Dict[str, SymbolElement], 
                                      counterfactual_symbols: Dict[str, SymbolElement]) -> Dict[str, Any]:
        """
        Compare actual symbols with counterfactual ones to evaluate impact
        
        Returns analysis of differences and hypothesis validation
        """
        # Compare symbol sets
        actual_ids = set(actual_symbols.keys())
        cf_ids = set(counterfactual_symbols.keys())
        
        # Find symbols only in actual or counterfactual
        only_actual = actual_ids - cf_ids
        only_cf = cf_ids - actual_ids
        shared = actual_ids.intersection(cf_ids)
        
        # Calculate meaning differences for shared symbols
        meaning_diffs = {}
        for symbol_id in shared:
            actual_symbol = actual_symbols[symbol_id]
            cf_symbol = counterfactual_symbols[symbol_id]
            
            # Create context for meaning actualization
            context = {'comparison': True}
            context_hash = str(hash(str(context)))
            
            # Actualize meanings
            actual_meaning = actual_symbol.actualize_meaning(context_hash, context)
            cf_meaning = cf_symbol.actualize_meaning(context_hash, context)
            
            # Calculate differences
            potential_diffs = {}
            for potential_id in set(list(actual_meaning.keys()) + list(cf_meaning.keys())):
                actual_val = actual_meaning.get(potential_id, 0.0)
                cf_val = cf_meaning.get(potential_id, 0.0)
                diff = cf_val - actual_val
                if abs(diff) > 0.1:  # Only track significant differences
                    potential_diffs[potential_id] = diff
            
            if potential_diffs:
                meaning_diffs[symbol_id] = potential_diffs
        
        return {
            'only_in_actual': list(only_actual),
            'only_in_counterfactual': list(only_cf),
            'shared_symbols': len(shared),
            'meaning_differences': meaning_diffs
        }



# === FILE: layer2_autopoietic_maintanance\enums.py ===

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



# === FILE: layer2_autopoietic_maintanance\multi_resolution_modeling.py ===




# === FILE: layer2_autopoietic_maintanance\network.py ===

import torch
from typing import Dict, List, Optional

class SparseTensorSymbolNetwork:
    """
    Efficient sparse tensor representation of the symbol network.
    Uses Graph Tensor Convolution inspired approach for rich semantic propagation.
    Enhanced with hardware acceleration capabilities.
    """
    def __init__(self, max_symbols: int = 10000, device=None):
        self.max_symbols = max_symbols
        self.symbol_to_idx: Dict[str, int] = {}
        self.idx_to_symbol: Dict[int, str] = {}
        self.potential_to_idx: Dict[str, int] = {}
        self.idx_to_potential: Dict[int, str] = {}
        
        # Sparse adjacency data structures
        self._row_indices: List[int] = []
        self._col_indices: List[int] = []
        self._edge_values: List[float] = []
        self._edge_types: List[int] = []
        
        # Cached tensor representations
        self._adjacency_tensor: Optional[List[torch.Tensor]] = None
        self._feature_tensor: Optional[torch.Tensor] = None
        self._need_rebuild: bool = True
        
        # Configuration for tensor operations
        self.semantic_channels = 3  # Multiple channels for rich semantic representation
        
        # Hardware acceleration support
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    
    def add_symbol(self, symbol_id: str) -> int:
        """Add a symbol to the network and return its index."""
        if symbol_id in self.symbol_to_idx:
            return self.symbol_to_idx[symbol_id]
            
        idx = len(self.symbol_to_idx)
        self.symbol_to_idx[symbol_id] = idx
        self.idx_to_symbol[idx] = symbol_id
        self._need_rebuild = True
        return idx
    
    def add_potential(self, potential_id: str) -> int:
        """Add a potential to the network and return its index."""
        if potential_id in self.potential_to_idx:
            return self.potential_to_idx[potential_id]
            
        idx = len(self.potential_to_idx)
        self.potential_to_idx[potential_id] = idx
        self.idx_to_potential[idx] = potential_id
        self._need_rebuild = True
        return idx
    
    def add_relation(self, source_id: str, target_id: str, relation_type: int, strength: float) -> None:
        """Add a relation between symbols or potentials."""
        # Determine if source/target are symbols or potentials
        source_is_symbol = source_id in self.symbol_to_idx
        target_is_symbol = target_id in self.symbol_to_idx
        
        # Get or create indices
        if source_is_symbol:
            source_idx = self.symbol_to_idx[source_id]
        else:
            source_idx = self.add_potential(source_id) + self.max_symbols
            
        if target_is_symbol:
            target_idx = self.symbol_to_idx[target_id]
        else:
            target_idx = self.add_potential(target_id) + self.max_symbols
            
        # Add relation to sparse representation
        self._row_indices.append(source_idx)
        self._col_indices.append(target_idx)
        self._edge_values.append(strength)
        self._edge_types.append(relation_type)
        
        self._need_rebuild = True
        
    def _build_tensors(self) -> None:
        """Build sparse tensor representations of the network with hardware acceleration."""
        if not self._need_rebuild:
            return
            
        # Convert to PyTorch sparse tensor with hardware acceleration
        indices = torch.tensor([self._row_indices, self._col_indices], 
                              dtype=torch.long, device=self.device)
        values = torch.tensor(self._edge_values, dtype=torch.float, device=self.device)
        edge_types = torch.tensor(self._edge_types, dtype=torch.long, device=self.device)
        
        # Size includes both symbols and potentials
        size = (self.max_symbols * 2, self.max_symbols * 2)
        
        # Create multi-channel adjacency tensor (GTCN-inspired)
        # Each channel represents different semantic relationship aspects
        max_edge_type = max(self._edge_types) if self._edge_types else 0
        channels = min(max_edge_type + 1, self.semantic_channels)
        
        adjacency_tensors = []
        for channel in range(channels):
            # Filter edges for this channel/type
            channel_mask = edge_types == channel
            if not torch.any(channel_mask):
                # Empty channel
                channel_tensor = torch.sparse.FloatTensor(
                    torch.zeros((2, 0), dtype=torch.long, device=self.device),
                    torch.tensor([], dtype=torch.float, device=self.device),
                    size
                )
            else:
                channel_indices = indices[:, channel_mask]
                channel_values = values[channel_mask]
                channel_tensor = torch.sparse.FloatTensor(
                    channel_indices, channel_values, size,
                    device=self.device
                )
            adjacency_tensors.append(channel_tensor)
            
        # Store as list of sparse tensors (more efficient than 3D sparse tensor)
        self._adjacency_tensor = adjacency_tensors
        
        # Initialize feature tensor
        total_nodes = len(self.symbol_to_idx) + len(self.potential_to_idx)
        self._feature_tensor = torch.zeros(
            (total_nodes, 16),
            dtype=torch.float,
            device=self.device
        )
        
        self._need_rebuild = False
        
    def propagate_activations(self, initial_activations: Dict[str, float], iterations: int = 2) -> Dict[str, float]:
        """
        Propagate activations through the network using tensor operations.
        Uses GTCN-inspired multi-channel propagation for rich semantic transfer.
        """
        self._build_tensors()
        
        # Initialize activation vector
        activation_vector = torch.zeros(
            self.max_symbols * 2,
            dtype=torch.float,
            device=self.device
        )
        
        # Set initial activations
        for node_id, activation in initial_activations.items():
            if node_id in self.symbol_to_idx:
                idx = self.symbol_to_idx[node_id]
                activation_vector[idx] = activation
            elif node_id in self.potential_to_idx:
                idx = self.potential_to_idx[node_id] + self.max_symbols
                activation_vector[idx] = activation
        
        # Propagate activations through the network
        for _ in range(iterations):
            # Propagate through each channel
            channel_results = []
            for channel_tensor in self._adjacency_tensor:
                if channel_tensor._nnz() > 0:  # Check if tensor has any non-zero elements
                    # Sparse matrix multiplication for efficiency
                    channel_result = torch.sparse.mm(channel_tensor, 
                                                   activation_vector.unsqueeze(1)).squeeze(1)
                    channel_results.append(channel_result)
            
            if channel_results:
                # Stack results from all channels
                stacked_results = torch.stack(channel_results, dim=0)
                # Use maximization-based aggregation (MGC-inspired)
                # This prevents over-smoothing compared to averaging
                new_activations, _ = torch.max(stacked_results, dim=0)
                # Update activations (with residual connection)
                activation_vector = (activation_vector + new_activations) / 2
        
        # Extract results for symbols
        result = {}
        for symbol_id, idx in self.symbol_to_idx.items():
            result[symbol_id] = activation_vector[idx].item()
            
        return result



# === FILE: layer2_autopoietic_maintanance\operational_closure.py ===

import time
import scipy.sparse as sp
from typing import Dict, List, Tuple, Set, Optional

from asf.symbolic_formation.enums import NonlinearityOrder
from asf.symbolic_formation.symbol import SymbolElement

class OperationalClosure:
    """
    Implements mechanisms for maintaining system coherence through
    operational closure as per Maturana and Varela.
    Optimized with sparse matrix representation for efficiency.
    """
    def __init__(self):
        self.boundary_elements: Set[str] = set()
        # Use sparse matrix for internal relations
        self._relation_matrix: Optional[sp.csr_matrix] = None
        self._element_indices: Dict[str, int] = {}
        self._index_elements: Dict[int, str] = {}
        self.closure_metrics: Dict[str, float] = {}
        self._need_rebuild: bool = True
        
        # Phase 2 enhancement: system integrity tracking
        self.integrity_history = []
        self.last_integrity_check = 0
    
    def add_boundary_element(self, element_id: str) -> None:
        """Add an element to the system boundary."""
        self.boundary_elements.add(element_id)
        # Ensure element is in index mapping
        if element_id not in self._element_indices:
            idx = len(self._element_indices)
            self._element_indices[element_id] = idx
            self._index_elements[idx] = element_id
            self._need_rebuild = True
    
    def add_internal_relation(self, source_id: str, target_id: str) -> None:
        """Add an internal relation between elements."""
        # Ensure elements are in index mapping
        for element_id in (source_id, target_id):
            if element_id not in self._element_indices:
                idx = len(self._element_indices)
                self._element_indices[element_id] = idx
                self._index_elements[idx] = element_id
                
        self._need_rebuild = True
        
    def _rebuild_matrix(self, relations: List[Tuple[str, str]]) -> None:
        """Rebuild the sparse relation matrix."""
        if not self._need_rebuild and self._relation_matrix is not None:
            return
            
        n_elements = len(self._element_indices)
        if n_elements == 0:
            self._relation_matrix = sp.csr_matrix((0, 0))
            return
            
        # Build sparse matrix
        rows, cols, data = [], [], []
        for source_id, target_id in relations:
            if source_id in self._element_indices and target_id in self._element_indices:
                source_idx = self._element_indices[source_id]
                target_idx = self._element_indices[target_id]
                rows.append(source_idx)
                cols.append(target_idx)
                data.append(1.0)
                
        self._relation_matrix = sp.csr_matrix(
            (data, (rows, cols)), shape=(n_elements, n_elements))
            
        self._need_rebuild = False
        
    def calculate_closure(self, elements: Dict[str, SymbolElement]) -> float:
        """
        Calculate degree of operational closure using efficient sparse operations.
        1.0 means perfect closure, 0.0 means completely open.
        """
        if not elements:
            return 0.0
            
        # Extract relations from elements
        relations = []
        for symbol_id, symbol in elements.items():
            # Add symbol to indices if needed
            if symbol_id not in self._element_indices:
                idx = len(self._element_indices)
                self._element_indices[symbol_id] = idx
                self._index_elements[idx] = symbol_id
                
            # Extract relations from potentials
            for potential_id, potential in symbol.potentials.items():
                source_key = f"{symbol_id}:{potential_id}"
                
                # Add relation source to indices
                if source_key not in self._element_indices:
                    idx = len(self._element_indices)
                    self._element_indices[source_key] = idx
                    self._index_elements[idx] = source_key
                    
                # Add relations to associations
                for assoc_id in potential._associations:
                    relations.append((source_key, assoc_id))
                    
        # Rebuild relation matrix
        self._rebuild_matrix(relations)
        
        # Calculate closure using matrix operations
        if self._relation_matrix.shape[0] == 0:
            return 0.0
            
        # Count total relations
        total_relations = self._relation_matrix.count_nonzero()
        if total_relations == 0:
            return 0.0
            
        # Count relations between elements in the system
        element_indices = [self._element_indices[e_id] for e_id in elements
                         if e_id in self._element_indices]
        if not element_indices:
            return 0.0
            
        # Extract submatrix for system elements
        system_matrix = self._relation_matrix[element_indices, :][:, element_indices]
        internal_relations = system_matrix.count_nonzero()
        
        # Record system integrity
        current_time = time.time()
        if current_time - self.last_integrity_check > 60:  # Check at most once per minute
            closure_score = internal_relations / total_relations
            self.integrity_history.append({
                'timestamp': current_time,
                'closure_score': closure_score,
                'total_relations': total_relations,
                'internal_relations': internal_relations,
                'element_count': len(elements)
            })
            self.last_integrity_check = current_time
            
        return internal_relations / total_relations
        
    def maintain_closure(self, elements: Dict[str, SymbolElement],
                      nonlinearity_tracker,
                      min_closure: float = 0.7) -> List[Tuple[str, str]]:
        """
        Maintain operational closure by suggesting new internal relations
        if closure falls below threshold. Prioritizes simpler relationships.
        """
        current_closure = self.calculate_closure(elements)
        if current_closure >= min_closure:
            return []
            
        # Find potential new relations to increase closure
        suggested_relations = []
        
        # Extract current relations
        current_relations = set()
        for symbol_id, symbol in elements.items():
            for potential_id, potential in symbol.potentials.items():
                source_key = f"{symbol_id}:{potential_id}"
                for assoc_id in potential._associations:
                    current_relations.add((source_key, assoc_id))
                    
        # Find candidate relations between system elements
        element_ids = set(elements.keys())
        potential_relations = []
        
        for symbol_id, symbol in elements.items():
            for potential_id, potential in symbol.potentials.items():
                source_key = f"{symbol_id}:{potential_id}"
                
                for target_id in element_ids:
                    if target_id != symbol_id:
                        for target_pot_id in elements[target_id].potentials:
                            target_key = f"{target_id}:{target_pot_id}"
                            
                            # Check if relation already exists
                            if (source_key, target_key) not in current_relations:
                                # Calculate potential relationship nonlinearity
                                nonlinearity = NonlinearityOrder.LINEAR
                                
                                if source_key in nonlinearity_tracker.potential_nonlinearity:
                                    source_nl = nonlinearity_tracker.potential_nonlinearity[source_key]
                                else:
                                    source_nl = NonlinearityOrder.LINEAR
                                    
                                if target_key in nonlinearity_tracker.potential_nonlinearity:
                                    target_nl = nonlinearity_tracker.potential_nonlinearity[target_key]
                                else:
                                    target_nl = NonlinearityOrder.LINEAR
                                
                                # Combine nonlinearities
                                nonlinearity = NonlinearityOrder(
                                    min(NonlinearityOrder.COMPOSITIONAL.value,
                                       max(source_nl.value, target_nl.value) + 1))
                                        
                                # Add as candidate with nonlinearity as score
                                potential_relations.append(
                                    (source_key, target_key, nonlinearity))
                                    
        # Sort potential relations by nonlinearity (simpler first)
        potential_relations.sort(key=lambda x: x[2].value)
        
        # Select top relations to suggest
        needed_relations = int((min_closure - current_closure) * 
                             len(current_relations) * 1.5) + 1
                              
        suggested_relations = [(src, tgt) for src, tgt, _ in 
                              potential_relations[:needed_relations]]
                               
        return suggested_relations



# === FILE: layer2_autopoietic_maintanance\potentials.py ===




# === FILE: layer2_autopoietic_maintanance\predictive_factory.py ===

import logging
from typing import Dict, Any, Optional

from asf.symbolic_formation.symbol import SymbolElement
from asf.symbolic_formation.potentials import SymbolicPotential
from asf.symbolic_formation.predictive_potentials import PredictiveSymbolicPotential
from asf.symbolic_formation.predictive_symbol import PredictiveSymbolElement
from asf.symbolic_formation.predictive_layer import PredictiveSymbolicFormationLayer

logger = logging.getLogger(__name__)

def create_predictive_layer2(config=None):
    """Factory function to create a predictive Layer 2"""
    logger.info("Creating predictive Layer 2 with Seth's Data Paradox enhancements")
    return PredictiveSymbolicFormationLayer(config)

def convert_to_predictive(layer):
    """Convert standard symbols and potentials to predictive variants"""
    logger.info(f"Converting {len(layer.symbols)} symbols to predictive variants")
    
    # Convert symbols to predictive variants
    predictive_symbols = {}
    for symbol_id, symbol in layer.symbols.items():
        # Create predictive symbol with same properties
        predictive_symbol = PredictiveSymbolElement(symbol.id, symbol.perceptual_anchors.copy())
        predictive_symbol.name = symbol.name
        predictive_symbol.confidence = symbol.confidence
        predictive_symbol.confidence_state = symbol.confidence_state
        predictive_symbol.confidence_evidence = symbol.confidence_evidence.copy()
        predictive_symbol.source_entities = symbol.source_entities.copy()
        predictive_symbol.created_at = symbol.created_at
        predictive_symbol.last_accessed = symbol.last_accessed
        predictive_symbol._nonlinearity = symbol._nonlinearity
        
        # Convert potentials to predictive variants
        for potential_id, potential in symbol.potentials.items():
            predictive_potential = PredictiveSymbolicPotential(
                potential.id,
                potential.strength,
                potential.nonlinearity
            )
            
            # Copy associations
            for assoc_id, assoc_strength in potential._associations.items():
                predictive_potential.add_association(assoc_id, assoc_strength)
                
            # Copy activations
            predictive_potential._activations = potential._activations.copy()
            
            # Add to symbol
            predictive_symbol.potentials[potential_id] = predictive_potential
            
        # Add to collection
        predictive_symbols[symbol_id] = predictive_symbol
    
    # Replace symbols in layer
    layer.symbols = predictive_symbols
    
    logger.info("Conversion to predictive variants complete")
    return layer



# === FILE: layer2_autopoietic_maintanance\predictive_layer.py ===

import time
import joblib
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

from asf.symbolic_formation.symbolic_layer import SymbolicFormationLayer
from asf.symbolic_formation.predictive_recognition import PredictiveSymbolRecognizer
from asf.symbolic_formation.counterfactual_network import CounterfactualAutocatalyticNetwork
from asf.symbolic_formation.predictive_processor import SymbolicPredictiveProcessor

class PredictiveSymbolicFormationLayer(SymbolicFormationLayer):
    """
    Enhanced symbolic formation layer implementing Seth's Data Paradox principles.
    Integrates predictive processing and counterfactual reasoning for improved
    symbol formation and recognition.
    """
    def __init__(self, config=None):
        super().__init__(config)
        # Replace standard components with predictive variants
        self.recognizer = PredictiveSymbolRecognizer(
            threshold=self.config.get('recognition_threshold', 0.7)
        )
        
        # Create predictive processor for coordinating predictions
        self.predictive_processor = SymbolicPredictiveProcessor()
        
        # Replace autocatalytic network with counterfactual version
        self.autocatalytic_network = CounterfactualAutocatalyticNetwork(self.nonlinearity_tracker)
        
        # Track perceptual predictions
        self.perceptual_predictions = {}
        self.perceptual_prediction_errors = defaultdict(list)
    
    async def process_perceptual_input(self, perceptual_data, context=None):
        """Override to include predictive processing"""
        # First predict what we expect to see
        context = context or {}
        predictions = await self.generate_perceptual_predictions(context)
        
        # Store prediction for later evaluation
        prediction_id = f"perceptual_{int(time.time())}"
        self.perceptual_predictions[prediction_id] = {
            'context': context,
            'predictions': predictions,
            'timestamp': time.time()
        }
        
        # Process input normally
        result = await super().process_perceptual_input(perceptual_data, context)
        
        # Add prediction info to result
        result['prediction_id'] = prediction_id
        
        # Evaluate prediction accuracy
        if prediction_id in self.perceptual_predictions:
            evaluation = self.evaluate_perceptual_prediction(
                prediction_id, perceptual_data
            )
            result['prediction_evaluation'] = evaluation
        
        return result
        
    async def generate_perceptual_predictions(self, context):
        """
        Generate predictions about perceptual entities that should appear
        based on activated symbols. These can be sent to Layer 1 to guide
        perception.
        """
        # Get most relevant symbols for this context
        context_hash = joblib.hash(context)
        relevant_symbols = self._get_context_relevant_symbols(context)
        
        # For each symbol, get perceptual anchors that would be expected
        predictions = defaultdict(dict)
        
        for symbol_id, relevance in relevant_symbols.items():
            symbol = self.symbols[symbol_id]
            
            # Extract perceptual anchors as predictions
            for anchor, strength in symbol.perceptual_anchors.items():
                # Parse anchor to get entity type and feature
                if ":" in anchor:
                    entity_type, feature = anchor.split(":", 1)
                    predictions[entity_type][feature] = max(
                        predictions[entity_type].get(feature, 0.0),
                        strength * relevance
                    )
        
        return dict(predictions)
    
    def _get_context_relevant_symbols(self, context):
        """Get symbols relevant to the current context with scores"""
        context_hash = joblib.hash(context)
        relevant_symbols = {}
        
        # Check each symbol's relevance to this context
        for symbol_id, symbol in self.symbols.items():
            # Actualize meaning to check relevance
            meaning = symbol.actualize_meaning(context_hash, context)
            if meaning:
                # Calculate overall relevance from meaning
                relevance = sum(meaning.values()) / max(1, len(meaning))
                if relevance > 0.2:  # Threshold for relevance
                    relevant_symbols[symbol_id] = relevance
        
        return relevant_symbols
    
    def evaluate_perceptual_prediction(self, prediction_id, actual_perceptual):
        """
        Evaluate a perceptual prediction against actual data
        """
        if prediction_id not in self.perceptual_predictions:
            return {'error': 'Prediction not found'}
        
        prediction = self.perceptual_predictions[prediction_id]
        predicted = prediction['predictions']
        
        # Flatten actual perceptual data
        flat_actual = {}
        for entity_id, features in actual_perceptual.items():
            entity_type = entity_id.split('_')[0]  # Extract type from ID
            for feature, value in features.items():
                flat_actual[f"{entity_type}:{feature}"] = value
        
        # Flatten predictions
        flat_predicted = {}
        for entity_type, features in predicted.items():
            for feature, value in features.items():
                flat_predicted[f"{entity_type}:{feature}"] = value
        
        # Calculate true positives, false positives, false negatives
        tp = 0
        fp = 0
        fn = 0
        
        # True positives and false positives
        for key, pred_value in flat_predicted.items():
            if pred_value > 0.3:  # Prediction threshold
                if key in flat_actual and flat_actual[key] > 0.3:
                    tp += 1
                else:
                    fp += 1
        
        # False negatives
        for key, actual_value in flat_actual.items():
            if actual_value > 0.3 and (key not in flat_predicted or flat_predicted[key] < 0.3):
                fn += 1
        
        # Calculate precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        }
    
    def run_counterfactual_simulation(self, perceptual_data, modification_rules, context=None):
        """
        Run a counterfactual simulation to see what symbols would form
        under modified perceptual conditions
        """
        context = context or {}
        
        # Process the actual perceptual data (async function needs to be called differently)
        loop = asyncio.get_event_loop()
        actual_result = loop.run_until_complete(
            self.process_perceptual_input(perceptual_data, context)
        )
        
        # Get all newly created symbols
        actual_symbols = {}
        for symbol_info in actual_result.get('new_symbols', []):
            symbol_id = symbol_info.get('symbol_id')
            if symbol_id in self.symbols:
                actual_symbols[symbol_id] = self.symbols[symbol_id]
        
        # Generate counterfactual symbols
        cf_symbols = self.autocatalytic_network.generate_counterfactual_symbols(
            self.symbols,
            self._flatten_perceptual_data(perceptual_data),
            modification_rules
        )
        
        # Compare outcomes
        comparison = self.autocatalytic_network.compare_counterfactual_outcomes(
            actual_symbols, cf_symbols
        )
        
        return {
            'actual_symbols': [symbol_id for symbol_id in actual_symbols],
            'counterfactual_symbols': [symbol_id for symbol_id in cf_symbols],
            'comparison': comparison,
            'modifications': modification_rules
        }
    
    def _flatten_perceptual_data(self, perceptual_data):
        """Flatten hierarchical perceptual data into key-value pairs"""
        flat_data = {}
        for entity_id, features in perceptual_data.items():
            for feature_name, value in features.items():
                key = f"{entity_id}:{feature_name}"
                flat_data[key] = value
        return flat_data



# === FILE: layer2_autopoietic_maintanance\predictive_potentials.py ===

import time
import numpy as np
import joblib
from collections import defaultdict
from typing import Dict, Any, List, Optional, Set, Tuple

from asf.symbolic_formation.potentials import SymbolicPotential
from asf.symbolic_formation.enums import NonlinearityOrder

class PredictiveSymbolicPotential(SymbolicPotential):
    """
    Symbolic potential with prediction capabilities.
    Uses precision-weighted activation for better uncertainty handling.
    """
    def __init__(self, id: str, strength: float = 1.0, nonlinearity: NonlinearityOrder = NonlinearityOrder.LINEAR):
        super().__init__(id, strength, nonlinearity)
        self.predicted_activations = {}  # Context hash -> predicted activation
        self.prediction_errors = defaultdict(list)  # Context hash -> list of errors
        self.precision_values = {}  # Context hash -> precision value
        
    def predict_activation(self, context: Dict[str, Any]) -> Optional[float]:
        """Predict activation before context actualization"""
        context_hash = joblib.hash(context)
        
        # If we don't have enough past activations, can't predict
        if len(self._activations) < 3:
            return None
            
        # Find similar contexts (simplified implementation)
        similar_contexts = list(self._activations.keys())[:5]  # Use 5 most recent contexts
        
        if not similar_contexts:
            return None
        
        # Calculate predicted activation as weighted average of similar contexts
        total_weight = 0.0
        weighted_sum = 0.0
        
        for past_hash in similar_contexts:
            weight = 1.0  # Equal weighting for simplicity
            activation = self._activations[past_hash]
            
            weighted_sum += activation * weight
            total_weight += weight
        
        if total_weight > 0:
            predicted = weighted_sum / total_weight
            self.predicted_activations[context_hash] = predicted
            return predicted
            
        return None
        
    def actualize(self, context: Dict[str, Any], potential_network: Optional[Dict[str, 'SymbolicPotential']] = None) -> float:
        """Predict activation, then actualize and calculate error"""
        context_hash = joblib.hash(context)
        
        # First make prediction
        predicted = self.predict_activation(context)
        
        # Then get actual activation
        actual = super().actualize(context, potential_network)
        
        # Calculate prediction error if prediction was made
        if predicted is not None:
            error = abs(predicted - actual)
            self.prediction_errors[context_hash].append(error)
            
            # Limit history size
            if len(self.prediction_errors[context_hash]) > 20:
                self.prediction_errors[context_hash] = self.prediction_errors[context_hash][-20:]
            
            # Update precision (inverse variance)
            if len(self.prediction_errors[context_hash]) > 1:
                variance = np.var(self.prediction_errors[context_hash])
                precision = 1.0 / (variance + 1e-6)  # Avoid division by zero
                self.precision_values[context_hash] = min(10.0, precision)  # Cap very high precision
        
        return actual
    
    def get_precision(self, context_hash=None):
        """Get precision for a context or overall"""
        if context_hash and context_hash in self.precision_values:
            return self.precision_values[context_hash]
        
        # Calculate overall precision
        all_errors = []
        for errors in self.prediction_errors.values():
            all_errors.extend(errors)
            
        if len(all_errors) < 2:
            return 1.0  # Default precision
            
        variance = np.var(all_errors)
        precision = 1.0 / (variance + 1e-6)  # Avoid division by zero
        return min(10.0, precision)  # Cap very high precision



# === FILE: layer2_autopoietic_maintanance\predictive_processor.py ===

import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

class SymbolicPredictiveProcessor:
    """
    Coordinates predictive processing across Layer 2 components.
    Implements Seth's predictive processing principles for symbolic operations.
    """
    
    def __init__(self):
        self.predictions = {}  # Prediction ID -> prediction data
        self.prediction_errors = defaultdict(list)  # Entity ID -> errors
        self.precision_values = {}  # Entity ID -> precision value
        self.learning_rates = {}  # Entity ID -> adaptive learning rate
        
    def register_prediction(self, context_id: str, entity_id: str, prediction: Any) -> str:
        """Register a prediction for later evaluation"""
        prediction_id = f"{context_id}_{entity_id}_{int(time.time()*1000)}"
        self.predictions[prediction_id] = {
            'context_id': context_id,
            'entity_id': entity_id,
            'value': prediction,
            'timestamp': time.time(),
            'evaluated': False
        }
        return prediction_id
        
    def evaluate_prediction(self, prediction_id: str, actual_value: Any) -> Optional[Dict[str, Any]]:
        """Evaluate prediction against actual value"""
        if prediction_id not in self.predictions:
            return None
            
        prediction = self.predictions[prediction_id]
        if prediction['evaluated']:
            return None
            
        # Calculate prediction error
        predicted = prediction['value']
        error = self._calculate_error(predicted, actual_value)
        
        # Update prediction record
        prediction['evaluated'] = True
        prediction['actual_value'] = actual_value
        prediction['error'] = error
        prediction['evaluation_time'] = time.time()
        
        # Track error for precision calculation
        entity_id = prediction['entity_id']
        self.prediction_errors[entity_id].append(error)
        
        # Limit history size
        if len(self.prediction_errors[entity_id]) > 20:
            self.prediction_errors[entity_id] = self.prediction_errors[entity_id][-20:]
        
        # Update precision (inverse variance)
        if len(self.prediction_errors[entity_id]) > 1:
            variance = np.var(self.prediction_errors[entity_id])
            precision = 1.0 / (variance + 1e-6)  # Avoid division by zero
            self.precision_values[entity_id] = min(10.0, precision)  # Cap very high precision
            
        # Calculate adaptive learning rate
        # Higher error = higher learning rate
        # Higher precision = lower learning rate (more cautious)
        precision = self.get_precision(entity_id)
        base_rate = min(0.8, error * 2)  # Error-proportional component
        precision_factor = max(0.1, min(0.9, 1.0 / (1.0 + precision * 0.2)))
        learning_rate = min(0.9, max(0.1, base_rate * precision_factor))
        self.learning_rates[entity_id] = learning_rate
        
        return {
            'prediction_id': prediction_id,
            'error': error,
            'precision': precision,
            'learning_rate': learning_rate
        }
        
    def get_precision(self, entity_id: str) -> float:
        """Get precision for a specific entity"""
        return self.precision_values.get(entity_id, 1.0)
        
    def get_learning_rate(self, entity_id: str) -> float:
        """Get adaptive learning rate for an entity"""
        return self.learning_rates.get(entity_id, 0.3)
        
    def _calculate_error(self, predicted: Any, actual: Any) -> float:
        """Calculate normalized error between predicted and actual values"""
        if isinstance(predicted, (int, float, np.number)) and isinstance(actual, (int, float, np.number)):
            # For numeric values, normalized absolute difference
            return abs(predicted - actual) / (1.0 + abs(actual))
            
        elif isinstance(predicted, (list, np.ndarray)) and isinstance(actual, (list, np.ndarray)):
            # For vectors, normalized Euclidean distance
            predicted_arr = np.array(predicted)
            actual_arr = np.array(actual)
            
            if predicted_arr.shape != actual_arr.shape:
                return 1.0  # Maximum error for shape mismatch
                
            if predicted_arr.size == 0 or actual_arr.size == 0:
                return 1.0  # Maximum error for empty arrays
                
            # Normalized Euclidean distance
            diff = np.linalg.norm(predicted_arr - actual_arr)
            norm = np.linalg.norm(actual_arr)
            return min(1.0, diff / (1.0 + norm))
            
        elif isinstance(predicted, dict) and isinstance(actual, dict):
            # For dictionaries, calculate average error across shared keys
            shared_keys = set(predicted.keys()) & set(actual.keys())
            
            if not shared_keys:
                return 1.0  # Maximum error if no shared keys
                
            errors = []
            for key in shared_keys:
                errors.append(self._calculate_error(predicted[key], actual[key]))
                
            return sum(errors) / len(errors)
            
        else:
            # Fallback for other types
            return 1.0 if predicted != actual else 0.0



# === FILE: layer2_autopoietic_maintanance\predictive_recognition.py ===

import time
import joblib
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple

from asf.symbolic_formation.recognition import SymbolRecognizer
from asf.symbolic_formation.symbol import SymbolElement

class PredictiveSymbolRecognizer(SymbolRecognizer):
    """
    Enhances standard symbol recognizer with predictive capabilities.
    Anticipates which symbols will appear before receiving perceptual data.
    """
    def __init__(self, threshold: float = 0.7):
        super().__init__(threshold)
        self.prediction_cache = {}  # Context hash -> predicted symbols
        self.prediction_errors = defaultdict(list)  # Track errors for precision
        self.precision_values = {}  # Symbol ID -> precision value
        
    async def predict_symbols(self, context, existing_symbols):
        """Predict which symbols are likely to be recognized in this context"""
        context_hash = joblib.hash(context)
        
        # Check cache first
        if context_hash in self.prediction_cache:
            # Return cached prediction if not too old
            cached = self.prediction_cache[context_hash]
            if time.time() - cached['timestamp'] < 300:  # 5 minutes validity
                return cached['predictions']
        
        # Generate predictions based on symbol relevance to context
        predictions = {}
        for symbol_id, symbol in existing_symbols.items():
            # Calculate context relevance based on actualized meaning
            meaning = symbol.actualize_meaning(context_hash, context)
            if meaning:
                # Use total activation as prediction confidence
                confidence = sum(meaning.values()) / len(meaning)
                predictions[symbol_id] = min(0.95, confidence)
            
        # Store prediction in cache
        self.prediction_cache[context_hash] = {
            'predictions': predictions,
            'timestamp': time.time()
        }
        
        return predictions
        
    async def recognize(self, perceptual_data, existing_symbols, context=None):
        """First predict symbols, then compare with actual recognition"""
        context = context or {}
        context_hash = joblib.hash(context)
        
        # Make prediction before actual recognition
        predictions = await self.predict_symbols(context, existing_symbols)
        
        # Perform actual recognition
        result = await super().recognize(perceptual_data, existing_symbols, context)
        
        # Calculate prediction error if recognition was successful
        if result['recognized']:
            symbol_id = result['symbol_id']
            predicted_confidence = predictions.get(symbol_id, 0.0)
            prediction_error = abs(predicted_confidence - result['confidence'])
            
            # Track prediction error
            self.prediction_errors[symbol_id].append(prediction_error)
            
            # Limit history size
            if len(self.prediction_errors[symbol_id]) > 20:
                self.prediction_errors[symbol_id] = self.prediction_errors[symbol_id][-20:]
            
            # Update precision (inverse variance of prediction errors)
            if len(self.prediction_errors[symbol_id]) > 1:
                variance = np.var(self.prediction_errors[symbol_id])
                precision = 1.0 / (variance + 1e-6)  # Avoid division by zero
                self.precision_values[symbol_id] = min(10.0, precision)  # Cap very high precision
            
            # Add prediction information to result
            result['predicted_confidence'] = predicted_confidence
            result['prediction_error'] = prediction_error
            result['precision'] = self.precision_values.get(symbol_id, 1.0)
            
        return result
    
    def get_prediction_precision(self, symbol_id):
        """Calculate precision (inverse variance) of predictions for a symbol"""
        errors = self.prediction_errors.get(symbol_id, [])
        if len(errors) < 2:
            return 1.0  # Default precision
            
        variance = np.var(errors)
        precision = 1.0 / (variance + 1e-6)  # Avoid division by zero
        return min(10.0, precision)  # Cap very high precision



# === FILE: layer2_autopoietic_maintanance\predictive_symbol.py ===

import time
import math
import numpy as np
import joblib
from collections import defaultdict
from typing import Dict, Any, List, Optional, Set, Tuple

from asf.symbolic_formation.symbol import SymbolElement
from asf.symbolic_formation.potentials import SymbolicPotential
from asf.symbolic_formation.enums import NonlinearityOrder, SymbolConfidenceState

class PredictiveSymbolElement(SymbolElement):
    """
    Enhanced symbol with predictive capabilities.
    Predicts meaning activations before observing context.
    """
    def __init__(self, symbol_id: str, perceptual_anchors: Dict[str, float] = None):
        super().__init__(symbol_id, perceptual_anchors)
        self.predicted_meanings = {}  # Context hash -> predicted meanings
        self.prediction_errors = defaultdict(list)  # Context -> errors
        self.precision_values = {}  # Context -> precision (1/variance)
        
    def predict_meaning(self, context_hash: str, context: Dict[str, Any]) -> Dict[str, float]:
        """Predict meaning before actualization based on similar contexts"""
        # If we don't have enough historical data, can't predict meaningfully
        if len(self._actual_meanings) < 3:
            return {}
            
        # Find similar contexts based on keys
        similar_contexts = []
        context_keys = set(context.keys())
        
        for past_hash in self._actual_meanings:
            # Simple similarity heuristic - add to similar if we have actual data
            if past_hash in self._actual_meanings:
                similar_contexts.append(past_hash)
                
        if not similar_contexts:
            return {}
            
        # Calculate predicted meaning based on similar contexts
        predicted = {}
        weights = {}
        
        for past_hash in similar_contexts:
            past_meaning = self._actual_meanings[past_hash]
            
            # Weight by recency (more recent contexts get higher weight)
            weight = 1.0
            if past_hash in self._activation_time:
                # Decay weight by time since activation
                time_elapsed = time.time() - self._activation_time[past_hash]
                decay_factor = math.exp(-time_elapsed / 3600)  # 1-hour half-life
                weight *= decay_factor
            
            # Accumulate weighted meanings
            for potential_id, activation in past_meaning.items():
                if potential_id not in predicted:
                    predicted[potential_id] = 0.0
                    weights[potential_id] = 0.0
                
                predicted[potential_id] += activation * weight
                weights[potential_id] += weight
        
        # Normalize by weights
        for potential_id in predicted:
            if weights[potential_id] > 0:
                predicted[potential_id] /= weights[potential_id]
        
        # Store prediction
        self.predicted_meanings[context_hash] = predicted
        return predicted
        
    def actualize_meaning(self, context_hash: str, context: Dict[str, Any]) -> Dict[str, float]:
        """First predict meaning, then actualize and compare"""
        # Make prediction if we have prior contexts
        predicted = {}
        if len(self._actual_meanings) >= 3:
            predicted = self.predict_meaning(context_hash, context)
        
        # Actual actualization from parent class
        actualized = super().actualize_meaning(context_hash, context)
        
        # Calculate prediction error if prediction was made
        if predicted:
            # Calculate error for each potential
            errors = []
            for potential_id in set(list(predicted.keys()) + list(actualized.keys())):
                pred_value = predicted.get(potential_id, 0.0)
                actual_value = actualized.get(potential_id, 0.0)
                error = abs(pred_value - actual_value)
                errors.append(error)
            
            # Average error across all potentials
            if errors:
                avg_error = sum(errors) / len(errors)
                self.prediction_errors[context_hash].append(avg_error)
                
                # Calculate precision (inverse variance)
                if len(self.prediction_errors[context_hash]) > 1:
                    variance = np.var(self.prediction_errors[context_hash])
                    precision = 1.0 / (variance + 1e-6)
                    self.precision_values[context_hash] = precision
        
        return actualized
    
    def get_prediction_precision(self, context_hash=None):
        """Get prediction precision for a context or overall"""
        if context_hash and context_hash in self.precision_values:
            return self.precision_values[context_hash]
            
        # Calculate overall precision across all contexts
        all_errors = []
        for errors in self.prediction_errors.values():
            all_errors.extend(errors)
            
        if len(all_errors) < 2:
            return 1.0  # Default precision
            
        variance = np.var(all_errors)
        precision = 1.0 / (variance + 1e-6)
        return min(10.0, precision)  # Cap very high precision



# === FILE: layer2_autopoietic_maintanance\recognition.py ===

import time
import numpy as np
import asyncio
from typing import Dict, List, Any, Tuple
from collections import defaultdict

from asf.symbolic_formation.symbol import SymbolElement

class SymbolRecognizer:
    """
    Recognizes existing symbols from perceptual data.
    Enhanced with multi-strategy recognition approaches.
    """
    def __init__(self, threshold: float = 0.7):
        self.recognition_threshold = threshold
        self.recognition_history = []
        # Phase 2 enhancement: multiple recognition strategies
        self.strategies = {
            'anchor_matching': self._recognize_by_anchors,
            'embedding_similarity': self._recognize_by_embedding,
            'feature_mapping': self._recognize_by_feature_mapping
        }
        self.strategy_weights = {
            'anchor_matching': 0.6,
            'embedding_similarity': 0.2,
            'feature_mapping': 0.2
        }
    
    async def recognize(self, perceptual_data: Dict[str, Dict[str, float]],
                      existing_symbols: Dict[str, SymbolElement],
                      context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Recognize symbols from perceptual data using multiple strategies.
        
        Args:
            perceptual_data: Dictionary of perceptual features
            existing_symbols: Dictionary of existing symbols
            context: Optional context information
            
        Returns:
            Recognition result
        """
        context = context or {}
        
        # Flatten perceptual data for processing
        flat_perceptual = self._flatten_perceptual_data(perceptual_data)
        
        # Results from each strategy
        strategy_results = {}
        
        # Apply each recognition strategy
        for strategy_name, strategy_func in self.strategies.items():
            weight = self.strategy_weights[strategy_name]
            result = await strategy_func(flat_perceptual, existing_symbols, context)
            
            if result['recognized']:
                strategy_results[strategy_name] = {
                    'symbol_id': result['symbol_id'],
                    'confidence': result['confidence'],
                    'weighted_confidence': result['confidence'] * weight
                }
                
        # Combine strategy results
        if not strategy_results:
            return {
                'recognized': False,
                'confidence': 0.0,
                'strategies_applied': list(self.strategies.keys())
            }
            
        # Find best match across strategies
        best_strategy = max(strategy_results.items(), 
                          key=lambda x: x[1]['weighted_confidence'])
        
        strategy_name, result = best_strategy
        weighted_confidence = result['weighted_confidence']
        
        # Final decision based on confidence threshold
        if weighted_confidence >= self.recognition_threshold:
            self.recognition_history.append({
                'timestamp': time.time(),
                'symbol_id': result['symbol_id'],
                'confidence': result['confidence'],
                'weighted_confidence': weighted_confidence,
                'strategy': strategy_name
            })
            
            return {
                'recognized': True,
                'symbol_id': result['symbol_id'],
                'confidence': result['confidence'],
                'weighted_confidence': weighted_confidence,
                'strategy': strategy_name,
                'strategies_applied': list(strategy_results.keys())
            }
            
        return {
            'recognized': False,
            'confidence': weighted_confidence,
            'best_match': result['symbol_id'],
            'strategies_applied': list(self.strategies.keys())
        }
    
    async def _recognize_by_anchors(self, flat_perceptual: Dict[str, float],
                                 existing_symbols: Dict[str, SymbolElement],
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recognize symbols based on perceptual anchor matching.
        """
        best_match = None
        best_score = 0.0
        
        # Group symbols by anchor keys for faster matching
        anchor_to_symbols = defaultdict(list)
        for symbol_id, symbol in existing_symbols.items():
            for anchor in symbol.perceptual_anchors:
                anchor_to_symbols[anchor].append(symbol_id)
                
        # For each perceptual feature, check candidate symbols
        candidates = {}
        for feature, strength in flat_perceptual.items():
            if feature in anchor_to_symbols and strength > 0.3:  # Threshold
                for symbol_id in anchor_to_symbols[feature]:
                    if symbol_id not in candidates:
                        candidates[symbol_id] = 0
                    candidates[symbol_id] += strength
                    
        # Detailed perceptual match for candidates
        for symbol_id, initial_score in candidates.items():
            symbol = existing_symbols[symbol_id]
            match_score = self._calculate_perceptual_match(symbol, flat_perceptual)
            
            if match_score > best_score:
                best_score = match_score
                best_match = symbol_id
                
        return {
            'recognized': best_score >= self.recognition_threshold,
            'symbol_id': best_match,
            'confidence': best_score
        }
    
    async def _recognize_by_embedding(self, flat_perceptual: Dict[str, float],
                                   existing_symbols: Dict[str, SymbolElement],
                                   context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recognize symbols based on embedding similarity.
        """
        # This is a simplified implementation. In a real system, this would use  
        # feature embeddings and compute semantic similarity.
        if not flat_perceptual:
            return {'recognized': False, 'symbol_id': None, 'confidence': 0.0}
            
        # Create a simplified feature vector from flat_perceptual
        percept_vec = np.zeros(128)
        for i, (key, value) in enumerate(flat_perceptual.items()):
            hash_val = hash(key) % 128
            percept_vec[hash_val] = value
            
        # Normalize
        norm = np.linalg.norm(percept_vec)
        if norm > 0:
            percept_vec = percept_vec / norm
            
        # Find most similar symbol
        best_match = None
        best_similarity = 0.0
        
        for symbol_id, symbol in existing_symbols.items():
            # Create similar simplified vector for symbol
            sym_vec = np.zeros(128)
            for anchor, strength in symbol.perceptual_anchors.items():
                hash_val = hash(anchor) % 128
                sym_vec[hash_val] = strength
                
            # Normalize
            norm = np.linalg.norm(sym_vec)
            if norm > 0:
                sym_vec = sym_vec / norm
                
            # Calculate cosine similarity
            similarity = np.dot(percept_vec, sym_vec)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = symbol_id
                
        return {
            'recognized': best_similarity >= self.recognition_threshold,
            'symbol_id': best_match,
            'confidence': best_similarity
        }
    
    async def _recognize_by_feature_mapping(self, flat_perceptual: Dict[str, float],
                                         existing_symbols: Dict[str, SymbolElement],
                                         context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recognize symbols based on detailed feature mapping.
        """
        if not existing_symbols:
            return {'recognized': False, 'symbol_id': None, 'confidence': 0.0}
            
        # Calculate match scores for all symbols
        match_scores = []
        for symbol_id, symbol in existing_symbols.items():
            context_hash = str(hash(str(context)))
            
            # Get actualized meaning in current context
            meaning = symbol.actualize_meaning(context_hash, context)
            
            # Calculate mapping between perceptual data and meaning
            match_score = self._calculate_feature_mapping(flat_perceptual, meaning)
            match_scores.append((symbol_id, match_score))
            
        # Find best match
        if not match_scores:
            return {'recognized': False, 'symbol_id': None, 'confidence': 0.0}
            
        best_match = max(match_scores, key=lambda x: x[1])
        symbol_id, score = best_match
        
        return {
            'recognized': score >= self.recognition_threshold,
            'symbol_id': symbol_id,
            'confidence': score
        }
    
    def _flatten_perceptual_data(self, perceptual_data: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Flatten hierarchical perceptual data into a simple key-value dictionary.
        """
        flat_data = {}
        for entity_id, features in perceptual_data.items():
            for feature_name, value in features.items():
                key = f"{entity_id}:{feature_name}"
                flat_data[key] = value
        return flat_data
    
    def _calculate_perceptual_match(self, symbol: SymbolElement, perceptual: Dict[str, float]) -> float:
        """
        Calculate match score between symbol anchors and perceptual data.
        """
        if not symbol.perceptual_anchors or not perceptual:
            return 0.0
            
        # Calculate overlap between anchors and perceptual features
        overlap_score = 0.0
        total_weight = 0.0
        
        for anchor, anchor_strength in symbol.perceptual_anchors.items():
            # Look for exact matches
            if anchor in perceptual:
                overlap_score += anchor_strength * perceptual[anchor]
                total_weight += anchor_strength
            else:
                # Look for partial matches
                for percept_key, percept_value in perceptual.items():
                    if anchor in percept_key or (isinstance(percept_key, str) and percept_key in anchor):
                        partial_score = 0.7 * anchor_strength * percept_value  # Reduce score for partial match
                        overlap_score += partial_score
                        total_weight += anchor_strength
                        break
        
        # Normalize score
        if total_weight > 0:
            return overlap_score / total_weight
        return 0.0
    
    def _calculate_feature_mapping(self, perceptual: Dict[str, float], meaning: Dict[str, float]) -> float:
        """
        Calculate mapping between perceptual features and symbol meaning.
        """
        if not perceptual or not meaning:
            return 0.0
            
        # Create simplified feature vectors
        perc_vec = np.zeros(256)
        mean_vec = np.zeros(256)
        
        # Fill perceptual vector
        for key, value in perceptual.items():
            idx = hash(key) % 256
            perc_vec[idx] = value
            
        # Fill meaning vector
        for key, value in meaning.items():
            idx = hash(key) % 256
            mean_vec[idx] = value
            
        # Calculate cosine similarity
        perc_norm = np.linalg.norm(perc_vec)
        mean_norm = np.linalg.norm(mean_vec)
        
        if perc_norm > 0 and mean_norm > 0:
            similarity = np.dot(perc_vec, mean_vec) / (perc_norm * mean_norm)
            return similarity
            
        return 0.0



# === FILE: layer2_autopoietic_maintanance\resolution_strategies.py ===




# === FILE: layer2_autopoietic_maintanance\symbol.py ===

import time
import numpy as np
from typing import Dict, Any, List
from asf.symbolic_formation.enums import NonlinearityOrder, SymbolConfidenceState
from asf.symbolic_formation.potentials import SymbolicPotential

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

