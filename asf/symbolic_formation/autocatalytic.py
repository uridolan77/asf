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
