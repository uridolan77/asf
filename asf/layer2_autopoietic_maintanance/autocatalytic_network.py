import numpy as np
import time
import uuid
from collections import defaultdict
from typing import Dict, List, Optional
import torch

from asf.layer2_autopoietic_maintanance.enums import NonlinearityOrder
from asf.layer2_autopoietic_maintanance.symbol import SymbolElement
from asf.layer2_autopoietic_maintanance.network import SparseTensorSymbolNetwork

class NonlinearityOrderTracker:
    """
    Tracks and optimizes order of nonlinearity in symbol transformations.
    Enhanced with learning capabilities for adaptive nonlinearity classification.
    """
    def __init__(self):
        self.symbol_nonlinearity: Dict[str, NonlinearityOrder] = {}
        self.potential_nonlinearity: Dict[str, NonlinearityOrder] = {}
        self.transformation_complexity: Dict[tuple, NonlinearityOrder] = {}
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
            
        order_counts = {}
        for obs in observations:
            if obs.value in order_counts:
                order_counts[obs.value] += 1
            else:
                order_counts[obs.value] = 1
                
        most_common_value = max(order_counts.items(), key=lambda x: x[1])[0]
        most_common_order = NonlinearityOrder(most_common_value)
        
        current_order = self.transformation_complexity.get((source_id, target_id), NonlinearityOrder.LINEAR)
        
        if most_common_order.value != current_order.value:
            if most_common_order.value > current_order.value:
                new_value = min(
                    NonlinearityOrder.COMPOSITIONAL.value,
                    current_order.value + round(self.learning_rate * (most_common_order.value - current_order.value))
                )
            else:
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
        
        self._tensor_network = SparseTensorSymbolNetwork()
        self._template_success_rate: Dict[str, float] = defaultdict(float)
        self._template_usage_count: Dict[str, int] = defaultdict(int)
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
        self.nonlinearity_tracker.register_transformation(
            "template", template_id, nonlinearity
        )
    
    def generate_symbols(self, existing_symbols: Dict[str, SymbolElement],
                       perceptual_inputs: Dict[str, float],
                       threshold: float = 0.5) -> Dict[str, SymbolElement]:
        new_symbols = {}
        initial_activations = {}
        for symbol_id, symbol in existing_symbols.items():
            initial_activations[symbol_id] = symbol.get_pregnancy() / 10  # Normalize
            
        propagated = self._tensor_network.propagate_activations(initial_activations)
        
        sorted_templates = sorted(
            self.production_templates.keys(),
            key=lambda t: (self.template_nonlinearity[t].value, -self._template_success_rate.get(t, 0.0))
        )
        
        for template_id in sorted_templates:
            required = self.production_templates[template_id]
            production_strength = self._calculate_production_strength(propagated, required)
            
            if production_strength > threshold:
                new_id = f"generated_{template_id}_{len(new_symbols)}_{uuid.uuid4().hex[:4]}"
                anchors = {k: v for k, v in perceptual_inputs.items() if v > 0.3}  # Threshold for inclusion
                new_symbol = SymbolElement(new_id, anchors)
                new_symbol.name = f"{template_id}_{new_id[-4:]}"
                
                for req_id, req_strength in required.items():
                    if req_id in propagated and propagated[req_id] > 0.2:
                        if req_id in existing_symbols:
                            self._transfer_potentials(
                                existing_symbols[req_id],
                                new_symbol,
                                req_strength
                            )
                
                if new_symbol.potentials:
                    new_symbols[new_id] = new_symbol
                    
                    self._template_usage_count[template_id] += 1
                    self._template_success_rate[template_id] = (
                        (self._template_success_rate[template_id] * (self._template_usage_count[template_id] - 1)
                         + 1.0) /
                        self._template_usage_count[template_id]
                    )
                    
                    self._adapt_template(template_id, True, new_symbol)
            else:
                self._template_usage_count[template_id] += 1
                self._template_success_rate[template_id] = (
                    (self._template_success_rate[template_id] * (self._template_usage_count[template_id] - 1)
                     + 0.0) /
                    self._template_usage_count[template_id]
                )
                
                self._adapt_template(template_id, False, None)
                
        return new_symbols
    
    def _calculate_production_strength(self, propagated_activations: Dict[str, float],
                                     required: Dict[str, float]) -> float:
        Transfer potentials from source to target symbol.
        Optimized to maintain lower nonlinearity order.
        Adapt template based on success or failure in generating symbols.
        
        Args:
            template_id: Template to adapt
            success: Whether the generation was successful
            new_symbol: The newly generated symbol if success is True