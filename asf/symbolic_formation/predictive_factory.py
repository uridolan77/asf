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
