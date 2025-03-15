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
