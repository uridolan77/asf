import time
import joblib
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

from asf.layer2_autopoietic_maintanance.autopoietic_maintanance_layer import SymbolicFormationLayer
from asf.layer2_autopoietic_maintanance.predictive_recognition import PredictiveSymbolRecognizer
from asf.layer2_autopoietic_maintanance.counterfactual_network import CounterfactualAutocatalyticNetwork
from asf.layer2_autopoietic_maintanance.predictive_processor import SymbolicPredictiveProcessor

class PredictiveSymbolicFormationLayer(SymbolicFormationLayer):
    """
    Enhanced symbolic formation layer implementing Seth's Data Paradox principles.
    Integrates predictive processing and counterfactual reasoning for improved
    symbol formation and recognition.
    """
    def __init__(self, config=None):
        super().__init__(config)
        self.recognizer = PredictiveSymbolRecognizer(
            threshold=self.config.get('recognition_threshold', 0.7)
        )
        
        self.predictive_processor = SymbolicPredictiveProcessor()
        
        self.autocatalytic_network = CounterfactualAutocatalyticNetwork(self.nonlinearity_tracker)
        
        self.perceptual_predictions = {}
        self.perceptual_prediction_errors = defaultdict(list)
    
    async def process_perceptual_input(self, perceptual_data, context=None):
        Generate predictions about perceptual entities that should appear
        based on activated symbols. These can be sent to Layer 1 to guide
        perception.
        context_hash = joblib.hash(context)
        relevant_symbols = {}
        
        for symbol_id, symbol in self.symbols.items():
            meaning = symbol.actualize_meaning(context_hash, context)
            if meaning:
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
        
        flat_actual = {}
        for entity_id, features in actual_perceptual.items():
            entity_type = entity_id.split('_')[0]  # Extract type from ID
            for feature, value in features.items():
                flat_actual[f"{entity_type}:{feature}"] = value
        
        flat_predicted = {}
        for entity_type, features in predicted.items():
            for feature, value in features.items():
                flat_predicted[f"{entity_type}:{feature}"] = value
        
        tp = 0
        fp = 0
        fn = 0
        
        for key, pred_value in flat_predicted.items():
            if pred_value > 0.3:  # Prediction threshold
                if key in flat_actual and flat_actual[key] > 0.3:
                    tp += 1
                else:
                    fp += 1
        
        for key, actual_value in flat_actual.items():
            if actual_value > 0.3 and (key not in flat_predicted or flat_predicted[key] < 0.3):
                fn += 1
        
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
        
        loop = asyncio.get_event_loop()
        actual_result = loop.run_until_complete(
            self.process_perceptual_input(perceptual_data, context)
        )
        
        actual_symbols = {}
        for symbol_info in actual_result.get('new_symbols', []):
            symbol_id = symbol_info.get('symbol_id')
            if symbol_id in self.symbols:
                actual_symbols[symbol_id] = self.symbols[symbol_id]
        
        cf_symbols = self.autocatalytic_network.generate_counterfactual_symbols(
            self.symbols,
            self._flatten_perceptual_data(perceptual_data),
            modification_rules
        )
        
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
