import time
import joblib
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple

from asf.layer2_autopoietic_maintanance.recognition import SymbolRecognizer
from asf.layer2_autopoietic_maintanance.symbol import SymbolElement

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
        context = context or {}
        context_hash = joblib.hash(context)
        
        predictions = await self.predict_symbols(context, existing_symbols)
        
        result = await super().recognize(perceptual_data, existing_symbols, context)
        
        if result['recognized']:
            symbol_id = result['symbol_id']
            predicted_confidence = predictions.get(symbol_id, 0.0)
            prediction_error = abs(predicted_confidence - result['confidence'])
            
            self.prediction_errors[symbol_id].append(prediction_error)
            
            if len(self.prediction_errors[symbol_id]) > 20:
                self.prediction_errors[symbol_id] = self.prediction_errors[symbol_id][-20:]
            
            if len(self.prediction_errors[symbol_id]) > 1:
                variance = np.var(self.prediction_errors[symbol_id])
                precision = 1.0 / (variance + 1e-6)  # Avoid division by zero
                self.precision_values[symbol_id] = min(10.0, precision)  # Cap very high precision
            
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
