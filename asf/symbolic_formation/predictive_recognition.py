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
