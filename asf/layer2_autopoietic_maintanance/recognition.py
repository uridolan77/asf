import time
import numpy as np
import asyncio
from typing import Dict, List, Any, Tuple
from collections import defaultdict

from asf.layer2_autopoietic_maintanance.symbol import SymbolElement

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
