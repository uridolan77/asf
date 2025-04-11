import asyncio
import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveContextTracker:
    """
    Tracks contextual information for environmental interactions.
    Enhanced with predictive context modeling based on Seth's principles.
    """
    def __init__(self):
        self.entity_contexts = {}  # Maps entity_id to context information
        self.coupling_contexts = {}  # Maps coupling_id to context information
        self.interaction_contexts = defaultdict(list)  # Maps (entity_id, environmental_id) to context history
        self.global_context = {
            'start_time': time.time(),
            'context_updates': 0,
            'active_entities': set()
        }
        
        self.predicted_contexts = {}  # Maps entity_id to predicted future contexts
        self.context_embeddings = {}  # Maps context_id to embedding vector
        self.context_evolution = defaultdict(list)  # Maps entity_id to context evolution history
        
        self.use_neural_model = False
        self.context_model = None
        
        self.logger = logging.getLogger("ASF.Layer4.AdaptiveContextTracker")
        
    async def create_context(self, interaction_data, source_id=None, interaction_type=None):
        context = {
            'timestamp': time.time(),
            'interaction_type': interaction_type,
            'source_id': source_id,
            'context_id': f"ctx_{int(time.time())}_{hash(str(interaction_data))%1000}"
        }
        
        context['global_state'] = {
            'uptime': time.time() - self.global_context['start_time'],
            'active_entities': len(self.global_context['active_entities'])
        }
        
        if source_id and source_id in self.entity_contexts:
            source_context = self.entity_contexts[source_id]
            context['source_context'] = {
                'last_interaction': source_context.get('last_interaction'),
                'interaction_count': source_context.get('interaction_count', 0),
                'typical_patterns': source_context.get('typical_patterns', {})
            }
            
        if source_id and source_id in self.predicted_contexts:
            predicted = self.predicted_contexts[source_id]
            
            accuracy = self._calculate_context_match(context, predicted)
            
            context['prediction'] = {
                'accuracy': accuracy,
                'predicted_time': predicted.get('predicted_time'),
                'prediction_delta': time.time() - predicted.get('prediction_time', 0)
            }
            
            context = self._merge_contexts(context, predicted)
            
        if self.use_neural_model and self.context_model is not None:
            context_embedding = await self._create_context_embedding(context)
            self.context_embeddings[context['context_id']] = context_embedding
            
        return context
        
    async def register_coupling(self, coupling):
        Record an interaction in the context tracker.
        Now enhanced with context prediction after interaction.
        Predict future context for an entity pair.
        Implements Seth's controlled hallucination principle for context.
        if entity_id not in self.entity_contexts:
            return 0.0
            
        entity_context = self.entity_contexts[entity_id]
        
        if 'last_interaction' not in entity_context or entity_context['last_interaction'] is None:
            return 0.0
            
        current_time = time.time()
        time_since_last = current_time - entity_context['last_interaction']
        interaction_count = entity_context.get('interaction_count', 0)
        
        recency_factor = np.exp(-time_since_last / 3600)  # 1-hour half-life
        
        frequency_factor = min(1.0, interaction_count / 10)
        
        return recency_factor * frequency_factor
        
    def _calculate_context_match(self, actual_context, predicted_context):
        """Calculate how well an actual context matches a prediction."""
        match_score = 0.0
        match_count = 0
        
        # Match timestamp
        if 'timestamp' in actual_context and 'predicted_time' in predicted_context:
            time_diff = abs(actual_context['timestamp'] - predicted_context['predicted_time'])
            time_threshold = predicted_context.get('avg_interval', 3600) / 2
            time_score = max(0.0, 1.0 - (time_diff / time_threshold))
            match_score += time_score
            match_count += 1
            
        # Match interaction type
        if 'interaction_type' in actual_context and 'predicted_type' in predicted_context:
            type_score = 1.0 if actual_context['interaction_type'] == predicted_context['predicted_type'] else 0.0
            match_score += type_score
            match_count += 1
            
        # Calculate overall match score
        if match_count > 0:
            return match_score / match_count
        return 0.0
        
    def _merge_contexts(self, actual_context, predicted_context):
        """Merge actual context with predicted context elements."""
        merged = actual_context.copy()
        
        for key, value in predicted_context.items():
            if key.endswith('_predicted') and key not in merged:
                merged[key] = value
                
        return merged
        
    async def _create_context_embedding(self, context):
        start_time = time.time()
        
        expiry_threshold = time.time() - 86400  # 24 hours
        expired_entities = []
        
        for entity_id, context in self.entity_contexts.items():
            last_interaction = context.get('last_interaction')
            if last_interaction and last_interaction < expiry_threshold:
                expired_entities.append(entity_id)
                
        for entity_id in expired_entities:
            del self.entity_contexts[entity_id]
            if entity_id in self.predicted_contexts:
                del self.predicted_contexts[entity_id]
            if entity_id in self.context_evolution:
                del self.context_evolution[entity_id]
                
        expired_interactions = []
        for key, history in self.interaction_contexts.items():
            new_history = [entry for entry in history if entry.get('timestamp', 0) > expiry_threshold]
            
            if not new_history and history:
                expired_interactions.append(key)
            else:
                self.interaction_contexts[key] = new_history
                
        for key in expired_interactions:
            del self.interaction_contexts[key]
            
        self.global_context['active_entities'] = set(self.entity_contexts.keys())
        
        return {
            'entity_contexts': len(self.entity_contexts),
            'coupling_contexts': len(self.coupling_contexts),
            'interaction_contexts': len(self.interaction_contexts),
            'predicted_contexts': len(self.predicted_contexts),
            'expired_entities': len(expired_entities),
            'expired_interactions': len(expired_interactions),
            'elapsed_time': time.time() - start_time
        }
        
    async def get_metrics(self):