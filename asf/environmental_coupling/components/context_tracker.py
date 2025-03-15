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
        
        # Seth's Data Paradox enhancements
        self.predicted_contexts = {}  # Maps entity_id to predicted future contexts
        self.context_embeddings = {}  # Maps context_id to embedding vector
        self.context_evolution = defaultdict(list)  # Maps entity_id to context evolution history
        
        # Context model
        self.use_neural_model = False
        self.context_model = None
        
        self.logger = logging.getLogger("ASF.Layer4.AdaptiveContextTracker")
        
    async def create_context(self, interaction_data, source_id=None, interaction_type=None):
        """
        Create a context for an interaction.
        Now enhanced with predictive context generation.
        """
        # Generate basic context
        context = {
            'timestamp': time.time(),
            'interaction_type': interaction_type,
            'source_id': source_id,
            'context_id': f"ctx_{int(time.time())}_{hash(str(interaction_data))%1000}"
        }
        
        # Add global context information
        context['global_state'] = {
            'uptime': time.time() - self.global_context['start_time'],
            'active_entities': len(self.global_context['active_entities'])
        }
        
        # Add source entity context if available
        if source_id and source_id in self.entity_contexts:
            source_context = self.entity_contexts[source_id]
            context['source_context'] = {
                'last_interaction': source_context.get('last_interaction'),
                'interaction_count': source_context.get('interaction_count', 0),
                'typical_patterns': source_context.get('typical_patterns', {})
            }
            
        # If we have a predicted context for this entity, compare and merge
        if source_id and source_id in self.predicted_contexts:
            predicted = self.predicted_contexts[source_id]
            
            # Calculate prediction accuracy
            accuracy = self._calculate_context_match(context, predicted)
            
            # Include prediction accuracy in context
            context['prediction'] = {
                'accuracy': accuracy,
                'predicted_time': predicted.get('predicted_time'),
                'prediction_delta': time.time() - predicted.get('prediction_time', 0)
            }
            
            # Merge predictive elements with current context
            context = self._merge_contexts(context, predicted)
            
        # Create embedding for this context
        if self.use_neural_model and self.context_model is not None:
            context_embedding = await self._create_context_embedding(context)
            self.context_embeddings[context['context_id']] = context_embedding
            
        return context
        
    async def register_coupling(self, coupling):
        """Register a new coupling in the context tracker."""
        coupling_id = coupling.id
        internal_id = coupling.internal_entity_id
        environmental_id = coupling.environmental_entity_id
        
        # Create coupling context
        self.coupling_contexts[coupling_id] = {
            'creation_time': time.time(),
            'internal_entity_id': internal_id,
            'environmental_entity_id': environmental_id,
            'interaction_count': 0,
            'last_interaction': None,
            'coupling_type': coupling.coupling_type.name,
            'context_history': []
        }
        
        # Ensure entity contexts exist
        if internal_id not in self.entity_contexts:
            self.entity_contexts[internal_id] = {
                'creation_time': time.time(),
                'interaction_count': 0,
                'last_interaction': None,
                'connections': set()
            }
            
        if environmental_id not in self.entity_contexts:
            self.entity_contexts[environmental_id] = {
                'creation_time': time.time(),
                'interaction_count': 0,
                'last_interaction': None,
                'connections': set()
            }
            
        # Update connections
        self.entity_contexts[internal_id]['connections'].add(environmental_id)
        self.entity_contexts[environmental_id]['connections'].add(internal_id)
        
        # Update active entities set
        self.global_context['active_entities'].add(internal_id)
        self.global_context['active_entities'].add(environmental_id)
        
        return True
        
    async def record_interaction(self, internal_id, environmental_id, interaction_type, update_result):
        """
        Record an interaction in the context tracker.
        Now enhanced with context prediction after interaction.
        """
        timestamp = time.time()
        
        # Update entity contexts
        for entity_id in [internal_id, environmental_id]:
            if entity_id and entity_id in self.entity_contexts:
                self.entity_contexts[entity_id]['interaction_count'] += 1
                self.entity_contexts[entity_id]['last_interaction'] = timestamp
                
                # Update typical patterns
                if 'typical_patterns' not in self.entity_contexts[entity_id]:
                    self.entity_contexts[entity_id]['typical_patterns'] = {}
                    
                patterns = self.entity_contexts[entity_id]['typical_patterns']
                if interaction_type not in patterns:
                    patterns[interaction_type] = 0
                patterns[interaction_type] += 1
                
        # Update interaction context
        if internal_id and environmental_id:
            key = (internal_id, environmental_id)
            
            # Create context for this interaction
            interaction_context = {
                'timestamp': timestamp,
                'interaction_type': interaction_type,
                'bayesian_confidence': update_result.get('new_confidence', 0.5),
                'contradiction': update_result.get('contradiction_detected', False)
            }
            
            # Add to history
            self.interaction_contexts[key].append(interaction_context)
            
            # Limit history size
            if len(self.interaction_contexts[key]) > 100:
                self.interaction_contexts[key] = self.interaction_contexts[key][-100:]
                
            # Update coupling context if we can find a matching coupling
            for coupling_id, context in self.coupling_contexts.items():
                if (context['internal_entity_id'] == internal_id and 
                    context['environmental_entity_id'] == environmental_id):
                    context['interaction_count'] += 1
                    context['last_interaction'] = timestamp
                    context['context_history'].append(interaction_context)
                    
                    # Limit history size
                    if len(context['context_history']) > 20:
                        context['context_history'] = context['context_history'][-20:]
                        
        # Increment global context update counter
        self.global_context['context_updates'] += 1
        
        # Generate context prediction for future interactions
        await self._predict_future_context(internal_id, environmental_id, interaction_type)
        
        # Track context evolution
        if internal_id and environmental_id:
            interaction_context = {
                'timestamp': timestamp,
                'interaction_type': interaction_type,
                'entities': (internal_id, environmental_id)
            }
            
            # Calculate context change from previous
            previous_contexts = self.context_evolution.get(internal_id, [])
            if previous_contexts:
                prev_context = previous_contexts[-1]
                delta = timestamp - prev_context['timestamp']
                
                interaction_context['time_delta'] = delta
                interaction_context['evolution_rate'] = 1.0 / max(1.0, delta)
            
            # Add to evolution history
            self.context_evolution[internal_id].append(interaction_context)
            
            # Limit history size
            if len(self.context_evolution[internal_id]) > 50:
                self.context_evolution[internal_id] = self.context_evolution[internal_id][-50:]
                
        return True
        
    async def _predict_future_context(self, internal_id, environmental_id, interaction_type):
        """
        Predict future context for an entity pair.
        Implements Seth's controlled hallucination principle for context.
        """
        if not internal_id or not environmental_id:
            return None
            
        # Get interaction history
        key = (internal_id, environmental_id)
        history = self.interaction_contexts.get(key, [])
        
        if len(history) < 2:
            return None
            
        # Calculate average time between interactions
        timestamps = [entry['timestamp'] for entry in history]
        intervals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
        avg_interval = np.mean(intervals)
        
        # Predict next interaction time
        next_time = time.time() + avg_interval
        
        # Count interaction types to predict most likely next type
        type_counts = defaultdict(int)
        for entry in history:
            entry_type = entry.get('interaction_type')
            if entry_type:
                type_counts[entry_type] += 1
                
        # Get most common type or use current
        if type_counts:
            predicted_type = max(type_counts.items(), key=lambda x: x[1])[0]
        else:
            predicted_type = interaction_type
            
        # Create predicted context
        predicted_context = {
            'internal_id': internal_id,
            'environmental_id': environmental_id,
            'predicted_time': next_time,
            'prediction_time': time.time(),
            'predicted_type': predicted_type,
            'confidence': min(0.9, len(history) / 10),  # Confidence increases with more history
            'avg_interval': avg_interval
        }
        
        # Add entity-specific context predictions
        for entity_id in [internal_id, environmental_id]:
            if entity_id in self.entity_contexts:
                entity_context = self.entity_contexts[entity_id]
                
                # Predict entity state at future time
                predicted_context[f'{entity_id}_predicted'] = {
                    'interaction_count': entity_context.get('interaction_count', 0) + 1,
                    'connections': len(entity_context.get('connections', set())),
                    'activity_level': self._calculate_activity_level(entity_id)
                }
                
        # Store prediction
        for entity_id in [internal_id, environmental_id]:
            self.predicted_contexts[entity_id] = predicted_context
            
        return predicted_context
        
    def _calculate_activity_level(self, entity_id):
        """Calculate activity level for an entity based on interaction history."""
        if entity_id not in self.entity_contexts:
            return 0.0
            
        entity_context = self.entity_contexts[entity_id]
        
        # No interactions yet
        if 'last_interaction' not in entity_context or entity_context['last_interaction'] is None:
            return 0.0
            
        # Calculate activity based on recency and frequency
        current_time = time.time()
        time_since_last = current_time - entity_context['last_interaction']
        interaction_count = entity_context.get('interaction_count', 0)
        
        # Activity decays exponentially with time
        recency_factor = np.exp(-time_since_last / 3600)  # 1-hour half-life
        
        # Frequency factor increases with interaction count but saturates
        frequency_factor = min(1.0, interaction_count / 10)
        
        # Combined activity level
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
        
        # Add predictive elements that aren't in the actual context
        for key, value in predicted_context.items():
            if key.endswith('_predicted') and key not in merged:
                merged[key] = value
                
        return merged
        
    async def _create_context_embedding(self, context):
        """Create an embedding vector for a context using neural model."""
        # This would use an actual embedding model in a real implementation
        # For this example, we create a simple vector representation
        
        if not self.use_neural_model or self.context_model is None:
            # Fallback to simple representation
            embedding = np.zeros(10)
            embedding[0] = hash(str(context.get('interaction_type', ''))) % 100 / 100
            embedding[1] = context.get('timestamp', 0) % 86400 / 86400  # Time of day
            embedding[2] = min(1.0, context.get('prediction', {}).get('accuracy', 0))
            
            return torch.tensor(embedding, dtype=torch.float32)
            
        # Neural model implementation would go here
        
        return torch.zeros(10, dtype=torch.float32)
        
    async def perform_maintenance(self):
        """Perform periodic maintenance on context tracking."""
        start_time = time.time()
        
        # Clean up expired contexts
        expiry_threshold = time.time() - 86400  # 24 hours
        expired_entities = []
        
        for entity_id, context in self.entity_contexts.items():
            last_interaction = context.get('last_interaction')
            if last_interaction and last_interaction < expiry_threshold:
                expired_entities.append(entity_id)
                
        # Remove expired entities
        for entity_id in expired_entities:
            del self.entity_contexts[entity_id]
            if entity_id in self.predicted_contexts:
                del self.predicted_contexts[entity_id]
            if entity_id in self.context_evolution:
                del self.context_evolution[entity_id]
                
        # Remove expired interaction contexts
        expired_interactions = []
        for key, history in self.interaction_contexts.items():
            # Filter out old entries
            new_history = [entry for entry in history if entry.get('timestamp', 0) > expiry_threshold]
            
            # If all entries expired, mark for removal
            if not new_history and history:
                expired_interactions.append(key)
            else:
                self.interaction_contexts[key] = new_history
                
        # Remove expired interaction contexts
        for key in expired_interactions:
            del self.interaction_contexts[key]
            
        # Update global context
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
        """Get metrics about the context tracker."""
        # Calculate prediction accuracy
        prediction_accuracies = []
        for entity_id, context in self.entity_contexts.items():
            last_interaction = context.get('last_interaction')
            if last_interaction and entity_id in self.predicted_contexts:
                predicted = self.predicted_contexts[entity_id]
                if 'predicted_time' in predicted:
                    time_diff = abs(last_interaction - predicted['predicted_time'])
                    time_threshold = predicted.get('avg_interval', 3600) / 2
                    accuracy = max(0.0, 1.0 - (time_diff / time_threshold))
                    prediction_accuracies.append(accuracy)
                    
        avg_prediction_accuracy = np.mean(prediction_accuracies) if prediction_accuracies else 0.0
        
        # Calculate context evolution rate
        evolution_rates = []
        for entity_id, history in self.context_evolution.items():
            if len(history) >= 2:
                rates = [entry.get('evolution_rate', 0) for entry in history if 'evolution_rate' in entry]
                if rates:
                    avg_rate = np.mean(rates)
                    evolution_rates.append(avg_rate)
                    
        avg_evolution_rate = np.mean(evolution_rates) if evolution_rates else 0.0
        
        return {
            'entity_contexts': len(self.entity_contexts),
            'coupling_contexts': len(self.coupling_contexts),
            'interaction_contexts': len(self.interaction_contexts),
            'context_embeddings': len(self.context_embeddings),
            'predicted_contexts': len(self.predicted_contexts),
            'avg_prediction_accuracy': avg_prediction_accuracy,
            'avg_evolution_rate': avg_evolution_rate,
            'uptime': time.time() - self.global_context['start_time'],
            'context_updates': self.global_context['context_updates']
        }
