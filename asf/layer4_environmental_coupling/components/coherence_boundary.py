import asyncio
import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

class CoherenceBoundaryController:
    """
    Controls the boundary between system and environment with predictive filtering.
    Enhanced with Seth's predictive processing to anticipate and filter interactions.
    """
    def __init__(self, knowledge_substrate):
        self.knowledge_substrate = knowledge_substrate
        self.boundary_rules = {}  # Maps rule_id to rule configuration
        self.interaction_history = defaultdict(list)  # Maps entity_id to interaction history
        self.boundary_metrics = {
            'total_interactions': 0,
            'rejected_interactions': 0,
            'total_distributions': 0,
            'rejected_distributions': 0
        }
        
        self.predicted_interactions = {}  # Maps entity_id to predicted interactions
        self.prediction_accuracy = defaultdict(list)  # Maps entity_id to prediction accuracy history
        self.precision_weights = {}  # Maps entity_id to prediction precision
        
        self.logger = logging.getLogger("ASF.Layer4.CoherenceBoundaryController")
        
    async def check_interaction_coherence(self, interaction_data, source_id, interaction_type, context):
        start_time = time.time()
        self.boundary_metrics['total_interactions'] += 1
        
        prediction_match = False
        if source_id in self.predicted_interactions:
            predicted = self.predicted_interactions[source_id]
            prediction_match = self._compare_with_prediction(interaction_data, predicted)
            
            if predicted.get('prediction_time'):
                accuracy = 1.0 if prediction_match else 0.0
                self.prediction_accuracy[source_id].append({
                    'timestamp': time.time(),
                    'accuracy': accuracy,
                    'prediction_time': predicted['prediction_time']
                })
                
                if len(self.prediction_accuracy[source_id]) > 20:
                    self.prediction_accuracy[source_id] = self.prediction_accuracy[source_id][-20:]
                    
                self._update_precision(source_id)
        
        allowed = True
        rejected_reason = None
        applied_rules = []
        
        for rule_id, rule in self.boundary_rules.items():
            if rule['enabled']:
                matches_rule = self._interaction_matches_rule(interaction_data, source_id, interaction_type, rule)
                applied_rules.append({
                    'rule_id': rule_id,
                    'matches': matches_rule
                })
                
                if matches_rule and rule['action'] == 'reject':
                    allowed = False
                    rejected_reason = rule.get('reason', 'Violates boundary rule')
                    break
                    
        if source_id:
            self.interaction_history[source_id].append({
                'timestamp': time.time(),
                'interaction_type': interaction_type,
                'allowed': allowed,
                'prediction_match': prediction_match
            })
            
            if len(self.interaction_history[source_id]) > 100:
                self.interaction_history[source_id] = self.interaction_history[source_id][-100:]
        
        await self._predict_future_interaction(source_id, interaction_data, interaction_type)
        
        if not allowed:
            self.boundary_metrics['rejected_interactions'] += 1
            
        return allowed, {
            'allowed': allowed,
            'reason': rejected_reason,
            'applied_rules': applied_rules,
            'prediction_match': prediction_match,
            'elapsed_time': time.time() - start_time
        }
        
    async def check_distribution_coherence(self, entity_id, target_id, distribution_type):
        start_time = time.time()
        self.boundary_metrics['total_distributions'] += 1
        
        allowed = True
        rejected_reason = None
        applied_rules = []
        
        entity = await self.knowledge_substrate.get_entity(entity_id)
        if not entity:
            allowed = False
            rejected_reason = "Entity not found"
        else:
            for rule_id, rule in self.boundary_rules.items():
                if rule['enabled'] and rule.get('applies_to_distribution', False):
                    matches_rule = self._distribution_matches_rule(entity, target_id, distribution_type, rule)
                    applied_rules.append({
                        'rule_id': rule_id,
                        'matches': matches_rule
                    })
                    
                    if matches_rule and rule['action'] == 'reject':
                        allowed = False
                        rejected_reason = rule.get('reason', 'Violates boundary rule')
                        break
        
        if not allowed:
            self.boundary_metrics['rejected_distributions'] += 1
            
        return allowed, {
            'allowed': allowed,
            'reason': rejected_reason,
            'applied_rules': applied_rules,
            'elapsed_time': time.time() - start_time
        }
        
    async def add_boundary_rule(self, rule):
        if rule_id in self.boundary_rules:
            del self.boundary_rules[rule_id]
            return {
                'success': True,
                'rule_id': rule_id
            }
        return {
            'success': False,
            'error': 'Rule not found'
        }
        
    async def _predict_future_interaction(self, entity_id, current_interaction, interaction_type):
        if not entity_id:
            return
            
        history = self.interaction_history.get(entity_id, [])
        if not history:
            return
            
        interaction_types = [h['interaction_type'] for h in history if h['interaction_type']]
        type_counts = {}
        for t in interaction_types:
            type_counts[t] = type_counts.get(t, 0) + 1
            
        if len(history) >= 2:
            intervals = []
            for i in range(1, len(history)):
                interval = history[i]['timestamp'] - history[i-1]['timestamp']
                intervals.append(interval)
                
            avg_interval = np.mean(intervals) if intervals else 60.0  # Default to 60 seconds
        else:
            avg_interval = 60.0  # Default to 60 seconds
            
        if interaction_types:
            most_common = max(type_counts.items(), key=lambda x: x[1])[0]
            predicted_type = most_common
        else:
            predicted_type = interaction_type
            
        next_time = time.time() + avg_interval
        
        prediction = {
            'entity_id': entity_id,
            'predicted_type': predicted_type,
            'predicted_time': next_time,
            'prediction_time': time.time(),
            'avg_interval': avg_interval,
            'confidence': min(0.9, len(history) / 10)  # Confidence increases with more history
        }
        
        self.predicted_interactions[entity_id] = prediction
        
        return prediction
        
    def _compare_with_prediction(self, interaction_data, prediction):
        """
        Compare actual interaction with prediction.
        Returns True if the interaction matches prediction.
        """
        if not prediction:
            return False
            
        current_time = time.time()
        time_window = max(30.0, prediction['avg_interval'] * 0.5)  # Adjust window based on avg interval
        
        time_match = abs(current_time - prediction['predicted_time']) < time_window
        
        type_match = interaction_data.get('interaction_type') == prediction.get('predicted_type')
        
        return time_match and type_match
        
    def _update_precision(self, entity_id):
        """
        Update precision for an entity based on prediction accuracy.
        Implements Seth's precision-weighted errors principle.
        """
        accuracy_history = self.prediction_accuracy.get(entity_id, [])
        if len(accuracy_history) < 2:
            self.precision_weights[entity_id] = 1.0
            return
            
        accuracies = [entry['accuracy'] for entry in accuracy_history]
        
        variance = np.var(accuracies)
        
        precision = 1.0 / (variance + 0.1)  # Add small constant to avoid division by zero
        
        precision = max(0.1, min(10.0, precision))
        
        self.precision_weights[entity_id] = precision
        
    def _interaction_matches_rule(self, interaction_data, source_id, interaction_type, rule):
        """Check if interaction matches a boundary rule."""
        condition = rule['condition']
        
        # Simple rule matching
        if 'source_id' in condition and condition['source_id'] != source_id:
            return False
            
        if 'interaction_type' in condition and condition['interaction_type'] != interaction_type:
            return False
            
        if 'content_contains' in condition:
            # Check if any field contains the specified text
            found = False
            search_text = condition['content_contains'].lower()
            
            if isinstance(interaction_data, dict):
                for key, value in interaction_data.items():
                    if isinstance(value, str) and search_text in value.lower():
                        found = True
                        break
                    elif isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            if isinstance(subvalue, str) and search_text in subvalue.lower():
                                found = True
                                break
                        if found:
                            break
                            
            if not found:
                return False
                
        # More complex rule conditions would go here
        
        return True
        
    def _distribution_matches_rule(self, entity, target_id, distribution_type, rule):
        """Check if distribution matches a boundary rule."""
        condition = rule['condition']
        
        if 'target_id' in condition and condition['target_id'] != target_id:
            return False
            
        if 'distribution_type' in condition and condition['distribution_type'] != distribution_type:
            return False
            
        if 'entity_type' in condition:
            entity_type = getattr(entity, 'type', None)
            if condition['entity_type'] != entity_type:
                return False
                
        
        return True
        
    async def get_metrics(self):