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
        
        # Seth's Data Paradox enhancements
        self.predicted_interactions = {}  # Maps entity_id to predicted interactions
        self.prediction_accuracy = defaultdict(list)  # Maps entity_id to prediction accuracy history
        self.precision_weights = {}  # Maps entity_id to prediction precision
        
        self.logger = logging.getLogger("ASF.Layer4.CoherenceBoundaryController")
        
    async def check_interaction_coherence(self, interaction_data, source_id, interaction_type, context):
        """
        Check if an incoming interaction is coherent with system boundaries.
        Now enhanced with predictive filtering based on Seth's principles.
        """
        start_time = time.time()
        self.boundary_metrics['total_interactions'] += 1
        
        # Compare with predicted interactions if available
        prediction_match = False
        if source_id in self.predicted_interactions:
            predicted = self.predicted_interactions[source_id]
            prediction_match = self._compare_with_prediction(interaction_data, predicted)
            
            # Update prediction accuracy
            if predicted.get('prediction_time'):
                accuracy = 1.0 if prediction_match else 0.0
                self.prediction_accuracy[source_id].append({
                    'timestamp': time.time(),
                    'accuracy': accuracy,
                    'prediction_time': predicted['prediction_time']
                })
                
                # Limit history size
                if len(self.prediction_accuracy[source_id]) > 20:
                    self.prediction_accuracy[source_id] = self.prediction_accuracy[source_id][-20:]
                    
                # Update precision
                self._update_precision(source_id)
        
        # Check against boundary rules
        allowed = True
        rejected_reason = None
        applied_rules = []
        
        # Apply each rule in order
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
                    
        # Record interaction in history
        if source_id:
            self.interaction_history[source_id].append({
                'timestamp': time.time(),
                'interaction_type': interaction_type,
                'allowed': allowed,
                'prediction_match': prediction_match
            })
            
            # Limit history size
            if len(self.interaction_history[source_id]) > 100:
                self.interaction_history[source_id] = self.interaction_history[source_id][-100:]
        
        # Generate new prediction for future interactions
        await self._predict_future_interaction(source_id, interaction_data, interaction_type)
        
        # Update metrics
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
        """
        Check if an outgoing distribution is coherent with system boundaries.
        """
        start_time = time.time()
        self.boundary_metrics['total_distributions'] += 1
        
        # Check against boundary rules
        allowed = True
        rejected_reason = None
        applied_rules = []
        
        # Get entity information
        entity = await self.knowledge_substrate.get_entity(entity_id)
        if not entity:
            allowed = False
            rejected_reason = "Entity not found"
        else:
            # Apply distribution rules
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
        
        # Update metrics
        if not allowed:
            self.boundary_metrics['rejected_distributions'] += 1
            
        return allowed, {
            'allowed': allowed,
            'reason': rejected_reason,
            'applied_rules': applied_rules,
            'elapsed_time': time.time() - start_time
        }
        
    async def add_boundary_rule(self, rule):
        """Add a new boundary rule."""
        rule_id = rule.get('id', f"rule_{time.time()}")
        self.boundary_rules[rule_id] = {
            'enabled': rule.get('enabled', True),
            'condition': rule['condition'],
            'action': rule.get('action', 'reject'),
            'reason': rule.get('reason', 'Boundary rule violation'),
            'priority': rule.get('priority', 0),
            'applies_to_distribution': rule.get('applies_to_distribution', False)
        }
        
        return {
            'success': True,
            'rule_id': rule_id
        }
        
    async def remove_boundary_rule(self, rule_id):
        """Remove a boundary rule."""
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
        """
        Predict future interactions from an entity.
        Implements Seth's controlled hallucination principle.
        """
        if not entity_id:
            return
            
        # Get interaction history
        history = self.interaction_history.get(entity_id, [])
        if not history:
            return
            
        # Analyze interaction patterns
        interaction_types = [h['interaction_type'] for h in history if h['interaction_type']]
        type_counts = {}
        for t in interaction_types:
            type_counts[t] = type_counts.get(t, 0) + 1
            
        # Calculate time intervals between interactions
        if len(history) >= 2:
            intervals = []
            for i in range(1, len(history)):
                interval = history[i]['timestamp'] - history[i-1]['timestamp']
                intervals.append(interval)
                
            avg_interval = np.mean(intervals) if intervals else 60.0  # Default to 60 seconds
        else:
            avg_interval = 60.0  # Default to 60 seconds
            
        # Predict next interaction type
        if interaction_types:
            # Most common type
            most_common = max(type_counts.items(), key=lambda x: x[1])[0]
            predicted_type = most_common
        else:
            predicted_type = interaction_type
            
        # Predict when it will happen
        next_time = time.time() + avg_interval
        
        # Create prediction
        prediction = {
            'entity_id': entity_id,
            'predicted_type': predicted_type,
            'predicted_time': next_time,
            'prediction_time': time.time(),
            'avg_interval': avg_interval,
            'confidence': min(0.9, len(history) / 10)  # Confidence increases with more history
        }
        
        # Store prediction
        self.predicted_interactions[entity_id] = prediction
        
        return prediction
        
    def _compare_with_prediction(self, interaction_data, prediction):
        """
        Compare actual interaction with prediction.
        Returns True if the interaction matches prediction.
        """
        if not prediction:
            return False
            
        # Check if interaction is within expected time
        current_time = time.time()
        time_window = max(30.0, prediction['avg_interval'] * 0.5)  # Adjust window based on avg interval
        
        time_match = abs(current_time - prediction['predicted_time']) < time_window
        
        # Check if interaction type matches
        type_match = interaction_data.get('interaction_type') == prediction.get('predicted_type')
        
        # Overall match score
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
            
        # Calculate precision as inverse variance of errors
        accuracies = [entry['accuracy'] for entry in accuracy_history]
        
        # Variance of accuracy (0 = perfect consistency, 1 = totally random)
        variance = np.var(accuracies)
        
        # Precision is inverse variance (higher = more reliable predictions)
        precision = 1.0 / (variance + 0.1)  # Add small constant to avoid division by zero
        
        # Limit to reasonable range
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
        
        # Simple rule matching
        if 'target_id' in condition and condition['target_id'] != target_id:
            return False
            
        if 'distribution_type' in condition and condition['distribution_type'] != distribution_type:
            return False
            
        if 'entity_type' in condition:
            entity_type = getattr(entity, 'type', None)
            if condition['entity_type'] != entity_type:
                return False
                
        # More complex rule conditions would go here
        
        return True
        
    async def get_metrics(self):
        """Get metrics about the coherence boundary."""
        # Calculate prediction accuracy
        all_accuracies = []
        for entity_id, history in self.prediction_accuracy.items():
            accuracies = [entry['accuracy'] for entry in history]
            if accuracies:
                avg_accuracy = np.mean(accuracies)
                all_accuracies.append(avg_accuracy)
                
        avg_prediction_accuracy = np.mean(all_accuracies) if all_accuracies else 0.0
        
        return {
            'total_interactions': self.boundary_metrics['total_interactions'],
            'rejected_interactions': self.boundary_metrics['rejected_interactions'],
            'total_distributions': self.boundary_metrics['total_distributions'],
            'rejected_distributions': self.boundary_metrics['rejected_distributions'],
            'rule_count': len(self.boundary_rules),
            'rejection_rate': self.boundary_metrics['rejected_interactions'] / max(1, self.boundary_metrics['total_interactions']),
            'entities_tracked': len(self.interaction_history),
            'avg_prediction_accuracy': avg_prediction_accuracy,
            'entities_with_predictions': len(self.predicted_interactions)
        }
