import asyncio
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict
import uuid

class InformationGainEstimator:
    """
    Estimates expected information gain from different inference actions.
    Helps prioritize actions that will most effectively reduce uncertainty.
    """
    def __init__(self):
        self.action_history = defaultdict(list)
        self.info_gain_cache = {}
        self.gain_models = {}
        
    async def estimate_information_gain(self, action_set, current_state, domain=None):
        gains = {}
        
        for action in action_set:
            action_id = self._get_action_id(action)
            action_type = action.get('type', 'unknown')
            
            cache_key = f"{action_id}_{self._hash_state(current_state)}"
            if cache_key in self.info_gain_cache:
                gains[action_id] = self.info_gain_cache[cache_key]
                continue
            
            if domain and domain in self.gain_models:
                gain = await self.gain_models[domain].estimate_gain(action, current_state)
            else:
                gain = await self._estimate_generic_gain(action, action_type, current_state)
                
            gains[action_id] = gain
            
            self.info_gain_cache[cache_key] = gain
            
        return gains
    
    async def update_from_results(self, action, prior_state, posterior_state, actual_gain):
        action_id = self._get_action_id(action)
        action_type = action.get('type', 'unknown')
        
        self.action_history[action_type].append({
            'action_id': action_id,
            'timestamp': time.time(),
            'prior_state_hash': self._hash_state(prior_state),
            'posterior_state_hash': self._hash_state(posterior_state),
            'actual_gain': actual_gain
        })
        
        if len(self.action_history[action_type]) > 50:
            self.action_history[action_type] = self.action_history[action_type][-50:]
            
        cache_keys_to_update = [k for k in self.info_gain_cache if k.startswith(f"{action_id}_")]
        for key in cache_keys_to_update:
            old_estimate = self.info_gain_cache[key]
            self.info_gain_cache[key] = 0.8 * old_estimate + 0.2 * actual_gain
    
    def register_domain_model(self, domain, model):
        """Register a domain-specific information gain model."""
        self.gain_models[domain] = model
    
    async def _estimate_generic_gain(self, action, action_type, current_state):
        """Estimate information gain using generic heuristics."""
        baseline_gain = 0.5
        
        history = self.action_history[action_type]
        if history:
            avg_gain = np.mean([h['actual_gain'] for h in history])
            baseline_gain = 0.3 * baseline_gain + 0.7 * avg_gain
            
        if action_type == 'question':
            baseline_gain *= 1.2
        elif action_type == 'verification':
            baseline_gain *= 1.0
        elif action_type == 'exploration':
            knowledge_factor = self._estimate_knowledge_coverage(action, current_state)
            baseline_gain *= (2.0 - knowledge_factor)
            
        specificity = self._estimate_action_specificity(action)
        baseline_gain *= (0.5 + 0.5 * specificity)
        
        return min(1.0, max(0.1, baseline_gain))
    
    def _estimate_knowledge_coverage(self, action, state):
        """Estimate how much we already know about the target of an action."""
        # Simple heuristic based on state size
        # In a real implementation, would be more sophisticated
        if not state:
            return 0.1
            
        if isinstance(state, dict):
            coverage = min(1.0, len(state) / 20.0)  # Normalize by expected size
        elif isinstance(state, (list, set)):
            coverage = min(1.0, len(state) / 50.0)
        else:
            coverage = 0.5  # Default
            
        return coverage
    
    def _estimate_action_specificity(self, action):
        """Estimate how specific/focused an action is."""
        if isinstance(action, dict):
            param_count = sum(1 for k in action.keys() if not k.startswith('_'))
            return min(1.0, param_count / 5.0)  # Normalize
        else:
            return 0.5  # Default
    
    def _get_action_id(self, action):
        """Get a stable identifier for an action."""
        if isinstance(action, dict) and 'id' in action:
            return action['id']
        elif isinstance(action, dict):
            # Create a stable hash of action properties
            action_items = sorted((str(k), str(v)) for k, v in action.items() 
                                 if not k.startswith('_'))
            return hash(tuple(action_items))
        else:
            return hash(str(action))
    
    def _hash_state(self, state):
        """Create a stable hash representation of a state."""
        if isinstance(state, dict):
            state_items = sorted((str(k), str(v)) for k, v in state.items())
            return hash(tuple(state_items))
        elif isinstance(state, (list, tuple, set)):
            return hash(tuple(sorted(str(x) for x in state)))
        else:
            return hash(str(state))


class CounterfactualScenarioGenerator:
    """
    Generates diverse counterfactual scenarios for active inference testing.
    Enables more thorough exploration of possible world states.
    """
    def __init__(self):
        self.scenario_templates = defaultdict(list)
        self.variation_strategies = {
            'numeric': self._generate_numeric_variations,
            'categorical': self._generate_categorical_variations,
            'structural': self._generate_structural_variations,
            'temporal': self._generate_temporal_variations
        }
        
    async def generate_counterfactuals(self, base_scenario, variation_types=None, count=3):
        counterfactuals = []
        
        if not variation_types:
            variation_types = self._detect_variation_types(base_scenario)
            
        for variation_type in variation_types:
            if variation_type in self.variation_strategies:
                variations = await self.variation_strategies[variation_type](
                    base_scenario, 
                    max(1, count // len(variation_types))
                )
                counterfactuals.extend(variations)
                
        if len(counterfactuals) < count:
            additional_needed = count - len(counterfactuals)
            
            for i in range(min(additional_needed, len(counterfactuals))):
                base_cf = counterfactuals[i]
                cf_types = [t for t in variation_types if t != base_cf.get('_variation_type')]
                
                if cf_types:
                    variation_type = cf_types[0]
                    variations = await self.variation_strategies[variation_type](
                        base_cf, 1
                    )
                    if variations:
                        variations[0]['_compound_variation'] = True
                        counterfactuals.append(variations[0])
        
        return counterfactuals[:count]
    
    async def learn_from_scenario(self, scenario, outcome, effectiveness):
        variation_type = scenario.get('_variation_type', 'unknown')
        
        if effectiveness > 0.7:
            template = self._extract_template(scenario)
            self.scenario_templates[variation_type].append({
                'template': template,
                'effectiveness': effectiveness,
                'timestamp': time.time()
            })
            
            if len(self.scenario_templates[variation_type]) > 20:
                self.scenario_templates[variation_type] = sorted(
                    self.scenario_templates[variation_type],
                    key=lambda x: x['effectiveness'],
                    reverse=True
                )[:20]
    
    def _detect_variation_types(self, scenario):
        """Automatically detect appropriate variation types for a scenario."""
        applicable_types = []
        
        if isinstance(scenario, dict):
            # Check for numeric values
            has_numeric = any(isinstance(v, (int, float)) for v in scenario.values())
            if has_numeric:
                applicable_types.append('numeric')
                
            # Check for categorical values
            has_categorical = any(isinstance(v, str) and not self._is_temporal_string(v) 
                                 for v in scenario.values())
            if has_categorical:
                applicable_types.append('categorical')
                
            # Check for temporal values
            has_temporal = any(self._is_temporal_string(v) if isinstance(v, str) else False
                              for v in scenario.values())
            has_temporal = has_temporal or any(k in ('time', 'date', 'timestamp') 
                                              for k in scenario.keys())
            if has_temporal:
                applicable_types.append('temporal')
                
            # Always consider structural variations for dictionaries
            applicable_types.append('structural')
            
        elif isinstance(scenario, (list, tuple)):
            # For lists, consider numeric if elements are numeric
            if all(isinstance(x, (int, float)) for x in scenario):
                applicable_types.append('numeric')
            
            # Always consider structural for lists
            applicable_types.append('structural')
            
        # Default if nothing detected
        if not applicable_types:
            applicable_types = ['structural']
            
        return applicable_types
    
    async def _generate_numeric_variations(self, base_scenario, count=1):
        """Generate variations by modifying numeric values."""
        variations = []
        
        for i in range(count):
            if isinstance(base_scenario, dict):
                variation = base_scenario.copy()
                
                numeric_fields = [(k, v) for k, v in variation.items() 
                                 if isinstance(v, (int, float))]
                
                if numeric_fields:
                    field, value = numeric_fields[np.random.choice(len(numeric_fields))]
                    
                    if abs(value) < 1e-6:  # Near zero
                        variation[field] = np.random.uniform(-1.0, 1.0)
                    else:
                        factor = np.random.uniform(0.2, 0.8)
                        direction = 1 if np.random.random() > 0.5 else -1
                        
                        variation[field] = value * (1 + direction * factor)
                        
                    variation['_variation_type'] = 'numeric'
                    variation['_modified_field'] = field
                    variations.append(variation)
                
            elif isinstance(base_scenario, (list, tuple)) and all(isinstance(x, (int, float)) for x in base_scenario):
                variation = list(base_scenario)
                
                num_to_modify = max(1, int(len(variation) * 0.3))
                indices = np.random.choice(len(variation), num_to_modify, replace=False)
                
                for idx in indices:
                    value = variation[idx]
                    if abs(value) < 1e-6:  # Near zero
                        variation[idx] = np.random.uniform(-1.0, 1.0)
                    else:
                        factor = np.random.uniform(0.2, 0.8)
                        direction = 1 if np.random.random() > 0.5 else -1
                        variation[idx] = value * (1 + direction * factor)
                
                variation = {
                    'values': variation,
                    '_original_type': 'list',
                    '_variation_type': 'numeric',
                    '_modified_indices': indices.tolist()
                }
                variations.append(variation)
        
        return variations
    
    async def _generate_categorical_variations(self, base_scenario, count=1):
        variations = []
        
        for i in range(count):
            if isinstance(base_scenario, dict):
                variation = base_scenario.copy()
                
                strategy = np.random.choice(['add', 'remove', 'rename'])
                
                if strategy == 'add' or len(variation) < 3:
                    new_field = f"additional_property_{uuid.uuid4().hex[:6]}"
                    if any(isinstance(v, (int, float)) for v in variation.values()):
                        variation[new_field] = np.random.uniform(0, 100)
                    elif any(isinstance(v, str) for v in variation.values()):
                        options = ['new value', 'additional data', 'supplementary information']
                        variation[new_field] = np.random.choice(options)
                    else:
                        variation[new_field] = True
                        
                    variation['_variation_type'] = 'structural'
                    variation['_structural_change'] = 'add'
                    variation['_modified_field'] = new_field
                    
                elif strategy == 'remove' and len(variation) > 2:
                    removable_fields = [k for k in variation.keys() 
                                       if not k.startswith('_')]
                    
                    if removable_fields:
                        field_to_remove = np.random.choice(removable_fields)
                        removed_value = variation.pop(field_to_remove)
                        
                        variation['_variation_type'] = 'structural'
                        variation['_structural_change'] = 'remove'
                        variation['_removed_field'] = field_to_remove
                        variation['_removed_value'] = str(removed_value)
                
                elif strategy == 'rename':
                    renamable_fields = [k for k in variation.keys() 
                                       if not k.startswith('_')]
                    
                    if renamable_fields:
                        field_to_rename = np.random.choice(renamable_fields)
                        new_field_name = f"{field_to_rename}_renamed_{uuid.uuid4().hex[:4]}"
                        
                        variation[new_field_name] = variation.pop(field_to_rename)
                        
                        variation['_variation_type'] = 'structural'
                        variation['_structural_change'] = 'rename'
                        variation['_original_field'] = field_to_rename
                        variation['_new_field'] = new_field_name
                
                variations.append(variation)
                
            elif isinstance(base_scenario, (list, tuple)):
                variation = list(base_scenario)
                
                strategy = np.random.choice(['add', 'remove', 'reorder'])
                
                if strategy == 'add' or len(variation) < 3:
                    if all(isinstance(x, (int, float)) for x in variation):
                        min_val = min(variation) if variation else 0
                        max_val = max(variation) if variation else 100
                        new_value = np.random.uniform(min_val, max_val)
                    elif all(isinstance(x, str) for x in variation):
                        new_value = f"new_element_{uuid.uuid4().hex[:6]}"
                    else:
                        existing_types = set(type(x) for x in variation)
                        if int in existing_types:
                            new_value = np.random.randint(0, 100)
                        elif float in existing_types:
                            new_value = np.random.uniform(0, 100)
                        elif str in existing_types:
                            new_value = f"new_element_{uuid.uuid4().hex[:6]}"
                        else:
                            new_value = True
                    
                    insert_pos = np.random.randint(0, len(variation) + 1)
                    variation.insert(insert_pos, new_value)
                    
                    variation_dict = {
                        'values': variation,
                        '_original_type': 'list',
                        '_variation_type': 'structural',
                        '_structural_change': 'add',
                        '_insert_position': insert_pos,
                        '_added_value': str(new_value)
                    }
                    variations.append(variation_dict)
                    
                elif strategy == 'remove' and len(variation) > 2:
                    remove_pos = np.random.randint(0, len(variation))
                    removed_value = variation.pop(remove_pos)
                    
                    variation_dict = {
                        'values': variation,
                        '_original_type': 'list',
                        '_variation_type': 'structural',
                        '_structural_change': 'remove',
                        '_remove_position': remove_pos,
                        '_removed_value': str(removed_value)
                    }
                    variations.append(variation_dict)
                    
                elif strategy == 'reorder':
                    if len(variation) > 2:
                        start = np.random.randint(0, len(variation) - 2)
                        end = np.random.randint(start + 2, len(variation) + 1)
                        sublist = variation[start:end]
                        
                        np.random.shuffle(sublist)
                        variation[start:end] = sublist
                        
                        variation_dict = {
                            'values': variation,
                            '_original_type': 'list',
                            '_variation_type': 'structural',
                            '_structural_change': 'reorder',
                            '_reorder_start': start,
                            '_reorder_end': end
                        }
                        variations.append(variation_dict)
        
        return variations
    
    async def _generate_temporal_variations(self, base_scenario, count=1):
        if isinstance(scenario, dict):
            template = {
                'type': 'dict',
                'modification': scenario.get('_structural_change', 'value_change'),
                'modified_field': scenario.get('_modified_field'),
                'variation_type': scenario.get('_variation_type')
            }
            return template
        elif isinstance(scenario, (list, tuple)) or (isinstance(scenario, dict) and scenario.get('_original_type') == 'list'):
            template = {
                'type': 'list',
                'modification': scenario.get('_structural_change', 'value_change'),
                'variation_type': scenario.get('_variation_type')
            }
            return template
        else:
            return {'type': 'unknown'}
    
    def _is_temporal_string(self, s):
        """Check if a string appears to represent a date/time."""
        if not isinstance(s, str):
            return False
            
        # Simple heuristic - check for date/time patterns
        temporal_indicators = [
            '/', '-', 'day', 'month', 'year', 'hour', 'minute', 'second',
            'am', 'pm', 'gmt', 'utc', 'est', 'pst', 'cet', 'jst', 'ist',
            'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
            'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
        ]
        
        s_lower = s.lower()
        # Check if string contains temporal indicators and has digits
        has_indicators = any(ind in s_lower for ind in temporal_indicators)
        has_digits = any(c.isdigit() for c in s)
        
        return has_indicators and has_digits


class AdaptiveActiveInferenceController:
    """
    Enhanced active inference controller with adaptive exploration strategies.
    Implements a more sophisticated approach to Seth's active inference principle.
        self.use_neural_model = True
        self.neural_model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
        
        self.optimizer = torch.optim.Adam(self.neural_model.parameters(), lr=0.001)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.neural_model.to(device)
        
        self.logger.info(f"Neural model initialized on {device}")
        
    async def anticipate_state(self, planned_operations, time_horizon=1.0, context=None):
        operation_id = f"ops_{int(time.time() * 1000)}"
        
        context_features = self._extract_context_features(context) if context else {}
        
        contradiction_result = await self.semantic_layer.conflict_detection.anticipate_contradictions(
            planned_operations
        )
        
        category_anticipations = await self._anticipate_category_changes(
            planned_operations, 
            context_features
        )
        
        concept_anticipations = await self._anticipate_concept_formations(
            planned_operations, 
            context_features
        )
        
        anticipated_state = {
            'operation_id': operation_id,
            'timestamp': time.time(),
            'time_horizon': time_horizon,
            'anticipated_contradictions': contradiction_result.get('anticipated_contradictions', []),
            'anticipated_categories': category_anticipations,
            'anticipated_concepts': concept_anticipations,
            'planned_operations': planned_operations,
            'context_features': context_features,
            'uncertainty_metrics': await self._calculate_state_uncertainty(
                contradiction_result.get('anticipated_contradictions', []),
                category_anticipations,
                concept_anticipations
            )
        }
        
        self.anticipated_states[operation_id] = anticipated_state
        
        return anticipated_state
        
    async def perform_active_inference(self, anticipated_state, optimization_targets=None, 
                                       explore_counterfactuals=True):
        if not anticipated_state:
            return {'status': 'error', 'message': 'No anticipated state provided'}
            
        targets = optimization_targets or self._get_current_optimization_targets()
        
        planned_operations = anticipated_state.get('planned_operations', [])
        context_features = anticipated_state.get('context_features', {})
        
        operation_variants = await self._generate_operation_variants(
            planned_operations, 
            context_features
        )
        
        if explore_counterfactuals:
            counterfactual_variants = await self._generate_counterfactual_variants(
                planned_operations,
                anticipated_state
            )
            operation_variants.update(counterfactual_variants)
        
        variant_scores = []
        
        for variant_name, variant_ops in operation_variants.items():
            variant_state = await self.anticipate_state(variant_ops, time_horizon=0.5)
            
            score = await self._score_anticipated_state(variant_state, targets, context_features)
            
            info_gain = await self.information_gain_estimator.estimate_information_gain(
                variant_ops, anticipated_state, context_features.get('domain')
            )
            
            avg_gain = np.mean(list(info_gain.values())) if info_gain else 0.5
            combined_score = score * 0.7 + avg_gain * 0.3
            
            variant_scores.append((variant_name, variant_ops, combined_score, score, avg_gain))
        
        variant_scores.sort(key=lambda x: x[2], reverse=True)
        
        if not variant_scores:
            return {'status': 'error', 'message': 'No viable operation variants found'}
            
        best_name, best_ops, best_combined, best_score, best_gain = variant_scores[0]
        
        original_state = await self.anticipate_state(planned_operations, time_horizon=0.5)
        original_score = await self._score_anticipated_state(original_state, targets, context_features)
        
        self.action_history.append({
            'timestamp': time.time(),
            'original_operation_count': len(planned_operations),
            'selected_variant': best_name,
            'score_improvement': best_score - original_score,
            'info_gain': best_gain,
            'variant_count': len(variant_scores),
            'context_features': context_features
        })
        
        await self._update_learning_parameters(
            best_score - original_score,
            best_gain,
            best_name,
            context_features
        )
        
        return {
            'status': 'success',
            'original_operations': planned_operations,
            'optimized_operations': best_ops,
            'variant_name': best_name,
            'score': best_score,
            'information_gain': best_gain,
            'combined_score': best_combined,
            'improvement': best_score - original_score,
            'all_variants': len(variant_scores),
            'top_variants': [
                {'name': name, 'score': score, 'info_gain': gain} 
                for name, _, _, score, gain in variant_scores[:3]
            ]
        }
        
    async def evaluate_anticipations(self, actual_state, operation_id):
        if operation_id not in self.anticipated_states:
            return {'status': 'error', 'message': 'No such anticipated state'}
            
        anticipated = self.anticipated_states[operation_id]
        
        evaluation_metrics = {}
        
        if 'anticipated_contradictions' in anticipated and 'actual_contradictions' in actual_state:
            contradiction_metrics = await self._evaluate_contradiction_predictions(
                anticipated['anticipated_contradictions'],
                actual_state['actual_contradictions']
            )
            evaluation_metrics['contradictions'] = contradiction_metrics
            
            for contradiction_type, error in contradiction_metrics.get('errors_by_type', {}).items():
                error_key = f"contradictions_{contradiction_type}"
                self.prediction_errors[error_key].append(error)
                
                if len(self.prediction_errors[error_key]) > 20:
                    self.prediction_errors[error_key] = self.prediction_errors[error_key][-20:]
                
                if len(self.prediction_errors[error_key]) > 1:
                    variance = np.var(self.prediction_errors[error_key])
                    precision = 1.0 / (variance + 1e-6)  # Avoid division by zero
                    self.precision_values[error_key] = precision
        
        if 'anticipated_categories' in anticipated and 'actual_categories' in actual_state:
            category_metrics = await self._evaluate_category_predictions(
                anticipated['anticipated_categories'],
                actual_state['actual_categories']
            )
            evaluation_metrics['categories'] = category_metrics
            
        if 'anticipated_concepts' in anticipated and 'actual_concepts' in actual_state:
            concept_metrics = await self._evaluate_concept_predictions(
                anticipated['anticipated_concepts'],
                actual_state['actual_concepts']
            )
            evaluation_metrics['concepts'] = concept_metrics
            
        error_scores = []
        precision_scores = []
        
        for category, metrics in evaluation_metrics.items():
            if 'overall_error' in metrics:
                error_scores.append(metrics['overall_error'])
            if 'precision' in metrics:
                precision_scores.append(metrics['precision'])
                
        overall_error = np.mean(error_scores) if error_scores else 0.0
        overall_precision = np.mean(precision_scores) if precision_scores else 1.0
        
        evaluation_summary = {
            'status': 'success',
            'operation_id': operation_id,
            'overall_error': overall_error,
            'overall_precision': overall_precision,
            'calibration_score': self._calculate_calibration_score(evaluation_metrics),
            'detailed_metrics': evaluation_metrics
        }
        
        await self._update_from_evaluation(evaluation_summary, anticipated)
            
        self._clean_old_anticipations()
        
        return evaluation_summary
    
    async def _anticipate_category_changes(self, operations, context_features=None):
        anticipated_concepts = []
        
        anticipated_concepts.append({
            'concept_id': f"concept_{uuid.uuid4().hex[:8]}",
            'label': "Anticipated Concept",
            'property_count': 7,
            'confidence': 0.75,
            'uncertainty': {
                'confidence_interval': (0.65, 0.85),
                'entropy': 0.3
            }
        })
        
        return anticipated_concepts
    
    async def _generate_operation_variants(self, operations, context_features=None):
        context_features = anticipated_state.get('context_features', {})
        
        base_scenario = {
            'operations': operations,
            'anticipated_contradictions': len(anticipated_state.get('anticipated_contradictions', [])),
            'domain': context_features.get('domain', 'general')
        }
        
        counterfactuals = await self.counterfactual_generator.generate_counterfactuals(
            base_scenario, count=2
        )
        
        variants = {}
        
        for i, cf in enumerate(counterfactuals):
            if isinstance(cf, dict) and 'operations' in cf:
                cf_ops = cf['operations']
            elif isinstance(cf, dict) and '_original_type' == 'list' and 'values' in cf:
                cf_ops = cf['values']
            else:
                cf_ops = operations.copy()  # Fallback
                
                if len(cf_ops) > 1:
                    idx1, idx2 = np.random.choice(len(cf_ops), 2, replace=False)
                    cf_ops[idx1], cf_ops[idx2] = cf_ops[idx2], cf_ops[idx1]
            
            variants[f"counterfactual_{i+1}"] = cf_ops
            
        return variants
    
    def _reorder_operations(self, operations, context_features=None):
        """Reorder operations to minimize conflicts."""
        # Enhanced version with context awareness
        domain = context_features.get('domain', 'general') if context_features else 'general'
        
        # Different reordering strategies based on domain
        if domain == 'knowledge_graph':
            # For knowledge graphs, create nodes first, then hierarchical relations, then properties
            node_ops = [op for op in operations if op.get('type') == 'create_node']
            hierarchy_ops = [op for op in operations if op.get('type') == 'create_relation' and 
                           op.get('relation_type') in ('is_a', 'part_of', 'subclass_of')]
            prop_ops = [op for op in operations if op.get('type') in ('add_property', 'update_property')]
            other_rel_ops = [op for op in operations if op.get('type') == 'create_relation' and
                           op.get('relation_type') not in ('is_a', 'part_of', 'subclass_of')]
            other_ops = [op for op in operations if op not in node_ops + hierarchy_ops + 
                        prop_ops + other_rel_ops]
            
            return node_ops + hierarchy_ops + prop_ops + other_rel_ops + other_ops
            
        else:
            # Default strategy - sort node creations first, then relations, then properties
            node_ops = [op for op in operations if op.get('type') == 'create_node']
            rel_ops = [op for op in operations if op.get('type') == 'create_relation']
            prop_ops = [op for op in operations if op.get('type') in ('add_property', 'update_property')]
            other_ops = [op for op in operations if op not in node_ops + rel_ops + prop_ops]
            
            return node_ops + rel_ops + prop_ops + other_ops
    
    def _prune_operations(self, operations, context_features=None):
        """Remove likely problematic operations."""
        domain = context_features.get('domain', 'general') if context_features else 'general'
        confidence_threshold = 0.3  # Default threshold
        
        if domain == 'medical':
            confidence_threshold = 0.5
        elif domain == 'experimental':
            confidence_threshold = 0.2
            
        pruned_ops = []
        
        for op in operations:
            if op.get('confidence', 0.5) <= confidence_threshold:
                continue
                
            if op.get('_prune', False):
                continue
                
            if domain == 'knowledge_graph':
                if op.get('type') == 'create_relation':
                    source_exists = any(other_op.get('node_id') == op.get('source_id') and 
                                      other_op.get('type') == 'create_node'
                                      for other_op in operations)
                    target_exists = any(other_op.get('node_id') == op.get('target_id') and 
                                      other_op.get('type') == 'create_node'
                                      for other_op in operations)
                    
                    if not (source_exists and target_exists):
                        continue
            
            pruned_ops.append(op)
            
        return pruned_ops
    
    async def _enhance_operations(self, operations, context_features=None):
        prioritized = operations.copy()
        
        def priority_key(op):
            confidence = op.get('confidence', 0.5)
            
            importance = 0.5  # Default
            op_type = op.get('type')
            
            if op_type == 'create_node':
                importance = 0.8  # Node creation is high priority
            elif op_type == 'create_relation':
                rel_type = op.get('relation_type')
                if rel_type in ('is_a', 'part_of', 'subclass_of'):
                    importance = 0.7  # Hierarchical relations are higher priority
                else:
                    importance = 0.6  # Other relations
            elif op_type in ('add_property', 'update_property'):
                importance = 0.5  # Properties are medium priority
                
            return confidence * 0.7 + importance * 0.3
        
        prioritized.sort(key=priority_key, reverse=True)
        
        return prioritized
    
    async def _generate_neural_variant(self, operations, context_features=None):
        score = 0.0
        
        contradiction_count = len(state.get('anticipated_contradictions', []))
        contradiction_score = 1.0 / (1.0 + contradiction_count)
        score += contradiction_score * targets.get('contradiction_reduction', 0.4)
        
        category_score = 0.7  # Default
        if context_features and 'domain' in context_features:
            domain = context_features['domain']
            if domain == 'knowledge_graph':
                categories = state.get('anticipated_categories', [])
                category_score = 0.5