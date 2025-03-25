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
        """
        Estimate expected information gain for a set of potential actions.
        
        Args:
            action_set: Set of potential actions to evaluate
            current_state: Current knowledge state
            domain: Optional domain context
            
        Returns:
            Dictionary mapping actions to expected information gain
        """
        gains = {}
        
        for action in action_set:
            action_id = self._get_action_id(action)
            action_type = action.get('type', 'unknown')
            
            # Check cache first
            cache_key = f"{action_id}_{self._hash_state(current_state)}"
            if cache_key in self.info_gain_cache:
                gains[action_id] = self.info_gain_cache[cache_key]
                continue
            
            # Get domain-specific model if available
            if domain and domain in self.gain_models:
                gain = await self.gain_models[domain].estimate_gain(action, current_state)
            else:
                # General estimation based on action type and history
                gain = await self._estimate_generic_gain(action, action_type, current_state)
                
            gains[action_id] = gain
            
            # Cache the result
            self.info_gain_cache[cache_key] = gain
            
        return gains
    
    async def update_from_results(self, action, prior_state, posterior_state, actual_gain):
        """
        Update estimator based on actual information gain from an action.
        
        Args:
            action: The action that was taken
            prior_state: State before the action
            posterior_state: State after the action
            actual_gain: Measured information gain
        """
        action_id = self._get_action_id(action)
        action_type = action.get('type', 'unknown')
        
        # Record in history
        self.action_history[action_type].append({
            'action_id': action_id,
            'timestamp': time.time(),
            'prior_state_hash': self._hash_state(prior_state),
            'posterior_state_hash': self._hash_state(posterior_state),
            'actual_gain': actual_gain
        })
        
        # Limit history size
        if len(self.action_history[action_type]) > 50:
            self.action_history[action_type] = self.action_history[action_type][-50:]
            
        # Update cache entries that match this action
        cache_keys_to_update = [k for k in self.info_gain_cache if k.startswith(f"{action_id}_")]
        for key in cache_keys_to_update:
            # Apply weighted update
            old_estimate = self.info_gain_cache[key]
            # Exponential moving average update
            self.info_gain_cache[key] = 0.8 * old_estimate + 0.2 * actual_gain
    
    def register_domain_model(self, domain, model):
        """Register a domain-specific information gain model."""
        self.gain_models[domain] = model
    
    async def _estimate_generic_gain(self, action, action_type, current_state):
        """Estimate information gain using generic heuristics."""
        # Start with a baseline estimate
        baseline_gain = 0.5
        
        # Adjust based on action history for this type
        history = self.action_history[action_type]
        if history:
            # Calculate average gain from past actions of this type
            avg_gain = np.mean([h['actual_gain'] for h in history])
            baseline_gain = 0.3 * baseline_gain + 0.7 * avg_gain
            
        # Apply heuristic adjustments based on action type
        if action_type == 'question':
            # Questions generally provide good information gain
            baseline_gain *= 1.2
        elif action_type == 'verification':
            # Verification usually provides moderate gain
            baseline_gain *= 1.0
        elif action_type == 'exploration':
            # Exploration has variable gain - higher when we know less
            knowledge_factor = self._estimate_knowledge_coverage(action, current_state)
            # Less knowledge = higher potential gain from exploration
            baseline_gain *= (2.0 - knowledge_factor)
            
        # Consider action specificity - more specific actions tend to give more gain
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
        # More parameters usually means more specific action
        if isinstance(action, dict):
            # Count non-metadata parameters
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
            # Hash of sorted items
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
        """
        Generate counterfactual scenarios based on a base scenario.
        
        Args:
            base_scenario: Base scenario to generate variations from
            variation_types: Types of variations to apply (or None for all)
            count: Number of counterfactuals to generate
            
        Returns:
            List of counterfactual scenarios
        """
        counterfactuals = []
        
        # Determine applicable variation types
        if not variation_types:
            # Auto-detect appropriate variation types
            variation_types = self._detect_variation_types(base_scenario)
            
        # Apply each variation type
        for variation_type in variation_types:
            if variation_type in self.variation_strategies:
                # Generate variations using this strategy
                variations = await self.variation_strategies[variation_type](
                    base_scenario, 
                    max(1, count // len(variation_types))
                )
                counterfactuals.extend(variations)
                
        # If we didn't generate enough, duplicate some variations with additional changes
        if len(counterfactuals) < count:
            additional_needed = count - len(counterfactuals)
            
            for i in range(min(additional_needed, len(counterfactuals))):
                # Take an existing counterfactual and modify it further
                base_cf = counterfactuals[i]
                # Choose a different variation type than the one used to create this counterfactual
                cf_types = [t for t in variation_types if t != base_cf.get('_variation_type')]
                
                if cf_types:
                    variation_type = cf_types[0]
                    variations = await self.variation_strategies[variation_type](
                        base_cf, 1
                    )
                    if variations:
                        # Add metadata about compound variation
                        variations[0]['_compound_variation'] = True
                        counterfactuals.append(variations[0])
        
        # Ensure we return exactly the requested count
        return counterfactuals[:count]
    
    async def learn_from_scenario(self, scenario, outcome, effectiveness):
        """
        Learn from a counterfactual scenario and its actual outcome.
        Improves future counterfactual generation.
        
        Args:
            scenario: The counterfactual scenario
            outcome: The actual outcome that occurred
            effectiveness: How effective the counterfactual was (0-1)
        """
        variation_type = scenario.get('_variation_type', 'unknown')
        
        # Record this scenario as a template if it was effective
        if effectiveness > 0.7:
            template = self._extract_template(scenario)
            self.scenario_templates[variation_type].append({
                'template': template,
                'effectiveness': effectiveness,
                'timestamp': time.time()
            })
            
            # Limit template history
            if len(self.scenario_templates[variation_type]) > 20:
                # Sort by effectiveness and keep the best ones
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
                
                # Find numeric fields to modify
                numeric_fields = [(k, v) for k, v in variation.items() 
                                 if isinstance(v, (int, float))]
                
                if numeric_fields:
                    # Select a random field to modify
                    field, value = numeric_fields[np.random.choice(len(numeric_fields))]
                    
                    # Determine magnitude of change (proportional to value)
                    if abs(value) < 1e-6:  # Near zero
                        variation[field] = np.random.uniform(-1.0, 1.0)
                    else:
                        # Change by 20-80%
                        factor = np.random.uniform(0.2, 0.8)
                        direction = 1 if np.random.random() > 0.5 else -1
                        
                        variation[field] = value * (1 + direction * factor)
                        
                    # Add metadata
                    variation['_variation_type'] = 'numeric'
                    variation['_modified_field'] = field
                    variations.append(variation)
                
            elif isinstance(base_scenario, (list, tuple)) and all(isinstance(x, (int, float)) for x in base_scenario):
                # For numeric lists, modify random elements
                variation = list(base_scenario)
                
                # Select random indices to modify (up to 30% of elements)
                num_to_modify = max(1, int(len(variation) * 0.3))
                indices = np.random.choice(len(variation), num_to_modify, replace=False)
                
                for idx in indices:
                    value = variation[idx]
                    if abs(value) < 1e-6:  # Near zero
                        variation[idx] = np.random.uniform(-1.0, 1.0)
                    else:
                        # Change by 20-80%
                        factor = np.random.uniform(0.2, 0.8)
                        direction = 1 if np.random.random() > 0.5 else -1
                        variation[idx] = value * (1 + direction * factor)
                
                # Add metadata
                variation = {
                    'values': variation,
                    '_original_type': 'list',
                    '_variation_type': 'numeric',
                    '_modified_indices': indices.tolist()
                }
                variations.append(variation)
        
        return variations
    
    async def _generate_categorical_variations(self, base_scenario, count=1):
        """Generate variations by changing categorical values."""
        variations = []
        
        # Define common alternatives for frequent categorical values
        alternatives = {
            'high': ['medium', 'low', 'very high', 'above average'],
            'medium': ['high', 'low', 'average', 'moderate'],
            'low': ['medium', 'high', 'very low', 'below average'],
            'yes': ['no', 'maybe', 'partially', 'conditionally'],
            'no': ['yes', 'maybe', 'partially', 'uncertain'],
            'true': ['false', 'partially true', 'unknown', 'conditional'],
            'false': ['true', 'partially false', 'unknown', 'conditional']
        }
        
        for i in range(count):
            if isinstance(base_scenario, dict):
                variation = base_scenario.copy()
                
                # Find categorical fields
                categorical_fields = [(k, v) for k, v in variation.items() 
                                     if isinstance(v, str) and not self._is_temporal_string(v)]
                
                if categorical_fields:
                    # Select a random field to modify
                    field, value = categorical_fields[np.random.choice(len(categorical_fields))]
                    
                    # Generate alternative value
                    value_lower = value.lower()
                    if value_lower in alternatives:
                        new_value = np.random.choice(alternatives[value_lower])
                        # Preserve original capitalization
                        if value.isupper():
                            new_value = new_value.upper()
                        elif value[0].isupper():
                            new_value = new_value.capitalize()
                    else:
                        # For values without predefined alternatives,
                        # either prefix with "not" or append a modifier
                        modifiers = ['alternative', 'modified', 'different', 'variant']
                        if np.random.random() > 0.5 and not value_lower.startswith('not'):
                            new_value = f"not {value}"
                        else:
                            new_value = f"{value} ({np.random.choice(modifiers)})"
                    
                    variation[field] = new_value
                    
                    # Add metadata
                    variation['_variation_type'] = 'categorical'
                    variation['_modified_field'] = field
                    variations.append(variation)
        
        return variations
    
    async def _generate_structural_variations(self, base_scenario, count=1):
        """Generate variations by modifying the structure."""
        variations = []
        
        for i in range(count):
            if isinstance(base_scenario, dict):
                variation = base_scenario.copy()
                
                # Choose a structural variation strategy
                strategy = np.random.choice(['add', 'remove', 'rename'])
                
                if strategy == 'add' or len(variation) < 3:
                    # Add a new field
                    new_field = f"additional_property_{uuid.uuid4().hex[:6]}"
                    # Generate appropriate value type based on existing fields
                    if any(isinstance(v, (int, float)) for v in variation.values()):
                        variation[new_field] = np.random.uniform(0, 100)
                    elif any(isinstance(v, str) for v in variation.values()):
                        options = ['new value', 'additional data', 'supplementary information']
                        variation[new_field] = np.random.choice(options)
                    else:
                        variation[new_field] = True
                        
                    # Add metadata
                    variation['_variation_type'] = 'structural'
                    variation['_structural_change'] = 'add'
                    variation['_modified_field'] = new_field
                    
                elif strategy == 'remove' and len(variation) > 2:
                    # Remove a field (but preserve metadata fields)
                    removable_fields = [k for k in variation.keys() 
                                       if not k.startswith('_')]
                    
                    if removable_fields:
                        field_to_remove = np.random.choice(removable_fields)
                        removed_value = variation.pop(field_to_remove)
                        
                        # Add metadata
                        variation['_variation_type'] = 'structural'
                        variation['_structural_change'] = 'remove'
                        variation['_removed_field'] = field_to_remove
                        variation['_removed_value'] = str(removed_value)
                
                elif strategy == 'rename':
                    # Rename a field
                    renamable_fields = [k for k in variation.keys() 
                                       if not k.startswith('_')]
                    
                    if renamable_fields:
                        field_to_rename = np.random.choice(renamable_fields)
                        new_field_name = f"{field_to_rename}_renamed_{uuid.uuid4().hex[:4]}"
                        
                        # Rename field
                        variation[new_field_name] = variation.pop(field_to_rename)
                        
                        # Add metadata
                        variation['_variation_type'] = 'structural'
                        variation['_structural_change'] = 'rename'
                        variation['_original_field'] = field_to_rename
                        variation['_new_field'] = new_field_name
                
                variations.append(variation)
                
            elif isinstance(base_scenario, (list, tuple)):
                # Convert to list to ensure mutability
                variation = list(base_scenario)
                
                # Choose a structural variation strategy
                strategy = np.random.choice(['add', 'remove', 'reorder'])
                
                if strategy == 'add' or len(variation) < 3:
                    # Add a new element
                    if all(isinstance(x, (int, float)) for x in variation):
                        # For numeric lists, add a value in the same range
                        min_val = min(variation) if variation else 0
                        max_val = max(variation) if variation else 100
                        new_value = np.random.uniform(min_val, max_val)
                    elif all(isinstance(x, str) for x in variation):
                        # For string lists, add a similar string
                        new_value = f"new_element_{uuid.uuid4().hex[:6]}"
                    else:
                        # Mixed type list
                        existing_types = set(type(x) for x in variation)
                        if int in existing_types:
                            new_value = np.random.randint(0, 100)
                        elif float in existing_types:
                            new_value = np.random.uniform(0, 100)
                        elif str in existing_types:
                            new_value = f"new_element_{uuid.uuid4().hex[:6]}"
                        else:
                            new_value = True
                    
                    # Add to a random position
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
                    # Remove a random element
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
                    # Shuffle a subset of elements
                    if len(variation) > 2:
                        # Choose a random subrange to shuffle
                        start = np.random.randint(0, len(variation) - 2)
                        end = np.random.randint(start + 2, len(variation) + 1)
                        sublist = variation[start:end]
                        
                        # Shuffle the sublist
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
        """Generate variations by modifying temporal aspects."""
        variations = []
        
        # Simple time shifting for demonstration
        # In a real implementation, would parse actual timestamps and modify them
        for i in range(count):
            if isinstance(base_scenario, dict):
                variation = base_scenario.copy()
                
                # Identify temporal fields
                temporal_fields = []
                for k, v in variation.items():
                    if k in ('time', 'date', 'timestamp', 'created_at', 'updated_at', 'occurs_at'):
                        temporal_fields.append(k)
                    elif isinstance(v, str) and self._is_temporal_string(v):
                        temporal_fields.append(k)
                
                if temporal_fields:
                    # Select a random temporal field
                    field = np.random.choice(temporal_fields)
                    
                    # Apply a temporal shift
                    # This is simplified - real implementation would parse and properly modify dates
                    current_value = variation[field]
                    if isinstance(current_value, str):
                        # Add a temporal shift indicator
                        shifts = [
                            " + 1 day", " - 2 days", " + 1 week", " - 3 hours", 
                            " + 30 minutes", " next month", " previous year"
                        ]
                        variation[field] = f"{current_value}{np.random.choice(shifts)}"
                    else:
                        # Treat as numeric timestamp if not string
                        # Add/subtract random time (simulated as seconds)
                        shift = np.random.randint(-86400 * 7, 86400 * 7)  # ±7 days
                        variation[field] = current_value + shift
                    
                    # Add metadata
                    variation['_variation_type'] = 'temporal'
                    variation['_modified_field'] = field
                    variations.append(variation)
        
        return variations
    
    def _extract_template(self, scenario):
        """Extract a reusable template from a scenario."""
        if isinstance(scenario, dict):
            # Extract pattern of modifications
            template = {
                'type': 'dict',
                'modification': scenario.get('_structural_change', 'value_change'),
                'modified_field': scenario.get('_modified_field'),
                'variation_type': scenario.get('_variation_type')
            }
            return template
        elif isinstance(scenario, (list, tuple)) or (isinstance(scenario, dict) and scenario.get('_original_type') == 'list'):
            # Extract pattern for list modifications
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
    """
    def __init__(self, semantic_layer):
        self.semantic_layer = semantic_layer
        self.logger = logging.getLogger("ASF.Layer3.AdaptiveActiveInference")
        
        # Enhanced components
        self.information_gain_estimator = InformationGainEstimator()
        self.counterfactual_generator = CounterfactualScenarioGenerator()
        
        # Configuration and state
        self.anticipated_states = {}
        self.prediction_errors = defaultdict(list)
        self.precision_values = {}
        self.action_history = []
        
        # Learning parameters
        self.learning_rate = 0.2
        self.exploration_rate = 0.3  # Probability of trying experimental strategies
        self.exploration_decay = 0.99  # Gradual reduction in exploration
        
        # Configure optimization targets with adaptive weights
        self.optimization_targets = {
            'contradiction_reduction': 0.4,
            'category_coherence': 0.3,
            'structural_efficiency': 0.3
        }
        
        # Neural network for predicting optimal inference strategies (optional)
        self.use_neural_model = False
        self.neural_model = None
        
    async def initialize_neural_model(self, input_dim=32, output_dim=8):
        """Initialize neural network for inference strategy prediction."""
        self.use_neural_model = True
        self.neural_model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.neural_model.parameters(), lr=0.001)
        
        # Move to available device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.neural_model.to(device)
        
        self.logger.info(f"Neural model initialized on {device}")
        
    async def anticipate_state(self, planned_operations, time_horizon=1.0, context=None):
        """
        Anticipate future semantic network state based on planned operations.
        Enhanced with contextual processing and uncertainty estimation.
        
        Args:
            planned_operations: List of planned semantic operations
            time_horizon: Time window for anticipation (in seconds)
            context: Additional context information
            
        Returns:
            Anticipated semantic state
        """
        # Generate operation ID for tracking
        operation_id = f"ops_{int(time.time() * 1000)}"
        
        # Extract context features if provided
        context_features = self._extract_context_features(context) if context else {}
        
        # Enhanced anticipation process
        # Anticipate contradictions
        contradiction_result = await self.semantic_layer.conflict_detection.anticipate_contradictions(
            planned_operations
        )
        
        # Anticipate category formations/changes with uncertainty estimation
        category_anticipations = await self._anticipate_category_changes(
            planned_operations, 
            context_features
        )
        
        # Anticipate concept formations with confidence levels
        concept_anticipations = await self._anticipate_concept_formations(
            planned_operations, 
            context_features
        )
        
        # Create enhanced anticipated state with uncertainty metrics
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
        
        # Store for later evaluation
        self.anticipated_states[operation_id] = anticipated_state
        
        return anticipated_state
        
    async def perform_active_inference(self, anticipated_state, optimization_targets=None, 
                                       explore_counterfactuals=True):
        """
        Perform enhanced active inference to minimize prediction error.
        Modifies planned operations to optimize future state.
        
        Args:
            anticipated_state: Previously anticipated state
            optimization_targets: Optional custom optimization weights
            explore_counterfactuals: Whether to generate counterfactual scenarios
            
        Returns:
            Modified operations and expected improvements
        """
        if not anticipated_state:
            return {'status': 'error', 'message': 'No anticipated state provided'}
            
        # Use provided optimization targets or defaults with learning-adjusted weights
        targets = optimization_targets or self._get_current_optimization_targets()
        
        # Original operations
        planned_operations = anticipated_state.get('planned_operations', [])
        context_features = anticipated_state.get('context_features', {})
        
        # Generate operation variants to test, enhanced with counterfactuals
        operation_variants = await self._generate_operation_variants(
            planned_operations, 
            context_features
        )
        
        # If counterfactual exploration enabled, add counterfactual scenarios
        if explore_counterfactuals:
            counterfactual_variants = await self._generate_counterfactual_variants(
                planned_operations,
                anticipated_state
            )
            operation_variants.update(counterfactual_variants)
        
        # Evaluate each variant with information gain estimation
        variant_scores = []
        
        for variant_name, variant_ops in operation_variants.items():
            # Anticipate state with this variant
            variant_state = await self.anticipate_state(variant_ops, time_horizon=0.5)
            
            # Score the variant with enhanced scoring
            score = await self._score_anticipated_state(variant_state, targets, context_features)
            
            # Calculate expected information gain
            info_gain = await self.information_gain_estimator.estimate_information_gain(
                variant_ops, anticipated_state, context_features.get('domain')
            )
            
            # Combined score: weighted combination of anticipated state quality and info gain
            avg_gain = np.mean(list(info_gain.values())) if info_gain else 0.5
            combined_score = score * 0.7 + avg_gain * 0.3
            
            variant_scores.append((variant_name, variant_ops, combined_score, score, avg_gain))
        
        # Sort by combined score (higher is better)
        variant_scores.sort(key=lambda x: x[2], reverse=True)
        
        if not variant_scores:
            return {'status': 'error', 'message': 'No viable operation variants found'}
            
        # Select best variant
        best_name, best_ops, best_combined, best_score, best_gain = variant_scores[0]
        
        # Calculate improvements
        original_state = await self.anticipate_state(planned_operations, time_horizon=0.5)
        original_score = await self._score_anticipated_state(original_state, targets, context_features)
        
        # Record in history for learning
        self.action_history.append({
            'timestamp': time.time(),
            'original_operation_count': len(planned_operations),
            'selected_variant': best_name,
            'score_improvement': best_score - original_score,
            'info_gain': best_gain,
            'variant_count': len(variant_scores),
            'context_features': context_features
        })
        
        # Apply adaptive learning to update parameters based on this inference decision
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
        """
        Evaluate anticipation accuracy against actual state.
        Enhanced with more sophisticated error metrics and learning.
        
        Args:
            actual_state: Actual observed state
            operation_id: ID of the anticipated operation to evaluate
            
        Returns:
            Evaluation results with detailed metrics
        """
        if operation_id not in self.anticipated_states:
            return {'status': 'error', 'message': 'No such anticipated state'}
            
        anticipated = self.anticipated_states[operation_id]
        
        # Enhanced evaluation metrics
        evaluation_metrics = {}
        
        # Evaluate contradiction predictions
        if 'anticipated_contradictions' in anticipated and 'actual_contradictions' in actual_state:
            contradiction_metrics = await self._evaluate_contradiction_predictions(
                anticipated['anticipated_contradictions'],
                actual_state['actual_contradictions']
            )
            evaluation_metrics['contradictions'] = contradiction_metrics
            
            # Track error history with attention to contradiction types
            for contradiction_type, error in contradiction_metrics.get('errors_by_type', {}).items():
                error_key = f"contradictions_{contradiction_type}"
                self.prediction_errors[error_key].append(error)
                
                # Limit history size
                if len(self.prediction_errors[error_key]) > 20:
                    self.prediction_errors[error_key] = self.prediction_errors[error_key][-20:]
                
                # Update precision
                if len(self.prediction_errors[error_key]) > 1:
                    variance = np.var(self.prediction_errors[error_key])
                    precision = 1.0 / (variance + 1e-6)  # Avoid division by zero
                    self.precision_values[error_key] = precision
        
        # Similar enhanced evaluations for categories and concepts
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
            
        # Calculate overall prediction accuracy with advanced metrics
        error_scores = []
        precision_scores = []
        
        for category, metrics in evaluation_metrics.items():
            if 'overall_error' in metrics:
                error_scores.append(metrics['overall_error'])
            if 'precision' in metrics:
                precision_scores.append(metrics['precision'])
                
        overall_error = np.mean(error_scores) if error_scores else 0.0
        overall_precision = np.mean(precision_scores) if precision_scores else 1.0
        
        # Summarize metrics
        evaluation_summary = {
            'status': 'success',
            'operation_id': operation_id,
            'overall_error': overall_error,
            'overall_precision': overall_precision,
            'calibration_score': self._calculate_calibration_score(evaluation_metrics),
            'detailed_metrics': evaluation_metrics
        }
        
        # Update learning based on evaluation results
        await self._update_from_evaluation(evaluation_summary, anticipated)
            
        # Clean up old anticipations
        self._clean_old_anticipations()
        
        return evaluation_summary
    
    async def _anticipate_category_changes(self, operations, context_features=None):
        """Enhanced category change anticipation with confidence estimation."""
        # Placeholder implementation - in real system would make API calls to CategoryFormationSystem
        anticipated_categories = []
        
        # Add a placeholder category prediction with uncertainty metrics
        anticipated_categories.append({
            'category_id': f"cat_{uuid.uuid4().hex[:8]}",
            'label': "Anticipated Category",
            'member_count': 5,
            'confidence': 0.85,
            'uncertainty': {
                'confidence_interval': (0.75, 0.95),
                'entropy': 0.2
            }
        })
        
        return anticipated_categories
        
    async def _anticipate_concept_formations(self, operations, context_features=None):
        """Enhanced concept formation anticipation with uncertainty metrics."""
        # Placeholder implementation - in real system would make API calls to ConceptFormationEngine
        anticipated_concepts = []
        
        # Add a placeholder concept prediction with uncertainty metrics
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
        """Generate variants of operation sequences for testing."""
        # Enhanced version with context-aware generation
        variants = {
            'original': operations,
            'reordered': self._reorder_operations(operations, context_features),
            'pruned': self._prune_operations(operations, context_features),
            'enhanced': await self._enhance_operations(operations, context_features),
            'prioritized': self._prioritize_operations(operations, context_features)
        }
        
        # Add neural model variant if available
        if self.use_neural_model and self.neural_model is not None:
            neural_variant = await self._generate_neural_variant(operations, context_features)
            variants['neural'] = neural_variant
        
        return variants
    
    async def _generate_counterfactual_variants(self, operations, anticipated_state):
        """Generate counterfactual variants to explore alternative scenarios."""
        context_features = anticipated_state.get('context_features', {})
        
        # Create a base scenario from the operations
        base_scenario = {
            'operations': operations,
            'anticipated_contradictions': len(anticipated_state.get('anticipated_contradictions', [])),
            'domain': context_features.get('domain', 'general')
        }
        
        # Generate counterfactuals
        counterfactuals = await self.counterfactual_generator.generate_counterfactuals(
            base_scenario, count=2
        )
        
        # Convert each counterfactual to operations
        variants = {}
        
        for i, cf in enumerate(counterfactuals):
            # Extract operations from counterfactual
            if isinstance(cf, dict) and 'operations' in cf:
                cf_ops = cf['operations']
            elif isinstance(cf, dict) and '_original_type' == 'list' and 'values' in cf:
                cf_ops = cf['values']
            else:
                cf_ops = operations.copy()  # Fallback
                
                # Apply some generic modifications if no specific operations
                if len(cf_ops) > 1:
                    # Swap two operations
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
        # Enhanced version with context-aware pruning
        domain = context_features.get('domain', 'general') if context_features else 'general'
        confidence_threshold = 0.3  # Default threshold
        
        # Adjust threshold based on domain
        if domain == 'medical':
            # Higher threshold for medical domain due to safety concerns
            confidence_threshold = 0.5
        elif domain == 'experimental':
            # Lower threshold for experimental domains to allow exploration
            confidence_threshold = 0.2
            
        # Prune based on confidence and other factors
        pruned_ops = []
        
        for op in operations:
            # Skip if confidence is too low
            if op.get('confidence', 0.5) <= confidence_threshold:
                continue
                
            # Skip if explicitly marked for removal
            if op.get('_prune', False):
                continue
                
            # Domain-specific pruning logic
            if domain == 'knowledge_graph':
                # For knowledge graphs, ensure relation endpoints exist
                if op.get('type') == 'create_relation':
                    source_exists = any(other_op.get('node_id') == op.get('source_id') and 
                                      other_op.get('type') == 'create_node'
                                      for other_op in operations)
                    target_exists = any(other_op.get('node_id') == op.get('target_id') and 
                                      other_op.get('type') == 'create_node'
                                      for other_op in operations)
                    
                    # Only include relation if both endpoints exist
                    if not (source_exists and target_exists):
                        continue
            
            # Include operation in pruned set
            pruned_ops.append(op)
            
        return pruned_ops
    
    async def _enhance_operations(self, operations, context_features=None):
        """Add preventive or corrective operations."""
        # Enhanced version with context awareness
        domain = context_features.get('domain', 'general') if context_features else 'general'
        enhanced_ops = operations.copy()
        
        # Domain-specific enhancements
        if domain == 'knowledge_graph':
            # Ensure all relation endpoints have nodes
            relation_ops = [op for op in enhanced_ops if op.get('type') == 'create_relation']
            
            for rel_op in relation_ops:
                source_id = rel_op.get('source_id')
                target_id = rel_op.get('target_id')
                
                # Check if source node creation exists
                source_exists = any(op.get('type') == 'create_node' and op.get('node_id') == source_id
                                  for op in enhanced_ops)
                # Check if target node creation exists
                target_exists = any(op.get('type') == 'create_node' and op.get('node_id') == target_id
                                  for op in enhanced_ops)
                
                # Add missing node creations
                if not source_exists:
                    enhanced_ops.append({
                        'type': 'create_node',
                        'node_id': source_id,
                        'label': f"Auto-created source for {rel_op.get('relation_type', 'relation')}",
                        'confidence': 0.6,
                        '_auto_enhanced': True
                    })
                    
                if not target_exists:
                    enhanced_ops.append({
                        'type': 'create_node',
                        'node_id': target_id,
                        'label': f"Auto-created target for {rel_op.get('relation_type', 'relation')}",
                        'confidence': 0.6,
                        '_auto_enhanced': True
                    })
        
        # Add metadata to mark as enhanced
        for op in enhanced_ops:
            if '_auto_enhanced' not in op:
                op['_enhanced'] = True
                
        return enhanced_ops
    
    def _prioritize_operations(self, operations, context_features=None):
        """Prioritize operations based on importance and dependencies."""
        # Create a copy to avoid modifying original
        prioritized = operations.copy()
        
        # Sort based on combined factors of confidence and importance
        def priority_key(op):
            confidence = op.get('confidence', 0.5)
            
            # Assign importance based on operation type
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
                
            # Combine factors (confidence weighted more heavily)
            return confidence * 0.7 + importance * 0.3
        
        # Sort in descending order of priority
        prioritized.sort(key=priority_key, reverse=True)
        
        return prioritized
    
    async def _generate_neural_variant(self, operations, context_features=None):
        """Generate operation variant using neural network prediction."""
        # Convert operations and context to feature vector
        op_features = self._extract_operation_features(operations)
        ctx_features = self._extract_context_features(context_features) if context_features else {}
        
        # Combine features into input vector
        combined_features = np.zeros(32)  # Fixed size input vector
        
        # Fill in operation features (first half)
        for i, val in enumerate(op_features.values()):
            if i < 16:
                combined_features[i] = val
                
        # Fill in context features (second half)
        for i, val in enumerate(ctx_features.values()):
            if i < 16:
                combined_features[16 + i] = val
                
        # Convert to tensor
        input_tensor = torch.tensor(combined_features, dtype=torch.float32)
        input_tensor = input_tensor.to(next(self.neural_model.parameters()).device)
        
        # Generate prediction
        with torch.no_grad():
            output = self.neural_model(input_tensor)
            
        # Interpret output as operation modification parameters
        parameters = {
            'reorder_strength': float(output[0]),
            'prune_threshold': float(output[1]),
            'enhance_strength': float(output[2]),
            'prioritize_strength': float(output[3]),
            'confidence_adjustment': float(output[4]),
            'split_operations': output[5] > 0,
            'merge_operations': output[6] > 0,
            'add_context': output[7] > 0
        }
        
        # Apply neural network suggested modifications
        neural_variant = operations.copy()
        
        # Apply reordering if suggested
        if parameters['reorder_strength'] > 0.5:
            neural_variant = self._reorder_operations(neural_variant, context_features)
            
        # Apply pruning if suggested
        if parameters['prune_threshold'] > 0:
            # Adjust each operation's confidence based on neural suggestion
            for op in neural_variant:
                current_confidence = op.get('confidence', 0.5)
                confidence_adjustment = parameters['confidence_adjustment']
                op['confidence'] = max(0.1, min(0.9, current_confidence + confidence_adjustment))
            
            # Prune with neural threshold
            neural_variant = [op for op in neural_variant if op.get('confidence', 0.5) > parameters['prune_threshold']]
            
        # Apply enhancements if suggested
        if parameters['enhance_strength'] > 0.6:
            # Simple enhancement - add a metadata operation
            neural_variant.append({
                'type': 'metadata_update',
                'metadata_key': 'neural_enhanced',
                'metadata_value': True,
                'confidence': 0.8
            })
            
        # Apply other modifications based on neural network suggestions
        if parameters['split_operations']:
            # Example: Split a complex operation into simpler ones
            # Here we just mark operations as candidates for splitting
            for op in neural_variant:
                if 'properties' in op and isinstance(op['properties'], dict) and len(op['properties']) > 2:
                    op['_split_candidate'] = True
                    
        if parameters['merge_operations']:
            # Mark operations as candidates for merging
            node_ops = defaultdict(list)
            for i, op in enumerate(neural_variant):
                if op.get('type') in ('add_property', 'update_property') and 'node_id' in op:
                    node_ops[op['node_id']].append(i)
                    
            # Mark operations affecting the same node
            for node_id, indices in node_ops.items():
                if len(indices) > 1:
                    for idx in indices:
                        neural_variant[idx]['_merge_candidate'] = True
                        neural_variant[idx]['_merge_group'] = node_id
        
        return neural_variant
    
    async def _score_anticipated_state(self, state, targets, context_features=None):
        """Enhanced scoring of anticipated state with context awareness."""
        score = 0.0
        
        # Score contradiction reduction
        contradiction_count = len(state.get('anticipated_contradictions', []))
        # Lower is better, so we use an inverse score
        contradiction_score = 1.0 / (1.0 + contradiction_count)
        score += contradiction_score * targets.get('contradiction_reduction', 0.4)
        
        # Score category coherence with context awareness
        category_score = 0.7  # Default
        if context_features and 'domain' in context_features:
            # Adjust category score based on domain
            domain = context_features['domain']
            if domain == 'knowledge_graph':
                # For knowledge graphs, category coherence is more important
                categories = state.get('anticipated_categories', [])
                category_score = 0.5