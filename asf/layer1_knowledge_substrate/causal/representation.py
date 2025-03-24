import time
import numpy as np
from collections import defaultdict

from asf.layer1_knowledge_substrate.causal.graph import CausalGraph

class CausalRepresentationLearner:
    """
    Learns causal relationships between features and entities.
    Philosophical Influence: Pearl's causality, Woodward's interventionism, Seth's counterfactual models
    """
    def __init__(self):
        self.causal_graph = CausalGraph()
        self.observation_history = []
        self.intervention_outcomes = []
        self.correlation_matrix = None
        # For counterfactual simulation
        self.counterfactual_history = []
        self.prediction_errors = defaultdict(list)  # Track prediction accuracy
    
    def update_from_observations(self, entity_features):
        """
        Update causal model based on observed correlations between features
        
        Parameters:
        - entity_features: Dictionary mapping entity IDs to their feature dictionaries
        """
        # Track this observation
        self.observation_history.append({
            "entity_features": entity_features,
            "timestamp": time.time()
        })
        
        # Extract feature values across entities
        feature_values = defaultdict(list)
        feature_entities = defaultdict(list)
        
        for entity_id, features in entity_features.items():
            for feature_name, feature in features.items():
                # Only use numeric features for correlation analysis
                if (isinstance(feature.value, (int, float, np.number)) or 
                    (isinstance(feature.value, (list, np.ndarray)) and len(feature.value) == 1)):
                    # Convert to scalar if needed
                    value = feature.value[0] if isinstance(feature.value, (list, np.ndarray)) else feature.value
                    feature_values[feature_name].append(value)
                    feature_entities[feature_name].append(entity_id)
        
        # Need at least 2 observations per feature for correlation
        valid_features = [f for f, values in feature_values.items() if len(values) >= 2]
        if len(valid_features) < 2:
            return  # Not enough data for correlation
        
        # Compute correlation matrix
        correlation_data = np.array([feature_values[f] for f in valid_features])
        if correlation_data.shape[0] < 2:
            return
        
        try:
            correlation_matrix = np.corrcoef(correlation_data)
            # Update our correlation matrix
            self.correlation_matrix = {
                "features": valid_features,
                "matrix": correlation_matrix
            }
            # Update causal graph based on correlations
            self._update_causal_graph_from_correlations()
        except Exception as e:
            print(f"Error computing correlations: {e}")
    
    def _update_causal_graph_from_correlations(self):
        """
        Update causal graph based on correlation matrix
        Uses correlation strength as initial estimate of causal strength
        """
        if self.correlation_matrix is None:
            return
        
        features = self.correlation_matrix["features"]
        matrix = self.correlation_matrix["matrix"]
        
        # Add variables to causal graph if they don't exist
        for feature in features:
            if feature not in self.causal_graph.variables:
                self.causal_graph.add_variable(feature)
        
        # Look for strong correlations as potential causal links
        correlation_threshold = 0.7
        for i, feature1 in enumerate(features):
            for j, feature2 in enumerate(features):
                if i != j and abs(matrix[i, j]) > correlation_threshold:
                    # Correlation strength as initial causal strength estimate
                    causal_strength = abs(matrix[i, j])
                    
                    # For simplicity, assume the first feature in order is the cause
                    # In real implementation, would use temporal precedence, etc.
                    if i < j:
                        self.causal_graph.add_causal_link(feature1, feature2, causal_strength)
                    else:
                        self.causal_graph.add_causal_link(feature2, feature1, causal_strength)
    
    def perform_causal_intervention(self, entity_id, feature_name, new_value, entities):
        """
        Perform a causal intervention by changing a feature value
        Track the effects on other features to refine causal model
        
        Parameters:
        - entity_id: ID of entity to modify
        - feature_name: Name of feature to intervene on
        - new_value: New value to set
        - entities: Dictionary of all entities for tracking effects
        """
        # Record pre-intervention state
        pre_state = self._capture_entity_states(entities)
        
        # Perform intervention
        if entity_id in entities and feature_name in entities[entity_id].features:
            # Record intervention in causal graph
            variable_name = f"{entity_id}:{feature_name}"
            old_value = entities[entity_id].features[feature_name].value
            self.causal_graph.perform_intervention(variable_name, new_value)
            
            # Actually update the entity
            entities[entity_id].update_feature(feature_name, new_value)
            
            # Wait for effects (in real system, would wait for next processing cycle)
            # For demo, we'll just capture current state as "after"
            post_state = self._capture_entity_states(entities)
            
            # Record outcome
            self.intervention_outcomes.append({
                "variable": variable_name,
                "old_value": old_value,
                "new_value": new_value,
                "pre_state": pre_state,
                "post_state": post_state,
                "timestamp": time.time()
            })
            
            # Update causal model based on observed effects
            self._update_from_intervention(variable_name, pre_state, post_state)
            
            return True
        return False
    
    def _capture_entity_states(self, entities):
        """Capture current state of all entities and their features"""
        state = {}
        for entity_id, entity in entities.items():
            state[entity_id] = {
                "features": {
                    name: {
                        "value": feature.value,
                        "confidence": feature.confidence
                    }
                    for name, feature in entity.features.items()
                },
                "confidence_state": entity.confidence_state,
                "confidence_score": entity.confidence_score
            }
        return state
    
    def _update_from_intervention(self, intervention_var, pre_state, post_state):
        """
        Update causal model based on effects of an intervention
        
        Parameters:
        - intervention_var: Name of variable that was intervened on
        - pre_state: State before intervention
        - post_state: State after intervention
        """
        # Identify changes between pre and post states
        changes = self._identify_changes(pre_state, post_state)
        
        # Skip the intervened variable itself
        changes = [change for change in changes if change["variable"] != intervention_var]
        
        # Update causal links based on observed effects
        for change in changes:
            effect_var = change["variable"]
            
            # Get entity and feature from variable name
            if ":" in effect_var:
                entity_id, feature_name = effect_var.split(":", 1)
                
                # Strengthen causal link due to observed effect
                self.causal_graph.add_causal_link(intervention_var, effect_var, 0.8)
                
                # If the change was substantial, strengthen the link more
                if change["magnitude"] > 0.5:  # Arbitrary threshold
                    self.causal_graph.update_causal_strength(intervention_var, effect_var, 0.9)
    
    def _identify_changes(self, pre_state, post_state):
        """Identify changes between pre and post intervention states"""
        changes = []
        
        # Check all entities in pre-state
        for entity_id, pre_entity in pre_state.items():
            if entity_id in post_state:
                post_entity = post_state[entity_id]
                
                # Check features
                for feature_name, pre_feature in pre_entity["features"].items():
                    if feature_name in post_entity["features"]:
                        post_feature = post_entity["features"][feature_name]
                        
                        # Check if value changed
                        pre_value = pre_feature["value"]
                        post_value = post_feature["value"]
                        
                        # Handle different value types
                        if isinstance(pre_value, (int, float, np.number)) and isinstance(post_value, (int, float, np.number)):
                            # Numeric comparison
                            if pre_value != post_value:
                                # Calculate magnitude of change (normalized)
                                magnitude = abs(post_value - pre_value) / (1 + abs(pre_value))
                                
                                changes.append({
                                    "variable": f"{entity_id}:{feature_name}",
                                    "pre_value": pre_value,
                                    "post_value": post_value,
                                    "magnitude": magnitude
                                })
                        elif isinstance(pre_value, (list, np.ndarray)) and isinstance(post_value, (list, np.ndarray)):
                            # Vector comparison
                            pre_array = np.array(pre_value)
                            post_array = np.array(post_value)
                            
                            if pre_array.shape != post_array.shape:
                                # Shape changed, count as maximum change
                                changes.append({
                                    "variable": f"{entity_id}:{feature_name}",
                                    "pre_value": pre_value,
                                    "post_value": post_value,
                                    "magnitude": 1.0
                                })
                            elif not np.array_equal(pre_array, post_array):
                                # Calculate normalized distance
                                if pre_array.size > 0:
                                    diff = np.linalg.norm(post_array - pre_array) / np.linalg.norm(pre_array)
                                    changes.append({
                                        "variable": f"{entity_id}:{feature_name}",
                                        "pre_value": pre_value,
                                        "post_value": post_value,
                                        "magnitude": min(1.0, diff)
                                    })
                
                # Check confidence state
                if pre_entity["confidence_state"] != post_entity["confidence_state"]:
                    changes.append({
                        "variable": f"{entity_id}:confidence_state",
                        "pre_value": pre_entity["confidence_state"],
                        "post_value": post_entity["confidence_state"],
                        "magnitude": 1.0  # State change is significant
                    })
                
                # Check confidence score
                if abs(pre_entity["confidence_score"] - post_entity["confidence_score"]) > 0.05:
                    magnitude = abs(post_entity["confidence_score"] - pre_entity["confidence_score"])
                    changes.append({
                        "variable": f"{entity_id}:confidence_score",
                        "pre_value": pre_entity["confidence_score"],
                        "post_value": post_entity["confidence_score"],
                        "magnitude": magnitude
                    })
        
        return changes
    
    def generate_causal_explanation(self, entity_id, feature_name):
        """
        Generate natural language explanation of causal relationships
        
        Parameters:
        - entity_id: ID of entity to explain
        - feature_name: Name of feature to explain
        
        Returns a string explanation of causal influences on this feature
        """
        variable_name = f"{entity_id}:{feature_name}"
        
        if variable_name not in self.causal_graph.variables:
            return f"No causal information available for {feature_name}"
        
        # Get causal parents (causes of this feature)
        parents = self.causal_graph.get_causal_parents(variable_name)
        
        if not parents:
            return f"{feature_name} does not appear to be caused by other observed features"
        
        # Sort parents by causal strength
        sorted_parents = sorted(parents.items(), key=lambda x: x[1], reverse=True)
        
        # Generate explanation
        explanation = f"{feature_name} is primarily influenced by:\n"
        
        for parent_name, strength in sorted_parents:
            # Extract readable feature name from variable format
            if ":" in parent_name:
                _, parent_feature = parent_name.split(":", 1)
            else:
                parent_feature = parent_name
            
            # Describe causal strength
            if strength > 0.8:
                strength_desc = "strongly"
            elif strength > 0.5:
                strength_desc = "moderately"
            else:
                strength_desc = "weakly"
            
            explanation += f"- {parent_feature} ({strength_desc} influences, strength: {strength:.2f})\n"
        
        return explanation

    # SETH'S COUNTERFACTUAL ENHANCEMENT
    def generate_counterfactuals(self, entity_id, feature_name, entities):
        """
        Generate counterfactual scenarios for testing causal hypotheses.
        Returns possible alternative states and their predicted effects.
        
        This implements Seth's insight about counterfactual reasoning being
        essential to causal understanding.
        
        Parameters:
        - entity_id: ID of entity to modify hypothetically
        - feature_name: Feature to generate counterfactuals for
        - entities: Dictionary of all entities
        
        Returns:
        - List of counterfactual scenarios and predicted effects
        """
        if entity_id not in entities or feature_name not in entities[entity_id].features:
            return []
        
        # Get current value
        current_value = entities[entity_id].features[feature_name].value
        
        # Generate alternative values based on feature type
        if isinstance(current_value, (int, float, np.number)):
            alternatives = [
                current_value * 0.5,  # Half
                current_value * 1.5,  # 50% more
                -current_value,       # Opposite
                0.0                   # Zero/neutral
            ]
        elif isinstance(current_value, (list, np.ndarray)):
            current_array = np.array(current_value)
            alternatives = [
                np.zeros_like(current_array),                # Zero vector
                -current_array,                              # Opposite vector
                current_array * 2,                           # Doubled vector
                current_array * 0.5                          # Half vector
            ]
        else:
            # For non-numeric types, just create one counterfactual with None
            alternatives = [None]
        
        # Current entity state for reference
        base_state = self._capture_entity_states({entity_id: entities[entity_id]})
        
        # Generate counterfactuals for each alternative
        counterfactuals = []
        variable_name = f"{entity_id}:{feature_name}"
        
        for alt_value in alternatives:
            # Skip if same as current value
            if alt_value is not None and np.array_equal(alt_value, current_value):
                continue
                
            # Create deep copy of current state for simulation
            cf_state = self._deep_copy_state(base_state)
            
            # Apply counterfactual change
            if alt_value is not None:
                cf_state[entity_id]["features"][feature_name]["value"] = alt_value
            
            # Predict effects using causal graph
            predicted_effects = self._predict_intervention_effects(
                variable_name, alt_value, cf_state)
            
            # Add to counterfactuals
            counterfactuals.append({
                "alternative_value": alt_value,
                "predicted_effects": predicted_effects
            })
        
        # Record counterfactual generation
        self.counterfactual_history.append({
            "entity_id": entity_id,
            "feature_name": feature_name,
            "current_value": current_value,
            "counterfactuals_count": len(counterfactuals),
            "timestamp": time.time()
        })
        
        return counterfactuals
    
    def _predict_intervention_effects(self, variable_name, new_value, current_state):
        """
        Predict effects of an intervention using causal model
        
        Parameters:
        - variable_name: Name of variable to intervene on
        - new_value: New value to set
        - current_state: Current state of all entities
        
        Returns:
        - Dictionary of predicted effects
        """
        # Get children of intervened variable
        children = self.causal_graph.get_causal_children(variable_name)
        
        if not children:
            return {}  # No causal children, no effects
        
        # For each child, predict effect based on causal strength and change
        effects = {}
        
        for child_name, causal_strength in children.items():
            # Skip if child not in state
            if ":" not in child_name:
                continue
                
            child_entity_id, child_feature = child_name.split(":", 1)
            
            if (child_entity_id not in current_state or 
                child_feature not in current_state[child_entity_id]["features"]):
                continue
            
            # Get current child value
            child_value = current_state[child_entity_id]["features"][child_feature]["value"]
            
            # Predict new value based on causal model
            # This is a simplified model; actual prediction would depend on relationship type
            if isinstance(child_value, (int, float, np.number)) and isinstance(new_value, (int, float, np.number)):
                # Simple linear effect for numeric values
                # Effect proportional to causal strength and change magnitude
                change_magnitude = abs(new_value) / (1.0 + abs(new_value))
                effect_magnitude = causal_strength * change_magnitude
                
                # Direction depends on correlation sign (simplified)
                effect_direction = 1.0 if new_value > 0 else -1.0
                new_child_value = child_value + effect_direction * effect_magnitude * abs(child_value)
                
                effects[child_name] = {
                    "current_value": child_value,
                    "predicted_value": new_child_value,
                    "change_magnitude": abs(new_child_value - child_value) / (1.0 + abs(child_value)),
                    "causal_strength": causal_strength
                }
            elif isinstance(child_value, (list, np.ndarray)) and isinstance(new_value, (list, np.ndarray)):
                # For vector values, simplified prediction
                try:
                    child_array = np.array(child_value)
                    new_array = np.array(new_value)
                    
                    # Simple scaling based on causal strength
                    scaling_factor = causal_strength * 0.5  # Reduce effect magnitude
                    predicted_array = child_array + (new_array - child_array) * scaling_factor
                    
                    effects[child_name] = {
                        "current_value": child_value,
                        "predicted_value": predicted_array.tolist(),
                        "change_magnitude": np.linalg.norm(predicted_array - child_array) / np.linalg.norm(child_array),
                        "causal_strength": causal_strength
                    }
                except Exception as e:
                    # Skip if vectors can't be processed
                    continue
        
        return effects
    
    def _deep_copy_state(self, state):
        """Create a deep copy of entity state dictionary"""
        copied_state = {}
        
        for entity_id, entity_data in state.items():
            copied_entity = {
                "features": {},
                "confidence_state": entity_data["confidence_state"],
                "confidence_score": entity_data["confidence_score"]
            }
            
            # Copy features
            for feature_name, feature_data in entity_data["features"].items():
                # Handle different value types
                if isinstance(feature_data["value"], (list, np.ndarray)):
                    # Copy arrays/lists
                    copied_value = np.array(feature_data["value"]).tolist()
                else:
                    # Simple copy for scalars
                    copied_value = feature_data["value"]
                    
                copied_entity["features"][feature_name] = {
                    "value": copied_value,
                    "confidence": feature_data["confidence"]
                }
                
            copied_state[entity_id] = copied_entity
            
        return copied_state
    
    def validate_counterfactual(self, entity_id, feature_name, alternative_value, actual_effects, entities):
        """
        Validate a counterfactual prediction against actual observed effects
        
        Parameters:
        - entity_id: Entity that was modified
        - feature_name: Feature that was modified
        - alternative_value: Value that was set
        - actual_effects: Observed effects after intervention
        - entities: Current entity state
        
        Returns:
        - Dictionary with validation results
        """
        # Generate counterfactual prediction for this specific alternative
        counterfactuals = self.generate_counterfactuals(entity_id, feature_name, entities)
        
        # Find matching counterfactual
        matching_cf = None
        for cf in counterfactuals:
            if np.array_equal(cf["alternative_value"], alternative_value):
                matching_cf = cf
                break
                
        if not matching_cf:
            return {"valid": False, "error": "No matching counterfactual found"}
            
        # Compare predicted vs actual effects
        predicted_effects = matching_cf["predicted_effects"]
        validation_results = {}
        
        for effect_var, actual in actual_effects.items():
            if effect_var in predicted_effects:
                predicted = predicted_effects[effect_var]
                
                # Calculate prediction error
                if isinstance(actual["value"], (int, float, np.number)) and isinstance(predicted["predicted_value"], (int, float, np.number)):
                    error = abs(actual["value"] - predicted["predicted_value"]) / (1.0 + abs(actual["value"]))
                elif isinstance(actual["value"], (list, np.ndarray)) and isinstance(predicted["predicted_value"], (list, np.ndarray)):
                    actual_array = np.array(actual["value"])
                    predicted_array = np.array(predicted["predicted_value"])
                    if actual_array.shape == predicted_array.shape:
                        error = np.linalg.norm(actual_array - predicted_array) / np.linalg.norm(actual_array)
                    else:
                        error = 1.0  # Different shapes = maximum error
                else:
                    error = 1.0  # Different types = maximum error
                
                # Track prediction error for this variable
                self.prediction_errors[effect_var].append(error)
                
                # Limit history size
                if len(self.prediction_errors[effect_var]) > 10:
                    self.prediction_errors[effect_var] = self.prediction_errors[effect_var][-10:]
                
                validation_results[effect_var] = {
                    "predicted": predicted["predicted_value"],
                    "actual": actual["value"],
                    "error": error,
                    "accurate": error < 0.3  # Threshold for accuracy
                }
        
        # Calculate overall accuracy
        if validation_results:
            accuracy = sum(1 for r in validation_results.values() if r["accurate"]) / len(validation_results)
        else:
            accuracy = 0
            
        return {
            "valid": True,
            "accuracy": accuracy,
            "effect_validations": validation_results
        }
