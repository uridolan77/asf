import time
import logging
import asyncio
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field

# --- Assuming these are your fully implemented components ---
# from chronograph_middleware_layer import ChronographMiddleware
# from chronograph_gnosis_layer import ChronoGnosisLayer, GnosisConfig
# from knowledge_substrate_layer import (  # Assuming this is where PerceptualInputType lives
#     PerceptualInputType,
#     KnowledgeSubstrateLayer,
# )

# For standalone testing (replace with actual imports)
class ChronographMiddleware:  # Mock
    async def get_entity(self, entity_id: str, include_history: bool = False) -> Optional[Dict]:
        print(f"Mock Chronograph: get_entity({entity_id}, {include_history})")
        return {"id": entity_id, "features": {}}
    async def record_entity_state(self, entity_id: str, state_data: Dict, confidence:float):
        print(f"Mock Chronograph: record({entity_id}, {state_data})")
class ChronoGnosisLayer:  # Mock
    async def generate_embeddings(self, entity_ids: List[str]) -> Dict[str, Dict]:
      return {entity_id: {"embedding": [0.1, 0.2, 0.3]} for entity_id in entity_ids}
    async def extract_causal_patterns(self, *args, **kwargs):
      return {}

class PerceptualInputType(str):  # Mock
    TEXT = "text"
    IMAGE = "image"

# --- Configuration ---
class CausalConfig(BaseModel):
    correlation_threshold: float = Field(
        0.7, description="Minimum correlation to consider a causal link."
    )
    intervention_wait_time: float = Field(
        1.0, description="Time to wait (seconds) after an intervention."
    )
    max_prediction_error_history: int = Field(
        10, description="Maximum number of prediction errors to store."
    )
    min_observations_for_correlation: int = Field(
        5, description="Minimum observations for a feature"
    )


# --- Causal Graph ---
class CausalGraph:
    """
    Represents a causal graph.  Enhanced implementation with do-calculus support.
    """

    def __init__(self):
        self.variables: Dict[str, Dict] = {}  # {variable_name: {type: str, values: list}}
        self.edges: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )  # {parent: {child: strength}}

    def add_variable(self, variable_name: str, var_type: str = "numeric", values: Optional[List] = None):
        """Adds a variable (node) to the graph."""
        if variable_name not in self.variables:
            self.variables[variable_name] = {"type": var_type, "values": values if values else []}
            self.edges[variable_name] = {}

    def add_causal_link(
        self, cause_variable: str, effect_variable: str, strength: float
    ):
        """Adds a directed causal link (edge) between two variables."""
        self.add_variable(cause_variable)
        self.add_variable(effect_variable)
        self.edges[cause_variable][effect_variable] = strength

    def update_causal_strength(
        self, cause_variable: str, effect_variable: str, new_strength: float
    ):
        """Updates the strength of an existing causal link."""
        if (
            cause_variable in self.edges
            and effect_variable in self.edges[cause_variable]
        ):
            self.edges[cause_variable][effect_variable] = new_strength

    def get_causal_parents(self, variable_name: str) -> Dict[str, float]:
        """Gets the causal parents (direct causes) of a variable."""
        parents = {}
        for parent, children in self.edges.items():
            if variable_name in children:
                parents[parent] = children[variable_name]
        return parents

    def get_causal_children(self, variable_name: str) -> Dict[str, float]:
        """Gets the causal children (direct effects) of a variable."""
        return self.edges.get(variable_name, {})

    def perform_intervention(self, variable_name: str, new_value: Any):
        """
        Simulates an intervention (do-operator) on a variable. Removes incoming edges.
        """
        if variable_name not in self.variables:
          raise ValueError(f"Variable {variable_name} does not exist")
        # Remove incoming edges (do-calculus)
        for parent in self.get_causal_parents(variable_name):
            del self.edges[parent][variable_name]

        # Update the variable's possible values (if it's a discrete variable)
        if self.variables[variable_name]["type"] == "discrete":
            if new_value not in self.variables[variable_name]["values"]:
                self.variables[variable_name]["values"].append(new_value)
        # Record that this variable has been intervened on (for explanation, etc.)
        self.variables[variable_name]["intervened"] = True

    def to_dict(self) -> Dict:
      """Converts the graph to a dictionary for easy serialization"""
      return {
        "variables": self.variables,
        "edges": self.edges
      }
    @classmethod
    def from_dict(cls, data:Dict):
      """Creates a Causal Graph object from a dictionary"""
      graph = cls()
      graph.variables = data["variables"]
      # Need to use defaultdict again for the edges
      graph.edges = defaultdict(lambda: defaultdict(float))
      for parent, children in data['edges'].items():
        for child, strength in children.items():
          graph.edges[parent][child] = strength
      return graph

class CausalRepresentationLearner:
    """
    Learns causal relationships between features and entities.
    """

    def __init__(self, config: CausalConfig):
        self.config = config
        self.causal_graph = CausalGraph()
        self.observation_history: List[Dict] = []
        self.intervention_outcomes: List[Dict] = []
        self.correlation_matrix: Optional[Dict] = None
        self.counterfactual_history: List[Dict] = []  # For Seth's counterfactuals
        self.prediction_errors: Dict[str, List[float]] = defaultdict(list)

    async def update_from_observations(self, entity_features: Dict[str, Dict]):
        """
        Update causal model based on observed correlations between features.
        """
        self.observation_history.append(
            {"entity_features": entity_features, "timestamp": time.time()}
        )

        feature_values: Dict[str, List[float]] = defaultdict(list)
        feature_entities: Dict[str, List[str]] = defaultdict(list)

        for entity_id, features in entity_features.items():
            for feature_name, feature_value in features.items():
                # Check for Feature object (assuming a class with .value)
                if hasattr(feature_value, "value"):
                    value = feature_value.value
                else:
                    value = feature_value #if it's not, use value directly

                # Handle lists and NumPy arrays (take the mean if numeric)
                if isinstance(value, (list, np.ndarray)):
                    if all(isinstance(x, (int, float, np.number)) for x in value):
                        value = np.mean(value)  # Take the mean
                    else:
                        continue  # Skip non-numeric lists/arrays

                if isinstance(value, (int, float, np.number)):
                    feature_values[feature_name].append(value)
                    feature_entities[feature_name].append(
                        entity_id
                    )  # Keep track for later
        # Check for enough observations
        min_obs = self.config.min_observations_for_correlation
        valid_features = [
            f for f, values in feature_values.items() if len(values) >= min_obs
        ]
        if len(valid_features) < 2:
            return  # Not enough data

        # Compute correlations
        correlation_data = np.array([feature_values[f] for f in valid_features])
        if correlation_data.shape[0] < 2:  # Check for sufficient features
            return

        try:
            correlation_matrix = np.corrcoef(correlation_data)
            self.correlation_matrix = {
                "features": valid_features,
                "matrix": correlation_matrix,
                "entities": feature_entities,  # Store entity mapping
            }
            await self._update_causal_graph_from_correlations()  # Make async

        except Exception as e:
            print(f"Error computing correlations: {e}")

    async def _update_causal_graph_from_correlations(self):
        """Updates the causal graph based on the correlation matrix."""
        if self.correlation_matrix is None:
            return

        features = self.correlation_matrix["features"]
        matrix = self.correlation_matrix["matrix"]
        feature_entities = (
            self.correlation_matrix["entities"]
        )  # Get entity mapping

        for i, feature1 in enumerate(features):
            # Determine variable type (simplified)
            var_type = "numeric"  # Assume numeric for now, could infer
            self.causal_graph.add_variable(feature1, var_type=var_type)


            for j, feature2 in enumerate(features):
                if i != j and abs(matrix[i, j]) > self.config.correlation_threshold:
                    causal_strength = abs(matrix[i, j])

                    # Simplified temporal precedence (using entity features, if available)
                    # Get entities that have both features.
                    common_entities = list(set(feature_entities[feature1]).intersection(feature_entities[feature2]))

                    if common_entities: # If there's overlap in entities, try to find temporal order.
                      entity_example = common_entities[0] # Just using the first one
                      # Find first observation with both features for any shared entity.
                      feature1_time = None
                      feature2_time = None
                      # Search observation history
                      for observation in self.observation_history:
                        if entity_example in observation["entity_features"]:
                          if feature1 in observation["entity_features"][entity_example]:
                            feature1_time = observation["timestamp"]
                          if feature2 in observation["entity_features"][entity_example]:
                            feature2_time = observation["timestamp"]
                          if feature1_time and feature2_time: #found
                            break

                      #If we have timestamps for both, use temporal order
                      if feature1_time and feature2_time:
                        if feature1_time < feature2_time:
                            self.causal_graph.add_causal_link(
                                feature1, feature2, causal_strength
                            )
                        elif feature2_time < feature1_time:
                            self.causal_graph.add_causal_link(
                                feature2, feature1, causal_strength
                            )
                        #if they're equal, don't create a link
                      else:
                        # Fallback if no temporal order found, still use correlation
                        if i < j:
                          self.causal_graph.add_causal_link(feature1, feature2, causal_strength)
                        else:
                          self.causal_graph.add_causal_link(feature2, feature1, causal_strength)
                    else:
                      # If no common entities, base solely on matrix index, as before
                      if i < j:
                          self.causal_graph.add_causal_link(feature1, feature2, causal_strength)
                      else:
                          self.causal_graph.add_causal_link(feature2, feature1, causal_strength)

    async def perform_causal_intervention(
        self, entity_id: str, feature_name: str, new_value: Any, entities: Dict
    ):
        """
        Performs a causal intervention by changing a feature value and tracks effects.
        """
        variable_name = f"{entity_id}:{feature_name}"
        self.causal_graph.add_variable(variable_name)  # Ensure variable exists

        pre_state = await self._capture_entity_states(entities)
        old_value = None

        if entity_id in entities and feature_name in entities[entity_id]:
            # Record the intervention
            old_value = entities[entity_id][feature_name] # Access the value directly
            self.causal_graph.perform_intervention(variable_name, new_value)

            entities[entity_id][feature_name] = new_value   # Directly set the new value

            # Allow some time for effects to propagate
            await asyncio.sleep(self.config.intervention_wait_time)
            post_state = await self._capture_entity_states(entities)

            self.intervention_outcomes.append(
                {
                    "variable": variable_name,
                    "old_value": old_value,
                    "new_value": new_value,
                    "pre_state": pre_state,
                    "post_state": post_state,
                    "timestamp": time.time(),
                }
            )

            await self._update_from_intervention(
                variable_name, pre_state, post_state
            )  # Make async
            return True
        return False

    async def _capture_entity_states(self, entities: Dict) -> Dict:
        """Capture current state of all entities and their features."""
        state = {}
        for entity_id, entity_data in entities.items():
            if isinstance(entity_data, dict):  # Handle dictionary case
                state[entity_id] = {
                    "features": entity_data,  # Store the entire feature dict
                }
            else:  # Assume it's an Entity object (or similar)
                state[entity_id] = {
                    "features": {
                        name: feature.value for name, feature in entity_data.features.items()
                    },
                }
        return state

    async def _update_from_intervention(
        self, intervention_var: str, pre_state: Dict, post_state: Dict
    ):
        """Updates the causal model based on intervention effects."""
        changes = self._identify_changes(pre_state, post_state)
        changes = [
            change for change in changes if change["variable"] != intervention_var
        ]  # Exclude intervened var

        for change in changes:
            effect_var = change["variable"]

            # Get entity and feature from variable name
            if ":" in effect_var:
                _, feature_name = effect_var.split(
                    ":", 1
                )  # Don't need entity ID for graph update

                # Strengthen causal link, using a fixed strength increase for now
                self.causal_graph.add_causal_link(intervention_var, effect_var, 0.8)

                if change["magnitude"] > 0.5:  # If substantial change
                    self.causal_graph.update_causal_strength(
                        intervention_var, effect_var, 0.9
                    )  # Further increase strength

    def _identify_changes(self, pre_state: Dict, post_state: Dict) -> List[Dict]:
        """Identifies changes between pre- and post-intervention states."""
        changes = []

        for entity_id, pre_entity in pre_state.items():
            if entity_id in post_state:
                post_entity = post_state[entity_id]

                for feature_name, pre_value in pre_entity["features"].items():
                    if feature_name in post_entity["features"]:
                        post_value = post_entity["features"][feature_name]

                        if isinstance(pre_value, (int, float, np.number)) and isinstance(
                            post_value, (int, float, np.number)
                        ):
                            if pre_value != post_value:
                                magnitude = abs(post_value - pre_value) / (
                                    1 + abs(pre_value)
                                )
                                changes.append(
                                    {
                                        "variable": f"{entity_id}:{feature_name}",
                                        "pre_value": pre_value,
                                        "post_value": post_value,
                                        "magnitude": magnitude,
                                    }
                                )
                        elif isinstance(pre_value, (list, np.ndarray)) and isinstance(
                            post_value, (list, np.ndarray)
                        ):
                            pre_array = np.array(pre_value)
                            post_array = np.array(post_value)

                            if pre_array.shape != post_array.shape:
                                changes.append(
                                    {
                                        "variable": f"{entity_id}:{feature_name}",
                                        "pre_value": pre_value,
                                        "post_value": post_value,
                                        "magnitude": 1.0,
                                    }
                                )
                            elif not np.array_equal(pre_array, post_array):
                                diff = np.linalg.norm(post_array - pre_array) / (
                                    np.linalg.norm(pre_array) + 1e-9
                                )  # Add small constant
                                changes.append(
                                    {
                                        "variable": f"{entity_id}:{feature_name}",
                                        "pre_value": pre_value,
                                        "post_value": post_value,
                                        "magnitude": min(1.0, diff),
                                    }
                                )
                        elif pre_value != post_value:  # For other types (string, bool)
                            changes.append({
                                "variable": f"{entity_id}:{feature_name}",
                                "pre_value": pre_value,
                                "post_value": post_value,
                                "magnitude": 1.0
                            })

        return changes


    async def generate_causal_explanation(self, entity_id: str, feature_name: str) -> str:
        """Generates a natural language explanation of causal relationships."""
        variable_name = f"{entity_id}:{feature_name}"

        if variable_name not in self.causal_graph.variables:
            return f"No causal information available for {feature_name} of entity {entity_id}."

        parents = self.causal_graph.get_causal_parents(variable_name)

        if not parents:
            return f"{feature_name} of entity {entity_id} does not appear to be caused by other observed features."

        sorted_parents = sorted(parents.items(), key=lambda x: x[1], reverse=True)
        explanation = f"{feature_name} of entity {entity_id} is primarily influenced by:\n"

        for parent_name, strength in sorted_parents:
            if ":" in parent_name:
                _, parent_feature = parent_name.split(":", 1)
            else:
                parent_feature = parent_name

            if strength > 0.8:
                strength_desc = "strongly"
            elif strength > 0.5:
                strength_desc = "moderately"
            else:
                strength_desc = "weakly"

            explanation += f"- {parent_feature} ({strength_desc} influences, strength: {strength:.2f})\n"

        return explanation

    async def generate_counterfactuals(
        self, entity_id: str, feature_name: str, entities: Dict
    ) -> List[Dict]:
        """Generates counterfactual scenarios and predicts their effects."""

        variable_name = f"{entity_id}:{feature_name}"
        if (
            entity_id not in entities
            or feature_name not in entities[entity_id]
        ):
            return []

        current_value = entities[entity_id][feature_name] #access directly

        if isinstance(current_value, (int, float, np.number)):
            alternatives = [
                current_value * 0.5,
                current_value * 1.5,
                -current_value,
                0.0,
            ]
        elif isinstance(current_value, (list, np.ndarray)):
            current_array = np.array(current_value)
            alternatives = [
                np.zeros_like(current_array),
                -current_array,
                current_array * 2,
                current_array * 0.5,
            ]
        else:
            alternatives = [None]

        base_state = await self._capture_entity_states(entities)
        counterfactuals = []

        for alt_value in alternatives:
            if alt_value is not None and np.array_equal(alt_value, current_value):
                continue

            cf_state = await self._deep_copy_state(base_state)
            # Apply counterfactual change
            if alt_value is not None:
              cf_state[entity_id]["features"][feature_name] = alt_value

            predicted_effects = await self._predict_intervention_effects(  # Make async
                variable_name, alt_value, cf_state
            )
            counterfactuals.append(
                {"alternative_value": alt_value, "predicted_effects": predicted_effects}
            )

        self.counterfactual_history.append(
            {
                "entity_id": entity_id,
                "feature_name": feature_name,
                "current_value": current_value,
                "counterfactuals_count": len(counterfactuals),
                "timestamp": time.time(),
            }
        )

        return counterfactuals

    async def _predict_intervention_effects(
        self, variable_name: str, new_value: Any, current_state: Dict
    ) -> Dict:
        """Predicts the effects of an intervention using the causal model."""
        children = self.causal_graph.get_causal_children(variable_name)

        if not children:
            return {}

        effects = {}

        for child_name, causal_strength in children.items():
            if ":" not in child_name:
                continue

            child_entity_id, child_feature = child_name.split(":", 1)

            if (
                child_entity_id not in current_state
                or child_feature not in current_state[child_entity_id]["features"]
            ):
                continue

            child_value = current_state[child_entity_id]["features"][child_feature]

            if isinstance(child_value, (int, float, np.number)) and isinstance(
                new_value, (int, float, np.number)
            ):
                change_magnitude = abs(new_value) / (1.0 + abs(new_value))
                effect_magnitude = causal_strength * change_magnitude
                effect_direction = 1.0 if new_value > 0 else -1.0
                new_child_value = (
                    child_value + effect_direction * effect_magnitude * abs(child_value)
                )
                effects[child_name] = {
                    "current_value": child_value,
                    "predicted_value": new_child_value,
                    "change_magnitude": abs(new_child_value - child_value)
                    / (1.0 + abs(child_value)),
                    "causal_strength": causal_strength,
                }

            elif isinstance(child_value, (list, np.ndarray)) and isinstance(
                new_value, (list, np.ndarray)
            ):
                try:
                    child_array = np.array(child_value)
                    new_array = np.array(new_value)
                    scaling_factor = causal_strength * 0.5
                    predicted_array = (
                        child_array + (new_array - child_array) * scaling_factor
                    )
                    effects[child_name] = {
                        "current_value": child_value,
                        "predicted_value": predicted_array.tolist(),
                        "change_magnitude": np.linalg.norm(predicted_array - child_array)
                        / (np.linalg.norm(child_array) + 1e-09),
                        "causal_strength": causal_strength,
                    }

                except Exception:
                    continue  # Skip if incompatible types/shapes

            # Could add handling for other types (bool, str) if needed

        return effects

    async def _deep_copy_state(self, state: Dict) -> Dict:
        """Creates a deep copy of the entity state dictionary."""
        copied_state = {}

        for entity_id, entity_data in state.items():
            copied_entity = {"features": {}}

            for feature_name, feature_value in entity_data["features"].items():
                if isinstance(feature_value, (list, np.ndarray)):
                    copied_value = np.array(feature_value).tolist()  # Copy arrays/lists
                else:
                    copied_value = feature_value  # Copy scalars

                copied_entity["features"][feature_name] = copied_value

            copied_state[entity_id] = copied_entity

        return copied_state

    async def validate_counterfactual(
        self,
        entity_id: str,
        feature_name: str,
        alternative_value: Any,
        actual_effects: Dict,
        entities: Dict,
    ) -> Dict:
        """Validates a counterfactual prediction against observed effects."""

        counterfactuals = await self.generate_counterfactuals(
            entity_id, feature_name, entities
        )
        matching_cf = None
        for cf in counterfactuals:
            if np.array_equal(cf["alternative_value"], alternative_value):
                matching_cf = cf
                break

        if not matching_cf:
            return {"valid": False, "error": "No matching counterfactual found"}

        predicted_effects = matching_cf["predicted_effects"]
        validation_results = {}
        total_error = 0.0
        num_valid_effects = 0

        for effect_var, actual in actual_effects.items():
            if effect_var in predicted_effects:
                predicted = predicted_effects[effect_var]

                if isinstance(actual, (int, float, np.number)) and isinstance(
                    predicted["predicted_value"], (int, float, np.number)
                ):
                    error = abs(actual - predicted["predicted_value"]) / (
                        1.0 + abs(actual)
                    )
                    total_error += error
                    num_valid_effects += 1
                    validation_results[effect_var] = {
                        "predicted": predicted["predicted_value"],
                        "actual": actual,
                        "error": error,
                        "accurate": error < 0.3,  # Example threshold
                    }

                elif isinstance(actual, (list, np.ndarray)) and isinstance(
                    predicted["predicted_value"], (list, np.ndarray)
                ):
                    actual_array = np.array(actual)
                    predicted_array = np.array(predicted["predicted_value"])
                    if actual_array.shape == predicted_array.shape:
                        error = np.linalg.norm(actual_array - predicted_array) / (
                            np.linalg.norm(actual_array) + 1e-9
                        )
                        total_error += error
                        num_valid_effects += 1
                        validation_results[effect_var] = {
                            "predicted": predicted["predicted_value"],
                            "actual": actual,
                            "error": error,
                            "accurate": error < 0.3,  # Example threshold
                        }

        if num_valid_effects > 0:
          accuracy = sum([res['accurate'] for res in validation_results.values()]) / num_valid_effects
          avg_error = total_error/num_valid_effects
        else:
          accuracy = 0.0
          avg_error = 1.0

        return {
            "valid": True,
            "accuracy": accuracy,
            "average_error": avg_error,
            "effect_validations": validation_results,
        }

# --- Example Usage ---
async def main():
  # Mock entities (for demonstration purposes)
  entities = {
    "entity_1": {
        "feature_A": 10,
        "feature_B": 25,
        "feature_C": 5
    },
    "entity_2": {
        "feature_A": 12,
        "feature_B": 20,
         "feature_D": "high"
    },
      "entity_3":{
          "feature_A": 13,
          "feature_B": 32
      }
  }

  config = CausalConfig()
  learner = CausalRepresentationLearner(config)

  # Initial observation
  await learner.update_from_observations(entities)
  print("Initial Causal Graph:", learner.causal_graph.edges)

  # Perform an intervention
  await learner.perform_causal_intervention("entity_1", "feature_A", 5, entities) #changed
  print("Causal Graph after intervention:", learner.causal_graph.edges)

  # Update again with new observations after intervention
  # In reality this data would come from Chronograph
  entities["entity_1"]["feature_A"] = 5  # Reflect the intervention
  entities["entity_1"]["feature_B"] = 22 # Say feature B changed
  entities["entity_2"]["feature_B"] = 21
  await learner.update_from_observations(entities)
  print("Causal Graph after update:", learner.causal_graph.edges)


  # Generate counterfactuals
  counterfactuals = await learner.generate_counterfactuals("entity_1", "feature_A", entities)
  print("Counterfactuals:", counterfactuals)

  # Generate an explanation
  explanation = await learner.generate_causal_explanation("entity_1", "feature_B")
  print(explanation)

  # Example of validating a counterfactual after a REAL intervention
  # and subsequent observations.
  # Assuming we did intervene on entity_1, feature_A to a value of 5:

  # 1. SIMULATE REAL WORLD:  We'd observe the ACTUAL effects
  actual_effects = {
      "entity_1:feature_B": {"value": 23},  # Let's say B actually became 23
      "entity_2:feature_B": {"value": 21} #and this stayed the same
  }

  # 2. Validate
  validation_results = await learner.validate_counterfactual(
      "entity_1", "feature_A", 5, actual_effects, entities
  )
  print("\nCounterfactual Validation Results:")
  print(validation_results)

if __name__ == "__main__":
  asyncio.run(main())