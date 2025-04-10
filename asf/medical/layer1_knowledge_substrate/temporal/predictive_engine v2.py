import time
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any

class Relationship:
    """
    Represents a relationship between two entities in the knowledge graph.
    """

    def __init__(self, source_id: str, target_id: str, rel_type: str, strength: float = 1.0, timestamp: Optional[float] = None, justifications: Optional[List[str]] = None):
        """
        Initializes a Relationship object.

        Args:
            source_id: The ID of the source entity.
            target_id: The ID of the target entity.
            rel_type: The type of relationship (e.g., "causes", "part_of", "is_a").
            strength: The strength of the relationship (0.0 to 1.0).
            timestamp: The time the relationship was established or updated.
            justifications: Optional list of justifications for the relationship.
        """
        self.source_id = source_id
        self.target_id = target_id
        self.rel_type = rel_type
        self.strength = max(0.0, min(1.0, strength))  # Ensure strength is within [0, 1]
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.justifications = justifications if justifications is not None else []

    def __repr__(self):
        return (f"Relationship(source_id='{self.source_id}', target_id='{self.target_id}', "
                f"rel_type='{self.rel_type}', strength={self.strength:.2f}, timestamp={self.timestamp})")

    def to_dict(self):
        """Converts the relationship to a dictionary."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "rel_type": self.rel_type,
            "strength": self.strength,
            "timestamp": self.timestamp,
            "justifications": self.justifications,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Relationship":
      """Creates a relationship object from a dictionary."""
      return cls(
          data["source_id"],
          data["target_id"],
          data["rel_type"],
          data["strength"],
          data.get("timestamp"),  # Use get() to handle missing keys
          data.get("justifications"),
      )

    def update_strength(self, new_strength: float):
        """Updates the strength of the relationship."""
        self.strength = max(0.0, min(1.0, new_strength))
        self.timestamp = time.time()  # Update timestamp

    def add_justification(self, justification: str):
        """Adds a justification to the relationship."""
        self.justifications.append(justification)
        self.timestamp = time.time()

    def is_similar(self, other: "Relationship", type_threshold: float = 0.1, strength_threshold: float = 0.2) -> bool:
      """Checks if another relationship is similar (same source, target, type, and close strength.)"""
      if not isinstance(other, Relationship):
        return False

      if self.source_id != other.source_id or self.target_id != other.target_id:
        return False

      # Compare type similarity (using a string similarity metric if necessary)
      if self.rel_type != other.rel_type:
        # Basic string similarity check (you might use a more sophisticated method)
        len_diff = abs(len(self.rel_type) - len(other.rel_type))
        max_len = max(len(self.rel_type), len(other.rel_type))
        type_similarity = 1.0 - (len_diff / max_len) if max_len > 0 else 1.0
        if type_similarity < type_threshold:
            return False

      #Compare Strength
      return abs(self.strength - other.strength) <= strength_threshold
    
# --- PredictiveTemporalEngine (Transformer-Based) ---
class PredictiveTemporalEngine(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # --- Transformer-Based Model ---
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=input_size,
                nhead=8,  # Number of attention heads.  Adjust as needed.
                dim_feedforward=hidden_size * 2,  # Feedforward dimension
                batch_first=True
            ),
            num_layers=3  # Number of transformer layers. Adjust as needed.
        )
        self.output_layer = nn.Linear(input_size, input_size)  # Output layer
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.output_layer.bias.data.zero_()
        self.output_layer.weight.data.uniform_(-initrange, initrange)


    def forward(self, input_sequence, hidden_state=None): #Hidden state not used with Transformers, but keep for compatibility
        # Input sequence: (batch_size, sequence_length, input_size)

        # --- Transformer Forward Pass ---
        transformer_output = self.temporal_transformer(input_sequence)
        #transformer output is (batch_size, sequence_length, input_size)

        # Predict next state
        predictions = self.output_layer(transformer_output)  # Linear layer
        #predictions: (batch_size, sequence_length, input_size)

        return predictions, None #Return None for hidden_state to maintain compatibility


# --- AutocatalyticNetwork (Modified for PyTorch and Relationships) ---

class AutocatalyticNetwork(nn.Module): #Make the whole class a nn.Module
    def __init__(self, num_nodes, initial_connections=None, knowledge_graph=None):
        super().__init__()  # Call superclass constructor
        self.num_nodes = num_nodes
        self.connections = defaultdict(lambda: {})

        # Initialize weights (now a Parameter for learning)
        self.weights = nn.Parameter(torch.randn(num_nodes, num_nodes) * 0.1) #Random, learnable
        self.activation_threshold = 0.5

        if initial_connections:
          for (source, target, strength, rel_type) in initial_connections:
            self.add_connection(source, target, strength, rel_type)

        # --- Inject Prior Knowledge (if provided) ---
        if knowledge_graph:
            self._inject_knowledge_graph(knowledge_graph)

    def add_connection(self, source, target, strength, rel_type="causes"):
        if not (0 <= source < self.num_nodes and 0 <= target < self.num_nodes):
            raise ValueError("Invalid node indices")
        self.connections[str(source)][str(target)] = {"strength": strength, "type": rel_type}
        # Directly update the weight matrix using the indices.
        self.weights.data[source, target] = strength  # Update the learnable weight


    def remove_connection(self, source, target):
        if str(source) in self.connections and str(target) in self.connections[str(source)]:
            del self.connections[str(source)][str(target)]
            self.weights.data[source, target] = 0.0 #Reset

    def get_connection_strength(self, source, target):
        # Now get strength directly from the weight matrix.
        return self.weights[source, target].item()

    def _inject_knowledge_graph(self, knowledge_graph):
        """
        Initializes network weights based on a knowledge graph.

        Args:
            knowledge_graph:  A representation of prior knowledge (e.g., a dictionary
                             of relationships, or a NetworkX graph).  The exact format
                             depends on your chosen knowledge representation.
        """
        # Example:  Assume knowledge_graph is a dictionary:
        # { (source_node, target_node): relationship_type }

        for (source, target), rel_type in knowledge_graph.items():
            if 0 <= source < self.num_nodes and 0 <= target < self.num_nodes:
                if rel_type == "causes":
                    self.weights.data[source, target] = 0.8  # Strong positive weight
                elif rel_type == "inhibits":
                    self.weights.data[source, target] = -0.8 # Strong negative weight
                # Add more relationship types as needed

    def forward(self, external_input: Optional[torch.Tensor] = None) -> torch.Tensor:
      """
      Performs a single forward pass of the network.

      Args:
        external_input: External input, a tensor of shape (batch_size, num_nodes)

      Returns:
        next_state: New activation states: Tensor of shape (batch_size, num_nodes)
      """
      if external_input is None:
        external_input = torch.zeros(1, self.num_nodes) #batch size 1 if none.

      net_input = torch.matmul(self.current_state, self.weights.T) + external_input
      next_state = torch.sigmoid(net_input) #Sigmoid
      next_state = (next_state > self.activation_threshold).float() #Threshold
      self.current_state = next_state

      return next_state

    def run_to_stability(self, initial_state: np.ndarray, max_steps: int = 100, external_input:Optional[np.ndarray]=None) -> Tuple[np.ndarray, bool]:
      """Runs the network to stability, or for max_steps.
      Args:
        initial_state: Initial activation state of network.
        max_steps: Max steps to run.
        external_input: optional tensor for constant external input.

      Returns:
          final_state: Final state.
          is_stable: Boolean
      """

      self.current_state = torch.tensor(initial_state, dtype=torch.float32).unsqueeze(0) #Add batch dimension
      if external_input is not None:
        external_input = torch.tensor(external_input, dtype=torch.float32).unsqueeze(0)
      else:
        external_input = torch.zeros_like(self.current_state)

      previous_state = self.current_state.clone()
      for _ in range(max_steps):
        current_state = self.forward(external_input)
        if torch.equal(current_state, previous_state):
          return current_state.squeeze(0).detach().cpu().numpy(), True
        previous_state = current_state.clone()
      return self.current_state.squeeze(0).detach().cpu().numpy(), False



# --- PredictiveProcessor (Modified for Integration) ---

class PredictiveProcessor:
    """
    Implements Seth's predictive processing principles across the ASF system.

    Manages the balance between predictions and data-driven updates,
    tracking prediction errors and precision weighting.

    This is a cross-cutting component that coordinates predictive processing
    across all parts of Layer 1.

    Philosophical influences: Seth's Predictive Processing, Friston's Free Energy
    """

    def __init__(self):
        # Store predictions for entities and contexts
        self.prediction_models = {}

        # Track prediction errors for assessing precision
        self.prediction_errors = defaultdict(list)

        # Precision values (inverse variance of prediction errors)
        self.precision_weights = {}

        # Learning rates (adaptive based on prediction error and precision)
        self.learning_rates = {}

        # Surprise history for monitoring
        self.surprise_history = []

        # Default parameters
        self.default_precision = 1.0
        self.min_learning_rate = 0.1
        self.max_learning_rate = 0.9

        # Maximum history length to prevent unbounded memory growth
        self.max_history_length = 50

        # Statistics
        self.stats = {
            "predictions_made": 0,
            "predictions_evaluated": 0,
            "avg_error": 0.0,
            "avg_precision": 1.0
        }

    def register_prediction(self, entity_id, context_id, predicted_value, metadata=None):
        """
        Register a prediction for later evaluation

        Args:
            entity_id: Identifier for entity being predicted
            context_id: Identifier for context in which prediction is made
            predicted_value: The predicted value
            metadata: Optional additional information about the prediction

        Returns:
            prediction_id: Unique identifier for this prediction
        """
        # Create unique ID for this prediction
        prediction_id = f"{entity_id}_{context_id}_{int(time.time()*1000)}"

        # Store prediction
        self.prediction_models[prediction_id] = {
            "entity_id": entity_id,
            "context_id": context_id,
            "value": predicted_value,
            "timestamp": time.time(),
            "evaluated": False,
            "metadata": metadata or {}
        }

        # Update statistics
        self.stats["predictions_made"] += 1

        return prediction_id

    def evaluate_prediction(self, entity_id, context_id, actual_value, prediction_id=None):
        """
        Evaluate prediction against actual value and update precision

        Args:
            entity_id: Entity identifier
            context_id: Context identifier
            actual_value: Observed value to compare against prediction
            prediction_id: Optional specific prediction to evaluate

        Returns:
            Dictionary with evaluation results
        """
        # Find matching prediction
        if prediction_id is not None and prediction_id in self.prediction_models:
            prediction = self.prediction_models[prediction_id]
        else:
            # Find most recent unevaluated prediction for this entity/context
            matching_predictions = [
                (pid, p) for pid, p in self.prediction_models.items()
                if p["entity_id"] == entity_id and p["context_id"] == context_id and not p["evaluated"]
            ]

            if not matching_predictions:
                return None  # No matching prediction found

            # Sort by timestamp (most recent first)
            matching_predictions.sort(key=lambda x: x[1]["timestamp"], reverse=True)
            prediction_id, prediction = matching_predictions[0]

        # Skip if already evaluated
        if prediction["evaluated"]:
            return None

        # Calculate prediction error
        predicted_value = prediction["value"]
        error = self._calculate_prediction_error(predicted_value, actual_value)

        # Track error for this entity
        self.prediction_errors[entity_id].append(error)

        # Trim history if needed
        if len(self.prediction_errors[entity_id]) > self.max_history_length:
            self.prediction_errors[entity_id] = self.prediction_errors[entity_id][-self.max_history_length:]

        # Update precision (inverse variance of prediction errors)
        if len(self.prediction_errors[entity_id]) > 1:
            variance = np.var(self.prediction_errors[entity_id])
            precision = 1.0 / (variance + 1e-6)  # Avoid division by zero

            # Cap precision to reasonable range
            precision = min(10.0, precision)

            # Store precision
            self.precision_weights[entity_id] = precision
        else:
            # Default precision for first observation
            precision = self.default_precision
            self.precision_weights[entity_id] = precision

        # Calculate adaptive learning rate based on prediction error and precision
        # Higher error -> higher learning rate (learn more from surprising events)
        # Higher precision -> lower learning rate (already well-predicted)
        base_learning_rate = min(0.8, error * 2)  # Error-proportional component
        precision_factor = max(0.1, min(0.9, 1.0 / (1.0 + precision * 0.2)))  # Precision-based modifier

        learning_rate = min(
            self.max_learning_rate,
            max(self.min_learning_rate, base_learning_rate * precision_factor)
        )

        # Store learning rate
        self.learning_rates[entity_id] = learning_rate

        # Mark prediction as evaluated
        prediction["evaluated"] = True
        prediction["error"] = error
        prediction["precision"] = precision
        prediction["learning_rate"] = learning_rate
        prediction["evaluation_time"] = time.time()

        # Update statistics
        self.stats["predictions_evaluated"] += 1
        total_evaluated = self.stats["predictions_evaluated"]
        self.stats["avg_error"] = (self.stats["avg_error"] * (total_evaluated - 1) + error) / total_evaluated
        self.stats["avg_precision"] = (self.stats["avg_precision"] * (total_evaluated - 1) + precision) / total_evaluated

        # Add to surprise history
        self.surprise_history.append({
            "entity_id": entity_id,
            "context_id": context_id,
            "error": error,
            "precision": precision,
            "timestamp": time.time()
        })

        # Trim surprise history
        if len(self.surprise_history) > self.max_history_length:
            self.surprise_history = self.surprise_history[-self.max_history_length:]

        # Return evaluation results
        return {
            "prediction_id": prediction_id,
            "error": error,
            "precision": precision,
            "learning_rate": learning_rate
        }

    def get_precision_weight(self, entity_id):
        """
        Get current precision weight for entity

        Higher precision = more predictable entity = higher weight
        """
        return self.precision_weights.get(entity_id, self.default_precision)

    def get_learning_rate(self, entity_id):
        """
        Get adaptive learning rate for entity

        This balances learning speed based on prediction errors and precision
        """
        return self.learning_rates.get(entity_id, 0.3)  # Default learning rate

    def _calculate_prediction_error(self, predicted, actual):
        """
        Calculate normalized prediction error between predicted and actual values

        Handles different data types appropriately
        """
        # Handle different value types
        if isinstance(predicted, (int, float, np.number)) and isinstance(actual, (int, float, np.number)):
            # For numeric values, normalized absolute difference
            return abs(predicted - actual) / (1.0 + abs(actual))

        elif isinstance(predicted, (list, np.ndarray)) and isinstance(actual, (list, np.ndarray)):
            # For vectors, cosine distance or normalized Euclidean distance
            predicted_arr = np.array(predicted)
            actual_arr = np.array(actual)

            if predicted_arr.shape != actual_arr.shape:
                return 1.0  # Maximum error for shape mismatch

            if predicted_arr.size == 0 or actual_arr.size == 0:
                return 1.0  # Maximum error for empty arrays

            # Try cosine similarity if vectors are non-zero
            pred_norm = np.linalg.norm(predicted_arr)
            actual_norm = np.linalg.norm(actual_arr)

            if pred_norm > 0 and actual_norm > 0:
                similarity = np.dot(predicted_arr, actual_arr) / (pred_norm * actual_norm)
                return max(0, 1.0 - similarity)  # Convert to distance
            else:
                # Fall back to normalized Euclidean for zero vectors
                diff = np.linalg.norm(predicted_arr - actual_arr)
                return min(1.0, diff / (1.0 + actual_norm))

        elif isinstance(predicted, dict) and isinstance(actual, dict):
            # For dictionaries, calculate average error across shared keys
            shared_keys = set(predicted.keys()) & set(actual.keys())

            if not shared_keys:
                return 1.0  # Maximum error if no shared keys

            errors = []
            for key in shared_keys:
                errors.append(self._calculate_prediction_error(predicted[key], actual[key]))

            return sum(errors) / len(errors)

        elif isinstance(predicted, bool) and isinstance(actual, bool):
            # For booleans, 0 for match, 1 for mismatch
            return 0.0 if predicted == actual else 1.0

        elif isinstance(predicted, str) and isinstance(actual, str):
            # For strings, Levenshtein distance would be ideal
            # For simplicity, use 0 for exact match, 1 for completely different
            if predicted == actual:
                return 0.0

            # Simple heuristic: fraction of length difference
            len_diff = abs(len(predicted) - len(actual))
            max_len = max(len(predicted), len(actual))

            if max_len == 0:
                return 0.0 if len_diff == 0 else 1.0
                
            return min(1.0, len_diff / max_len)

        else:
            # Fallback for other types
            return 1.0 if predicted != actual else 0.0

    def get_prediction_statistics(self, entity_id=None, time_window=None):
        """
        Get prediction statistics overall or for specific entity

        Args:
            entity_id: Optional entity to filter statistics for
            time_window: Optional time window in seconds

        Returns:
            Dictionary of prediction statistics
        """
        if entity_id is None:
            # Overall statistics
            return dict(self.stats)

        # Entity-specific statistics
        errors = self.prediction_errors.get(entity_id, [])

        if time_window is not None:
            # Filter to recent history
            cutoff_time = time.time() - time_window
            relevant_history = [
                entry for entry in self.surprise_history
                if entry["entity_id"] == entity_id and entry["timestamp"] >= cutoff_time
            ]
        else:
            # Use all history for this entity
            relevant_history = [
                entry for entry in self.surprise_history
                if entry["entity_id"] == entity_id
            ]

        if not relevant_history:
            return {
                "entity_id": entity_id,
                "avg_error": None,
                "precision": self.precision_weights.get(entity_id, None),
                "learning_rate": self.learning_rates.get(entity_id, None),
                "prediction_count": 0
            }

        # Calculate statistics
        avg_error = sum(entry["error"] for entry in relevant_history) / len(relevant_history)

        return {
            "entity_id": entity_id,
            "avg_error": avg_error,
            "precision": self.precision_weights.get(entity_id, None),
            "learning_rate": self.learning_rates.get(entity_id, None),
            "prediction_count": len(relevant_history)
        }

    def predict_with_precision(self, predictions, entity_id=None):
      """
      Adjust predictions based on precision weights

      For multiple predictions, weight them by precision and combine

      Args:
          predictions: Dictionary of predictions from different sources
                       {source_id: (predicted_value, base_confidence)}
          entity_id: Optional entity ID for precision lookup

      Returns:
          Final prediction and confidence
      """
      if not predictions:
          return None, 0.0

      if len(predictions) == 1:
          # Only one prediction source, just return it
          source_id, (value, confidence) = next(iter(predictions.items()))
          return value, confidence

      # Multiple prediction sources, combine with precision weighting
      weighted_values = []
      total_weight = 0.0

      for source_id, (value, confidence) in predictions.items():
          # Get precision for this source
          precision = self.precision_weights.get(source_id, self.default_precision)

          # Weight = confidence * precision
          weight = confidence * precision
          weighted_values.append((value, weight))
          total_weight += weight

      if total_weight == 0:
          # No weights, return first prediction
          return next(iter(predictions.values()))[0], 0.0

      # For numeric values, calculate weighted average
      first_value = next(iter(predictions.values()))[0]

      if isinstance(first_value, (int, float, np.number)):
          # Weighted average for numbers
          weighted_sum = sum(value * weight for value, weight in weighted_values)
          combined_value = weighted_sum / total_weight
          combined_confidence = min(1.0, total_weight / len(predictions))
          return combined_value, combined_confidence

      elif isinstance(first_value, (list, np.ndarray)) and all(
          isinstance(v[0], (list, np.ndarray)) for v in weighted_values
      ):
          # For vectors, try weighted average if shapes match
          try:
              arrays = [np.array(v) * w for v, w in weighted_values]
              if all(a.shape == arrays[0].shape for a in arrays):
                  combined_array = sum(arrays) / total_weight
                  combined_confidence = min(1.0, total_weight / len(predictions))
                  return combined_array.tolist(), combined_confidence
          except:
              # Fall back to highest weighted prediction
              pass

      # For other types or if vector combination fails, return highest weighted prediction
      best_prediction = max(weighted_values, key=lambda x: x[1])
      best_confidence = best_prediction[1] / total_weight
      return best_prediction[0], min(1.0, best_confidence)
    def integrate_prediction(self, prediction_id, integration_function, *args, **kwargs):
        """
        Integrates a prediction into the system using a provided function.

        This allows for flexible handling of predictions, which could involve
        updating knowledge, triggering actions, etc.

        Args:
            prediction_id: The ID of the prediction to integrate.
            integration_function: A callable that takes the predicted value
                                 and any other necessary arguments and performs
                                 the integration.
            *args, **kwargs:  Arguments to be passed to the integration_function.

        Returns:
            The result of the integration_function.
        """
        if prediction_id not in self.prediction_models:
            raise ValueError(f"Prediction with ID '{prediction_id}' not found.")

        prediction = self.prediction_models[prediction_id]
        predicted_value = prediction["value"]
        return integration_function(predicted_value, *args, **kwargs)

# --- OperationalClosure (Simplified) ---
class OperationalClosure:
    """Simplified Operational Closure for managing relationships"""
    def __init__(self):
        self.relationships: Dict[str, Relationship] = {} #Store Relationships

    def add_relationship(self, source_id: str, target_id: str, rel_type: str, strength: float = 1.0):
        """Adds a relationship"""
        rel_id = f"{source_id}->{target_id}:{rel_type}"
        self.relationships[rel_id] = Relationship(source_id, target_id, rel_type, strength)
        return rel_id

    def remove_relationship(self, rel_id):
        """Removes a relationship"""
        if rel_id in self.relationships:
            del self.relationships[rel_id]

    def get_relationships(self) -> Dict[str, Relationship]:
      """Returns a copy of all relationships."""
      return self.relationships.copy()

# --- TemporalKnowledgeManager (Refactored) ---

class TemporalKnowledgeManager:
    """
    Manages knowledge with temporal context, confidence, and relationships.
    Interacts with PredictiveProcessor for prediction and updates.
    """

    def __init__(self, predictive_processor: PredictiveProcessor):
        self.knowledge: Dict[str, Dict] = {}  # {entity_id: {attributes, relationships, ...}}
        self.next_entity_id = 1
        self.predictive_processor = predictive_processor  # Dependency injection

    def add_knowledge(self, domain: str, content: Dict, context: Optional[Dict] = None,
                      justifications: Optional[List[str]] = None,
                      initial_confidence: float = 0.9) -> str:
      """Adds a new knowledge entity.

      Args:
          domain (str): The domain the knowledge belongs to (e.g. 'physics', 'politics').
          content (Dict):  The core content of the knowledge (attributes).
          context (Optional[Dict]):  Contextual information.
          justifications (Optional[List[str]]): Reasons for believing the knowledge.
          initial_confidence (float):  Starting confidence (0.0 - 1.0).

      Returns:
          str: The new entity's ID.
      """
      entity_id = str(self.next_entity_id)
      self.next_entity_id +=1

      self.knowledge[entity_id] = {
          "domain": domain,
          "content": content,  # Store content as a dictionary
          "context": context if context is not None else {},
          "justifications": justifications if justifications is not None else [],
          "confidence": initial_confidence,
          "relationships": {},  # {related_entity_id: {type, strength}}
          "timestamp": time.time(),  # Use time.time() for simplicity
      }

      return entity_id

    def get_knowledge(self, entity_id: str) -> Optional[Dict]:
        """Retrieves a knowledge entity by ID."""
        return self.knowledge.get(entity_id)

    def update_knowledge(self, entity_id: str, updates: Dict):
      """Updates an entities knowledge with a dictionary of changes."""
      if entity_id not in self.knowledge:
        return
      
      #Apply updates to content.
      if 'content' in updates:
        self.knowledge[entity_id]['content'].update(updates['content'])

      #Apply updates to context
      if 'context' in updates:
        self.knowledge[entity_id]['context'].update(updates['context'])

      #Add justifications
      if 'justifications' in updates:
        if isinstance(updates['justifications'], list):
          self.knowledge[entity_id]['justifications'].extend(updates['justifications'])

      #Update confidence.
      if 'confidence' in updates:
        new_confidence = updates['confidence']
        if 0.0 <= new_confidence <= 1.0:
          self.knowledge[entity_id]['confidence'] = new_confidence

      #Update timestamp.
      self.knowledge[entity_id]['timestamp'] = time.time()

    def add_relationship(self, source_entity_id: str, target_entity_id: str,
                         rel_type: str, strength: float = 1.0):
        """Adds a relationship between two knowledge entities."""
        if source_entity_id not in self.knowledge or target_entity_id not in self.knowledge:
            raise ValueError("Invalid entity IDs")

        # Add the relationship to both entities
        self.knowledge[source_entity_id]["relationships"][target_entity_id] = {"type": rel_type, "strength": strength}
        self.knowledge[target_entity_id]["relationships"][source_entity_id] = {"type": rel_type, "strength": strength} #Bidirectional for now

    def get_related_entities(self, entity_id: str) -> Dict[str, Dict]:
        """Retrieves all entities related to a given entity."""
        if entity_id not in self.knowledge:
            return {}
        return self.knowledge[entity_id]["relationships"]

    def integrate_prediction(self, prediction_id: str, entity_id: str):
        """
        Integrates a prediction into the knowledge base.  This is a simplified
        example of how the PredictiveProcessor and TemporalKnowledgeManager
        would interact.

        Args:
            prediction_id: The ID of the prediction to integrate.
            entity_id: The ID of the entity to which the prediction applies.
        """

        # --- 1. Retrieve Prediction ---
        prediction = self.predictive_processor.prediction_models.get(prediction_id)
        if not prediction:
          raise ValueError(f"Prediction not found {prediction_id}")
        predicted_value = prediction["value"]

        # --- 2. Get current knowledge entity data
        entity_data = self.get_knowledge(entity_id)
        if not entity_data:
          raise ValueError(f"Entity Not found: {entity_id}")

        # --- 3.  Update Knowledge based on Prediction ---
        # Example: Assume the prediction is about an attribute named "temperature"
        #  and the predicted_value is a number.
        if isinstance(predicted_value, (int, float, np.number)):
          updates = {"content": {
              "temperature": predicted_value #Update temperature
          }, "justifications": [f"Prediction ID: {prediction_id}"]}
          self.update_knowledge(entity_id, updates)
          print(f"Integrated Prediction: Updated entity {entity_id} with predicted temperature: {predicted_value}")
        else:
          print(f"Unhandled prediction value {predicted_value}")

    def evaluate_predictions_and_update_confidence(self):
      """
      Evaluates all pending predictions and updates confidence based on errors.
      Iterates through ALL predictions.  This should be optimized for large datasets
      (e.g., by only evaluating predictions that are "due" to be evaluated based on
      their timestamps and prediction horizons).
      """

      for prediction_id, prediction_data in list(self.predictive_processor.prediction_models.items()): #iterate over copy
        entity_id = prediction_data['entity_id']
        context_id = prediction_data['context_id']

        if entity_id in self.knowledge and not prediction_data["evaluated"]:
            # Get the actual value to compare against.  This *depends* on
            # how your knowledge is structured.  The following is a simplified
            # example, assuming that the prediction is about an attribute
            # of the entity.
            actual_value = self.knowledge[entity_id]["content"].get("temperature") #Example

            if actual_value is not None: #If we have current knowledge
                evaluation_result = self.predictive_processor.evaluate_prediction(
                    entity_id, context_id, actual_value, prediction_id
                )

                if evaluation_result: #If it was able to be evaluated
                  print(f"Prediction Evaluation Result: {evaluation_result}")
                  # --- Update Confidence Based on Prediction Error ---
                  # Get the learning rate from the PredictiveProcessor
                  learning_rate = self.predictive_processor.get_learning_rate(entity_id)

                  # Adjust confidence (simplified example)
                  current_confidence = self.knowledge[entity_id]["confidence"]
                  new_confidence = current_confidence + learning_rate * (1 - evaluation_result["error"])
                  new_confidence = max(0.0, min(1.0, new_confidence))  # Clamp to [0, 1]
                  self.update_knowledge(entity_id, {"confidence": new_confidence})

    def run_prediction_and_evaluation_cycle(self):
        """
        Runs a complete cycle:

        1. Make Predictions (using PredictiveTemporalEngine).
        2. Register Predictions with PredictiveProcessor
        3. Simulate the passage of time/new data arrival.
        4. Evaluate the predictions using PredictiveProcessor.
        5. Update confidences in knowledge.
        """
        print("Running prediction and evaluation cycle...")

        # --- 1. Make Predictions (using PredictiveTemporalEngine) ---
        # Example: Predict the temperature of entity "sensor_1" over the next hour
        predictions = self.predictive_temporal_engine.predict_next_events(
            "sensor_1", "temperature", time_horizon=3600
        )

        # --- 2. Register Predictions with PredictiveProcessor ---
        prediction_ids = []
        for prediction in predictions:
            # In a real system, the context_id would be more meaningful
            context_id = "environment_1"
            pred_id = self.predictive_processor.register_prediction(
                "sensor_1", context_id, prediction["predicted_time"]
            )  #  Register the *time* of the prediction
            prediction_ids.append(pred_id)

        # --- 3. Simulate Time Passing / New Data Arrival ---
        # In a real system, this would involve waiting for new sensor readings
        # or other data sources.  For this example, we'll just simulate it.
        print("...Simulating time passing...")
        time.sleep(2)  # Simulate a delay

        # Simulate receiving a new temperature reading for sensor_1
        new_temperature = 28.5  # Example new temperature
        new_data = {"content": {"temperature": new_temperature}}
        #In a complete system, we'd probably have a timestamp here.

        # Add this new data to the knowledge base (treat it as a new entity for now).
        new_knowledge_id = self.add_knowledge("environment", new_data, initial_confidence = 0.95)


        # --- 4. Evaluate Predictions ---
        self.evaluate_predictions_and_update_confidence()


        # --- 5. (Optional) Display updated knowledge ---
        print("\nUpdated Knowledge:")
        for entity_id, entity_data in self.knowledge.items():
            print(f"  Entity {entity_id}: {entity_data}")

# --- Example Usage (Putting it all together) ---

# 1.  Create the components
predictive_processor = PredictiveProcessor()
knowledge_manager = TemporalKnowledgeManager(predictive_processor)
temporal_engine = PredictiveTemporalEngine(input_size=1, hidden_size=32) # Example sizes
knowledge_manager.predictive_temporal_engine = temporal_engine #Give it the temporal engine
# 2.  Add initial knowledge
entity_id = knowledge_manager.add_knowledge(
    domain="environment",
    content={"temperature": 25.0, "humidity": 60.0},
    context={"location": "office"},
    justifications=["initial sensor reading"],
    initial_confidence=0.9
)
knowledge_manager.add_relationship(entity_id, entity_id, "self", 1.0) #Example Relationship

#Run a cycle.
knowledge_manager.run_prediction_and_evaluation_cycle()

#Show Prediction Stats:
print(predictive_processor.get_prediction_statistics())