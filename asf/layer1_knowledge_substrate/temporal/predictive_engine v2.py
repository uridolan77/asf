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
      Performs a single forward pass of the network.

      Args:
        external_input: External input, a tensor of shape (batch_size, num_nodes)

      Returns:
        next_state: New activation states: Tensor of shape (batch_size, num_nodes)
      Args:
        initial_state: Initial activation state of network.
        max_steps: Max steps to run.
        external_input: optional tensor for constant external input.

      Returns:
          final_state: Final state.
          is_stable: Boolean
    Implements Seth's predictive processing principles across the ASF system.

    Manages the balance between predictions and data-driven updates,
    tracking prediction errors and precision weighting.

    This is a cross-cutting component that coordinates predictive processing
    across all parts of Layer 1.

    Philosophical influences: Seth's Predictive Processing, Friston's Free Energy
        Register a prediction for later evaluation

        Args:
            entity_id: Identifier for entity being predicted
            context_id: Identifier for context in which prediction is made
            predicted_value: The predicted value
            metadata: Optional additional information about the prediction

        Returns:
            prediction_id: Unique identifier for this prediction
        Evaluate prediction against actual value and update precision

        Args:
            entity_id: Entity identifier
            context_id: Context identifier
            actual_value: Observed value to compare against prediction
            prediction_id: Optional specific prediction to evaluate

        Returns:
            Dictionary with evaluation results
        Get current precision weight for entity

        Higher precision = more predictable entity = higher weight
        Get adaptive learning rate for entity

        This balances learning speed based on prediction errors and precision
        Calculate normalized prediction error between predicted and actual values

        Handles different data types appropriately
        Get prediction statistics overall or for specific entity

        Args:
            entity_id: Optional entity to filter statistics for
            time_window: Optional time window in seconds

        Returns:
            Dictionary of prediction statistics
      Adjust predictions based on precision weights

      For multiple predictions, weight them by precision and combine

      Args:
          predictions: Dictionary of predictions from different sources
                       {source_id: (predicted_value, base_confidence)}
          entity_id: Optional entity ID for precision lookup

      Returns:
          Final prediction and confidence
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

      Args:
          domain (str): The domain the knowledge belongs to (e.g. 'physics', 'politics').
          content (Dict):  The core content of the knowledge (attributes).
          context (Optional[Dict]):  Contextual information.
          justifications (Optional[List[str]]): Reasons for believing the knowledge.
          initial_confidence (float):  Starting confidence (0.0 - 1.0).

      Returns:
          str: The new entity's ID.
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
      Evaluates all pending predictions and updates confidence based on errors.
      Iterates through ALL predictions.  This should be optimized for large datasets
      (e.g., by only evaluating predictions that are "due" to be evaluated based on
      their timestamps and prediction horizons).
        Runs a complete cycle:

        1. Make Predictions (using PredictiveTemporalEngine).
        2. Register Predictions with PredictiveProcessor
        3. Simulate the passage of time/new data arrival.
        4. Evaluate the predictions using PredictiveProcessor.
        5. Update confidences in knowledge.