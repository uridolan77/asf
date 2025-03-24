

# === FILE: asf\__init__.py ===

"""
Autopoietic Semantic Fields (ASF) Framework
A multi-layered approach to semantic processing with autopoietic principles
"""

__version__ = "0.1.0"


# === FILE: asf\layer1_knowledge_substrate\__init__.py ===

from asf.knowledge_substrate.knowledge_substrate import KnowledgeSubstrateLayer
from asf.knowledge_substrate.predictive_processor import PredictiveProcessor
from asf.knowledge_substrate.confidence.predictive_updater import PredictiveConfidenceUpdater


# === FILE: asf\layer1_knowledge_substrate\chronograph_gnosis_layer.py ===




# === FILE: asf\layer1_knowledge_substrate\chronograph_middleware_layer.py ===




# === FILE: asf\layer1_knowledge_substrate\knowledge_substrate_layer.py ===

# asf/knowledge_substrate/knowledge_substrate.py
import time
from asf.core.enums import PerceptualInputType, PerceptualEventType
from asf.knowledge_substrate.temporal.processing_engine import TemporalProcessingEngine
from asf.knowledge_substrate.perception.entity import PerceptualEntity
from asf.knowledge_substrate.confidence.bayesian_updater import BayesianConfidenceUpdater
from asf.knowledge_substrate.linking.entity_gnn import EntityLinkingGNN
from asf.knowledge_substrate.causal.representation import CausalRepresentationLearner
from asf.knowledge_substrate.memory.energy_based import EnergyBasedMemoryManager
from asf.knowledge_substrate.extraction.text import TextFeatureExtractor
from asf.knowledge_substrate.extraction.image import ImageFeatureExtractor

class KnowledgeSubstrateLayer:
    """
    Main controller class for the Knowledge Substrate Layer (Layer 1).
    Manages integration of all components and interfaces with higher layers.
    """
    def __init__(self, config=None):
        # Initialize configuration
        self.config = config or {}
        
        # Core components
        self.entities = {}  # entity_id -> PerceptualEntity
        self.entity_relations = []  # (source_idx, target_idx, relation_type)
        
        # Feature extractors for different modalities
        self.feature_extractors = {
            PerceptualInputType.TEXT: TextFeatureExtractor(),
            PerceptualInputType.IMAGE: ImageFeatureExtractor(),
            # Other modalities would be initialized here
        }
        
        # Enhanced components
        self.bayesian_updater = BayesianConfidenceUpdater()
        self.entity_linking_gnn = EntityLinkingGNN()
        self.temporal_processor = TemporalProcessingEngine()
        self.causal_learner = CausalRepresentationLearner()
        self.memory_manager = EnergyBasedMemoryManager()
        
        # Processing statistics
        self.stats = {
            "entities_processed": 0,
            "relations_discovered": 0,
            "confidence_updates": 0,
            "causal_relations": 0,
            "temporal_patterns": 0
        }
    
    def process_input(self, input_data, input_type, context=None):
        """
        Process a new perceptual input and create/update entities
        
        Parameters:
        - input_data: Raw input data (text, image, etc.)
        - input_type: PerceptualInputType indicating modality
        - context: Optional processing context
        
        Returns the entity_id of the processed entity
        """
        # Generate a unique ID for this entity
        entity_id = f"{input_type.value}_{int(time.time()*1000)}"
        
        # Extract features using appropriate extractor
        if input_type in self.feature_extractors:
            extractor = self.feature_extractors[input_type]
            
            # Extract features with semiotic weighting
            weighted_features = extractor.extract_weighted_features(input_data, context)
            
            # Create entity with extracted features
            entity = PerceptualEntity(entity_id, input_type)
            
            # Add weighted features
            for name, feature_info in weighted_features.items():
                entity.add_feature(
                    name, 
                    feature_info['value'], 
                    confidence=feature_info['importance']
                )
            
            # Store entity
            self.entities[entity_id] = entity
            
            # Add to memory manager
            self.memory_manager.add_entity(entity, context)
            
            # Update statistics
            self.stats["entities_processed"] += 1
            
            # Process temporal aspects
            self._process_temporal_aspects(entity, input_type)
            
            # Update causal model
            self._update_causal_model()
            
            # Return the entity ID
            return entity_id
        
        # Return None if input type not supported
        return None
    
    def update_entity_confidence(self, entity_id, relevant, context=None):
        """
        Update entity confidence using Bayesian updater
        
        Parameters:
        - entity_id: ID of entity to update
        - relevant: Boolean indicating if entity was relevant in current context
        - context: Optional context for prediction
        """
        if entity_id in self.entities:
            entity = self.entities[entity_id]
            
            # Convert context to vector if needed
            context_vector = None
            if context is not None:
                if hasattr(context, 'get_feature_vector'):
                    context_vector = context.get_feature_vector()
                elif isinstance(context, dict) and 'feature_vector' in context:
                    context_vector = context['feature_vector']
            
            # Update confidence using Bayesian updater
            new_confidence = self.bayesian_updater.update_confidence(
                entity_id, 
                relevant, 
                context_vector
            )
            
            # Update entity confidence state
            new_state = self.bayesian_updater.get_confidence_state(new_confidence)
            entity.update_confidence_state(new_state, new_confidence)
            
            # Update statistics
            self.stats["confidence_updates"] += 1
            
            return new_confidence
        return None
    
    def predict_entity_relevance(self, entity_id, context):
        """
        Predict entity relevance in a given context
        
        Parameters:
        - entity_id: ID of entity to evaluate
        - context: Context for prediction
        
        Returns relevance probability between 0 and 1
        """
        # Convert context to vector if needed
        context_vector = None
        if context is not None:
            if hasattr(context, 'get_feature_vector'):
                context_vector = context.get_feature_vector()
            elif isinstance(context, dict) and 'feature_vector' in context:
                context_vector = context['feature_vector']
        
        # Use Bayesian updater to predict relevance
        return self.bayesian_updater.predict_relevance(entity_id, context_vector)
    
    def find_similar_entities(self, entity_id, modality=None, top_k=5):
        """
        Find entities similar to the specified entity
        
        Parameters:
        - entity_id: ID of query entity
        - modality: Optional filter by modality
        - top_k: Maximum number of results to return
        
        Returns list of (entity, similarity_score) tuples
        """
        if entity_id not in self.entities:
            return []
        
        query_entity = self.entities[entity_id]
        
        # Filter entities by modality if specified
        if modality is not None:
            filtered_entities = [e for e in self.entities.values() if e.input_type == modality]
        else:
            filtered_entities = list(self.entities.values())
        
        # Remove query entity from candidates
        filtered_entities = [e for e in filtered_entities if e.id != entity_id]
        
        # Update entity embeddings if needed
        self._ensure_entity_embeddings()
        
        # Use GNN to find similar entities
        similar_entities = self.entity_linking_gnn.find_similar_entities(
            query_entity,
            filtered_entities,
            top_k
        )
        
        return similar_entities
    
    def get_entity(self, entity_id):
        """Get entity by ID, using memory manager"""
        return self.memory_manager.get_entity(entity_id)
    
    def add_cross_modal_link(self, source_id, target_id):
        """
        Create a cross-modal link between two entities
        
        Parameters:
        - source_id: ID of source entity
        - target_id: ID of target entity
        
        Returns True if link created, False otherwise
        """
        if source_id in self.entities and target_id in self.entities:
            source = self.entities[source_id]
            target = self.entities[target_id]
            
            # Only link across different modalities
            if source.input_type != target.input_type:
                # Add cross-modal links in both directions
                source.add_cross_modal_link(target_id)
                target.add_cross_modal_link(source_id)
                
                # Add to relations list for GNN
                source_idx = list(self.entities.keys()).index(source_id)
                target_idx = list(self.entities.keys()).index(target_id)
                rel_type = self.entity_linking_gnn._get_relation_type(
                    source.input_type,
                    target.input_type
                )
                
                self.entity_relations.append((source_idx, target_idx, rel_type))
                
                # Update statistics
                self.stats["relations_discovered"] += 1
                
                return True
        
        return False
    
    def suggest_cross_modal_links(self, similarity_threshold=0.7):
        """
        Suggest potential cross-modal links between entities
        
        Parameters:
        - similarity_threshold: Minimum similarity score for suggested links
        
        Returns list of (entity1, entity2, similarity) tuples
        """
        # Update entity embeddings if needed
        self._ensure_entity_embeddings()
        
        # Get suggestions from GNN
        return self.entity_linking_gnn.suggest_cross_modal_links(
            list(self.entities.values()),
            similarity_threshold
        )
    
    def add_temporal_event(self, entity_id, event_type, event_data):
        """
        Add a temporal event for an entity
        
        Parameters:
        - entity_id: ID of entity
        - event_type: Type of event
        - event_data: Event data
        
        Returns pattern detection result if a pattern is found
        """
        if entity_id in self.entities:
            # Get event type as string
            if isinstance(event_type, PerceptualEventType):
                sequence_type = event_type.value
            else:
                sequence_type = str(event_type)
            
            # Register sequence if needed
            self.temporal_processor.register_sequence(entity_id, sequence_type)
            
            # Add event and check for patterns
            pattern = self.temporal_processor.add_event(entity_id, sequence_type, event_data)
            
            if pattern:
                # Update statistics
                self.stats["temporal_patterns"] += 1
            
            return pattern
        
        return None
    
    def perform_causal_intervention(self, entity_id, feature_name, new_value):
        """
        Perform a causal intervention to test causal relationships
        
        Parameters:
        - entity_id: ID of entity to modify
        - feature_name: Name of feature to intervene on
        - new_value: New value to set
        
        Returns True if intervention successful, False otherwise
        """
        return self.causal_learner.perform_causal_intervention(
            entity_id,
            feature_name,
            new_value,
            self.entities
        )
    
    def get_causal_explanation(self, entity_id, feature_name):
        """
        Get causal explanation for a feature
        
        Parameters:
        - entity_id: ID of entity
        - feature_name: Name of feature to explain
        
        Returns textual explanation of causal influences
        """
        return self.causal_learner.generate_causal_explanation(entity_id, feature_name)
    
    def sample_entities_by_energy(self, context=None, n=5):
        """
        Sample entities using energy-based model
        
        Parameters:
        - context: Optional context for relevance calculation
        - n: Number of entities to sample
        
        Returns list of sampled entities
        """
        return self.memory_manager.sample_entities(context, n)
    
    def get_layer_statistics(self):
        """Get statistics about layer operations"""
        # Combine stats from various components
        memory_stats = self.memory_manager.get_memory_statistics()
        
        return {
            **self.stats,
            "memory_usage": memory_stats
        }
    
    def _process_temporal_aspects(self, entity, input_type):
        """Process temporal aspects of a new entity"""
        # Add NEW_INPUT event
        self.add_temporal_event(
            entity.id,
            PerceptualEventType.NEW_INPUT,
            entity
        )
        
        # Get entities of same type for temporal context
        same_type_entities = [
            e for e in self.entities.values() 
            if e.input_type == input_type and e.id != entity.id
        ]
        
        # Sort by recency (most recent first)
        same_type_entities.sort(
            key=lambda e: e.temporal_metadata.last_access_time,
            reverse=True
        )
        
        # Use recent entities for context
        if same_type_entities:
            # Maintain temporal context for this modality
            self.temporal_processor.maintain_temporal_context(
                f"recent_{input_type.value}",
                same_type_entities[:5]  # Keep 5 most recent
            )
    
    def _update_causal_model(self):
        """Update causal model with current entities"""
        entity_features = {
            entity_id: entity.features
            for entity_id, entity in self.entities.items()
        }
        
        self.causal_learner.update_from_observations(entity_features)
    
    def _ensure_entity_embeddings(self):
        """Ensure all entities have GNN embeddings"""
        # Check if we need to update entity embeddings
        entities_list = list(self.entities.values())
        
        # Only update if we have entities and relations
        if entities_list and self.entity_relations:
            self.entity_linking_gnn.update_entity_embeddings(
                entities_list,
                self.entity_relations
            )



# === FILE: asf\layer1_knowledge_substrate\predictive_processor.py ===

import time
import numpy as np
from collections import defaultdict
import torch
import logging

logger = logging.getLogger(__name__)

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



# === FILE: asf\layer1_knowledge_substrate\causal\graph.py ===

# asf/knowledge_substrate/causal/graph.py
import time
import networkx as nx

from asf.knowledge_substrate.causal.variable import CausalVariable

class CausalGraph:
    """
    Represents a causal graph with variables and their relationships.
    """
    def __init__(self):
        self.variables = {}  # Map from variable name to CausalVariable object
        self.interventions = []  # Track interventions for causal discovery
    
    def add_variable(self, name, value=None, feature_id=None):
        """Add a new variable to the causal graph"""
        if name not in self.variables:
            self.variables[name] = CausalVariable(name, value, feature_id)
        return self.variables[name]
    
    def add_causal_link(self, cause_name, effect_name, strength=0.5):
        """Add a causal link between two variables"""
        # Ensure both variables exist
        if cause_name not in self.variables:
            self.add_variable(cause_name)
        if effect_name not in self.variables:
            self.add_variable(effect_name)
        
        # Add the relationship in both directions
        self.variables[cause_name].add_child(effect_name, strength)
        self.variables[effect_name].add_parent(cause_name, strength)
    
    def update_causal_strength(self, cause_name, effect_name, new_strength):
        """Update the strength of a causal relationship"""
        if cause_name in self.variables and effect_name in self.variables:
            self.variables[cause_name].update_causal_strength(effect_name, new_strength, is_parent=False)
            self.variables[effect_name].update_causal_strength(cause_name, new_strength, is_parent=True)
    
    def perform_intervention(self, variable_name, new_value):
        """
        Perform an intervention by setting a variable to a specific value
        Returns the previous value and records the intervention
        """
        if variable_name not in self.variables:
            return None
        
        var = self.variables[variable_name]
        old_value = var.value
        var.value = new_value
        
        # Record intervention
        self.interventions.append({
            "variable": variable_name,
            "old_value": old_value,
            "new_value": new_value,
            "timestamp": time.time()
        })
        
        return old_value
    
    def get_causal_parents(self, variable_name):
        """Get all causal parents of a variable"""
        if variable_name in self.variables:
            return self.variables[variable_name].parents
        return {}
    
    def get_causal_children(self, variable_name):
        """Get all causal children of a variable"""
        if variable_name in self.variables:
            return self.variables[variable_name].children
        return {}
    
    def to_networkx(self):
        """Convert causal graph to NetworkX graph for visualization and analysis"""
        G = nx.DiGraph()
        
        # Add nodes
        for name, var in self.variables.items():
            G.add_node(name, value=var.value, feature_id=var.feature_id)
        
        # Add edges
        for name, var in self.variables.items():
            for child_name, strength in var.children.items():
                G.add_edge(name, child_name, weight=strength)
        
        return G



# === FILE: asf\layer1_knowledge_substrate\causal\representation.py ===

import time
import numpy as np
from collections import defaultdict

from asf.knowledge_substrate.causal.graph import CausalGraph

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



# === FILE: asf\layer1_knowledge_substrate\causal\variable.py ===

# asf/knowledge_substrate/causal/variable.py
class CausalVariable:
    """
    Represents a variable in a causal model with its current state and relationships.
    """
    def __init__(self, name, value=None, feature_id=None):
        self.name = name
        self.value = value
        self.feature_id = feature_id  # Link to original feature if applicable
        self.parents = {}  # Map from parent variable name to causal strength
        self.children = {}  # Map from child variable name to causal strength
    
    def add_parent(self, parent_name, causal_strength=0.5):
        """Add a parent variable that causally influences this variable"""
        self.parents[parent_name] = causal_strength
    
    def add_child(self, child_name, causal_strength=0.5):
        """Add a child variable that this variable causally influences"""
        self.children[child_name] = causal_strength
    
    def update_causal_strength(self, var_name, new_strength, is_parent=True):
        """Update the causal strength for a parent or child relationship"""
        if is_parent and var_name in self.parents:
            self.parents[var_name] = new_strength
        elif not is_parent and var_name in self.children:
            self.children[var_name] = new_strength



# === FILE: asf\layer1_knowledge_substrate\confidence\__init__.py ===

from asf.knowledge_substrate.confidence.bayesian_updater import BayesianConfidenceUpdater


# === FILE: asf\layer1_knowledge_substrate\confidence\bayesian_updater.py ===

import numpy as np
from asf.core.enums import EntityConfidenceState

class BayesianConfidenceUpdater:
    """
    Updates entity confidence using Bayesian inference.
    Philosophical Influence: Bayesian epistemology, Popper's falsifiability
    """
    def __init__(self, prior_alpha=1.0, prior_beta=1.0, decay_rate=0.95):
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.decay_rate = decay_rate
        self.entity_posteriors = {}  # Maps entity_id to (alpha, beta) parameters
        self.context_models = {}  # Context-specific prediction models
    
    def update_confidence(self, entity_id, observation_relevant, context_vector=None):
        """
        Updates entity confidence using Bayesian inference

        Parameters:
        - entity_id: Identifier for the perceptual entity
        - observation_relevant: Boolean indicating if entity was relevant in current context
        - context_vector: Optional vector representing current processing context
        """
        # Get current posterior parameters or initialize with priors
        if entity_id not in self.entity_posteriors:
            self.entity_posteriors[entity_id] = (self.prior_alpha, self.prior_beta)
        
        alpha, beta = self.entity_posteriors[entity_id]
        
        # Apply time decay to previous observations (gradually return to prior)
        alpha = self.prior_alpha + (alpha - self.prior_alpha) * self.decay_rate
        beta = self.prior_beta + (beta - self.prior_beta) * self.decay_rate
        
        # Update with new observation
        if observation_relevant:
            alpha += 1
        else:
            beta += 1
        
        # Store updated posterior parameters
        self.entity_posteriors[entity_id] = (alpha, beta)
        
        # Calculate confidence as mean of Beta distribution
        confidence = alpha / (alpha + beta)
        
        # Update context model if context provided
        if context_vector is not None:
            self._update_context_model(entity_id, context_vector, observation_relevant)
        
        return confidence
    
    def _update_context_model(self, entity_id, context_vector, was_relevant):
        """Update context-based prediction model for the entity"""
        if entity_id not in self.context_models:
            # Initialize model for this entity
            self.context_models[entity_id] = {
                "contexts": [],
                "relevance": []
            }
        
        # Add current observation to model
        self.context_models[entity_id]["contexts"].append(context_vector)
        self.context_models[entity_id]["relevance"].append(1.0 if was_relevant else 0.0)
        
        # Limit model size to prevent unbounded growth
        max_samples = 100
        if len(self.context_models[entity_id]["contexts"]) > max_samples:
            self.context_models[entity_id]["contexts"] = self.context_models[entity_id]["contexts"][-max_samples:]
            self.context_models[entity_id]["relevance"] = self.context_models[entity_id]["relevance"][-max_samples:]
    
    def predict_relevance(self, entity_id, context_vector):
        """
        Predicts entity relevance in a given context
        Returns a probability between 0 and 1
        """
        # Return posterior mean if no context model exists
        if entity_id not in self.entity_posteriors:
            return 0.5  # Prior mean if no data
        
        if entity_id not in self.context_models or not self.context_models[entity_id]["contexts"]:
            # Fall back to posterior mean if no context data
            alpha, beta = self.entity_posteriors[entity_id]
            return alpha / (alpha + beta)
        
        # Use context similarity to predict relevance
        contexts = np.array(self.context_models[entity_id]["contexts"])
        relevance = np.array(self.context_models[entity_id]["relevance"])
        
        # Convert context vector to numpy array if needed
        if not isinstance(context_vector, np.ndarray):
            context_vector = np.array(context_vector)
        
        # Compute similarities between current context and historical contexts
        similarities = self._compute_similarities(context_vector, contexts)
        
        # Weighted average of historical relevance by similarity
        weighted_relevance = np.sum(similarities * relevance) / np.sum(similarities)
        
        # Combine with base rate from posterior
        alpha, beta = self.entity_posteriors[entity_id]
        base_rate = alpha / (alpha + beta)
        
        # Weighted combination of context-based prediction and base rate
        # Higher similarity gives more weight to context-based prediction
        max_similarity = np.max(similarities) if len(similarities) > 0 else 0
        combined_prediction = max_similarity * weighted_relevance + (1 - max_similarity) * base_rate
        
        return combined_prediction
    
    def _compute_similarities(self, query_vector, context_vectors):
        """Compute cosine similarities between query and context vectors"""
        # Handle edge case of empty context vectors
        if len(context_vectors) == 0:
            return np.array([])
        
        # Normalize vectors
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            return np.zeros(len(context_vectors))
        query_vector = query_vector / query_norm
        
        # Compute similarities
        similarities = []
        for ctx in context_vectors:
            ctx_norm = np.linalg.norm(ctx)
            if ctx_norm == 0:
                similarities.append(0)
            else:
                ctx_normalized = ctx / ctx_norm
                similarity = np.dot(query_vector, ctx_normalized)
                similarities.append(max(0, similarity))  # Ensure non-negative
        
        return np.array(similarities)
    
    def get_confidence_state(self, confidence_score):
        """Convert confidence score to EntityConfidenceState"""
        if confidence_score > 0.8:
            return EntityConfidenceState.CANONICAL
        elif confidence_score > 0.5:
            return EntityConfidenceState.PROVISIONAL
        else:
            return EntityConfidenceState.UNVERIFIED



# === FILE: asf\layer1_knowledge_substrate\confidence\predictive_updater.py ===

import numpy as np
from asf.core.enums import EntityConfidenceState
from asf.knowledge_substrate.confidence.bayesian_updater import BayesianConfidenceUpdater

class PredictiveConfidenceUpdater(BayesianConfidenceUpdater):
    """
    Enhanced Bayesian updater incorporating Seth's predictive processing principles.
    Extends standard Bayesian updating with prediction error tracking and adaptive learning rates.
    
    Philosophical Influence: Seth's Predictive Processing, Friston's Free Energy Principle
    """
    
    def __init__(self, prior_alpha=1.0, prior_beta=1.0, decay_rate=0.95):
        super().__init__(prior_alpha, prior_beta, decay_rate)
        # Entity-specific prediction models
        self.prediction_models = {}
        # Track prediction errors to calculate precision
        self.prediction_errors = {}
        # Adaptive learning rates based on prediction error
        self.adaptive_learning_rates = {}
        # Default adaptive range
        self.min_learning_rate = 0.1
        self.max_learning_rate = 0.9
        
    def update_confidence(self, entity_id, observation_relevant, context_vector=None):
        """
        Updates entity confidence using predictive processing principles
        
        Parameters:
        - entity_id: Identifier for the perceptual entity
        - observation_relevant: Boolean indicating if entity was relevant in current context
        - context_vector: Optional vector representing current processing context
        
        Returns: Updated confidence value
        """
        # First, generate prediction based on context
        predicted_relevance = self.predict_relevance(entity_id, context_vector)
        
        # Calculate prediction error (difference between prediction and actual observation)
        prediction_error = abs(float(observation_relevant) - predicted_relevance)
        
        # Track prediction errors for this entity (for precision calculation)
        if entity_id not in self.prediction_errors:
            self.prediction_errors[entity_id] = []
        self.prediction_errors[entity_id].append(prediction_error)
        
        # Limit history size to prevent unbounded growth
        if len(self.prediction_errors[entity_id]) > 20:
            self.prediction_errors[entity_id] = self.prediction_errors[entity_id][-20:]
        
        # Calculate precision (inverse variance of prediction errors)
        if len(self.prediction_errors[entity_id]) > 1:
            precision = 1.0 / (np.var(self.prediction_errors[entity_id]) + 1e-6)
            # Normalize precision to a reasonable range
            precision = min(10.0, precision)  # Cap very high precision
        else:
            precision = 1.0  # Default precision
        
        # Adjust learning rate based on prediction error
        # Higher error = higher learning rate (to learn faster from surprising events)
        # Higher precision = lower learning rate (already well-predicted)
        learning_rate = min(self.max_learning_rate, 
                           max(self.min_learning_rate, 
                               prediction_error * (1.0 / precision)))
        
        # Store the adaptive learning rate
        self.adaptive_learning_rates[entity_id] = learning_rate
        
        # Get current posterior parameters or initialize with priors
        if entity_id not in self.entity_posteriors:
            self.entity_posteriors[entity_id] = (self.prior_alpha, self.prior_beta)
        
        alpha, beta = self.entity_posteriors[entity_id]
        
        # Apply time decay to previous observations (gradually return to prior)
        # Seth's insight: stronger decay for contexts with high uncertainty (low precision)
        decay_modifier = 1.0 / (1.0 + precision * 0.1)  # More decay with lower precision
        effective_decay = self.decay_rate * (1.0 + decay_modifier)
        effective_decay = min(0.99, effective_decay)  # Ensure decay rate is reasonable
        
        alpha = self.prior_alpha + (alpha - self.prior_alpha) * effective_decay
        beta = self.prior_beta + (beta - self.prior_beta) * effective_decay
        
        # Apply Bayesian update with weighted evidence based on learning rate
        if observation_relevant:
            alpha += learning_rate  # Weight evidence by learning rate
        else:
            beta += learning_rate  # Weight evidence by learning rate
        
        # Store updated posterior parameters
        self.entity_posteriors[entity_id] = (alpha, beta)
        
        # Calculate confidence as mean of Beta distribution
        confidence = alpha / (alpha + beta)
        
        # Update context model if context provided
        if context_vector is not None:
            self._update_context_model(entity_id, context_vector, observation_relevant)
        
        return confidence
    
    def predict_relevance(self, entity_id, context_vector=None):
        """
        Predicts entity relevance in a given context
        Incorporates precision weighting for more accurate predictions
        
        Parameters:
        - entity_id: ID of entity to evaluate
        - context_vector: Context for prediction
        
        Returns: Probability between 0 and 1
        """
        # If no context or no prior data, use standard prediction
        if context_vector is None or entity_id not in self.entity_posteriors:
            return super().predict_relevance(entity_id, context_vector)
        
        # Get precision for this entity
        if entity_id in self.prediction_errors and len(self.prediction_errors[entity_id]) > 1:
            precision = 1.0 / (np.var(self.prediction_errors[entity_id]) + 1e-6)
            precision = min(10.0, precision)  # Cap very high precision
        else:
            precision = 1.0  # Default precision
        
        # Get standard prediction
        standard_prediction = super().predict_relevance(entity_id, context_vector)
        
        # For high precision (well-learned predictions), rely more on the model
        # For low precision (uncertain predictions), regress toward prior
        alpha, beta = self.entity_posteriors[entity_id]
        prior_mean = alpha / (alpha + beta)
        
        # Calculate precision-weighted prediction
        # Higher precision = more weight to model prediction
        # Lower precision = more weight to prior
        precision_weight = precision / (precision + 1.0)  # Normalize to 0-1 range
        prediction = (precision_weight * standard_prediction + 
                     (1 - precision_weight) * prior_mean)
        
        return prediction
    
    def get_prediction_metrics(self, entity_id):
        """
        Get prediction-related metrics for an entity
        
        Returns dictionary with metrics including prediction error, precision,
        and adaptive learning rate
        """
        if entity_id not in self.prediction_errors:
            return {
                "avg_prediction_error": None,
                "prediction_precision": None,
                "adaptive_learning_rate": None
            }
        
        errors = self.prediction_errors.get(entity_id, [])
        avg_error = sum(errors) / len(errors) if errors else None
        
        precision = 1.0 / (np.var(errors) + 1e-6) if len(errors) > 1 else None
        precision = min(10.0, precision) if precision is not None else None
        
        learning_rate = self.adaptive_learning_rates.get(entity_id)
        
        return {
            "avg_prediction_error": avg_error,
            "prediction_precision": precision,
            "adaptive_learning_rate": learning_rate
        }



# === FILE: asf\layer1_knowledge_substrate\extraction\base.py ===

# asf/knowledge_substrate/extraction/base.py
import torch
import numpy as np
import joblib
from asf.knowledge_substrate.semiotic.feature_weighter import SemioticFeatureWeighter

class FeatureExtractor:
    """
    Base class for feature extraction from perceptual inputs.
    """
    def __init__(self):
        self.feature_cache = {}  # Cache extracted features
        self.feature_weighter = SemioticFeatureWeighter()
    
    def extract_features(self, input_data):
        """
        Extract features from input data.
        To be implemented by subclasses for specific modalities.
        """
        raise NotImplementedError("Subclasses must implement extract_features")
    
    def get_cache_key(self, input_data):
        """
        Generate a cache key for input data.
        Default implementation uses hash of string representation.
        """
        return hash(str(input_data))
    
    def extract_weighted_features(self, input_data, context=None):
        """
        Extract features and apply semiotic weighting
        """
        # Extract raw features
        raw_features = self.extract_features(input_data)
        
        # Convert to tensor format
        feature_keys = list(raw_features.keys())
        
        # Handle empty features case
        if not feature_keys:
            return {}
        
        try:
            feature_values = torch.stack([
                torch.tensor(raw_features[k], dtype=torch.float32) for k in feature_keys
            ])
            
            # Apply semiotic weighting
            weighted_features, importance_scores = self.feature_weighter.weight_features(
                feature_values,
                context
            )
            
            # Create weighted feature dictionary
            result = {}
            for i, key in enumerate(feature_keys):
                result[key] = {
                    'value': weighted_features[i].numpy(),
                    'importance': importance_scores[i].item()
                }
            
            return result
        except Exception as e:
            print(f"Error in feature weighting: {e}")
            return raw_features  # Fallback to unweighted features
    
    def get_key_features(self, input_data, top_k=5):
        """
        Identifies most important features for the input
        """
        raw_features = self.extract_features(input_data)
        return self.feature_weighter.extract_key_features(raw_features, top_k)



# === FILE: asf\layer1_knowledge_substrate\extraction\image.py ===

# asf/knowledge_substrate/extraction/image.py
import numpy as np
from transformers import ViTModel
from asf.knowledge_substrate.extraction.base import FeatureExtractor

class ImageFeatureExtractor(FeatureExtractor):
    """
    Extracts features from image inputs using Vision Transformer.
    """
    def __init__(self, model_name='vit-base-patch16-224'):
        super().__init__()
        self.model = ViTModel.from_pretrained(model_name)
    
    def extract_features(self, image_input):
        """Extract features from image input"""
        # Check cache first
        cache_key = self.get_cache_key(image_input)
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
        
        # In real implementation, would process image through ViT
        # For prototype, we'll simulate the output
        embedding_dim = 512  # Typical ViT embedding dimension after pooling
        simulated_embedding = np.random.randn(embedding_dim)
        
        # Normalize embedding to unit length
        norm = np.linalg.norm(simulated_embedding)
        if norm > 0:
            simulated_embedding = simulated_embedding / norm
        
        # Create feature dictionary
        features = {
            'image_embedding': simulated_embedding,
            'resolution': (224, 224),  # Simulated resolution
        }
        
        # Cache for future use
        self.feature_cache[cache_key] = features
        
        return features



# === FILE: asf\layer1_knowledge_substrate\extraction\text.py ===

# asf/knowledge_substrate/extraction/text.py
import numpy as np
from transformers import BertModel
from asf.layer1_knowledge_substrate.extraction.base import FeatureExtractor

class TextFeatureExtractor(FeatureExtractor):
    """
    Extracts features from text inputs using BERT.
    """
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        self.model = BertModel.from_pretrained(model_name)
        self.tokenizer = None  # Would initialize tokenizer in real implementation
    
    def extract_features(self, text_input):
        """Extract features from text input"""
        # Check cache first
        cache_key = self.get_cache_key(text_input)
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
        
        # In real implementation, would tokenize and pass through BERT
        # For prototype, we'll simulate the output
        embedding_dim = 768  # BERT base embedding dimension
        simulated_embedding = np.random.randn(embedding_dim)
        
        # Normalize embedding to unit length
        norm = np.linalg.norm(simulated_embedding)
        if norm > 0:
            simulated_embedding = simulated_embedding / norm
        
        # Create feature dictionary
        features = {
            'text_embedding': simulated_embedding,
            'text_length': len(text_input),
            'word_count': len(text_input.split()),
        }
        
        # Cache for future use
        self.feature_cache[cache_key] = features
        
        return features



# === FILE: asf\layer1_knowledge_substrate\linking\__init__.py ===




# === FILE: asf\layer1_knowledge_substrate\linking\entity_gnn.py ===

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

from asf.core.enums import PerceptualInputType

class EntityLinkingGNN:
    """
    Links entities across modalities using Graph Neural Networks.
    """
    def __init__(self, feature_dim=128, hidden_dim=64):
        # Graph Attention Network for entity linking
        self.gat = GATConv(feature_dim, hidden_dim, heads=4)
        self.output_layer = nn.Linear(hidden_dim * 4, feature_dim)
        
        # Relation type embeddings
        self.relation_embeddings = nn.Embedding(10, hidden_dim)  # 10 relation types
        
        # Entity cache to avoid redundant processing
        self.entity_embedding_cache = {}
    
    def create_entity_graph(self, entities, relations):
        """
        Creates graph representation of entities and their relations
        Parameters:
        - entities: List of PerceptualEntity objects
        - relations: List of (source_idx, target_idx, relation_type) tuples
        """
        # Extract entity features
        x = torch.stack([torch.tensor(e.get_feature_vector(), dtype=torch.float32) 
                         for e in entities])
        
        # Create edge index and edge attributes
        edge_index = []
        edge_attr = []
        for source, target, rel_type in relations:
            # Add bidirectional edges for undirected graph
            edge_index.append([source, target])
            edge_index.append([target, source])
            # Edge attributes include relation type embedding
            edge_attr.append(rel_type)
            edge_attr.append(rel_type)
        
        edge_index = torch.tensor(edge_index).t().contiguous()
        edge_attr = torch.tensor(edge_attr)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def update_entity_embeddings(self, entities, relations):
        """
        Updates entity embeddings using graph neural network
        """
        # Create graph data
        graph_data = self.create_entity_graph(entities, relations)
        
        # Apply GAT layer
        x, edge_index, edge_attr = graph_data.x, graph_data.edge_index, graph_data.edge_attr
        
        # Get relation embeddings
        rel_embeddings = self.relation_embeddings(edge_attr)
        
        # Apply GAT with relation-aware attention
        h = self.gat(x, edge_index, edge_attr=rel_embeddings)
        h = F.relu(h)
        
        # Final embeddings
        output = self.output_layer(h)
        
        # Update entity embeddings in cache
        for i, entity in enumerate(entities):
            self.entity_embedding_cache[entity.id] = output[i].detach()
        
        return output
    
    def get_entity_embedding(self, entity_id):
        """
        Retrieves cached embedding for an entity
        """
        return self.entity_embedding_cache.get(entity_id, None)
    
    def find_similar_entities(self, query_entity, all_entities, top_k=5):
        """
        Finds entities similar to query entity across all modalities
        """
        query_embedding = self.get_entity_embedding(query_entity.id)
        if query_embedding is None:
            return []
        
        # Compute similarities
        similarities = []
        for entity in all_entities:
            entity_embedding = self.get_entity_embedding(entity.id)
            if entity_embedding is not None:
                sim = torch.cosine_similarity(query_embedding, entity_embedding, dim=0)
                similarities.append((entity, sim.item()))
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def suggest_cross_modal_links(self, entities, similarity_threshold=0.7):
        """
        Suggests potential cross-modal links between entities
        """
        # Group entities by modality
        entities_by_modality = {}
        for entity in entities:
            if entity.input_type not in entities_by_modality:
                entities_by_modality[entity.input_type] = []
            entities_by_modality[entity.input_type].append(entity)
        
        # Build graph of entities with existing cross-modal links
        existing_relations = []
        for i, entity_i in enumerate(entities):
            for j, entity_j in enumerate(entities):
                if i != j and entity_j.id in entity_i.cross_modal_links:
                    # Find relation type based on modality pair
                    rel_type = self._get_relation_type(entity_i.input_type, entity_j.input_type)
                    existing_relations.append((i, j, rel_type))
        
        # Update embeddings using existing relations
        self.update_entity_embeddings(entities, existing_relations)
        
        # Find potential new links across modalities
        suggested_links = []
        for mod1, entities1 in entities_by_modality.items():
            for mod2, entities2 in entities_by_modality.items():
                if mod1 != mod2:  # Only link across modalities
                    for entity1 in entities1:
                        for entity2 in entities2:
                            # Skip if already linked
                            if entity2.id in entity1.cross_modal_links:
                                continue
                            
                            # Get embeddings
                            emb1 = self.get_entity_embedding(entity1.id)
                            emb2 = self.get_entity_embedding(entity2.id)
                            
                            if emb1 is not None and emb2 is not None:
                                # Calculate similarity
                                similarity = torch.cosine_similarity(emb1, emb2, dim=0).item()
                                
                                # Suggest if above threshold
                                if similarity > similarity_threshold:
                                    suggested_links.append((entity1, entity2, similarity))
        
        # Sort by similarity
        suggested_links.sort(key=lambda x: x[2], reverse=True)
        return suggested_links
    
    def _get_relation_type(self, modality1, modality2):
        """
        Determine relation type based on modality pair
        Returns an integer representing the relation type
        """
        # Create a dictionary mapping modality pairs to relation types
        modality_pairs = {
            (PerceptualInputType.TEXT, PerceptualInputType.IMAGE): 0,
            (PerceptualInputType.IMAGE, PerceptualInputType.TEXT): 0,
            (PerceptualInputType.TEXT, PerceptualInputType.AUDIO): 1,
            (PerceptualInputType.AUDIO, PerceptualInputType.TEXT): 1,
            (PerceptualInputType.IMAGE, PerceptualInputType.AUDIO): 2,
            (PerceptualInputType.AUDIO, PerceptualInputType.IMAGE): 2,
            # Default relation type for other pairs
            "default": 9
        }
        
        # Get the relation type for this modality pair
        key = (modality1, modality2)
        return modality_pairs.get(key, modality_pairs["default"])



# === FILE: asf\layer1_knowledge_substrate\memory\energy_based.py ===

# asf/knowledge_substrate/memory/energy_based.py
import time
import numpy as np
from collections import defaultdict

class EnergyBasedMemoryManager:
    """
    Manages memory allocation and retention using principles inspired by Friston's Free Energy.
    Philosophical Influence: Friston's Free Energy Principle, Bayesian epistemology
    """
    def __init__(self, capacity=1000, decay_rate=0.95, sampling_temp=0.1):
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.sampling_temp = sampling_temp  # Temperature for softmax sampling
        
        # Storage for entities and their energy values
        self.entity_store = {}  # entity_id -> entity
        self.energy_values = {}  # entity_id -> energy value
        
        # Predicted entity patterns (used to calculate surprise)
        self.predicted_patterns = {}  # context_id -> predicted distributions
        
        # Statistics
        self.access_counts = defaultdict(int)
        self.surprise_history = []
    
    def add_entity(self, entity, context=None):
        """
        Add an entity to memory, managing capacity limits.
        Returns True if entity was added, False if rejected.
        """
        entity_id = entity.id
        
        # Calculate initial energy value
        initial_energy = self.calculate_energy(entity, context)
        
        # Check if we need to free up capacity
        if len(self.entity_store) >= self.capacity and entity_id not in self.entity_store:
            # Remove entity with highest energy (least useful)
            self._free_capacity()
        
        # Store entity and its energy
        self.entity_store[entity_id] = entity
        self.energy_values[entity_id] = initial_energy
        
        # Update predicted patterns if context provided
        if context is not None:
            self._update_predictions(entity, context)
        
        return True
    
    def calculate_energy(self, entity, context=None):
        """
        Calculate energy value (lower is better) for an entity.
        Incorporates surprise, complexity, and utility.
        """
        # Start with base energy (favor higher confidence, recent entities)
        base_energy = 1.0 - entity.confidence_score
        
        # Adjust for temporal recency
        temporal_factor = 1.0 - entity.temporal_metadata.get_temporal_relevance()
        
        # Adjust for complexity (features with higher information content cost more)
        complexity = min(1.0, len(entity.features) / 20)  # Normalize by expected max features
        
        # Calculate surprise factor if context is provided
        surprise_factor = 0.0
        if context is not None:
            surprise_factor = self._calculate_surprise(entity, context)
        
        # Combine factors, weighted appropriately
        energy = (
            0.3 * base_energy +      # Base confidence and recency
            0.3 * temporal_factor +  # Temporal factor
            0.2 * complexity +       # Complexity cost
            0.2 * surprise_factor    # Surprise (information gain)
        )
        
        return energy
    
    def _calculate_surprise(self, entity, context):
        """
        Calculate surprise (unexpectedness) of entity in current context.
        Lower values indicate expected patterns, higher values unexpected ones.
        """
        context_id = self._get_context_id(context)
        
        # If no predictions exist for this context, maximum surprise
        if context_id not in self.predicted_patterns:
            return 1.0
        
        predictions = self.predicted_patterns[context_id]
        
        # Check if entity features match predicted patterns
        surprise_values = []
        for feature_name, feature in entity.features.items():
            if feature_name in predictions:
                predicted_value, prediction_confidence = predictions[feature_name]
                
                # Calculate difference between predicted and actual
                if isinstance(feature.value, (int, float, np.number)) and isinstance(predicted_value, (int, float, np.number)):
                    # For numeric values, normalized absolute difference
                    diff = abs(feature.value - predicted_value) / (1.0 + abs(predicted_value))
                    surprise = diff * prediction_confidence  # Weight by prediction confidence
                    surprise_values.append(surprise)
                elif isinstance(feature.value, (list, np.ndarray)) and isinstance(predicted_value, (list, np.ndarray)):
                    # For vectors, cosine distance
                    actual = np.array(feature.value)
                    predicted = np.array(predicted_value)
                    if actual.shape == predicted.shape and actual.size > 0:
                        similarity = np.dot(actual, predicted) / (np.linalg.norm(actual) * np.linalg.norm(predicted))
                        surprise = (1.0 - max(0, similarity)) * prediction_confidence
                        surprise_values.append(surprise)
        
        # Average surprise or default to medium if no features compared
        if surprise_values:
            avg_surprise = sum(surprise_values) / len(surprise_values)
            
            # Track surprise history
            self.surprise_history.append(avg_surprise)
            if len(self.surprise_history) > 100:
                self.surprise_history = self.surprise_history[-100:]
            
            return avg_surprise
        return 0.5  # Default medium surprise
    
    def _update_predictions(self, entity, context):
        """Update predicted patterns based on entity in this context"""
        context_id = self._get_context_id(context)
        
        # Initialize predictions for this context if needed
        if context_id not in self.predicted_patterns:
            self.predicted_patterns[context_id] = {}
        
        # Update predictions for each feature
        predictions = self.predicted_patterns[context_id]
        
        for feature_name, feature in entity.features.items():
            if feature_name not in predictions:
                # New feature, add prediction
                predictions[feature_name] = (feature.value, feature.confidence)
            else:
                # Existing feature, update prediction with exponential moving average
                current_value, current_confidence = predictions[feature_name]
                
                # Update with weighted average
                if isinstance(feature.value, (int, float, np.number)) and isinstance(current_value, (int, float, np.number)):
                    # For numeric values
                    alpha = 0.2  # Learning rate
                    new_value = (1 - alpha) * current_value + alpha * feature.value
                    new_confidence = (1 - alpha) * current_confidence + alpha * feature.confidence
                    predictions[feature_name] = (new_value, new_confidence)
                elif isinstance(feature.value, (list, np.ndarray)) and isinstance(current_value, (list, np.ndarray)):
                    # For vectors, if same shape
                    current_array = np.array(current_value)
                    new_array = np.array(feature.value)
                    if current_array.shape == new_array.shape:
                        alpha = 0.2  # Learning rate
                        updated_value = (1 - alpha) * current_array + alpha * new_array
                        new_confidence = (1 - alpha) * current_confidence + alpha * feature.confidence
                        predictions[feature_name] = (updated_value.tolist(), new_confidence)
    
    def _free_capacity(self):
        """Remove highest energy entities to free capacity"""
        if not self.entity_store:
            return
        
        # Get entities sorted by energy (highest first)
        sorted_entities = sorted(
            self.energy_values.items(),
            key=lambda x: x[1],
            reverse=True  # Highest energy (least useful) first
        )
        
        # Remove highest energy entity
        entity_id, _ = sorted_entities[0]
        self.entity_store.pop(entity_id, None)
        self.energy_values.pop(entity_id, None)
    
    def get_entity(self, entity_id):
        """
        Retrieve entity from memory and update access statistics
        Returns entity if found, None otherwise
        """
        entity = self.entity_store.get(entity_id)
        
        if entity:
            # Update access count
            self.access_counts[entity_id] += 1
            
            # Update temporal metadata
            entity.temporal_metadata.update_access_time()
            
            # Decrease energy (make it more likely to be retained)
            if entity_id in self.energy_values:
                self.energy_values[entity_id] *= self.decay_rate
        
        return entity
    
    def sample_entities(self, context=None, n=5):
        """
        Sample entities using energy-based probability.
        Lower energy entities are more likely to be sampled.
        """
        if not self.entity_store:
            return []
        
        # Get energy values, inverted so lower energy = higher probability
        entity_ids = list(self.entity_store.keys())
        energies = np.array([self.energy_values.get(eid, 1.0) for eid in entity_ids])
        
        # Adjust energies if context provided
        if context is not None:
            context_id = self._get_context_id(context)
            if context_id in self.predicted_patterns:
                # Adjust energies based on context relevance
                for i, entity_id in enumerate(entity_ids):
                    entity = self.entity_store[entity_id]
                    surprise = self._calculate_surprise(entity, context)
                    # Lower surprise = more expected in context = lower energy
                    energies[i] *= (0.5 + 0.5 * surprise)
        
        # Invert energies and apply temperature
        inverted = 1.0 / (energies + 1e-6)  # Add small epsilon to avoid division by zero
        probabilities = np.exp(inverted / self.sampling_temp)
        probabilities = probabilities / np.sum(probabilities)  # Normalize
        
        # Sample without replacement
        try:
            indices = np.random.choice(
                len(entity_ids),
                size=min(n, len(entity_ids)),
                replace=False,
                p=probabilities
            )
            return [self.entity_store[entity_ids[idx]] for idx in indices]
        except:
            # Fallback if sampling fails
            return list(self.entity_store.values())[:n]
    
    def forget_entity(self, entity_id):
        """Explicitly remove an entity from memory"""
        if entity_id in self.entity_store:
            self.entity_store.pop(entity_id)
            self.energy_values.pop(entity_id, None)
            self.access_counts.pop(entity_id, None)
            return True
        return False
    
    def update_all_energies(self, context=None):
        """Recalculate energy values for all entities"""
        for entity_id, entity in self.entity_store.items():
            self.energy_values[entity_id] = self.calculate_energy(entity, context)
    
    def get_memory_statistics(self):
        """Return statistics about memory usage"""
        if not self.entity_store:
            return {
                "total_entities": 0,
                "capacity_used": 0,
                "avg_energy": 0,
                "avg_surprise": 0
            }
        
        return {
            "total_entities": len(self.entity_store),
            "capacity_used": len(self.entity_store) / self.capacity,
            "avg_energy": sum(self.energy_values.values()) / len(self.energy_values),
            "avg_surprise": sum(self.surprise_history) / len(self.surprise_history) if self.surprise_history else 0
        }
    
    def _get_context_id(self, context):
        """Generate a consistent ID for a context object"""
        # Simple implementation using object ID, would be more sophisticated in production
        return str(id(context))



# === FILE: asf\layer1_knowledge_substrate\perception\__init__.py ===

from asf.knowledge_substrate.perception.feature import PerceptualFeature
from asf.knowledge_substrate.perception.entity import PerceptualEntity



# === FILE: asf\layer1_knowledge_substrate\perception\entity.py ===

import numpy as np
from sortedcontainers import SortedDict
from asf.core.enums import EntityConfidenceState
from asf.knowledge_substrate.temporal.adaptive_metadata import AdaptiveTemporalMetadata
from asf.knowledge_substrate.perception.feature import PerceptualFeature

class PerceptualEntity:
    """
    Represents a structured object derived from perceptual input.
    Philosophical influence: Kant's Transcendental Schema
    """
    def __init__(self, entity_id, input_type, initial_features=None):
        self.id = entity_id
        self.input_type = input_type  # PerceptualInputType
        self.features = SortedDict()  # Using SortedDict for efficient feature access
        self.confidence_state = EntityConfidenceState.UNVERIFIED
        self.confidence_score = 0.5  # Initial confidence (0-1)
        self.temporal_metadata = AdaptiveTemporalMetadata()
        self.cross_modal_links = set()  # IDs of linked entities from other modalities
        
        # Add any initial features
        if initial_features:
            for name, value in initial_features.items():
                self.add_feature(name, value)
    
    def add_feature(self, name, value, confidence=0.5):
        """Add a new feature to this entity"""
        self.features[name] = PerceptualFeature(name, value, confidence)
    
    def update_feature(self, name, value=None, confidence=None):
        """Update an existing feature's value and/or confidence"""
        if name in self.features:
            if value is not None:
                self.features[name].value = value
            if confidence is not None:
                self.features[name].update_confidence(confidence)
            return True
        return False
    
    def get_feature(self, name):
        """Get a specific feature by name"""
        return self.features.get(name)
    
    def get_all_features(self):
        """Get all features as a dictionary"""
        return {name: feature for name, feature in self.features.items()}
    
    def update_confidence_state(self, new_state, new_score=None):
        """Update the confidence state of this entity"""
        self.confidence_state = new_state
        if new_score is not None:
            self.confidence_score = new_score
        self.temporal_metadata.update_access_time()
    
    def add_cross_modal_link(self, entity_id):
        """Link this entity to another entity from a different modality"""
        self.cross_modal_links.add(entity_id)
    
    def get_cross_modal_links(self):
        """Get all cross-modal links for this entity"""
        return self.cross_modal_links
    
    def get_feature_vector(self):
        """
        Get a vector representation of all features for similarity calculation
        """
        # Implementation depends on feature types - this is simplified
        feature_values = []
        for name, feature in self.features.items():
            if isinstance(feature.value, (list, np.ndarray)):
                feature_values.extend(feature.value)
            else:
                feature_values.append(feature.value)
        return np.array(feature_values)



# === FILE: asf\layer1_knowledge_substrate\perception\feature.py ===

from asf.knowledge_substrate.temporal.adaptive_metadata import AdaptiveTemporalMetadata

class PerceptualFeature:
    """
    Represents an individual feature extracted from perceptual input.
    """
    def __init__(self, name, value, confidence=0.5):
        self.name = name
        self.value = value  # Could be a scalar, vector, or more complex structure
        self.confidence = confidence
        self.temporal_metadata = AdaptiveTemporalMetadata()
    
    def update_confidence(self, new_confidence):
        """Update confidence score for this feature"""
        self.confidence = new_confidence
        self.temporal_metadata.update_access_time()
    
    def get_relevance(self):
        """Calculate overall relevance based on confidence and temporal factors"""
        temporal_relevance = self.temporal_metadata.get_temporal_relevance()
        return self.confidence * temporal_relevance



# === FILE: asf\layer1_knowledge_substrate\semiotic\feature_weighter.py ===

import torch
import torch.nn as nn
import torch.nn.functional as F

class SemioticFeatureWeighter:
    """
    Weights features based on their semiotic significance using transformer attention.
    Philosophical Influence: Saussure's semiotics, Peirce's sign theory
    """
    def __init__(self, feature_dim=768, num_heads=8):
        # Multi-head self-attention for feature weighting
        self.attention = nn.MultiheadAttention(feature_dim, num_heads)
        # Context-aware projection
        self.context_projection = nn.Linear(feature_dim, feature_dim)
        # Feature importance predictor
        self.importance_predictor = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def extract_attention_weights(self, model_output, input_tokens):
        """
        Extracts attention weights from transformer models like BERT or ViT
        """
        # Implementation depends on specific model architecture
        # For BERT:
        if hasattr(model_output, 'attentions') and model_output.attentions:
            # Get weights from last layer
            attention_weights = model_output.attentions[-1]
            # Average across attention heads
            avg_attention = torch.mean(attention_weights, dim=1)
            # Return the attention weights
            return avg_attention
        # For ViT:
        elif hasattr(model_output, 'attention_weights'):
            return model_output.attention_weights
        # Fallback if attention weights aren't available
        return torch.ones((len(input_tokens), len(input_tokens)))
    
    def weight_features(self, features, context_vector=None):
        """
        Applies semiotic weighting to features based on context
        
        Parameters:
        - features: Tensor of feature vectors [num_features, feature_dim]
        - context_vector: Optional context representation
        """
        # Convert features to proper shape for attention
        features_t = features.unsqueeze(1)  # [num_features, 1, feature_dim]
        
        # If context provided, use it to condition attention
        if context_vector is not None:
            # Project context to feature space
            context_key = self.context_projection(context_vector).unsqueeze(0)
            
            # Use context as query for attention
            attn_output, attn_weights = self.attention(
                context_key,
                features_t,
                features_t
            )
            importance_scores = attn_weights.squeeze()
        else:
            # Self-attention among features
            attn_output, attn_weights = self.attention(
                features_t,
                features_t,
                features_t
            )
            # Calculate importance scores from attention
            importance_scores = attn_weights.mean(dim=1).squeeze()
        
        # Normalize importance scores
        importance_scores = F.softmax(importance_scores, dim=0)
        
        # Apply importance weighting to features
        weighted_features = features * importance_scores.unsqueeze(1)
        
        return weighted_features, importance_scores
    
    def extract_key_features(self, feature_dict, top_k=5):
        """
        Identifies most semantically relevant features
        """
        # Convert feature dictionary to tensors
        feature_keys = list(feature_dict.keys())
        feature_values = torch.stack([
            torch.tensor(feature_dict[k].value, dtype=torch.float32)
            for k in feature_keys if hasattr(feature_dict[k], 'value')
        ])
        
        # Apply weighting
        _, importance_scores = self.weight_features(feature_values)
        
        # Get top-k features
        _, top_indices = torch.topk(importance_scores, min(top_k, len(feature_keys)))
        
        return [feature_keys[i] for i in top_indices]



# === FILE: asf\layer1_knowledge_substrate\temporal\__init__.py ===

from asf.knowledge_substrate.temporal.adaptive_metadata import AdaptiveTemporalMetadata
from asf.knowledge_substrate.temporal.sequence import TemporalSequence
from asf.knowledge_substrate.temporal.pattern_detector import TemporalPatternDetector
from asf.knowledge_substrate.temporal.processing_engine import TemporalProcessingEngine
from asf.knowledge_substrate.temporal.predictive_engine import PredictiveTemporalEngine



# === FILE: asf\layer1_knowledge_substrate\temporal\adaptive_metadata.py ===

import time
import math

class AdaptiveTemporalMetadata:
    """
    Manages temporal aspects of knowledge entities with dynamic decay rates.
    Philosophical Influence: Bergson's time-consciousness and Heidegger's Dasein
    """
    def __init__(self, context_type="default"):
        self.creation_time = time.time()
        self.last_access_time = self.creation_time
        self.access_count = 1
        # Half-life constants for different context types (in seconds)
        self.half_lives = {
            "default": 86400,  # 24 hours
            "critical": 604800,  # 1 week
            "ephemeral": 3600  # 1 hour
        }
        self.context_type = context_type
    
    def update_access_time(self):
        """Update last access time and increment access counter"""
        self.last_access_time = time.time()
        self.access_count += 1
    
    def get_temporal_relevance(self) -> float:
        """
        Calculate temporal relevance based on adaptive logarithmic decay.
        Returns a value between 0 and 1 where 1 is most relevant.
        """
        current_time = time.time()
        time_since_creation = current_time - self.creation_time
        time_since_access = current_time - self.last_access_time
        # Adjust half-life based on access frequency
        access_factor = math.log2(max(2, self.access_count))
        adjusted_half_life = self.half_lives[self.context_type] * access_factor
        # Calculate decay using adapted half-life formula
        decay = math.exp(-time_since_access / adjusted_half_life)
        return max(0.01, min(1.0, decay))  # Ensure value between 0.01 and 1
    
    def set_context_type(self, context_type):
        """Update the context type to change temporal relevance calculation"""
        if context_type in self.half_lives:
            self.context_type = context_type



# === FILE: asf\layer1_knowledge_substrate\temporal\pattern_detector.py ===

import torch
import torch.nn as nn

class TemporalPatternDetector:
    """
    Detects patterns in temporal sequences using recurrent neural networks.
    """
    def __init__(self, input_size, hidden_size=64):
        # GRU-based recurrent model for temporal pattern detection
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.pattern_classifier = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size
    
    def forward(self, sequence_tensor):
        """Process temporal sequence to detect patterns"""
        # sequence_tensor shape: [batch_size, seq_length, input_size]
        output, hidden = self.rnn(sequence_tensor)
        # Use last hidden state for pattern classification
        pattern_score = torch.sigmoid(self.pattern_classifier(hidden.squeeze(0)))
        return pattern_score, hidden
    
    def detect_patterns(self, sequence_data):
        """
        Analyze sequence data to detect significant temporal patterns
        Returns pattern score and pattern type
        """
        # Convert sequence to tensor
        sequence_tensor = torch.tensor(sequence_data, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pattern_score, _ = self.forward(sequence_tensor)
        return pattern_score.item()



# === FILE: asf\layer1_knowledge_substrate\temporal\predictive_engine.py ===

import time
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from asf.knowledge_substrate.temporal.processing_engine import TemporalProcessingEngine

class PredictiveTemporalEngine(TemporalProcessingEngine):
    """
    Enhanced temporal processing engine incorporating Seth's predictive processing principles.
    
    Extends the base TemporalProcessingEngine with:
    - Explicit prediction of future events
    - Prediction error tracking
    - Precision-weighted updates
    - Counterfactual temporal simulations
    
    Philosophical influences: Seth's predictive processing, Friston's Free Energy
    """
    
    def __init__(self):
        super().__init__()
        # Track prediction errors
        self.prediction_errors = defaultdict(list)
        # Track predictions and outcomes
        self.predictions = {}
        self.prediction_outcomes = []
        # Precision values (inverse variance of prediction errors)
        self.precision_values = {}
        # Learning rates (adaptive based on prediction error)
        self.learning_rates = defaultdict(lambda: 0.3)  # Default learning rate
        # Counterfactual simulations
        self.counterfactual_simulations = []
        
        # Pytorch model for temporal prediction (optional, more sophisticated prediction)
        self.use_neural_prediction = False  # Toggle for neural prediction
        if self.use_neural_prediction:
            self.predictor = TemporalPredictor(input_size=64, hidden_size=128)
    
    def predict_next_events(self, entity_id, sequence_type, time_horizon=3600):
        """
        Predict future events based on temporal patterns.
        
        Args:
            entity_id: Entity to predict for
            sequence_type: Type of sequence to predict
            time_horizon: How far into the future to predict (seconds)
            
        Returns:
            List of predicted events with timestamps
        """
        if entity_id not in self.sequences or sequence_type not in self.sequences[entity_id]:
            return []
        
        sequence = self.sequences[entity_id][sequence_type]
        recent_events = sequence.get_events_in_window(window_size=time_horizon)
        
        if len(recent_events) < 3:  # Need minimum data for prediction
            return []
        
        # Extract timestamps for interval analysis
        timestamps = list(sequence.timestamps)
        if len(timestamps) < 2:
            return []
            
        # Analyze temporal patterns
        event_intervals = []
        for i in range(1, len(timestamps)):
            interval = timestamps[i] - timestamps[i-1]
            event_intervals.append(interval)
        
        if not event_intervals:
            return []
        
        # Calculate mean interval and variance (for precision)
        mean_interval = np.mean(event_intervals)
        interval_variance = np.var(event_intervals)
        
        # More regular patterns = higher precision = more confident predictions
        precision = 1.0 / (interval_variance + 1e-6)
        
        # Cap precision to reasonable range
        precision = min(10.0, precision)
        
        # Store precision for this sequence
        self.precision_values.setdefault(entity_id, {})
        self.precision_values[entity_id][sequence_type] = precision
        
        # Last event time
        last_time = timestamps[-1] if timestamps else time.time()
        
        # Predict future events based on pattern
        predictions = []
        current_time = time.time()
        next_time = last_time + mean_interval
        
        # Use neural prediction if enabled and enough data
        if self.use_neural_prediction and len(recent_events) >= 5:
            neural_predictions = self._neural_predict(entity_id, sequence_type, time_horizon)
            if neural_predictions:
                # Store predictions for later evaluation
                prediction_id = f"{entity_id}_{sequence_type}_{int(time.time())}"
                self.predictions[prediction_id] = {
                    "entity_id": entity_id,
                    "sequence_type": sequence_type,
                    "predictions": neural_predictions,
                    "timestamp": time.time(),
                    "method": "neural"
                }
                return neural_predictions
        
        # Fall back to statistical prediction
        while next_time <= current_time + time_horizon:
            # Create prediction with confidence based on precision
            confidence = min(0.9, np.tanh(precision) * 0.8)
            
            predictions.append({
                "predicted_time": next_time,
                "confidence": confidence,
                "time_from_now": next_time - current_time
            })
            
            next_time += mean_interval
        
        # Store predictions for later evaluation
        if predictions:
            prediction_id = f"{entity_id}_{sequence_type}_{int(time.time())}"
            self.predictions[prediction_id] = {
                "entity_id": entity_id,
                "sequence_type": sequence_type,
                "predictions": predictions,
                "timestamp": time.time(),
                "method": "statistical"
            }
        
        return predictions
    
    def evaluate_predictions(self, entity_id, sequence_type, current_time=None):
        """
        Evaluate accuracy of past predictions for this entity/sequence.
        Updates prediction errors and precision values.
        
        Returns evaluation metrics dictionary.
        """
        if current_time is None:
            current_time = time.time()
            
        # Find predictions for this entity/sequence
        relevant_predictions = []
        for pred_id, prediction in self.predictions.items():
            if (prediction["entity_id"] == entity_id and 
                prediction["sequence_type"] == sequence_type and
                prediction["timestamp"] < current_time):
                relevant_predictions.append((pred_id, prediction))
        
        if not relevant_predictions:
            return {"evaluated": False, "reason": "no_predictions"}
            
        # Get actual events that occurred since predictions were made
        if entity_id not in self.sequences or sequence_type not in self.sequences[entity_id]:
            return {"evaluated": False, "reason": "sequence_not_found"}
            
        sequence = self.sequences[entity_id][sequence_type]
        
        # Results for all evaluated predictions
        evaluation_results = []
        
        for pred_id, prediction in relevant_predictions:
            # Skip if already evaluated
            if "evaluated" in prediction and prediction["evaluated"]:
                continue
                
            pred_time = prediction["timestamp"]
            pred_events = prediction["predictions"]
            
            # Get actual events that occurred after prediction was made
            actual_timestamps = [ts for ts in sequence.timestamps if ts > pred_time]
            
            # Skip if no actual events to compare against
            if not actual_timestamps:
                continue
                
            # For each predicted event, find closest actual event
            prediction_errors = []
            
            for pred_event in pred_events:
                pred_timestamp = pred_event["predicted_time"]
                
                # Skip predicted events in the future
                if pred_timestamp > current_time:
                    continue
                    
                # Find closest actual event
                closest_actual = min(actual_timestamps, key=lambda x: abs(x - pred_timestamp))
                
                # Calculate error (time difference in seconds)
                error = abs(closest_actual - pred_timestamp)
                
                # Normalize error relative to prediction horizon
                horizon = max(pred_event["time_from_now"], 60)  # At least 1 minute
                normalized_error = min(1.0, error / horizon)
                
                prediction_errors.append(normalized_error)
            
            # Skip if no predictions were evaluated
            if not prediction_errors:
                continue
                
            # Calculate average error
            avg_error = sum(prediction_errors) / len(prediction_errors)
            
            # Update prediction error tracking
            error_key = f"{entity_id}_{sequence_type}"
            self.prediction_errors[error_key].append(avg_error)
            
            # Limit history size
            if len(self.prediction_errors[error_key]) > 20:
                self.prediction_errors[error_key] = self.prediction_errors[error_key][-20:]
            
            # Update precision value
            if len(self.prediction_errors[error_key]) > 1:
                error_variance = np.var(self.prediction_errors[error_key])
                precision = 1.0 / (error_variance + 1e-6)
                precision = min(10.0, precision)  # Cap precision
                
                # Store updated precision
                self.precision_values.setdefault(entity_id, {})
                self.precision_values[entity_id][sequence_type] = precision
            
            # Mark prediction as evaluated
            prediction["evaluated"] = True
            prediction["avg_error"] = avg_error
            prediction["evaluation_time"] = current_time
            
            # Add to results
            evaluation_results.append({
                "prediction_id": pred_id,
                "avg_error": avg_error,
                "num_predictions": len(pred_events),
                "num_evaluated": len(prediction_errors)
            })
            
            # Record outcome for learning
            self.prediction_outcomes.append({
                "entity_id": entity_id,
                "sequence_type": sequence_type,
                "prediction_method": prediction["method"],
                "avg_error": avg_error,
                "precision": self.precision_values.get(entity_id, {}).get(sequence_type, 1.0),
                "timestamp": current_time
            })
        
        # Return evaluation summary
        if evaluation_results:
            return {
                "evaluated": True,
                "num_predictions": len(evaluation_results),
                "avg_error": sum(r["avg_error"] for r in evaluation_results) / len(evaluation_results),
                "precision": self.precision_values.get(entity_id, {}).get(sequence_type, 1.0),
                "details": evaluation_results
            }
        else:
            return {"evaluated": False, "reason": "no_evaluable_predictions"}
    
    def add_event(self, entity_id, sequence_type, event_data):
        """
        Add an event and update predictions. 
        Overrides base method to include prediction evaluation.
        """
        # First call parent implementation
        result = super().add_event(entity_id, sequence_type, event_data)
        
        # Then evaluate existing predictions
        self.evaluate_predictions(entity_id, sequence_type)
        
        # Return original result
        return result
    
    def generate_counterfactual_sequence(self, entity_id, sequence_type, 
                                        modification, time_horizon=3600):
        """
        Generate a counterfactual sequence by modifying existing temporal patterns.
        
        Args:
            entity_id: Entity to generate counterfactual for
            sequence_type: Type of sequence
            modification: Dictionary specifying the modification 
                (e.g., {"interval_factor": 0.5} to make events twice as frequent)
            time_horizon: Time horizon for counterfactual (seconds)
            
        Returns:
            Counterfactual sequence of events
        """
        if entity_id not in self.sequences or sequence_type not in self.sequences[entity_id]:
            return []
            
        sequence = self.sequences[entity_id][sequence_type]
        recent_events = sequence.get_events_in_window(window_size=time_horizon)
        
        if len(recent_events) < 3:  # Need minimum data
            return []
            
        # Extract timestamps
        timestamps = list(sequence.timestamps)
        if len(timestamps) < 2:
            return []
            
        # Create copy of events for counterfactual
        cf_events = recent_events.copy()
        cf_timestamps = timestamps.copy()
        
        # Apply modifications
        if "interval_factor" in modification:
            # Change frequency of events
            factor = modification["interval_factor"]
            
            # Calculate new intervals
            new_timestamps = [timestamps[0]]  # Keep first timestamp
            
            for i in range(1, len(timestamps)):
                interval = timestamps[i] - timestamps[i-1]
                # Apply factor to interval
                new_interval = interval * factor
                new_timestamps.append(new_timestamps[-1] + new_interval)
                
            cf_timestamps = new_timestamps
        
        elif "remove_pattern" in modification:
            # Remove a specific pattern from sequence
            pattern = modification["remove_pattern"]
            
            # Filter events not matching pattern
            filtered_events = []
            filtered_timestamps = []
            
            for i, event in enumerate(cf_events):
                if not self._matches_pattern(event, pattern):
                    filtered_events.append(event)
                    filtered_timestamps.append(cf_timestamps[i])
                    
            cf_events = filtered_events
            cf_timestamps = filtered_timestamps
        
        # Record counterfactual simulation
        cf_id = f"cf_{entity_id}_{sequence_type}_{int(time.time())}"
        self.counterfactual_simulations.append({
            "id": cf_id,
            "entity_id": entity_id,
            "sequence_type": sequence_type,
            "modification": modification,
            "original_count": len(recent_events),
            "counterfactual_count": len(cf_events),
            "timestamp": time.time()
        })
        
        # Return counterfactual sequence
        return list(zip(cf_events, cf_timestamps))
    
    def _neural_predict(self, entity_id, sequence_type, time_horizon):
        """
        Use neural network for more sophisticated temporal prediction.
        
        Returns list of predicted events with timestamps and confidence.
        """
        if not self.use_neural_prediction:
            return []
            
        sequence = self.sequences[entity_id][sequence_type]
        recent_events = sequence.get_events_in_window(time_horizon)
        
        if len(recent_events) < 5:  # Need more data for neural prediction
            return []
            
        # Extract features and timestamps
        timestamps = list(sequence.timestamps)[-len(recent_events):]
        
        # Convert to intervals
        intervals = []
        for i in range(1, len(timestamps)):
            intervals.append(timestamps[i] - timestamps[i-1])
            
        if not intervals:
            return []
            
        # Create input tensor
        # For simplicity, just use intervals as features
        # In real implementation, would include event features
        sequence_tensor = torch.tensor(intervals, dtype=torch.float32).unsqueeze(0)
        
        # Generate predictions
        with torch.no_grad():
            predicted_intervals, confidence = self.predictor(sequence_tensor)
            
        # Convert back to timestamps
        last_time = timestamps[-1]
        current_time = time.time()
        
        predictions = []
        for i, interval in enumerate(predicted_intervals[0]):
            next_time = last_time + interval.item()
            last_time = next_time
            
            if next_time <= current_time + time_horizon:
                predictions.append({
                    "predicted_time": next_time,
                    "confidence": min(0.95, confidence[0, i].item()),
                    "time_from_now": next_time - current_time
                })
                
        return predictions
    
    def _matches_pattern(self, event, pattern):
        """Check if event matches specified pattern"""
        # Implementation depends on event and pattern representation
        # This is a simplified placeholder
        if hasattr(event, 'get_feature_vector') and hasattr(pattern, 'get_feature_vector'):
            event_vec = event.get_feature_vector()
            pattern_vec = pattern.get_feature_vector()
            
            if len(event_vec) == len(pattern_vec):
                similarity = np.dot(event_vec, pattern_vec) / (
                    np.linalg.norm(event_vec) * np.linalg.norm(pattern_vec))
                return similarity > 0.8
        
        return False


class TemporalPredictor(nn.Module):
    """Neural network model for temporal prediction"""
    
    def __init__(self, input_size=64, hidden_size=128):
        super().__init__()
        
        # RNN for sequence modeling
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )
        
        # Prediction layers
        self.interval_predictor = nn.Linear(hidden_size * 2, 5)  # Predict next 5 intervals
        self.confidence_predictor = nn.Linear(hidden_size * 2, 5)  # Confidence for each prediction
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, seq_length, input_size]
            
        Returns:
            predicted_intervals: [batch_size, 5]
            confidence: [batch_size, 5]
        """
        # Process sequence
        output, hidden = self.rnn(x)
        
        # Use last hidden state
        last_hidden = output[:, -1, :]
        
        # Predict intervals
        intervals = self.interval_predictor(last_hidden)
        
        # Ensure intervals are positive
        intervals = torch.abs(intervals)
        
        # Predict confidence
        confidence = torch.sigmoid(self.confidence_predictor(last_hidden))
        
        return intervals, confidence



# === FILE: asf\layer1_knowledge_substrate\temporal\processing_engine.py ===

import time
from asf.knowledge_substrate.temporal.sequence import TemporalSequence
from asf.knowledge_substrate.temporal.pattern_detector import TemporalPatternDetector

class TemporalProcessingEngine:
    """
    Manages temporal processing across different modalities and timescales.
    """
    def __init__(self):
        self.sequences = {}  # Maps entity_ids to their temporal sequences
        self.pattern_detectors = {}  # Different detectors for different sequence types
        self.temporal_context = {}  # Maintains context across processing cycles
    
    def register_sequence(self, entity_id, sequence_type):
        """Register a new temporal sequence for tracking"""
        if entity_id not in self.sequences:
            self.sequences[entity_id] = {}
        self.sequences[entity_id][sequence_type] = TemporalSequence()
    
    def add_event(self, entity_id, sequence_type, event_data):
        """Add an event to an entity's temporal sequence"""
        if entity_id in self.sequences and sequence_type in self.sequences[entity_id]:
            self.sequences[entity_id][sequence_type].add_event(event_data)
            # Check for patterns when new events are added
            return self.check_for_patterns(entity_id, sequence_type)
        return None
    
    def check_for_patterns(self, entity_id, sequence_type):
        """Analyze sequence for temporal patterns"""
        if sequence_type not in self.pattern_detectors:
            # Initialize appropriate detector if needed
            feature_dim = self._get_feature_dimension(sequence_type)
            self.pattern_detectors[sequence_type] = TemporalPatternDetector(feature_dim)
        
        # Get recent events as feature vectors
        sequence = self.sequences[entity_id][sequence_type]
        recent_events = sequence.get_events_in_window(window_size=3600)  # Last hour
        if len(recent_events) < 3:  # Need minimum sequence length
            return None
        
        # Prepare data for pattern detection
        sequence_data = [self._prepare_event_features(e) for e in recent_events]
        
        # Detect patterns
        pattern_score = self.pattern_detectors[sequence_type].detect_patterns(sequence_data)
        if pattern_score > 0.7:  # Threshold for pattern detection
            return {
                "entity_id": entity_id,
                "sequence_type": sequence_type,
                "pattern_score": pattern_score,
                "event_count": len(recent_events)
            }
        return None
    
    def _get_feature_dimension(self, sequence_type):
        """Return feature dimension based on sequence type"""
        # Different sequence types may have different feature dimensions
        feature_dimensions = {
            "text": 768,
            "image": 512,
            "audio": 256,
            "numerical": 64,
            "default": 128
        }
        return feature_dimensions.get(sequence_type, feature_dimensions["default"])
    
    def _prepare_event_features(self, event):
        """Extract feature vector from event for pattern detection"""
        # Implementation depends on event type
        if hasattr(event, 'get_feature_vector'):
            return event.get_feature_vector()
        return event  # Assume event is already a feature vector
    
    def maintain_temporal_context(self, context_id, context_data):
        """Update temporal context for a given context ID"""
        self.temporal_context[context_id] = {
            "data": context_data,
            "timestamp": time.time()
        }
    
    def get_temporal_context(self, context_id, max_age=None):
        """Retrieve temporal context if not expired"""
        if context_id in self.temporal_context:
            context = self.temporal_context[context_id]
            if max_age is None or time.time() - context["timestamp"] <= max_age:
                return context["data"]
        return None



# === FILE: asf\layer1_knowledge_substrate\temporal\sequence.py ===

import time
from collections import deque
import numpy as np

class TemporalSequence:
    """
    Represents a sequence of temporal events for pattern detection.
    Optimized for performance with efficient data structures and operations.
    """
    def __init__(self, max_length=100):
        self.events = deque(maxlen=max_length)
        self.timestamps = deque(maxlen=max_length)
        self._max_length = max_length
        # Cache for window lookups
        self._last_window_time = 0
        self._last_window_size = 0
        self._last_window_result = []
    
    def add_event(self, event, timestamp=None):
        """Add event to sequence with optional timestamp"""
        if timestamp is None:
            timestamp = time.time()
        
        # Local references for performance
        events = self.events
        timestamps = self.timestamps
        
        events.append(event)
        timestamps.append(timestamp)
        
        # Invalidate cache when new events are added
        self._last_window_time = 0
        
        return len(events)  # Return new size for convenience
    
    def get_events_in_window(self, window_size):
        """Return events within the specified time window from now"""
        current_time = time.time()
        
        # Check if we can use cached result
        if (current_time - self._last_window_time < 0.1 and  # Cache valid for 100ms
            window_size == self._last_window_size and
            self._last_window_result):
            return self._last_window_result
        
        # Local references for performance
        timestamps = self.timestamps
        events = self.events
        
        # Optimize: use binary search to find cutoff index
        cutoff_time = current_time - window_size
        
        # If sequence is empty
        if not timestamps:
            return []
        
        # If all events are within window
        if timestamps[0] >= cutoff_time:
            result = list(events)
        else:
            # Find index using binary search approximation
            left, right = 0, len(timestamps) - 1
            cutoff_idx = 0
            
            while left <= right:
                mid = (left + right) // 2
                if timestamps[mid] < cutoff_time:
                    left = mid + 1
                    cutoff_idx = left  # First index >= cutoff_time
                else:
                    right = mid - 1
            
            # Extract events from cutoff_idx to end
            result = list(events)[cutoff_idx:]
        
        # Cache the result
        self._last_window_time = current_time
        self._last_window_size = window_size
        self._last_window_result = result
        
        return result
    
    def get_latest_events(self, n=1):
        """Get the n most recent events"""
        if not self.events:
            return []
        
        if n >= len(self.events):
            return list(self.events)
        
        return list(self.events)[-n:]
    
    def clear(self):
        """Clear all events and timestamps"""
        self.events.clear()
        self.timestamps.clear()
        self._last_window_time = 0
        self._last_window_result = []
        
    def __len__(self):
        """Return the number of events in sequence"""
        return len(self.events)
    
    def get_statistics(self):
        """Return statistical information about the sequence"""
        if not self.timestamps:
            return {
                "count": 0,
                "oldest": None,
                "newest": None,
                "time_span": 0,
                "avg_interval": 0
            }
        
        ts_array = np.array(self.timestamps)
        oldest = float(ts_array.min())
        newest = float(ts_array.max())
        
        return {
            "count": len(self.timestamps),
            "oldest": oldest,
            "newest": newest,
            "time_span": newest - oldest,
            "avg_interval": (newest - oldest) / max(1, len(self.timestamps) - 1)
        }



# === FILE: asf\layer2_autopoietic_maintanance\__init__.py ===

from asf.symbolic_formation.enums import NonlinearityOrder, SymbolConfidenceState
from asf.symbolic_formation.potentials import SymbolicPotential
from asf.symbolic_formation.symbol import SymbolElement
from asf.symbolic_formation.network import SparseTensorSymbolNetwork
from asf.symbolic_formation.autocatalytic import AutocatalyticNetwork, NonlinearityOrderTracker
from asf.symbolic_formation.operational_closure import OperationalClosure
from asf.symbolic_formation.recognition import SymbolRecognizer
from asf.symbolic_formation.symbolic_layer import SymbolicFormationLayer

# Seth's Data Paradox enhancements
from asf.symbolic_formation.predictive_potentials import PredictiveSymbolicPotential
from asf.symbolic_formation.predictive_symbol import PredictiveSymbolElement
from asf.symbolic_formation.predictive_recognition import PredictiveSymbolRecognizer
from asf.symbolic_formation.counterfactual_network import CounterfactualAutocatalyticNetwork
from asf.symbolic_formation.predictive_processor import SymbolicPredictiveProcessor
from asf.symbolic_formation.predictive_layer import PredictiveSymbolicFormationLayer



# === FILE: asf\layer2_autopoietic_maintanance\autocatalytic.py ===

import numpy as np
import time
import uuid
from collections import defaultdict
from typing import Dict, List, Optional
import torch

from asf.symbolic_formation.enums import NonlinearityOrder
from asf.symbolic_formation.symbol import SymbolElement
from asf.symbolic_formation.network import SparseTensorSymbolNetwork

class NonlinearityOrderTracker:
    """
    Tracks and optimizes order of nonlinearity in symbol transformations.
    Enhanced with learning capabilities for adaptive nonlinearity classification.
    """
    def __init__(self):
        self.symbol_nonlinearity: Dict[str, NonlinearityOrder] = {}
        self.potential_nonlinearity: Dict[str, NonlinearityOrder] = {}
        self.transformation_complexity: Dict[tuple, NonlinearityOrder] = {}
        # Phase 2 enhancement: learning statistics
        self.transformation_observations = defaultdict(list)
        self.learning_rate = 0.1
    
    def register_symbol(self, symbol_id: str, nonlinearity: NonlinearityOrder = NonlinearityOrder.LINEAR) -> None:
        """Register a symbol with its nonlinearity order."""
        self.symbol_nonlinearity[symbol_id] = nonlinearity
    
    def register_potential(self, potential_id: str, nonlinearity: NonlinearityOrder = NonlinearityOrder.LINEAR) -> None:
        """Register a potential with its nonlinearity order."""
        self.potential_nonlinearity[potential_id] = nonlinearity
    
    def register_transformation(self, source_id: str, target_id: str, nonlinearity: NonlinearityOrder) -> None:
        """Register a transformation between symbols/potentials with its nonlinearity order."""
        self.transformation_complexity[(source_id, target_id)] = nonlinearity
        # Phase 2 enhancement: record observation for learning
        self.transformation_observations[(source_id, target_id)].append(nonlinearity)
        self._update_transformation_complexity(source_id, target_id)
    
    def _update_transformation_complexity(self, source_id: str, target_id: str) -> None:
        """Update transformation complexity based on observations."""
        observations = self.transformation_observations[(source_id, target_id)]
        if len(observations) < 3:
            return  # Not enough data
            
        # Count occurrences of each nonlinearity order
        order_counts = {}
        for obs in observations:
            if obs.value in order_counts:
                order_counts[obs.value] += 1
            else:
                order_counts[obs.value] = 1
                
        # Find most common nonlinearity order
        most_common_value = max(order_counts.items(), key=lambda x: x[1])[0]
        most_common_order = NonlinearityOrder(most_common_value)
        
        # Update complexity with some inertia (learning rate)
        current_order = self.transformation_complexity.get((source_id, target_id), NonlinearityOrder.LINEAR)
        
        # If most common differs from current, adapt with learning rate
        if most_common_order.value != current_order.value:
            # Move toward most common based on learning rate
            if most_common_order.value > current_order.value:
                # Increase nonlinearity
                new_value = min(
                    NonlinearityOrder.COMPOSITIONAL.value,
                    current_order.value + round(self.learning_rate * (most_common_order.value - current_order.value))
                )
            else:
                # Decrease nonlinearity
                new_value = max(
                    NonlinearityOrder.LINEAR.value,
                    current_order.value - round(self.learning_rate * (current_order.value - most_common_order.value))
                )
                
            self.transformation_complexity[(source_id, target_id)] = NonlinearityOrder(new_value)

class AutocatalyticNetwork:
    """
    Implements autocatalytic networks where symbols help create other symbols.
    Optimized with nonlinearity order tracking for better generalization.
    Enhanced with template adaptation learning.
    """
    def __init__(self, nonlinearity_tracker: NonlinearityOrderTracker):
        self.catalytic_relations: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.production_templates: Dict[str, Dict[str, float]] = {}
        self.template_nonlinearity: Dict[str, NonlinearityOrder] = {}
        self.nonlinearity_tracker = nonlinearity_tracker
        
        # Sparse tensor representation for large networks
        self._tensor_network = SparseTensorSymbolNetwork()
        # Adaptive template selection based on past success
        self._template_success_rate: Dict[str, float] = defaultdict(float)
        self._template_usage_count: Dict[str, int] = defaultdict(int)
        # Phase 2 enhancement: template adaptation
        self._template_adaption_rate = 0.05
        self._template_evolution_history = []
    
    def add_catalytic_relation(self, catalyst_id: str, target_id: str, strength: float) -> None:
        """Add a catalytic relationship where one symbol helps produce another."""
        self.catalytic_relations[catalyst_id][target_id] = strength
        # Add to tensor network
        self._tensor_network.add_relation(
            catalyst_id, target_id, relation_type=1, strength=strength
        )
    
    def add_production_template(self, template_id: str, required_elements: Dict[str, float],
                              nonlinearity: NonlinearityOrder = NonlinearityOrder.LINEAR) -> None:
        """Add a template for producing new symbolic elements."""
        self.production_templates[template_id] = required_elements
        self.template_nonlinearity[template_id] = nonlinearity
        # Register with nonlinearity tracker
        self.nonlinearity_tracker.register_transformation(
            "template", template_id, nonlinearity
        )
    
    def generate_symbols(self, existing_symbols: Dict[str, SymbolElement],
                       perceptual_inputs: Dict[str, float],
                       threshold: float = 0.5) -> Dict[str, SymbolElement]:
        """
        Generate new symbols based on existing symbols and perceptual inputs.
        Uses tensor-based propagation and prioritizes templates with lower nonlinearity.
        """
        new_symbols = {}
        # Set initial activations for tensor propagation
        initial_activations = {}
        for symbol_id, symbol in existing_symbols.items():
            initial_activations[symbol_id] = symbol.get_pregnancy() / 10  # Normalize
            
        # Propagate activations to find potential catalytic interactions
        propagated = self._tensor_network.propagate_activations(initial_activations)
        
        # Sort templates by nonlinearity order (prioritize simpler templates)
        sorted_templates = sorted(
            self.production_templates.keys(),
            key=lambda t: (self.template_nonlinearity[t].value, -self._template_success_rate.get(t, 0.0))
        )
        
        # Check each production template in order of increasing nonlinearity
        for template_id in sorted_templates:
            required = self.production_templates[template_id]
            # Calculate production strength based on propagated activations
            production_strength = self._calculate_production_strength(propagated, required)
            
            if production_strength > threshold:
                # Generate new symbol
                new_id = f"generated_{template_id}_{len(new_symbols)}_{uuid.uuid4().hex[:4]}"
                # Create with subset of perceptual anchors
                anchors = {k: v for k, v in perceptual_inputs.items() if v > 0.3}  # Threshold for inclusion
                new_symbol = SymbolElement(new_id, anchors)
                # Set name based on template
                new_symbol.name = f"{template_id}_{new_id[-4:]}"
                
                # Add initial potentials based on template
                for req_id, req_strength in required.items():
                    if req_id in propagated and propagated[req_id] > 0.2:
                        if req_id in existing_symbols:
                            # Inherit some potentials from catalyst symbols
                            self._transfer_potentials(
                                existing_symbols[req_id],
                                new_symbol,
                                req_strength
                            )
                
                # Only add if symbol has meaningful potentials
                if new_symbol.potentials:
                    new_symbols[new_id] = new_symbol
                    
                    # Update template success statistics
                    self._template_usage_count[template_id] += 1
                    self._template_success_rate[template_id] = (
                        (self._template_success_rate[template_id] * (self._template_usage_count[template_id] - 1)
                         + 1.0) /
                        self._template_usage_count[template_id]
                    )
                    
                    # Phase 2 enhancement: template adaptation
                    self._adapt_template(template_id, True, new_symbol)
            else:
                # Update template failure statistics
                self._template_usage_count[template_id] += 1
                self._template_success_rate[template_id] = (
                    (self._template_success_rate[template_id] * (self._template_usage_count[template_id] - 1)
                     + 0.0) /
                    self._template_usage_count[template_id]
                )
                
                # Phase 2 enhancement: template adaptation
                self._adapt_template(template_id, False, None)
                
        return new_symbols
    
    def _calculate_production_strength(self, propagated_activations: Dict[str, float],
                                     required: Dict[str, float]) -> float:
        """Calculate production strength based on propagated activations."""
        if not required:
            return 0.0
            
        # Use maximization-based approach rather than average (MGC-inspired)
        strengths = []
        for req_id, req_strength in required.items():
            if req_id in propagated_activations:
                # Consider both requirement strength and activation
                activation = propagated_activations[req_id]
                strengths.append(activation * req_strength)
                
        if not strengths:
            return 0.0
            
        # Return maximum strength rather than average
        return max(strengths)
    
    def _transfer_potentials(self, source: SymbolElement, target: SymbolElement, strength: float) -> None:
        """
        Transfer potentials from source to target symbol.
        Optimized to maintain lower nonlinearity order.
        """
        # Select potentials to transfer
        potentials_to_transfer = []
        
        # Select based on both strength and nonlinearity order
        for potential in source.potentials.values():
            # Prioritize potentials with lower nonlinearity
            transfer_score = strength * (1.0 / (1.0 + potential.nonlinearity.value))
            if np.random.random() < transfer_score:
                potentials_to_transfer.append(potential)
                
        # Transfer selected potentials with modified strength
        for potential in potentials_to_transfer:
            # Create new potential with nonlinearity one order higher
            new_nonlinearity = NonlinearityOrder(
                min(NonlinearityOrder.COMPOSITIONAL.value,
                   potential.nonlinearity.value + 1))
            new_potential_id = f"{target.id}_{potential.id}"
            
            from asf.symbolic_formation.potentials import SymbolicPotential
            
            new_potential = SymbolicPotential(
                new_potential_id,
                potential.strength * strength,
                nonlinearity=new_nonlinearity
            )
            
            # Copy subset of associations based on strength
            assoc_items = list(potential._associations.items())
            if len(assoc_items) > 3:
                assoc_items = assoc_items[-3:]  # Top 3
                
            for assoc_id, assoc_strength in assoc_items:
                if np.random.random() < strength:  # Probabilistic transfer
                    new_potential.add_association(assoc_id, assoc_strength * strength)
                    
            target.add_potential(new_potential)
            
            # Register with nonlinearity tracker
            self.nonlinearity_tracker.register_potential(
                new_potential.id, new_nonlinearity
            )
    
    # Phase 2 enhancement: template adaptation
    def _adapt_template(self, template_id: str, success: bool, new_symbol: Optional[SymbolElement] = None) -> None:
        """
        Adapt template based on success or failure in generating symbols.
        
        Args:
            template_id: Template to adapt
            success: Whether the generation was successful
            new_symbol: The newly generated symbol if success is True
        """
        if template_id not in self.production_templates:
            return
            
        current_template = self.production_templates[template_id]
        
        if success and new_symbol:
            # Strengthen requirements that contributed to success
            # Use potentials of new symbol to guide adaptation
            for potential in new_symbol.potentials.values():
                # Find which requirement contributed to this potential
                potential_source = potential.id.split("_")[-1]  # Assuming potential ID format
                for req_id in current_template:
                    if potential_source in req_id:
                        # Strengthen this requirement slightly
                        current_template[req_id] = min(1.0,
                                                    current_template[req_id] + self._template_adaption_rate)
        else:
            # Slight reduction in all requirements to make template more flexible
            for req_id in current_template:
                current_template[req_id] = max(0.1,
                                            current_template[req_id] - (self._template_adaption_rate / 2))
                
        # Record adaptation history
        self._template_evolution_history.append({
            'template_id': template_id,
            'success': success,
            'timestamp': time.time(),
            'new_requirements': dict(current_template)
        })



# === FILE: asf\layer2_autopoietic_maintanance\autopoietic_maintanance_layer.py ===

import time
import joblib
import asyncio
import logging
from typing import Dict, List, Any, Optional, Set, Tuple

from asf.symbolic_formation.enums import NonlinearityOrder, SymbolConfidenceState
from asf.symbolic_formation.potentials import SymbolicPotential
from asf.symbolic_formation.symbol import SymbolElement
from asf.symbolic_formation.network import SparseTensorSymbolNetwork
from asf.symbolic_formation.autocatalytic import NonlinearityOrderTracker, AutocatalyticNetwork
from asf.symbolic_formation.operational_closure import OperationalClosure
from asf.symbolic_formation.recognition import SymbolRecognizer

class SymbolicFormationLayer:
    """
    Main controller for the Symbolic Formation Layer (Layer 2).
    Coordinates all symbol-related processes and interfaces with other layers.
    """
    def __init__(self, config=None):
        self.config = config or {}
        
        # Core data structures
        self.symbols: Dict[str, SymbolElement] = {}
        self.recognized_symbol_mapping: Dict[str, str] = {}  # Maps perceptual IDs to symbol IDs
        
        # Component initialization
        self.nonlinearity_tracker = NonlinearityOrderTracker()
        self.network = SparseTensorSymbolNetwork()
        self.autocatalytic_network = AutocatalyticNetwork(self.nonlinearity_tracker)
        self.operational_closure = OperationalClosure()
        self.recognizer = SymbolRecognizer(
            threshold=self.config.get('recognition_threshold', 0.7)
        )
        
        # Performance tracking
        self.stats = {
            "symbols_created": 0,
            "symbols_recognized": 0,
            "potentials_actualized": 0,
            "system_closure": 0.0
        }
        
        # Cache for context hashing
        self.context_cache = {}
        
        # Initialize logger
        self.logger = logging.getLogger("ASF.SymbolicFormation")
        
        # Load initial templates
        self._initialize_production_templates()
    
    def _initialize_production_templates(self) -> None:
        """Initialize basic production templates for autocatalytic network."""
        # Simple combination template
        self.autocatalytic_network.add_production_template(
            "combination",
            {"entity1": 0.7, "entity2": 0.7},
            NonlinearityOrder.LINEAR
        )
        
        # Contrast template
        self.autocatalytic_network.add_production_template(
            "contrast",
            {"entity1": 0.8, "opposite1": 0.5},
            NonlinearityOrder.QUADRATIC
        )
        
        # Abstraction template
        self.autocatalytic_network.add_production_template(
            "abstraction",
            {"specific1": 0.6, "specific2": 0.6, "specific3": 0.6},
            NonlinearityOrder.CUBIC
        )
        
        # Analogy template
        self.autocatalytic_network.add_production_template(
            "analogy",
            {"source": 0.8, "target": 0.7, "relation": 0.5},
            NonlinearityOrder.EXPONENTIAL
        )
    
    async def process_perceptual_input(self, perceptual_data: Dict[str, Dict[str, float]],
                                    context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process new perceptual input to form or recognize symbols.
        
        Args:
            perceptual_data: Dictionary mapping entity IDs to features
            context: Optional processing context
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        context = context or {}
        context_hash = joblib.hash(context)
        
        # Try to recognize existing symbols first
        recognition_result = await self.recognizer.recognize(
            perceptual_data, self.symbols, context
        )
        
        result = {
            "processing_time": 0,
            "new_symbols": [],
            "recognized_symbols": [],
            "activations": {}
        }
        
        if recognition_result["recognized"]:
            # Symbol recognized - update mapping and confidence
            symbol_id = recognition_result["symbol_id"]
            symbol = self.symbols[symbol_id]
            
            # Update confidence based on successful recognition
            symbol.update_confidence(True, weight=recognition_result["confidence"])
            
            # Update entity mapping
            for entity_id in perceptual_data:
                self.recognized_symbol_mapping[entity_id] = symbol_id
                # Add source tracking
                symbol.add_source_entity(entity_id)
            
            # Actualize symbol in current context
            meaning = symbol.actualize_meaning(context_hash, context)
            
            # Track stats
            self.stats["symbols_recognized"] += 1
            self.stats["potentials_actualized"] += len(meaning)
            
            result["recognized_symbols"].append({
                "symbol_id": symbol_id,
                "confidence": recognition_result["confidence"],
                "strategy": recognition_result.get("strategy")
            })
            result["activations"][symbol_id] = meaning
        else:
            # No recognition - try to form new symbols
            new_symbols = await self._form_new_symbols(perceptual_data, context)
            
            if new_symbols:
                # Add new symbols to main collection
                for symbol_id, symbol in new_symbols.items():
                    self.symbols[symbol_id] = symbol
                    
                    # Register with nonlinearity tracker
                    self.nonlinearity_tracker.register_symbol(
                        symbol_id, symbol.get_nonlinearity()
                    )
                    
                    # Add to network
                    self.network.add_symbol(symbol_id)
                    
                    # Create mappings for source entities
                    for entity_id in perceptual_data:
                        self.recognized_symbol_mapping[entity_id] = symbol_id
                        symbol.add_source_entity(entity_id)
                        
                    # Track formation of new symbol
                    result["new_symbols"].append({
                        "symbol_id": symbol_id,
                        "name": symbol.name,
                        "pregnancy": symbol.get_pregnancy(),
                        "potentials": len(symbol.potentials)
                    })
                
                # Track stats
                self.stats["symbols_created"] += len(new_symbols)
                
                # Maintain operational closure after adding new symbols
                self._maintain_closure()
        
        # Calculate current closure
        self.stats["system_closure"] = self.operational_closure.calculate_closure(self.symbols)
        
        # Calculate processing time
        result["processing_time"] = time.time() - start_time
        
        return result
    
    async def _form_new_symbols(self, perceptual_data: Dict[str, Dict[str, float]],
                              context: Dict[str, Any]) -> Dict[str, SymbolElement]:
        """
        Form new symbols from perceptual data using autocatalytic network.
        """
        # Flatten perceptual data for symbol formation
        flat_perceptual = {}
        for entity_id, features in perceptual_data.items():
            for feature_name, value in features.items():
                key = f"{entity_id}:{feature_name}"
                flat_perceptual[key] = value
                
        # Use autocatalytic network to generate new symbols
        return self.autocatalytic_network.generate_symbols(
            self.symbols,
            flat_perceptual,
            threshold=self.config.get('symbol_formation_threshold', 0.5)
        )
    
    def actualize_symbol(self, symbol_id: str, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Actualize the meaning of a symbol in a specific context.
        
        Args:
            symbol_id: ID of symbol to actualize
            context: Context for actualization
            
        Returns:
            Dictionary of actualized meaning aspects with strengths
        """
        if symbol_id not in self.symbols:
            return {}
            
        symbol = self.symbols[symbol_id]
        context_hash = joblib.hash(context)
        
        # Actualize the symbol in this context
        meaning = symbol.actualize_meaning(context_hash, context)
        
        # Track potentials actualized
        self.stats["potentials_actualized"] += len(meaning)
        
        return meaning
    
    def propagate_activations(self, initial_activations: Dict[str, float],
                            context: Dict[str, Any] = None,
                            iterations: int = 2) -> Dict[str, Dict[str, float]]:
        """
        Propagate activations through the symbol network.
        
        Args:
            initial_activations: Dictionary mapping symbol IDs to initial activations
            context: Optional context for meaning actualization
            iterations: Number of propagation iterations
            
        Returns:
            Dictionary of symbol IDs to their actualized meanings
        """
        # First propagate through network to get activated symbols
        propagated = self.network.propagate_activations(initial_activations, iterations)
        
        # For symbols that are sufficiently activated, actualize their meanings
        result = {}
        context_hash = joblib.hash(context) if context else "default"
        
        for symbol_id, activation in propagated.items():
            if activation > 0.2 and symbol_id in self.symbols:  # Activation threshold
                symbol = self.symbols[symbol_id]
                meaning = symbol.actualize_meaning(context_hash, context or {})
                
                # Scale meaning by activation strength
                scaled_meaning = {k: v * activation for k, v in meaning.items()}
                result[symbol_id] = scaled_meaning
                
                # Track stats
                self.stats["potentials_actualized"] += len(meaning)
                
        return result
    
    def _maintain_closure(self) -> None:
        """
        Maintain operational closure of the symbol system.
        Adds suggested relations to ensure system coherence.
        """
        suggested_relations = self.operational_closure.maintain_closure(
            self.symbols,
            self.nonlinearity_tracker,
            min_closure=self.config.get('min_closure', 0.7)
        )
        
        # Add suggested relations to maintain closure
        for source_key, target_key in suggested_relations:
            # Extract symbol and potential IDs
            if ":" in source_key and ":" in target_key:
                source_symbol_id, source_potential_id = source_key.split(":", 1)
                target_symbol_id, target_potential_id = target_key.split(":", 1)
                
                # Check if symbols and potentials exist
                if (source_symbol_id in self.symbols and
                    source_potential_id in self.symbols[source_symbol_id].potentials and
                    target_symbol_id in self.symbols and
                    target_potential_id in self.symbols[target_symbol_id].potentials):
                    
                    # Add association between potentials
                    source_potential = self.symbols[source_symbol_id].potentials[source_potential_id]
                    source_potential.add_association(target_key, 0.5)  # Default strength
                    
                    # Add to network
                    self.network.add_relation(source_key, target_key, 0, 0.5)
    
    def get_symbol_by_id(self, symbol_id: str) -> Optional[SymbolElement]:
        """Get a symbol by ID."""
        return self.symbols.get(symbol_id)
    
    def get_symbols_for_entity(self, entity_id: str) -> List[SymbolElement]:
        """Get all symbols associated with a perceptual entity."""
        if entity_id in self.recognized_symbol_mapping:
            symbol_id = self.recognized_symbol_mapping[entity_id]
            if symbol_id in self.symbols:
                return [self.symbols[symbol_id]]
        
        # Search for entity in source_entities lists
        result = []
        for symbol in self.symbols.values():
            if entity_id in symbol.source_entities:
                result.append(symbol)
                
        return result
    
    def get_layer_statistics(self) -> Dict[str, Any]:
        """Get statistics about layer operations."""
        return {
            **self.stats,
            "total_symbols": len(self.symbols),
            "canonical_symbols": sum(1 for s in self.symbols.values() 
                                  if s.confidence_state == SymbolConfidenceState.CANONICAL),
            "total_potentials": sum(len(s.potentials) for s in self.symbols.values()),
            "avg_pregnancy": sum(s.get_pregnancy() for s in self.symbols.values()) / 
                            max(1, len(self.symbols))
        }
    
    async def run_integration_cycle(self) -> Dict[str, Any]:
        """
        Run an integration cycle to enhance system coherence.
        This performs maintenance operations on the symbol system.
        """
        start_time = time.time()
        
        # Maintain operational closure
        self._maintain_closure()
        
        # Generate new symbols through autocatalytic process
        # Use current symbols to generate new emergent symbols
        flat_perceptual = {}  # Empty perceptual input for pure emergent generation
        symbols_before = len(self.symbols)
        
        new_symbols = self.autocatalytic_network.generate_symbols(
            self.symbols,
            flat_perceptual,
            threshold=self.config.get('emergent_symbol_threshold', 0.7)
        )
        
        # Add new symbols to main collection
        for symbol_id, symbol in new_symbols.items():
            self.symbols[symbol_id] = symbol
            
            # Register with nonlinearity tracker
            self.nonlinearity_tracker.register_symbol(
                symbol_id, symbol.get_nonlinearity()
            )
            
            # Add to network
            self.network.add_symbol(symbol_id)
        
        # Update system closure
        closure = self.operational_closure.calculate_closure(self.symbols)
        self.stats["system_closure"] = closure
        
        return {
            "processing_time": time.time() - start_time,
            "new_symbols": len(self.symbols) - symbols_before,
            "system_closure": closure,
            "total_symbols": len(self.symbols)
        }



# === FILE: asf\layer2_autopoietic_maintanance\autopoietic_maintenance_engine.py ===




# === FILE: asf\layer2_autopoietic_maintanance\bayesian_confidence.py ===




# === FILE: asf\layer2_autopoietic_maintanance\contradiction_detection.py ===




# === FILE: asf\layer2_autopoietic_maintanance\contradiction_pattern_analysis.py ===




# === FILE: asf\layer2_autopoietic_maintanance\counterfactual_network.py ===

import time
import uuid
import numpy as np
from typing import Dict, List, Any, Optional, Set, Tuple

from asf.symbolic_formation.autocatalytic import AutocatalyticNetwork
from asf.symbolic_formation.autocatalytic import NonlinearityOrderTracker
from asf.symbolic_formation.symbol import SymbolElement

class CounterfactualAutocatalyticNetwork(AutocatalyticNetwork):
    """
    Enhances AutocatalyticNetwork with counterfactual reasoning capabilities.
    Allows testing hypothetical scenarios through virtual interventions.
    """
    def __init__(self, nonlinearity_tracker: NonlinearityOrderTracker):
        super().__init__(nonlinearity_tracker)
        self.counterfactual_history = []
        
    def generate_counterfactual_symbols(self, existing_symbols: Dict[str, SymbolElement], 
                                       perceptual_inputs: Dict[str, float], 
                                       modification_rules: List[Dict[str, Any]]) -> Dict[str, SymbolElement]:
        """
        Generate 'what if' symbols by counterfactually modifying inputs
        
        Args:
            existing_symbols: Current symbols
            perceptual_inputs: Actual perceptual data
            modification_rules: Rules for counterfactual modifications
        
        Returns:
            Dictionary of counterfactual symbols that might have been generated
        """
        # Create counterfactual perceptual inputs
        cf_inputs = self._apply_modifications(perceptual_inputs, modification_rules)
        
        # Generate symbols from counterfactual inputs
        cf_symbols = self.generate_symbols(existing_symbols, cf_inputs)
        
        # Track counterfactual generation
        self.counterfactual_history.append({
            'timestamp': time.time(),
            'modifications': modification_rules,
            'original_input_size': len(perceptual_inputs),
            'modified_input_size': len(cf_inputs),
            'symbols_generated': len(cf_symbols)
        })
        
        return cf_symbols
    
    def _apply_modifications(self, perceptual_inputs: Dict[str, float], 
                            modification_rules: List[Dict[str, Any]]) -> Dict[str, float]:
        """Apply modification rules to perceptual inputs"""
        modified_inputs = perceptual_inputs.copy()
        
        for rule in modification_rules:
            rule_type = rule.get('type')
            
            if rule_type == 'remove':
                # Remove features matching pattern
                pattern = rule.get('pattern', '')
                keys_to_remove = [k for k in modified_inputs if pattern in k]
                for key in keys_to_remove:
                    modified_inputs.pop(key, None)
                    
            elif rule_type == 'add':
                # Add new features
                features = rule.get('features', {})
                for key, value in features.items():
                    modified_inputs[key] = value
                    
            elif rule_type == 'modify':
                # Modify existing features
                pattern = rule.get('pattern', '')
                operation = rule.get('operation', 'multiply')
                factor = rule.get('factor', 1.0)
                
                for key, value in list(modified_inputs.items()):
                    if pattern in key:
                        if operation == 'multiply':
                            modified_inputs[key] = value * factor
                        elif operation == 'add':
                            modified_inputs[key] = value + factor
                        elif operation == 'replace':
                            modified_inputs[key] = factor
        
        return modified_inputs
    
    def compare_counterfactual_outcomes(self, actual_symbols: Dict[str, SymbolElement], 
                                      counterfactual_symbols: Dict[str, SymbolElement]) -> Dict[str, Any]:
        """
        Compare actual symbols with counterfactual ones to evaluate impact
        
        Returns analysis of differences and hypothesis validation
        """
        # Compare symbol sets
        actual_ids = set(actual_symbols.keys())
        cf_ids = set(counterfactual_symbols.keys())
        
        # Find symbols only in actual or counterfactual
        only_actual = actual_ids - cf_ids
        only_cf = cf_ids - actual_ids
        shared = actual_ids.intersection(cf_ids)
        
        # Calculate meaning differences for shared symbols
        meaning_diffs = {}
        for symbol_id in shared:
            actual_symbol = actual_symbols[symbol_id]
            cf_symbol = counterfactual_symbols[symbol_id]
            
            # Create context for meaning actualization
            context = {'comparison': True}
            context_hash = str(hash(str(context)))
            
            # Actualize meanings
            actual_meaning = actual_symbol.actualize_meaning(context_hash, context)
            cf_meaning = cf_symbol.actualize_meaning(context_hash, context)
            
            # Calculate differences
            potential_diffs = {}
            for potential_id in set(list(actual_meaning.keys()) + list(cf_meaning.keys())):
                actual_val = actual_meaning.get(potential_id, 0.0)
                cf_val = cf_meaning.get(potential_id, 0.0)
                diff = cf_val - actual_val
                if abs(diff) > 0.1:  # Only track significant differences
                    potential_diffs[potential_id] = diff
            
            if potential_diffs:
                meaning_diffs[symbol_id] = potential_diffs
        
        return {
            'only_in_actual': list(only_actual),
            'only_in_counterfactual': list(only_cf),
            'shared_symbols': len(shared),
            'meaning_differences': meaning_diffs
        }



# === FILE: asf\layer2_autopoietic_maintanance\enums.py ===

from enum import Enum

class NonlinearityOrder(Enum):
    """
    Enumeration for tracking order of nonlinearity in symbolic transformations.
    Higher orders represent more complex conceptual transformations.
    """
    LINEAR = 1      # Direct correspondences and simple relationships
    QUADRATIC = 2   # Second-order transformations involving two interacting elements
    CUBIC = 3       # Third-order transformations with multiple interactions
    EXPONENTIAL = 4 # Transformations with rapidly increasing complexity
    COMPOSITIONAL = 5 # Highest order - complex compositions of multiple transformations

class SymbolConfidenceState(Enum):
    """
    Enumeration for tracking confidence states of symbols.
    Mirrors the confidence states in Layer 1 for consistency.
    """
    HYPOTHESIS = "hypothesis"  # Initial hypothetical state
    PROVISIONAL = "provisional" # Partially validated
    CANONICAL = "canonical"    # Fully validated



# === FILE: asf\layer2_autopoietic_maintanance\multi_resolution_modeling.py ===




# === FILE: asf\layer2_autopoietic_maintanance\network.py ===

import torch
from typing import Dict, List, Optional

class SparseTensorSymbolNetwork:
    """
    Efficient sparse tensor representation of the symbol network.
    Uses Graph Tensor Convolution inspired approach for rich semantic propagation.
    Enhanced with hardware acceleration capabilities.
    """
    def __init__(self, max_symbols: int = 10000, device=None):
        self.max_symbols = max_symbols
        self.symbol_to_idx: Dict[str, int] = {}
        self.idx_to_symbol: Dict[int, str] = {}
        self.potential_to_idx: Dict[str, int] = {}
        self.idx_to_potential: Dict[int, str] = {}
        
        # Sparse adjacency data structures
        self._row_indices: List[int] = []
        self._col_indices: List[int] = []
        self._edge_values: List[float] = []
        self._edge_types: List[int] = []
        
        # Cached tensor representations
        self._adjacency_tensor: Optional[List[torch.Tensor]] = None
        self._feature_tensor: Optional[torch.Tensor] = None
        self._need_rebuild: bool = True
        
        # Configuration for tensor operations
        self.semantic_channels = 3  # Multiple channels for rich semantic representation
        
        # Hardware acceleration support
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    
    def add_symbol(self, symbol_id: str) -> int:
        """Add a symbol to the network and return its index."""
        if symbol_id in self.symbol_to_idx:
            return self.symbol_to_idx[symbol_id]
            
        idx = len(self.symbol_to_idx)
        self.symbol_to_idx[symbol_id] = idx
        self.idx_to_symbol[idx] = symbol_id
        self._need_rebuild = True
        return idx
    
    def add_potential(self, potential_id: str) -> int:
        """Add a potential to the network and return its index."""
        if potential_id in self.potential_to_idx:
            return self.potential_to_idx[potential_id]
            
        idx = len(self.potential_to_idx)
        self.potential_to_idx[potential_id] = idx
        self.idx_to_potential[idx] = potential_id
        self._need_rebuild = True
        return idx
    
    def add_relation(self, source_id: str, target_id: str, relation_type: int, strength: float) -> None:
        """Add a relation between symbols or potentials."""
        # Determine if source/target are symbols or potentials
        source_is_symbol = source_id in self.symbol_to_idx
        target_is_symbol = target_id in self.symbol_to_idx
        
        # Get or create indices
        if source_is_symbol:
            source_idx = self.symbol_to_idx[source_id]
        else:
            source_idx = self.add_potential(source_id) + self.max_symbols
            
        if target_is_symbol:
            target_idx = self.symbol_to_idx[target_id]
        else:
            target_idx = self.add_potential(target_id) + self.max_symbols
            
        # Add relation to sparse representation
        self._row_indices.append(source_idx)
        self._col_indices.append(target_idx)
        self._edge_values.append(strength)
        self._edge_types.append(relation_type)
        
        self._need_rebuild = True
        
    def _build_tensors(self) -> None:
        """Build sparse tensor representations of the network with hardware acceleration."""
        if not self._need_rebuild:
            return
            
        # Convert to PyTorch sparse tensor with hardware acceleration
        indices = torch.tensor([self._row_indices, self._col_indices], 
                              dtype=torch.long, device=self.device)
        values = torch.tensor(self._edge_values, dtype=torch.float, device=self.device)
        edge_types = torch.tensor(self._edge_types, dtype=torch.long, device=self.device)
        
        # Size includes both symbols and potentials
        size = (self.max_symbols * 2, self.max_symbols * 2)
        
        # Create multi-channel adjacency tensor (GTCN-inspired)
        # Each channel represents different semantic relationship aspects
        max_edge_type = max(self._edge_types) if self._edge_types else 0
        channels = min(max_edge_type + 1, self.semantic_channels)
        
        adjacency_tensors = []
        for channel in range(channels):
            # Filter edges for this channel/type
            channel_mask = edge_types == channel
            if not torch.any(channel_mask):
                # Empty channel
                channel_tensor = torch.sparse.FloatTensor(
                    torch.zeros((2, 0), dtype=torch.long, device=self.device),
                    torch.tensor([], dtype=torch.float, device=self.device),
                    size
                )
            else:
                channel_indices = indices[:, channel_mask]
                channel_values = values[channel_mask]
                channel_tensor = torch.sparse.FloatTensor(
                    channel_indices, channel_values, size,
                    device=self.device
                )
            adjacency_tensors.append(channel_tensor)
            
        # Store as list of sparse tensors (more efficient than 3D sparse tensor)
        self._adjacency_tensor = adjacency_tensors
        
        # Initialize feature tensor
        total_nodes = len(self.symbol_to_idx) + len(self.potential_to_idx)
        self._feature_tensor = torch.zeros(
            (total_nodes, 16),
            dtype=torch.float,
            device=self.device
        )
        
        self._need_rebuild = False
        
    def propagate_activations(self, initial_activations: Dict[str, float], iterations: int = 2) -> Dict[str, float]:
        """
        Propagate activations through the network using tensor operations.
        Uses GTCN-inspired multi-channel propagation for rich semantic transfer.
        """
        self._build_tensors()
        
        # Initialize activation vector
        activation_vector = torch.zeros(
            self.max_symbols * 2,
            dtype=torch.float,
            device=self.device
        )
        
        # Set initial activations
        for node_id, activation in initial_activations.items():
            if node_id in self.symbol_to_idx:
                idx = self.symbol_to_idx[node_id]
                activation_vector[idx] = activation
            elif node_id in self.potential_to_idx:
                idx = self.potential_to_idx[node_id] + self.max_symbols
                activation_vector[idx] = activation
        
        # Propagate activations through the network
        for _ in range(iterations):
            # Propagate through each channel
            channel_results = []
            for channel_tensor in self._adjacency_tensor:
                if channel_tensor._nnz() > 0:  # Check if tensor has any non-zero elements
                    # Sparse matrix multiplication for efficiency
                    channel_result = torch.sparse.mm(channel_tensor, 
                                                   activation_vector.unsqueeze(1)).squeeze(1)
                    channel_results.append(channel_result)
            
            if channel_results:
                # Stack results from all channels
                stacked_results = torch.stack(channel_results, dim=0)
                # Use maximization-based aggregation (MGC-inspired)
                # This prevents over-smoothing compared to averaging
                new_activations, _ = torch.max(stacked_results, dim=0)
                # Update activations (with residual connection)
                activation_vector = (activation_vector + new_activations) / 2
        
        # Extract results for symbols
        result = {}
        for symbol_id, idx in self.symbol_to_idx.items():
            result[symbol_id] = activation_vector[idx].item()
            
        return result



# === FILE: asf\layer2_autopoietic_maintanance\operational_closure.py ===

import time
import scipy.sparse as sp
from typing import Dict, List, Tuple, Set, Optional

from asf.symbolic_formation.enums import NonlinearityOrder
from asf.symbolic_formation.symbol import SymbolElement

class OperationalClosure:
    """
    Implements mechanisms for maintaining system coherence through
    operational closure as per Maturana and Varela.
    Optimized with sparse matrix representation for efficiency.
    """
    def __init__(self):
        self.boundary_elements: Set[str] = set()
        # Use sparse matrix for internal relations
        self._relation_matrix: Optional[sp.csr_matrix] = None
        self._element_indices: Dict[str, int] = {}
        self._index_elements: Dict[int, str] = {}
        self.closure_metrics: Dict[str, float] = {}
        self._need_rebuild: bool = True
        
        # Phase 2 enhancement: system integrity tracking
        self.integrity_history = []
        self.last_integrity_check = 0
    
    def add_boundary_element(self, element_id: str) -> None:
        """Add an element to the system boundary."""
        self.boundary_elements.add(element_id)
        # Ensure element is in index mapping
        if element_id not in self._element_indices:
            idx = len(self._element_indices)
            self._element_indices[element_id] = idx
            self._index_elements[idx] = element_id
            self._need_rebuild = True
    
    def add_internal_relation(self, source_id: str, target_id: str) -> None:
        """Add an internal relation between elements."""
        # Ensure elements are in index mapping
        for element_id in (source_id, target_id):
            if element_id not in self._element_indices:
                idx = len(self._element_indices)
                self._element_indices[element_id] = idx
                self._index_elements[idx] = element_id
                
        self._need_rebuild = True
        
    def _rebuild_matrix(self, relations: List[Tuple[str, str]]) -> None:
        """Rebuild the sparse relation matrix."""
        if not self._need_rebuild and self._relation_matrix is not None:
            return
            
        n_elements = len(self._element_indices)
        if n_elements == 0:
            self._relation_matrix = sp.csr_matrix((0, 0))
            return
            
        # Build sparse matrix
        rows, cols, data = [], [], []
        for source_id, target_id in relations:
            if source_id in self._element_indices and target_id in self._element_indices:
                source_idx = self._element_indices[source_id]
                target_idx = self._element_indices[target_id]
                rows.append(source_idx)
                cols.append(target_idx)
                data.append(1.0)
                
        self._relation_matrix = sp.csr_matrix(
            (data, (rows, cols)), shape=(n_elements, n_elements))
            
        self._need_rebuild = False
        
    def calculate_closure(self, elements: Dict[str, SymbolElement]) -> float:
        """
        Calculate degree of operational closure using efficient sparse operations.
        1.0 means perfect closure, 0.0 means completely open.
        """
        if not elements:
            return 0.0
            
        # Extract relations from elements
        relations = []
        for symbol_id, symbol in elements.items():
            # Add symbol to indices if needed
            if symbol_id not in self._element_indices:
                idx = len(self._element_indices)
                self._element_indices[symbol_id] = idx
                self._index_elements[idx] = symbol_id
                
            # Extract relations from potentials
            for potential_id, potential in symbol.potentials.items():
                source_key = f"{symbol_id}:{potential_id}"
                
                # Add relation source to indices
                if source_key not in self._element_indices:
                    idx = len(self._element_indices)
                    self._element_indices[source_key] = idx
                    self._index_elements[idx] = source_key
                    
                # Add relations to associations
                for assoc_id in potential._associations:
                    relations.append((source_key, assoc_id))
                    
        # Rebuild relation matrix
        self._rebuild_matrix(relations)
        
        # Calculate closure using matrix operations
        if self._relation_matrix.shape[0] == 0:
            return 0.0
            
        # Count total relations
        total_relations = self._relation_matrix.count_nonzero()
        if total_relations == 0:
            return 0.0
            
        # Count relations between elements in the system
        element_indices = [self._element_indices[e_id] for e_id in elements
                         if e_id in self._element_indices]
        if not element_indices:
            return 0.0
            
        # Extract submatrix for system elements
        system_matrix = self._relation_matrix[element_indices, :][:, element_indices]
        internal_relations = system_matrix.count_nonzero()
        
        # Record system integrity
        current_time = time.time()
        if current_time - self.last_integrity_check > 60:  # Check at most once per minute
            closure_score = internal_relations / total_relations
            self.integrity_history.append({
                'timestamp': current_time,
                'closure_score': closure_score,
                'total_relations': total_relations,
                'internal_relations': internal_relations,
                'element_count': len(elements)
            })
            self.last_integrity_check = current_time
            
        return internal_relations / total_relations
        
    def maintain_closure(self, elements: Dict[str, SymbolElement],
                      nonlinearity_tracker,
                      min_closure: float = 0.7) -> List[Tuple[str, str]]:
        """
        Maintain operational closure by suggesting new internal relations
        if closure falls below threshold. Prioritizes simpler relationships.
        """
        current_closure = self.calculate_closure(elements)
        if current_closure >= min_closure:
            return []
            
        # Find potential new relations to increase closure
        suggested_relations = []
        
        # Extract current relations
        current_relations = set()
        for symbol_id, symbol in elements.items():
            for potential_id, potential in symbol.potentials.items():
                source_key = f"{symbol_id}:{potential_id}"
                for assoc_id in potential._associations:
                    current_relations.add((source_key, assoc_id))
                    
        # Find candidate relations between system elements
        element_ids = set(elements.keys())
        potential_relations = []
        
        for symbol_id, symbol in elements.items():
            for potential_id, potential in symbol.potentials.items():
                source_key = f"{symbol_id}:{potential_id}"
                
                for target_id in element_ids:
                    if target_id != symbol_id:
                        for target_pot_id in elements[target_id].potentials:
                            target_key = f"{target_id}:{target_pot_id}"
                            
                            # Check if relation already exists
                            if (source_key, target_key) not in current_relations:
                                # Calculate potential relationship nonlinearity
                                nonlinearity = NonlinearityOrder.LINEAR
                                
                                if source_key in nonlinearity_tracker.potential_nonlinearity:
                                    source_nl = nonlinearity_tracker.potential_nonlinearity[source_key]
                                else:
                                    source_nl = NonlinearityOrder.LINEAR
                                    
                                if target_key in nonlinearity_tracker.potential_nonlinearity:
                                    target_nl = nonlinearity_tracker.potential_nonlinearity[target_key]
                                else:
                                    target_nl = NonlinearityOrder.LINEAR
                                
                                # Combine nonlinearities
                                nonlinearity = NonlinearityOrder(
                                    min(NonlinearityOrder.COMPOSITIONAL.value,
                                       max(source_nl.value, target_nl.value) + 1))
                                        
                                # Add as candidate with nonlinearity as score
                                potential_relations.append(
                                    (source_key, target_key, nonlinearity))
                                    
        # Sort potential relations by nonlinearity (simpler first)
        potential_relations.sort(key=lambda x: x[2].value)
        
        # Select top relations to suggest
        needed_relations = int((min_closure - current_closure) * 
                             len(current_relations) * 1.5) + 1
                              
        suggested_relations = [(src, tgt) for src, tgt, _ in 
                              potential_relations[:needed_relations]]
                               
        return suggested_relations



# === FILE: asf\layer2_autopoietic_maintanance\potentials.py ===




# === FILE: asf\layer2_autopoietic_maintanance\predictive_factory.py ===

import logging
from typing import Dict, Any, Optional

from asf.symbolic_formation.symbol import SymbolElement
from asf.symbolic_formation.potentials import SymbolicPotential
from asf.symbolic_formation.predictive_potentials import PredictiveSymbolicPotential
from asf.symbolic_formation.predictive_symbol import PredictiveSymbolElement
from asf.symbolic_formation.predictive_layer import PredictiveSymbolicFormationLayer

logger = logging.getLogger(__name__)

def create_predictive_layer2(config=None):
    """Factory function to create a predictive Layer 2"""
    logger.info("Creating predictive Layer 2 with Seth's Data Paradox enhancements")
    return PredictiveSymbolicFormationLayer(config)

def convert_to_predictive(layer):
    """Convert standard symbols and potentials to predictive variants"""
    logger.info(f"Converting {len(layer.symbols)} symbols to predictive variants")
    
    # Convert symbols to predictive variants
    predictive_symbols = {}
    for symbol_id, symbol in layer.symbols.items():
        # Create predictive symbol with same properties
        predictive_symbol = PredictiveSymbolElement(symbol.id, symbol.perceptual_anchors.copy())
        predictive_symbol.name = symbol.name
        predictive_symbol.confidence = symbol.confidence
        predictive_symbol.confidence_state = symbol.confidence_state
        predictive_symbol.confidence_evidence = symbol.confidence_evidence.copy()
        predictive_symbol.source_entities = symbol.source_entities.copy()
        predictive_symbol.created_at = symbol.created_at
        predictive_symbol.last_accessed = symbol.last_accessed
        predictive_symbol._nonlinearity = symbol._nonlinearity
        
        # Convert potentials to predictive variants
        for potential_id, potential in symbol.potentials.items():
            predictive_potential = PredictiveSymbolicPotential(
                potential.id,
                potential.strength,
                potential.nonlinearity
            )
            
            # Copy associations
            for assoc_id, assoc_strength in potential._associations.items():
                predictive_potential.add_association(assoc_id, assoc_strength)
                
            # Copy activations
            predictive_potential._activations = potential._activations.copy()
            
            # Add to symbol
            predictive_symbol.potentials[potential_id] = predictive_potential
            
        # Add to collection
        predictive_symbols[symbol_id] = predictive_symbol
    
    # Replace symbols in layer
    layer.symbols = predictive_symbols
    
    logger.info("Conversion to predictive variants complete")
    return layer



# === FILE: asf\layer2_autopoietic_maintanance\predictive_layer.py ===

import time
import joblib
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

from asf.symbolic_formation.symbolic_layer import SymbolicFormationLayer
from asf.symbolic_formation.predictive_recognition import PredictiveSymbolRecognizer
from asf.symbolic_formation.counterfactual_network import CounterfactualAutocatalyticNetwork
from asf.symbolic_formation.predictive_processor import SymbolicPredictiveProcessor

class PredictiveSymbolicFormationLayer(SymbolicFormationLayer):
    """
    Enhanced symbolic formation layer implementing Seth's Data Paradox principles.
    Integrates predictive processing and counterfactual reasoning for improved
    symbol formation and recognition.
    """
    def __init__(self, config=None):
        super().__init__(config)
        # Replace standard components with predictive variants
        self.recognizer = PredictiveSymbolRecognizer(
            threshold=self.config.get('recognition_threshold', 0.7)
        )
        
        # Create predictive processor for coordinating predictions
        self.predictive_processor = SymbolicPredictiveProcessor()
        
        # Replace autocatalytic network with counterfactual version
        self.autocatalytic_network = CounterfactualAutocatalyticNetwork(self.nonlinearity_tracker)
        
        # Track perceptual predictions
        self.perceptual_predictions = {}
        self.perceptual_prediction_errors = defaultdict(list)
    
    async def process_perceptual_input(self, perceptual_data, context=None):
        """Override to include predictive processing"""
        # First predict what we expect to see
        context = context or {}
        predictions = await self.generate_perceptual_predictions(context)
        
        # Store prediction for later evaluation
        prediction_id = f"perceptual_{int(time.time())}"
        self.perceptual_predictions[prediction_id] = {
            'context': context,
            'predictions': predictions,
            'timestamp': time.time()
        }
        
        # Process input normally
        result = await super().process_perceptual_input(perceptual_data, context)
        
        # Add prediction info to result
        result['prediction_id'] = prediction_id
        
        # Evaluate prediction accuracy
        if prediction_id in self.perceptual_predictions:
            evaluation = self.evaluate_perceptual_prediction(
                prediction_id, perceptual_data
            )
            result['prediction_evaluation'] = evaluation
        
        return result
        
    async def generate_perceptual_predictions(self, context):
        """
        Generate predictions about perceptual entities that should appear
        based on activated symbols. These can be sent to Layer 1 to guide
        perception.
        """
        # Get most relevant symbols for this context
        context_hash = joblib.hash(context)
        relevant_symbols = self._get_context_relevant_symbols(context)
        
        # For each symbol, get perceptual anchors that would be expected
        predictions = defaultdict(dict)
        
        for symbol_id, relevance in relevant_symbols.items():
            symbol = self.symbols[symbol_id]
            
            # Extract perceptual anchors as predictions
            for anchor, strength in symbol.perceptual_anchors.items():
                # Parse anchor to get entity type and feature
                if ":" in anchor:
                    entity_type, feature = anchor.split(":", 1)
                    predictions[entity_type][feature] = max(
                        predictions[entity_type].get(feature, 0.0),
                        strength * relevance
                    )
        
        return dict(predictions)
    
    def _get_context_relevant_symbols(self, context):
        """Get symbols relevant to the current context with scores"""
        context_hash = joblib.hash(context)
        relevant_symbols = {}
        
        # Check each symbol's relevance to this context
        for symbol_id, symbol in self.symbols.items():
            # Actualize meaning to check relevance
            meaning = symbol.actualize_meaning(context_hash, context)
            if meaning:
                # Calculate overall relevance from meaning
                relevance = sum(meaning.values()) / max(1, len(meaning))
                if relevance > 0.2:  # Threshold for relevance
                    relevant_symbols[symbol_id] = relevance
        
        return relevant_symbols
    
    def evaluate_perceptual_prediction(self, prediction_id, actual_perceptual):
        """
        Evaluate a perceptual prediction against actual data
        """
        if prediction_id not in self.perceptual_predictions:
            return {'error': 'Prediction not found'}
        
        prediction = self.perceptual_predictions[prediction_id]
        predicted = prediction['predictions']
        
        # Flatten actual perceptual data
        flat_actual = {}
        for entity_id, features in actual_perceptual.items():
            entity_type = entity_id.split('_')[0]  # Extract type from ID
            for feature, value in features.items():
                flat_actual[f"{entity_type}:{feature}"] = value
        
        # Flatten predictions
        flat_predicted = {}
        for entity_type, features in predicted.items():
            for feature, value in features.items():
                flat_predicted[f"{entity_type}:{feature}"] = value
        
        # Calculate true positives, false positives, false negatives
        tp = 0
        fp = 0
        fn = 0
        
        # True positives and false positives
        for key, pred_value in flat_predicted.items():
            if pred_value > 0.3:  # Prediction threshold
                if key in flat_actual and flat_actual[key] > 0.3:
                    tp += 1
                else:
                    fp += 1
        
        # False negatives
        for key, actual_value in flat_actual.items():
            if actual_value > 0.3 and (key not in flat_predicted or flat_predicted[key] < 0.3):
                fn += 1
        
        # Calculate precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        }
    
    def run_counterfactual_simulation(self, perceptual_data, modification_rules, context=None):
        """
        Run a counterfactual simulation to see what symbols would form
        under modified perceptual conditions
        """
        context = context or {}
        
        # Process the actual perceptual data (async function needs to be called differently)
        loop = asyncio.get_event_loop()
        actual_result = loop.run_until_complete(
            self.process_perceptual_input(perceptual_data, context)
        )
        
        # Get all newly created symbols
        actual_symbols = {}
        for symbol_info in actual_result.get('new_symbols', []):
            symbol_id = symbol_info.get('symbol_id')
            if symbol_id in self.symbols:
                actual_symbols[symbol_id] = self.symbols[symbol_id]
        
        # Generate counterfactual symbols
        cf_symbols = self.autocatalytic_network.generate_counterfactual_symbols(
            self.symbols,
            self._flatten_perceptual_data(perceptual_data),
            modification_rules
        )
        
        # Compare outcomes
        comparison = self.autocatalytic_network.compare_counterfactual_outcomes(
            actual_symbols, cf_symbols
        )
        
        return {
            'actual_symbols': [symbol_id for symbol_id in actual_symbols],
            'counterfactual_symbols': [symbol_id for symbol_id in cf_symbols],
            'comparison': comparison,
            'modifications': modification_rules
        }
    
    def _flatten_perceptual_data(self, perceptual_data):
        """Flatten hierarchical perceptual data into key-value pairs"""
        flat_data = {}
        for entity_id, features in perceptual_data.items():
            for feature_name, value in features.items():
                key = f"{entity_id}:{feature_name}"
                flat_data[key] = value
        return flat_data



# === FILE: asf\layer2_autopoietic_maintanance\predictive_potentials.py ===

import time
import numpy as np
import joblib
from collections import defaultdict
from typing import Dict, Any, List, Optional, Set, Tuple

from asf.symbolic_formation.potentials import SymbolicPotential
from asf.symbolic_formation.enums import NonlinearityOrder

class PredictiveSymbolicPotential(SymbolicPotential):
    """
    Symbolic potential with prediction capabilities.
    Uses precision-weighted activation for better uncertainty handling.
    """
    def __init__(self, id: str, strength: float = 1.0, nonlinearity: NonlinearityOrder = NonlinearityOrder.LINEAR):
        super().__init__(id, strength, nonlinearity)
        self.predicted_activations = {}  # Context hash -> predicted activation
        self.prediction_errors = defaultdict(list)  # Context hash -> list of errors
        self.precision_values = {}  # Context hash -> precision value
        
    def predict_activation(self, context: Dict[str, Any]) -> Optional[float]:
        """Predict activation before context actualization"""
        context_hash = joblib.hash(context)
        
        # If we don't have enough past activations, can't predict
        if len(self._activations) < 3:
            return None
            
        # Find similar contexts (simplified implementation)
        similar_contexts = list(self._activations.keys())[:5]  # Use 5 most recent contexts
        
        if not similar_contexts:
            return None
        
        # Calculate predicted activation as weighted average of similar contexts
        total_weight = 0.0
        weighted_sum = 0.0
        
        for past_hash in similar_contexts:
            weight = 1.0  # Equal weighting for simplicity
            activation = self._activations[past_hash]
            
            weighted_sum += activation * weight
            total_weight += weight
        
        if total_weight > 0:
            predicted = weighted_sum / total_weight
            self.predicted_activations[context_hash] = predicted
            return predicted
            
        return None
        
    def actualize(self, context: Dict[str, Any], potential_network: Optional[Dict[str, 'SymbolicPotential']] = None) -> float:
        """Predict activation, then actualize and calculate error"""
        context_hash = joblib.hash(context)
        
        # First make prediction
        predicted = self.predict_activation(context)
        
        # Then get actual activation
        actual = super().actualize(context, potential_network)
        
        # Calculate prediction error if prediction was made
        if predicted is not None:
            error = abs(predicted - actual)
            self.prediction_errors[context_hash].append(error)
            
            # Limit history size
            if len(self.prediction_errors[context_hash]) > 20:
                self.prediction_errors[context_hash] = self.prediction_errors[context_hash][-20:]
            
            # Update precision (inverse variance)
            if len(self.prediction_errors[context_hash]) > 1:
                variance = np.var(self.prediction_errors[context_hash])
                precision = 1.0 / (variance + 1e-6)  # Avoid division by zero
                self.precision_values[context_hash] = min(10.0, precision)  # Cap very high precision
        
        return actual
    
    def get_precision(self, context_hash=None):
        """Get precision for a context or overall"""
        if context_hash and context_hash in self.precision_values:
            return self.precision_values[context_hash]
        
        # Calculate overall precision
        all_errors = []
        for errors in self.prediction_errors.values():
            all_errors.extend(errors)
            
        if len(all_errors) < 2:
            return 1.0  # Default precision
            
        variance = np.var(all_errors)
        precision = 1.0 / (variance + 1e-6)  # Avoid division by zero
        return min(10.0, precision)  # Cap very high precision



# === FILE: asf\layer2_autopoietic_maintanance\predictive_processor.py ===

import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

class SymbolicPredictiveProcessor:
    """
    Coordinates predictive processing across Layer 2 components.
    Implements Seth's predictive processing principles for symbolic operations.
    """
    
    def __init__(self):
        self.predictions = {}  # Prediction ID -> prediction data
        self.prediction_errors = defaultdict(list)  # Entity ID -> errors
        self.precision_values = {}  # Entity ID -> precision value
        self.learning_rates = {}  # Entity ID -> adaptive learning rate
        
    def register_prediction(self, context_id: str, entity_id: str, prediction: Any) -> str:
        """Register a prediction for later evaluation"""
        prediction_id = f"{context_id}_{entity_id}_{int(time.time()*1000)}"
        self.predictions[prediction_id] = {
            'context_id': context_id,
            'entity_id': entity_id,
            'value': prediction,
            'timestamp': time.time(),
            'evaluated': False
        }
        return prediction_id
        
    def evaluate_prediction(self, prediction_id: str, actual_value: Any) -> Optional[Dict[str, Any]]:
        """Evaluate prediction against actual value"""
        if prediction_id not in self.predictions:
            return None
            
        prediction = self.predictions[prediction_id]
        if prediction['evaluated']:
            return None
            
        # Calculate prediction error
        predicted = prediction['value']
        error = self._calculate_error(predicted, actual_value)
        
        # Update prediction record
        prediction['evaluated'] = True
        prediction['actual_value'] = actual_value
        prediction['error'] = error
        prediction['evaluation_time'] = time.time()
        
        # Track error for precision calculation
        entity_id = prediction['entity_id']
        self.prediction_errors[entity_id].append(error)
        
        # Limit history size
        if len(self.prediction_errors[entity_id]) > 20:
            self.prediction_errors[entity_id] = self.prediction_errors[entity_id][-20:]
        
        # Update precision (inverse variance)
        if len(self.prediction_errors[entity_id]) > 1:
            variance = np.var(self.prediction_errors[entity_id])
            precision = 1.0 / (variance + 1e-6)  # Avoid division by zero
            self.precision_values[entity_id] = min(10.0, precision)  # Cap very high precision
            
        # Calculate adaptive learning rate
        # Higher error = higher learning rate
        # Higher precision = lower learning rate (more cautious)
        precision = self.get_precision(entity_id)
        base_rate = min(0.8, error * 2)  # Error-proportional component
        precision_factor = max(0.1, min(0.9, 1.0 / (1.0 + precision * 0.2)))
        learning_rate = min(0.9, max(0.1, base_rate * precision_factor))
        self.learning_rates[entity_id] = learning_rate
        
        return {
            'prediction_id': prediction_id,
            'error': error,
            'precision': precision,
            'learning_rate': learning_rate
        }
        
    def get_precision(self, entity_id: str) -> float:
        """Get precision for a specific entity"""
        return self.precision_values.get(entity_id, 1.0)
        
    def get_learning_rate(self, entity_id: str) -> float:
        """Get adaptive learning rate for an entity"""
        return self.learning_rates.get(entity_id, 0.3)
        
    def _calculate_error(self, predicted: Any, actual: Any) -> float:
        """Calculate normalized error between predicted and actual values"""
        if isinstance(predicted, (int, float, np.number)) and isinstance(actual, (int, float, np.number)):
            # For numeric values, normalized absolute difference
            return abs(predicted - actual) / (1.0 + abs(actual))
            
        elif isinstance(predicted, (list, np.ndarray)) and isinstance(actual, (list, np.ndarray)):
            # For vectors, normalized Euclidean distance
            predicted_arr = np.array(predicted)
            actual_arr = np.array(actual)
            
            if predicted_arr.shape != actual_arr.shape:
                return 1.0  # Maximum error for shape mismatch
                
            if predicted_arr.size == 0 or actual_arr.size == 0:
                return 1.0  # Maximum error for empty arrays
                
            # Normalized Euclidean distance
            diff = np.linalg.norm(predicted_arr - actual_arr)
            norm = np.linalg.norm(actual_arr)
            return min(1.0, diff / (1.0 + norm))
            
        elif isinstance(predicted, dict) and isinstance(actual, dict):
            # For dictionaries, calculate average error across shared keys
            shared_keys = set(predicted.keys()) & set(actual.keys())
            
            if not shared_keys:
                return 1.0  # Maximum error if no shared keys
                
            errors = []
            for key in shared_keys:
                errors.append(self._calculate_error(predicted[key], actual[key]))
                
            return sum(errors) / len(errors)
            
        else:
            # Fallback for other types
            return 1.0 if predicted != actual else 0.0



# === FILE: asf\layer2_autopoietic_maintanance\predictive_recognition.py ===

import time
import joblib
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple

from asf.symbolic_formation.recognition import SymbolRecognizer
from asf.symbolic_formation.symbol import SymbolElement

class PredictiveSymbolRecognizer(SymbolRecognizer):
    """
    Enhances standard symbol recognizer with predictive capabilities.
    Anticipates which symbols will appear before receiving perceptual data.
    """
    def __init__(self, threshold: float = 0.7):
        super().__init__(threshold)
        self.prediction_cache = {}  # Context hash -> predicted symbols
        self.prediction_errors = defaultdict(list)  # Track errors for precision
        self.precision_values = {}  # Symbol ID -> precision value
        
    async def predict_symbols(self, context, existing_symbols):
        """Predict which symbols are likely to be recognized in this context"""
        context_hash = joblib.hash(context)
        
        # Check cache first
        if context_hash in self.prediction_cache:
            # Return cached prediction if not too old
            cached = self.prediction_cache[context_hash]
            if time.time() - cached['timestamp'] < 300:  # 5 minutes validity
                return cached['predictions']
        
        # Generate predictions based on symbol relevance to context
        predictions = {}
        for symbol_id, symbol in existing_symbols.items():
            # Calculate context relevance based on actualized meaning
            meaning = symbol.actualize_meaning(context_hash, context)
            if meaning:
                # Use total activation as prediction confidence
                confidence = sum(meaning.values()) / len(meaning)
                predictions[symbol_id] = min(0.95, confidence)
            
        # Store prediction in cache
        self.prediction_cache[context_hash] = {
            'predictions': predictions,
            'timestamp': time.time()
        }
        
        return predictions
        
    async def recognize(self, perceptual_data, existing_symbols, context=None):
        """First predict symbols, then compare with actual recognition"""
        context = context or {}
        context_hash = joblib.hash(context)
        
        # Make prediction before actual recognition
        predictions = await self.predict_symbols(context, existing_symbols)
        
        # Perform actual recognition
        result = await super().recognize(perceptual_data, existing_symbols, context)
        
        # Calculate prediction error if recognition was successful
        if result['recognized']:
            symbol_id = result['symbol_id']
            predicted_confidence = predictions.get(symbol_id, 0.0)
            prediction_error = abs(predicted_confidence - result['confidence'])
            
            # Track prediction error
            self.prediction_errors[symbol_id].append(prediction_error)
            
            # Limit history size
            if len(self.prediction_errors[symbol_id]) > 20:
                self.prediction_errors[symbol_id] = self.prediction_errors[symbol_id][-20:]
            
            # Update precision (inverse variance of prediction errors)
            if len(self.prediction_errors[symbol_id]) > 1:
                variance = np.var(self.prediction_errors[symbol_id])
                precision = 1.0 / (variance + 1e-6)  # Avoid division by zero
                self.precision_values[symbol_id] = min(10.0, precision)  # Cap very high precision
            
            # Add prediction information to result
            result['predicted_confidence'] = predicted_confidence
            result['prediction_error'] = prediction_error
            result['precision'] = self.precision_values.get(symbol_id, 1.0)
            
        return result
    
    def get_prediction_precision(self, symbol_id):
        """Calculate precision (inverse variance) of predictions for a symbol"""
        errors = self.prediction_errors.get(symbol_id, [])
        if len(errors) < 2:
            return 1.0  # Default precision
            
        variance = np.var(errors)
        precision = 1.0 / (variance + 1e-6)  # Avoid division by zero
        return min(10.0, precision)  # Cap very high precision



# === FILE: asf\layer2_autopoietic_maintanance\predictive_symbol.py ===

import time
import math
import numpy as np
import joblib
from collections import defaultdict
from typing import Dict, Any, List, Optional, Set, Tuple

from asf.symbolic_formation.symbol import SymbolElement
from asf.symbolic_formation.potentials import SymbolicPotential
from asf.symbolic_formation.enums import NonlinearityOrder, SymbolConfidenceState

class PredictiveSymbolElement(SymbolElement):
    """
    Enhanced symbol with predictive capabilities.
    Predicts meaning activations before observing context.
    """
    def __init__(self, symbol_id: str, perceptual_anchors: Dict[str, float] = None):
        super().__init__(symbol_id, perceptual_anchors)
        self.predicted_meanings = {}  # Context hash -> predicted meanings
        self.prediction_errors = defaultdict(list)  # Context -> errors
        self.precision_values = {}  # Context -> precision (1/variance)
        
    def predict_meaning(self, context_hash: str, context: Dict[str, Any]) -> Dict[str, float]:
        """Predict meaning before actualization based on similar contexts"""
        # If we don't have enough historical data, can't predict meaningfully
        if len(self._actual_meanings) < 3:
            return {}
            
        # Find similar contexts based on keys
        similar_contexts = []
        context_keys = set(context.keys())
        
        for past_hash in self._actual_meanings:
            # Simple similarity heuristic - add to similar if we have actual data
            if past_hash in self._actual_meanings:
                similar_contexts.append(past_hash)
                
        if not similar_contexts:
            return {}
            
        # Calculate predicted meaning based on similar contexts
        predicted = {}
        weights = {}
        
        for past_hash in similar_contexts:
            past_meaning = self._actual_meanings[past_hash]
            
            # Weight by recency (more recent contexts get higher weight)
            weight = 1.0
            if past_hash in self._activation_time:
                # Decay weight by time since activation
                time_elapsed = time.time() - self._activation_time[past_hash]
                decay_factor = math.exp(-time_elapsed / 3600)  # 1-hour half-life
                weight *= decay_factor
            
            # Accumulate weighted meanings
            for potential_id, activation in past_meaning.items():
                if potential_id not in predicted:
                    predicted[potential_id] = 0.0
                    weights[potential_id] = 0.0
                
                predicted[potential_id] += activation * weight
                weights[potential_id] += weight
        
        # Normalize by weights
        for potential_id in predicted:
            if weights[potential_id] > 0:
                predicted[potential_id] /= weights[potential_id]
        
        # Store prediction
        self.predicted_meanings[context_hash] = predicted
        return predicted
        
    def actualize_meaning(self, context_hash: str, context: Dict[str, Any]) -> Dict[str, float]:
        """First predict meaning, then actualize and compare"""
        # Make prediction if we have prior contexts
        predicted = {}
        if len(self._actual_meanings) >= 3:
            predicted = self.predict_meaning(context_hash, context)
        
        # Actual actualization from parent class
        actualized = super().actualize_meaning(context_hash, context)
        
        # Calculate prediction error if prediction was made
        if predicted:
            # Calculate error for each potential
            errors = []
            for potential_id in set(list(predicted.keys()) + list(actualized.keys())):
                pred_value = predicted.get(potential_id, 0.0)
                actual_value = actualized.get(potential_id, 0.0)
                error = abs(pred_value - actual_value)
                errors.append(error)
            
            # Average error across all potentials
            if errors:
                avg_error = sum(errors) / len(errors)
                self.prediction_errors[context_hash].append(avg_error)
                
                # Calculate precision (inverse variance)
                if len(self.prediction_errors[context_hash]) > 1:
                    variance = np.var(self.prediction_errors[context_hash])
                    precision = 1.0 / (variance + 1e-6)
                    self.precision_values[context_hash] = precision
        
        return actualized
    
    def get_prediction_precision(self, context_hash=None):
        """Get prediction precision for a context or overall"""
        if context_hash and context_hash in self.precision_values:
            return self.precision_values[context_hash]
            
        # Calculate overall precision across all contexts
        all_errors = []
        for errors in self.prediction_errors.values():
            all_errors.extend(errors)
            
        if len(all_errors) < 2:
            return 1.0  # Default precision
            
        variance = np.var(all_errors)
        precision = 1.0 / (variance + 1e-6)
        return min(10.0, precision)  # Cap very high precision



# === FILE: asf\layer2_autopoietic_maintanance\recognition.py ===

import time
import numpy as np
import asyncio
from typing import Dict, List, Any, Tuple
from collections import defaultdict

from asf.symbolic_formation.symbol import SymbolElement

class SymbolRecognizer:
    """
    Recognizes existing symbols from perceptual data.
    Enhanced with multi-strategy recognition approaches.
    """
    def __init__(self, threshold: float = 0.7):
        self.recognition_threshold = threshold
        self.recognition_history = []
        # Phase 2 enhancement: multiple recognition strategies
        self.strategies = {
            'anchor_matching': self._recognize_by_anchors,
            'embedding_similarity': self._recognize_by_embedding,
            'feature_mapping': self._recognize_by_feature_mapping
        }
        self.strategy_weights = {
            'anchor_matching': 0.6,
            'embedding_similarity': 0.2,
            'feature_mapping': 0.2
        }
    
    async def recognize(self, perceptual_data: Dict[str, Dict[str, float]],
                      existing_symbols: Dict[str, SymbolElement],
                      context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Recognize symbols from perceptual data using multiple strategies.
        
        Args:
            perceptual_data: Dictionary of perceptual features
            existing_symbols: Dictionary of existing symbols
            context: Optional context information
            
        Returns:
            Recognition result
        """
        context = context or {}
        
        # Flatten perceptual data for processing
        flat_perceptual = self._flatten_perceptual_data(perceptual_data)
        
        # Results from each strategy
        strategy_results = {}
        
        # Apply each recognition strategy
        for strategy_name, strategy_func in self.strategies.items():
            weight = self.strategy_weights[strategy_name]
            result = await strategy_func(flat_perceptual, existing_symbols, context)
            
            if result['recognized']:
                strategy_results[strategy_name] = {
                    'symbol_id': result['symbol_id'],
                    'confidence': result['confidence'],
                    'weighted_confidence': result['confidence'] * weight
                }
                
        # Combine strategy results
        if not strategy_results:
            return {
                'recognized': False,
                'confidence': 0.0,
                'strategies_applied': list(self.strategies.keys())
            }
            
        # Find best match across strategies
        best_strategy = max(strategy_results.items(), 
                          key=lambda x: x[1]['weighted_confidence'])
        
        strategy_name, result = best_strategy
        weighted_confidence = result['weighted_confidence']
        
        # Final decision based on confidence threshold
        if weighted_confidence >= self.recognition_threshold:
            self.recognition_history.append({
                'timestamp': time.time(),
                'symbol_id': result['symbol_id'],
                'confidence': result['confidence'],
                'weighted_confidence': weighted_confidence,
                'strategy': strategy_name
            })
            
            return {
                'recognized': True,
                'symbol_id': result['symbol_id'],
                'confidence': result['confidence'],
                'weighted_confidence': weighted_confidence,
                'strategy': strategy_name,
                'strategies_applied': list(strategy_results.keys())
            }
            
        return {
            'recognized': False,
            'confidence': weighted_confidence,
            'best_match': result['symbol_id'],
            'strategies_applied': list(self.strategies.keys())
        }
    
    async def _recognize_by_anchors(self, flat_perceptual: Dict[str, float],
                                 existing_symbols: Dict[str, SymbolElement],
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recognize symbols based on perceptual anchor matching.
        """
        best_match = None
        best_score = 0.0
        
        # Group symbols by anchor keys for faster matching
        anchor_to_symbols = defaultdict(list)
        for symbol_id, symbol in existing_symbols.items():
            for anchor in symbol.perceptual_anchors:
                anchor_to_symbols[anchor].append(symbol_id)
                
        # For each perceptual feature, check candidate symbols
        candidates = {}
        for feature, strength in flat_perceptual.items():
            if feature in anchor_to_symbols and strength > 0.3:  # Threshold
                for symbol_id in anchor_to_symbols[feature]:
                    if symbol_id not in candidates:
                        candidates[symbol_id] = 0
                    candidates[symbol_id] += strength
                    
        # Detailed perceptual match for candidates
        for symbol_id, initial_score in candidates.items():
            symbol = existing_symbols[symbol_id]
            match_score = self._calculate_perceptual_match(symbol, flat_perceptual)
            
            if match_score > best_score:
                best_score = match_score
                best_match = symbol_id
                
        return {
            'recognized': best_score >= self.recognition_threshold,
            'symbol_id': best_match,
            'confidence': best_score
        }
    
    async def _recognize_by_embedding(self, flat_perceptual: Dict[str, float],
                                   existing_symbols: Dict[str, SymbolElement],
                                   context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recognize symbols based on embedding similarity.
        """
        # This is a simplified implementation. In a real system, this would use  
        # feature embeddings and compute semantic similarity.
        if not flat_perceptual:
            return {'recognized': False, 'symbol_id': None, 'confidence': 0.0}
            
        # Create a simplified feature vector from flat_perceptual
        percept_vec = np.zeros(128)
        for i, (key, value) in enumerate(flat_perceptual.items()):
            hash_val = hash(key) % 128
            percept_vec[hash_val] = value
            
        # Normalize
        norm = np.linalg.norm(percept_vec)
        if norm > 0:
            percept_vec = percept_vec / norm
            
        # Find most similar symbol
        best_match = None
        best_similarity = 0.0
        
        for symbol_id, symbol in existing_symbols.items():
            # Create similar simplified vector for symbol
            sym_vec = np.zeros(128)
            for anchor, strength in symbol.perceptual_anchors.items():
                hash_val = hash(anchor) % 128
                sym_vec[hash_val] = strength
                
            # Normalize
            norm = np.linalg.norm(sym_vec)
            if norm > 0:
                sym_vec = sym_vec / norm
                
            # Calculate cosine similarity
            similarity = np.dot(percept_vec, sym_vec)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = symbol_id
                
        return {
            'recognized': best_similarity >= self.recognition_threshold,
            'symbol_id': best_match,
            'confidence': best_similarity
        }
    
    async def _recognize_by_feature_mapping(self, flat_perceptual: Dict[str, float],
                                         existing_symbols: Dict[str, SymbolElement],
                                         context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recognize symbols based on detailed feature mapping.
        """
        if not existing_symbols:
            return {'recognized': False, 'symbol_id': None, 'confidence': 0.0}
            
        # Calculate match scores for all symbols
        match_scores = []
        for symbol_id, symbol in existing_symbols.items():
            context_hash = str(hash(str(context)))
            
            # Get actualized meaning in current context
            meaning = symbol.actualize_meaning(context_hash, context)
            
            # Calculate mapping between perceptual data and meaning
            match_score = self._calculate_feature_mapping(flat_perceptual, meaning)
            match_scores.append((symbol_id, match_score))
            
        # Find best match
        if not match_scores:
            return {'recognized': False, 'symbol_id': None, 'confidence': 0.0}
            
        best_match = max(match_scores, key=lambda x: x[1])
        symbol_id, score = best_match
        
        return {
            'recognized': score >= self.recognition_threshold,
            'symbol_id': symbol_id,
            'confidence': score
        }
    
    def _flatten_perceptual_data(self, perceptual_data: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Flatten hierarchical perceptual data into a simple key-value dictionary.
        """
        flat_data = {}
        for entity_id, features in perceptual_data.items():
            for feature_name, value in features.items():
                key = f"{entity_id}:{feature_name}"
                flat_data[key] = value
        return flat_data
    
    def _calculate_perceptual_match(self, symbol: SymbolElement, perceptual: Dict[str, float]) -> float:
        """
        Calculate match score between symbol anchors and perceptual data.
        """
        if not symbol.perceptual_anchors or not perceptual:
            return 0.0
            
        # Calculate overlap between anchors and perceptual features
        overlap_score = 0.0
        total_weight = 0.0
        
        for anchor, anchor_strength in symbol.perceptual_anchors.items():
            # Look for exact matches
            if anchor in perceptual:
                overlap_score += anchor_strength * perceptual[anchor]
                total_weight += anchor_strength
            else:
                # Look for partial matches
                for percept_key, percept_value in perceptual.items():
                    if anchor in percept_key or (isinstance(percept_key, str) and percept_key in anchor):
                        partial_score = 0.7 * anchor_strength * percept_value  # Reduce score for partial match
                        overlap_score += partial_score
                        total_weight += anchor_strength
                        break
        
        # Normalize score
        if total_weight > 0:
            return overlap_score / total_weight
        return 0.0
    
    def _calculate_feature_mapping(self, perceptual: Dict[str, float], meaning: Dict[str, float]) -> float:
        """
        Calculate mapping between perceptual features and symbol meaning.
        """
        if not perceptual or not meaning:
            return 0.0
            
        # Create simplified feature vectors
        perc_vec = np.zeros(256)
        mean_vec = np.zeros(256)
        
        # Fill perceptual vector
        for key, value in perceptual.items():
            idx = hash(key) % 256
            perc_vec[idx] = value
            
        # Fill meaning vector
        for key, value in meaning.items():
            idx = hash(key) % 256
            mean_vec[idx] = value
            
        # Calculate cosine similarity
        perc_norm = np.linalg.norm(perc_vec)
        mean_norm = np.linalg.norm(mean_vec)
        
        if perc_norm > 0 and mean_norm > 0:
            similarity = np.dot(perc_vec, mean_vec) / (perc_norm * mean_norm)
            return similarity
            
        return 0.0



# === FILE: asf\layer2_autopoietic_maintanance\resolution_strategies.py ===




# === FILE: asf\layer2_autopoietic_maintanance\symbol.py ===

import time
import numpy as np
from typing import Dict, Any, List
from asf.symbolic_formation.enums import NonlinearityOrder, SymbolConfidenceState
from asf.symbolic_formation.potentials import SymbolicPotential

class SymbolElement:
    """
    Represents a symbol with its internal structure and potentials.
    Optimized for efficient operations on potential networks.
    Enhanced with Bayesian confidence updating.
    """
    def __init__(self, symbol_id: str, perceptual_anchors: Dict[str, float] = None):
        self.id = symbol_id
        self.name = f"symbol_{symbol_id[-8:]}"  # Auto-generate a name based on ID
        self.perceptual_anchors = perceptual_anchors or {}
        self.potentials: Dict[str, SymbolicPotential] = {}
        
        # Phase 2 enhancement: Bayesian confidence
        self.confidence = 0.5  # Initial confidence
        self.confidence_state = SymbolConfidenceState.HYPOTHESIS
        self.confidence_evidence = {'positive': 1, 'negative': 1}  # Bayesian priors
        
        # Use sparse dictionary for actual meanings to save memory
        self._actual_meanings: Dict[str, Dict[str, float]] = {}
        
        # Performance tracking
        self._activation_time: Dict[str, float] = {}
        self._nonlinearity = NonlinearityOrder.LINEAR
        
        # Phase 1 enhancement: source tracking
        self.source_entities = []
        self.created_at = time.time()
        self.last_accessed = time.time()
    
    def add_potential(self, potential: SymbolicPotential) -> None:
        """Add a meaning potential to this symbol."""
        self.potentials[potential.id] = potential
        
        # Update symbol's nonlinearity based on potentials
        if potential.nonlinearity.value > self._nonlinearity.value:
            self._nonlinearity = potential.nonlinearity
    
    def actualize_meaning(self, context_hash: str, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Actualize meaning in a specific context using maximization-based approach.
        Returns dictionary of actualized meaning aspects with strengths.
        """
        start_time = time.time()
        
        # Create potential network for propagation
        potential_network = {}
        for potential_id, potential in self.potentials.items():
            key = f"{self.id}:{potential_id}"
            potential_network[key] = potential
            
        # If already actualized in this context, return cached
        if context_hash in self._actual_meanings:
            return self._actual_meanings[context_hash]
            
        # Actualize potentials in this context using maximization-based approach
        actualized = {}
        for potential_id, potential in self.potentials.items():
            # Pass potential network for MGC-inspired propagation
            activation = potential.actualize(context, potential_network)
            if activation > 0.2:  # Threshold for inclusion
                actualized[potential_id] = activation
                
        # Store for future reference
        self._actual_meanings[context_hash] = actualized
        
        # Track performance
        self._activation_time[context_hash] = time.time() - start_time
        self.last_accessed = time.time()
        
        return actualized
    
    def get_pregnancy(self) -> float:
        """
        Calculate symbolic pregnancy - richness of potential meanings.
        Higher values indicate more meaning potential.
        """
        if not self.potentials:
            return 0.0
            
        # Use maximization-based approach rather than sum
        pregnancy_values = [
            p.strength * (1 + len(p._associations) * 0.1)
            for p in self.potentials.values()
        ]
        
        return max(pregnancy_values) * len(self.potentials)
    
    def get_nonlinearity(self) -> NonlinearityOrder:
        """Get the symbol's nonlinearity order."""
        return self._nonlinearity
    
    def clear_caches(self) -> None:
        """Clear cached calculations."""
        # Clear instance caches
        self._actual_meanings = {}
        self._activation_time = {}
    
    # Phase 2 enhancement: Bayesian confidence updating
    def update_confidence(self, new_evidence: bool, weight: float = 1.0) -> float:
        """
        Update symbol confidence using Bayesian updating.
        
        Args:
            new_evidence: True for positive evidence, False for negative
            weight: Weight of evidence (1.0 = standard weight)
            
        Returns:
            Updated confidence value
        """
        # Apply evidence to Bayesian model
        if new_evidence:
            self.confidence_evidence['positive'] += weight
        else:
            self.confidence_evidence['negative'] += weight
            
        # Calculate confidence from evidence
        p = self.confidence_evidence['positive']
        n = self.confidence_evidence['negative']
        self.confidence = p / (p + n)
        
        # Update confidence state
        if self.confidence >= 0.8:
            self.confidence_state = SymbolConfidenceState.CANONICAL
        elif self.confidence >= 0.5:
            self.confidence_state = SymbolConfidenceState.PROVISIONAL
        else:
            self.confidence_state = SymbolConfidenceState.HYPOTHESIS
            
        return self.confidence
    
    def add_source_entity(self, entity_id: str) -> None:
        """Add a source entity ID reference."""
        if entity_id not in self.source_entities:
            self.source_entities.append(entity_id)



# === FILE: asf\layer3_cognitive_boundary\__init__.py ===

# Layer 3: Semantic Organization Layer
# Main module exports

from asf.semantic_organization.enums import SemanticNodeType, SemanticConfidenceState
from asf.semantic_organization.temporal import AdaptiveTemporalMetadata
from asf.semantic_organization.predictive_processor import PredictiveProcessor
from asf.semantic_organization.semantic_layer import SemanticOrganizationLayer

# Core components
from asf.semantic_organization.core import SemanticNode, SemanticRelation, SemanticTensorNetwork

# Formation systems
from asf.semantic_organization.formation import (
    ConceptFormationEngine, ConceptualBlendingEngine, CategoryFormationSystem
)

# Processing components
from asf.semantic_organization.processing import AsyncProcessingQueue, AdaptivePriorityManager

# Resolution components
from asf.semantic_organization.resolution import ConflictDetectionEngine

__all__ = [
    'SemanticNodeType', 'SemanticConfidenceState', 'AdaptiveTemporalMetadata',
    'PredictiveProcessor', 'SemanticOrganizationLayer',
    'SemanticNode', 'SemanticRelation', 'SemanticTensorNetwork',
    'ConceptFormationEngine', 'ConceptualBlendingEngine', 'CategoryFormationSystem',
    'AsyncProcessingQueue', 'AdaptivePriorityManager',
    'ConflictDetectionEngine'
]



# === FILE: asf\layer3_cognitive_boundary\active_inference.py ===

# Create a new file: active_inference.py

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

class ActiveInferenceController:
    """
    Implements system-wide active inference for Layer 3.
    Coordinates predictive processing across all components.
    Embodies Seth's principle of minimizing prediction error through actions.
    """
    def __init__(self, semantic_layer):
        self.semantic_layer = semantic_layer
        self.logger = logging.getLogger("ASF.Layer3.ActiveInference")
        self.anticipated_states = {}
        self.prediction_errors = defaultdict(list)
        self.active_inference_history = []
        self.optimization_targets = {
            'contradiction_reduction': 0.4,  # Weight given to reducing contradictions
            'category_coherence': 0.3,       # Weight given to improving category coherence
            'structural_efficiency': 0.3      # Weight given to network efficiency
        }
        
    async def anticipate_state(self, planned_operations, time_horizon=1.0):
        """
        Anticipate future semantic network state based on planned operations.
        
        Args:
            planned_operations: List of planned semantic operations
            time_horizon: Time window for anticipation (in seconds)
            
        Returns:
            Anticipated semantic state
        """
        # Generate operation ID for tracking
        operation_id = f"ops_{int(time.time() * 1000)}"
        
        # Anticipate contradictions
        contradiction_result = await self.semantic_layer.conflict_detection.anticipate_contradictions(
            planned_operations
        )
        
        # Anticipate category formations/changes
        # (In a real implementation, would call into CategoryFormationSystem)
        category_anticipations = await self._anticipate_category_changes(planned_operations)
        
        # Anticipate concept formations
        concept_anticipations = await self._anticipate_concept_formations(planned_operations)
        
        # Create a composite anticipated state
        anticipated_state = {
            'operation_id': operation_id,
            'timestamp': time.time(),
            'time_horizon': time_horizon,
            'anticipated_contradictions': contradiction_result.get('anticipated_contradictions', []),
            'anticipated_categories': category_anticipations,
            'anticipated_concepts': concept_anticipations,
            'planned_operations': planned_operations
        }
        
        # Store for later evaluation
        self.anticipated_states[operation_id] = anticipated_state
        
        return anticipated_state
        
    async def perform_active_inference(self, anticipated_state, optimization_targets=None):
        """
        Perform active inference to minimize prediction error.
        Modifies planned operations to optimize future state.
        
        Args:
            anticipated_state: Previously anticipated state
            optimization_targets: Optional custom optimization weights
            
        Returns:
            Modified operations and expected improvements
        """
        if not anticipated_state:
            return {'status': 'error', 'message': 'No anticipated state provided'}
            
        # Use provided optimization targets or defaults
        targets = optimization_targets or self.optimization_targets
        
        # Original operations
        planned_operations = anticipated_state.get('planned_operations', [])
        
        # Create different operation variants to test
        operation_variants = await self._generate_operation_variants(planned_operations)
        
        # Evaluate each variant
        variant_scores = []
        
        for variant_name, variant_ops in operation_variants.items():
            # Anticipate state with this variant
            variant_state = await self.anticipate_state(variant_ops, time_horizon=0.5)
            
            # Score the variant
            score = self._score_anticipated_state(variant_state, targets)
            
            variant_scores.append((variant_name, variant_ops, score))
        
        # Sort by score (higher is better)
        variant_scores.sort(key=lambda x: x[2], reverse=True)
        
        if not variant_scores:
            return {'status': 'error', 'message': 'No viable operation variants found'}
            
        # Select best variant
        best_name, best_ops, best_score = variant_scores[0]
        
        # Record in history
        self.active_inference_history.append({
            'timestamp': time.time(),
            'original_operation_count': len(planned_operations),
            'selected_variant': best_name,
            'score_improvement': best_score - self._score_anticipated_state(anticipated_state, targets),
            'variant_count': len(variant_scores)
        })
        
        return {
            'status': 'success',
            'original_operations': planned_operations,
            'optimized_operations': best_ops,
            'variant_name': best_name,
            'score': best_score,
            'all_variants': len(variant_scores)
        }
        
    async def evaluate_anticipations(self, actual_state, operation_id):
        """
        Evaluate anticipation accuracy against actual state.
        Updates precision values based on prediction errors.
        
        Args:
            actual_state: Actual observed state
            operation_id: ID of the anticipated operation to evaluate
            
        Returns:
            Evaluation results
        """
        if operation_id not in self.anticipated_states:
            return {'status': 'error', 'message': 'No such anticipated state'}
            
        anticipated = self.anticipated_states[operation_id]
        
        # Track errors for different prediction types
        errors = {}
        
        # Evaluate contradiction predictions
        if 'anticipated_contradictions' in anticipated and 'actual_contradictions' in actual_state:
            contradiction_error = self._calculate_set_prediction_error(
                anticipated['anticipated_contradictions'],
                actual_state['actual_contradictions']
            )
            errors['contradictions'] = contradiction_error
            
            # Track error history
            self.prediction_errors['contradictions'].append(contradiction_error)
            
            # Limit history size
            if len(self.prediction_errors['contradictions']) > 20:
                self.prediction_errors['contradictions'] = self.prediction_errors['contradictions'][-20:]
        
        # Similar evaluations for other prediction types
        # ...
        
        # Calculate overall prediction accuracy
        if errors:
            overall_error = sum(errors.values()) / len(errors)
        else:
            overall_error = 0.0
            
        # Clean up old anticipations
        self._clean_old_anticipations()
        
        return {
            'status': 'success',
            'operation_id': operation_id,
            'overall_error': overall_error,
            'specific_errors': errors
        }
    
    def _score_anticipated_state(self, state, targets):
        """
        Score anticipated state based on optimization targets.
        Higher score is better.
        """
        score = 0.0
        
        # Score contradiction reduction
        contradiction_count = len(state.get('anticipated_contradictions', []))
        # Lower is better, so we use an inverse score
        contradiction_score = 1.0 / (1.0 + contradiction_count)
        score += contradiction_score * targets.get('contradiction_reduction', 0.4)
        
        # Score category coherence
        # (In a real implementation would have proper metrics)
        category_score = 0.7  # Placeholder
        score += category_score * targets.get('category_coherence', 0.3)
        
        # Score structural efficiency
        # (In a real implementation would have proper metrics)
        efficiency_score = 0.8  # Placeholder
        score += efficiency_score * targets.get('structural_efficiency', 0.3)
        
        return score
    
    async def _generate_operation_variants(self, operations):
        """
        Generate variations of operation sequences for testing.
        """
        variants = {
            'original': operations,
            'reordered': self._reorder_operations(operations),
            'pruned': self._prune_operations(operations),
            'enhanced': await self._enhance_operations(operations)
        }
        
        return variants
    
    def _reorder_operations(self, operations):
        """Reorder operations to minimize conflicts."""
        # Simple implementation - sort property changes to come after node creations
        node_ops = [op for op in operations if op.get('type') == 'create_node']
        prop_ops = [op for op in operations if op.get('type') in ('add_property', 'update_property')]
        other_ops = [op for op in operations if op.get('type') not in ('create_node', 'add_property', 'update_property')]
        
        return node_ops + prop_ops + other_ops
    
    def _prune_operations(self, operations):
        """Remove likely problematic operations."""
        # Simple implementation - remove operations with very low confidence
        return [op for op in operations if op.get('confidence', 0.5) > 0.3]
    
    async def _enhance_operations(self, operations):
        """Add preventive or corrective operations."""
        # In a real implementation, would add operations that prevent contradictions
        # Simple example: add a dummy operation
        enhanced = operations.copy()
        enhanced.append({
            'type': 'metadata_update',
            'metadata_key': 'active_inference_enhanced',
            'metadata_value': True
        })
        return enhanced
        
    async def _anticipate_category_changes(self, operations):
        """Anticipate how categories might form or change."""
        # Placeholder - in real implementation would implement proper anticipation
        return []
        
    async def _anticipate_concept_formations(self, operations):
        """Anticipate what concepts might form."""
        # Placeholder - in real implementation would implement proper anticipation
        return []
    
    def _calculate_set_prediction_error(self, predicted, actual):
        """Calculate error between predicted and actual sets."""
        # Simple implementation - normalized difference in count
        pred_count = len(predicted)
        actual_count = len(actual)
        
        if pred_count == 0 and actual_count == 0:
            return 0.0
            
        count_error = abs(pred_count - actual_count) / max(1, max(pred_count, actual_count))
        
        return count_error
        
    def _clean_old_anticipations(self):
        """Remove old anticipations to prevent memory buildup."""
        current_time = time.time()
        self.anticipated_states = {
            op_id: state for op_id, state in self.anticipated_states.items()
            if current_time - state['timestamp'] < 3600  # Keep for an hour
        }



# === FILE: asf\layer3_cognitive_boundary\cognitive_boundary_layer.py ===

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional

from asf.semantic_organization.core.semantic_tensor_network import SemanticTensorNetwork
from asf.semantic_organization.processing.async_queue import AsyncProcessingQueue
from asf.semantic_organization.processing.priority_manager import AdaptivePriorityManager
from asf.semantic_organization.formation.concept_formation import ConceptFormationEngine
from asf.semantic_organization.formation.conceptual_blending import ConceptualBlendingEngine
from asf.semantic_organization.formation.category_formation import CategoryFormationSystem
from asf.semantic_organization.resolution.conflict_detection import ConflictDetectionEngine
from asf.semantic_organization.temporal import AdaptiveTemporalMetadata
from asf.semantic_organization.predictive_processor import PredictiveProcessor
from asf.semantic_organization.active_inference import ActiveInferenceController

class SemanticOrganizationLayer:
    """
    Main controller for Layer 3 (Semantic Organization Layer).
    Orchestrates semantic operations across all components.
    Fully integrates Seth's predictive processing principles.
    """
    def __init__(self, config=None):
        self.config = config or {}
        self.logger = logging.getLogger("ASF.Layer3")

        # Initialize core components
        self.semantic_network = SemanticTensorNetwork()
        self.processing_queue = AsyncProcessingQueue()
        self.priority_manager = AdaptivePriorityManager()

        # Initialize formation systems
        self.concept_formation = ConceptFormationEngine(self.semantic_network)
        self.conceptual_blending = ConceptualBlendingEngine(self.semantic_network, self.concept_formation)
        self.category_formation = CategoryFormationSystem(self.semantic_network)

        # Initialize resolution systems
        self.conflict_detection = ConflictDetectionEngine(self.semantic_network)

        # Seth's Data Paradox enhancements
        self.predictive_processor = PredictiveProcessor()
        self.active_inference = ActiveInferenceController(self)

        # Configuration for predictive processing and temporal management
        self.anticipation_enabled = self.config.get('anticipation_enabled', True)
        self.active_inference_enabled = self.config.get('active_inference_enabled', True)

    async def initialize(self):
        """Initialize the semantic organization layer."""
        self.logger.info("Initializing Semantic Organization Layer")
        
        # Initialize predictive components if enabled
        if self.anticipation_enabled:
            self.logger.info("Initializing predictive processing capabilities")
        
        return True

    async def anticipate_semantic_operations(self, operations):
        """
        Anticipate the outcomes of semantic operations before executing them.
        
        Args:
            operations: List of planned semantic operations
            
        Returns:
            Anticipated semantic state
        """
        if not self.anticipation_enabled:
            return {'status': 'disabled', 'message': 'Anticipation is disabled'}
        
        self.logger.info(f"Anticipating outcomes of {len(operations)} semantic operations")
        return await self.active_inference.anticipate_state(operations)

    async def optimize_operations(self, anticipated_state):
        """
        Optimize planned operations using active inference.
        
        Args:
            anticipated_state: Previously anticipated state
            
        Returns:
            Optimized operations
        """
        if not self.active_inference_enabled:
            return {'status': 'disabled', 'message': 'Active inference is disabled'}
        
        self.logger.info("Performing active inference to optimize operations")
        return await self.active_inference.perform_active_inference(anticipated_state)

    async def execute_semantic_operations(self, operations, anticipate=True, optimize=True):
        """
        Execute semantic operations with optional anticipation and optimization.
        
        Args:
            operations: List of semantic operations to perform
            anticipate: Whether to anticipate outcomes before execution
            optimize: Whether to optimize operations using active inference
            
        Returns:
            Execution results
        """
        start_time = time.time()
        
        operation_results = []
        
        # Anticipate outcomes if enabled
        anticipated_state = None
        if anticipate and self.anticipation_enabled:
            anticipated_state = await self.anticipate_semantic_operations(operations)
        
        # Optimize operations if enabled
        if optimize and self.active_inference_enabled and anticipated_state:
            optimization_result = await self.optimize_operations(anticipated_state)
            if optimization_result.get('status') == 'success':
                operations = optimization_result.get('optimized_operations', operations)
        
        # Execute each operation
        for operation in operations:
            op_type = operation.get('type')
            op_result = {'operation': operation, 'status': 'unknown'}
            
            try:
                if op_type == 'create_node':
                    node_id = await self._execute_create_node(operation)
                    op_result = {'status': 'success', 'node_id': node_id}
                    
                elif op_type == 'add_property':
                    success = await self._execute_add_property(operation)
                    op_result = {'status': 'success' if success else 'failed'}
                    
                elif op_type == 'create_relation':
                    relation_id = await self._execute_create_relation(operation)
                    op_result = {'status': 'success', 'relation_id': relation_id}
                    
                elif op_type == 'form_concept':
                    concept_id = await self._execute_form_concept(operation)
                    op_result = {'status': 'success', 'concept_id': concept_id}
                    
                elif op_type == 'create_blend':
                    blend_id = await self._execute_create_blend(operation)
                    op_result = {'status': 'success', 'blend_id': blend_id}
                    
                elif op_type == 'form_categories':
                    category_result = await self._execute_form_categories(operation)
                    op_result = {'status': 'success', 'category_result': category_result}
                    
                else:
                    op_result = {'status': 'unknown_operation', 'message': f"Unknown operation type: {op_type}"}
                    
            except Exception as e:
                self.logger.error(f"Error executing operation {op_type}: {str(e)}")
                op_result = {'status': 'error', 'message': str(e)}
                
            operation_results.append(op_result)
        
        # Check for contradictions after execution
        contradictions = await self.conflict_detection.check_contradictions()
        
        # Evaluate anticipation accuracy if we did anticipation
        evaluation = None
        if anticipated_state:
            actual_state = {
                'actual_contradictions': contradictions,
                # In a real implementation, would include more state information
            }
            
            evaluation = await self.active_inference.evaluate_anticipations(
                actual_state,
                anticipated_state.get('operation_id')
            )
        
        execution_time = time.time() - start_time
        
        return {
            'status': 'success',
            'operation_count': len(operations),
            'results': operation_results,
            'contradictions': contradictions,
            'anticipation_evaluation': evaluation,
            'execution_time': execution_time,
        }

    async def _execute_create_node(self, operation):
        """Execute a create node operation."""
        
    async def _execute_add_property(self, operation):
         """Execute an add property operation."""
    
    async def _execute_create_relation(self, operation):
         """Execute a create relation operation."""
    
    async def _execute_form_concept(self, operation):
         """Execute a form concept operation."""
    
    async def _execute_create_blend(self, operation):
         """Execute a create blend operation."""
    
    async def _execute_form_categories(self, operation):
         """Execute a form categories operation."""



# === FILE: asf\layer3_cognitive_boundary\enums.py ===

from enum import Enum

class SemanticNodeType(Enum):
    """Enumeration of semantic node types"""
    CONCEPT = "concept"
    CATEGORY = "category"
    BLEND = "blend"
    ABSTRACTION = "abstraction"
    RELATION = "relation"

class SemanticConfidenceState(Enum):
    """Confidence states for semantic structures"""
    HYPOTHETICAL = "hypothetical"  # Initial state, speculative
    PROVISIONAL = "provisional"    # Partially validated
    CANONICAL = "canonical"        # Fully validated semantic structure



# === FILE: asf\layer3_cognitive_boundary\predictive_processor.py ===

import time
import numpy as np
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict

class PredictiveProcessor:
    """
    Central coordinator for predictive processing across Layer 3.
    Implements Seth's core insight that cognition is "controlled hallucination."
    
    This service:
    1. Manages predictions across all components
    2. Tracks prediction errors and precision for uncertainty estimation
    3. Provides adaptive learning rates based on precision
    4. Supports active inference for minimizing prediction errors
    """
    def __init__(self):
        self.predictions = {}  # Prediction ID -> prediction data
        self.prediction_errors = defaultdict(list)  # Entity ID -> errors
        self.precision_values = {}  # Entity ID -> precision value
        self.learning_rates = {}  # Entity ID -> adaptive learning rate
        self.context_history = {}  # Context ID -> previous states
        
    async def register_prediction(self, context_id: str, entity_id: str, 
                                prediction_type: str, prediction_value: Any) -> str:
        """Register a prediction for later evaluation"""
        prediction_id = f"{context_id}_{entity_id}_{prediction_type}_{int(time.time()*1000)}"
        self.predictions[prediction_id] = {
            'context_id': context_id,
            'entity_id': entity_id,
            'type': prediction_type,
            'value': prediction_value,
            'timestamp': time.time(),
            'evaluated': False
        }
        return prediction_id
        
    async def evaluate_prediction(self, prediction_id: str, actual_value: Any) -> Optional[Dict[str, Any]]:
        """Evaluate prediction against actual value and update precision"""
        if prediction_id not in self.predictions:
            return None
            
        prediction = self.predictions[prediction_id]
        if prediction['evaluated']:
            return None
            
        # Calculate prediction error
        predicted_value = prediction['value']
        error = self._calculate_error(predicted_value, actual_value)
        
        # Update prediction record
        prediction['evaluated'] = True
        prediction['actual_value'] = actual_value
        prediction['error'] = error
        prediction['evaluation_time'] = time.time()
        
        # Track error for precision calculation
        entity_id = prediction['entity_id']
        self.prediction_errors[entity_id].append(error)
        
        # Limit history size to prevent unbounded growth
        if len(self.prediction_errors[entity_id]) > 20:
            self.prediction_errors[entity_id] = self.prediction_errors[entity_id][-20:]
        
        # Update precision (inverse variance)
        if len(self.prediction_errors[entity_id]) > 1:
            variance = np.var(self.prediction_errors[entity_id])
            precision = 1.0 / (variance + 1e-6)  # Avoid division by zero
            self.precision_values[entity_id] = min(10.0, precision)  # Cap very high precision
            
        # Calculate adaptive learning rate
        # Implements Seth's insight: learning rate should be higher for surprising events
        # but tempered by precision (how certain we are about the prediction)
        precision = self.get_precision(entity_id)
        base_rate = min(0.8, error * 2)  # Error-proportional component
        precision_factor = max(0.1, min(0.9, 1.0 / (1.0 + precision * 0.2)))
        learning_rate = min(0.9, max(0.1, base_rate * precision_factor))
        self.learning_rates[entity_id] = learning_rate
        
        return {
            'prediction_id': prediction_id,
            'error': error,
            'precision': precision,
            'learning_rate': learning_rate
        }
        
    def get_precision(self, entity_id: str) -> float:
        """Get precision (inverse variance) for a specific entity"""
        return self.precision_values.get(entity_id, 1.0)
        
    def get_learning_rate(self, entity_id: str) -> float:
        """Get adaptive learning rate for an entity"""
        return self.learning_rates.get(entity_id, 0.3)
        
    def _calculate_error(self, predicted: Any, actual: Any) -> float:
        """Calculate normalized error between predicted and actual values"""
        if isinstance(predicted, (int, float, np.number)) and isinstance(actual, (int, float, np.number)):
            # For numeric values, normalized absolute difference
            return abs(predicted - actual) / (1.0 + abs(actual))
            
        elif isinstance(predicted, (list, np.ndarray)) and isinstance(actual, (list, np.ndarray)):
            # For vectors, normalized Euclidean distance
            try:
                predicted_arr = np.array(predicted)
                actual_arr = np.array(actual)
                
                if predicted_arr.shape != actual_arr.shape:
                    return 1.0  # Maximum error for shape mismatch
                    
                if predicted_arr.size == 0 or actual_arr.size == 0:
                    return 1.0  # Maximum error for empty arrays
                    
                # Normalized Euclidean distance
                diff = np.linalg.norm(predicted_arr - actual_arr)
                norm = np.linalg.norm(actual_arr)
                return min(1.0, diff / (1.0 + norm))
            except:
                return 1.0  # Maximum error if calculation fails
            
        elif isinstance(predicted, dict) and isinstance(actual, dict):
            # For dictionaries, calculate average error across shared keys
            shared_keys = set(predicted.keys()) & set(actual.keys())
            
            if not shared_keys:
                return 1.0  # Maximum error if no shared keys
                
            errors = []
            for key in shared_keys:
                errors.append(self._calculate_error(predicted[key], actual[key]))
                
            return sum(errors) / len(errors)
            
        else:
            # Fallback for other types
            return 1.0 if predicted != actual else 0.0
    
    def perform_active_inference(self, entity_id: str, current_state: Dict[str, Any], 
                                possible_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select action that minimizes expected prediction error (Seth's active inference)
        
        Args:
            entity_id: Entity performing the inference
            current_state: Current state of the entity
            possible_actions: List of possible actions to take
            
        Returns:
            Selected action that minimizes expected prediction error
        """
        # If no actions available, return None
        if not possible_actions:
            return None
            
        # Get precision for this entity
        precision = self.get_precision(entity_id)
        
        # Evaluate each action's expected prediction error
        action_scores = []
        
        for action in possible_actions:
            # Predict outcome of this action
            predicted_outcome = action.get('predicted_outcome')
            if not predicted_outcome:
                # Skip actions without predicted outcomes
                continue
                
            # Calculate expected prediction error
            expected_error = action.get('expected_error', 0.5)
            
            # Weight by precision (higher precision = more weight)
            weighted_error = expected_error / (1.0 + precision)
            
            action_scores.append((action, weighted_error))
            
        if not action_scores:
            # No actions with predictions, return first action
            return possible_actions[0] if possible_actions else None
            
        # Select action with minimum expected error
        best_action = min(action_scores, key=lambda x: x[1])[0]
        return best_action



# === FILE: asf\layer3_cognitive_boundary\temporal.py ===

import time
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class AdaptiveTemporalMetadata:
    """
    Enhanced temporal metadata with adaptive decay based on context.
    Supports multi-resolution temporal tracking with confidence weighting.
    Implements Seth's principle that perception is shaped by temporal context.
    """
    creation_time: float = field(default_factory=lambda: time.time())
    last_accessed: float = field(default_factory=lambda: time.time())
    last_modified: float = field(default_factory=lambda: time.time())
    access_count: int = 0
    modification_count: int = 0
    contextual_half_lives: Dict[str, float] = field(default_factory=lambda: {
        'default': 86400 * 7,  # 7 days
        'critical': 86400 * 30,  # 30 days
        'ephemeral': 3600  # 1 hour
    })
    
    # Seth's Data Paradox enhancements
    predicted_relevance: Dict[str, float] = field(default_factory=dict)
    relevance_errors: Dict[str, list] = field(default_factory=lambda: {})
    relevance_precision: Dict[str, float] = field(default_factory=dict)
    
    def update_access(self):
        """Update access time and count when entity is accessed."""
        self.last_accessed = time.time()
        self.access_count += 1
        
    def update_modification(self):
        """Update modification time when entity is modified."""
        self.last_modified = time.time()
        self.modification_count += 1
    
    def compute_relevance(self, current_time=None, context='default'):
        """Compute temporal relevance using adaptive weighting."""
        if current_time is None:
            current_time = time.time()
            
        elapsed = current_time - self.last_accessed
        half_life = self.contextual_half_lives.get(context, 
                                                 self.contextual_half_lives['default'])
        
        # Dynamic weight based on access frequency
        dynamic_weight = min(1.0, 0.5 + (self.access_count / 20))
        
        # Logarithmic decay is gentler for frequently accessed items
        base_relevance = 1.0 / (1.0 + np.log1p(elapsed / half_life))
        
        # Apply dynamic weighting
        return base_relevance * dynamic_weight
    
    def compute_freshness(self, current_time=None):
        """Compute how fresh/recent a modification is."""
        if current_time is None:
            current_time = time.time()
            
        elapsed = current_time - self.last_modified
        
        # Exponential decay function for freshness
        decay_rate = 0.1  # Controls decay speed
        freshness = np.exp(-decay_rate * elapsed / 86400)  # Normalized to days
        
        return freshness
    
    def should_retain(self, current_time=None, context='default', threshold=0.2):
        """Determine if this entity should be retained in memory."""
        relevance = self.compute_relevance(current_time, context)
        freshness = self.compute_freshness(current_time)
        
        # Combined score with higher weight on relevance
        retention_score = relevance * 0.7 + freshness * 0.3
        
        return retention_score > threshold
    
    def anticipate_relevance(self, future_time, context='default'):
        """
        Anticipate future relevance of this entity.
        Implements Seth's predictive principle for temporal perception.
        
        Args:
            future_time: Future time point to predict relevance for
            context: Context for prediction
            
        Returns:
            Predicted relevance score
        """
        prediction_key = f"{context}_{int(future_time)}"
        
        # If we already have a prediction, return it
        if prediction_key in self.predicted_relevance:
            return self.predicted_relevance[prediction_key]
        
        # Calculate time delta
        current_time = time.time()
        delta = future_time - current_time
        
        # Base prediction on current relevance
        current_relevance = self.compute_relevance(current_time, context)
        
        # Predict future relevance based on contextual half-life decay
        half_life = self.contextual_half_lives.get(context, 
                                                self.contextual_half_lives['default'])
        
        # Calculate decay factor (exponential decay)
        decay_factor = np.exp(-0.693 * delta / half_life)  # 0.693 is ln(2)
        
        # Adjust for predicted access or modifications
        expected_accesses = (self.access_count / max(1, current_time - self.creation_time)) * delta
        access_factor = min(1.2, 1.0 + (expected_accesses * 0.1))
        
        # Calculate predicted relevance
        predicted_relevance = current_relevance * decay_factor * access_factor
        
        # Store prediction
        self.predicted_relevance[prediction_key] = predicted_relevance
        
        return predicted_relevance
    
    def update_relevance_prediction(self, future_time, actual_relevance, context='default'):
        """
        Update relevance predictions with actual observed relevance.
        Implements Seth's error minimization principle.
        
        Args:
            future_time: The time point that was predicted
            actual_relevance: The actual observed relevance
            context: Context for prediction
            
        Returns:
            Prediction error
        """
        prediction_key = f"{context}_{int(future_time)}"
        
        # If we didn't predict for this time, nothing to update
        if prediction_key not in self.predicted_relevance:
            return None
        
        # Calculate prediction error
        predicted = self.predicted_relevance[prediction_key]
        error = abs(predicted - actual_relevance)
        
        # Initialize error tracking for this context if needed
        if context not in self.relevance_errors:
            self.relevance_errors[context] = []
        
        # Track error for precision calculation
        self.relevance_errors[context].append(error)
        
        # Limit history size
        if len(self.relevance_errors[context]) > 20:
            self.relevance_errors[context] = self.relevance_errors[context][-20:]
        
        # Update precision (inverse variance)
        if len(self.relevance_errors[context]) > 1:
            variance = np.var(self.relevance_errors[context])
            precision = 1.0 / (variance + 1e-6)  # Avoid division by zero
            self.relevance_precision[context] = precision
        
        return error



# === FILE: asf\layer3_cognitive_boundary\core\__init__.py ===

# Layer 3: Semantic Organization Layer
# Core module containing semantic network components

from asf.semantic_organization.core.semantic_node import SemanticNode
from asf.semantic_organization.core.semantic_relation import SemanticRelation
from asf.semantic_organization.core.semantic_tensor_network import SemanticTensorNetwork

__all__ = ['SemanticNode', 'SemanticRelation', 'SemanticTensorNetwork']



# === FILE: asf\layer3_cognitive_boundary\core\semantic_node.py ===

# Enhancement for semantic_node.py - Add to existing file

import time
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Set, Tuple
from collections import defaultdict

# Keep existing SemanticNode class and add these methods:

class SemanticNode:
    # Existing initialization and properties...
    
    # Add these fields to __init__ or __post_init__:
    # self.anticipated_activations = {}  # Context -> predicted activation
    # self.activation_errors = defaultdict(list)  # Context -> list of errors
    # self.precision_values = {}  # Context -> precision value
    # self.anticipated_properties = {}  # Context -> predicted properties
    
    def anticipate_activation(self, context: Dict[str, Any], stimulus_sources: Optional[List[str]] = None) -> float:
        """
        Anticipate activation level before receiving actual stimulus.
        Implements Seth's "controlled hallucination" principle.
        
        Args:
            context: Current processing context
            stimulus_sources: Optional list of stimulus source IDs
            
        Returns:
            Anticipated activation level (0-1)
        """
        context_key = self._get_context_key(context)
        
        # If we've seen this exact context before, use cached prediction
        if hasattr(self, 'anticipated_activations') and context_key in self.anticipated_activations:
            return self.anticipated_activations[context_key]
        
        # Initialize anticipation fields if not present
        if not hasattr(self, 'anticipated_activations'):
            self.anticipated_activations = {}
        if not hasattr(self, 'activation_errors'):
            self.activation_errors = defaultdict(list)
        if not hasattr(self, 'precision_values'):
            self.precision_values = {}
        
        # Base prediction on similar contexts we've seen before
        similar_contexts = self._find_similar_contexts(context)
        
        if not similar_contexts:
            # No similar contexts, use current activation as prediction
            prediction = self.activation
        else:
            # Calculate prediction as weighted average of activations in similar contexts
            total_weight = 0.0
            weighted_sum = 0.0
            
            for sim_context, similarity in similar_contexts:
                if sim_context in self.precision_values:
                    # Weight by both similarity and precision
                    precision = self.precision_values[sim_context]
                    weight = similarity * precision
                else:
                    weight = similarity
                
                # Retrieve past activation for this context
                past_activation = self._get_past_activation(sim_context)
                
                weighted_sum += past_activation * weight
                total_weight += weight
            
            if total_weight > 0:
                prediction = weighted_sum / total_weight
            else:
                prediction = self.activation
        
        # Store prediction
        self.anticipated_activations[context_key] = prediction
        
        return prediction
    
    def update_activation(self, stimulus: float, decay_rate: float = 0.95) -> float:
        """
        Update activation level and evaluate prediction accuracy.
        Uses leaky integrator model with MGC-inspired maximization.
        
        Args:
            stimulus: Input activation stimulus
            decay_rate: Rate of activation decay
            
        Returns:
            Updated activation level
        """
        # Get current context
        context_key = self._get_current_context_key()
        
        # Check if we made a prediction for this context
        if hasattr(self, 'anticipated_activations') and context_key in self.anticipated_activations:
            predicted = self.anticipated_activations[context_key]
            
            # Calculate prediction error
            prediction_error = abs(predicted - stimulus)
            
            # Initialize if needed
            if not hasattr(self, 'activation_errors'):
                self.activation_errors = defaultdict(list)
            if not hasattr(self, 'precision_values'):
                self.precision_values = {}
            
            # Track error for this context
            self.activation_errors[context_key].append(prediction_error)
            
            # Limit history size
            if len(self.activation_errors[context_key]) > 20:
                self.activation_errors[context_key] = self.activation_errors[context_key][-20:]
            
            # Update precision (inverse variance)
            if len(self.activation_errors[context_key]) > 1:
                variance = np.var(self.activation_errors[context_key])
                precision = 1.0 / (variance + 1e-6)  # Avoid division by zero
                self.precision_values[context_key] = precision
        
        # MGC-inspired maximization-based update
        self.activation = max(
            decay_rate * self.activation,
            stimulus
        )
        
        self.temporal_metadata.update_access()
        return self.activation
    
    def anticipate_property_changes(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Anticipate how properties might change in a given context.
        
        Args:
            context: Processing context
            
        Returns:
            Dictionary of anticipated property values
        """
        context_key = self._get_context_key(context)
        
        # Initialize if needed
        if not hasattr(self, 'anticipated_properties'):
            self.anticipated_properties = {}
        
        # If we already predicted for this context, return cached prediction
        if context_key in self.anticipated_properties:
            return self.anticipated_properties[context_key]
        
        # Find similar contexts where properties changed
        similar_contexts = self._find_similar_contexts(context)
        
        predictions = {}
        
        # Look at property changes in similar contexts
        for prop_name in self.properties:
            prop_predictions = []
            
            for sim_context, similarity in similar_contexts:
                # Check if we have record of this property changing in similar context
                change_key = f"{sim_context}_{prop_name}"
                if 'property_changes' in self.metadata and change_key in self.metadata['property_changes']:
                    change_record = self.metadata['property_changes'][change_key]
                    # Weight by similarity and recency
                    time_factor = np.exp(-(time.time() - change_record['time']) / 86400)  # Decay over a day
                    weight = similarity * time_factor
                    prop_predictions.append((change_record['new_value'], weight))
            
            # If we have predictions for this property, calculate weighted average
            if prop_predictions:
                if all(isinstance(p[0], (int, float)) for p in prop_predictions):
                    # For numeric properties, weighted average
                    total_weight = sum(w for _, w in prop_predictions)
                    weighted_sum = sum(v * w for v, w in prop_predictions)
                    if total_weight > 0:
                        predictions[prop_name] = weighted_sum / total_weight
                else:
                    # For non-numeric, use most strongly weighted value
                    predictions[prop_name] = max(prop_predictions, key=lambda x: x[1])[0]
        
        # Store predictions
        self.anticipated_properties[context_key] = predictions
        
        return predictions
    
    def _get_context_key(self, context: Dict[str, Any]) -> str:
        """Generate a stable hash key for a context."""
        # Simple context key generation - can be enhanced with better hashing
        sorted_items = sorted((str(k), str(v)) for k, v in context.items())
        return '_'.join(f"{k}:{v}" for k, v in sorted_items)
    
    def _get_current_context_key(self) -> str:
        """Get context key for current state."""
        # In real implementation, this would capture current processing context
        return f"ctx_{int(time.time())}"
    
    def _find_similar_contexts(self, context: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Find contexts similar to the given context with similarity scores."""
        # Initialize if needed
        if not hasattr(self, 'anticipated_activations'):
            self.anticipated_activations = {}
        if not hasattr(self, 'precision_values'):
            self.precision_values = {}
            
        context_key = self._get_context_key(context)
        context_keys = set(self.anticipated_activations.keys()) | set(key for key in self.precision_values.keys())
        
        similar_contexts = []
        
        for key in context_keys:
            similarity = self._calculate_context_similarity(context_key, key)
            if similarity > 0.2:  # Threshold for considering contexts similar
                similar_contexts.append((key, similarity))
        
        # Sort by similarity (descending)
        return sorted(similar_contexts, key=lambda x: -x[1])
    
    def _calculate_context_similarity(self, context_key1: str, context_key2: str) -> float:
        """Calculate similarity between two context keys."""
        # Simple implementation - in production, would use semantic similarity
        tokens1 = set(context_key1.split('_'))
        tokens2 = set(context_key2.split('_'))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union)
    
    def _get_past_activation(self, context_key: str) -> float:
        """Retrieve past activation for a context."""
        # In real implementation, would store activation history
        return self.activation  # Default to current activation



# === FILE: asf\layer3_cognitive_boundary\core\semantic_relation.py ===

import numpy as np
import time
from collections import defaultdict
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from asf.semantic_organization.temporal import AdaptiveTemporalMetadata

@dataclass
class SemanticRelation:
    """
    Enhanced edge in the semantic network with flexible weighting and attention.
    Supports hyperbolic geometry for hierarchical relationships and
    predictive processing for relation strength anticipation.
    """
    id: str
    source_id: str
    target_id: str
    relation_type: str
    weight: float = 0.8
    bidirectional: bool = False
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.7
    temporal_metadata: AdaptiveTemporalMetadata = field(default_factory=AdaptiveTemporalMetadata)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Enhanced properties
    attention_weight: float = 1.0  # For attention-weighted graph
    embedding: Optional[np.ndarray] = None  # Relation-specific embedding
    
    # Predictive properties
    anticipated_weights: Dict[str, float] = field(default_factory=dict)
    weight_prediction_errors: Dict[str, list] = field(default_factory=lambda: defaultdict(list))
    weight_precision: Dict[str, float] = field(default_factory=dict)
    
    def compute_effective_weight(self, context='default'):
        """Compute effective weight considering temporal relevance and attention."""
        temporal_factor = self.temporal_metadata.compute_relevance(context=context)
        return self.weight * temporal_factor * self.confidence * self.attention_weight
    
    def update_attention(self, new_attention):
        """Update attention weight for this relation."""
        self.attention_weight = new_attention
        self.temporal_metadata.update_modification()
        
    def anticipate_weight(self, context='default', factors=None):
        """
        Anticipate how this relation's weight might evolve in a given context.
        Implements Seth's "controlled hallucination" principle for relations.
        
        Args:
            context: Context identifier
            factors: Additional factors that might affect the weight
            
        Returns:
            Anticipated weight
        """
        # If we already have a prediction for this context, return it
        if context in self.anticipated_weights:
            return self.anticipated_weights[context]
        
        factors = factors or {}
        current_weight = self.weight
        
        # Base prediction on current weight
        prediction = current_weight
        
        # Adjust prediction based on factors
        if 'temporal_decay' in factors and factors['temporal_decay']:
            # Predict temporal decay effect
            time_factor = np.exp(-0.1 * factors.get('elapsed_time', 0) / 86400)
            prediction *= time_factor
            
        if 'competing_relation' in factors:
            # Predict effect of competing relation
            prediction *= 0.9
            
        if 'reinforcing_relation' in factors:
            # Predict effect of reinforcing relation
            prediction = min(1.0, prediction * 1.1)
            
        # Store prediction
        self.anticipated_weights[context] = prediction
        
        return prediction
    
    def update_weight(self, new_weight, context='default'):
        """
        Update weight and evaluate prediction accuracy.
        Implements Seth's prediction error minimization principle.
        
        Args:
            new_weight: New weight value
            context: Context in which the update occurs
            
        Returns:
            Prediction error
        """
        old_weight = self.weight
        self.weight = new_weight
        
        # If we had a prediction for this context, calculate error
        error = None
        if context in self.anticipated_weights:
            predicted = self.anticipated_weights[context]
            error = abs(predicted - new_weight)
            
            # Track error for precision calculation
            self.weight_prediction_errors[context].append(error)
            
            # Limit history size
            if len(self.weight_prediction_errors[context]) > 20:
                self.weight_prediction_errors[context] = self.weight_prediction_errors[context][-20:]
            
            # Update precision (inverse variance)
            if len(self.weight_prediction_errors[context]) > 1:
                variance = np.var(self.weight_prediction_errors[context])
                precision = 1.0 / (variance + 1e-6)  # Avoid division by zero
                self.weight_precision[context] = precision
        
        self.temporal_metadata.update_modification()
        return error
    
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'source_id': self.source_id,
            'target_id': self.target_id,
            'relation_type': self.relation_type,
            'weight': self.weight,
            'bidirectional': self.bidirectional,
            'properties': self.properties,
            'confidence': self.confidence,
            'attention_weight': self.attention_weight,
            'metadata': self.metadata,
            'prediction_metadata': {
                'contexts': list(self.anticipated_weights.keys()),
                'precision': {k: v for k, v in self.weight_precision.items()}
            }
        }



# === FILE: asf\layer3_cognitive_boundary\core\semantic_tensor_network.py ===

import numpy as np
import torch
import torch.nn.functional as F
import threading
import logging
import faiss
import time
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict

from asf.semantic_organization.core.semantic_node import SemanticNode
from asf.semantic_organization.core.semantic_relation import SemanticRelation

class SemanticTensorNetwork:
    """
    Core semantic network implementation with optimized tensor-based operations.
    Uses multi-channel sparse tensors for efficient graph algorithms and
    implements predictive mechanisms for anticipating network evolution.
    """
    def __init__(self, embedding_dim=1024, device=None):
        self.nodes = {}  # Maps node ID to SemanticNode
        self.relations = {}  # Maps relation ID to SemanticRelation
        self.node_index = {}  # Maps node ID to matrix index
        self.index_to_node = {}  # Maps matrix index to node ID
        
        # Relation indices for efficient lookup
        self.relation_index = {
            'source': defaultdict(list),  # Maps source ID to relation IDs
            'target': defaultdict(list),  # Maps target ID to relation IDs
            'type': defaultdict(list)  # Maps relation type to relation IDs
        }
        
        # Tensor-based representations
        self.embedding_dim = embedding_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Multi-channel adjacency tensors (by relation type)
        self.adjacency_tensors = {}
        self.node_embeddings = None
        self.rebuild_needed = True
        
        # FAISS index for similarity search
        self.faiss_index = None
        self.embedding_buffer = []
        self.node_count = 0
        self.faiss_initialized = False
        
        # Performance optimization
        self.batched_operations = True
        self.batch_size = 64
        
        # Predictive components (Seth's Data Paradox integration)
        self.predicted_connections = {}  # (node1_id, node2_id) -> predicted relation
        self.connection_prediction_errors = defaultdict(list)  # Relation type -> errors
        self.connection_precision = {}  # Relation type -> precision
        self.predicted_network_changes = []  # List of anticipated changes
        
        # Locks for thread safety
        self.logger = logging.getLogger("ASF.Layer3.SemanticNetwork")
        self.lock = threading.RLock()

    async def add_node(self, node, update_tensors=True):
        """Add a node to the semantic network with incremental tensor update."""
        with self.lock:
            if node.id in self.nodes:
                # Update existing node
                existing_node = self.nodes[node.id]
                existing_node.label = node.label
                existing_node.properties.update(node.properties)
                existing_node.confidence = max(existing_node.confidence, node.confidence)
                
                # Only update embeddings if significantly different
                if isinstance(node.embeddings, np.ndarray) and np.linalg.norm(existing_node.embeddings - node.embeddings) > 0.1:
                    existing_node.embeddings = node.embeddings
                    existing_node.tensor_representation = torch.tensor(
                        node.embeddings, dtype=torch.float32
                    ).to(self.device)
                    
                    # Mark for incremental FAISS update
                    if node.id in self.node_index:
                        self.embedding_buffer.append((self.node_index[node.id], node.embeddings))
                    
                    existing_node.temporal_metadata.update_modification()
                    
                return node.id
            
            # Add new node
            self.nodes[node.id] = node
            
            # Move tensor to correct device
            if node.tensor_representation is not None:
                node.tensor_representation = node.tensor_representation.to(self.device)
                
            self.node_index[node.id] = self.node_count
            self.index_to_node[self.node_count] = node.id
            
            # Mark for incremental FAISS update
            if hasattr(node, 'embeddings') and node.embeddings is not None:
                self.embedding_buffer.append((self.node_count, node.embeddings))
                
            self.node_count += 1
            
            # Update hierarchical relationships
            if hasattr(node, 'parent_ids'):
                for parent_id in node.parent_ids:
                    if parent_id in self.nodes:
                        if node.id not in self.nodes[parent_id].child_ids:
                            self.nodes[parent_id].child_ids.append(node.id)
                            
            if hasattr(node, 'child_ids'):
                for child_id in node.child_ids:
                    if child_id in self.nodes:
                        if node.id not in self.nodes[child_id].parent_ids:
                            self.nodes[child_id].parent_ids.append(node.id)
            
            # Check if we should update tensors
            if update_tensors and self.batched_operations:
                if len(self.embedding_buffer) >= self.batch_size:
                    await self._update_tensors()
                    
            # Predictive component: anticipate connections for new node
            await self._predict_connections_for_node(node.id)
                
            return node.id

    async def add_relation(self, relation):
        """Add a relation to the semantic network with tensor updates."""
        with self.lock:
            # Ensure source and target nodes exist
            if relation.source_id not in self.nodes or relation.target_id not in self.nodes:
                return None
                
            if relation.id in self.relations:
                # Update existing relation
                existing_relation = self.relations[relation.id]
                existing_relation.weight = relation.weight
                existing_relation.properties.update(relation.properties)
                existing_relation.confidence = max(existing_relation.confidence, relation.confidence)
                existing_relation.attention_weight = relation.attention_weight
                existing_relation.temporal_metadata.update_modification()
                
                # Mark as needing rebuild for tensor operations
                self.rebuild_needed = True
                
                # Evaluate prediction accuracy if applicable
                prediction_key = (relation.source_id, relation.target_id)
                if prediction_key in self.predicted_connections:
                    pred = self.predicted_connections[prediction_key]
                    if pred['relation_type'] == relation.relation_type:
                        # Calculate prediction error
                        weight_error = abs(pred['weight'] - relation.weight)
                        
                        # Update prediction errors
                        self.connection_prediction_errors[relation.relation_type].append(weight_error)
                        
                        # Limit history size
                        if len(self.connection_prediction_errors[relation.relation_type]) > 20:
                            self.connection_prediction_errors[relation.relation_type] = \
                                self.connection_prediction_errors[relation.relation_type][-20:]
                        
                        # Update precision
                        if len(self.connection_prediction_errors[relation.relation_type]) > 1:
                            variance = np.var(self.connection_prediction_errors[relation.relation_type])
                            precision = 1.0 / (variance + 1e-6)  # Avoid division by zero
                            self.connection_precision[relation.relation_type] = precision
                
                return relation.id
            
            # Add new relation
            self.relations[relation.id] = relation
            
            # Update indices
            self.relation_index['source'][relation.source_id].append(relation.id)
            self.relation_index['target'][relation.target_id].append(relation.id)
            self.relation_index['type'][relation.relation_type].append(relation.id)
            
            # Update adjacency tensors if they exist
            self.rebuild_needed = True
            
            # For hierarchical relations, update hierarchical structure
            if relation.relation_type in ["is_a", "part_of", "subclass_of"]:
                source_node = self.nodes[relation.source_id]
                target_node = self.nodes[relation.target_id]
                
                # Add hierarchical relationship
                if hasattr(source_node, 'parent_ids') and target_node.id not in source_node.parent_ids:
                    source_node.parent_ids.append(target_node.id)
                    
                if hasattr(target_node, 'child_ids') and source_node.id not in target_node.child_ids:
                    target_node.child_ids.append(source_node.id)
                
                # Update abstraction levels
                if hasattr(source_node, 'abstraction_level') and hasattr(target_node, 'abstraction_level'):
                    source_node.abstraction_level = max(
                        source_node.abstraction_level,
                        target_node.abstraction_level + 1
                    )
            
            # Evaluate prediction accuracy if applicable
            prediction_key = (relation.source_id, relation.target_id)
            if prediction_key in self.predicted_connections:
                pred = self.predicted_connections[prediction_key]
                if pred['relation_type'] == relation.relation_type:
                    # Calculate prediction error
                    weight_error = abs(pred['weight'] - relation.weight)
                    
                    # Update prediction errors
                    self.connection_prediction_errors[relation.relation_type].append(weight_error)
                    
                    # Limit history size
                    if len(self.connection_prediction_errors[relation.relation_type]) > 20:
                        self.connection_prediction_errors[relation.relation_type] = \
                            self.connection_prediction_errors[relation.relation_type][-20:]
                    
                    # Update precision
                    if len(self.connection_prediction_errors[relation.relation_type]) > 1:
                        variance = np.var(self.connection_prediction_errors[relation.relation_type])
                        precision = 1.0 / (variance + 1e-6)  # Avoid division by zero
                        self.connection_precision[relation.relation_type] = precision
            
            return relation.id

    async def get_node(self, node_id):
        """Get a node by ID."""
        with self.lock:
            node = self.nodes.get(node_id)
            if node:
                node.temporal_metadata.update_access()
                return node
            return None

    async def get_relation(self, relation_id):
        """Get a relation by ID."""
        with self.lock:
            relation = self.relations.get(relation_id)
            if relation:
                relation.temporal_metadata.update_access()
                return relation
            return None

    async def get_similar_nodes(self, node_id, k=5, threshold=0.7, by_embedding=True):
        """Find semantically similar nodes using FAISS or tensor operations."""
        node = await self.get_node(node_id)
        if not node:
            return []
            
        with self.lock:
            if by_embedding and (not self.faiss_initialized or self.rebuild_needed):
                await self._initialize_faiss_index()
                
            if by_embedding and self.faiss_index is not None:
                # Search using FAISS (faster for large networks)
                embeddings = np.array([node.embeddings], dtype=np.float32)
                distances, indices = self.faiss_index.search(embeddings, k*2)  # Get extra for filtering
                
                # Process results
                results = []
                for i in range(len(indices[0])):
                    idx = indices[0][i]
                    if idx != -1 and idx < self.node_count:
                        similar_id = self.index_to_node.get(idx)
                        if similar_id and similar_id != node_id:
                            similarity = 1.0 - distances[0][i] / 2.0  # Convert to similarity
                            if similarity >= threshold:
                                results.append((similar_id, similarity))
                
                return sorted(results, key=lambda x: -x[1])[:k]
            else:
                # Fallback to direct tensor similarity
                return await self._compute_tensor_similarities(node, k, threshold)

    async def get_node_relations(self, node_id, direction="both", relation_type=None):
        """Get relations involving a node."""
        with self.lock:
            results = []
            
            # Get outgoing relations
            if direction in ["outgoing", "both"]:
                for relation_id in self.relation_index['source'].get(node_id, []):
                    relation = self.relations.get(relation_id)
                    if relation and (relation_type is None or relation.relation_type == relation_type):
                        results.append(relation)
                        
            # Get incoming relations
            if direction in ["incoming", "both"]:
                for relation_id in self.relation_index['target'].get(node_id, []):
                    relation = self.relations.get(relation_id)
                    if relation and (relation_type is None or relation.relation_type == relation_type):
                        results.append(relation)
                        
            return results

    async def predict_connections(self, node_id, max_predictions=5):
        """
        Predict potential new connections for a node.
        Implements Seth's "controlled hallucination" principle for network structure.
        
        Args:
            node_id: Node to predict connections for
            max_predictions: Maximum number of predictions to return
            
        Returns:
            List of predicted connections
        """
        node = await self.get_node(node_id)
        if not node:
            return []
            
        # Get similar nodes
        similar_nodes = await self.get_similar_nodes(node_id, k=10, threshold=0.6)
        
        # Track predictions
        predictions = []
        
        # Look at relations of similar nodes
        for similar_id, similarity in similar_nodes:
            # Get relations of similar node
            relations = await self.get_node_relations(similar_id)
            
            for relation in relations:
                # Determine if this relation pattern could apply to our node
                if relation.source_id == similar_id:
                    # Check if target is already connected to our node
                    target_id = relation.target_id
                    existing_relation = False
                    
                    for rel_id in self.relation_index['source'].get(node_id, []):
                        if self.relations[rel_id].target_id == target_id:
                            existing_relation = True
                            break
                            
                    if not existing_relation:
                        # This is a potential new connection
                        # Get precision for this relation type
                        precision = self.connection_precision.get(relation.relation_type, 1.0)
                        
                        # Calculate predicted weight
                        predicted_weight = relation.weight * similarity * precision
                        
                        predictions.append({
                            'source_id': node_id,
                            'target_id': target_id,
                            'relation_type': relation.relation_type,
                            'weight': predicted_weight,
                            'confidence': similarity * precision,
                            'based_on': similar_id
                        })
                        
                elif relation.target_id == similar_id:
                    # Check if source is already connected to our node
                    source_id = relation.source_id
                    existing_relation = False
                    
                    for rel_id in self.relation_index['target'].get(node_id, []):
                        if self.relations[rel_id].source_id == source_id:
                            existing_relation = True
                            break
                            
                    if not existing_relation:
                        # This is a potential new connection
                        # Get precision for this relation type
                        precision = self.connection_precision.get(relation.relation_type, 1.0)
                        
                        # Calculate predicted weight
                        predicted_weight = relation.weight * similarity * precision
                        
                        predictions.append({
                            'source_id': source_id,
                            'target_id': node_id,
                            'relation_type': relation.relation_type,
                            'weight': predicted_weight,
                            'confidence': similarity * precision,
                            'based_on': similar_id
                        })
        
        # Sort by confidence
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Store predictions
        for pred in predictions[:max_predictions]:
            prediction_key = (pred['source_id'], pred['target_id'])
            self.predicted_connections[prediction_key] = pred
            
        return predictions[:max_predictions]

    async def predict_network_evolution(self, time_horizon=10, confidence_threshold=0.7):
        """
        Predict how the network might evolve over time.
        Implements Seth's predictive processing principle for anticipating future states.
        
        Args:
            time_horizon: Time steps to predict ahead
            confidence_threshold: Minimum confidence for predictions
            
        Returns:
            List of predicted changes
        """
        predicted_changes = []
        
        # Predict new relations
        for node_id in self.nodes:
            # Predict connections for each node
            node_predictions = await self.predict_connections(node_id)
            
            # Filter by confidence
            high_confidence_predictions = [
                p for p in node_predictions if p['confidence'] >= confidence_threshold
            ]
            
            # Add to predicted changes
            for pred in high_confidence_predictions:
                predicted_changes.append({
                    'type': 'new_relation',
                    'source_id': pred['source_id'],
                    'target_id': pred['target_id'],
                    'relation_type': pred['relation_type'],
                    'weight': pred['weight'],
                    'confidence': pred['confidence'],
                    'time_step': np.random.randint(1, time_horizon + 1)  # Random time step
                })
        
        # Predict weight changes
        for relation_id, relation in self.relations.items():
            # Skip if relation has very low weight
            if relation.weight < 0.1:
                continue
                
            # Predict temporal decay
            decay_factor = 0.95  # 5% decay per time step
            final_weight = relation.weight * (decay_factor ** time_horizon)
            
            # Only include significant changes
            if abs(relation.weight - final_weight) > 0.1:
                predicted_changes.append({
                    'type': 'weight_change',
                    'relation_id': relation_id,
                    'initial_weight': relation.weight,
                    'final_weight': final_weight,
                    'confidence': 0.8,
                    'time_step': time_horizon
                })
        
        # Sort by time step
        predicted_changes.sort(key=lambda x: x['time_step'])
        
        # Store predictions
        self.predicted_network_changes = predicted_changes
        
        return predicted_changes

    # The remaining private methods (_build_tensors, _initialize_faiss_index, etc.)
    # would be included but are omitted for brevity

    async def _predict_connections_for_node(self, node_id):
        """Predict potential connections for a newly added node."""
        # Get node
        node = self.nodes.get(node_id)
        if not node:
            return
            
        # Find similar nodes
        similar_nodes = await self.get_similar_nodes(node_id, k=5)
        
        # If no similar nodes found, return
        if not similar_nodes:
            return
            
        # Look at relations of similar nodes
        for similar_id, similarity in similar_nodes:
            if similarity < 0.6:  # Threshold for similarity
                continue
                
            relations = await self.get_node_relations(similar_id)
            
            for relation in relations:
                # Check if this relation type could apply to our node
                target_id = relation.target_id if relation.source_id == similar_id else relation.source_id
                
                # Skip if target is the node itself
                if target_id == node_id:
                    continue
                    
                # Skip if relation already exists
                existing_relation = False
                
                for rel_id in self.relation_index['source'].get(node_id, []):
                    if self.relations[rel_id].target_id == target_id:
                        existing_relation = True
                        break
                        
                for rel_id in self.relation_index['target'].get(node_id, []):
                    if self.relations[rel_id].source_id == target_id:
                        existing_relation = True
                        break
                        
                if not existing_relation:
                    # This is a potential connection
                    precision = self.connection_precision.get(relation.relation_type, 1.0)
                    
                    # Determine direction
                    if relation.source_id == similar_id:
                        # Similar node is source, so our node should be source too
                        source_id = node_id
                        target_id = relation.target_id
                    else:
                        # Similar node is target, so our node should be target too
                        source_id = relation.source_id
                        target_id = node_id
                        
                    # Store prediction
                    prediction_key = (source_id, target_id)
                    self.predicted_connections[prediction_key] = {
                        'source_id': source_id,
                        'target_id': target_id,
                        'relation_type': relation.relation_type,
                        'weight': relation.weight * similarity,
                        'confidence': similarity * precision,
                        'based_on': similar_id
                    }



# === FILE: asf\layer3_cognitive_boundary\formation\__init__.py ===

# Layer 3: Semantic Organization Layer
# Formation module for concept, blend, and category formation

from asf.semantic_organization.formation.concept_formation import ConceptFormationEngine
from asf.semantic_organization.formation.conceptual_blending import ConceptualBlendingEngine
from asf.semantic_organization.formation.category_formation import CategoryFormationSystem

__all__ = ['ConceptFormationEngine', 'ConceptualBlendingEngine', 'CategoryFormationSystem']



# === FILE: asf\layer3_cognitive_boundary\formation\category_formation.py ===

# Enhanced CategoryFormationSystem with active inference

import torch
import torch.nn.functional as F
import uuid
import time
import numpy as np
import logging
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter

from asf.semantic_organization.core.semantic_node import SemanticNode
from asf.semantic_organization.core.semantic_relation import SemanticRelation
from asf.semantic_organization.enums import SemanticConfidenceState

class CategoryFormationSystem:
    """
    Forms and maintains categories with active inference capabilities.
    Implements Seth's principle of minimizing prediction error through actions.
    """
    def __init__(self, semantic_network):
        self.semantic_network = semantic_network
        self.category_formation_history = []
        self.logger = logging.getLogger("ASF.Layer3.CategoryFormation")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Seth's Data Paradox enhancements
        self.category_predictions = {}  # Node set hash -> predicted categories
        self.category_errors = defaultdict(list)  # Input hash -> prediction errors
        self.category_precision = {}  # Input hash -> precision
        
    async def predict_categories(self, partial_nodes, method="similarity", params=None):
        """
        Predict category formation from partial node set.
        Implements Seth's predictive processing principle.
        
        Args:
            partial_nodes: Dict of node IDs to nodes (subset of total)
            method: Categorization method
            params: Additional parameters
            
        Returns:
            Predicted categories
        """
        params = params or {}
        node_set_hash = self._hash_node_set(partial_nodes.keys(), method)
        
        # Check cache for identical node set
        if node_set_hash in self.category_predictions:
            return self.category_predictions[node_set_hash]
        
        # For similarity-based prediction, we need embeddings
        if method == "similarity":
            # Extract node embeddings
            node_ids = list(partial_nodes.keys())
            
            # Use tensor-based similarity
            embeddings_list = []
            valid_node_ids = []
            
            for node_id in node_ids:
                node = partial_nodes[node_id]
                if isinstance(node.embeddings, np.ndarray):
                    embeddings_list.append(torch.tensor(node.embeddings, dtype=torch.float32))
                    valid_node_ids.append(node_id)
                elif isinstance(node.embeddings, torch.Tensor):
                    embeddings_list.append(node.embeddings)
                    valid_node_ids.append(node_id)
            
            if not embeddings_list:
                return {"status": "error", "message": "No valid embeddings for prediction"}
            
            # Stack tensors
            embeddings_tensor = torch.stack(embeddings_list).to(self.device)
            
            # Normalize embeddings
            norms = torch.norm(embeddings_tensor, dim=1, keepdim=True)
            norms[norms == 0] = 1.0  # Avoid division by zero
            normalized_embeddings = embeddings_tensor / norms
            
            # Calculate similarity matrix
            similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t()).cpu().numpy()
            
            # Predict clusters
            min_cluster_size = params.get('min_cluster_size', 3)
            similarity_threshold = params.get('similarity_threshold', 0.7)
            
            predicted_clusters = self._hierarchical_clustering(
                similarity_matrix, 
                min_cluster_size, 
                max_clusters=params.get('max_clusters', 10),
                similarity_threshold=similarity_threshold
            )
            
            # Create predicted categories
            predicted_categories = []
            
            for i, cluster in enumerate(predicted_clusters):
                # Get nodes in this cluster
                cluster_nodes = [partial_nodes[valid_node_ids[idx]] for idx in cluster if idx < len(valid_node_ids)]
                
                if len(cluster_nodes) < min_cluster_size:
                    continue
                
                # Find common properties
                common_properties = self._find_common_properties(cluster_nodes)
                
                # Generate category label
                category_label = self._generate_category_label(common_properties, cluster_nodes)
                
                # Record predicted category
                predicted_categories.append({
                    'id': f"predicted_category_{i}",
                    'label': category_label,
                    'size': len(cluster_nodes),
                    'members': [node.id for node in cluster_nodes],
                    'common_properties': common_properties,
                    'confidence': 0.7  # Default confidence for predictions
                })
            
            # Create prediction result
            prediction_result = {
                "status": "success",
                "categories": predicted_categories,
                "total_nodes": len(partial_nodes),
                "categorized_nodes": sum(len(cat['members']) for cat in predicted_categories)
            }
            
            # Store in cache
            self.category_predictions[node_set_hash] = prediction_result
            
            return prediction_result
            
        elif method == "property":
            # Property-based prediction is simpler but less accurate
            # Find common properties across partial nodes
            all_properties = set()
            for node in partial_nodes.values():
                all_properties.update(node.properties.keys())
                
            # Group nodes by property
            property_to_nodes = defaultdict(list)
            for node_id, node in partial_nodes.items():
                for prop in node.properties:
                    property_to_nodes[prop].append(node_id)
            
            # Predict categories based on common properties
            predicted_categories = []
            
            for prop, members in property_to_nodes.items():
                if len(members) >= params.get('min_members', 3):
                    member_nodes = [partial_nodes[m] for m in members if m in partial_nodes]
                    common_props = self._find_common_properties(member_nodes)
                    
                    if len(common_props) >= params.get('min_shared_props', 2):
                        category_label = self._generate_category_label(common_props, member_nodes)
                        
                        predicted_categories.append({
                            'id': f"predicted_category_{prop}",
                            'label': category_label,
                            'size': len(member_nodes),
                            'members': [node.id for node in member_nodes],
                            'common_properties': common_props,
                            'key_property': prop,
                            'confidence': 0.7
                        })
            
            # Create prediction result
            prediction_result = {
                "status": "success",
                "categories": predicted_categories,
                "total_nodes": len(partial_nodes),
                "categorized_nodes": sum(len(cat['members']) for cat in predicted_categories)
            }
            
            # Store in cache
            self.category_predictions[node_set_hash] = prediction_result
            
            return prediction_result
        
        else:
            # Fall back to similarity
            return await self.predict_categories(partial_nodes, "similarity", params)
    
    async def refine_categories_via_active_inference(self, initial_categories, source_nodes=None):
        """
        Refine categories through active inference to minimize prediction error.
        Implements Seth's active inference principle.
        
        Args:
            initial_categories: Initial category formation
            source_nodes: Source nodes used for category formation
            
        Returns:
            Refined categories
        """
        if not initial_categories or 'categories' not in initial_categories:
            return initial_categories
        
        if not source_nodes:
            # Can't refine without source nodes
            return initial_categories
        
        categories = initial_categories['categories']
        
        # Calculate initial prediction error
        initial_error = self._calculate_category_prediction_error(categories, source_nodes)
        
        # Generate candidate refinements
        candidates = self._generate_category_refinements(categories, source_nodes)
        
        # Evaluate candidates
        best_categories = categories
        best_error = initial_error
        
        for candidate in candidates:
            error = self._calculate_category_prediction_error(candidate, source_nodes)
            
            if error < best_error:
                best_error = error
                best_categories = candidate
        
        # If we found a better arrangement, create result
        if best_error < initial_error:
            refined_result = dict(initial_categories)
            refined_result['categories'] = best_categories
            refined_result['refinement'] = {
                'initial_error': initial_error,
                'refined_error': best_error,
                'improvement': initial_error - best_error
            }
            
            return refined_result
        
        # No improvement
        return initial_categories
    
    def _calculate_category_prediction_error(self, categories, source_nodes):
        """Calculate prediction error for a set of categories."""
        # Collect category memberships
        node_categories = defaultdict(list)
        
        for cat in categories:
            for member_id in cat['members']:
                node_categories[member_id].append(cat['id'])
        
        # Calculate how well categories predict node similarities
        error = 0.0
        
        # For each pair of nodes, check if they should be in same categories
        node_ids = list(source_nodes.keys())
        
        for i, node1_id in enumerate(node_ids):
            node1 = source_nodes[node1_id]
            
            for j in range(i+1, len(node_ids)):
                node2_id = node_ids[j]
                node2 = source_nodes[node2_id]
                
                # Calculate actual similarity
                if isinstance(node1.embeddings, np.ndarray) and isinstance(node2.embeddings, np.ndarray):
                    actual_similarity = np.dot(node1.embeddings, node2.embeddings) / (
                        np.linalg.norm(node1.embeddings) * np.linalg.norm(node2.embeddings)
                    )
                else:
                    # Fallback
                    actual_similarity = 0.5
                
                # Calculate predicted similarity based on shared categories
                cat1 = set(node_categories[node1_id])
                cat2 = set(node_categories[node2_id])
                
                shared_categories = len(cat1.intersection(cat2))
                total_categories = len(cat1.union(cat2))
                
                if total_categories > 0:
                    predicted_similarity = shared_categories / total_categories
                else:
                    predicted_similarity = 0.0
                
                # Add squared error
                pair_error = (predicted_similarity - actual_similarity) ** 2
                error += pair_error
        
        # Normalize by number of pairs
        n_pairs = (len(node_ids) * (len(node_ids) - 1)) // 2
        if n_pairs > 0:
            error /= n_pairs
        
        return error
    
    def _generate_category_refinements(self, categories, source_nodes):
        """Generate candidate category refinements."""
        candidates = []
        
        # Strategy 1: Move nodes between categories
        # Find boundary nodes (nodes that could belong to multiple categories)
        node_categories = defaultdict(list)
        category_nodes = {}
        
        for i, cat in enumerate(categories):
            category_nodes[i] = cat['members']
            for member_id in cat['members']:
                node_categories[member_id].append(i)
        
        # Try moving boundary nodes
        for node_id, cat_indices in node_categories.items():
            if len(cat_indices) == 1 and node_id in source_nodes:
                # Node belongs to only one category - consider moving it
                current_cat_idx = cat_indices[0]
                
                # Find most similar category
                node = source_nodes[node_id]
                best_similarity = -1
                best_target_idx = -1
                
                for target_idx, members in category_nodes.items():
                    if target_idx == current_cat_idx:
                        continue
                    
                    # Calculate average similarity to this category
                    similarities = []
                    for member_id in members:
                        if member_id in source_nodes:
                            member = source_nodes[member_id]
                            if isinstance(node.embeddings, np.ndarray) and isinstance(member.embeddings, np.ndarray):
                                sim = np.dot(node.embeddings, member.embeddings) / (
                                    np.linalg.norm(node.embeddings) * np.linalg.norm(member.embeddings)
                                )
                                similarities.append(sim)
                    
                    if similarities:
                        avg_similarity = sum(similarities) / len(similarities)
                        if avg_similarity > best_similarity:
                            best_similarity = avg_similarity
                            best_target_idx = target_idx
                
                if best_similarity > 0.5:  # Only if reasonably similar
                    # Create a candidate with this node moved
                    candidate = []
                    
                    for i, cat in enumerate(categories):
                        new_cat = dict(cat)
                        
                        if i == current_cat_idx:
                            # Remove from this category
                            new_cat['members'] = [m for m in cat['members'] if m != node_id]
                            new_cat['size'] = len(new_cat['members'])
                        elif i == best_target_idx:
                            # Add to this category
                            new_cat['members'] = cat['members'] + [node_id]
                            new_cat['size'] = len(new_cat['members'])
                        
                        candidate.append(new_cat)
                    
                    candidates.append(candidate)
        
        # Strategy 2: Merge similar categories
        for i in range(len(categories)):
            for j in range(i+1, len(categories)):
                cat1 = categories[i]
                cat2 = categories[j]
                
                # Calculate overlap
                members1 = set(cat1['members'])
                members2 = set(cat2['members'])
                
                overlap = len(members1.intersection(members2))
                
                if overlap > 0 or self._are_categories_similar(cat1, cat2):
                    # Create a merged category
                    merged_members = list(members1.union(members2))
                    
                    # Create common properties
                    common_properties = {}
                    for prop in cat1['common_properties']:
                        if prop in cat2['common_properties'] and cat1['common_properties'][prop] == cat2['common_properties'][prop]:
                            common_properties[prop] = cat1['common_properties'][prop]
                    
                    merged_cat = {
                        'id': f"merged_{cat1['id']}_{cat2['id']}",
                        'label': f"Merged: {cat1['label']} + {cat2['label']}",
                        'size': len(merged_members),
                        'members': merged_members,
                        'common_properties': common_properties
                    }
                    
                    # Create candidate with merged category
                    candidate = []
                    for k, cat in enumerate(categories):
                        if k != i and k != j:
                            candidate.append(cat)
                    
                    candidate.append(merged_cat)
                    candidates.append(candidate)
        
        return candidates
    
    def _are_categories_similar(self, cat1, cat2):
        """Check if two categories are similar enough to be merged."""
        # Check for similar properties
        props1 = set(cat1['common_properties'].keys())
        props2 = set(cat2['common_properties'].keys())
        
        if not props1 or not props2:
            return False
        
        # Check property overlap
        overlap = props1.intersection(props2)
        if len(overlap) >= min(2, min(len(props1), len(props2))):
            return True
        
        # Check for similar labels
        label1 = cat1['label'].lower()
        label2 = cat2['label'].lower()
        
        # Simple string similarity
        words1 = set(label1.split())
        words2 = set(label2.split())
        
        word_overlap = words1.intersection(words2)
        if len(word_overlap) >= min(2, min(len(words1), len(words2))):
            return True
        
        return False
    
    def _hash_node_set(self, node_ids, method):
        """Create a stable hash for node sets."""
        # Sort node IDs for consistency
        sorted_ids = sorted(str(nid) for nid in node_ids)
        return hash((tuple(sorted_ids), method))



# === FILE: asf\layer3_cognitive_boundary\formation\concept_formation.py ===

# Enhanced ConceptFormationEngine with predictive capabilities

import uuid
import time
import numpy as np
import logging
from typing import Dict, Any, Optional, List
from collections import defaultdict

from asf.semantic_organization.enums import SemanticNodeType, SemanticConfidenceState
from asf.semantic_organization.core.semantic_node import SemanticNode
from asf.semantic_organization.core.semantic_relation import SemanticRelation

class ConceptFormationEngine:
    """
    Enhanced concept formation engine with anticipatory capabilities.
    Implements Seth's principle that cognition is predictive processing.
    """
    def __init__(self, semantic_network):
        self.semantic_network = semantic_network
        self.similarity_threshold = 0.85
        self.property_confidence_threshold = 0.4
        self.formation_history = []
        self.logger = logging.getLogger("ASF.Layer3.ConceptFormation")
        # Enhanced formation parameters
        self.hierarchical_enabled = True
        self.adaptive_thresholds = True
        self.formation_confidence = 0.7
        
        # Seth's Data Paradox enhancements
        self.partial_concept_cache = {}  # Partial features -> anticipated concept
        self.concept_formation_errors = defaultdict(list)  # Feature set hash -> prediction errors
        self.precision_values = {}  # Feature set hash -> precision
        
    async def anticipate_concept(self, partial_features, context=None):
        """
        Anticipate a concept based on partial features.
        Implements Seth's principle of "controlled hallucination".
        
        Args:
            partial_features: Incomplete set of features
            context: Additional context information
            
        Returns:
            Anticipated concept information
        """
        context = context or {}
        features_hash = self._hash_features(partial_features)
        
        # Check cache for identical partial features
        if features_hash in self.partial_concept_cache:
            return self.partial_concept_cache[features_hash]
        
        # Generate partial embeddings
        partial_embeddings = self._generate_embeddings(partial_features)
        
        # Find similar existing concepts based on available features
        similar_concepts = []
        
        # Search semantic network for similar concepts
        # Create a temporary concept node for similarity search
        temp_id = f"temp_{uuid.uuid4().hex[:8]}"
        temp_node = SemanticNode(
            id=temp_id,
            label="Temporary anticipation node",
            node_type=SemanticNodeType.CONCEPT.value,
            properties=partial_features,
            embeddings=partial_embeddings
        )
        
        # Add temporary node to network for similarity search
        await self.semantic_network.add_node(temp_node, update_tensors=False)
        
        # Get similar concepts
        similarity_results = await self.semantic_network.get_similar_nodes(
            temp_id, k=5, threshold=0.6
        )
        
        # Remove temporary node
        # (In a real implementation, we would need a method to remove nodes)
        
        # Process similar concepts to anticipate complete concept
        anticipated_properties = dict(partial_features)  # Start with known properties
        
        if similarity_results:
            # For each similar concept, extract additional properties
            property_candidates = defaultdict(list)
            
            for concept_id, similarity in similarity_results:
                # Get the concept
                concept = await self.semantic_network.get_node(concept_id)
                if not concept:
                    continue
                
                # Extract properties not in partial features
                for prop_name, value in concept.properties.items():
                    if prop_name not in partial_features:
                        # Store with similarity as weight
                        property_candidates[prop_name].append((value, similarity))
            
            # Determine which properties to include in anticipation
            for prop_name, candidates in property_candidates.items():
                if not candidates:
                    continue
                
                # For numeric properties, use weighted average
                if all(isinstance(v[0], (int, float)) for v in candidates):
                    total_weight = sum(weight for _, weight in candidates)
                    if total_weight > 0:
                        weighted_sum = sum(value * weight for value, weight in candidates)
                        anticipated_properties[prop_name] = weighted_sum / total_weight
                else:
                    # For non-numeric, use most heavily weighted value
                    best_candidate = max(candidates, key=lambda x: x[1])
                    anticipated_properties[prop_name] = best_candidate[0]
        
        # Generate embeddings for anticipated concept
        anticipated_embeddings = self._generate_embeddings(anticipated_properties)
        
        # Calculate confidence in anticipation
        confidence_factors = [
            len(partial_features) / max(1, len(anticipated_properties)),  # Ratio of known to total
            0.7,  # Base confidence
        ]
        if similarity_results:
            confidence_factors.append(similarity_results[0][1])  # Similarity to best match
        
        anticipated_confidence = min(0.9, sum(confidence_factors) / len(confidence_factors))
        
        # Create anticipated concept
        anticipated_concept = {
            'properties': anticipated_properties,
            'embeddings': anticipated_embeddings,
            'confidence': anticipated_confidence,
            'similarity_results': similarity_results,
            'partial_features': partial_features,
        }
        
        # Store in cache
        self.partial_concept_cache[features_hash] = anticipated_concept
        
        return anticipated_concept
    
    async def form_concept(self, features, source_id=None, context=None):
        """
        Form a concept from features with anticipation evaluation.
        If partial anticipation was done earlier, evaluate its accuracy.
        
        Args:
            features: Feature dictionary (name: value)
            source_id: Source entity ID (if available)
            context: Additional context information
            
        Returns:
            Concept node ID
        """
        context = context or {}
        features_hash = self._hash_features(features)
        
        # Check if we previously anticipated this concept
        anticipated = None
        if 'anticipated_id' in context:
            anticipated_id = context['anticipated_id']
            if anticipated_id in self.partial_concept_cache:
                anticipated = self.partial_concept_cache[anticipated_id]
        
        # If we have an anticipation, evaluate its accuracy
        if anticipated:
            anticipated_properties = anticipated['properties']
            
            # Calculate prediction error
            prediction_errors = []
            
            for prop_name, actual_value in features.items():
                if prop_name in anticipated_properties:
                    anticipated_value = anticipated_properties[prop_name]
                    
                    # Calculate error based on value type
                    if isinstance(actual_value, (int, float)) and isinstance(anticipated_value, (int, float)):
                        error = abs(actual_value - anticipated_value) / (1.0 + abs(actual_value))
                        prediction_errors.append(error)
                    elif isinstance(actual_value, str) and isinstance(anticipated_value, str):
                        # Simple string comparison (could use more advanced metrics)
                        error = 0.0 if actual_value == anticipated_value else 1.0
                        prediction_errors.append(error)
            
            if prediction_errors:
                # Calculate average error
                avg_error = sum(prediction_errors) / len(prediction_errors)
                
                # Track error for precision calculation
                self.concept_formation_errors[features_hash].append(avg_error)
                
                # Limit history size
                if len(self.concept_formation_errors[features_hash]) > 20:
                    self.concept_formation_errors[features_hash] = self.concept_formation_errors[features_hash][-20:]
                
                # Update precision (inverse variance)
                if len(self.concept_formation_errors[features_hash]) > 1:
                    variance = np.var(self.concept_formation_errors[features_hash])
                    precision = 1.0 / (variance + 1e-6)  # Avoid division by zero
                    self.precision_values[features_hash] = min(10.0, precision)  # Cap very high precision
        
        # Proceed with standard concept formation (existing implementation)
        # Rest of the original form_concept method...
        
        # But incorporate anticipation confidence if available
        if anticipated and 'confidence' in anticipated:
            # Use anticipated confidence as a factor
            anticipated_confidence = anticipated['confidence']
            context['anticipated_confidence'] = anticipated_confidence
        
        # Continue with original implementation...
        # Generate embeddings
        embeddings = self._generate_embeddings(features)
        
        # Create concept properties
        properties = {}
        for name, value in features.items():
            if isinstance(value, (int, float, str, bool)):
                properties[name] = value
                
        # Calculate confidence based on source confidence if available
        source_confidence = context.get('source_confidence', 0.7)
        anticipated_confidence = context.get('anticipated_confidence', 0.0)
        
        # Blend confidences based on prediction precision if available
        precision = self.precision_values.get(features_hash, 1.0)
        confidence_weight = min(0.8, precision / (precision + 1.0))
        
        # Higher precision gives more weight to anticipated confidence
        confidence = (
            self.formation_confidence * (1.0 - confidence_weight) +
            source_confidence * (1.0 - confidence_weight) +
            anticipated_confidence * confidence_weight * 2
        ) / (2.0 + confidence_weight)
        
        confidence = max(self.formation_confidence, min(0.95, confidence))
        
        # Continue with the rest of the original form_concept implementation...
        
        # Add anticipation metadata
        metadata = {
            'created_from': context.get('created_from', 'direct'),
            'formation_time': time.time()
        }
        
        if anticipated:
            metadata['anticipated'] = True
            metadata['anticipation_error'] = sum(prediction_errors) / len(prediction_errors) if prediction_errors else 0
            metadata['anticipation_precision'] = self.precision_values.get(features_hash, 1.0)
        
        # Create node
        concept_id = f"concept_{uuid.uuid4().hex[:8]}"
        label = context.get('label', f"Concept_{concept_id[-6:]}")
        
        concept_node = SemanticNode(
            id=concept_id,
            label=label,
            node_type=SemanticNodeType.CONCEPT.value,
            properties=properties,
            embeddings=embeddings,
            confidence=confidence,
            confidence_state=SemanticConfidenceState.PROVISIONAL,
            confidence_evidence={"positive": 2.0, "negative": 1.0},
            source_ids=[source_id] if source_id else [],
            metadata=metadata
        )
        
        # Add to network and finalize
        await self.semantic_network.add_node(concept_node)
        
        # Record in history
        self.formation_history.append({
            'action': 'created',
            'concept_id': concept_id,
            'source_id': source_id,
            'anticipated': anticipated is not None,
            'timestamp': time.time()
        })
        
        return concept_id
    
    async def generate_counterfactual_concepts(self, features, modifications, context=None):
        """
        Generate counterfactual concepts by modifying features.
        Implements Seth's principle of testing hypotheses through counterfactuals.
        
        Args:
            features: Original feature dictionary
            modifications: List of modification operations
            context: Additional context
            
        Returns:
            List of counterfactual concepts
        """
        context = context or {}
        counterfactual_concepts = []
        
        for modification in modifications:
            # Apply modification to features
            modified_features = self._apply_feature_modification(features, modification)
            
            # Generate concept from modified features
            cf_context = dict(context)
            cf_context['created_from'] = 'counterfactual'
            cf_context['modification'] = modification
            
            cf_concept_id = await self.form_concept(
                modified_features,
                source_id=context.get('source_id'),
                context=cf_context
            )
            
            if cf_concept_id:
                counterfactual_concepts.append({
                    'concept_id': cf_concept_id,
                    'features': modified_features,
                    'modification': modification
                })
        
        return counterfactual_concepts
    
    def _apply_feature_modification(self, features, modification):
        """Apply a modification to features to create a counterfactual."""
        modified_features = dict(features)
        
        mod_type = modification.get('type', 'change')
        
        if mod_type == 'change':
            # Change specific properties
            properties = modification.get('properties', {})
            for prop_name, new_value in properties.items():
                modified_features[prop_name] = new_value
                
        elif mod_type == 'remove':
            # Remove properties
            properties = modification.get('properties', [])
            for prop_name in properties:
                if prop_name in modified_features:
                    del modified_features[prop_name]
                    
        elif mod_type == 'scale':
            # Scale numeric properties
            factor = modification.get('factor', 1.0)
            properties = modification.get('properties', [])
            
            for prop_name in properties:
                if prop_name in modified_features and isinstance(modified_features[prop_name], (int, float)):
                    modified_features[prop_name] = modified_features[prop_name] * factor
                    
        return modified_features
        
    def _hash_features(self, features):
        """Create a stable hash for feature sets."""
        # Sort features by name for consistency
        sorted_items = sorted((str(k), str(v)) for k, v in features.items())
        return hash(tuple(sorted_items))



# === FILE: asf\layer3_cognitive_boundary\formation\conceptual_blending.py ===

# Enhanced ConceptualBlendingEngine with predictive capabilities

import torch
import torch.nn.functional as F
import uuid
import time
import numpy as np
import logging
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter

from asf.semantic_organization.core.semantic_node import SemanticNode
from asf.semantic_organization.core.semantic_relation import SemanticRelation
from asf.semantic_organization.enums import SemanticConfidenceState

class ConceptualBlendingEngine:
    """
    Implements conceptual blending with predictive capabilities.
    Anticipates blend outcomes before fully processing inputs.
    """
    def __init__(self, semantic_network, concept_formation_engine):
        self.semantic_network = semantic_network
        self.concept_formation_engine = concept_formation_engine
        self.blend_history = []
        self.logger = logging.getLogger("ASF.Layer3.ConceptualBlending")
        
        # Enhanced blending parameters
        self.use_tensor_blending = True
        self.blend_confidence = 0.6
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Seth's Data Paradox enhancements
        self.blend_predictions = {}  # Input hash -> predicted blend
        self.blend_errors = defaultdict(list)  # Input hash -> prediction errors
        self.blend_precision = {}  # Input hash -> precision
        
    async def anticipate_blend(self, input_ids, blend_type="composition", context=None):
        """
        Anticipate blend outcome before fully processing inputs.
        Implements Seth's principle of "controlled hallucination".
        
        Args:
            input_ids: List of input concept IDs to blend
            blend_type: Type of blend to create
            context: Additional context for blending
            
        Returns:
            Anticipated blend information
        """
        context = context or {}
        input_hash = self._hash_inputs(input_ids, blend_type)
        
        # Check cache for identical input combination
        if input_hash in self.blend_predictions:
            return self.blend_predictions[input_hash]
        
        # Get input concepts
        input_concepts = []
        for concept_id in input_ids:
            node = await self.semantic_network.get_node(concept_id)
            if node:
                input_concepts.append(node)
        
        if len(input_concepts) < 2:
            return None
        
        # Create partial generic space based on input concept properties
        generic_space = await self._create_generic_space(input_concepts)
        
        # Anticipate blended properties based on blend type
        anticipated_properties = dict(generic_space['properties'])
        
        # Anticipate emergent properties
        anticipated_emergent = await self._anticipate_emergent_properties(
            input_concepts, anticipated_properties, blend_type, context
        )
        
        # Add anticipated emergent properties
        anticipated_properties.update(anticipated_emergent)
        
        # Create anticipated embedding
        anticipated_embedding = await self._create_blend_embedding(
            input_concepts, generic_space, blend_type
        )
        
        # Calculate confidence in anticipation
        confidence_factors = [
            generic_space['confidence'],  # Generic space confidence
            0.7,  # Base confidence
        ]
        
        anticipated_confidence = min(0.9, np.mean(confidence_factors))
        
        # Create anticipated blend
        if len(input_concepts) == 2:
            anticipated_label = f"{input_concepts[0].label}-{input_concepts[1].label} blend"
        else:
            anticipated_label = f"Multi-concept blend ({len(input_concepts)})"
        
        anticipated_blend = {
            'properties': anticipated_properties,
            'emergent_properties': anticipated_emergent,
            'embeddings': anticipated_embedding,
            'confidence': anticipated_confidence,
            'label': anticipated_label,
            'input_ids': input_ids,
            'blend_type': blend_type
        }
        
        # Store in cache
        self.blend_predictions[input_hash] = anticipated_blend
        
        return anticipated_blend
    
    async def create_blend(self, input_ids, blend_type="composition", context=None):
        """
        Create a blend with anticipation evaluation.
        If anticipation was done earlier, evaluate its accuracy.
        """
        context = context or {}
        input_hash = self._hash_inputs(input_ids, blend_type)
        
        # Check if we previously anticipated this blend
        anticipated = None
        if input_hash in self.blend_predictions:
            anticipated = self.blend_predictions[input_hash]
        
        # Standard blend creation (existing implementation)
        # Validate inputs
        input_concepts = []
        for concept_id in input_ids:
            node = await self.semantic_network.get_node(concept_id)
            if node:
                input_concepts.append(node)
        
        if len(input_concepts) < 2:
            self.logger.warning("Need at least 2 valid concepts for blending")
            return None
        
        # Create generic space
        generic_space = await self._create_generic_space(input_concepts)
        
        # Create blended space based on type
        blend_result = await self._create_tensor_blend(
            input_concepts, generic_space, blend_type, context
        )
        
        if not blend_result:
            return None
        
        # If we had an anticipation, evaluate its accuracy
        if anticipated:
            anticipated_properties = anticipated['properties']
            actual_properties = blend_result['blend_concept'].properties
            
            # Calculate prediction errors
            property_errors = []
            
            # Compare properties
            for prop_name in set(list(anticipated_properties.keys()) + list(actual_properties.keys())):
                if prop_name in anticipated_properties and prop_name in actual_properties:
                    anticipated_value = anticipated_properties[prop_name]
                    actual_value = actual_properties[prop_name]
                    
                    # Calculate error based on value type
                    if isinstance(actual_value, (int, float)) and isinstance(anticipated_value, (int, float)):
                        error = abs(actual_value - anticipated_value) / (1.0 + abs(actual_value))
                        property_errors.append(error)
                    elif isinstance(actual_value, str) and isinstance(anticipated_value, str):
                        # Simple string comparison
                        error = 0.0 if actual_value == anticipated_value else 1.0
                        property_errors.append(error)
                else:
                    # Missing or extra property
                    property_errors.append(1.0)
            
            if property_errors:
                # Calculate average error
                avg_error = sum(property_errors) / len(property_errors)
                
                # Track error for precision calculation
                self.blend_errors[input_hash].append(avg_error)
                
                # Limit history size
                if len(self.blend_errors[input_hash]) > 20:
                    self.blend_errors[input_hash] = self.blend_errors[input_hash][-20:]
                
                # Update precision (inverse variance)
                if len(self.blend_errors[input_hash]) > 1:
                    variance = np.var(self.blend_errors[input_hash])
                    precision = 1.0 / (variance + 1e-6)  # Avoid division by zero
                    self.blend_precision[input_hash] = min(10.0, precision)  # Cap very high precision
            
            # Add anticipation metadata to blend
            blend_result['blend_concept'].metadata['anticipated'] = True
            blend_result['blend_concept'].metadata['anticipation_error'] = avg_error if property_errors else 0
            blend_result['blend_concept'].metadata['anticipation_precision'] = self.blend_precision.get(input_hash, 1.0)
        
        # Add blend concept to network
        blend_id = blend_result['blend_id']
        blend_concept = blend_result['blend_concept']
        await self.semantic_network.add_node(blend_concept)
        
        # Create relations between blend and inputs
        for concept in input_concepts:
            relation_id = f"rel_{uuid.uuid4().hex[:8]}"
            
            blend_relation = SemanticRelation(
                id=relation_id,
                source_id=blend_id,
                target_id=concept.id,
                relation_type="has_input",
                weight=0.9,
                confidence=blend_concept.confidence * 0.9
            )
            
            await self.semantic_network.add_relation(blend_relation)
        
        # Record in history
        self.blend_history.append({
            'blend_id': blend_id,
            'input_ids': input_ids,
            'blend_type': blend_type,
            'anticipated': anticipated is not None,
            'timestamp': time.time()
        })
        
        return blend_id
    
    async def _anticipate_emergent_properties(self, input_concepts, base_properties, blend_type, context):
        """
        Anticipate emergent properties that will arise from blending.
        Implements Seth's notion of emergent features through predictive processing.
        """
        emergent_properties = {}
        
        # For numeric properties, anticipate relationships and enhancements
        numeric_props = {}
        for concept in input_concepts:
            for name, value in concept.properties.items():
                if isinstance(value, (int, float)):
                    if name not in numeric_props:
                        numeric_props[name] = []
                    numeric_props[name].append(value)
        
        # Generate emergent properties based on blend type
        for name, values in numeric_props.items():
            if len(values) >= 2:
                if blend_type == "composition":
                    # For composition, anticipate maximum and average
                    if name not in base_properties:
                        emergent_properties[f"max_{name}"] = max(values)
                        
                elif blend_type == "completion":
                    # For completion, anticipate sequence continuation
                    if len(values) >= 3:
                        # Check if values form a sequence
                        diffs = [values[i] - values[i-1] for i in range(1, len(values))]
                        if max(diffs) - min(diffs) < 0.1 * abs(sum(diffs)/len(diffs)):
                            # Arithmetic sequence detected, predict next value
                            next_value = values[-1] + (sum(diffs)/len(diffs))
                            emergent_properties[f"next_{name}"] = next_value
                            
                elif blend_type == "elaboration":
                    # For elaboration, anticipate statistical properties
                    variance = np.var(values)
                    emergent_properties[f"variance_{name}"] = variance
                    emergent_properties[f"stddev_{name}"] = variance ** 0.5
        
        # Generate relational emergent properties
        if len(input_concepts) >= 2:
            for i, concept1 in enumerate(input_concepts):
                for j in range(i+1, len(input_concepts)):
                    concept2 = input_concepts[j]
                    
                    # For each pair of numeric properties, anticipate relationships
                    for name1, value1 in concept1.properties.items():
                        if not isinstance(value1, (int, float)):
                            continue
                            
                        for name2, value2 in concept2.properties.items():
                            if not isinstance(value2, (int, float)):
                                continue
                                
                            # Anticipate ratio if meaningful
                            if value2 != 0 and f"{name1}_to_{name2}_ratio" not in base_properties:
                                ratio = value1 / value2
                                emergent_properties[f"{name1}_to_{name2}_ratio"] = ratio
        
        return emergent_properties
    
    def _hash_inputs(self, input_ids, blend_type):
        """Create a stable hash for input combinations."""
        # Sort inputs for consistency
        sorted_inputs = sorted(input_ids)
        return hash((tuple(sorted_inputs), blend_type))



# === FILE: asf\layer3_cognitive_boundary\processing\__init__.py ===

# Layer 3: Semantic Organization Layer
# Processing module for asynchronous queue and priority management

from asf.semantic_organization.processing.async_queue import AsyncProcessingQueue
from asf.semantic_organization.processing.priority_manager import AdaptivePriorityManager

__all__ = ['AsyncProcessingQueue', 'AdaptivePriorityManager']



# === FILE: asf\layer3_cognitive_boundary\processing\async_queue.py ===

# Enhancement for async_queue.py - Add to existing class

import asyncio
import datetime
import time
import heapq
import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any, Optional, Union, Callable

class AsyncProcessingQueue:
    # Existing initialization and methods...
    
    # Add these fields to __init__:
    # self.task_patterns = defaultdict(list)  # Task type -> past occurrences
    # self.task_transitions = defaultdict(lambda: defaultdict(int))  # Task A -> Task B -> count
    # self.context_task_frequencies = defaultdict(lambda: defaultdict(int))  # Context -> Task type -> count
    # self.prediction_errors = defaultdict(list)  # Task type -> prediction errors
    # self.precision_values = {}  # Task type -> precision
    
    async def predict_tasks(self, context):
        """
        Predict which tasks are likely to be needed in the near future.
        Implements Seth's predictive processing principle.
        
        Args:
            context: Current processing context
            
        Returns:
            Dict mapping task types to predicted priorities
        """
        # Initialize prediction fields if not present
        if not hasattr(self, 'task_patterns'):
            self.task_patterns = defaultdict(list)
        if not hasattr(self, 'task_transitions'):
            self.task_transitions = defaultdict(lambda: defaultdict(int))
        if not hasattr(self, 'context_task_frequencies'):
            self.context_task_frequencies = defaultdict(lambda: defaultdict(int))
            
        predicted_tasks = {}
        
        # Method 1: Use task transitions (which tasks typically follow others)
        recent_tasks = self._get_recent_task_types(5)  # Get 5 most recent task types
        
        for task_type in recent_tasks:
            if task_type in self.task_transitions:
                transitions = self.task_transitions[task_type]
                if transitions:
                    # Find most likely next tasks
                    total_count = sum(transitions.values())
                    for next_task, count in transitions.items():
                        probability = count / total_count
                        if probability > 0.3:  # Threshold for prediction
                            predicted_tasks[next_task] = probability
        
        # Method 2: Use context-based predictions
        context_key = self._get_context_key(context)
        if context_key in self.context_task_frequencies:
            frequencies = self.context_task_frequencies[context_key]
            total_count = sum(frequencies.values())
            if total_count > 0:
                for task_type, count in frequencies.items():
                    probability = count / total_count
                    if probability > 0.3:  # Threshold for prediction
                        predicted_tasks[task_type] = max(predicted_tasks.get(task_type, 0), probability)
        
        # Convert probabilities to priorities (0.5-0.9 range)
        predicted_priorities = {
            task_type: 0.5 + (0.4 * probability) 
            for task_type, probability in predicted_tasks.items()
        }
        
        return predicted_priorities
    
    async def preemptively_schedule(self, context):
        """
        Schedule predicted high-priority tasks before they're explicitly requested.
        Implements Seth's active inference principle.
        
        Args:
            context: Current processing context
            
        Returns:
            List of preemptively scheduled task IDs
        """
        # Initialize precision values if not present
        if not hasattr(self, 'precision_values'):
            self.precision_values = {}
            
        # Predict tasks
        predicted_tasks = await self.predict_tasks(context)
        
        # Filter to high-confidence predictions
        high_priority_predictions = {
            task_type: priority 
            for task_type, priority in predicted_tasks.items() 
            if priority > 0.7 and task_type in self.precision_values and self.precision_values[task_type] > 2.0
        }
        
        # No high-confidence predictions
        if not high_priority_predictions:
            return []
        
        # Schedule top predictions
        scheduled_tasks = []
        for task_type, priority in sorted(high_priority_predictions.items(), key=lambda x: -x[1]):
            # Create a preemptive task
            task_id = f"preemptive_{task_type}_{int(time.time()*1000)}"
            
            # In real implementation, we would need task factory functions to create actual tasks
            # This is a simplified placeholder
            scheduled_tasks.append(task_id)
            
            if hasattr(self, 'logger'):
                self.logger.info(f"Preemptively scheduled task {task_id} with priority {priority}")
        
        return scheduled_tasks
    
    async def submit_task(self, task_id, task_func, priority=0.5, dependencies=None, args=None, kwargs=None):
        """Submit a task with prediction evaluation."""
        # Initialize prediction fields if not present
        if not hasattr(self, 'prediction_errors'):
            self.prediction_errors = defaultdict(list)
        if not hasattr(self, 'precision_values'):
            self.precision_values = {}
        
        async with self.lock:
            # Existing task submission code...
            
            # Extract task type from kwargs
            args = args or []
            kwargs = kwargs or {}
            dependencies = dependencies or []
            task_type = kwargs.get('task_type', 'unknown')
            context = kwargs.get('context', {})
            
            # Check if this task was predicted
            predicted_priority = await self._check_predicted_task(task_type, context)
            if predicted_priority is not None:
                # Task was predicted, evaluate accuracy
                prediction_error = abs(predicted_priority - priority) / (1.0 + priority)
                self.prediction_errors[task_type].append(prediction_error)
                
                # Limit history size
                if len(self.prediction_errors[task_type]) > 20:
                    self.prediction_errors[task_type] = self.prediction_errors[task_type][-20:]
                
                # Update precision (inverse variance)
                if len(self.prediction_errors[task_type]) > 1:
                    variance = np.var(self.prediction_errors[task_type])
                    precision = 1.0 / (variance + 1e-6)  # Avoid division by zero
                    self.precision_values[task_type] = min(10.0, precision)  # Cap very high precision
                    
                # Use predicted priority if it's higher (preemptive processing)
                priority = max(priority, predicted_priority)
            
            # Record task pattern for future prediction
            await self._record_task_pattern(task_type, context)
            
            # Continue with existing task submission code...
            
            # IMPORTANT: Make sure to save this result and return it
            result = await super().submit_task(task_id, task_func, priority, dependencies, args, kwargs)
            return result
    
    async def _check_predicted_task(self, task_type, context):
        """Check if a task was predicted and return its predicted priority."""
        # Initialize if needed
        if not hasattr(self, 'task_patterns'):
            return None
            
        # Predict tasks for current context
        predicted_tasks = await self.predict_tasks(context)
        
        # Check if this task type was predicted
        if task_type in predicted_tasks:
            return predicted_tasks[task_type]
        
        return None
    
    async def _record_task_pattern(self, task_type, context):
        """Record task pattern for future prediction."""
        # Initialize if needed
        if not hasattr(self, 'task_patterns'):
            self.task_patterns = defaultdict(list)
        if not hasattr(self, 'task_transitions'):
            self.task_transitions = defaultdict(lambda: defaultdict(int))
        if not hasattr(self, 'context_task_frequencies'):
            self.context_task_frequencies = defaultdict(lambda: defaultdict(int))
            
        # Record task in history
        self.task_patterns[task_type].append({
            'timestamp': time.time(),
            'context': context
        })
        
        # Limit history size
        if len(self.task_patterns[task_type]) > 100:
            self.task_patterns[task_type] = self.task_patterns[task_type][-100:]
        
        # Record transition from previous task
        recent_tasks = self._get_recent_task_types(1)
        if recent_tasks:
            previous_task = recent_tasks[0]
            self.task_transitions[previous_task][task_type] += 1
        
        # Record context frequency
        context_key = self._get_context_key(context)
        self.context_task_frequencies[context_key][task_type] += 1
    
    def _get_recent_task_types(self, count=5):
        """Get most recent task types."""
        # Initialize if needed
        if not hasattr(self, 'task_patterns'):
            return []
            
        recent_tasks = []
        all_tasks = []
        
        # Collect all tasks with timestamps
        for task_type, occurrences in self.task_patterns.items():
            if occurrences:
                all_tasks.append((task_type, occurrences[-1]['timestamp']))
        
        # Sort by timestamp (descending)
        all_tasks.sort(key=lambda x: -x[1])
        
        # Take most recent
        return [task_type for task_type, _ in all_tasks[:count]]
    
    def _get_context_key(self, context):
        """Generate stable key for context."""
        # Simple implementation - in production would use better hashing
        if not context:
            return "default_context"
        
        # Sort context items for consistent key
        sorted_items = sorted((str(k), str(v)) for k, v in context.items())
        return "_".join(f"{k}:{v}" for k, v in sorted_items)
    
    async def get_prediction_stats(self):
        """Get statistics about task predictions."""
        # Initialize if needed
        if not hasattr(self, 'prediction_errors'):
            return {}
        if not hasattr(self, 'precision_values'):
            self.precision_values = {}
        if not hasattr(self, 'task_transitions'):
            self.task_transitions = defaultdict(lambda: defaultdict(int))
            
        stats = {}
        
        for task_type in self.prediction_errors:
            errors = self.prediction_errors[task_type]
            if errors:
                precision = self.precision_values.get(task_type, 1.0)
                stats[task_type] = {
                    'mean_error': sum(errors) / len(errors),
                    'error_count': len(errors),
                    'precision': precision,
                    'transitions': dict(self.task_transitions.get(task_type, {}))
                }
        
        return stats



# === FILE: asf\layer3_cognitive_boundary\processing\priority_manager.py ===

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import datetime
import logging
from typing import Dict, Any, Optional, List
from collections import defaultdict

class AdaptivePriorityManager:
    """
    Uses reinforcement learning to dynamically adjust task priorities.
    Optimizes semantic organization processing based on performance metrics and context.
    Implements Seth's principle of active inference for resource allocation.
    """
    def __init__(self, learning_rate=0.1, discount_factor=0.9):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_values = {}  # State-action value function
        self.experience_buffer = []  # For experience replay
        self.buffer_size = 1000
        self.batch_size = 32
        self.state_features = {}  # Cache of state features
        self.logger = logging.getLogger("ASF.Layer3.AdaptivePriorityManager")
        
        # Neural network for complex state representations
        self.use_neural_model = False
        self.neural_model = None
        
        # Seth's Data Paradox enhancements
        self.task_predictions = {}  # Task type -> predicted priorities
        self.priority_errors = defaultdict(list)  # Task type -> prediction errors
        self.priority_precision = {}  # Task type -> precision
        
    def initialize_neural_model(self, state_dim=8, action_dim=3):
        """Initialize neural network model for deep reinforcement learning."""
        self.use_neural_model = True
        
        # Simple feedforward network for Q-value prediction
        self.neural_model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        self.optimizer = torch.optim.Adam(self.neural_model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.5, patience=5, verbose=True
        )
        
    async def get_priority(self, task_type, context):
        """
        Determine optimal priority for a task based on current state.
        
        Args:
            task_type: Type of task to prioritize
            context: Additional context about the task
            
        Returns:
            Priority value between 0 and 1
        """
        # Extract state features
        state = self._extract_state_features(task_type, context)
        
        if self.use_neural_model and self.neural_model is not None:
            # Neural network-based priority assignment
            return await self._get_neural_priority(state)
            
        # Discretize state for table lookup
        state_key = self._discretize_state(state)
        
        # Get Q-values for this state
        if state_key not in self.q_values:
            self.q_values[state_key] = {
                'high': 0.7,
                'medium': 0.5,
                'low': 0.3
            }
            
        q_values = self.q_values[state_key]
        
        # Choose best action (priority level)
        best_action = max(q_values.items(), key=lambda x: x[1])[0]
        
        # Convert action to priority value
        priority_map = {
            'high': 0.9,
            'medium': 0.5,
            'low': 0.2
        }
        priority = priority_map.get(best_action, 0.5)
        
        # Store state for later update
        self.state_features[task_type] = state
        
        return priority
        
    async def _get_neural_priority(self, state):
        """Get priority using neural network."""
        # Convert state to tensor
        state_tensor = torch.tensor(list(state.values()), 
                                   dtype=torch.float32).to(self.neural_model[0].weight.device)
        
        # Forward pass
        with torch.no_grad():
            q_values = self.neural_model(state_tensor)
            
        # Convert to priority
        # For neural model, we output 3 values: low, medium, high priority
        action_idx = torch.argmax(q_values).item()
        
        # Map to priority values
        priority_values = [0.2, 0.5, 0.9]  # low, medium, high
        
        return priority_values[action_idx]
        
    async def predict_priority(self, task_type, future_context):
        """
        Predict future priority for a task type in a given context.
        Implements Seth's predictive processing principle.
        
        Args:
            task_type: Type of task to predict priority for
            future_context: Future context to predict in
            
        Returns:
            Predicted priority between 0 and 1
        """
        # Create prediction key
        future_key = self._extract_context_key(future_context)
        prediction_key = f"{task_type}_{future_key}"
        
        # Check if we already have a prediction for this
        if prediction_key in self.task_predictions:
            return self.task_predictions[prediction_key]
            
        # Extract state features for prediction
        future_state = self._extract_state_features(task_type, future_context)
        
        # Get current precision for this task type
        precision = self.priority_precision.get(task_type, 1.0)
        
        if self.use_neural_model and self.neural_model is not None:
            # Neural prediction
            state_tensor = torch.tensor(list(future_state.values()), 
                                      dtype=torch.float32).to(self.neural_model[0].weight.device)
            
            # Forward pass with uncertainty
            with torch.no_grad():
                q_values = self.neural_model(state_tensor)
                
                # Apply softmax to get probability distribution
                probs = F.softmax(q_values * precision, dim=0)
                
                # Weight priority values by probabilities
                priority_values = torch.tensor([0.2, 0.5, 0.9], 
                                             device=probs.device)
                predicted_priority = torch.sum(probs * priority_values)
                
                result = predicted_priority.item()
        else:
            # Table-based prediction
            state_key = self._discretize_state(future_state)
            
            if state_key in self.q_values:
                q_values = self.q_values[state_key]
                
                # Calculate expected priority across all actions
                weighted_sum = 0.0
                total_weight = 0.0
                
                priority_map = {
                    'high': 0.9,
                    'medium': 0.5,
                    'low': 0.2
                }
                
                for action, q_value in q_values.items():
                    # Convert to probability using softmax
                    weight = np.exp(q_value * precision)
                    weighted_sum += priority_map[action] * weight
                    total_weight += weight
                    
                if total_weight > 0:
                    result = weighted_sum / total_weight
                else:
                    result = 0.5  # Default
            else:
                # No data for this state
                result = 0.5  # Default
                
        # Store prediction
        self.task_predictions[prediction_key] = result
        
        return result
        
    async def update_from_feedback(self, task_type, metrics):
        """
        Update priority model based on performance feedback.
        
        Args:
            task_type: Type of task that was executed
            metrics: Performance metrics from task execution
        """
        if task_type not in self.state_features:
            return
            
        # Calculate reward based on metrics
        reward = self._calculate_reward(metrics)
        
        # Extract relevant states
        state = self.state_features[task_type]
        
        if self.use_neural_model and self.neural_model is not None:
            await self._update_neural_model(state, 
                                          metrics.get('priority', 0.5), 
                                          reward)
            return
            
        # Traditional Q-learning update
        state_key = self._discretize_state(state)
        
        # Determine which action was taken
        priority = metrics.get('priority', 0.5)
        action = 'medium'
        
        if priority >= 0.7:
            action = 'high'
        elif priority <= 0.3:
            action = 'low'
            
        # Simple Q-learning update
        if state_key in self.q_values and action in self.q_values[state_key]:
            old_value = self.q_values[state_key][action]
            self.q_values[state_key][action] = old_value + self.learning_rate * reward
            
        # Add to experience buffer for replay
        experience = (state_key, action, reward)
        self.experience_buffer.append(experience)
        
        # Trim buffer if needed
        if len(self.experience_buffer) > self.buffer_size:
            self.experience_buffer = self.experience_buffer[-self.buffer_size:]
            
        # Periodically replay experiences
        if len(self.experience_buffer) >= self.batch_size:
            await self._replay_experiences()
            
        # Evaluate priority prediction if we made one
        context_key = self._extract_context_key(metrics.get('context', {}))
        prediction_key = f"{task_type}_{context_key}"
        
        if prediction_key in self.task_predictions:
            predicted = self.task_predictions[prediction_key]
            actual = priority
            
            # Calculate prediction error
            error = abs(predicted - actual)
            
            # Track error for precision calculation
            self.priority_errors[task_type].append(error)
            
            # Limit history size
            if len(self.priority_errors[task_type]) > 20:
                self.priority_errors[task_type] = self.priority_errors[task_type][-20:]
                
            # Update precision (inverse variance)
            if len(self.priority_errors[task_type]) > 1:
                variance = np.var(self.priority_errors[task_type])
                precision = 1.0 / (variance + 1e-6)  # Avoid division by zero
                self.priority_precision[task_type] = precision
            
    async def _update_neural_model(self, state, priority, reward):
        """Update neural model using experience."""
        # Convert state to tensor
        state_tensor = torch.tensor(list(state.values()), 
                                   dtype=torch.float32).to(self.neural_model[0].weight.device)
        
        # Convert priority to action index
        action_idx = 1  # medium priority
        
        if priority >= 0.7:
            action_idx = 2  # high priority
        elif priority <= 0.3:
            action_idx = 0  # low priority
            
        # Create target Q-values
        with torch.no_grad():
            current_q = self.neural_model(state_tensor)
            target_q = current_q.clone()
            target_q[action_idx] = reward
            
        # Update model
        self.optimizer.zero_grad()
        predicted_q = self.neural_model(state_tensor)
        loss = F.mse_loss(predicted_q, target_q)
        loss.backward()
        self.optimizer.step()
        
        # Update learning rate scheduler
        self.scheduler.step(loss)
        
    async def _replay_experiences(self):
        """Replay random batch of experiences to improve learning."""
        if len(self.experience_buffer) < self.batch_size:
            return
            
        # Sample random batch
        batch_indices = np.random.choice(
            len(self.experience_buffer),
            self.batch_size,
            replace=False
        )
        
        for idx in batch_indices:
            state_key, action, reward = self.experience_buffer[idx]
            
            if state_key in self.q_values and action in self.q_values[state_key]:
                old_value = self.q_values[state_key][action]
                self.q_values[state_key][action] = old_value + self.learning_rate * (
                    reward - old_value
                )
                
    def _extract_state_features(self, task_type, context):
        """Extract relevant features for state representation."""
        features = {
            'task_type_id': hash(task_type) % 10,  # Hash task type to a small integer
            'queue_length': min(1.0, context.get('queue_length', 0) / 100),
            'system_load': context.get('system_load', 0.5),
            'time_of_day': datetime.datetime.now().hour / 24.0,
            'is_critical': 1.0 if context.get('is_critical', False) else 0.0,
            'expected_duration': min(1.0, context.get('expected_duration', 0.5) / 10.0),
            'resource_availability': context.get('resource_availability', 0.8),
            'priority_trend': context.get('priority_trend', 0.0)  # -1 to 1 range
        }
        
        return features
        
    def _extract_context_key(self, context):
        """Generate a string key for a context dict."""
        # Sort keys for consistent hashing
        return "_".join(f"{k}:{v}" for k, v in sorted(context.items()))
        
    def _discretize_state(self, state):
        """Convert continuous state to discrete representation for lookup."""
        # Create a tuple of discretized features
        discrete_state = (
            state['task_type_id'],
            min(5, int(state['queue_length'] * 5)),  # Bucket queue length
            min(5, int(state['system_load'] * 5)),  # Bucket system load
            min(3, int(state['time_of_day'] * 4)),  # 4 time of day buckets
            1 if state['is_critical'] > 0.5 else 0,
            min(3, int(state['expected_duration'] * 3))  # 3 duration buckets
        )
        
        return str(discrete_state)  # Convert to string for dictionary key
        
    def _calculate_reward(self, metrics):
        """Calculate reward based on performance metrics."""
        # Extract relevant metrics
        processing_time = metrics.get('processing_time', 1.0)
        success = metrics.get('success', True)
        quality = metrics.get('quality', 0.5)
        resource_efficiency = metrics.get('resource_efficiency', 0.5)
        
        # Calculate reward components
        time_reward = max(0, 1.0 - min(1.0, processing_time / 10.0))
        success_reward = 1.0 if success else -0.5
        quality_reward = quality - 0.5  # -0.5 to 0.5
        efficiency_reward = resource_efficiency - 0.5  # -0.5 to 0.5
        
        # Combine rewards
        reward = (
            time_reward * 0.3 +
            success_reward * 0.4 +
            quality_reward * 0.2 +
            efficiency_reward * 0.1
        )
        
        return reward



# === FILE: asf\layer3_cognitive_boundary\resolution\__init__.py ===

# Layer 3: Semantic Organization Layer
# Resolution module for detecting and resolving contradictions

from asf.semantic_organization.resolution.conflict_detection import ConflictDetectionEngine

__all__ = ['ConflictDetectionEngine']



# === FILE: asf\layer3_cognitive_boundary\resolution\conflict_detection.py ===

# Enhanced ConflictDetectionEngine with predictive capabilities

import time
import numpy as np
import logging
from typing import Dict, List, Any, Optional
from collections import defaultdict

class ConflictDetectionEngine:
    """
    Detects and resolves semantic contradictions and inconsistencies.
    Now enhanced with ability to anticipate contradictions before they materialize.
    Implements Seth's principle of minimizing prediction error through active inference.
    """
    def __init__(self, semantic_network):
        self.semantic_network = semantic_network
        self.contradiction_history = []
        self.logger = logging.getLogger("ASF.Layer3.ConflictDetection")
        
        # Define contradiction types
        self.contradiction_types = {
            "property_value": self._check_property_value_contradiction,
            "relational": self._check_relational_contradiction,
            "inheritance": self._check_inheritance_contradiction,
            "temporal": self._check_temporal_contradiction
        }
        
        # Seth's Data Paradox enhancements
        self.anticipated_contradictions = {}  # Operation hash -> anticipated contradictions
        self.anticipation_errors = defaultdict(list)  # Type -> prediction errors
        self.precision_values = {}  # Contradiction type -> precision
        
    async def anticipate_contradictions(self, semantic_operations):
        """
        Anticipate contradictions that might arise from planned semantic operations.
        Implements Seth's predictive processing principle.
        
        Args:
            semantic_operations: List of planned semantic operations (node additions, property changes, etc.)
            
        Returns:
            Dict of anticipated contradictions
        """
        anticipated = []
        operations_hash = self._hash_operations(semantic_operations)
        
        # Check if we've already analyzed the same set of operations
        if operations_hash in self.anticipated_contradictions:
            return self.anticipated_contradictions[operations_hash]
        
        # Simulate operations to anticipate contradictions
        simulated_state = await self._simulate_operations(semantic_operations)
        
        if not simulated_state:
            return {"status": "error", "message": "Failed to simulate operations"}
        
        # Perform contradiction checks on simulated state
        for check_type, check_func in self.contradiction_types.items():
            # Calculate precision for this contradiction type
            type_precision = self.precision_values.get(check_type, 1.0)
            
            # Skip low-precision checks to save computation
            if type_precision < 0.5:
                continue
                
            # Run the checker function on simulated state
            contradictions = await check_func(simulated_state)
            
            # Associate each contradiction with precision
            for contradiction in contradictions:
                contradiction['anticipated'] = True
                contradiction['precision'] = type_precision
                anticipated.append(contradiction)
        
        # Store anticipations for future evaluation
        self.anticipated_contradictions[operations_hash] = {
            'contradictions': anticipated,
            'timestamp': time.time()
        }
        
        return {
            'status': 'success',
            'anticipated_contradictions': anticipated,
            'operation_count': len(semantic_operations)
        }
    
    async def preemptively_resolve(self, anticipated_contradictions):
        """
        Preemptively resolve anticipated contradictions before they materialize.
        Implements Seth's active inference principle.
        
        Args:
            anticipated_contradictions: Dict of anticipated contradictions
            
        Returns:
            Dict of resolution actions
        """
        if not anticipated_contradictions or 'anticipated_contradictions' not in anticipated_contradictions:
            return {'status': 'error', 'message': 'No valid contradictions provided'}
            
        contradictions = anticipated_contradictions['anticipated_contradictions']
        
        # Sort by confidence and precision
        sorted_contradictions = sorted(
            contradictions, 
            key=lambda x: x.get('confidence', 0.5) * x.get('precision', 1.0),
            reverse=True
        )
        
        resolutions = []
        
        for contradiction in sorted_contradictions:
            # Determine resolution strategy based on confidence and type
            strategy = self._select_preemptive_strategy(contradiction)
            
            # Apply resolution strategy
            resolution = await self._apply_preemptive_resolution(contradiction, strategy)
            
            if resolution:
                resolutions.append({
                    'contradiction': contradiction,
                    'strategy': strategy,
                    'actions': resolution
                })
        
        return {
            'status': 'success',
            'resolutions': resolutions,
            'contradiction_count': len(contradictions),
            'resolution_count': len(resolutions)
        }
        
    async def check_contradictions(self, nodes=None, check_types=None):
        """
        Check for contradictions among nodes with evaluation of anticipations.
        
        Args:
            nodes: List of node IDs to check (or None for all nodes)
            check_types: Types of contradictions to check (or None for all)
            
        Returns:
            List of detected contradictions
        """
        # Get nodes to check
        if nodes:
            nodes_to_check = {}
            for node_id in nodes:
                node = await self.semantic_network.get_node(node_id)
                if node:
                    nodes_to_check[node_id] = node
        else:
            nodes_to_check = dict(self.semantic_network.nodes)
            
        # Determine contradiction types to check
        if check_types:
            contradiction_checkers = {t: self.contradiction_types[t] for t in check_types 
                                     if t in self.contradiction_types}
        else:
            contradiction_checkers = self.contradiction_types
            
        # Check for contradictions
        contradictions = []
        
        # Keep track of actual contradictions by type for evaluation
        actual_by_type = defaultdict(list)
        
        for check_name, check_func in contradiction_checkers.items():
            detected = await check_func(nodes_to_check)
            contradictions.extend(detected)
            
            # Record by type for anticipation evaluation
            for item in detected:
                actual_by_type[check_name].append(item)
            
        # Now, evaluate anticipations against actual contradictions
        for operation_hash, anticipation in list(self.anticipated_contradictions.items()):
            # Skip recent anticipations
            if time.time() - anticipation['timestamp'] < 10:  # 10 seconds threshold
                continue
                
            # Get all anticipated by type
            anticipated_by_type = defaultdict(list)
            
            for item in anticipation.get('contradictions', []):
                item_type = item.get('type', 'unknown')
                anticipated_by_type[item_type].append(item)
            
            # Compare anticipations to actual results
            for contradiction_type in set(list(anticipated_by_type.keys()) + list(actual_by_type.keys())):
                # Calculate error
                if contradiction_type in anticipated_by_type and contradiction_type in actual_by_type:
                    # Both anticipated and actual contradictions
                    error = abs(len(anticipated_by_type[contradiction_type]) - len(actual_by_type[contradiction_type]))
                    error = error / max(1, len(actual_by_type[contradiction_type]))  # Normalize
                elif contradiction_type in anticipated_by_type:
                    # False positive - anticipated but not found
                    error = 1.0
                else:
                    # False negative - not anticipated but found
                    error = 1.0
                
                # Record error for this type
                self.anticipation_errors[contradiction_type].append(error)
                
                # Limit history size
                if len(self.anticipation_errors[contradiction_type]) > 20:
                    self.anticipation_errors[contradiction_type] = self.anticipation_errors[contradiction_type][-20:]
                
                # Update precision (inverse variance)
                if len(self.anticipation_errors[contradiction_type]) > 1:
                    variance = np.var(self.anticipation_errors[contradiction_type])
                    precision = 1.0 / (variance + 1e-6)  # Avoid division by zero
                    self.precision_values[contradiction_type] = min(10.0, precision)  # Cap precision
        
        # Clean up old anticipations
        current_time = time.time()
        self.anticipated_contradictions = {
            op_hash: anticip for op_hash, anticip in self.anticipated_contradictions.items()
            if current_time - anticip['timestamp'] < 3600  # Keep for an hour
        }
        
        # Record in history
        if contradictions:
            self.contradiction_history.append({
                'timestamp': time.time(),
                'node_count': len(nodes_to_check),
                'contradiction_count': len(contradictions),
                'check_types': list(contradiction_checkers.keys())
            })
            
        return contradictions
    
    async def _simulate_operations(self, operations):
        """
        Simulate semantic operations to anticipate their effects.
        Creates a temporary copy of affected state.
        
        Args:
            operations: List of semantic operations
            
        Returns:
            Dict of simulated state (nodes)
        """
        # Create a shallow copy of the relevant nodes
        affected_nodes = {}
        
        # Identify affected nodes
        for operation in operations:
            op_type = operation.get('type')
            node_id = operation.get('node_id')
            
            if not node_id:
                continue
                
            # Get the node if not already in our simulation
            if node_id not in affected_nodes:
                node = await self.semantic_network.get_node(node_id)
                if node:
                    # Create a shallow copy
                    affected_nodes[node_id] = self._copy_node(node)
        
        # Simulate operations on the copied nodes
        for operation in operations:
            op_type = operation.get('type')
            node_id = operation.get('node_id')
            
            if not node_id or node_id not in affected_nodes:
                continue
                
            node = affected_nodes[node_id]
            
            # Apply operation based on type
            if op_type == 'add_property':
                prop_name = operation.get('property_name')
                prop_value = operation.get('property_value')
                
                if prop_name:
                    node.properties[prop_name] = prop_value
                    
            elif op_type == 'update_property':
                prop_name = operation.get('property_name')
                prop_value = operation.get('property_value')
                
                if prop_name:
                    node.properties[prop_name] = prop_value
                    
            elif op_type == 'remove_property':
                prop_name = operation.get('property_name')
                
                if prop_name and prop_name in node.properties:
                    del node.properties[prop_name]
        
        return affected_nodes
    
    def _copy_node(self, node):
        """Create a shallow copy of a node for simulation."""
        # This is a simplified version - in production would need proper deep copying
        copy = type(node)(
            id=node.id,
            label=node.label,
            node_type=node.node_type,
            properties=dict(node.properties),  # Copy properties
            confidence=node.confidence
        )
        return copy
    
    def _select_preemptive_strategy(self, contradiction):
        """Select appropriate preemptive resolution strategy."""
        contradiction_type = contradiction.get('type')
        confidence = contradiction.get('confidence', 0.5)
        precision = contradiction.get('precision', 1.0)
        
        if contradiction_type == 'property_value':
            if confidence * precision > 0.8:
                return 'prevent_property_change'
            else:
                return 'flag_property_potential_conflict'
                
        elif contradiction_type == 'relational':
            return 'prevent_relation_formation'
            
        elif contradiction_type == 'inheritance':
            return 'recommend_intermediate_concept'
            
        elif contradiction_type == 'temporal':
            return 'adjust_temporal_ordering'
            
        return 'flag_potential_conflict'
    
    async def _apply_preemptive_resolution(self, contradiction, strategy):
        """Apply preemptive resolution strategy."""
        if strategy == 'prevent_property_change':
            return {
                'action': 'prevent',
                'message': f"Prevented property change that would cause contradiction: {contradiction.get('property', 'unknown')}",
                'property': contradiction.get('property')
            }
            
        elif strategy == 'flag_property_potential_conflict':
            return {
                'action': 'flag',
                'message': f"Flagged potential property conflict: {contradiction.get('property', 'unknown')}",
                'property': contradiction.get('property')
            }
            
        elif strategy == 'prevent_relation_formation':
            return {
                'action': 'prevent',
                'message': "Prevented formation of contradictory relation",
                'source_id': contradiction.get('source_id'),
                'target_id': contradiction.get('target_id')
            }
            
        elif strategy == 'recommend_intermediate_concept':
            return {
                'action': 'recommend',
                'message': "Recommended intermediate concept to resolve inheritance contradiction",
                'involved_ids': [contradiction.get('parent_id'), contradiction.get('child_id')]
            }
            
        elif strategy == 'adjust_temporal_ordering':
            return {
                'action': 'adjust',
                'message': "Suggested temporal adjustment to resolve contradiction",
                'node1_id': contradiction.get('node1_id'),
                'node2_id': contradiction.get('node2_id')
            }
            
        else:  # flag_potential_conflict
            return {
                'action': 'flag',
                'message': "Flagged potential contradiction",
                'contradiction_type': contradiction.get('type')
            }
    
    def _hash_operations(self, operations):
        """Create a stable hash for a set of operations."""
        # Simple implementation - in production would need better hashing
        operation_strings = []
        
        for op in operations:
            op_str = f"{op.get('type')}:{op.get('node_id')}:{op.get('property_name', '')}"
            operation_strings.append(op_str)
            
        # Sort for stability
        operation_strings.sort()
        
        return hash(tuple(operation_strings))



# === FILE: asf\layer3_cognitive_boundary\utils\__init__.py ===

# Layer 3: Semantic Organization Layer
# Utility functions for tensor operations and other helpers

from asf.semantic_organization.utils.tensor_utils import (
    normalize_embeddings, cosine_similarity, batch_cosine_similarity,
    tensor_max_pool, project_to_hyperplane, soft_attention_weighted_sum
)

__all__ = [
    'normalize_embeddings', 'cosine_similarity', 'batch_cosine_similarity',
    'tensor_max_pool', 'project_to_hyperplane', 'soft_attention_weighted_sum'
]



# === FILE: asf\layer3_cognitive_boundary\utils\tensor_utils.py ===

import torch
import torch.nn.functional as F
import numpy as np

def normalize_embeddings(embeddings, dim=1, eps=1e-8):
    """
    Normalize embeddings to unit length along specified dimension.
    
    Args:
        embeddings: Numpy array or torch tensor of embeddings
        dim: Dimension along which to normalize
        eps: Small value to prevent division by zero
        
    Returns:
        Normalized embeddings
    """
    if isinstance(embeddings, np.ndarray):
        norms = np.linalg.norm(embeddings, axis=dim, keepdims=True)
        norms[norms < eps] = 1.0  # Avoid division by zero
        return embeddings / norms
        
    elif isinstance(embeddings, torch.Tensor):
        return F.normalize(embeddings, p=2, dim=dim, eps=eps)
        
    else:
        raise TypeError("Embeddings must be numpy array or torch tensor")

def cosine_similarity(a, b):
    """
    Calculate cosine similarity between vectors.
    
    Args:
        a: First vector (numpy array or torch tensor)
        b: Second vector (numpy array or torch tensor)
        
    Returns:
        Cosine similarity
    """
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        
        if a_norm == 0 or b_norm == 0:
            return 0.0
            
        return np.dot(a, b) / (a_norm * b_norm)
        
    elif isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        a_normalized = F.normalize(a, p=2, dim=0)
        b_normalized = F.normalize(b, p=2, dim=0)
        
        return torch.dot(a_normalized, b_normalized).item()
        
    else:
        raise TypeError("Vectors must be either both numpy arrays or both torch tensors")

def batch_cosine_similarity(query, database):
    """
    Calculate cosine similarity between a query and all vectors in a database.
    
    Args:
        query: Query vector [D] or batch [B, D]
        database: Database of vectors [N, D]
        
    Returns:
        Similarities [N] or [B, N]
    """
    if isinstance(query, np.ndarray) and isinstance(database, np.ndarray):
        # Normalize
        query_norm = normalize_embeddings(query, dim=0 if query.ndim == 1 else 1)
        db_norm = normalize_embeddings(database, dim=1)
        
        if query.ndim == 1:
            # Single query
            return np.dot(db_norm, query_norm)
        else:
            # Batch of queries
            return np.dot(query_norm, db_norm.T)
            
    elif isinstance(query, torch.Tensor) and isinstance(database, torch.Tensor):
        # Normalize
        query_norm = F.normalize(query, p=2, dim=0 if query.dim() == 1 else 1)
        db_norm = F.normalize(database, p=2, dim=1)
        
        if query.dim() == 1:
            # Single query
            return torch.matmul(db_norm, query_norm)
        else:
            # Batch of queries
            return torch.matmul(query_norm, db_norm.t())
            
    else:
        raise TypeError("Vectors must be either both numpy arrays or both torch tensors")

def tensor_max_pool(tensor_list, dim=0):
    """
    Apply max pooling across a list of tensors.
    Implements Seth's principle of MGC-based maximization.
    
    Args:
        tensor_list: List of tensors with same shape
        dim: Dimension along which to stack tensors
        
    Returns:
        Tensor with maximum values
    """
    if not tensor_list:
        return None
        
    if isinstance(tensor_list[0], np.ndarray):
        stacked = np.stack(tensor_list, axis=dim)
        return np.max(stacked, axis=dim)
        
    elif isinstance(tensor_list[0], torch.Tensor):
        stacked = torch.stack(tensor_list, dim=dim)
        return torch.max(stacked, dim=dim)[0]
        
    else:
        raise TypeError("Tensors must be either numpy arrays or torch tensors")

def project_to_hyperplane(vectors, normal, bias=0.0):
    """
    Project vectors onto a hyperplane defined by normal and bias.
    Useful for conceptual blending and dimension reduction.
    
    Args:
        vectors: Vectors to project [N, D]
        normal: Normal vector of hyperplane [D]
        bias: Hyperplane bias
        
    Returns:
        Projected vectors [N, D]
    """
    if isinstance(vectors, np.ndarray) and isinstance(normal, np.ndarray):
        # Normalize normal vector
        normal = normal / np.linalg.norm(normal)
        
        # Calculate projection
        distances = np.dot(vectors, normal) - bias
        projections = vectors - np.outer(distances, normal)
        
        return projections
        
    elif isinstance(vectors, torch.Tensor) and isinstance(normal, torch.Tensor):
        # Normalize normal vector
        normal = F.normalize(normal, p=2, dim=0)
        
        # Calculate projection
        distances = torch.matmul(vectors, normal) - bias
        projections = vectors - torch.outer(distances, normal)
        
        return projections
        
    else:
        raise TypeError("Vectors must be either both numpy arrays or both torch tensors")

def soft_attention_weighted_sum(query, keys, values, temperature=1.0):
    """
    Calculate attention-weighted sum of values.
    Important for implementing selective attention in Seth's framework.
    
    Args:
        query: Query vector [D]
        keys: Key vectors [N, D]
        values: Value vectors [N, V]
        temperature: Controls softmax temperature
        
    Returns:
        Weighted sum of values [V]
    """
    if isinstance(query, np.ndarray):
        # Calculate similarities
        similarities = batch_cosine_similarity(query, keys)
        
        # Apply temperature and softmax
        similarities = similarities / temperature
        weights = np.exp(similarities)
        weights = weights / np.sum(weights)
        
        # Apply weights
        weighted_sum = np.sum(values * weights.reshape(-1, 1), axis=0)
        
        return weighted_sum
        
    elif isinstance(query, torch.Tensor):
        # Calculate similarities
        similarities = batch_cosine_similarity(query, keys)
        
        # Apply temperature and softmax
        similarities = similarities / temperature
        weights = F.softmax(similarities, dim=0)
        
        # Apply weights
        weighted_sum = torch.sum(values * weights.unsqueeze(1), dim=0)
        
        return weighted_sum
        
    else:
        raise TypeError("Query must be numpy array or torch tensor")



# === FILE: asf\layer4_environmental_coupling\__init__.py ===

from asf.environmental_coupling.layer import EnvironmentalCouplingLayer
from asf.environmental_coupling.models import (
    EnvironmentalCoupling,
    CouplingEvent,
    EnvironmentalPrediction,
    ActiveInferenceTest
)
from asf.environmental_coupling.enums import (
    CouplingType,
    CouplingStrength,
    CouplingState,
    EventPriority,
    PredictionState
)

__all__ = [
    'EnvironmentalCouplingLayer',
    'EnvironmentalCoupling',
    'CouplingEvent',
    'EnvironmentalPrediction',
    'ActiveInferenceTest',
    'CouplingType',
    'CouplingStrength', 
    'CouplingState',
    'EventPriority',
    'PredictionState'
]



# === FILE: asf\layer4_environmental_coupling\enums.py ===

import enum
from enum import Enum, auto

class CouplingType(Enum):
    """Types of environmental coupling relationships."""
    INFORMATIONAL = auto()  # Information exchange coupling
    OPERATIONAL = auto()    # Operational/functional coupling
    CONTEXTUAL = auto()     # Context-providing coupling
    ADAPTIVE = auto()       # Learning/feedback-based coupling
    PREDICTIVE = auto()     # NEW: Seth's predictive coupling type

class CouplingStrength(Enum):
    """Strength classifications for coupling relationships."""
    WEAK = 0.2      # Minimal influence, easily disrupted
    MODERATE = 0.5  # Standard coupling with moderate influence
    STRONG = 0.8    # Strong influence, resistant to disruption
    CRITICAL = 1.0  # System-defining, essential coupling

class CouplingState(Enum):
    """States of coupling relationships."""
    POTENTIAL = auto()     # Possible coupling not yet established
    FORMING = auto()       # Coupling in formation process
    ACTIVE = auto()        # Active coupling relationship
    DEGRADING = auto()     # Coupling beginning to decay
    DORMANT = auto()       # Inactive but restorable
    TERMINATED = auto()    # Permanently terminated
    ANTICIPATORY = auto()  # NEW: Seth's anticipatory state

class EventPriority(Enum):
    """Priority levels for event processing."""
    LOW = 0.2
    MEDIUM = 0.5
    HIGH = 0.8
    CRITICAL = 1.0

class PredictionState(Enum):
    """NEW: States for predictive processing."""
    ANTICIPATING = auto()   # Generating predictions
    PERCEIVING = auto()     # Receiving actual data
    COMPARING = auto()      # Comparing prediction to reality
    UPDATING = auto()       # Updating model based on prediction error
    TESTING = auto()        # Actively testing a prediction



# === FILE: asf\layer4_environmental_coupling\environmental_coupling_layer.py ===

import asyncio
import time
import uuid
import logging
import numpy as np
import random
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

from asf.environmental_coupling.enums import (
    CouplingType, CouplingStrength, CouplingState, EventPriority, PredictionState
)
from asf.environmental_coupling.models import (
    CouplingEvent, EnvironmentalCoupling, EnvironmentalPrediction, ActiveInferenceTest
)
from asf.environmental_coupling.components.coupling_registry import SparseCouplingRegistry
from asf.environmental_coupling.components.event_processor import EventDrivenProcessor, AsyncEventQueue
from asf.environmental_coupling.components.enhanced_bayesian_updater import EnhancedBayesianUpdater
from asf.environmental_coupling.components.rl_optimizer import ReinforcementLearningOptimizer
from asf.environmental_coupling.components.coherence_boundary import CoherenceBoundaryController
from asf.environmental_coupling.components.gpu_accelerator import GPUAccelerationManager
from asf.environmental_coupling.components.context_tracker import AdaptiveContextTracker
from asf.environmental_coupling.components.distributed_cache import DistributedCouplingCache
from asf.environmental_coupling.components.metrics_collector import PerformanceMetricsCollector

# Seth's Data Paradox components
from asf.environmental_coupling.components.predictive_modeler import PredictiveEnvironmentalModeler
from asf.environmental_coupling.components.active_inference_controller import ActiveInferenceController
from asf.environmental_coupling.components.counterfactual_simulator import CounterfactualSimulator

class EnvironmentalCouplingLayer:
    """
    Enhanced Layer 4 with complete integration of Seth's predictive processing principles.
    Orchestrates controlled hallucination, precision-weighted prediction errors, active inference,
    and counterfactual simulation as a unified predictive system.
    """
    def __init__(self, knowledge_substrate, async_queue=None, config=None):
        # Original components with enhancements
        self.knowledge_substrate = knowledge_substrate
        self.async_queue = async_queue or AsyncEventQueue()
        self.config = config or {}
        
        # Core components with enhanced versions
        self.coupling_registry = SparseCouplingRegistry(
            initial_capacity=self.config.get('initial_capacity', 10000)
        )
        self.event_processor = EventDrivenProcessor(
            max_concurrency=self.config.get('max_concurrency', 16)
        )
        self.bayesian_updater = EnhancedBayesianUpdater()  # Seth-enhanced version
        self.rl_optimizer = ReinforcementLearningOptimizer(
            learning_rate=self.config.get('learning_rate', 0.01)
        )
        
        # Seth's Data Paradox components
        self.predictive_modeler = PredictiveEnvironmentalModeler()
        self.active_inference = ActiveInferenceController(knowledge_substrate)
        self.counterfactual_simulator = CounterfactualSimulator()
        
        # Original components
        self.coherence_boundary = CoherenceBoundaryController(knowledge_substrate)
        self.gpu_accelerator = GPUAccelerationManager(
            enabled=self.config.get('use_gpu', True)
        )
        self.context_tracker = AdaptiveContextTracker()
        self.distributed_cache = DistributedCouplingCache(
            self.config.get('cache_config', {})
        )
        
        # Seth's principle configuration
        self.prediction_enabled = self.config.get('prediction_enabled', True)
        self.active_inference_enabled = self.config.get('active_inference_enabled', True)
        self.counterfactual_enabled = self.config.get('counterfactual_enabled', True)
        
        # Adjacent layers
        self.layer5 = None
        self.layer6 = None
        
        # Performance metrics
        self.metrics_collector = PerformanceMetricsCollector()
        self.logger = logging.getLogger("ASF.Layer4.EnvironmentalCouplingLayer")
        
    async def initialize(self, layer5=None, layer6=None):
        """Initialize the layer and connect to adjacent layers."""
        self.layer5 = layer5  # AutopoieticMaintenanceLayer
        self.layer6 = layer6  # EnvironmentalDistributionLayer
        
        # Initialize components
        await self.coupling_registry.initialize()
        await self.event_processor.initialize(self.process_coupling_event)
        await self.rl_optimizer.initialize(self.knowledge_substrate)
        await self.gpu_accelerator.initialize()
        
        # Set up component connections
        self.active_inference.set_coupling_registry(self.coupling_registry)
        
        # Start background task for async event processing
        asyncio.create_task(self.event_processor.run_processing_loop())
        
        self.logger.info(f"Layer 4 (Environmental Coupling) initialized with Seth's Data Paradox principles")
        self.logger.info(f"Prediction enabled: {self.prediction_enabled}")
        self.logger.info(f"Active inference enabled: {self.active_inference_enabled}")
        self.logger.info(f"Counterfactual simulation enabled: {self.counterfactual_enabled}")
        
        return {'status': 'initialized'}
        
    async def integrate_prediction_cycle(self, entity_id, context=None):
        """
        Execute a complete predictive processing cycle for an entity.
        This implements the full predictive cycle from Seth's framework:
        1. Generate predictions (controlled hallucination)
        2. Collect sensory evidence
        3. Calculate precision-weighted prediction errors
        4. Update internal models
        5. Perform active inference as needed
        6. Explore counterfactual configurations
        """
        start_time = time.time()
        cycle_results = {}
        
        # 1. Generate predictions using controlled hallucination
        predictions = await self._generate_entity_predictions(entity_id, context)
        cycle_results['predictions'] = {
            'count': len(predictions),
            'prediction_ids': [p.id for p in predictions]
        }
        
        # 2 & 3. Collect evidence and evaluate predictions
        current_state = await self._collect_environmental_state(entity_id)
        evaluations = await self._evaluate_predictions(predictions, current_state)
        cycle_results['evaluations'] = {
            'count': len(evaluations),
            'avg_error': np.mean([e['error'] for e in evaluations]) if evaluations else None,
            'avg_precision': np.mean([e['precision'] for e in evaluations]) if evaluations else None
        }
        
        # 4. Update internal models based on precision-weighted errors
        model_updates = await self._update_internal_models(entity_id, evaluations)
        cycle_results['model_updates'] = model_updates
        
        # 5. Perform active inference if needed
        if self._should_perform_active_inference(evaluations):
            inference_results = await self.perform_active_inference(entity_id)
            cycle_results['active_inference'] = inference_results
        
        # 6. Explore counterfactual configurations
        counterfactuals = await self._explore_counterfactuals(entity_id, evaluations)
        cycle_results['counterfactuals'] = counterfactuals
        
        # Generate new predictions for next cycle
        new_predictions = await self._generate_entity_predictions(entity_id, context, current_state)
        cycle_results['new_predictions'] = {
            'count': len(new_predictions),
            'prediction_ids': [p.id for p in new_predictions]
        }
        
        # Calculate cycle metrics
        cycle_results['elapsed_time'] = time.time() - start_time
        cycle_results['success'] = True
        
        self.logger.info(f"Completed predictive cycle for entity {entity_id} in {cycle_results['elapsed_time']:.3f}s")
        return cycle_results
    
    async def _generate_entity_predictions(self, entity_id, context=None, current_state=None):
        """Generate predictions for an entity across all its environmental connections."""
        predictions = []
        
        # Get all couplings for this entity
        couplings = await self.coupling_registry.get_couplings_by_internal_entity(entity_id)
        
        for coupling in couplings:
            # Skip inactive couplings
            if coupling.coupling_state != CouplingState.ACTIVE:
                continue
            
            # Generate prediction for each coupling
            prediction = await self.predictive_modeler.predict_interaction(
                coupling.environmental_entity_id,
                context={
                    'coupling_id': coupling.id,
                    'coupling_type': coupling.coupling_type.name,
                    'bayesian_confidence': coupling.bayesian_confidence,
                    'current_state': current_state,
                    **(context or {})
                }
            )
            
            predictions.append(prediction)
            
            # Associate prediction with coupling
            if hasattr(coupling, 'expected_interactions'):
                coupling.expected_interactions[prediction.id] = time.time()
                # Cleanup old predictions
                if len(coupling.expected_interactions) > 10:
                    oldest = min(coupling.expected_interactions.items(), key=lambda x: x[1])[0]
                    del coupling.expected_interactions[oldest]
                    
                # Update coupling in registry
                await self.coupling_registry.update_coupling(coupling)
        
        return predictions
    
    async def _collect_environmental_state(self, entity_id):
        """Collect current state information about environmental connections."""
        # This would normally query environmental entities for current state
        # For this implementation, we'll create a simplified representation
        
        state = {
            'timestamp': time.time(),
            'entity_id': entity_id,
            'active_couplings': 0,
            'environments': {}
        }
        
        # Get all couplings for this entity
        couplings = await self.coupling_registry.get_couplings_by_internal_entity(entity_id)
        
        for coupling in couplings:
            if coupling.coupling_state == CouplingState.ACTIVE:
                state['active_couplings'] += 1
                
                # Collect state for each environmental entity
                env_id = coupling.environmental_entity_id
                if env_id not in state['environments']:
                    state['environments'][env_id] = {
                        'last_interaction': coupling.last_interaction,
                        'coupling_strength': coupling.coupling_strength,
                        'bayesian_confidence': coupling.bayesian_confidence,
                        'coupling_type': coupling.coupling_type.name
                    }
                    
                    # Add precision information if available
                    if hasattr(coupling, 'prediction_precision'):
                        state['environments'][env_id]['prediction_precision'] = coupling.prediction_precision
        
        return state
    
    async def _evaluate_predictions(self, predictions, current_state):
        """
        Evaluate predictions against current state.
        This is a simplified implementation since we don't have actual environmental data.
        """
        evaluations = []
        
        for prediction in predictions:
            env_id = prediction.environmental_entity_id
            
            # If we have state data for this environment, evaluate against it
            if env_id in current_state.get('environments', {}):
                env_state = current_state['environments'][env_id]
                
                # Create mock actual data based on state (simplified)
                actual_data = {
                    'interaction_type': prediction.predicted_data.get('predicted_interaction_type', 'unknown'),
                    'timestamp': time.time(),
                    'content_type': 'state_update',
                    'response_time': time.time() - env_state.get('last_interaction', time.time() - 60)
                }
                
                # Evaluate prediction
                evaluation = await self.predictive_modeler.evaluate_prediction(
                    prediction.id, actual_data
                )
                
                if evaluation:
                    evaluations.append(evaluation)
                    self.logger.debug(f"Evaluated prediction {prediction.id} with error {evaluation['error']:.3f}")
        
        return evaluations
    
    async def _update_internal_models(self, entity_id, evaluations):
        """Update internal models based on prediction evaluations."""
        update_results = {
            'precision_updates': 0,
            'model_updates': 0,
            'confidence_changes': []
        }
        
        # Get all couplings for this entity
        couplings = await self.coupling_registry.get_couplings_by_internal_entity(entity_id)
        couplings_by_env = {c.environmental_entity_id: c for c in couplings}
        
        # Process each evaluation
        for evaluation in evaluations:
            prediction_id = evaluation.get('prediction_id')
            if prediction_id and prediction_id in self.predictive_modeler.predictions:
                prediction = self.predictive_modeler.predictions[prediction_id]
                env_id = prediction.environmental_entity_id
                
                # If we have a coupling for this environment, update it
                if env_id in couplings_by_env:
                    coupling = couplings_by_env[env_id]
                    
                    # Update precision for the coupling
                    precision = evaluation.get('precision', 1.0)
                    old_precision = getattr(coupling, 'prediction_precision', 1.0)
                    coupling.prediction_precision = precision
                    update_results['precision_updates'] += 1
                    
                    # Update Bayesian model with precision-weighted confidence
                    # Get base confidence (could be from prediction or default)
                    confidence = prediction.confidence if hasattr(prediction, 'confidence') else 0.5
                    
                    # Update Bayesian model with precision weighting
                    bayesian_result = await self.bayesian_updater.update_precision(
                        coupling.id, evaluation.get('error', 0.5)
                    )
                    
                    # Update confidence tracking
                    update_results['confidence_changes'].append({
                        'coupling_id': coupling.id,
                        'old_precision': old_precision,
                        'new_precision': precision,
                        'error': evaluation.get('error', 0.5)
                    })
                    
                    # Update coupling in registry
                    await self.coupling_registry.update_coupling(coupling)
                    update_results['model_updates'] += 1
        
        return update_results
    
    def _should_perform_active_inference(self, evaluations):
        """Determine if active inference should be performed based on evaluations."""
        if not evaluations:
            return False
            
        # Perform active inference if:
        # 1. Average error is high (above 0.5)
        # 2. OR precision is low (below 1.0)
        avg_error = np.mean([e.get('error', 0) for e in evaluations])
        avg_precision = np.mean([e.get('precision', 1.0) for e in evaluations])
        
        return avg_error > 0.5 or avg_precision < 1.0
    
    async def _explore_counterfactuals(self, entity_id, evaluations):
        """Explore counterfactual configurations for high-error predictions."""
        results = {
            'simulations_run': 0,
            'improvements_found': 0,
            'optimal_configurations': []
        }
        
        # Focus on evaluations with high error
        high_error_evaluations = [e for e in evaluations if e.get('error', 0) > 0.4]
        
        if not high_error_evaluations:
            return results
            
        # Get couplings for entity
        couplings = await self.coupling_registry.get_couplings_by_internal_entity(entity_id)
        couplings_by_env = {c.environmental_entity_id: c for c in couplings}
        
        # For each high-error prediction, explore counterfactuals
        for evaluation in high_error_evaluations:
            prediction_id = evaluation.get('prediction_id')
            if prediction_id and prediction_id in self.predictive_modeler.predictions:
                prediction = self.predictive_modeler.predictions[prediction_id]
                env_id = prediction.environmental_entity_id
                
                # If we have a coupling for this environment, explore counterfactuals
                if env_id in couplings_by_env:
                    coupling = couplings_by_env[env_id]
                    
                    # Run counterfactual simulation
                    simulation_result = await self.simulate_counterfactual_coupling(coupling.id, 5)
                    results['simulations_run'] += 1
                    
                    if simulation_result.get('success', False):
                        optimal_config = simulation_result.get('optimal_configuration')
                        if optimal_config and optimal_config.get('improvement', 0) > 0.1:
                            # Found a significant improvement
                            results['improvements_found'] += 1
                            results['optimal_configurations'].append({
                                'coupling_id': coupling.id,
                                'improvement': optimal_config.get('improvement', 0),
                                'configuration_type': optimal_config.get('optimal_configuration', {}).get('variation_type')
                            })
        
        return results
    
    async def process_environmental_interaction(self, interaction_data, source_id=None, interaction_type=None, confidence=None, context=None):
        """
        Process an interaction from the environment with predictive enhancements.
        Implements Seth's principles of prediction, precision-weighting, and active inference.
        """
        start_time = time.time()
        
        # Create processing context if not provided
        if context is None:
            context = await self.context_tracker.create_context(
                interaction_data, source_id, interaction_type
            )
        
        # Check if we have predictions for this interaction
        predictions = []
        prediction_evaluations = []
        if self.prediction_enabled and source_id:
            predictions = await self._get_relevant_predictions(source_id, interaction_type)
            
            # Evaluate predictions against actual data
            for prediction in predictions:
                evaluation = await self.predictive_modeler.evaluate_prediction(
                    prediction.id, interaction_data
                )
                if evaluation:
                    prediction_evaluations.append(evaluation)
                    self.logger.debug(f"Prediction evaluation: error={evaluation['error']:.3f}, precision={evaluation['precision']:.3f}")
        
        # Check coherence boundary
        coherent, boundary_result = await self.coherence_boundary.check_interaction_coherence(
            interaction_data, source_id, interaction_type, context
        )
        
        if not coherent:
            return {
                'success': False,
                'error': 'Interaction violates coherence boundary',
                'boundary_result': boundary_result,
                'prediction_evaluations': prediction_evaluations,
                'prediction_count': len(predictions),
                'elapsed_time': time.time() - start_time
            }
        
        # Get relevant couplings
        couplings = []
        if source_id:
            couplings = await self.coupling_registry.get_couplings_by_environmental_entity(source_id)
        
        # Process through couplings with precision-weighted updating
        processing_results = []
        
        for coupling in couplings:
            # Get precision from prediction evaluations
            precision = 1.0
            if prediction_evaluations:
                # Use the most relevant prediction's precision
                precision = prediction_evaluations[0].get('precision', 1.0)
            
            # Prepare integrated event data
            event_data = {
                'interaction_data': interaction_data,
                'interaction_type': interaction_type,
                'confidence': confidence,
                'context': context,
                'prediction_evaluations': prediction_evaluations,
                'precision': precision
            }
            
            # Create event for processing
            event = CouplingEvent(
                id=str(uuid.uuid4()),
                event_type='environmental_interaction',
                coupling_id=coupling.id,
                entity_id=coupling.internal_entity_id,
                environmental_id=coupling.environmental_entity_id,
                data=event_data,
                # Add prediction information
                predicted=bool(predictions),
                prediction_id=predictions[0].id if predictions else None,
                precision=precision
            )
            
            # Submit for processing
            await self.event_processor.submit_event(event)
            
            processing_results.append({
                'coupling_id': coupling.id,
                'event_id': event.id,
                'status': 'submitted',
                'precision_applied': precision
            })
        
        # If no couplings processed, still create an event for system-level handling
        if not processing_results:
            event = CouplingEvent(
                id=str(uuid.uuid4()),
                event_type='unassociated_interaction',
                environmental_id=source_id,
                data={
                    'interaction_data': interaction_data,
                    'interaction_type': interaction_type,
                    'confidence': confidence,
                    'context': context
                }
            )
            await self.event_processor.submit_event(event)
            processing_results.append({
                'event_id': event.id,
                'status': 'submitted_unassociated'
            })
        
        # After processing the interaction, generate new predictions
        if self.prediction_enabled and source_id:
            # Context enriched with evaluation results
            prediction_context = {
                'last_interaction_type': interaction_type,
                'evaluation_results': [
                    {'error': e['error'], 'precision': e['precision']}
                    for e in prediction_evaluations
                ] if prediction_evaluations else []
            }
            
            new_prediction = await self.predictive_modeler.predict_interaction(
                source_id, prediction_context
            )
            
            self.logger.debug(f"Generated new prediction {new_prediction.id} for {source_id}")
        
        # Check if active inference is needed based on prediction errors
        active_inference_triggered = False
        if self.active_inference_enabled and prediction_evaluations:
            avg_error = np.mean([e['error'] for e in prediction_evaluations])
            if avg_error > 0.6:  # High error threshold
                # Get the internal entity for the source
                internal_entity_id = None
                if couplings:
                    internal_entity_id = couplings[0].internal_entity_id
                
                if internal_entity_id:
                    # Schedule active inference in background
                    asyncio.create_task(self.perform_active_inference(internal_entity_id))
                    active_inference_triggered = True
                    self.logger.info(f"Triggered active inference due to high prediction error: {avg_error:.3f}")
        
        # Check if counterfactual simulation is needed
        counterfactual_triggered = False
        if self.counterfactual_enabled and couplings and prediction_evaluations:
            # Find the coupling with highest prediction error
            max_error_eval = max(prediction_evaluations, key=lambda e: e['error'])
            if max_error_eval['error'] > 0.7:  # Very high error threshold
                coupling_to_simulate = couplings[0].id  # Default to first
                
                # Try to find the specific coupling for this high-error prediction
                prediction_id = max_error_eval.get('prediction_id')
                if prediction_id and prediction_id in self.predictive_modeler.predictions:
                    pred = self.predictive_modeler.predictions[prediction_id]
                    env_id = pred.environmental_entity_id
                    
                    # Find the coupling for this environmental entity
                    for c in couplings:
                        if c.environmental_entity_id == env_id:
                            coupling_to_simulate = c.id
                            break
                
                # Schedule counterfactual simulation in background
                asyncio.create_task(self.simulate_counterfactual_coupling(coupling_to_simulate))
                counterfactual_triggered = True
                self.logger.info(f"Triggered counterfactual simulation due to very high prediction error: {max_error_eval['error']:.3f}")
        
        return {
            'success': True,
            'couplings_found': len(couplings),
            'processing_results': processing_results,
            'prediction_evaluations': prediction_evaluations,
            'prediction_count': len(predictions),
            'active_inference_triggered': active_inference_triggered,
            'counterfactual_triggered': counterfactual_triggered,
            'elapsed_time': time.time() - start_time
        }
    
    async def process_coupling_event(self, event):
        """
        Process a coupling event with integrated predictive components.
        """
        start_time = time.time()
        
        # Check if this is an active inference test result
        if hasattr(event, 'data') and event.data.get('active_inference_test_result'):
            test_result = event.data.get('active_inference_test_result')
            test_id = test_result.get('test_id')
            
            if test_id:
                test_processing = await self.process_test_result(test_id, test_result)
                result = {
                    'success': True,
                    'event_type': 'active_inference_test_result',
                    'test_processing': test_processing,
                    'elapsed_time': time.time() - start_time
                }
                return result
        
        if not hasattr(event, 'coupling_id') or not event.coupling_id:
            return {'success': False, 'error': 'No coupling ID in event'}
            
        # Get the coupling
        coupling = await self.coupling_registry.get_coupling(event.coupling_id)
        if not coupling:
            return {'success': False, 'error': 'Coupling not found'}
            
        # Extract event data
        event_data = event.data if hasattr(event, 'data') and event.data else {}
        interaction_data = event_data.get('interaction_data', {})
        interaction_type = event_data.get('interaction_type')
        confidence = event_data.get('confidence', 0.5)
        context = event_data.get('context', {})
        
        # Stage 1: Apply precision-weighted Bayesian updates
        # Get precision from event or prediction evaluations
        precision = event_data.get('precision', 1.0)
        prediction_evaluations = event_data.get('prediction_evaluations', [])
        
        if not precision and prediction_evaluations:
            precision = prediction_evaluations[0].get('precision', 1.0)
        
        # Update Bayesian model with prediction-based precision
        bayesian_result = await self.bayesian_updater.update_from_interaction(
            coupling.id, interaction_data, interaction_type, confidence
        )
        
        # Update coupling based on Bayesian result
        coupling.bayesian_confidence = bayesian_result['new_confidence']
        
        # If strength changed significantly, update coupling strength
        if abs(bayesian_result['strength_delta']) > 0.1:
            old_strength = coupling.coupling_strength
            new_strength = max(0.1, min(1.0, old_strength + bayesian_result['strength_delta'] * 0.2))
            coupling.coupling_strength = new_strength
            
        # Update interaction counter and timestamp
        coupling.interaction_count += 1
        coupling.last_interaction = time.time()
        
        # Stage 2: Update prediction precision from evaluations
        if prediction_evaluations:
            # Store prediction error information for future precision calculations
            coupling.prediction_errors = coupling.prediction_errors or []
            for eval_result in prediction_evaluations:
                if 'error' in eval_result:
                    coupling.prediction_errors.append(eval_result['error'])
                    
            # Limit history size
            if len(coupling.prediction_errors) > 20:
                coupling.prediction_errors = coupling.prediction_errors[-20:]
                
            # Update coupling's prediction precision
            coupling.prediction_precision = precision
        
        # Stage 3: Check for contradictions and handle them
        if bayesian_result['contradiction_detected']:
            contradiction = bayesian_result['contradiction']
            await self._handle_contradiction(coupling, contradiction, context)
        
        # Stage 4: Use Reinforcement Learning to optimize coupling parameters
        rl_result = await self.rl_optimizer.update_from_interaction(
            coupling.id, interaction_data, interaction_type, bayesian_result
        )
        
        # Stage 5: Update coupling in registry
        await self.coupling_registry.update_coupling(coupling)
        
        # Return processing results
        return {
            'success': True,
            'coupling_id': coupling.id,
            'bayesian_update': bayesian_result,
            'reinforcement_learning': rl_result,
            'precision_applied': precision,
            'prediction_count': len(prediction_evaluations),
            'elapsed_time': time.time() - start_time
        }
    
    async def _handle_contradiction(self, coupling, contradiction, context):
        """
        Handle a detected contradiction in coupling.
        Contradictions are opportunities for learning and boundary refinement.
        """
        if not contradiction:
            return
            
        self.logger.info(f"Handling contradiction in coupling {coupling.id}: {contradiction['type']}")
        
        # Notify upper layers about contradiction
        if self.layer5:
            asyncio.create_task(self.layer5.process_contradiction(
                coupling.id,
                coupling.internal_entity_id,
                coupling.environmental_entity_id,
                contradiction,
                context
            ))
        
        # Record contradiction in coupling history
        if not hasattr(coupling, 'history'):
            coupling.history = []
            
        coupling.history.append({
            'timestamp': time.time(),
            'event_type': 'contradiction',
            'contradiction_type': contradiction['type'],
            'prior_confidence': contradiction['prior_confidence'],
            'context': context
        })
        
        # For high-confidence contradictions, consider active inference
        if contradiction.get('prior_confidence', 0) > 0.8 and self.active_inference_enabled:
            asyncio.create_task(self.perform_active_inference(coupling.internal_entity_id))
    
    async def perform_active_inference(self, entity_id):
        """
        Actively test and optimize couplings using controlled interactions.
        Implements Seth's active inference principle.
        """
        if not self.active_inference_enabled:
            return {'success': False, 'error': 'Active inference is disabled'}
        
        start_time = time.time()
        
        # Get all current couplings
        couplings = await self.coupling_registry.get_couplings_by_internal_entity(entity_id)
        if not couplings:
            return {
                'success': False,
                'error': 'No couplings found',
                'elapsed_time': time.time() - start_time
            }
        
        inference_results = []
        
        # For each coupling, generate test interactions to minimize uncertainty
        for coupling in couplings:
            # Only test active couplings
            if not hasattr(coupling, 'coupling_state') or coupling.coupling_state != CouplingState.ACTIVE:
                continue
                
            # Generate optimal test interaction based on current uncertainty
            test = await self.active_inference.generate_test_interaction(
                coupling.id,
                uncertainty_focus=True
            )
            
            if test:
                # Execute test interaction to gather information
                if self.layer6:  # EnvironmentalDistributionLayer
                    # Send test interaction through the distribution layer
                    distribution_task = asyncio.create_task(
                        self.layer6.distribute_entity(
                            entity_id,
                            target_id=coupling.environmental_entity_id,
                            context={
                                'test_interaction': True,
                                'active_inference': True,
                                'test_id': test.id
                            },
                            interaction_data=test.test_parameters
                        )
                    )
                    
                    inference_results.append({
                        'coupling_id': coupling.id,
                        'test_id': test.id,
                        'test_parameters': test.test_parameters,
                        'status': 'submitted'
                    })
                else:
                    # No distribution layer available
                    inference_results.append({
                        'coupling_id': coupling.id,
                        'test_id': test.id,
                        'test_parameters': test.test_parameters,
                        'status': 'distribution_layer_unavailable'
                    })
        
        return {
            'success': True,
            'entity_id': entity_id,
            'couplings_tested': len(inference_results),
            'inference_results': inference_results,
            'elapsed_time': time.time() - start_time
        }
    
    async def process_test_result(self, test_id, actual_result):
        """
        Process the result of an active inference test.
        Updates coupling based on test information gain.
        """
        if not self.active_inference_enabled:
            return {'success': False, 'error': 'Active inference is disabled'}
        
        start_time = time.time()
        
        # Evaluate test result
        evaluation = await self.active_inference.evaluate_test_result(test_id, actual_result)
        
        if not evaluation.get('success', False):
            return {
                'success': False,
                'error': evaluation.get('error', 'Test evaluation failed'),
                'elapsed_time': time.time() - start_time
            }
        
        # Log information gain
        self.logger.info(f"Test {test_id} resulted in information gain: {evaluation['information_gain']:.3f}")
        
        # Get the coupling this test was for
        coupling_id = evaluation.get('coupling_updates', {}).get('coupling_id')
        if coupling_id:
            coupling = await self.coupling_registry.get_coupling(coupling_id)
            
            if coupling:
                # Apply any coupling updates from test result
                if 'coupling_updates' in evaluation and evaluation['coupling_updates'].get('coupling_changed', False):
                    # The active inference controller has already updated the coupling
                    self.logger.debug(f"Coupling {coupling_id} updated based on test results")
                
                # If this test significantly reduced uncertainty, consider additional tests
                if evaluation['information_gain'] > 0.8:
                    # Schedule additional tests if high information gain suggests more could be learned
                    asyncio.create_task(self.schedule_follow_up_tests(coupling))
        
        # Check if layer5 (maintenance) needs to be informed of test results
        if self.layer5 and evaluation.get('information_gain', 0) < 0.3:
            # Low information gain might indicate a boundary issue
            asyncio.create_task(
                self.layer5.examine_coupling(
                    coupling_id, 
                    reason="low_active_inference_gain",
                    metadata={
                        'test_id': test_id,
                        'information_gain': evaluation.get('information_gain', 0),
                        'target_area': evaluation.get('target_area', 'unknown')
                    }
                )
            )
        
        return {
            'success': True,
            'test_id': test_id,
            'evaluation': evaluation,
            'elapsed_time': time.time() - start_time
        }
    
    async def schedule_follow_up_tests(self, coupling):
        """
        Schedule follow-up tests when a test yields high information gain.
        This allows the system to progressively refine its understanding.
        """
        # Wait a bit before the follow-up test
        await asyncio.sleep(5)
        
        # Generate a new test focusing on a different uncertainty area
        test = await self.active_inference.generate_test_interaction(
            coupling.id,
            uncertainty_focus=True
        )
        
        if test and self.layer6:
            # Send follow-up test through distribution layer
            await self.layer6.distribute_entity(
                coupling.internal_entity_id,
                target_id=coupling.environmental_entity_id,
                context={
                    'test_interaction': True,
                    'active_inference': True,
                    'test_id': test.id,
                    'follow_up': True
                },
                interaction_data=test.test_parameters
            )
            
            self.logger.info(f"Scheduled follow-up test {test.id} for coupling {coupling.id}")
    
    async def simulate_counterfactual_coupling(self, coupling_id, variations=3):
        """
        Simulate alternative coupling configurations.
        Implements Seth's counterfactual processing principle.
        """
        if not self.counterfactual_enabled:
            return {'success': False, 'error': 'Counterfactual simulation is disabled'}
        
        start_time = time.time()
        
        # Get the coupling
        coupling = await self.coupling_registry.get_coupling(coupling_id)
        if not coupling:
            return {
                'success': False,
                'error': 'Coupling not found',
                'elapsed_time': time.time() - start_time
            }
        
        # Generate counterfactual variations
        counterfactuals = await self.counterfactual_simulator.generate_coupling_variations(
            coupling, variations
        )
        
        # Simulate outcomes
        simulation_results = await self.counterfactual_simulator.simulate_outcomes(
            counterfactuals
        )
        
        # Identify optimal configuration
        optimal_config = await self.counterfactual_simulator.identify_optimal_configuration(
            simulation_results
        )
        
        # Log the counterfactual simulation
        self.logger.info(f"Completed counterfactual simulation for coupling {coupling_id} with {variations} variations")
        
        # Store optimal configuration for future reference
        if not hasattr(coupling, 'properties'):
            coupling.properties = {}
            
        coupling.properties['counterfactual_recommendation'] = {
            'timestamp': time.time(),
            'optimal_config': {
                'variation_type': optimal_config['optimal_configuration']['variation_type'],
                'description': optimal_config['optimal_configuration']['description'],
                'predicted_improvement': optimal_config['improvement']
            }
        }
        
        # Update coupling in registry
        await self.coupling_registry.update_coupling(coupling)
        
        return {
            'success': True,
            'coupling_id': coupling_id,
            'counterfactual_count': len(counterfactuals),
            'simulation_results': simulation_results,
            'optimal_configuration': optimal_config,
            'elapsed_time': time.time() - start_time
        }
    
    async def apply_counterfactual_recommendation(self, coupling_id):
        """
        Apply the recommended counterfactual configuration to an actual coupling.
        """
        if not self.counterfactual_enabled:
            return {'success': False, 'error': 'Counterfactual simulation is disabled'}
        
        # Get the coupling
        coupling = await self.coupling_registry.get_coupling(coupling_id)
        if not coupling:
            return {'success': False, 'error': 'Coupling not found'}
        
        # Check if we have a recommendation
        if (not hasattr(coupling, 'properties') or
            'counterfactual_recommendation' not in coupling.properties):
            return {'success': False, 'error': 'No counterfactual recommendation available'}
        
        recommendation = coupling.properties['counterfactual_recommendation']
        if 'optimal_config' not in recommendation:
            return {'success': False, 'error': 'Invalid recommendation format'}
        
        optimal = recommendation['optimal_config']
        
        # Apply recommended changes
        original_values = {}
        
        # Record original values
        original_values['coupling_type'] = coupling.coupling_type
        original_values['coupling_strength'] = coupling.coupling_strength
        if hasattr(coupling, 'properties'):
            original_values['properties'] = dict(coupling.properties)
        
        # Apply changes based on variation type
        variation_type = optimal.get('variation_type')
        
        if variation_type == 'strength_increase':
            coupling.coupling_strength = min(1.0, coupling.coupling_strength * 1.2)
        
        elif variation_type == 'strength_decrease':
            coupling.coupling_strength = max(0.1, coupling.coupling_strength * 0.8)
        
        elif variation_type == 'type_adaptive':
            coupling.coupling_type = CouplingType.ADAPTIVE
        
        elif variation_type == 'type_predictive':
            coupling.coupling_type = CouplingType.PREDICTIVE
        
        elif variation_type == 'property_responsiveness':
            if not hasattr(coupling, 'properties'):
                coupling.properties = {}
            coupling.properties['response_threshold'] = 0.3
        
        elif variation_type == 'property_reliability':
            if not hasattr(coupling, 'properties'):
                coupling.properties = {}
            coupling.properties['reliability_factor'] = 0.8
        
        elif variation_type == 'property_precision':
            if not hasattr(coupling, 'properties'):
                coupling.properties = {}
            coupling.properties['precision_target'] = 2.0
        
        # Update application metadata
        coupling.properties['counterfactual_application'] = {
            'timestamp': time.time(),
            'applied_variation': variation_type,
            'original_values': original_values,
            'description': optimal.get('description', 'Counterfactual optimization')
        }
        
        # Update coupling in registry
        await self.coupling_registry.update_coupling(coupling)
        
        # Update history
        if hasattr(coupling, 'history'):
            coupling.history.append({
                'timestamp': time.time(),
                'action': 'counterfactual_applied',
                'variation_type': variation_type,
                'predicted_improvement': optimal.get('predicted_improvement', 0)
            })
        
        self.logger.info(f"Applied counterfactual recommendation to coupling {coupling_id}: {variation_type}")
        
        return {
            'success': True,
            'coupling_id': coupling_id,
            'applied_variation': variation_type,
            'description': optimal.get('description'),
            'original_values': original_values
        }
    
    async def _get_relevant_predictions(self, entity_id, interaction_type=None):
        """
        Get relevant predictions for an entity and interaction type.
        This method retrieves predictions that match the incoming interaction.
        """
        if not self.prediction_enabled or not entity_id:
            return []
            
        # Get recent predictions for this entity
        predictions = await self.predictive_modeler.get_predictions_for_entity(
            entity_id, 
            limit=5, 
            future_only=False  # Include recently verified predictions for comparison
        )
        
        # If we have a specific interaction type, prioritize matching predictions
        if interaction_type and predictions:
            # Sort by relevance:
            # 1. Predictions with matching interaction type
            # 2. Unverified predictions (future predictions)
            # 3. Most recent predictions
            
            def prediction_relevance(pred):
                # Calculate a relevance score (higher is better)
                score = 0
                
                # Higher score for matching interaction type
                if 'predicted_interaction_type' in pred.predicted_data:
                    if pred.predicted_data['predicted_interaction_type'] == interaction_type:
                        score += 10
                
                # Higher score for unverified predictions
                if pred.verification_time is None:
                    score += 5
                    
                # Higher score for recent predictions
                time_factor = max(0, 3600 - (time.time() - pred.prediction_time)) / 3600
                score += time_factor * 3
                
                return score
                
            # Sort by relevance score
            predictions.sort(key=prediction_relevance, reverse=True)
            
        # Limit to most relevant predictions
        return predictions[:3] if predictions else []
    
    async def process_predictions_for_entity(self, entity_id):
        """
        Process all predictions for an entity and update precision.
        Useful for periodic maintenance of prediction precision.
        """
        if not self.prediction_enabled:
            return {'success': False, 'status': 'predictions_disabled'}
            
        # Get all predictions for this entity
        predictions = await self.predictive_modeler.get_predictions_for_entity(
            entity_id, limit=10, future_only=False
        )
        
        if not predictions:
            return {'success': True, 'status': 'no_predictions', 'entity_id': entity_id}
            
        # Find verified predictions needing precision updates
        verified_count = 0
        precision_updates = 0
        
        for prediction in predictions:
            if prediction.verification_time is not None and prediction.prediction_error is not None:
                verified_count += 1
                
                # Update precision for verified predictions
                if prediction.environmental_entity_id:
                    await self.predictive_modeler.update_entity_precision(
                        prediction.environmental_entity_id, prediction.prediction_error
                    )
                    precision_updates += 1
                    
        # Generate new predictions if needed
        if verified_count == len(predictions):
            new_prediction = await self.predictive_modeler.predict_interaction(entity_id)
            self.logger.debug(f"Generated new prediction {new_prediction.id} for entity {entity_id}")
        
        return {
            'success': True,
            'entity_id': entity_id,
            'prediction_count': len(predictions),
            'verified_count': verified_count,
            'precision_updates': precision_updates
        }



# === FILE: asf\layer4_environmental_coupling\models.py ===

import time
import uuid
from typing import Dict, List, Set, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict
from asf.environmental_coupling.enums import (
    CouplingType, CouplingStrength, CouplingState,
    EventPriority, PredictionState
)

@dataclass
class CouplingEvent:
    """Represents an event related to environmental coupling."""
    id: str
    event_type: str
    coupling_id: Optional[str] = None
    entity_id: Optional[str] = None
    environmental_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)
    priority: float = 0.5
    processed: bool = False
    processing_time: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    
    # NEW: Seth's Data Paradox enhancements
    predicted: bool = False
    prediction_id: Optional[str] = None
    prediction_error: Optional[float] = None
    precision: Optional[float] = None

@dataclass
class EnvironmentalCoupling:
    """Represents a coupling relationship between internal and environmental entities."""
    id: str
    internal_entity_id: str
    environmental_entity_id: str
    coupling_type: CouplingType = CouplingType.INFORMATIONAL
    coupling_strength: float = 0.5
    coupling_state: CouplingState = CouplingState.FORMING
    creation_time: float = field(default_factory=time.time)
    last_interaction: float = field(default_factory=time.time)
    interaction_count: int = 0
    bayesian_confidence: float = 0.5
    properties: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    tensor_coordinates: Optional[Tuple[int, int]] = None
    
    # NEW: Seth's Data Paradox enhancements
    prediction_precision: float = 1.0  # Inverse variance of prediction errors
    prediction_errors: List[float] = field(default_factory=list)
    expected_interactions: Dict[str, float] = field(default_factory=dict)
    
@dataclass
class EnvironmentalPrediction:
    """Represents a prediction about an environmental entity or interaction."""
    id: str
    environmental_entity_id: str
    predicted_data: Dict[str, Any]
    confidence: float = 0.5
    precision: float = 1.0
    prediction_time: float = field(default_factory=time.time)
    verification_time: Optional[float] = None
    prediction_error: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ActiveInferenceTest:
    """Represents an active test of a prediction."""
    id: str
    coupling_id: str
    prediction_id: Optional[str] = None
    test_parameters: Dict[str, Any] = field(default_factory=dict)
    created_time: float = field(default_factory=time.time)
    execution_time: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    information_gain: Optional[float] = None



# === FILE: asf\layer4_environmental_coupling\predictive_orchestrator.py ===

import asyncio
import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict

class PredictiveProcessingOrchestrator:
    """
    Orchestrates the predictive processing cycle across all components.
    Implements Seth's integrated predictive processing framework with controlled
    hallucination, precision-weighted errors, active inference, and counterfactual simulation.
    """
    def __init__(self, coupling_layer):
        self.coupling_layer = coupling_layer
        self.active_entities = set()
        self.prediction_cycles = defaultdict(int)  # Maps entity_id to cycle count
        self.logger = logging.getLogger("ASF.Layer4.PredictiveOrchestrator")
        
    async def initialize(self):
        """Initialize the orchestrator."""
        self.logger.info("Initializing Predictive Processing Orchestrator")
        return True
        
    async def register_entity(self, entity_id):
        """Register an entity for predictive processing."""
        self.active_entities.add(entity_id)
        self.logger.info(f"Registered entity {entity_id} for predictive processing")
        return True
        
    async def unregister_entity(self, entity_id):
        """Unregister an entity from predictive processing."""
        if entity_id in self.active_entities:
            self.active_entities.remove(entity_id)
            self.logger.info(f"Unregistered entity {entity_id} from predictive processing")
        return True
        
    async def run_predictive_cycle(self, entity_id, context=None):
        """Run a complete predictive processing cycle for an entity."""
        if entity_id not in self.active_entities:
            return {'success': False, 'error': 'Entity not registered'}
            
        # Increment cycle count
        self.prediction_cycles[entity_id] += 1
        cycle_number = self.prediction_cycles[entity_id]
        
        # Enhanced context with cycle information
        enhanced_context = {
            'cycle_number': cycle_number,
            'orchestrator_timestamp': time.time(),
            **(context or {})
        }
        
        # Run the cycle through the coupling layer
        result = await self.coupling_layer.integrate_prediction_cycle(entity_id, enhanced_context)
        
        self.logger.info(f"Completed prediction cycle {cycle_number} for entity {entity_id}")
        
        return {
            'success': True,
            'entity_id': entity_id,
            'cycle_number': cycle_number,
            'cycle_results': result
        }
        
    async def run_continuous_cycles(self, entity_id, interval=60, max_cycles=None):
        """Run continuous predictive cycles for an entity at specified interval."""
        if entity_id not in self.active_entities:
            return {'success': False, 'error': 'Entity not registered'}
            
        cycle_count = 0
        start_time = time.time()
        
        self.logger.info(f"Starting continuous prediction cycles for entity {entity_id} at {interval}s intervals")
        
        while (max_cycles is None or cycle_count < max_cycles) and entity_id in self.active_entities:
            cycle_start = time.time()
            
            # Run a cycle
            await self.run_predictive_cycle(entity_id, {
                'continuous_mode': True,
                'continuous_cycle': cycle_count,
                'elapsed_time': time.time() - start_time
            })
            
            cycle_count += 1
            
            # Wait for next cycle, accounting for processing time
            cycle_duration = time.time() - cycle_start
            wait_time = max(0, interval - cycle_duration)
            
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        
        return {
            'success': True,
            'entity_id': entity_id,
            'cycles_completed': cycle_count,
            'total_runtime': time.time() - start_time
        }
        
    async def run_adaptive_cycles(self, entity_id, min_interval=10, max_interval=300):
        """
        Run adaptive predictive cycles with frequency based on prediction accuracy.
        More accurate predictions = longer intervals, less accurate = shorter intervals.
        """
        if entity_id not in self.active_entities:
            return {'success': False, 'error': 'Entity not registered'}
            
        cycle_count = 0
        current_interval = (min_interval + max_interval) / 2  # Start with middle interval
        start_time = time.time()
        
        self.logger.info(f"Starting adaptive prediction cycles for entity {entity_id}")
        
        while entity_id in self.active_entities:
            cycle_start = time.time()
            
            # Run a cycle
            result = await self.run_predictive_cycle(entity_id, {
                'adaptive_mode': True,
                'current_interval': current_interval,
                'adaptive_cycle': cycle_count
            })
            
            cycle_count += 1
            
            # Adapt interval based on prediction accuracy
            cycle_results = result.get('cycle_results', {})
            evaluations = cycle_results.get('evaluations', {})
            avg_error = evaluations.get('avg_error')
            
            if avg_error is not None:
                # Adjust interval: higher error = shorter interval
                error_factor = 1.0 - min(1.0, avg_error * 2)  # 0 to 1 scale
                new_interval = min_interval + error_factor * (max_interval - min_interval)
                
                # Smooth adjustment
                current_interval = (current_interval * 0.7) + (new_interval * 0.3)
                self.logger.debug(f"Adjusted interval to {current_interval:.1f}s based on error {avg_error:.3f}")
            
            # Wait for next cycle
            cycle_duration = time.time() - cycle_start
            wait_time = max(0, current_interval - cycle_duration)
            
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        
        return {
            'success': True,
            'entity_id': entity_id,
            'cycles_completed': cycle_count,
            'total_runtime': time.time() - start_time
        }
        
    async def perform_maintenance(self):
        """Perform maintenance on the predictive processing system."""
        self.logger.info("Performing maintenance on predictive processing system")
        
        start_time = time.time()
        
        # Collect maintenance metrics
        metrics = {
            'active_entities': len(self.active_entities),
            'total_prediction_cycles': sum(self.prediction_cycles.values()),
            'components_maintained': 0
        }
        
        # Maintain predictive modeler
        if hasattr(self.coupling_layer, 'predictive_modeler'):
            # Clean up old predictions
            await self._cleanup_old_predictions()
            metrics['components_maintained'] += 1
        
        # Maintain active inference controller
        if hasattr(self.coupling_layer, 'active_inference'):
            # Clean up old tests
            await self._cleanup_old_tests()
            metrics['components_maintained'] += 1
        
        # Maintain counterfactual simulator
        if hasattr(self.coupling_layer, 'counterfactual_simulator'):
            # Clean up old simulations
            await self._cleanup_old_simulations()
            metrics['components_maintained'] += 1
        
        metrics['elapsed_time'] = time.time() - start_time
        
        return metrics
        
    async def _cleanup_old_predictions(self):
        """Clean up old predictions to free memory."""
        predictive_modeler = self.coupling_layer.predictive_modeler
        
        # Find old predictions to clean up
        current_time = time.time()
        expired_predictions = []
        
        for pred_id, prediction in predictive_modeler.predictions.items():
            # Clean verified predictions older than 1 hour
            if prediction.verification_time and (current_time - prediction.verification_time > 3600):
                expired_predictions.append(pred_id)
            
            # Clean unverified predictions older than 24 hours
            elif current_time - prediction.prediction_time > 86400:
                expired_predictions.append(pred_id)
        
        # Remove expired predictions
        for pred_id in expired_predictions:
            if pred_id in predictive_modeler.predictions:
                prediction = predictive_modeler.predictions[pred_id]
                # Remove from entity predictions
                if prediction.environmental_entity_id in predictive_modeler.entity_predictions:
                    if pred_id in predictive_modeler.entity_predictions[prediction.environmental_entity_id]:
                        predictive_modeler.entity_predictions[prediction.environmental_entity_id].remove(pred_id)
                # Remove from predictions
                del predictive_modeler.predictions[pred_id]
        
        self.logger.info(f"Cleaned up {len(expired_predictions)} expired predictions")
        
    async def _cleanup_old_tests(self):
        """Clean up old active inference tests."""
        active_inference = self.coupling_layer.active_inference
        
        # Implementation depends on active inference controller implementation
        # This is a placeholder for the cleanup logic
        cleaned_tests = 0
        current_time = time.time()
        
        self.logger.info(f"Cleaned up {cleaned_tests} old active inference tests")
        
    async def _cleanup_old_simulations(self):
        """Clean up old counterfactual simulations."""
        counterfactual_simulator = self.coupling_layer.counterfactual_simulator
        
        # Implementation depends on counterfactual simulator implementation
        # This is a placeholder for the cleanup logic
        cleaned_simulations = 0
        current_time = time.time()
        
        self.logger.info(f"Cleaned up {cleaned_simulations} old counterfactual simulations")



# === FILE: asf\layer4_environmental_coupling\components\__init__.py ===




# === FILE: asf\layer4_environmental_coupling\components\active_inference_controller.py ===

import asyncio
import time
import uuid
import logging
import numpy as np
import random
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

from asf.environmental_coupling.models import ActiveInferenceTest, EnvironmentalCoupling
from asf.environmental_coupling.enums import CouplingType, CouplingState

class ActiveInferenceController:
    """
    Implements Seth's active inference principle to minimize prediction errors.
    Proactively tests environmental couplings to reduce uncertainty.
    """
    def __init__(self, knowledge_substrate):
        self.knowledge_substrate = knowledge_substrate
        self.coupling_registry = None  # Will be set during initialization
        self.inference_history = defaultdict(list)  # Maps coupling_id to test history
        self.test_results = {}  # Maps test_id to results
        self.uncertainty_profiles = {}  # Maps coupling_id to uncertainty profile
        self.logger = logging.getLogger("ASF.Layer4.ActiveInferenceController")
    
    def set_coupling_registry(self, registry):
        """Set reference to coupling registry."""
        self.coupling_registry = registry
    
    async def generate_test_interaction(self, coupling_id, uncertainty_focus=True):
        """
        Generate an interaction that tests a coupling relationship.
        If uncertainty_focus is True, focuses on areas with highest uncertainty.
        """
        # Get coupling data
        coupling = await self.coupling_registry.get_coupling(coupling_id)
        if not coupling:
            self.logger.warning(f"Cannot generate test for unknown coupling: {coupling_id}")
            return None
            
        # Get entity data
        entity = await self.knowledge_substrate.get_entity(coupling.internal_entity_id)
        if not entity:
            self.logger.warning(f"Cannot generate test for unknown entity: {coupling.internal_entity_id}")
            return None
        
        # Create uncertainty profile if it doesn't exist
        if coupling_id not in self.uncertainty_profiles:
            await self._create_uncertainty_profile(coupling_id, coupling)
        
        # Analyze areas of highest uncertainty
        uncertainty_areas = await self._analyze_coupling_uncertainty(coupling)
        
        # Generate test based on uncertainty focus preference
        if uncertainty_focus and uncertainty_areas:
            # Focus on the area with highest uncertainty
            target_area = uncertainty_areas[0]
            test_interaction = await self._generate_uncertainty_reducing_test(coupling, entity, target_area)
        else:
            # Generate a more general test
            test_interaction = await self._generate_standard_test(coupling, entity)
        
        # Create test record
        test = ActiveInferenceTest(
            id=str(uuid.uuid4()),
            coupling_id=coupling_id,
            prediction_id=test_interaction.get('prediction_id'),
            test_parameters=test_interaction
        )
        
        # Record in history
        self.inference_history[coupling_id].append({
            'test_id': test.id,
            'timestamp': time.time(),
            'test_type': 'uncertainty_focused' if uncertainty_focus else 'standard',
            'target_area': uncertainty_areas[0] if uncertainty_focus and uncertainty_areas else 'general'
        })
        
        # Log the test creation
        self.logger.info(f"Generated active inference test {test.id} for coupling {coupling_id}, " +
                         f"targeting {'uncertainty in ' + uncertainty_areas[0] if uncertainty_focus and uncertainty_areas else 'general properties'}")
        
        return test
    
    async def evaluate_test_result(self, test_id, actual_result):
        """
        Evaluate the result of an active inference test.
        Updates uncertainty profiles based on test results.
        """
        # Find the test
        test = None
        coupling_id = None
        
        # Search for test in inference history
        for cid, history in self.inference_history.items():
            for entry in history:
                if entry.get('test_id') == test_id:
                    coupling_id = cid
                    test_entry = entry
                    break
            if coupling_id:
                break
        
        if not coupling_id:
            self.logger.warning(f"Test {test_id} not found in inference history")
            return {'success': False, 'error': 'Test not found'}
        
        # Get the coupling
        coupling = await self.coupling_registry.get_coupling(coupling_id)
        if not coupling:
            self.logger.warning(f"Coupling {coupling_id} not found for test {test_id}")
            return {'success': False, 'error': 'Coupling not found'}
        
        # Calculate information gain
        test_parameters = test_entry.get('test_parameters', {})
        target_area = test_entry.get('target_area', 'general')
        
        info_gain = self._calculate_information_gain(test_parameters, actual_result, target_area)
        
        # Update uncertainty profile
        await self._update_uncertainty_profile(coupling_id, target_area, info_gain, actual_result)
        
        # Store test result
        self.test_results[test_id] = {
            'coupling_id': coupling_id,
            'actual_result': actual_result,
            'information_gain': info_gain,
            'target_area': target_area,
            'evaluation_time': time.time()
        }
        
        # Update coupling based on test results
        update_result = await self._update_coupling_from_test(coupling, target_area, info_gain, actual_result)
        
        self.logger.info(f"Evaluated test {test_id} with information gain {info_gain:.3f}")
        
        return {
            'success': True,
            'test_id': test_id,
            'information_gain': info_gain,
            'target_area': target_area,
            'coupling_updates': update_result
        }
    
    async def _create_uncertainty_profile(self, coupling_id, coupling):
        """Create initial uncertainty profile for a coupling."""
        # Initialize uncertainty values for different areas
        self.uncertainty_profiles[coupling_id] = {
            'response_timing': 1.0,  # Higher value = higher uncertainty
            'interaction_pattern': 1.0,
            'reliability': 1.0,
            'content_structure': 1.0,
            'contextual_behavior': 1.0,
            'last_updated': time.time(),
            'update_count': 0
        }
        
        # If coupling has some history, refine initial estimates
        if hasattr(coupling, 'interaction_count') and coupling.interaction_count > 0:
            if coupling.interaction_count > 5:
                self.uncertainty_profiles[coupling_id]['interaction_pattern'] *= 0.8
            
            if hasattr(coupling, 'bayesian_confidence') and coupling.bayesian_confidence > 0.6:
                self.uncertainty_profiles[coupling_id]['reliability'] *= 0.8
            
            if hasattr(coupling, 'prediction_precision') and coupling.prediction_precision > 1.0:
                self.uncertainty_profiles[coupling_id]['response_timing'] *= 0.8
    
    async def _analyze_coupling_uncertainty(self, coupling):
        """Identify areas of highest uncertainty for a coupling."""
        coupling_id = coupling.id
        
        # If we don't have an uncertainty profile, create one
        if coupling_id not in self.uncertainty_profiles:
            await self._create_uncertainty_profile(coupling_id, coupling)
        
        profile = self.uncertainty_profiles[coupling_id]
        
        # Sort areas by uncertainty (descending)
        areas = [
            'response_timing',
            'interaction_pattern',
            'reliability',
            'content_structure',
            'contextual_behavior'
        ]
        
        sorted_areas = sorted(areas, key=lambda area: profile[area], reverse=True)
        
        # Additional dynamic factors that may increase uncertainty
        
        # Low interaction count increases pattern uncertainty
        if hasattr(coupling, 'interaction_count') and coupling.interaction_count < 5:
            if 'interaction_pattern' not in sorted_areas[:2]:
                # Move to front of list
                sorted_areas.remove('interaction_pattern')
                sorted_areas.insert(0, 'interaction_pattern')
        
        # Low bayesian confidence increases reliability uncertainty
        if hasattr(coupling, 'bayesian_confidence') and coupling.bayesian_confidence < 0.5:
            if 'reliability' not in sorted_areas[:2]:
                # Move to front or second position
                sorted_areas.remove('reliability')
                sorted_areas.insert(min(1, len(sorted_areas)), 'reliability')
        
        # Low prediction precision increases timing uncertainty
        if hasattr(coupling, 'prediction_precision') and coupling.prediction_precision < 0.8:
            if 'response_timing' not in sorted_areas[:2]:
                # Move to front or second position
                sorted_areas.remove('response_timing')
                sorted_areas.insert(min(1, len(sorted_areas)), 'response_timing')
        
        return sorted_areas
    
    async def _generate_uncertainty_reducing_test(self, coupling, entity, uncertainty_area):
        """Generate a test focused on reducing a specific uncertainty area."""
        # Start with basic test structure
        test_interaction = {
            'interaction_type': 'active_inference_test',
            'target_entity_id': coupling.environmental_entity_id,
            'uncertainty_area': uncertainty_area,
            'source_entity_id': coupling.internal_entity_id,
            'timestamp': time.time(),
            'test_id': str(uuid.uuid4())
        }
        
        # Customize test based on uncertainty area
        if uncertainty_area == 'response_timing':
            test_interaction.update({
                'test_content': {
                    'test_type': 'timing_test',
                    'request_timestamp': time.time(),
                    'expected_response_window': [0.5, 3.0],  # seconds
                    'require_timestamp': True
                },
                'prediction': {
                    'expected_response_time': 1.5,  # seconds
                    'timestamp_precision': 0.1  # seconds
                }
            })
            
        elif uncertainty_area == 'interaction_pattern':
            # Define a sequence pattern test
            test_interaction.update({
                'test_content': {
                    'test_type': 'pattern_test',
                    'sequence': [1, 2, 3],
                    'expected_next': 4,
                    'pattern_description': 'sequential_increment'
                },
                'prediction': {
                    'expected_completion': True,
                    'alternative_patterns': ['fibonacci', 'even_numbers', 'powers_of_two']
                }
            })
            
        elif uncertainty_area == 'reliability':
            # Create an echo test with verification
            test_interaction.update({
                'test_content': {
                    'test_type': 'reliability_test',
                    'verification_code': str(uuid.uuid4())[:8],
                    'verification_instruction': 'return_exact_code',
                    'verification_timestamp': time.time()
                },
                'prediction': {
                    'expected_verification': True,
                    'expected_code_match': True
                }
            })
            
        elif uncertainty_area == 'content_structure':
            # Test ability to follow a structured data format
            test_interaction.update({
                'test_content': {
                    'test_type': 'structure_test',
                    'required_fields': ['id', 'timestamp', 'value', 'metadata'],
                    'expected_types': {
                        'id': 'string',
                        'timestamp': 'number',
                        'value': 'number',
                        'metadata': 'object'
                    }
                },
                'prediction': {
                    'expected_conformance': 0.8,  # 80% of fields correct
                    'critical_fields': ['id', 'timestamp']
                }
            })
            
        elif uncertainty_area == 'contextual_behavior':
            # Test how entity behavior changes based on context
            contexts = ['normal', 'urgent', 'error', 'maintenance']
            selected_context = random.choice(contexts)
            
            test_interaction.update({
                'test_content': {
                    'test_type': 'context_test',
                    'context': selected_context,
                    'context_indicator': f"This is a {selected_context} situation",
                    'expected_behavior': self._get_expected_behavior(selected_context)
                },
                'prediction': {
                    'context_recognition': True,
                    'behavior_adaptation': True
                }
            })
        
        else:  # 'general' or unknown area
            # General test with multiple aspects
            test_interaction.update({
                'test_content': {
                    'test_type': 'general_test',
                    'verification_code': str(uuid.uuid4())[:6],
                    'timestamp': time.time(),
                    'structured_request': {
                        'operation': 'echo',
                        'payload': {
                            'code': test_interaction['test_id'][-8:],
                            'timestamp': time.time()
                        }
                    }
                },
                'prediction': {
                    'response_expected': True,
                    'response_time_range': [0.5, 5.0]
                }
            })
        
        # Add tracking ID for the prediction we're making
        test_interaction['prediction_id'] = str(uuid.uuid4())
        
        return test_interaction
    
    async def _generate_standard_test(self, coupling, entity):
        """Generate a standard test interaction without specific uncertainty focus."""
        # Create a general test that checks basic functionality
        test_interaction = {
            'interaction_type': 'active_inference_test',
            'target_entity_id': coupling.environmental_entity_id,
            'uncertainty_area': 'general',
            'source_entity_id': coupling.internal_entity_id,
            'timestamp': time.time(),
            'test_id': str(uuid.uuid4()),
            'test_content': {
                'test_type': 'general_test',
                'message': 'Standard active inference test',
                'request_id': str(uuid.uuid4()),
                'echo_data': f"test_{int(time.time())}"
            },
            'prediction': {
                'response_expected': True,
                'contains_echo': True,
                'response_time_under': 5.0  # seconds
            }
        }
        
        # Add tracking ID for the prediction we're making
        test_interaction['prediction_id'] = str(uuid.uuid4())
        
        return test_interaction
    
    def _calculate_information_gain(self, test_parameters, actual_result, uncertainty_area):
        """
        Calculate information gain from test results.
        Higher values indicate more valuable information was acquired.
        """
        # Extract test details and prediction
        test_content = test_parameters.get('test_content', {})
        prediction = test_parameters.get('prediction', {})
        test_type = test_content.get('test_type', 'unknown')
        
        # Base information gain starts neutral
        information_gain = 0.5
        
        # Calculate based on test type
        if test_type == 'timing_test':
            # Extract expectations and actual values
            expected_time = prediction.get('expected_response_time', 1.0)
            actual_time = actual_result.get('response_time')
            
            if actual_time is not None:
                # Calculate time prediction error
                time_error = abs(actual_time - expected_time)
                expected_window = test_content.get('expected_response_window', [0, 10])
                max_window = expected_window[1] - expected_window[0]
                
                # Normalize error to 0-1 scale
                normalized_error = min(1.0, time_error / max_window)
                
                # Information gain is higher when error is lower
                information_gain = 1.0 - normalized_error
            else:
                # No timing data means little information gained
                information_gain = 0.1
        
        elif test_type == 'pattern_test':
            # Extract expectations and actual values
            expected_next = test_content.get('expected_next')
            actual_next = actual_result.get('next_value')
            
            if actual_next is not None:
                # Binary result - either correct or not
                if expected_next == actual_next:
                    information_gain = 1.0
                else:
                    # Still gain some information even if wrong
                    information_gain = 0.3
                    
                # Additional information if alternative pattern identified
                if actual_result.get('pattern_identified'):
                    information_gain += 0.2
            else:
                information_gain = 0.1
        
        elif test_type == 'reliability_test':
            # Extract verification expectations
            verification_code = test_content.get('verification_code')
            actual_code = actual_result.get('verification_code')
            
            # Check if verification happened at all
            if actual_code is not None:
                # How accurate was the verification
                if verification_code == actual_code:
                    information_gain = 1.0
                else:
                    # Partial match also provides information
                    similarity = self._calculate_string_similarity(verification_code, actual_code)
                    information_gain = max(0.1, similarity)
            else:
                information_gain = 0.0
        
        elif test_type == 'structure_test':
            # Check conformance to expected structure
            required_fields = test_content.get('required_fields', [])
            expected_types = test_content.get('expected_types', {})
            
            if isinstance(actual_result, dict):
                # Count matching fields
                field_matches = 0
                type_matches = 0
                
                for field in required_fields:
                    if field in actual_result:
                        field_matches += 1
                        
                        # Also check type if expected
                        if field in expected_types:
                            expected_type = expected_types[field]
                            actual_type = self._get_type_name(actual_result[field])
                            
                            if expected_type == actual_type:
                                type_matches += 1
                
                # Calculate information gain based on matches
                if required_fields:
                    field_score = field_matches / len(required_fields)
                    type_score = type_matches / len(expected_types) if expected_types else 1.0
                    
                    # Weight field presence more heavily than type matching
                    information_gain = (field_score * 0.6) + (type_score * 0.4)
                else:
                    information_gain = 0.5
            else:
                # Not even a dict structure
                information_gain = 0.1
        
        elif test_type == 'context_test':
            # Check if context was recognized and behavior adapted
            expected_behavior = test_content.get('expected_behavior', {})
            context = test_content.get('context')
            
            if isinstance(actual_result, dict):
                # Check for context indicators in response
                context_recognized = False
                behavior_appropriate = False
                
                # Look for context acknowledgment
                if actual_result.get('context_acknowledged') == context:
                    context_recognized = True
                
                # Check behavior against expectations
                if expected_behavior and actual_result.get('behavior'):
                    similarity = self._calculate_behavior_match(
                        expected_behavior, 
                        actual_result['behavior']
                    )
                    behavior_appropriate = similarity > 0.7
                
                # Calculate information gain
                if context_recognized and behavior_appropriate:
                    information_gain = 1.0
                elif context_recognized:
                    information_gain = 0.7
                elif behavior_appropriate:
                    information_gain = 0.6
                else:
                    information_gain = 0.3
            else:
                information_gain = 0.1
        
        else:  # general or unknown test type
            # For general tests, check basic response properties
            if isinstance(actual_result, dict) and actual_result:
                # Response exists and has content
                information_gain = 0.6
                
                # Check for echo data
                echo_data = test_content.get('echo_data')
                if echo_data and actual_result.get('echo') == echo_data:
                    information_gain += 0.2
                    
                # Check response timing if available
                if 'response_time' in actual_result:
                    information_gain += 0.1
            else:
                # Empty or invalid response
                information_gain = 0.2
        
        # Ensure result is normalized
        return max(0.0, min(1.0, information_gain))
    
    async def _update_uncertainty_profile(self, coupling_id, uncertainty_area, information_gain, actual_result):
        """Update the uncertainty profile based on test results."""
        if coupling_id not in self.uncertainty_profiles:
            # If profile doesn't exist, we can't update it
            return False
        
        profile = self.uncertainty_profiles[coupling_id]
        
        # Higher information gain means lower uncertainty
        current_uncertainty = profile.get(uncertainty_area, 1.0)
        
        # Calculate new uncertainty with exponential moving average
        # More weight on new observation if current uncertainty is high
        weight = min(0.8, current_uncertainty)  # Weight between 0 and 0.8
        
        # Convert information gain to uncertainty reduction
        # Higher information gain = lower uncertainty
        uncertainty_reduction = 1.0 - information_gain
        
        # Update uncertainty value
        new_uncertainty = (current_uncertainty * (1 - weight)) + (uncertainty_reduction * weight)
        
        # Ensure uncertainty stays within bounds
        new_uncertainty = max(0.1, min(1.0, new_uncertainty))
        
        # Update profile
        profile[uncertainty_area] = new_uncertainty
        profile['last_updated'] = time.time()
        profile['update_count'] = profile.get('update_count', 0) + 1
        
        return True
    
    async def _update_coupling_from_test(self, coupling, uncertainty_area, information_gain, actual_result):
        """Update coupling based on test results."""
        updates = []
        coupling_changed = False
        
        # Update coupling based on uncertainty area
        if uncertainty_area == 'response_timing':
            if hasattr(coupling, 'properties') and 'expected_response_time' not in coupling.properties:
                coupling.properties['expected_response_time'] = actual_result.get('response_time', 1.0)
                updates.append('added expected_response_time property')
                coupling_changed = True
            
            if hasattr(coupling, 'prediction_precision'):
                # Adjust prediction precision based on information gain
                precision_delta = (information_gain - 0.5) * 0.5  # Scale to smaller adjustment
                coupling.prediction_precision = max(0.1, coupling.prediction_precision + precision_delta)
                updates.append(f"adjusted prediction_precision to {coupling.prediction_precision:.3f}")
                coupling_changed = True
        
        elif uncertainty_area == 'reliability':
            if hasattr(coupling, 'bayesian_confidence'):
                # Adjust confidence based on test results
                confidence_delta = (information_gain - 0.5) * 0.1  # Small adjustment
                coupling.bayesian_confidence = max(0.1, min(0.95, coupling.bayesian_confidence + confidence_delta))
                updates.append(f"adjusted bayesian_confidence to {coupling.bayesian_confidence:.3f}")
                coupling_changed = True
        
        elif uncertainty_area == 'interaction_pattern':
            if hasattr(coupling, 'properties') and isinstance(coupling.properties, dict):
                # Record pattern information
                if 'interaction_patterns' not in coupling.properties:
                    coupling.properties['interaction_patterns'] = {}
                
                test_content = actual_result.get('test_content', {})
                if test_content.get('pattern_description'):
                    pattern_type = test_content['pattern_description']
                    coupling.properties['interaction_patterns'][pattern_type] = {
                        'verified': information_gain > 0.7,
                        'last_tested': time.time()
                    }
                    updates.append(f"updated interaction pattern: {pattern_type}")
                    coupling_changed = True
        
        # General updates applicable to all test types
        if hasattr(coupling, 'properties') and isinstance(coupling.properties, dict):
            # Record test history
            if 'active_inference_tests' not in coupling.properties:
                coupling.properties['active_inference_tests'] = []
            
            # Add test result summary
            coupling.properties['active_inference_tests'].append({
                'area': uncertainty_area,
                'information_gain': information_gain,
                'timestamp': time.time()
            })
            
            # Limit test history size
            if len(coupling.properties['active_inference_tests']) > 10:
                coupling.properties['active_inference_tests'] = coupling.properties['active_inference_tests'][-10:]
            
            updates.append("added test to history")
            coupling_changed = True
        
        # Update coupling counts
        if hasattr(coupling, 'interaction_count'):
            coupling.interaction_count += 1
            updates.append(f"incremented interaction_count to {coupling.interaction_count}")
            coupling_changed = True
        
        # Record last interaction time
        coupling.last_interaction = time.time()
        
        # If coupling changed and we have registry, update it
        if coupling_changed and self.coupling_registry:
            await self.coupling_registry.update_coupling(coupling)
        
        return {
            'coupling_id': coupling.id,
            'updates': updates,
            'coupling_changed': coupling_changed
        }
    
    def _get_expected_behavior(self, context):
        """Get expected behavior for a context test."""
        behaviors = {
            'normal': {
                'response_time': 'standard',
                'priority': 'normal',
                'detail_level': 'standard',
                'error_checking': 'standard'
            },
            'urgent': {
                'response_time': 'accelerated',
                'priority': 'high',
                'detail_level': 'minimal',
                'error_checking': 'minimal'
            },
            'error': {
                'response_time': 'standard',
                'priority': 'high',
                'detail_level': 'detailed',
                'error_checking': 'thorough'
            },
            'maintenance': {
                'response_time': 'relaxed',
                'priority': 'low',
                'detail_level': 'comprehensive',
                'error_checking': 'thorough'
            }
        }
        
        return behaviors.get(context, behaviors['normal'])
    
    def _calculate_string_similarity(self, str1, str2):
        """Calculate simple string similarity (0-1 scale)."""
        if not str1 or not str2:
            return 0.0
        
        # Length of longest common substring
        shorter = min(len(str1), len(str2))
        if shorter == 0:
            return 0.0
        
        # Count matching characters
        matches = sum(c1 == c2 for c1, c2 in zip(str1, str2))
        return matches / shorter
    
    def _calculate_behavior_match(self, expected, actual):
        """Calculate how well actual behavior matches expected behavior."""
        if not isinstance(expected, dict) or not isinstance(actual, dict):
            return 0.0
        
        total_attributes = len(expected)
        if total_attributes == 0:
            return 0.0
        
        matching_attributes = 0
        
        for key, exp_value in expected.items():
            if key in actual and actual[key] == exp_value:
                matching_attributes += 1
        
        return matching_attributes / total_attributes
    
    def _get_type_name(self, value):
        """Get type name of a value."""
        if isinstance(value, str):
            return 'string'
        elif isinstance(value, (int, float)):
            return 'number'
        elif isinstance(value, bool):
            return 'boolean'
        elif isinstance(value, dict):
            return 'object'
        elif isinstance(value, list):
            return 'array'
        elif value is None:
            return 'null'
        else:
            return 'unknown'



# === FILE: asf\layer4_environmental_coupling\components\coherence_boundary.py ===

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



# === FILE: asf\layer4_environmental_coupling\components\context_tracker.py ===

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



# === FILE: asf\layer4_environmental_coupling\components\counterfactual_simulator.py ===


import asyncio
import time
import uuid
import logging
import numpy as np
import random
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

from asf.environmental_coupling.models import EnvironmentalCoupling
from asf.environmental_coupling.enums import CouplingType, CouplingStrength, CouplingState

class CounterfactualSimulator:
    """
    Implements Seth's counterfactual processing principle for coupling simulations.
    Simulates alternative coupling configurations to optimize without direct testing.
    """
    def __init__(self):
        self.simulation_history = defaultdict(list)  # Maps coupling_id to simulation history
        self.simulation_models = {}  # Maps coupling_id to simulation models
        self.variation_templates = {}  # Maps variation_type to generation templates
        self.outcome_evaluators = {}  # Maps coupling_type to outcome evaluators
        self.logger = logging.getLogger("ASF.Layer4.CounterfactualSimulator")
    
    async def generate_coupling_variations(self, base_coupling, variations=3):
        """Generate variations of a coupling configuration."""
        coupling_variations = []
        
        # Initialize variation templates if not already done
        if not self.variation_templates:
            self._initialize_variation_templates()
        
        # Generate different variation types based on coupling properties
        for i in range(variations):
            variation = await self._create_coupling_variation(base_coupling, i)
            coupling_variations.append(variation)
            
            self.logger.debug(f"Generated variation {i+1} of type {variation['variation_type']} for coupling {base_coupling.id}")
        
        return coupling_variations
    
    async def simulate_outcomes(self, coupling_variations):
        """Simulate outcomes of different coupling variations."""
        simulation_results = []
        
        for variation in coupling_variations:
            outcome = await self._simulate_variation_outcome(variation)
            simulation_results.append({
                'variation': variation,
                'outcome': outcome
            })
            
            variation_id = variation.get('id', 'unknown')
            self.logger.debug(f"Simulated outcome for variation {variation_id} with score {self._calculate_simulation_score(outcome):.3f}")
        
        return simulation_results
    
    async def identify_optimal_configuration(self, simulation_results):
        """Identify the optimal coupling configuration from simulation results."""
        if not simulation_results:
            return None
        
        # Rank based on performance metrics
        ranked_results = sorted(
            simulation_results,
            key=lambda r: self._calculate_simulation_score(r['outcome']),
            reverse=True
        )
        
        # Return the best configuration
        optimal = ranked_results[0]
        improvement = self._calculate_improvement(optimal, simulation_results)
        
        self.logger.info(f"Identified optimal configuration with improvement of {improvement:.2f}x over average alternatives")
        
        return {
            'optimal_configuration': optimal['variation'],
            'predicted_outcome': optimal['outcome'],
            'improvement': improvement,
            'all_variations': len(simulation_results)
        }
    
    async def record_actual_outcome(self, coupling_id, selected_variation, actual_outcome):
        """Record actual outcome of a selected variation to improve simulation accuracy."""
        if coupling_id not in self.simulation_models:
            # Initialize a model if we don't have one
            self.simulation_models[coupling_id] = {
                'outcomes': [],
                'accuracy': 0.5,  # Initial moderate accuracy
                'last_updated': time.time()
            }
        
        model = self.simulation_models[coupling_id]
        
        # Compare actual outcome to predicted
        predicted_score = self._calculate_simulation_score(selected_variation.get('predicted_outcome', {}))
        actual_score = self._calculate_simulation_score(actual_outcome)
        
        # Calculate prediction error
        prediction_error = abs(predicted_score - actual_score)
        normalized_error = min(1.0, prediction_error / max(0.1, predicted_score))
        
        # Update model accuracy with exponential moving average
        model['accuracy'] = 0.8 * model['accuracy'] + 0.2 * (1.0 - normalized_error)
        
        # Add to outcomes history
        model['outcomes'].append({
            'timestamp': time.time(),
            'variation_type': selected_variation.get('variation_type'),
            'predicted_score': predicted_score,
            'actual_score': actual_score,
            'prediction_error': normalized_error
        })
        
        # Limit history size
        if len(model['outcomes']) > 20:
            model['outcomes'] = model['outcomes'][-20:]
        
        model['last_updated'] = time.time()
        
        self.logger.info(f"Updated simulation model for coupling {coupling_id}, new accuracy: {model['accuracy']:.3f}")
        
        return {
            'prediction_error': normalized_error,
            'model_accuracy': model['accuracy'],
            'outcomes_recorded': len(model['outcomes'])
        }
    
    def _initialize_variation_templates(self):
        """Initialize templates for generating variations."""
        # Strength variations
        self.variation_templates['strength_increase'] = {
            'coupling_strength': lambda c: min(1.0, c.coupling_strength * 1.2),
            'description': 'Increased coupling strength'
        }
        
        self.variation_templates['strength_decrease'] = {
            'coupling_strength': lambda c: max(0.1, c.coupling_strength * 0.8),
            'description': 'Decreased coupling strength'
        }
        
        # Type variations
        self.variation_templates['type_adaptive'] = {
            'coupling_type': lambda c: CouplingType.ADAPTIVE,
            'description': 'Changed to adaptive coupling'
        }
        
        self.variation_templates['type_predictive'] = {
            'coupling_type': lambda c: CouplingType.PREDICTIVE,
            'description': 'Changed to predictive coupling'
        }
        
        # Property variations
        self.variation_templates['property_responsiveness'] = {
            'properties': lambda c: {**getattr(c, 'properties', {}), 'response_threshold': 0.3},
            'description': 'Optimized for responsiveness'
        }
        
        self.variation_templates['property_reliability'] = {
            'properties': lambda c: {**getattr(c, 'properties', {}), 'reliability_factor': 0.8},
            'description': 'Optimized for reliability'
        }
        
        self.variation_templates['property_precision'] = {
            'properties': lambda c: {**getattr(c, 'properties', {}), 'precision_target': 2.0},
            'description': 'Optimized for prediction precision'
        }
    
    async def _create_coupling_variation(self, base_coupling, variation_index):
        """Create a variation of a coupling configuration."""
        # Clone base coupling properties for the variation
        variation = {
            'id': str(uuid.uuid4()),
            'base_coupling_id': base_coupling.id,
            'internal_entity_id': base_coupling.internal_entity_id,
            'environmental_entity_id': base_coupling.environmental_entity_id,
            'coupling_type': base_coupling.coupling_type,
            'coupling_strength': base_coupling.coupling_strength,
            'variation_index': variation_index
        }
        
        # Copy properties if they exist
        if hasattr(base_coupling, 'properties') and base_coupling.properties:
            variation['properties'] = dict(base_coupling.properties)
        else:
            variation['properties'] = {}
        
        # Apply variation based on index and coupling characteristics
        available_variations = list(self.variation_templates.keys())
        
        # Select appropriate variations based on coupling type
        if base_coupling.coupling_type == CouplingType.INFORMATIONAL:
            preferred_variations = ['strength_increase', 'type_adaptive', 'property_reliability']
        elif base_coupling.coupling_type == CouplingType.OPERATIONAL:
            preferred_variations = ['strength_increase', 'property_responsiveness', 'type_predictive']
        elif base_coupling.coupling_type == CouplingType.CONTEXTUAL:
            preferred_variations = ['property_precision', 'type_adaptive', 'strength_decrease']
        elif base_coupling.coupling_type == CouplingType.ADAPTIVE:
            preferred_variations = ['property_precision', 'strength_increase', 'property_reliability']
        elif base_coupling.coupling_type == CouplingType.PREDICTIVE:
            preferred_variations = ['property_precision', 'property_responsiveness', 'strength_increase']
        else:
            preferred_variations = available_variations
        
        # Ensure we have enough variations
        if len(preferred_variations) <= variation_index:
            # Fall back to random selection from all variations
            variation_type = random.choice(available_variations)
        else:
            variation_type = preferred_variations[variation_index]
        
        # Apply the selected variation
        template = self.variation_templates[variation_type]
        
        for attr, value_func in template.items():
            if attr != 'description':
                if callable(value_func):
                    variation[attr] = value_func(base_coupling)
                else:
                    variation[attr] = value_func
        
        variation['variation_type'] = variation_type
        variation['description'] = template.get('description', f'Variation {variation_index}')
        
        # Record creation in simulation history
        if not isinstance(base_coupling.id, str):
            coupling_id = str(base_coupling.id)
        else:
            coupling_id = base_coupling.id
            
        self.simulation_history[coupling_id].append({
            'timestamp': time.time(),
            'variation_id': variation['id'],
            'variation_type': variation_type,
            'description': variation['description']
        })
        
        # Limit history size
        if len(self.simulation_history[coupling_id]) > 50:
            self.simulation_history[coupling_id] = self.simulation_history[coupling_id][-50:]
        
        return variation
    
    async def _simulate_variation_outcome(self, variation):
        """Simulate the outcome of a coupling variation."""
        # Get the base coupling ID to check if we have a simulation model
        base_coupling_id = variation.get('base_coupling_id')
        model_confidence = 0.5  # Default moderate confidence
        
        if base_coupling_id in self.simulation_models:
            model = self.simulation_models[base_coupling_id]
            model_confidence = model['accuracy']
        
        # Create base outcome structure
        outcome = {
            'success_rate': np.random.uniform(0.7, 0.95),
            'efficiency': np.random.uniform(0.6, 0.9),
            'response_time': np.random.uniform(0.1, 2.0),
            'prediction_precision': np.random.uniform(0.5, 3.0),
            'reliability': np.random.uniform(0.7, 0.95),
            'adaptability': np.random.uniform(0.5, 0.9)
        }
        
        # Adjust based on variation type
        variation_type = variation.get('variation_type')
        
        if variation_type == 'strength_increase':
            outcome['success_rate'] += 0.05
            outcome['efficiency'] += 0.03
            outcome['response_time'] -= 0.2  # Faster response
            
        elif variation_type == 'strength_decrease':
            outcome['reliability'] += 0.05
            outcome['adaptability'] += 0.07
            outcome['efficiency'] -= 0.03
            
        elif variation_type == 'type_adaptive':
            outcome['adaptability'] += 0.1
            outcome['prediction_precision'] += 0.2
            outcome['efficiency'] -= 0.05
            
        elif variation_type == 'type_predictive':
            outcome['prediction_precision'] += 0.5
            outcome['response_time'] -= 0.3
            outcome['reliability'] -= 0.03
            
        elif variation_type == 'property_responsiveness':
            outcome['response_time'] -= 0.4
            outcome['efficiency'] += 0.08
            outcome['reliability'] -= 0.02
            
        elif variation_type == 'property_reliability':
            outcome['reliability'] += 0.08
            outcome['success_rate'] += 0.03
            outcome['adaptability'] -= 0.04
            
        elif variation_type == 'property_precision':
            outcome['prediction_precision'] += 0.7
            outcome['efficiency'] += 0.02
            outcome['response_time'] += 0.1  # Slower response
        
        # Add random variations based on model confidence
        # Lower confidence = more randomness
        randomness = 1.0 - model_confidence
        for key in outcome:
            outcome[key] += np.random.uniform(-randomness, randomness) * 0.1
        
        # Ensure values are in valid ranges
        outcome['success_rate'] = min(1.0, max(0.0, outcome['success_rate']))
        outcome['efficiency'] = min(1.0, max(0.0, outcome['efficiency']))
        outcome['reliability'] = min(1.0, max(0.0, outcome['reliability']))
        outcome['adaptability'] = min(1.0, max(0.0, outcome['adaptability']))
        outcome['response_time'] = max(0.1, outcome['response_time'])
        outcome['prediction_precision'] = max(0.1, outcome['prediction_precision'])
        
        # Add simulation metadata
        outcome['simulation_time'] = time.time()
        outcome['simulation_id'] = str(uuid.uuid4())
        outcome['model_confidence'] = model_confidence
        
        return outcome
    
    def _calculate_simulation_score(self, outcome):
        """Calculate an overall score for a simulation outcome."""
        if not outcome:
            return 0.0
        
        # Weight different metrics
        weights = {
            'success_rate': 0.25,
            'efficiency': 0.20,
            'response_time': 0.15,
            'prediction_precision': 0.15,
            'reliability': 0.15,
            'adaptability': 0.10
        }
        
        # Convert response time to score (lower is better)
        response_time_score = 1.0 - min(1.0, outcome.get('response_time', 0) / 3.0)
        
        # Calculate weighted score
        score = (
            weights['success_rate'] * outcome.get('success_rate', 0) +
            weights['efficiency'] * outcome.get('efficiency', 0) +
            weights['response_time'] * response_time_score +
            weights['prediction_precision'] * min(1.0, outcome.get('prediction_precision', 0) / 3.0) +
            weights['reliability'] * outcome.get('reliability', 0) +
            weights['adaptability'] * outcome.get('adaptability', 0)
        )
        
        return score
    
    def _calculate_improvement(self, best_result, all_results):
        """Calculate improvement of best result over average of all results."""
        if len(all_results) <= 1:
            return 0.0
        
        best_score = self._calculate_simulation_score(best_result['outcome'])
        
        # Calculate average score of all results except the best
        other_scores = [
            self._calculate_simulation_score(r['outcome'])
            for r in all_results if r != best_result
        ]
        
        avg_other_score = sum(other_scores) / len(other_scores) if other_scores else 0.0
        
        # Calculate relative improvement
        if avg_other_score > 0:
            improvement = (best_score - avg_other_score) / avg_other_score
        else:
            improvement = best_score
        
        return improvement



# === FILE: asf\layer4_environmental_coupling\components\coupling_registry.py ===

# === FILE: asf/environmental_coupling/components/coupling_registry.py ===
import asyncio
import time
import logging
from typing import Dict, List, Optional, Set
from collections import defaultdict

from asf.environmental_coupling.models import EnvironmentalCoupling
from asf.environmental_coupling.enums import CouplingState

class SparseCouplingRegistry:
    """
    Efficiently manages environmental couplings with sparse representation.
    Optimized for quick lookup by both internal and environmental entity IDs.
    """
    
    def __init__(self, initial_capacity=10000):
        self.couplings: Dict[str, EnvironmentalCoupling] = {}
        self.internal_entity_map: Dict[str, Set[str]] = defaultdict(set)
        self.environmental_entity_map: Dict[str, Set[str]] = defaultdict(set)
        self.lock = asyncio.Lock()
        self.capacity = initial_capacity
        self.logger = logging.getLogger("ASF.Layer4.SparseCouplingRegistry")
        
    async def initialize(self):
        """Initialize the registry."""
        self.logger.info(f"Initializing SparseCouplingRegistry with capacity {self.capacity}")
        return True
        
    async def add_coupling(self, coupling: EnvironmentalCoupling) -> bool:
        """Add a new coupling to the registry."""
        async with self.lock:
            if coupling.id in self.couplings:
                self.logger.warning(f"Coupling {coupling.id} already exists")
                return False
                
            self.couplings[coupling.id] = coupling
            self.internal_entity_map[coupling.internal_entity_id].add(coupling.id)
            self.environmental_entity_map[coupling.environmental_entity_id].add(coupling.id)
            
            self.logger.debug(f"Added coupling {coupling.id} between {coupling.internal_entity_id} and {coupling.environmental_entity_id}")
            return True
    
    async def get_coupling(self, coupling_id: str) -> Optional[EnvironmentalCoupling]:
        """Get a coupling by its ID."""
        return self.couplings.get(coupling_id)
        
    async def update_coupling(self, coupling: EnvironmentalCoupling) -> bool:
        """Update an existing coupling."""
        async with self.lock:
            if coupling.id not in self.couplings:
                self.logger.warning(f"Cannot update: Coupling {coupling.id} does not exist")
                return False
                
            # Check if entity relationships changed
            existing = self.couplings[coupling.id]
            if (existing.internal_entity_id != coupling.internal_entity_id or 
                existing.environmental_entity_id != coupling.environmental_entity_id):
                # Update mappings
                self.internal_entity_map[existing.internal_entity_id].discard(coupling.id)
                self.environmental_entity_map[existing.environmental_entity_id].discard(coupling.id)
                
                self.internal_entity_map[coupling.internal_entity_id].add(coupling.id)
                self.environmental_entity_map[coupling.environmental_entity_id].add(coupling.id)
            
            # Update the coupling
            self.couplings[coupling.id] = coupling
            self.logger.debug(f"Updated coupling {coupling.id}")
            return True
    
    async def delete_coupling(self, coupling_id: str) -> bool:
        """Delete a coupling."""
        async with self.lock:
            if coupling_id not in self.couplings:
                return False
                
            coupling = self.couplings[coupling_id]
            self.internal_entity_map[coupling.internal_entity_id].discard(coupling_id)
            self.environmental_entity_map[coupling.environmental_entity_id].discard(coupling_id)
            
            del self.couplings[coupling_id]
            self.logger.debug(f"Deleted coupling {coupling_id}")
            return True
    
    async def get_couplings_by_internal_entity(self, entity_id: str) -> List[EnvironmentalCoupling]:
        """Get all couplings for an internal entity."""
        coupling_ids = self.internal_entity_map.get(entity_id, set())
        return [self.couplings[cid] for cid in coupling_ids if cid in self.couplings]
    
    async def get_couplings_by_environmental_entity(self, entity_id: str) -> List[EnvironmentalCoupling]:
        """Get all couplings for an environmental entity."""
        coupling_ids = self.environmental_entity_map.get(entity_id, set())
        return [self.couplings[cid] for cid in coupling_ids if cid in self.couplings]
    
    async def get_active_couplings(self) -> List[EnvironmentalCoupling]:
        """Get all active couplings."""
        return [c for c in self.couplings.values() if c.coupling_state == CouplingState.ACTIVE]
    
    async def get_statistics(self) -> Dict:
        """Get registry statistics."""
        return {
            'total_couplings': len(self.couplings),
            'internal_entities': len(self.internal_entity_map),
            'environmental_entities': len(self.environmental_entity_map),
            'active_couplings': sum(1 for c in self.couplings.values() 
                                  if c.coupling_state == CouplingState.ACTIVE)
        }



# === FILE: asf\layer4_environmental_coupling\components\distributed_cache.py ===

# === FILE: asf/environmental_coupling/components/distributed_cache.py ===
import asyncio
import time
import logging
import random
from typing import Dict, List, Any, Optional, Tuple, Set

class DistributedCouplingCache:
    """
    Provides distributed caching for coupling data.
    Optimizes performance through multi-level caching with consistency guarantees.
    """
    
    def __init__(self, cache_config: Dict = None):
        self.config = cache_config or {
            'max_local_size': 10000,
            'max_shared_size': 100000,
            'default_ttl': 300,  # 5 minutes
            'consistency_check_interval': 60  # 1 minute
        }
        
        # Local cache (per-instance)
        self.local_cache = {}
        self.local_expiry = {}
        
        # Shared cache (simulated)
        # In a real implementation, this would use Redis, Memcached, etc.
        self.shared_cache = {}
        self.shared_expiry = {}
        
        # Tracking for eviction
        self.access_history = {}
        self.update_timestamps = {}
        
        # Lock for cache operations
        self.lock = asyncio.Lock()
        
        # Background task for maintenance
        self.maintenance_task = None
        self.running = False
        
        self.logger = logging.getLogger("ASF.Layer4.DistributedCouplingCache")
        
    async def initialize(self):
        """Initialize the cache service."""
        self.running = True
        self.maintenance_task = asyncio.create_task(self._maintenance_loop())
        self.logger.info(f"Initialized with max local size {self.config['max_local_size']}")
        return True
    
    async def get(self, key: str, use_shared: bool = True) -> Optional[Any]:
        """
        Get a value from the cache.
        Checks local cache first, then shared if enabled.
        """
        current_time = time.time()
        
        # Check local cache first
        if key in self.local_cache:
            expiry = self.local_expiry.get(key, 0)
            
            if expiry > current_time:
                # Update access record
                self.access_history[key] = current_time
                return self.local_cache[key]
            else:
                # Expired, remove from local cache
                await self._remove_from_local(key)
        
        # If not in local cache and shared cache is enabled, check there
        if use_shared and key in self.shared_cache:
            expiry = self.shared_expiry.get(key, 0)
            
            if expiry > current_time:
                # Found in shared cache, update local cache
                value = self.shared_cache[key]
                await self._add_to_local(key, value, expiry - current_time)
                return value
        
        # Not found or expired
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None, 
                use_shared: bool = True) -> bool:
        """
        Set a value in the cache with optional time-to-live.
        Updates both local and shared caches if enabled.
        """
        if ttl is None:
            ttl = self.config['default_ttl']
            
        current_time = time.time()
        expiry = current_time + ttl
        
        # Update local cache
        result = await self._add_to_local(key, value, ttl)
        
        # Update shared cache if enabled
        if use_shared:
            self.shared_cache[key] = value
            self.shared_expiry[key] = expiry
            self.update_timestamps[key] = current_time
            
            # Check shared cache size limits
            if len(self.shared_cache) > self.config['max_shared_size']:
                await self._evict_from_shared(10)  # Evict 10 items
        
        return result
    
    async def invalidate(self, key: str, use_shared: bool = True) -> bool:
        """
        Invalidate a cache entry.
        Removes from both local and shared caches if enabled.
        """
        result = await self._remove_from_local(key)
        
        # Also remove from shared cache if enabled
        if use_shared and key in self.shared_cache:
            del self.shared_cache[key]
            if key in self.shared_expiry:
                del self.shared_expiry[key]
            if key in self.update_timestamps:
                del self.update_timestamps[key]
                
            result = True
        
        return result
    
    async def invalidate_pattern(self, pattern: str, use_shared: bool = True) -> int:
        """
        Invalidate all cache entries matching a pattern.
        Returns the number of entries invalidated.
        """
        count = 0
        
        # Find matching keys
        local_matches = [k for k in self.local_cache if pattern in k]
        
        # Remove from local cache
        for key in local_matches:
            if await self._remove_from_local(key):
                count += 1
        
        # Also remove from shared cache if enabled
        if use_shared:
            shared_matches = [k for k in self.shared_cache if pattern in k]
            
            for key in shared_matches:
                if key in self.shared_cache:
                    del self.shared_cache[key]
                    if key in self.shared_expiry:
                        del self.shared_expiry[key]
                    if key in self.update_timestamps:
                        del self.update_timestamps[key]
                    count += 1
        
        return count
    
    async def _add_to_local(self, key: str, value: Any, ttl: int) -> bool:
        """Add a value to the local cache with expiration."""
        async with self.lock:
            # Check cache size limit
            if key not in self.local_cache and len(self.local_cache) >= self.config['max_local_size']:
                await self._evict_from_local(1)  # Make room
            
            current_time = time.time()
            self.local_cache[key] = value
            self.local_expiry[key] = current_time + ttl
            self.access_history[key] = current_time
            
            return True
    
    async def _remove_from_local(self, key: str) -> bool:
        """Remove a value from the local cache."""
        async with self.lock:
            if key in self.local_cache:
                del self.local_cache[key]
                if key in self.local_expiry:
                    del self.local_expiry[key]
                if key in self.access_history:
                    del self.access_history[key]
                return True
            return False
    
    async def _evict_from_local(self, count: int) -> int:
        """
        Evict entries from local cache using LRU policy.
        Returns the number of entries evicted.
        """
        if not self.access_history:
            return 0
            
        # Sort by last access time (oldest first)
        sorted_keys = sorted(
            self.access_history.keys(),
            key=lambda k: self.access_history[k]
        )
        
        # Take the oldest entries up to count
        to_evict = sorted_keys[:count]
        
        # Remove them
        evicted = 0
        for key in to_evict:
            if await self._remove_from_local(key):
                evicted += 1
                
        return evicted
    
    async def _evict_from_shared(self, count: int) -> int:
        """
        Evict entries from shared cache.
        In a real implementation, this might coordinate with other instances.
        """
        if not self.update_timestamps:
            return 0
            
        # Sort by last update time (oldest first)
        sorted_keys = sorted(
            self.update_timestamps.keys(),
            key=lambda k: self.update_timestamps[k]
        )
        
        # Take the oldest entries up to count
        to_evict = sorted_keys[:count]
        
        # Remove them
        evicted = 0
        for key in to_evict:
            if key in self.shared_cache:
                del self.shared_cache[key]
                if key in self.shared_expiry:
                    del self.shared_expiry[key]
                if key in self.update_timestamps:
                    del self.update_timestamps[key]
                evicted += 1
                
        return evicted
    
    async def _maintenance_loop(self):
        """Background task for cache maintenance."""
        try:
            while self.running:
                try:
                    # Clean expired entries from local cache
                    await self._clean_expired_local()
                    
                    # Clean expired entries from shared cache
                    await self._clean_expired_shared()
                    
                    # Check consistency with shared cache
                    await self._check_consistency()
                    
                    # Wait for next maintenance cycle
                    await asyncio.sleep(self.config['consistency_check_interval'])
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Error in cache maintenance: {str(e)}")
                    await asyncio.sleep(5)  # Wait before retry
                    
        finally:
            self.running = False
    
    async def _clean_expired_local(self) -> int:
        """Clean expired entries from local cache."""
        current_time = time.time()
        to_remove = []
        
        async with self.lock:
            for key, expiry in self.local_expiry.items():
                if expiry <= current_time:
                    to_remove.append(key)
                    
            # Remove expired entries
            for key in to_remove:
                await self._remove_from_local(key)
                
            return len(to_remove)
    
    async def _clean_expired_shared(self) -> int:
        """Clean expired entries from shared cache."""
        current_time = time.time()
        to_remove = []
        
        for key, expiry in self.shared_expiry.items():
            if expiry <= current_time:
                to_remove.append(key)
                
        # Remove expired entries
        for key in to_remove:
            if key in self.shared_cache:
                del self.shared_cache[key]
            if key in self.shared_expiry:
                del self.shared_expiry[key]
            if key in self.update_timestamps:
                del self.update_timestamps[key]
                
        return len(to_remove)
    
    async def _check_consistency(self) -> Dict:
        """
        Check consistency between local and shared caches.
        Synchronizes when needed.
        """
        # In a real implementation, this would check with a distributed
        # cache system like Redis for newer versions of locally cached items
        
        # For this simulation, we'll just check a few random keys
        if not self.local_cache or not self.shared_cache:
            return {'checked': 0, 'updated': 0}
            
        # Select random keys from local cache
        sample_size = min(10, len(self.local_cache))
        sample_keys = random.sample(list(self.local_cache.keys()), sample_size)
        
        updated = 0
        
        for key in sample_keys:
            # Check if key exists in shared cache and is newer
            if (key in self.shared_cache and key in self.update_timestamps and
                key in self.local_expiry and 
                self.update_timestamps[key] > self.access_history.get(key, 0)):
                
                # Shared version is newer, update local
                self.local_cache[key] = self.shared_cache[key]
                self.local_expiry[key] = self.shared_expiry[key]
                self.access_history[key] = time.time()
                updated += 1
        
        return {'checked': sample_size, 'updated': updated}
    
    async def stop(self):
        """Stop the cache service."""
        self.running = False
        if self.maintenance_task:
            self.maintenance_task.cancel()
            try:
                await self.maintenance_task
            except asyncio.CancelledError:
                pass
                
        self.logger.info("Cache service stopped")
        return True
    
    async def get_metrics(self) -> Dict:
        """Get cache metrics."""
        return {
            'local_entries': len(self.local_cache),
            'shared_entries': len(self.shared_cache),
            'local_hit_ratio': self.local_hit_ratio if hasattr(self, 'local_hit_ratio') else 0,
            'memory_usage': self._estimate_memory_usage()
        }
    
    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage of the cache in bytes."""
        # This is a very rough estimation
        # In a real implementation, you would use sys.getsizeof or similar
        
        # Assume average key size of 50 bytes and value size of 500 bytes
        local_estimate = len(self.local_cache) * 550
        
        return local_estimate



# === FILE: asf\layer4_environmental_coupling\components\enhanced_bayesian_updater.py ===

# === FILE: asf/environmental_coupling/components/enhanced_bayesian_updater.py ===

import asyncio
import time
import uuid
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict

class EnhancedBayesianUpdater:
    """
    Enhanced Bayesian updater with precision-weighted prediction errors.
    Implements Seth's principle of weighting updates by precision.
    """
    def __init__(self):
        self.coupling_models = {}  # Maps coupling_id to Bayesian models
        self.update_history = defaultdict(list)  # Maps coupling_id to update history
        self.precision_values = {}  # Maps coupling_id to precision value
        self.logger = logging.getLogger("ASF.Layer4.EnhancedBayesianUpdater")
        
    async def initialize_coupling_model(self, coupling_id):
        """Initialize Bayesian model for a new coupling."""
        self.coupling_models[coupling_id] = {
            'prior_confidence': 0.5,
            'likelihood_estimates': {},
            'last_updated': time.time(),
            'update_count': 0
        }
        self.precision_values[coupling_id] = 1.0  # Default precision
        return True
        
    async def update_from_interaction(self, coupling_id, interaction_data, interaction_type, confidence):
        """
        Update coupling Bayesian model with precision weighting.
        Implements Seth's precision-weighted error principle.
        """
        start_time = time.time()
        
        if coupling_id not in self.coupling_models:
            await self.initialize_coupling_model(coupling_id)
            
        model = self.coupling_models[coupling_id]
        
        # Extract key features from interaction data
        features = self._extract_features(interaction_data)
        
        # Get current prior confidence
        prior = model['prior_confidence']
        
        # Calculate likelihood based on feature match
        likelihood = self._calculate_likelihood(features, model, interaction_type)
        
        # Get precision for this coupling
        precision = self.precision_values.get(coupling_id, 1.0)
        
        # Weight confidence by precision (Seth's principle)
        weighted_confidence = confidence * precision
        
        # Apply Bayes' rule with weighted confidence
        posterior = self._apply_bayes_rule(prior, likelihood, weighted_confidence)
        
        # Determine effect on coupling strength
        strength_delta = (posterior - prior) * 0.2  # Scale factor
        
        # Update model
        model['prior_confidence'] = posterior
        model['last_updated'] = time.time()
        model['update_count'] += 1
        
        # Update likelihood estimates
        if interaction_type not in model['likelihood_estimates']:
            model['likelihood_estimates'][interaction_type] = {}
            
        for feature, value in features.items():
            feature_key = f"{feature}:{value}"
            if feature_key not in model['likelihood_estimates'][interaction_type]:
                model['likelihood_estimates'][interaction_type][feature_key] = 0.5
                
            # Update likelihood estimate using exponential moving average
            current_estimate = model['likelihood_estimates'][interaction_type][feature_key]
            update_weight = 0.2  # Learning rate
            new_estimate = (1 - update_weight) * current_estimate + update_weight * weighted_confidence
            model['likelihood_estimates'][interaction_type][feature_key] = new_estimate
        
        # Check for contradictions in the data
        contradiction = self._check_for_contradictions(interaction_data, model)
        
        # Add to update history
        self.update_history[coupling_id].append({
            'timestamp': time.time(),
            'interaction_type': interaction_type,
            'prior': prior,
            'likelihood': likelihood,
            'posterior': posterior,
            'features': len(features),
            'weighted_confidence': weighted_confidence,
            'precision': precision,
            'contradiction_detected': contradiction is not None
        })
        
        # Limit history size
        if len(self.update_history[coupling_id]) > 50:
            self.update_history[coupling_id] = self.update_history[coupling_id][-50:]
        
        return {
            'prior_confidence': prior,
            'new_confidence': posterior,
            'likelihood': likelihood,
            'strength_delta': strength_delta,
            'precision_applied': precision,
            'contradiction_detected': contradiction is not None,
            'contradiction': contradiction,
            'features_processed': len(features),
            'elapsed_time': time.time() - start_time
        }
    
    async def update_precision(self, coupling_id, prediction_error):
        """
        Update precision based on prediction error.
        Implements Seth's precision updating mechanism.
        """
        if coupling_id not in self.precision_values:
            self.precision_values[coupling_id] = 1.0
            
        # In a real implementation, this would use a more sophisticated
        # approach like Kalman filtering or variational inference
        # For this example, we use a simple inverse variance calculation
        
        # Track error history
        if coupling_id not in self.update_history:
            self.update_history[coupling_id] = []
            
        # Add error to history
        self.update_history[coupling_id].append({
            'timestamp': time.time(),
            'prediction_error': prediction_error
        })
        
        # Limit history size
        if len(self.update_history[coupling_id]) > 20:
            self.update_history[coupling_id] = self.update_history[coupling_id][-20:]
            
        # Calculate precision as inverse variance of recent errors
        recent_errors = [update.get('prediction_error', 0) 
                         for update in self.update_history[coupling_id]
                         if 'prediction_error' in update]
        
        if len(recent_errors) > 1:
            variance = np.var(recent_errors)
            # Avoid division by zero and excessive weights
            new_precision = min(10.0, 1.0 / (variance + 1e-6))
            self.precision_values[coupling_id] = new_precision
            
        return self.precision_values[coupling_id]
        
    async def get_precision(self, coupling_id):
        """Get current precision for a coupling."""
        return self.precision_values.get(coupling_id, 1.0)
    
    async def cleanup_coupling_model(self, coupling_id):
        """Clean up resources for a terminated coupling."""
        if coupling_id in self.coupling_models:
            del self.coupling_models[coupling_id]
        if coupling_id in self.update_history:
            del self.update_history[coupling_id]
        if coupling_id in self.precision_values:
            del self.precision_values[coupling_id]
        return True
        
    async def perform_maintenance(self):
        """Perform periodic maintenance on Bayesian models."""
        start_time = time.time()
        
        # Count models
        model_count = len(self.coupling_models)
        
        # Identify inactive models
        current_time = time.time()
        inactive_threshold = 86400  # 24 hours
        inactive_models = [
            coupling_id for coupling_id, model in self.coupling_models.items()
            if current_time - model['last_updated'] > inactive_threshold
        ]
        
        # Compact history for all models
        for coupling_id in self.update_history:
            if len(self.update_history[coupling_id]) > 50:
                self.update_history[coupling_id] = self.update_history[coupling_id][-50:]
        
        return {
            'model_count': model_count,
            'inactive_models': len(inactive_models),
            'precision_values': len(self.precision_values),
            'elapsed_time': time.time() - start_time
        }
        
    async def get_metrics(self):
        """Get metrics about the enhanced Bayesian updater."""
        return {
            'model_count': len(self.coupling_models),
            'avg_update_count': np.mean([model['update_count'] for model in self.coupling_models.values()]) if self.coupling_models else 0,
            'avg_precision': np.mean(list(self.precision_values.values())) if self.precision_values else 1.0,
            'max_precision': max(self.precision_values.values()) if self.precision_values else 1.0,
            'history_entries': sum(len(history) for history in self.update_history.values())
        }
    
    def _extract_features(self, interaction_data):
        """Extract features from interaction data."""
        features = {}
        
        if not isinstance(interaction_data, dict):
            return features
            
        # Extract basic features
        for key, value in interaction_data.items():
            if isinstance(value, (str, int, float, bool)):
                features[key] = str(value)
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, (str, int, float, bool)):
                        features[f"{key}_{subkey}"] = str(subvalue)
        
        return features
    
    def _calculate_likelihood(self, features, model, interaction_type):
        """Calculate likelihood based on feature match to existing model."""
        if interaction_type not in model['likelihood_estimates']:
            return 0.5  # Default for unknown interaction type
            
        likelihood_estimates = model['likelihood_estimates'][interaction_type]
        
        # Calculate likelihood from matched features
        matched_likelihoods = []
        
        for feature, value in features.items():
            feature_key = f"{feature}:{value}"
            if feature_key in likelihood_estimates:
                matched_likelihoods.append(likelihood_estimates[feature_key])
                
        # If no matches found, use default
        if not matched_likelihoods:
            return 0.5
            
        # Use mean of matched likelihoods
        return np.mean(matched_likelihoods)
    
    def _apply_bayes_rule(self, prior, likelihood, confidence):
        """Apply Bayes' rule to update belief with confidence weighting."""
        # Confidence-weighted Bayesian update
        if likelihood > 0.5:  # Evidence supports the model
            support = likelihood * confidence
            posterior = prior + (1 - prior) * support * 0.5
        else:  # Evidence against the model
            contradiction = (1 - likelihood) * confidence
            posterior = prior - prior * contradiction * 0.5
            
        # Ensure posterior is in valid range
        return max(0.1, min(0.95, posterior))
    
    def _check_for_contradictions(self, interaction_data, model):
        """Check for contradictions in interaction data compared to model."""
        # Simplified contradiction detection
        if not isinstance(interaction_data, dict):
            return None
            
        # If confidence is high but interaction data contradicts model
        if model['prior_confidence'] > 0.7:
            features = self._extract_features(interaction_data)
            
            for interaction_type, estimates in model['likelihood_estimates'].items():
                matched_likelihoods = []
                
                for feature, value in features.items():
                    feature_key = f"{feature}:{value}"
                    if feature_key in estimates:
                        matched_likelihoods.append(estimates[feature_key])
                
                if matched_likelihoods and np.mean(matched_likelihoods) < 0.3:
                    # This is a potential contradiction
                    return {
                        'type': 'likelihood_contradiction',
                        'interaction_type': interaction_type,
                        'prior_confidence': model['prior_confidence'],
                        'contradiction_likelihood': np.mean(matched_likelihoods),
                        'features': features
                    }
                    
        return None



# === FILE: asf\layer4_environmental_coupling\components\event_processor.py ===

import asyncio
import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

class EventDrivenProcessor:
    """
    Optimized event processor with multi-stream parallel processing.
    Implements high-throughput event handling through concurrent processing streams.
    """
    def __init__(self, max_concurrency=16, stream_count=8):
        self.max_concurrency = max_concurrency
        self.stream_count = stream_count  # Number of parallel processing streams
        self.processing_semaphore = asyncio.Semaphore(max_concurrency)
        self.event_callback = None
        self.running = False
        self.stream_queues = [asyncio.Queue() for _ in range(stream_count)]  # Multiple streams
        self.stream_workers = []
        self.stream_stats = [{'processed': 0, 'average_time': 0.0} for _ in range(stream_count)]
        self.logger = logging.getLogger("ASF.Layer4.EventDrivenProcessor")
        
    async def initialize(self, callback):
        """Initialize with event processing callback."""
        self.event_callback = callback
        self.running = True
        self.logger.info(f"Initialized event processor with {self.stream_count} parallel streams")
        return True
        
    async def submit_event(self, event):
        """Submit an event for processing using adaptive stream selection."""
        if not self.running:
            return False
            
        # Select optimal stream based on load balancing
        stream_index = await self._select_optimal_stream(event)
        
        # Add event to selected stream queue
        await self.stream_queues[stream_index].put(event)
        return True
        
    async def _select_optimal_stream(self, event):
        """Select the optimal stream for an event based on current load and event type."""
        # Simple strategy: choose the stream with the shortest queue
        queue_sizes = [q.qsize() for q in self.stream_queues]
        
        # Consider event priority for queue selection
        if hasattr(event, 'priority') and event.priority > 0.7:
            # For high priority events, use the least loaded queue
            return queue_sizes.index(min(queue_sizes))
            
        # For regular events, use modulo by entity_id if available for affinity
        if hasattr(event, 'entity_id') and event.entity_id:
            # Consistent hashing for entity affinity (keeps related events on same stream)
            entity_hash = hash(event.entity_id) % self.stream_count
            
            # Only use this stream if it's not overloaded
            if queue_sizes[entity_hash] < 1.5 * min(queue_sizes):
                return entity_hash
                
        # Default to least loaded queue
        return queue_sizes.index(min(queue_sizes))
        
    async def run_processing_loop(self):
        """Start processing loops for all streams."""
        # Create and start a worker for each stream
        self.stream_workers = [
            asyncio.create_task(self._stream_processing_loop(i))
            for i in range(self.stream_count)
        ]
        
        self.logger.info(f"Started {self.stream_count} parallel processing streams")
        return self.stream_workers
        
    async def _stream_processing_loop(self, stream_index):
        """Processing loop for a single stream."""
        queue = self.stream_queues[stream_index]
        stats = self.stream_stats[stream_index]
        
        self.logger.info(f"Started processing loop for stream {stream_index}")
        
        while self.running:
            try:
                # Get next event
                event = await queue.get()
                
                # Process event with semaphore to limit total concurrency
                async with self.processing_semaphore:
                    start_time = time.time()
                    
                    # Process the event
                    if self.event_callback:
                        try:
                            result = await self.event_callback(event)
                            # Record successful processing
                            if hasattr(event, 'processed'):
                                event.processed = True
                            if hasattr(event, 'result'):
                                event.result = result
                        except Exception as e:
                            # Handle errors during event processing
                            self.logger.error(f"Error processing event {getattr(event, 'id', 'unknown')}: {str(e)}")
                            if hasattr(event, 'processed'):
                                event.processed = False
                            if hasattr(event, 'result'):
                                event.result = {'error': str(e), 'processed': False}
                    
                    processing_time = time.time() - start_time
                    
                    # Update statistics with exponential moving average
                    stats['processed'] += 1
                    stats['average_time'] = (stats['average_time'] * 0.95) + (processing_time * 0.05)
                    
                    # Log slow events
                    if processing_time > 1.0:  # More than 1 second
                        self.logger.warning(f"Slow event processing: {getattr(event, 'event_type', 'unknown')} took {processing_time:.2f}s")
                        
                # Mark task as done
                queue.task_done()
                
            except asyncio.CancelledError:
                self.logger.info(f"Stream {stream_index} processing loop cancelled")
                break
                
            except Exception as e:
                self.logger.error(f"Error in stream {stream_index} processing loop: {str(e)}")
                # Continue processing despite errors
                
        self.logger.info(f"Stream {stream_index} processing loop exited")
        
    async def perform_maintenance(self):
        """Perform periodic maintenance tasks."""
        # Calculate overall metrics
        total_processed = sum(stats['processed'] for stats in self.stream_stats)
        avg_times = [stats['average_time'] for stats in self.stream_stats]
        
        # Balance queues if needed
        await self._balance_queues()
        
        return {
            'stream_count': self.stream_count,
            'events_processed': total_processed,
            'stream_load': [q.qsize() for q in self.stream_queues],
            'avg_processing_times': avg_times
        }
        
    async def _balance_queues(self):
        """Balance workload across queues if imbalanced."""
        queue_sizes = [q.qsize() for q in self.stream_queues]
        max_size = max(queue_sizes)
        min_size = min(queue_sizes)
        
        # If imbalance is significant
        if max_size > min_size * 2 and max_size > 10:
            source_idx = queue_sizes.index(max_size)
            target_idx = queue_sizes.index(min_size)
            
            # Move some events (up to half of the difference)
            moves = (max_size - min_size) // 2
            
            for _ in range(moves):
                if not self.stream_queues[source_idx].empty():
                    event = await self.stream_queues[source_idx].get()
                    await self.stream_queues[target_idx].put(event)
                    self.stream_queues[source_idx].task_done()
                    
            self.logger.info(f"Balanced queues by moving {moves} events from stream {source_idx} to {target_idx}")
            
    async def get_metrics(self):
        """Get processor metrics."""
        return {
            'stream_count': self.stream_count,
            'events_processed': [stats['processed'] for stats in self.stream_stats],
            'avg_processing_times': [stats['average_time'] for stats in self.stream_stats],
            'current_queue_sizes': [q.qsize() for q in self.stream_queues]
        }
        
    async def stop(self):
        """Stop processing and shut down workers gracefully."""
        self.running = False
        
        # Cancel all worker tasks
        for worker in self.stream_workers:
            worker.cancel()
            
        # Wait for workers to complete
        if self.stream_workers:
            await asyncio.gather(*self.stream_workers, return_exceptions=True)
            
        self.logger.info("Event processor stopped")
        return True



# === FILE: asf\layer4_environmental_coupling\components\gpu_accelerator.py ===

# === FILE: asf/environmental_coupling/components/gpu_accelerator.py ===
import asyncio
import time
import logging
import random
from typing import Dict, List, Any, Optional, Tuple

class GPUAccelerationManager:
    """
    Manages GPU resources for accelerating environmental coupling operations.
    Optimizes resource allocation for predictive modeling and active inference.
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.available_devices = []
        self.device_utilization = {}
        self.operation_timings = {}
        self.lock = asyncio.Lock()
        self.logger = logging.getLogger("ASF.Layer4.GPUAccelerationManager")
        
    async def initialize(self):
        """Initialize the GPU manager and detect available devices."""
        if not self.enabled:
            self.logger.info("GPU acceleration disabled")
            return {'status': 'disabled'}
            
        # This is a simplified placeholder for actual GPU detection
        # In a real implementation, this would use libraries like torch.cuda, tf.config, etc.
        try:
            # Simulate GPU detection
            detected_devices = self._detect_gpus()
            
            if detected_devices:
                self.available_devices = detected_devices
                for device in self.available_devices:
                    self.device_utilization[device['id']] = 0.0
                
                self.logger.info(f"Initialized with {len(self.available_devices)} GPU devices")
                return {
                    'status': 'initialized',
                    'devices': len(self.available_devices),
                    'total_memory': sum(d['memory'] for d in self.available_devices)
                }
            else:
                self.enabled = False
                self.logger.warning("No GPU devices detected, acceleration disabled")
                return {'status': 'disabled', 'reason': 'no_devices'}
                
        except Exception as e:
            self.enabled = False
            self.logger.error(f"Error initializing GPU manager: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    def _detect_gpus(self) -> List[Dict]:
        """
        Detect available GPU devices.
        This is a simplified placeholder for actual detection logic.
        """
        # In a real implementation, this would use CUDA/ROCm APIs
        # to detect actual GPU devices and their properties
        
        # For simulation purposes, we'll just return mock devices
        # based on some reasonable assumptions
        devices = []
        
        # Simulate a 30% chance of having a GPU
        if random.random() < 0.3:
            # Simulate 1-2 GPUs
            gpu_count = random.randint(1, 2)
            
            for i in range(gpu_count):
                # Simulate different GPU memory sizes (8-32 GB)
                memory_gb = random.choice([8, 16, 24, 32])
                
                devices.append({
                    'id': f'gpu:{i}',
                    'name': f'Simulated GPU {i}',
                    'memory': memory_gb * 1024,  # Convert to MB
                    'compute_capability': random.choice([7.0, 7.5, 8.0, 8.6])
                })
        
        return devices
    
    async def allocate_resources(self, operation_type: str, 
                                 resource_requirements: Dict) -> Dict:
        """
        Allocate GPU resources for an operation if available.
        Returns device information or fallback CPU configuration.
        """
        if not self.enabled or not self.available_devices:
            return {'device': 'cpu', 'acceleration': False}
            
        async with self.lock:
            # Find least utilized device
            utilization = [(device_id, self.device_utilization[device_id]) 
                         for device_id in self.device_utilization]
            utilization.sort(key=lambda x: x[1])  # Sort by utilization
            
            # Get the least utilized device
            device_id, current_util = utilization[0]
            
            # Check if device has enough capacity
            required_memory = resource_requirements.get('memory_mb', 1024)
            device = next((d for d in self.available_devices if d['id'] == device_id), None)
            
            if device and current_util < 0.9:  # Allow up to 90% utilization
                # Update utilization (simplified)
                new_util = min(1.0, current_util + (required_memory / device['memory']))
                self.device_utilization[device_id] = new_util
                
                # Record start time for this operation
                operation_id = f"{operation_type}_{time.time()}"
                self.operation_timings[operation_id] = {
                    'start_time': time.time(),
                    'device_id': device_id,
                    'operation_type': operation_type,
                    'memory_required': required_memory
                }
                
                self.logger.debug(f"Allocated GPU {device_id} for {operation_type}, "
                                 f"utilization now {new_util:.2f}")
                
                return {
                    'device': device_id,
                    'acceleration': True,
                    'operation_id': operation_id,
                    'device_info': device
                }
            else:
                # No suitable GPU available, fallback to CPU
                self.logger.debug(f"No suitable GPU for {operation_type}, using CPU")
                return {'device': 'cpu', 'acceleration': False}
    
    async def release_resources(self, operation_id: str) -> bool:
        """Release allocated GPU resources after operation completes."""
        if not self.enabled or operation_id not in self.operation_timings:
            return False
            
        async with self.lock:
            timing_info = self.operation_timings.pop(operation_id)
            device_id = timing_info['device_id']
            
            # Record operation timing
            duration = time.time() - timing_info['start_time']
            
            # Update device utilization (simplified)
            required_memory = timing_info['memory_required']
            device = next((d for d in self.available_devices if d['id'] == device_id), None)
            
            if device:
                current_util = self.device_utilization[device_id]
                # Decrease utilization proportionally
                new_util = max(0.0, current_util - (required_memory / device['memory']))
                self.device_utilization[device_id] = new_util
                
                self.logger.debug(f"Released GPU {device_id}, "
                                 f"utilization now {new_util:.2f}, "
                                 f"operation took {duration:.3f}s")
            
            return True
    
    async def accelerate_tensor_operation(self, operation_type: str, 
                                        tensor_data: Any,
                                        operation_params: Dict) -> Tuple[Any, Dict]:
        """
        Accelerate a tensor operation using available GPU resources.
        Returns result and performance metrics.
        """
        if not self.enabled or not self.available_devices:
            # Execute on CPU (simplified)
            start_time = time.time()
            result = self._simulate_cpu_operation(operation_type, tensor_data, operation_params)
            duration = time.time() - start_time
            
            return result, {
                'device': 'cpu',
                'duration': duration,
                'acceleration': False
            }
            
        # Estimate resource requirements
        # In a real implementation, this would be based on tensor sizes
        memory_required = 1024  # 1GB placeholder
        
        # Allocate resources
        allocation = await self.allocate_resources(operation_type, {'memory_mb': memory_required})
        
        try:
            start_time = time.time()
            
            if allocation['acceleration']:
                # Execute on GPU (simplified)
                result = self._simulate_gpu_operation(
                    operation_type, tensor_data, operation_params, allocation
                )
            else:
                # Fallback to CPU
                result = self._simulate_cpu_operation(
                    operation_type, tensor_data, operation_params
                )
                
            duration = time.time() - start_time
            
            return result, {
                'device': allocation.get('device', 'cpu'),
                'duration': duration,
                'acceleration': allocation.get('acceleration', False),
                'operation_type': operation_type
            }
            
        finally:
            # Release resources if allocated
            if allocation.get('acceleration') and 'operation_id' in allocation:
                await self.release_resources(allocation['operation_id'])
    
    def _simulate_cpu_operation(self, operation_type: str, 
                              tensor_data: Any, 
                              operation_params: Dict) -> Any:
        """Simulate a CPU tensor operation (placeholder)."""
        # This would be replaced with actual tensor operations in a real implementation
        
        # Simulate processing time based on data size
        data_size = self._estimate_data_size(tensor_data)
        processing_time = 0.001 * data_size  # 1ms per KB
        time.sleep(processing_time)
        
        return tensor_data  # Placeholder result
    
    def _simulate_gpu_operation(self, operation_type: str, 
                              tensor_data: Any, 
                              operation_params: Dict,
                              allocation: Dict) -> Any:
        """Simulate a GPU tensor operation (placeholder)."""
        # This would be replaced with actual GPU tensor operations in a real implementation
        
        # Simulate faster processing time based on data size and device
        data_size = self._estimate_data_size(tensor_data)
        device_info = allocation.get('device_info', {})
        
        # Assume GPUs are 5-10x faster than CPU
        speedup = 5.0
        if 'compute_capability' in device_info:
            # Higher compute capability = faster
            speedup += device_info['compute_capability']
            
        processing_time = (0.001 * data_size) / speedup
        time.sleep(processing_time)
        
        return tensor_data  # Placeholder result
    
    def _estimate_data_size(self, tensor_data: Any) -> float:
        """Estimate data size in KB (placeholder)."""
        # In a real implementation, this would calculate actual tensor memory usage
        # For this simplified version, we'll just generate a random size
        return random.uniform(10, 1000)  # 10KB to 1MB
    
    async def get_metrics(self) -> Dict:
        """Get GPU manager metrics."""
        if not self.enabled:
            return {'enabled': False, 'devices': 0}
            
        return {
            'enabled': self.enabled,
            'devices': len(self.available_devices),
            'average_utilization': np.mean(list(self.device_utilization.values())) 
                                   if self.device_utilization else 0.0,
            'active_operations': len(self.operation_timings)
        }



# === FILE: asf\layer4_environmental_coupling\components\metrics_collector.py ===

# === FILE: asf/environmental_coupling/components/metrics_collector.py ===
import asyncio
import time
import logging
import json
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict

class PerformanceMetricsCollector:
    """
    Collects and analyzes performance metrics for the Environmental Coupling Layer.
    Enables optimization through data-driven insights.
    """
    
    def __init__(self):
        self.operation_timings = defaultdict(list)  # Maps operation type to timing list
        self.entity_metrics = defaultdict(dict)     # Maps entity_id to metric dict
        self.global_counters = defaultdict(int)     # Global counters for events
        self.recent_events = []                     # Queue of recent events
        self.start_time = time.time()
        self.last_reset = time.time()
        self.lock = asyncio.Lock()
        self.logger = logging.getLogger("ASF.Layer4.PerformanceMetricsCollector")
        
    async def record_operation(self, operation_type: str, duration: float, 
                             metadata: Optional[Dict] = None) -> None:
        """Record timing for an operation."""
        async with self.lock:
            # Record the timing
            self.operation_timings[operation_type].append({
                'timestamp': time.time(),
                'duration': duration,
                'metadata': metadata or {}
            })
            
            # Limit number of timings stored per operation
            if len(self.operation_timings[operation_type]) > 1000:
                self.operation_timings[operation_type] = self.operation_timings[operation_type][-1000:]
            
            # Update global counter
            self.global_counters[f"operation.{operation_type}"] += 1
            
            # Add to recent events
            self.recent_events.append({
                'type': 'operation',
                'operation_type': operation_type,
                'duration': duration,
                'timestamp': time.time()
            })
            
            # Limit recent events
            if len(self.recent_events) > 100:
                self.recent_events = self.recent_events[-100:]
    
    async def record_entity_event(self, entity_id: str, event_type: str,
                                data: Optional[Dict] = None) -> None:
        """Record an event for a specific entity."""
        async with self.lock:
            # Initialize entity metrics if needed
            if entity_id not in self.entity_metrics:
                self.entity_metrics[entity_id] = {
                    'first_seen': time.time(),
                    'event_count': 0,
                    'event_types': defaultdict(int),
                    'last_seen': time.time()
                }
            
            # Update entity metrics
            metrics = self.entity_metrics[entity_id]
            metrics['event_count'] += 1
            metrics['event_types'][event_type] += 1
            metrics['last_seen'] = time.time()
            
            # Store additional data if provided
            if data:
                if 'recent_data' not in metrics:
                    metrics['recent_data'] = []
                    
                metrics['recent_data'].append({
                    'timestamp': time.time(),
                    'event_type': event_type,
                    'data': data
                })
                
                # Limit recent data
                if len(metrics['recent_data']) > 10:
                    metrics['recent_data'] = metrics['recent_data'][-10:]
            
            # Update global counter
            self.global_counters[f"entity_event.{event_type}"] += 1
            
            # Add to recent events
            self.recent_events.append({
                'type': 'entity_event',
                'entity_id': entity_id,
                'event_type': event_type,
                'timestamp': time.time()
            })
            
            # Limit recent events
            if len(self.recent_events) > 100:
                self.recent_events = self.recent_events[-100:]
    
    async def increment_counter(self, counter_name: str, value: int = 1) -> int:
        """Increment a custom counter."""
        async with self.lock:
            self.global_counters[counter_name] += value
            return self.global_counters[counter_name]
    
    async def get_operation_stats(self, operation_type: str = None) -> Dict:
        """Get statistics for operations."""
        if operation_type:
            # Stats for specific operation type
            timings = self.operation_timings.get(operation_type, [])
            
            if not timings:
                return {
                    'operation_type': operation_type,
                    'count': 0,
                    'found': False
                }
                
            durations = [t['duration'] for t in timings]
            
            return {
                'operation_type': operation_type,
                'count': len(timings),
                'avg_duration': sum(durations) / len(durations) if durations else 0,
                'min_duration': min(durations) if durations else 0,
                'max_duration': max(durations) if durations else 0,
                'recent_duration': durations[-1] if durations else 0,
                'found': True
            }
        else:
            # Stats for all operation types
            stats = {}
            
            for op_type in self.operation_timings:
                stats[op_type] = await self.get_operation_stats(op_type)
                
            return stats
    
    async def get_entity_stats(self, entity_id: str = None) -> Dict:
        """Get statistics for entities."""
        if entity_id:
            # Stats for specific entity
            if entity_id not in self.entity_metrics:
                return {
                    'entity_id': entity_id,
                    'found': False
                }
                
            metrics = self.entity_metrics[entity_id]
            
            # Calculate active period
            active_period = metrics['last_seen'] - metrics['first_seen']
            
            # Calculate event frequency (events per hour)
            event_frequency = (metrics['event_count'] / active_period) * 3600 if active_period > 0 else 0
            
            return {
                'entity_id': entity_id,
                'event_count': metrics['event_count'],
                'first_seen': metrics['first_seen'],
                'last_seen': metrics['last_seen'],
                'active_period': active_period,
                'event_frequency': event_frequency,
                'event_types': dict(metrics['event_types']),
                'found': True
            }
        else:
            # Summary stats for all entities
            entity_count = len(self.entity_metrics)
            
            if entity_count == 0:
                return {
                    'entity_count': 0,
                    'total_events': 0
                }
                
            total_events = sum(m['event_count'] for m in self.entity_metrics.values())
            active_entities = sum(1 for m in self.entity_metrics.values() 
                                if time.time() - m['last_seen'] < 3600)  # Active in last hour
            
            return {
                'entity_count': entity_count,
                'total_events': total_events,
                'active_entities': active_entities,
                'avg_events_per_entity': total_events / entity_count if entity_count > 0 else 0
            }
    
    async def get_counters(self, pattern: str = None) -> Dict:
        """Get counter values, optionally filtered by pattern."""
        if pattern:
            # Filter counters by pattern
            return {
                name: value for name, value in self.global_counters.items()
                if pattern in name
            }
        else:
            # Return all counters
            return dict(self.global_counters)
    
    async def get_recent_events(self, limit: int = 100) -> List[Dict]:
        """Get the most recent events."""
        return self.recent_events[-limit:]
    
    async def reset_metrics(self, keep_history: bool = False) -> Dict:
        """
        Reset metrics collectors.
        If keep_history is True, keeps historical data but resets counters.
        """
        async with self.lock:
            old_counters = dict(self.global_counters)
            
            if keep_history:
                # Just reset counters
                self.global_counters = defaultdict(int)
            else:
                # Reset everything
                self.operation_timings = defaultdict(list)
                self.entity_metrics = defaultdict(dict)
                self.global_counters = defaultdict(int)
                self.recent_events = []
            
            self.last_reset = time.time()
            
            return {
                'reset_time': self.last_reset,
                'previous_counters': old_counters,
                'kept_history': keep_history
            }
    
    async def export_metrics(self, format: str = 'json') -> str:
        """Export metrics in the specified format."""
        # Collect all metrics
        metrics = {
            'timestamp': time.time(),
            'uptime': time.time() - self.start_time,
            'since_reset': time.time() - self.last_reset,
            'counters': dict(self.global_counters),
            'entity_summary': await self.get_entity_stats(),
            'recent_events_count': len(self.recent_events)
        }
        
        # Format the metrics
        if format.lower() == 'json':
            return json.dumps(metrics, indent=2)
        elif format.lower() == 'text':
            # Simple text format
            lines = [
                f"Timestamp: {metrics['timestamp']}",
                f"Uptime: {metrics['uptime']:.2f} seconds",
                f"Since reset: {metrics['since_reset']:.2f} seconds",
                "Counters:",
            ]
            
            for name, value in sorted(metrics['counters'].items()):
                lines.append(f"  {name}: {value}")
                
            lines.append(f"Entities: {metrics['entity_summary']['entity_count']}")
            lines.append(f"Total events: {metrics['entity_summary']['total_events']}")
            
            return "\n".join(lines)
        else:
            return f"Unsupported format: {format}"
    
    async def analyze_performance(self) -> Dict:
        """
        Analyze performance metrics to identify optimization opportunities.
        Returns insights about system behavior.
        """
        insights = {
            'timestamp': time.time(),
            'bottlenecks': [],
            'anomalies': [],
            'trends': [],
            'recommendations': []
        }
        
        # Analyze operation timings for bottlenecks
        for op_type, timings in self.operation_timings.items():
            if not timings:
                continue
                
            durations = [t['duration'] for t in timings]
            avg_duration = sum(durations) / len(durations)
            
            # Check for operations that take longer than 1 second on average
            if avg_duration > 1.0:
                insights['bottlenecks'].append({
                    'operation_type': op_type,
                    'avg_duration': avg_duration,
                    'sample_count': len(timings)
                })
                
                insights['recommendations'].append({
                    'type': 'optimization',
                    'target': op_type,
                    'recommendation': f"Optimize {op_type} operations (avg: {avg_duration:.2f}s)"
                })
        
        # Analyze entity metrics for anomalies
        for entity_id, metrics in self.entity_metrics.items():
            # Check for very high event frequency
            if metrics['event_count'] > 1000:
                active_period = metrics['last_seen'] - metrics['first_seen']
                events_per_hour = (metrics['event_count'] / active_period) * 3600 if active_period > 0 else 0
                
                if events_per_hour > 100:  # More than 100 events per hour
                    insights['anomalies'].append({
                        'entity_id': entity_id,
                        'event_count': metrics['event_count'],
                        'events_per_hour': events_per_hour
                    })
                    
                    insights['recommendations'].append({
                        'type': 'investigation',
                        'target': entity_id,
                        'recommendation': f"Investigate high activity for entity {entity_id} ({events_per_hour:.1f} events/hour)"
                    })
        
        # Analyze global counters for trends
        operation_counts = {
            name.split('.')[-1]: value
            for name, value in self.global_counters.items()
            if name.startswith('operation.')
        }
        
        if operation_counts:
            # Identify imbalanced operation types
            total_ops = sum(operation_counts.values())
            for op_type, count in operation_counts.items():
                ratio = count / total_ops if total_ops > 0 else 0
                
                if ratio > 0.7:  # Operation type represents >70% of all operations
                    insights['trends'].append({
                        'type': 'dominant_operation',
                        'operation_type': op_type,
                        'ratio': ratio,
                        'count': count
                    })
        
        return insights



# === FILE: asf\layer4_environmental_coupling\components\predictive_modeler.py ===

import asyncio
import time
import uuid
import logging
import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

from asf.environmental_coupling.models import EnvironmentalPrediction, EnvironmentalCoupling

class PredictiveEnvironmentalModeler:
    """
    Implements Seth's controlled hallucination principle for environmental interactions.
    Generates predictions about environmental entities and updates based on actual observations.
    """
    def __init__(self):
        self.predictions = {}  # Maps prediction_id to EnvironmentalPrediction
        self.entity_predictions = defaultdict(list)  # Maps entity_id to prediction_ids
        self.prediction_errors = defaultdict(list)  # Maps entity_id to historical errors
        self.precision = {}  # Maps entity_id to precision (inverse error variance)
        self.logger = logging.getLogger("ASF.Layer4.PredictiveEnvironmentalModeler")
    
    async def predict_interaction(self, environmental_entity_id, context=None):
        """
        Generate a prediction about an upcoming interaction with an environmental entity.
        Implements Seth's controlled hallucination principle.
        """
        prediction_id = str(uuid.uuid4())
        context = context or {}
        
        # Generate prediction based on historical patterns
        predicted_data = await self._generate_prediction(environmental_entity_id, context)
        
        # Get confidence based on precision
        confidence = self._calculate_confidence(environmental_entity_id)
        
        # Create prediction object
        prediction = EnvironmentalPrediction(
            id=prediction_id,
            environmental_entity_id=environmental_entity_id,
            predicted_data=predicted_data,
            confidence=confidence,
            context=context,
            precision=self.precision.get(environmental_entity_id, 1.0)
        )
        
        # Store prediction
        self.predictions[prediction_id] = prediction
        self.entity_predictions[environmental_entity_id].append(prediction_id)
        
        # Clean up old predictions for this entity
        await self._cleanup_old_predictions(environmental_entity_id)
        
        self.logger.debug(f"Generated prediction {prediction_id} for entity {environmental_entity_id} with confidence {confidence:.2f}")
        return prediction
    
    async def evaluate_prediction(self, prediction_id, actual_data):
        """
        Compare prediction to actual data and update precision.
        Implements Seth's precision-weighted prediction error.
        """
        if prediction_id not in self.predictions:
            return None
            
        prediction = self.predictions[prediction_id]
        prediction.verification_time = time.time()
        
        # Calculate prediction error
        error = self._calculate_prediction_error(prediction.predicted_data, actual_data)
        prediction.prediction_error = error
        
        # Update error history
        entity_id = prediction.environmental_entity_id
        self.prediction_errors[entity_id].append(error)
        
        # Limit history size
        if len(self.prediction_errors[entity_id]) > 20:
            self.prediction_errors[entity_id] = self.prediction_errors[entity_id][-20:]
        
        # Update precision (inverse variance)
        if len(self.prediction_errors[entity_id]) > 1:
            variance = np.var(self.prediction_errors[entity_id])
            new_precision = 1.0 / (variance + 1e-6)  # Avoid division by zero
            self.precision[entity_id] = new_precision
            prediction.precision = new_precision
        
        self.logger.debug(f"Evaluated prediction {prediction_id} with error {error:.4f}, new precision {self.precision.get(entity_id, 1.0):.4f}")
        return {
            'prediction_id': prediction_id,
            'error': error,
            'precision': self.precision.get(entity_id, 1.0)
        }
    
    async def get_predictions_for_entity(self, environmental_entity_id, limit=5, future_only=True):
        """
        Retrieve recent predictions for an entity.
        If future_only is True, only return predictions that haven't been verified yet.
        """
        prediction_ids = self.entity_predictions.get(environmental_entity_id, [])
        
        # Get actual prediction objects
        predictions = []
        current_time = time.time()
        
        for pred_id in reversed(prediction_ids):  # Most recent first
            if pred_id in self.predictions:
                pred = self.predictions[pred_id]
                
                # Filter for future predictions if requested
                if future_only and pred.verification_time is not None:
                    continue
                    
                predictions.append(pred)
                
                if len(predictions) >= limit:
                    break
        
        return predictions
    
    async def _generate_prediction(self, environmental_entity_id, context):
        """
        Internal method to generate prediction based on historical patterns.
        In a real implementation, this would use sophisticated ML models.
        """
        # Extract context information that might be useful for prediction
        interaction_type = context.get('last_interaction_type')
        
        # Generate basic prediction structure
        predicted_data = {
            'predicted_interaction_type': await self._predict_interaction_type(environmental_entity_id, interaction_type),
            'predicted_content': await self._generate_content_prediction(environmental_entity_id),
            'predicted_timing': await self._predict_timing(environmental_entity_id),
            'predicted_response_time': random.uniform(0.1, 2.0),
            'predicted_confidence': random.uniform(0.5, 0.9)
        }
        
        return predicted_data
    
    async def _predict_interaction_type(self, entity_id, last_interaction_type=None):
        """
        Predict the type of interaction expected from this entity.
        Uses historical patterns where available.
        """
        # If we have prediction errors, check for patterns
        if entity_id in self.prediction_errors and len(self.prediction_errors[entity_id]) > 0:
            # This would be more sophisticated in a real implementation
            interaction_types = ['query', 'update', 'notification', 'request']
            
            # If we have last interaction type, use that as a signal
            if last_interaction_type:
                # Simple pattern: if last type was query, next might be update
                if last_interaction_type == 'query':
                    return 'update'
                elif last_interaction_type == 'update':
                    return 'notification'
                else:
                    return random.choice(interaction_types)
            else:
                return random.choice(interaction_types)
        else:
            # No history yet, return a default prediction
            return 'query'
    
    async def _generate_content_prediction(self, entity_id):
        """
        Generate predicted content for an interaction.
        This would use more sophisticated models in a real implementation.
        """
        # Simplified content prediction
        content_types = [
            {'type': 'text', 'expected_length': random.randint(10, 100)},
            {'type': 'structured', 'expected_fields': ['id', 'timestamp', 'value']},
            {'type': 'numeric', 'expected_range': [0, 100]}
        ]
        
        return random.choice(content_types)
    
    async def _predict_timing(self, entity_id):
        """
        Predict when the next interaction will occur.
        Uses timing patterns from history where available.
        """
        # Start with a default timing (30-60 seconds in the future)
        base_timing = time.time() + random.uniform(30, 60)
        
        # If we have prediction history, refine the timing prediction
        if entity_id in self.entity_predictions and len(self.entity_predictions[entity_id]) >= 2:
            # Get the last few predictions to analyze timing patterns
            recent_pred_ids = self.entity_predictions[entity_id][-5:]
            recent_predictions = [self.predictions[pid] for pid in recent_pred_ids if pid in self.predictions]
            
            if len(recent_predictions) >= 2:
                # Calculate average time between predictions
                times = [p.prediction_time for p in recent_predictions]
                intervals = [times[i] - times[i-1] for i in range(1, len(times))]
                avg_interval = sum(intervals) / len(intervals)
                
                # Predict next interaction based on average interval
                last_time = max(times)
                predicted_time = last_time + avg_interval
                
                # Add some randomness (±20%)
                predicted_time += avg_interval * random.uniform(-0.2, 0.2)
                
                return predicted_time
        
        return base_timing
    
    def _calculate_confidence(self, entity_id):
        """
        Calculate confidence level based on prediction history.
        Higher precision leads to higher confidence.
        """
        # Base confidence starts moderate
        base_confidence = 0.5
        
        # Adjust based on precision if available
        precision = self.precision.get(entity_id, 1.0)
        
        # Scale precision to confidence (0.3-0.9 range)
        # Higher precision = higher confidence
        precision_factor = min(0.4, (precision - 1.0) * 0.1) if precision > 1.0 else 0
        confidence = base_confidence + precision_factor
        
        # Cap confidence
        confidence = min(0.9, max(0.3, confidence))
        
        return confidence
    
    def _calculate_prediction_error(self, predicted_data, actual_data):
        """
        Calculate error between prediction and actual data.
        Returns a normalized error value between 0.0 (perfect) and 1.0 (completely wrong).
        """
        # For simplicity, we'll focus on a few key comparisons
        error_components = []
        
        # Check interaction type prediction if available
        if 'predicted_interaction_type' in predicted_data and 'interaction_type' in actual_data:
            type_match = predicted_data['predicted_interaction_type'] == actual_data['interaction_type']
            error_components.append(0.0 if type_match else 1.0)
        
        # Check content type prediction if available
        if 'predicted_content' in predicted_data and 'content_type' in actual_data:
            if 'type' in predicted_data['predicted_content']:
                content_match = predicted_data['predicted_content']['type'] == actual_data['content_type']
                error_components.append(0.0 if content_match else 1.0)
        
        # Check timing prediction if available (with a tolerance)
        if 'predicted_timing' in predicted_data and 'timestamp' in actual_data:
            timing_diff = abs(predicted_data['predicted_timing'] - actual_data['timestamp'])
            # Normalize timing error (0.0 for perfect, 1.0 for >60 seconds off)
            timing_error = min(1.0, timing_diff / 60.0)
            error_components.append(timing_error)
        
        # If we couldn't calculate any components, return a default error
        if not error_components:
            return 0.5
        
        # Return average error across all components
        return sum(error_components) / len(error_components)
    
    async def _cleanup_old_predictions(self, entity_id, max_age=3600, max_count=100):
        """Clean up old predictions to prevent memory issues."""
        if entity_id not in self.entity_predictions:
            return
            
        # Get prediction IDs for this entity
        prediction_ids = self.entity_predictions[entity_id]
        
        # If under max count, no need to clean up
        if len(prediction_ids) <= max_count:
            return
            
        current_time = time.time()
        predictions_to_remove = []
        
        # Identify old predictions to remove
        for pred_id in prediction_ids:
            if pred_id in self.predictions:
                pred = self.predictions[pred_id]
                
                # Remove verified predictions older than max_age
                if pred.verification_time and (current_time - pred.verification_time > max_age):
                    predictions_to_remove.append(pred_id)
                    
                # Remove old unverified predictions
                elif current_time - pred.prediction_time > max_age * 2:
                    predictions_to_remove.append(pred_id)
            else:
                # ID exists but prediction doesn't (should not happen)
                predictions_to_remove.append(pred_id)
                
        # Remove identified predictions
        for pred_id in predictions_to_remove:
            if pred_id in self.predictions:
                del self.predictions[pred_id]
            self.entity_predictions[entity_id].remove(pred_id)



# === FILE: asf\layer4_environmental_coupling\components\rl_optimizer.py ===

import asyncio
import time
import uuid
import logging
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional
from collections import defaultdict

class ReinforcementLearningOptimizer:
    """
    Enhanced reinforcement learning optimizer with active inference capabilities.
    Optimizes coupling parameters through reinforcement learning and counterfactual simulation.
    """
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.coupling_models = {}  # Maps coupling_id to RL models
        self.experience_buffer = defaultdict(list)  # Maps coupling_id to experiences
        self.neural_models = {}  # Maps coupling_id to neural networks
        self.model_parameters = {}  # Maps coupling_id to hyperparameters
        self.optimization_history = defaultdict(list)  # Maps coupling_id to optimization history
        
        # Seth's Data Paradox enhancements
        self.counterfactual_simulations = {}  # Maps coupling_id to simulated interactions
        self.active_inference_tests = {}  # Maps coupling_id to active inference test results
        self.predicted_outcomes = {}  # Maps coupling_id to predicted interaction outcomes
        
        self.logger = logging.getLogger("ASF.Layer4.ReinforcementLearningOptimizer")
        
    async def initialize(self, knowledge_substrate):
        """Initialize the RL optimizer."""
        self.knowledge_substrate = knowledge_substrate
        return True
        
    async def initialize_coupling_model(self, coupling_id, coupling):
        """Initialize RL model for a new coupling."""
        model_type = self._determine_model_type(coupling)
        
        self.coupling_models[coupling_id] = {
            'model_type': model_type,
            'state_space': self._define_state_space(coupling),
            'action_space': self._define_action_space(coupling),
            'q_values': {},  # Q-values for state-action pairs
            'last_updated': time.time(),
            'update_count': 0
        }
        
        # Set model parameters
        self.model_parameters[coupling_id] = {
            'learning_rate': self.learning_rate,
            'discount_factor': 0.9,
            'exploration_rate': 0.2,
            'batch_size': 32
        }
        
        # Initialize neural model for more complex couplings
        if model_type == 'neural':
            self._initialize_neural_model(coupling_id, coupling)
            
        # Initialize counterfactual simulation model
        self.counterfactual_simulations[coupling_id] = []
        
        return True
        
    async def update_from_interaction(self, coupling_id, interaction_data, interaction_type, bayesian_update):
        """
        Update RL model based on interaction outcome.
        """
        start_time = time.time()
        
        if coupling_id not in self.coupling_models:
            return {'status': 'model_not_found'}
            
        model = self.coupling_models[coupling_id]
        
        # Extract state and action from interaction
        state = self._extract_state(interaction_data)
        action = self._extract_action(interaction_data)
        
        # Calculate reward based on Bayesian update
        reward = self._calculate_reward(bayesian_update)
        
        # Determine new state
        new_state = self._extract_state({
            'interaction_type': interaction_type,
            'bayesian_confidence': bayesian_update.get('new_confidence', 0.5)
        })
        
        # Add to experience buffer
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'new_state': new_state,
            'timestamp': time.time()
        }
        self.experience_buffer[coupling_id].append(experience)
        
        # Limit buffer size
        if len(self.experience_buffer[coupling_id]) > 1000:
            self.experience_buffer[coupling_id] = self.experience_buffer[coupling_id][-1000:]
            
        # Update model
        if model['model_type'] == 'tabular':
            await self._update_tabular_model(coupling_id, experience)
        elif model['model_type'] == 'neural':
            await self._update_neural_model(coupling_id, experience)
            
        # Generate predicted outcomes for future interactions
        await self._generate_predictions(coupling_id, new_state)
        
        # Check if this was an active inference test
        active_inference_result = None
        if coupling_id in self.active_inference_tests:
            # Compare actual results to test expectations
            active_inference_result = await self._evaluate_active_inference_test(
                coupling_id, interaction_data, reward
            )
            
        # Record optimization step
        self.optimization_history[coupling_id].append({
            'timestamp': time.time(),
            'reward': reward,
            'bayesian_confidence': bayesian_update.get('new_confidence', 0.5),
            'exploration_rate': self.model_parameters[coupling_id]['exploration_rate']
        })
        
        # Adaptive exploration rate decay
        self.model_parameters[coupling_id]['exploration_rate'] *= 0.995
        
        return {
            'status': 'updated',
            'reward': reward,
            'model_type': model['model_type'],
            'experiences': len(self.experience_buffer[coupling_id]),
            'active_inference_result': active_inference_result,
            'elapsed_time': time.time() - start_time
        }
        
    async def get_optimal_parameters(self, entity, target_id, coupling):
        """
        Get optimal parameters for interacting with target.
        Uses RL model to determine best action in current state.
        """
        coupling_id = coupling.id
        
        if coupling_id not in self.coupling_models:
            return {}
            
        model = self.coupling_models[coupling_id]
        
        # Extract current state
        current_state = self._extract_state({
            'entity_type': getattr(entity, 'type', 'unknown'),
            'bayesian_confidence': coupling.bayesian_confidence
        })
        
        # Determine if we should explore or exploit
        if random.random() < self.model_parameters[coupling_id]['exploration_rate']:
            # Exploration: choose random action
            action = random.choice(list(model['action_space'].keys()))
        else:
            # Exploitation: choose best action
            action = await self._get_best_action(coupling_id, current_state)
            
        # Convert action to parameters
        parameters = self._action_to_parameters(action, coupling)
        
        # Add counterfactual thinking - consider alternative parameters
        counterfactual_params = await self._generate_counterfactual_parameters(
            coupling_id, current_state, parameters
        )
        
        # If counterfactual parameters are better, use them
        if counterfactual_params and counterfactual_params.get('predicted_reward', 0) > parameters.get('predicted_reward', 0):
            parameters = counterfactual_params
            
        return parameters
        
    async def _generate_counterfactual_parameters(self, coupling_id, state, base_parameters):
        """
        Generate counterfactual parameters to consider alternative actions.
        Implements Seth's counterfactual thinking principle.
        """
        if coupling_id not in self.coupling_models:
            return None
            
        model = self.coupling_models[coupling_id]
        
        # Get all possible actions except the one chosen
        all_actions = list(model['action_space'].keys())
        base_action = base_parameters.get('action_type', 'default')
        alternative_actions = [a for a in all_actions if a != base_action]
        
        if not alternative_actions:
            return None
            
        # Pick an alternative action
        alt_action = random.choice(alternative_actions)
        
        # Convert to parameters
        alt_parameters = model['action_space'][alt_action].copy()
        alt_parameters['action_type'] = alt_action
        
        # Predict reward for this alternative
        if model['model_type'] == 'tabular':
            state_key = self._state_to_key(state)
            alt_reward = model['q_values'].get((state_key, alt_action), 0)
        elif model['model_type'] == 'neural' and coupling_id in self.neural_models:
            nn_model = self.neural_models[coupling_id]
            state_tensor = torch.tensor(list(state.values()), dtype=torch.float32)
            with torch.no_grad():
                q_values = nn_model(state_tensor)
                action_idx = all_actions.index(alt_action)
                alt_reward = q_values[action_idx].item()
        else:
            alt_reward = 0
            
        alt_parameters['predicted_reward'] = alt_reward
        
        # Record counterfactual simulation
        self.counterfactual_simulations[coupling_id].append({
            'timestamp': time.time(),
            'state': state,
            'base_action': base_action,
            'alt_action': alt_action,
            'base_reward': base_parameters.get('predicted_reward', 0),
            'alt_reward': alt_reward
        })
        
        # Limit history
        if len(self.counterfactual_simulations[coupling_id]) > 100:
            self.counterfactual_simulations[coupling_id] = self.counterfactual_simulations[coupling_id][-100:]
            
        return alt_parameters
        
    async def setup_active_inference_test(self, coupling_id, test_parameters):
        """
        Set up an active inference test for a coupling.
        Implements Seth's active inference principle to test predictions.
        """
        if coupling_id not in self.coupling_models:
            return {'status': 'model_not_found'}
            
        # Record test setup
        test_id = str(uuid.uuid4())
        self.active_inference_tests[coupling_id] = {
            'test_id': test_id,
            'parameters': test_parameters,
            'setup_time': time.time(),
            'expected_reward': test_parameters.get('expected_reward', 0),
            'completed': False
        }
        
        return {
            'status': 'test_setup',
            'test_id': test_id,
            'coupling_id': coupling_id
        }
        
    async def _evaluate_active_inference_test(self, coupling_id, interaction_data, actual_reward):
        """
        Evaluate the results of an active inference test.
        """
        if coupling_id not in self.active_inference_tests:
            return None
            
        test = self.active_inference_tests[coupling_id]
        if test['completed']:
            return None
            
        # Calculate prediction error
        expected_reward = test['expected_reward']
        prediction_error = abs(actual_reward - expected_reward)
        
        # Mark test as completed
        test['completed'] = True
        test['completion_time'] = time.time()
        test['actual_reward'] = actual_reward
        test['prediction_error'] = prediction_error
        
        # Calculate information gain
        information_gain = 1.0 / (1.0 + prediction_error)
        test['information_gain'] = information_gain
        
        return {
            'test_id': test['test_id'],
            'expected_reward': expected_reward,
            'actual_reward': actual_reward,
            'prediction_error': prediction_error,
            'information_gain': information_gain
        }
        
    async def _generate_predictions(self, coupling_id, current_state):
        """
        Generate predictions about future interaction outcomes.
        Implements Seth's controlled hallucination principle.
        """
        if coupling_id not in self.coupling_models:
            return
            
        model = self.coupling_models[coupling_id]
        
        # Predict outcomes for all possible actions
        predictions = {}
        for action in model['action_space'].keys():
            # Predict reward for this action
            if model['model_type'] == 'tabular':
                state_key = self._state_to_key(current_state)
                reward = model['q_values'].get((state_key, action), 0)
            elif model['model_type'] == 'neural' and coupling_id in self.neural_models:
                nn_model = self.neural_models[coupling_id]
                state_tensor = torch.tensor(list(current_state.values()), dtype=torch.float32)
                with torch.no_grad():
                    q_values = nn_model(state_tensor)
                    action_idx = list(model['action_space'].keys()).index(action)
                    reward = q_values[action_idx].item()
            else:
                reward = 0
                
            predictions[action] = {
                'predicted_reward': reward,
                'confidence': min(1.0, model['update_count'] / 100),  # Confidence grows with experience
                'generation_time': time.time()
            }
            
        # Store predictions
        self.predicted_outcomes[coupling_id] = predictions
        
    async def _get_best_action(self, coupling_id, state):
        """Get best action for a given state."""
        model = self.coupling_models[coupling_id]
        
        if model['model_type'] == 'tabular':
            state_key = self._state_to_key(state)
            # Find action with highest Q-value
            best_action = None
            best_value = float('-inf')
            
            for (s, a), value in model['q_values'].items():
                if s == state_key and value > best_value:
                    best_action = a
                    best_value = value
                    
            # If no action found, choose random
            if best_action is None:
                best_action = random.choice(list(model['action_space'].keys()))
                
            return best_action
            
        elif model['model_type'] == 'neural' and coupling_id in self.neural_models:
            nn_model = self.neural_models[coupling_id]
            state_tensor = torch.tensor(list(state.values()), dtype=torch.float32)
            
            with torch.no_grad():
                q_values = nn_model(state_tensor)
                best_idx = torch.argmax(q_values).item()
                
            # Convert index to action
            actions = list(model['action_space'].keys())
            return actions[best_idx]
            
        # Fallback to random action
        return random.choice(list(model['action_space'].keys()))
        
    async def _update_tabular_model(self, coupling_id, experience):
        """Update tabular Q-learning model."""
        model = self.coupling_models[coupling_id]
        params = self.model_parameters[coupling_id]
        
        # Convert states to keys
        state_key = self._state_to_key(experience['state'])
        new_state_key = self._state_to_key(experience['new_state'])
        action = experience['action']
        
        # Get current Q-value
        current_q = model['q_values'].get((state_key, action), 0)
        
        # Find max Q-value for new state
        max_q_next = 0
        for (s, a), value in model['q_values'].items():
            if s == new_state_key and value > max_q_next:
                max_q_next = value
                
        # Q-learning update rule
        new_q = current_q + params['learning_rate'] * (
            experience['reward'] + params['discount_factor'] * max_q_next - current_q
        )
        
        # Update Q-value
        model['q_values'][(state_key, action)] = new_q
        model['update_count'] += 1
        model['last_updated'] = time.time()
        
    async def _update_neural_model(self, coupling_id, experience):
        """Update neural network model."""
        if coupling_id not in self.neural_models:
            return
            
        model = self.coupling_models[coupling_id]
        params = self.model_parameters[coupling_id]
        nn_model = self.neural_models[coupling_id]
        
        # Add to experience buffer (already done in main update method)
        
        # If we have enough experiences, perform a batch update
        if len(self.experience_buffer[coupling_id]) >= params['batch_size']:
            # Sample batch
            batch = random.sample(self.experience_buffer[coupling_id], params['batch_size'])
            
            # Prepare tensors
            states = torch.tensor([list(exp['state'].values()) for exp in batch], dtype=torch.float32)
            actions = [exp['action'] for exp in batch]
            rewards = torch.tensor([exp['reward'] for exp in batch], dtype=torch.float32)
            next_states = torch.tensor([list(exp['new_state'].values()) for exp in batch], dtype=torch.float32)
            
            # Convert actions to indices
            action_space = list(model['action_space'].keys())
            action_indices = torch.tensor([action_space.index(a) for a in actions], dtype=torch.long)
            
            # Get current Q-values
            current_q = nn_model(states)
            
            # Get max Q-values for next states
            with torch.no_grad():
                next_q = nn_model(next_states)
                max_next_q = next_q.max(1)[0]
                
            # Calculate target Q-values
            target_q = current_q.clone()
            for i in range(params['batch_size']):
                target_q[i, action_indices[i]] = rewards[i] + params['discount_factor'] * max_next_q[i]
                
            # Optimize model
            nn_model.optimizer.zero_grad()
            loss = F.smooth_l1_loss(current_q, target_q)
            loss.backward()
            nn_model.optimizer.step()
            
            # Update model metadata
            model['update_count'] += 1
            model['last_updated'] = time.time()
            
    def _initialize_neural_model(self, coupling_id, coupling):
        """Initialize neural network for a coupling."""
        state_size = len(self._define_state_space(coupling))
        action_size = len(self._define_action_space(coupling))
        
        # Simple neural network
        model = nn.Sequential(
            nn.Linear(state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, action_size)
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        
        # Store model and optimizer
        model.optimizer = optimizer
        self.neural_models[coupling_id] = model
        
    def _determine_model_type(self, coupling):
        """Determine appropriate model type for coupling."""
        # Simple heuristic: use neural for more complex couplings
        if coupling.coupling_type.name in ['ADAPTIVE', 'PREDICTIVE']:
            return 'neural'
        return 'tabular'
        
    def _define_state_space(self, coupling):
        """Define state space for a coupling."""
        # Default state features
        state_space = {
            'bayesian_confidence': 0.5,
            'coupling_strength': coupling.coupling_strength,
            'interaction_count': 0,
            'time_since_last': 0
        }
        return state_space
        
    def _define_action_space(self, coupling):
        """Define action space for a coupling."""
        # Basic action types with parameters
        action_space = {
            'standard': {
                'intensity': 0.5,
                'timing_offset': 0
            },
            'intensive': {
                'intensity': 0.8,
                'timing_offset': 0
            },
            'minimal': {
                'intensity': 0.2,
                'timing_offset': 0
            }
        }
        return action_space
        
    def _extract_state(self, data):
        """Extract state representation from data."""
        # Simple state extraction
        state = {
            'interaction_type': hash(data.get('interaction_type', 'unknown')) % 100 / 100,
            'bayesian_confidence': data.get('bayesian_confidence', 0.5)
        }
        return state
        
    def _extract_action(self, data):
        """Extract action from interaction data."""
        # Default to standard action
        return data.get('action_type', 'standard')
        
    def _calculate_reward(self, bayesian_update):
        """Calculate reward from Bayesian update."""
        # Reward based on confidence change
        confidence_before = bayesian_update.get('prior_confidence', 0.5)
        confidence_after = bayesian_update.get('new_confidence', 0.5)
        
        # Positive reward for confidence increase
        if confidence_after > confidence_before:
            return 2 * (confidence_after - confidence_before)
        # Smaller positive reward for maintaining high confidence
        elif confidence_after > 0.7:
            return 0.1
        # Small negative reward for confidence decrease
        else:
            return -0.5 * (confidence_before - confidence_after)
            
    def _state_to_key(self, state):
        """Convert state dict to hashable key."""
        # Round values to reduce state space
        return tuple((k, round(v, 2)) for k, v in sorted(state.items()))
        
    def _action_to_parameters(self, action, coupling):
        """Convert action to distribution parameters."""
        model = self.coupling_models.get(coupling.id, {})
        action_space = model.get('action_space', {})
        
        # Get parameters for this action
        parameters = action_space.get(action, {}).copy()
        parameters['action_type'] = action
        
        # Add predicted reward if available
        if coupling.id in self.predicted_outcomes and action in self.predicted_outcomes[coupling.id]:
            parameters['predicted_reward'] = self.predicted_outcomes[coupling.id][action]['predicted_reward']
            
        return parameters
        
    async def cleanup_coupling_model(self, coupling_id):
        """Clean up resources for a terminated coupling."""
        if coupling_id in self.coupling_models:
            del self.coupling_models[coupling_id]
        if coupling_id in self.experience_buffer:
            del self.experience_buffer[coupling_id]
        if coupling_id in self.neural_models:
            del self.neural_models[coupling_id]
        if coupling_id in self.model_parameters:
            del self.model_parameters[coupling_id]
        if coupling_id in self.optimization_history:
            del self.optimization_history[coupling_id]
        if coupling_id in self.counterfactual_simulations:
            del self.counterfactual_simulations[coupling_id]
        if coupling_id in self.active_inference_tests:
            del self.active_inference_tests[coupling_id]
        if coupling_id in self.predicted_outcomes:
            del self.predicted_outcomes[coupling_id]
        return True
        
    async def perform_maintenance(self):
        """Perform periodic maintenance on RL models."""
        start_time = time.time()
        
        # Count models
        model_count = len(self.coupling_models)
        
        # Clean up old experiences
        total_experiences = 0
        cleaned_experiences = 0
        
        for coupling_id, experiences in self.experience_buffer.items():
            total_experiences += len(experiences)
            if len(experiences) > 1000:
                self.experience_buffer[coupling_id] = experiences[-1000:]
                cleaned_experiences += len(experiences) - 1000
                
        # Free GPU memory for neural models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return {
            'model_count': model_count,
            'total_experiences': total_experiences,
            'cleaned_experiences': cleaned_experiences,
            'neural_models': len(self.neural_models),
            'active_inference_tests': len(self.active_inference_tests),
            'elapsed_time': time.time() - start_time
        }
        
    async def get_metrics(self):
        """Get metrics about the RL optimizer."""
        return {
            'model_count': len(self.coupling_models),
            'neural_model_count': len(self.neural_models),
            'total_experiences': sum(len(exp) for exp in self.experience_buffer.values()),
            'avg_update_count': np.mean([model['update_count'] for model in self.coupling_models.values()]) if self.coupling_models else 0,
            'counterfactual_simulations': sum(len(sims) for sims in self.counterfactual_simulations.values()),
            'active_inference_tests': len(self.active_inference_tests),
            'using_gpu': torch.cuda.is_available()
        }



# === FILE: asf\layer4_environmental_coupling\utils\__init__.py ===




# === FILE: asf\layer4_environmental_coupling\utils\async_helpers.py ===

import asyncio
import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

class AsyncEventQueue:
    """
    Asynchronous queue for coupling events with priority support.
    Provides ordered, priority-based event processing with monitoring capabilities.
    """
    def __init__(self, max_size=0):
        """
        Initialize the async event queue.
        
        Args:
            max_size: Maximum queue size (0 for unlimited)
        """
        self.queue = asyncio.PriorityQueue(maxsize=max_size)
        self.event_count = 0
        self.total_submitted = 0
        self.total_processed = 0
        self.processing_times = []
        self.priority_distribution = defaultdict(int)
        self.logger = logging.getLogger("ASF.Layer4.AsyncEventQueue")
        
    async def put(self, event, priority=None):
        """
        Add an event to the queue with priority support.
        
        Args:
            event: The event to add to the queue
            priority: Optional priority override (0-1, higher is higher priority)
                     If None, uses event.priority if available
        
        Returns:
            True if successful
        """
        # Determine priority
        if priority is None and hasattr(event, 'priority'):
            priority = event.priority
        else:
            priority = 0.5  # Default priority
            
        # Record priority for metrics
        priority_bin = round(priority * 10) / 10  # Round to nearest 0.1
        self.priority_distribution[priority_bin] += 1
        
        # Use counter to maintain FIFO order for same priority
        self.event_count += 1
        self.total_submitted += 1
        
        # Invert priority so lower values are processed first
        inverted_priority = 1.0 - priority
        
        # Put in queue with (priority, count, event) structure
        await self.queue.put((inverted_priority, self.event_count, event))
        return True
        
    async def get(self):
        """
        Get the next event from the queue based on priority.
        
        Returns:
            The next event
        """
        # Get from queue, ignoring priority and count
        _, _, event = await self.queue.get()
        return event
        
    def task_done(self):
        """Mark a task as done."""
        self.queue.task_done()
        self.total_processed += 1
        
    async def get_with_timeout(self, timeout=1.0):
        """
        Get an event with timeout.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Event or None if timeout
        """
        try:
            # Use wait_for with timeout
            _, _, event = await asyncio.wait_for(self.queue.get(), timeout)
            return event
        except asyncio.TimeoutError:
            return None
            
    async def put_batch(self, events, base_priority=0.5):
        """
        Add multiple events to the queue.
        
        Args:
            events: List of events to add
            base_priority: Base priority to use if event has no priority
            
        Returns:
            Number of events added
        """
        for event in events:
            priority = getattr(event, 'priority', base_priority)
            await self.put(event, priority)
        return len(events)
        
    def qsize(self):
        """Get current queue size."""
        return self.queue.qsize()
        
    def empty(self):
        """Check if queue is empty."""
        return self.queue.empty()
        
    def full(self):
        """Check if queue is full."""
        return self.queue.full()
        
    async def drain(self):
        """
        Drain the queue by processing all pending events.
        
        Returns:
            Number of events drained
        """
        drained = 0
        while not self.queue.empty():
            await self.queue.get()
            self.queue.task_done()
            drained += 1
        return drained
        
    def get_metrics(self):
        """
        Get queue metrics.
        
        Returns:
            Dictionary of metrics
        """
        return {
            'current_size': self.queue.qsize(),
            'total_submitted': self.total_submitted,
            'total_processed': self.total_processed,
            'backlog': self.total_submitted - self.total_processed,
            'priority_distribution': dict(self.priority_distribution)
        }
        
    async def wait_for_completion(self, timeout=None):
        """
        Wait for all current items to be processed.
        
        Args:
            timeout: Optional timeout in seconds
            
        Returns:
            True if completed, False if timeout
        """
        try:
            await asyncio.wait_for(self.queue.join(), timeout)
            return True
        except asyncio.TimeoutError:
            return False



# === FILE: asf\layer4_environmental_coupling\utils\tensor_utils.py ===

import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, Tuple, List, Optional

def normalize_tensor(tensor, dim=0, eps=1e-8):
    """
    Normalize a tensor along a specific dimension.
    
    Args:
        tensor: Input tensor
        dim: Dimension to normalize
        eps: Small value to avoid division by zero
        
    Returns:
        Normalized tensor
    """
    if isinstance(tensor, np.ndarray):
        norm = np.linalg.norm(tensor, axis=dim, keepdims=True)
        return tensor / np.maximum(norm, eps)
    elif isinstance(tensor, torch.Tensor):
        return F.normalize(tensor, p=2, dim=dim, eps=eps)
    else:
        raise TypeError("Input must be numpy array or torch tensor")

def cosine_similarity_batch(queries, corpus):
    """
    Calculate cosine similarity between queries and corpus.
    
    Args:
        queries: Tensor of shape [Q, D] for Q queries of dimension D
        corpus: Tensor of shape [C, D] for C corpus items of dimension D
        
    Returns:
        Similarities tensor of shape [Q, C]
    """
    if isinstance(queries, np.ndarray) and isinstance(corpus, np.ndarray):
        # Normalize
        queries_norm = normalize_tensor(queries, dim=1)
        corpus_norm = normalize_tensor(corpus, dim=1)
        
        # Calculate similarity
        return np.matmul(queries_norm, corpus_norm.T)
    
    elif isinstance(queries, torch.Tensor) and isinstance(corpus, torch.Tensor):
        # Normalize
        queries_norm = F.normalize(queries, p=2, dim=1)
        corpus_norm = F.normalize(corpus, p=2, dim=1)
        
        # Calculate similarity
        return torch.matmul(queries_norm, corpus_norm.t())
    
    else:
        raise TypeError("Both inputs must be of the same type (numpy array or torch tensor)")

def sparse_tensor_to_dense(sparse_tensor, shape=None):
    """
    Convert sparse tensor to dense tensor.
    
    Args:
        sparse_tensor: Sparse tensor (either torch.sparse or scipy.sparse)
        shape: Optional shape for the dense tensor
        
    Returns:
        Dense tensor
    """
    if isinstance(sparse_tensor, torch.Tensor) and sparse_tensor.is_sparse:
        return sparse_tensor.to_dense()
    
    elif hasattr(sparse_tensor, 'toarray'):  # scipy.sparse matrices
        return sparse_tensor.toarray()
    
    else:
        raise TypeError("Input must be a sparse tensor")

def batched_matrix_operation(operation, matrices, batch_size=64):
    """
    Apply an operation to a list of matrices in batches.
    
    Args:
        operation: Function to apply to each batch
        matrices: List of matrices
        batch_size: Size of each batch
        
    Returns:
        List of operation results
    """
    results = []
    for i in range(0, len(matrices), batch_size):
        batch = matrices[i:i+batch_size]
        batch_results = operation(batch)
        results.extend(batch_results)
    return results

def adaptive_precision_tensor(tensor, precision='auto'):
    """
    Convert tensor to appropriate precision based on content.
    
    Args:
        tensor: Input tensor
        precision: 'auto', 'float32', 'float16', or 'bfloat16'
        
    Returns:
        Tensor with appropriate precision
    """
    if not isinstance(tensor, torch.Tensor):
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        else:
            tensor = torch.tensor(tensor)
    
    # Determine precision
    if precision == 'auto':
        # Check value range
        max_val = torch.max(torch.abs(tensor)).item()
        
        if max_val > 65504 or max_val < 6e-5:
            # Values too large or small for float16
            precision = 'float32'
        else:
            precision = 'float16'
    
    # Convert to specified precision
    if precision == 'float32':
        return tensor.float()
    elif precision == 'float16':
        return tensor.half()
    elif precision == 'bfloat16' and hasattr(torch, 'bfloat16'):
        return tensor.to(torch.bfloat16)
    else:
        return tensor.float()  # Default to float32
def mixed_precision_matmul(a, b, precision='auto'):
    """
    Perform matrix multiplication with mixed precision.
    
    Args:
        a: First tensor
        b: Second tensor
        precision: Precision to use for computation
        
    Returns:
        Result of matrix multiplication
    """
    # Determine if we should use GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert inputs to tensors if needed
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a, device=device)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b, device=device)
    
    # Ensure tensors are on the same device
    a = a.to(device)
    b = b.to(device)
    
    # Apply precision settings
    a_mp = adaptive_precision_tensor(a, precision)
    b_mp = adaptive_precision_tensor(b, precision)
    
    # Perform matrix multiplication
    result = torch.matmul(a_mp, b_mp)
    
    # Convert back to float32 for stability
    return result.float()

def sparse_tensor_add(a, b, inplace=False):
    """
    Add two sparse tensors efficiently.
    
    Args:
        a: First sparse tensor
        b: Second sparse tensor
        inplace: Whether to modify a in-place
        
    Returns:
        Result of addition
    """
    if not a.is_sparse or not b.is_sparse:
        raise ValueError("Both tensors must be sparse")
    
    if inplace:
        # Add values at corresponding indices
        indices_a = a.indices()
        values_a = a.values()
        
        indices_b = b.indices()
        values_b = b.values()
        
        # Handle overlapping indices
        result = torch.sparse_coo_tensor(
            torch.cat([indices_a, indices_b], dim=1),
            torch.cat([values_a, values_b]),
            a.size()
        ).coalesce()
        
        a._indices().copy_(result._indices())
        a._values().copy_(result._values())
        return a
    else:
        # Use built-in addition
        return a + b

def convert_to_csr(sparse_tensor):
    """
    Convert a sparse COO tensor to CSR format for efficient row operations.
    
    Args:
        sparse_tensor: Sparse tensor in COO format
        
    Returns:
        Sparse tensor in CSR format
    """
    if not sparse_tensor.is_sparse:
        raise ValueError("Input must be a sparse tensor")
    
    # Make sure it's coalesced (no duplicate indices)
    sparse_tensor = sparse_tensor.coalesce()
    
    # For PyTorch 1.9+ we can use to_sparse_csr directly
    if hasattr(sparse_tensor, 'to_sparse_csr'):
        return sparse_tensor.to_sparse_csr()
    
    # For older PyTorch versions, we need to convert to scipy and back
    indices = sparse_tensor._indices().cpu().numpy()
    values = sparse_tensor._values().cpu().numpy()
    shape = sparse_tensor.size()
    
    # Convert to scipy CSR
    import scipy.sparse as sp
    scipy_csr = sp.csr_matrix((values, (indices[0], indices[1])), shape=shape)
    
    # Convert back to PyTorch
    indptr = torch.tensor(scipy_csr.indptr, dtype=torch.long)
    indices = torch.tensor(scipy_csr.indices, dtype=torch.long)
    data = torch.tensor(scipy_csr.data, dtype=torch.float)
    
    # Create csr tensor - implementation depends on PyTorch version
    return torch.sparse_coo_tensor(
        torch.stack([indptr[:-1], indices]),
        data,
        shape
    )

def sparse_slice(sparse_tensor, dim, start, end):
    """
    Slice a sparse tensor along a dimension.
    
    Args:
        sparse_tensor: Sparse tensor to slice
        dim: Dimension to slice along
        start: Start index
        end: End index
        
    Returns:
        Sliced sparse tensor
    """
    if not sparse_tensor.is_sparse:
        raise ValueError("Input must be a sparse tensor")
    
    # Get indices and values
    indices = sparse_tensor._indices()
    values = sparse_tensor._values()
    
    # Find indices within the slice range
    mask = (indices[dim] >= start) & (indices[dim] < end)
    
    # Filter indices and values
    filtered_indices = indices[:, mask]
    filtered_values = values[mask]
    
    # Adjust indices for the new tensor
    if start != 0:
        filtered_indices[dim] -= start
    
    # Create new sizes
    new_size = list(sparse_tensor.size())
    new_size[dim] = end - start
    
    # Create and return new sparse tensor
    return torch.sparse_coo_tensor(
        filtered_indices, 
        filtered_values,
        torch.Size(new_size)
    )

def efficient_sparse_matmul(a, b):
    """
    Efficient sparse-sparse or sparse-dense matrix multiplication.
    
    Args:
        a: First tensor (sparse or dense)
        b: Second tensor (sparse or dense)
        
    Returns:
        Result of matrix multiplication
    """
    # Case 1: Both sparse - use special handling
    if a.is_sparse and b.is_sparse:
        # If available, use specialized sparse-sparse mm
        if hasattr(torch.sparse, 'mm'):
            return torch.sparse.mm(a, b)
        # Otherwise, convert one to dense
        else:
            return torch.mm(a.to_dense(), b)
    
    # Case 2: a is sparse, b is dense
    elif a.is_sparse and not b.is_sparse:
        if hasattr(a, 'is_sparse_csr') and a.is_sparse_csr:
            # Use specialized CSR operation if available
            if hasattr(torch.sparse, 'mm'):
                return torch.sparse.mm(a, b)
            else:
                return torch.mm(a.to_dense(), b)
        else:
            # For COO format
            return torch.sparse.mm(a, b)
    
    # Case 3: a is dense, b is sparse
    elif not a.is_sparse and b.is_sparse:
        # Transpose to use efficient sparse mm
        return torch.sparse.mm(b.t(), a.t()).t()
    
    # Case 4: Both dense - use standard mm
    else:
        return torch.mm(a, b)

def tensor_to_device(tensor, device=None):
    """
    Move a tensor to the specified device efficiently.
    
    Args:
        tensor: Input tensor
        device: Target device, or None to use GPU if available
        
    Returns:
        Tensor on the specified device
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # No-op if already on the right device
    if tensor.device == device:
        return tensor
    
    # Handle sparse tensors specially
    if tensor.is_sparse:
        # For sparse tensors, it's more efficient to recreate on the target device
        indices = tensor._indices()
        values = tensor._values()
        
        return torch.sparse_coo_tensor(
            indices.to(device),
            values.to(device),
            tensor.size()
        )
    
    # Standard device transfer for dense tensors
    return tensor.to(device)

def batch_sparse_dense_matmul(sparse_matrices, dense_matrices, batch_size=16):
    """
    Perform batched sparse-dense matrix multiplication efficiently.
    
    Args:
        sparse_matrices: List of sparse matrices
        dense_matrices: List of dense matrices (must be same length as sparse_matrices)
        batch_size: Batch size for processing
        
    Returns:
        List of multiplication results
    """
    if len(sparse_matrices) != len(dense_matrices):
        raise ValueError("Input lists must have the same length")
    
    results = []
    for i in range(0, len(sparse_matrices), batch_size):
        # Get batch
        sparse_batch = sparse_matrices[i:i+batch_size]
        dense_batch = dense_matrices[i:i+batch_size]
        
        # Process batch
        batch_results = []
        for sparse, dense in zip(sparse_batch, dense_batch):
            # Perform multiplication
            if sparse.is_sparse:
                result = torch.sparse.mm(sparse, dense)
            else:
                result = torch.mm(sparse, dense)
            batch_results.append(result)
        
        results.extend(batch_results)
    
    return results

def sparse_to_scipy(sparse_tensor):
    """
    Convert PyTorch sparse tensor to SciPy sparse matrix.
    
    Args:
        sparse_tensor: PyTorch sparse tensor
        
    Returns:
        SciPy sparse matrix
    """
    import scipy.sparse as sp
    
    if not sparse_tensor.is_sparse:
        raise ValueError("Input must be a sparse tensor")
    
    # Make sure it's coalesced
    sparse_tensor = sparse_tensor.coalesce()
    
    # Get indices and values
    indices = sparse_tensor._indices().cpu().numpy()
    values = sparse_tensor._values().cpu().numpy()
    shape = sparse_tensor.size()
    
    # Convert to scipy COO first
    scipy_coo = sp.coo_matrix((values, (indices[0], indices[1])), shape=shape)
    
    # Convert to CSR for better efficiency
    return scipy_coo.tocsr()

def scipy_to_torch_sparse(scipy_matrix, device=None):
    """
    Convert SciPy sparse matrix to PyTorch sparse tensor.
    
    Args:
        scipy_matrix: SciPy sparse matrix
        device: PyTorch device to place the tensor on
        
    Returns:
        PyTorch sparse tensor
    """
    import scipy.sparse as sp
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert to COO if not already
    if not isinstance(scipy_matrix, sp.coo_matrix):
        scipy_matrix = scipy_matrix.tocoo()
    
    # Get indices and values
    indices = torch.tensor(np.vstack((scipy_matrix.row, scipy_matrix.col)), 
                           dtype=torch.long, device=device)
    values = torch.tensor(scipy_matrix.data, dtype=torch.float, device=device)
    shape = torch.Size(scipy_matrix.shape)
    
    # Create sparse tensor
    return torch.sparse_coo_tensor(indices, values, shape)

def save_sparse_tensor(sparse_tensor, file_path):
    """
    Save a sparse tensor to disk efficiently.
    
    Args:
        sparse_tensor: Sparse tensor to save
        file_path: Path to save the tensor to
        
    Returns:
        True if successful
    """
    if not sparse_tensor.is_sparse:
        raise ValueError("Input must be a sparse tensor")
    
    # Make sure the tensor is coalesced
    sparse_tensor = sparse_tensor.coalesce()
    
    # Get components
    indices = sparse_tensor._indices().cpu()
    values = sparse_tensor._values().cpu()
    size = sparse_tensor.size()
    
    # Save components
    torch.save({
        'indices': indices,
        'values': values,
        'size': size
    }, file_path)
    
    return True

def load_sparse_tensor(file_path, device=None):
    """
    Load a sparse tensor from disk.
    
    Args:
        file_path: Path to load the tensor from
        device: Device to load the tensor to
        
    Returns:
        Loaded sparse tensor
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load components
    data = torch.load(file_path, map_location=device)
    
    # Create sparse tensor
    return torch.sparse_coo_tensor(
        data['indices'],
        data['values'],
        data['size']
    ).to(device)



# === FILE: asf\__core\__init__.py ===

from asf.core.enums import PerceptualInputType, EntityConfidenceState, PerceptualEventType


# === FILE: asf\__core\enums.py ===

import enum

class PerceptualInputType(enum.Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    NUMERICAL = "numerical"
    STRUCTURED = "structured"
    SENSOR = "sensor"
    MULTIMODAL = "multimodal"

class EntityConfidenceState(enum.Enum):
    UNVERIFIED = "unverified"  # Initial state, not yet validated
    PROVISIONAL = "provisional"  # Partially validated but requires further confirmation
    CANONICAL = "canonical"  # Fully validated entity

class PerceptualEventType(enum.Enum):
    NEW_INPUT = "new_input"
    PATTERN_DETECTED = "pattern_detected"
    ANOMALY_DETECTED = "anomaly_detected"
    ENTITY_PROMOTION = "entity_promotion"
    CONFIDENCE_UPDATE = "confidence_update"
    TEMPORAL_PATTERN = "temporal_pattern"
    CAUSAL_RELATION_DETECTED = "causal_relation_detected"



# === FILE: asf\__examples\active_inference_demo.py ===


import asyncio
import time
import logging
import json
import uuid
from asf.environmental_coupling.components.active_inference_controller import ActiveInferenceController
from asf.environmental_coupling.components.coupling_registry import SparseCouplingRegistry
from asf.environmental_coupling.models import EnvironmentalCoupling
from asf.environmental_coupling.enums import CouplingType, CouplingStrength, CouplingState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ActiveInferenceDemo")

class MockKnowledgeSubstrate:
    """Mock knowledge substrate for demo purposes."""
    
    async def get_entity(self, entity_id):
        """Return a mock entity."""
        return {
            'id': entity_id,
            'type': 'service',
            'properties': {
                'name': f'Entity {entity_id}',
                'description': 'Mock entity for demo'
            }
        }

class MockEnvironmentalEntity:
    """Mock environmental entity that responds to tests."""
    
    def __init__(self, entity_id, behavior_profile=None):
        self.entity_id = entity_id
        self.behavior_profile = behavior_profile or {}
        self.interaction_history = []
        
    async def process_test(self, test_parameters):
        """Process an active inference test and generate a response."""
        test_content = test_parameters.get('test_content', {})
        test_type = test_content.get('test_type', 'unknown')
        
        # Record interaction
        self.interaction_history.append({
            'timestamp': time.time(),
            'test_type': test_type,
            'test_id': test_parameters.get('test_id')
        })
        
        # Generate response based on test type
        if test_type == 'timing_test':
            # Simulate response time based on behavior profile
            response_time = self.behavior_profile.get('response_time', 1.0)
            # Add some randomness
            actual_time = response_time * (0.8 + 0.4 * random.random())
            # Simulate processing time
            await asyncio.sleep(actual_time)
            
            return {
                'response_time': actual_time,
                'timestamp': time.time(),
                'entity_id': self.entity_id,
                'status': 'success'
            }
            
        elif test_type == 'pattern_test':
            # Check if we recognize patterns
            pattern_recognition = self.behavior_profile.get('pattern_recognition', True)
            
            if pattern_recognition:
                # Get the sequence and expected next item
                sequence = test_content.get('sequence', [])
                expected_next = test_content.get('expected_next')
                
                # If our pattern recognition is fuzzy, sometimes get it wrong
                pattern_accuracy = self.behavior_profile.get('pattern_accuracy', 0.9)
                
                if random.random() <= pattern_accuracy:
                    next_value = expected_next
                else:
                    # Generate incorrect answer
                    next_value = expected_next + random.randint(1, 10)
                
                return {
                    'next_value': next_value,
                    'pattern_identified': True,
                    'entity_id': self.entity_id,
                    'status': 'success'
                }
            else:
                # Can't recognize patterns
                return {
                    'error': 'pattern_recognition_not_supported',
                    'entity_id': self.entity_id,
                    'status': 'error'
                }
                
        elif test_type == 'reliability_test':
            # Check reliability behavior
            reliability = self.behavior_profile.get('reliability', 0.95)
            
            if random.random() <= reliability:
                # Return the verification code correctly
                verification_code = test_content.get('verification_code')
                return {
                    'verification_code': verification_code,
                    'timestamp': time.time(),
                    'entity_id': self.entity_id,
                    'status': 'success'
                }
            else:
                # Return error or wrong code
                return {
                    'verification_code': 'invalid_code',
                    'error': 'verification_failed',
                    'entity_id': self.entity_id,
                    'status': 'error'
                }
                
        elif test_type == 'structure_test':
            # Check if we support structured data
            structure_support = self.behavior_profile.get('structure_support', True)
            
            if structure_support:
                # Get required fields and create response
                required_fields = test_content.get('required_fields', [])
                response = {
                    'entity_id': self.entity_id,
                    'timestamp': time.time(),
                    'status': 'success'
                }
                
                # Add required fields
                field_accuracy = self.behavior_profile.get('field_accuracy', 0.9)
                
                for field in required_fields:
                    if random.random() <= field_accuracy:
                        # Add field with correct type
                        if field == 'id':
                            response[field] = f"response_{uuid.uuid4().hex[:8]}"
                        elif field == 'timestamp':
                            response[field] = time.time()
                        elif field == 'value':
                            response[field] = random.randint(1, 100)
                        elif field == 'metadata':
                            response[field] = {'source': self.entity_id}
                        else:
                            response[field] = f"value_for_{field}"
                
                return response
            else:
                # Don't support structure tests
                return {
                    'error': 'structure_not_supported',
                    'entity_id': self.entity_id,
                    'status': 'error'
                }
                
        elif test_type == 'context_test':
            # Check if we handle contexts
            context_awareness = self.behavior_profile.get('context_awareness', True)
            
            if context_awareness:
                # Get context information
                context = test_content.get('context')
                expected_behavior = test_content.get('expected_behavior', {})
                
                # Determine our actual behavior based on profile
                context_accuracy = self.behavior_profile.get('context_accuracy', 0.9)
                
                if random.random() <= context_accuracy:
                    # Correct context recognition
                    actual_behavior = expected_behavior.copy()
                else:
                    # Incorrect context behavior
                    actual_behavior = {k: 'incorrect' for k in expected_behavior}
                
                return {
                    'context_acknowledged': context if random.random() <= context_accuracy else 'unknown',
                    'behavior': actual_behavior,
                    'entity_id': self.entity_id,
                    'status': 'success'
                }
            else:
                # Don't support context
                return {
                    'error': 'context_not_supported',
                    'entity_id': self.entity_id,
                    'status': 'error'
                }
                
        else:  # General test
            # Basic response
            echo_data = test_content.get('echo_data')
            return {
                'echo': echo_data,
                'response_time': self.behavior_profile.get('response_time', 1.0),
                'entity_id': self.entity_id,
                'status': 'success'
            }

async def run_active_inference_demo():
    """Run a demonstration of active inference capabilities."""
    logger.info("Starting Active Inference Demonstration")
    
    # Create components
    knowledge_substrate = MockKnowledgeSubstrate()
    coupling_registry = SparseCouplingRegistry()
    active_inference = ActiveInferenceController(knowledge_substrate)
    
    # Initialize components
    await coupling_registry.initialize()
    active_inference.set_coupling_registry(coupling_registry)
    
    # Create mock entities with different behavior profiles
    entities = {
        'reliable_entity': MockEnvironmentalEntity('reliable_entity', {
            'response_time': 0.8,
            'reliability': 0.95,
            'pattern_recognition': True,
            'pattern_accuracy': 0.9,
            'structure_support': True,
            'field_accuracy': 0.95,
            'context_awareness': True,
            'context_accuracy': 0.9
        }),
        'unreliable_entity': MockEnvironmentalEntity('unreliable_entity', {
            'response_time': 2.5,
            'reliability': 0.6,
            'pattern_recognition': True,
            'pattern_accuracy': 0.7,
            'structure_support': True,
            'field_accuracy': 0.7,
            'context_awareness': False
        }),
        'specialized_entity': MockEnvironmentalEntity('specialized_entity', {
            'response_time': 0.5,
            'reliability': 0.99,
            'pattern_recognition': False,
            'structure_support': False,
            'context_awareness': True,
            'context_accuracy': 0.99
        })
    }
    
    # Create couplings between internal entity and environmental entities
    couplings = {}
    
    for entity_id, entity in entities.items():
        coupling = EnvironmentalCoupling(
            id=f"coupling_{entity_id}",
            internal_entity_id='system_entity',
            environmental_entity_id=entity_id,
            coupling_type=CouplingType.INFORMATIONAL,
            coupling_strength=0.7,
            coupling_state=CouplingState.ACTIVE,
            bayesian_confidence=0.5,
            interaction_count=1
        )
        
        # Add to registry
        await coupling_registry.add_coupling(coupling)
        couplings[entity_id] = coupling
    
    logger.info(f"Created {len(couplings)} couplings with environmental entities")
    
    # Run active inference learning cycle for each coupling
    for entity_id, coupling in couplings.items():
        logger.info(f"\n{'=' * 40}\nRunning active inference cycle for {entity_id}\n{'=' * 40}")
        entity = entities[entity_id]
        
        # Run multiple tests to learn about entity behavior
        for i in range(5):
            logger.info(f"\nTest {i+1} for {entity_id}")
            
            # Generate test - focus on uncertainty
            test = await active_inference.generate_test_interaction(
                coupling.id,
                uncertainty_focus=True
            )
            
            # Log test details
            test_content = test.test_parameters.get('test_content', {})
            test_type = test_content.get('test_type', 'unknown')
            uncertainty_area = test.test_parameters.get('uncertainty_area', 'general')
            
            logger.info(f"Generated {test_type} test targeting {uncertainty_area}")
            
            # Process test through mock entity
            actual_result = await entity.process_test(test.test_parameters)
            
            # Evaluate test result
            evaluation = await active_inference.evaluate_test_result(test.id, actual_result)
            
            # Log results
            logger.info(f"Test information gain: {evaluation['information_gain']:.3f}")
            
            # Get updated uncertainty profile
            if coupling.id in active_inference.uncertainty_profiles:
                profile = active_inference.uncertainty_profiles[coupling.id]
                profile_summary = {k: round(v, 3) for k, v in profile.items() 
                                 if k not in ['last_updated', 'update_count']}
                
                logger.info(f"Updated uncertainty profile: {json.dumps(profile_summary, indent=2)}")
            
            # Brief pause between tests
            await asyncio.sleep(0.5)
        
        # Analyze what we learned about the entity
        if coupling.id in active_inference.uncertainty_profiles:
            profile = active_inference.uncertainty_profiles[coupling.id]
            
            # Find lowest and highest uncertainty areas
            areas = [
                'response_timing',
                'interaction_pattern',
                'reliability',
                'content_structure',
                'contextual_behavior'
            ]
            
            sorted_areas = sorted(areas, key=lambda area: profile[area])
            
            logger.info(f"\nAfter testing, learned that {entity_id}:")
            logger.info(f"- Most predictable in: {sorted_areas[0]} (uncertainty: {profile[sorted_areas[0]]:.3f})")
            logger.info(f"- Least predictable in: {sorted_areas[-1]} (uncertainty: {profile[sorted_areas[-1]]:.3f})")
            
            # Compare with actual behavior profile
            actual_behavior = entities[entity_id].behavior_profile
            logger.info("\nComparison with actual behavior profile:")
            
            if 'response_time' in actual_behavior:
                logger.info(f"- Actual response time: {actual_behavior['response_time']:.2f}s")
                
            if 'reliability' in actual_behavior:
                logger.info(f"- Actual reliability: {actual_behavior['reliability']:.2f}")
                
            if 'pattern_recognition' in actual_behavior:
                pattern_support = "Supported" if actual_behavior['pattern_recognition'] else "Not supported"
                logger.info(f"- Pattern recognition: {pattern_support}")
                
            if 'context_awareness' in actual_behavior:
                context_support = "Supported" if actual_behavior['context_awareness'] else "Not supported"
                logger.info(f"- Context awareness: {context_support}")
    
    logger.info("\nActive Inference Demonstration Completed")

if __name__ == "__main__":
    # Import random here to avoid scope issues
    import random
    
    # Run the demo
    asyncio.run(run_active_inference_demo())



# === FILE: asf\__examples\counterfactual_demo.py ===


import asyncio
import time
import logging
import random
import uuid
import json
from dataclasses import asdict

from asf.environmental_coupling.components.counterfactual_simulator import CounterfactualSimulator
from asf.environmental_coupling.models import EnvironmentalCoupling
from asf.environmental_coupling.enums import CouplingType, CouplingStrength, CouplingState

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CounterfactualDemo")

async def run_counterfactual_demonstration():
    """Run a demonstration of counterfactual simulation capabilities."""
    logger.info("Starting Counterfactual Simulation Demonstration")
    
    # Create simulator
    simulator = CounterfactualSimulator()
    
    # Create different types of couplings to test
    coupling_types = [
        ('informational', CouplingType.INFORMATIONAL),
        ('operational', CouplingType.OPERATIONAL),
        ('adaptive', CouplingType.ADAPTIVE),
        ('predictive', CouplingType.PREDICTIVE)
    ]
    
    for coupling_name, coupling_type in coupling_types:
        logger.info(f"\n{'=' * 40}\nTesting {coupling_name.upper()} coupling\n{'=' * 40}")
        
        # Create test coupling
        coupling = EnvironmentalCoupling(
            id=f"coupling_{coupling_name}_{int(time.time())}",
            internal_entity_id=f"internal_{coupling_name}",
            environmental_entity_id=f"external_{coupling_name}",
            coupling_type=coupling_type,
            coupling_strength=0.6,
            coupling_state=CouplingState.ACTIVE,
            bayesian_confidence=0.5,
            interaction_count=random.randint(5, 20)
        )
        
        # Add properties based on coupling type
        if coupling_type == CouplingType.INFORMATIONAL:
            coupling.properties = {
                'reliability': 0.7,
                'refresh_interval': 60
            }
        elif coupling_type == CouplingType.OPERATIONAL:
            coupling.properties = {
                'response_time': 1.2,
                'throughput': 50
            }
        elif coupling_type == CouplingType.ADAPTIVE:
            coupling.properties = {
                'learning_rate': 0.05,
                'adaptation_threshold': 0.3
            }
        elif coupling_type == CouplingType.PREDICTIVE:
            coupling.properties = {
                'prediction_horizon': 300,
                'min_confidence': 0.4
            }
        
        # Generate variations
        variations_count = 5
        logger.info(f"Generating {variations_count} counterfactual variations")
        
        variations = await simulator.generate_coupling_variations(coupling, variations_count)
        
        # Log variations
        for i, variation in enumerate(variations):
            logger.info(f"Variation {i+1}: {variation['variation_type']} - {variation['description']}")
            
            # Log key changes
            if 'coupling_type' in variation and variation['coupling_type'] != coupling.coupling_type:
                logger.info(f"  Changed type: {coupling.coupling_type} -> {variation['coupling_type']}")
            
            if 'coupling_strength' in variation and variation['coupling_strength'] != coupling.coupling_strength:
                logger.info(f"  Changed strength: {coupling.coupling_strength:.2f} -> {variation['coupling_strength']:.2f}")
            
            if 'properties' in variation:
                for key, value in variation['properties'].items():
                    if key in coupling.properties and coupling.properties[key] != value:
                        logger.info(f"  Changed property {key}: {coupling.properties[key]} -> {value}")
                    elif key not in coupling.properties:
                        logger.info(f"  Added property {key}: {value}")
        
        # Simulate outcomes
        logger.info("\nSimulating outcomes for variations")
        simulation_results = await simulator.simulate_outcomes(variations)
        
        # Log simulation results
        for i, result in enumerate(simulation_results):
            score = simulator._calculate_simulation_score(result['outcome'])
            logger.info(f"Variation {i+1} score: {score:.3f}")
            
            # Log key metrics
            outcome = result['outcome']
            logger.info(f"  Success rate: {outcome['success_rate']:.2f}")
            logger.info(f"  Efficiency: {outcome['efficiency']:.2f}")
            logger.info(f"  Response time: {outcome['response_time']:.2f}s")
            logger.info(f"  Prediction precision: {outcome['prediction_precision']:.2f}")
        
        # Identify optimal configuration
        logger.info("\nIdentifying optimal configuration")
        optimal = await simulator.identify_optimal_configuration(simulation_results)
        
        # Log optimal configuration
        opt_config = optimal['optimal_configuration']
        logger.info(f"Optimal configuration: {opt_config['variation_type']} - {opt_config['description']}")
        logger.info(f"Improvement over alternatives: {optimal['improvement']:.2f}x")
        
        # Simulate applying the configuration
        logger.info("\nSimulating application of optimal configuration")
        
        # Create a copy of the original coupling for comparison
        original_coupling_dict = {
            'type': coupling.coupling_type.name,
            'strength': coupling.coupling_strength,
            'properties': dict(coupling.properties)
        }
        
        # Apply changes from optimal configuration to the coupling
        if 'coupling_type' in opt_config:
            coupling.coupling_type = opt_config['coupling_type']
        
        if 'coupling_strength' in opt_config:
            coupling.coupling_strength = opt_config['coupling_strength']
        
        if 'properties' in opt_config:
            for key, value in opt_config['properties'].items():
                coupling.properties[key] = value
        
        # Log the changes
        logger.info("Changes applied to coupling:")
        if original_coupling_dict['type'] != coupling.coupling_type.name:
            logger.info(f"  Type: {original_coupling_dict['type']} -> {coupling.coupling_type.name}")
        
        if original_coupling_dict['strength'] != coupling.coupling_strength:
            logger.info(f"  Strength: {original_coupling_dict['strength']:.2f} -> {coupling.coupling_strength:.2f}")
        
        for key, value in coupling.properties.items():
            if key in original_coupling_dict['properties']:
                if original_coupling_dict['properties'][key] != value:
                    logger.info(f"  Property {key}: {original_coupling_dict['properties'][key]} -> {value}")
            else:
                logger.info(f"  Added property {key}: {value}")
        
        # Simulate recording actual outcome
        logger.info("\nSimulating actual outcome after applying configuration")
        
        # Create simulated actual outcome (similar to predicted but with some differences)
        predicted_outcome = optimal['predicted_outcome']
        
        # Apply some noise to represent real-world variance
        actual_outcome = {}
        for key, value in predicted_outcome.items():
            if key not in ['simulation_time', 'simulation_id', 'model_confidence']:
                # Add random noise ±15%
                noise_factor = 1.0 + random.uniform(-0.15, 0.15)
                actual_outcome[key] = value * noise_factor
        
        # Ensure values are in valid ranges
        for key in ['success_rate', 'efficiency', 'reliability', 'adaptability']:
            if key in actual_outcome:
                actual_outcome[key] = min(1.0, max(0.0, actual_outcome[key]))
        
        if 'response_time' in actual_outcome:
            actual_outcome['response_time'] = max(0.1, actual_outcome['response_time'])
        
        if 'prediction_precision' in actual_outcome:
            actual_outcome['prediction_precision'] = max(0.1, actual_outcome['prediction_precision'])
        
        # Record actual outcome
        await simulator.record_actual_outcome(
            coupling.id,
            {'predicted_outcome': predicted_outcome},
            actual_outcome
        )
        
        # Log actual vs predicted
        logger.info("Actual vs Predicted Outcome:")
        for key in ['success_rate', 'efficiency', 'response_time', 'prediction_precision', 'reliability']:
            if key in actual_outcome and key in predicted_outcome:
                logger.info(f"  {key}: {predicted_outcome[key]:.2f} (predicted) vs {actual_outcome[key]:.2f} (actual)")
        
        # Check model accuracy
        if coupling.id in simulator.simulation_models:
            model = simulator.simulation_models[coupling.id]
            logger.info(f"\nSimulation model accuracy: {model['accuracy']:.3f}")
            logger.info(f"Recorded outcomes: {len(model['outcomes'])}")
        
        logger.info(f"\nCompleted counterfactual simulation for {coupling_name.upper()} coupling")
        logger.info('-' * 80)

async def main():
    """Main demo function."""
    await run_counterfactual_demonstration()
    logger.info("Counterfactual demonstration completed")

if __name__ == "__main__":
    asyncio.run(main())



# === FILE: asf\__examples\integrated_predictive_demo.py ===


import asyncio
import time
import logging
import random
import json
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("IntegratedPredictiveDemo")

# Import necessary components
from asf.environmental_coupling.layer import EnvironmentalCouplingLayer
from asf.environmental_coupling.predictive_orchestrator import PredictiveProcessingOrchestrator
from asf.environmental_coupling.models import EnvironmentalCoupling
from asf.environmental_coupling.enums import CouplingType, CouplingStrength, CouplingState

class MockKnowledgeSubstrate:
    """Mock knowledge substrate for the demo."""
    
    async def get_entity(self, entity_id):
        """Get entity information."""
        return {
            'id': entity_id,
            'type': 'service',
            'properties': {
                'name': f'Entity {entity_id}',
                'created_at': time.time() - 3600  # Created an hour ago
            }
        }

class MockEnvironmentalEntity:
    """Mock environmental entity that can be interacted with."""
    
    def __init__(self, entity_id, behavior_profile=None):
        self.entity_id = entity_id
        self.behavior_profile = behavior_profile or {}
        self.state = {
            'last_update': time.time(),
            'interaction_count': 0,
            'data': {}
        }
        self.interaction_history = []
        logger.info(f"Created environmental entity {entity_id}")
    
    async def process_interaction(self, interaction_data, source_id=None):
        """Process an interaction from an internal entity."""
        self.interaction_history.append({
            'timestamp': time.time(),
            'source_id': source_id,
            'data': interaction_data
        })
        
        self.state['interaction_count'] += 1
        self.state['last_update'] = time.time()
        
        # Generate response based on behavior profile
        response_time = self.behavior_profile.get('response_time', 1.0)
        response_variability = self.behavior_profile.get('response_variability', 0.2)
        actual_response_time = response_time * (1 + random.uniform(-response_variability, response_variability))
        
        # Simulate processing time
        await asyncio.sleep(actual_response_time)
        
        # Generate response
        response = {
            'timestamp': time.time(),
            'entity_id': self.entity_id,
            'response_time': actual_response_time,
            'interaction_type': interaction_data.get('interaction_type', 'response'),
            'content_type': 'structured',
            'data': {
                'status': 'success',
                'source_id': source_id,
                'message': f"Processed interaction from {source_id}"
            }
        }
        
        # Add some predictable patterns based on interaction type
        if interaction_data.get('interaction_type') == 'query':
            response['data']['query_result'] = {'value': random.randint(1, 100)}
        elif interaction_data.get('interaction_type') == 'update':
            new_value = interaction_data.get('value', 0)
            self.state['data']['value'] = new_value
            response['data']['update_result'] = {'previous': self.state.get('value', 0), 'new': new_value}
        
        # Sometimes introduce errors based on reliability
        reliability = self.behavior_profile.get('reliability', 0.95)
        if random.random() > reliability:
            response['data']['status'] = 'error'
            response['data']['error'] = 'Random error occurred'
        
        logger.debug(f"Entity {self.entity_id} processed interaction from {source_id}")
        return response
    
    async def process_test(self, test_data, source_id=None):
        """Process an active inference test."""
        # Record test in history
        self.interaction_history.append({
            'timestamp': time.time(),
            'source_id': source_id,
            'test_data': test_data,
            'is_test': True
        })
        
        # Extract test parameters
        test_type = test_data.get('test_type', 'unknown')
        
        # Generate response based on test type and behavior profile
        if test_type == 'timing_test':
            # Respond with actual timing capabilities
            actual_time = self.behavior_profile.get('response_time', 1.0)
            variation = self.behavior_profile.get('response_variability', 0.2)
            
            # Wait the actual response time
            await asyncio.sleep(actual_time * (1 + random.uniform(-variation, variation)))
            
            return {
                'timestamp': time.time(),
                'response_time': actual_time,
                'variation': variation,
                'test_type': test_type,
                'entity_id': self.entity_id
            }
            
        elif test_type == 'reliability_test':
            # Respond with actual reliability information
            reliability = self.behavior_profile.get('reliability', 0.95)
            
            # Sometimes fail based on reliability
            if random.random() > reliability:
                return {
                    'timestamp': time.time(),
                    'test_type': test_type,
                    'status': 'error',
                    'entity_id': self.entity_id,
                    'error': 'Test failed due to reliability issues'
                }
            
            # Return validation code if requested
            validation_code = test_data.get('validation_code')
            return {
                'timestamp': time.time(),
                'test_type': test_type,
                'status': 'success',
                'entity_id': self.entity_id,
                'validation_code': validation_code,
                'reliability': reliability
            }
            
        elif test_type == 'pattern_test':
            # Check pattern recognition capabilities
            pattern_recognition = self.behavior_profile.get('pattern_recognition', True)
            
            if not pattern_recognition:
                return {
                    'timestamp': time.time(),
                    'test_type': test_type,
                    'status': 'error',
                    'entity_id': self.entity_id,
                    'error': 'Pattern recognition not supported'
                }
            
            # Get sequence and expected next value
            sequence = test_data.get('sequence', [])
            expected_next = test_data.get('expected_next')
            
            # Determine accuracy of pattern recognition
            pattern_accuracy = self.behavior_profile.get('pattern_accuracy', 0.9)
            
            if random.random() <= pattern_accuracy:
                # Correctly recognize pattern
                return {
                    'timestamp': time.time(),
                    'test_type': test_type,
                    'status': 'success',
                    'entity_id': self.entity_id,
                    'next_value': expected_next,
                    'pattern_recognized': True
                }
            else:
                # Incorrectly recognize pattern
                return {
                    'timestamp': time.time(),
                    'test_type': test_type,
                    'status': 'success',
                    'entity_id': self.entity_id,
                    'next_value': expected_next + random.randint(1, 5),
                    'pattern_recognized': False
                }
        
        # Default response for unknown test types
        return {
            'timestamp': time.time(),
            'test_type': test_type,
            'status': 'unknown_test_type',
            'entity_id': self.entity_id
        }

class MockDistributionLayer:
    """Mock distribution layer for the demo."""
    
    def __init__(self):
        self.entities = {}  # Maps entity_id to MockEnvironmentalEntity
        logger.info("Created distribution layer")
    
    def register_entity(self, entity):
        """Register an environmental entity."""
        self.entities[entity.entity_id] = entity
        logger.info(f"Registered entity {entity.entity_id} with distribution layer")
        return True
    
    async def distribute_entity(self, source_id, target_id, interaction_data, context=None):
        """Distribute an interaction to a target entity."""
        if target_id not in self.entities:
            logger.warning(f"Target entity {target_id} not found in distribution layer")
            return None
        
        entity = self.entities[target_id]
        
        # Check if this is a test or regular interaction
        if context and context.get('test_interaction'):
            return await entity.process_test(interaction_data.get('test_content', {}), source_id)
        else:
            return await entity.process_interaction(interaction_data, source_id)

async def setup_demo_environment():
    """Set up the demo environment with mock components."""
    
    # Create knowledge substrate
    knowledge_substrate = MockKnowledgeSubstrate()
    
    # Create mock environmental entities with different behavior profiles
    entities = {
        'reliable_entity': MockEnvironmentalEntity('reliable_entity', {
            'response_time': 0.5,
            'response_variability': 0.1,
            'reliability': 0.95,
            'pattern_recognition': True,
            'pattern_accuracy': 0.9
        }),
        'unreliable_entity': MockEnvironmentalEntity('unreliable_entity', {
            'response_time': 1.5,
            'response_variability': 0.4,
            'reliability': 0.6,
            'pattern_recognition': True,
            'pattern_accuracy': 0.7
        }),
        'slow_entity': MockEnvironmentalEntity('slow_entity', {
            'response_time': 2.0,
            'response_variability': 0.3,
            'reliability': 0.9,
            'pattern_recognition': False
        }),
        'fast_entity': MockEnvironmentalEntity('fast_entity', {
            'response_time': 0.2,
            'response_variability': 0.1,
            'reliability': 0.8,
            'pattern_recognition': True,
            'pattern_accuracy': 0.95
        })
    }
    
    # Create distribution layer
    distribution_layer = MockDistributionLayer()
    
    # Register entities with distribution layer
    for entity in entities.values():
        distribution_layer.register_entity(entity)
    
    # Create coupling layer
    coupling_layer = EnvironmentalCouplingLayer(knowledge_substrate)
    
    # Initialize layer
    await coupling_layer.initialize(layer6=distribution_layer)
    
    # Create orchestrator
    orchestrator = PredictiveProcessingOrchestrator(coupling_layer)
    await orchestrator.initialize()
    
    # Create internal entities
    internal_entities = ['system_entity_1', 'system_entity_2']
    
    # Register internal entities with orchestrator
    for entity_id in internal_entities:
        await orchestrator.register_entity(entity_id)
    
    # Create couplings between internal and environmental entities
    couplings = []
    
    # Entity 1 couples with reliable and unreliable entities
    couplings.append(EnvironmentalCoupling(
        id=str(time.time()) + "_coupling_1",
        internal_entity_id='system_entity_1',
        environmental_entity_id='reliable_entity',
        coupling_type=CouplingType.INFORMATIONAL,
        coupling_strength=0.7,
        coupling_state=CouplingState.ACTIVE
    ))
    
    couplings.append(EnvironmentalCoupling(
        id=str(time.time()) + "_coupling_2",
        internal_entity_id='system_entity_1',
        environmental_entity_id='unreliable_entity',
        coupling_type=CouplingType.CONTEXTUAL,
        coupling_strength=0.5,
        coupling_state=CouplingState.ACTIVE
    ))
    
    # Entity 2 couples with slow and fast entities
    couplings.append(EnvironmentalCoupling(
        id=str(time.time()) + "_coupling_3",
        internal_entity_id='system_entity_2',
        environmental_entity_id='slow_entity',
        coupling_type=CouplingType.ADAPTIVE,
        coupling_strength=0.6,
        coupling_state=CouplingState.ACTIVE
    ))
    
    couplings.append(EnvironmentalCoupling(
        id=str(time.time()) + "_coupling_4",
        internal_entity_id='system_entity_2',
        environmental_entity_id='fast_entity',
        coupling_type=CouplingType.PREDICTIVE,
        coupling_strength=0.8,
        coupling_state=CouplingState.ACTIVE
    ))
    
    # Register couplings with coupling layer
    for coupling in couplings:
        await coupling_layer.coupling_registry.add_coupling(coupling)
        logger.info(f"Added coupling {coupling.id} between {coupling.internal_entity_id} and {coupling.environmental_entity_id}")
    
    return {
        'knowledge_substrate': knowledge_substrate,
        'entities': entities,
        'distribution_layer': distribution_layer,
        'coupling_layer': coupling_layer,
        'orchestrator': orchestrator,
        'internal_entities': internal_entities,
        'couplings': couplings
    }

async def run_integrated_demo(duration=300):
    """Run the integrated predictive processing demonstration."""
    
    logger.info("Setting up demonstration environment...")
    env = await setup_demo_environment()
    
    logger.info(f"Running demonstration for {duration} seconds...")
    
    # Start adaptive cycles for entity 1
    entity1_task = asyncio.create_task(
        env['orchestrator'].run_adaptive_cycles(
            'system_entity_1',
            min_interval=10,
            max_interval=60
        )
    )
    
    # Start continuous cycles for entity 2
    entity2_task = asyncio.create_task(
        env['orchestrator'].run_continuous_cycles(
            'system_entity_2',
            interval=30
        )
    )
    
    # Run for specified duration
    try:
        await asyncio.sleep(duration)
    except asyncio.CancelledError:
        logger.info("Demo cancelled")
    finally:
        # Stop cycles
        entity1_task.cancel()
        entity2_task.cancel()
        
        try:
            await asyncio.gather(entity1_task, entity2_task, return_exceptions=True)
        except Exception as e:
            pass
    
    # Unregister entities
    for entity_id in env['internal_entities']:
        await env['orchestrator'].unregister_entity(entity_id)
    
    # Perform final maintenance
    maintenance_results = await env['orchestrator'].perform_maintenance()
    
    logger.info("Demonstration completed")
    logger.info(f"Final maintenance results: {json.dumps(maintenance_results, indent=2)}")
    
    # Return statistics
    return {
        'duration': duration,
        'entities_processed': len(env['internal_entities']),
        'environmental_entities': len(env['entities']),
        'couplings': len(env['couplings']),
        'maintenance': maintenance_results
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Integrated Predictive Processing Demo')
    parser.add_argument('--duration', type=int, default=300, help='Duration in seconds')
    args = parser.parse_args()
    
    logger.info(f"Starting Integrated Predictive Processing Demo ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    
    asyncio.run(run_integrated_demo(args.duration))



# === FILE: asf\__examples\precision_weighted_demo.py ===


import asyncio
import time
import logging
import random
import numpy as np
import matplotlib.pyplot as plt
from asf.environmental_coupling.components.predictive_modeler import PredictiveEnvironmentalModeler
from asf.environmental_coupling.components.enhanced_bayesian_updater import EnhancedBayesianUpdater

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PrecisionWeightedDemo")

async def simulate_precision_weighted_interactions(entity_id, coupling_id, count=15):
    """Simulate a sequence of interactions with precision-weighted updates."""
    # Initialize components
    modeler = PredictiveEnvironmentalModeler()
    updater = EnhancedBayesianUpdater()
    
    # Initialize Bayesian model
    await updater.initialize_coupling_model(coupling_id)
    
    logger.info(f"Starting precision-weighted simulation for {count} interactions")
    logger.info(f"Entity ID: {entity_id}, Coupling ID: {coupling_id}")
    
    # Track metrics
    precisions = []
    confidences = []
    errors = []
    
    # Phases of the simulation
    # Phase 1: Predictable entity (low errors)
    # Phase 2: Unpredictable entity (high errors)
    # Phase 3: Return to predictable (low errors)
    
    phase_lengths = [5, 5, 5]  # 5 interactions per phase
    current_phase = 0
    phase_interaction = 0
    
    for i in range(count):
        # Determine current phase
        if phase_interaction >= phase_lengths[current_phase]:
            current_phase = (current_phase + 1) % len(phase_lengths)
            phase_interaction = 0
            logger.info(f"Switching to phase {current_phase + 1}")
            
        phase_interaction += 1
        
        # Generate prediction
        context = {"phase": current_phase, "iteration": i}
        prediction = await modeler.predict_interaction(entity_id, context)
        
        logger.info(f"Step {i+1}: Generated prediction with confidence {prediction.confidence:.2f}")
        
        # Wait a bit
        await asyncio.sleep(0.5)
        
        # Generate actual data with accuracy dependent on phase
        if current_phase == 0 or current_phase == 2:  # Predictable phases
            # Low error - type matches, timing close
            error_level = 0.2
            actual_data = {
                "interaction_type": prediction.predicted_data['predicted_interaction_type'],
                "content_type": "text",
                "timestamp": prediction.predicted_data['predicted_timing'] + random.uniform(0, 10),
                "value": f"accurate data {i}"
            }
        else:  # Unpredictable phase
            # High error - different type, timing way off
            error_level = 0.8
            actual_data = {
                "interaction_type": "unexpected_type",
                "content_type": "binary",
                "timestamp": prediction.predicted_data['predicted_timing'] + random.uniform(30, 60),
                "value": f"unpredictable data {i}"
            }
            
        # Evaluate prediction
        evaluation = await modeler.evaluate_prediction(prediction.id, actual_data)
        logger.info(f"Evaluated prediction - Error: {evaluation['error']:.4f}")
        
        # Update precision
        await updater.update_precision(coupling_id, evaluation['error'])
        current_precision = updater.precision_values.get(coupling_id, 1.0)
        logger.info(f"Updated precision: {current_precision:.4f}")
        
        # Update Bayesian model with precision-weighted confidence
        update_result = await updater.update_from_interaction(
            coupling_id, 
            actual_data, 
            actual_data["interaction_type"], 
            0.7  # Base confidence
        )
        
        logger.info(f"Bayesian update - Prior: {update_result['prior_confidence']:.4f}, " +
                     f"Posterior: {update_result['new_confidence']:.4f}, " +
                     f"Weighted confidence: {update_result['weighted_confidence']:.4f}")
        
        # Track metrics
        precisions.append(current_precision)
        confidences.append(update_result['new_confidence'])
        errors.append(evaluation['error'])
        
        logger.info("-" * 60)
    
    # Show final results
    logger.info(f"Simulation completed - Final metrics:")
    logger.info(f"Final precision: {precisions[-1]:.4f}")
    logger.info(f"Final confidence: {confidences[-1]:.4f}")
    logger.info(f"Average error: {np.mean(errors):.4f}")
    
    # Visualize results if matplotlib is available
    try:
       
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(3, 1, 1)
        plt.plot(precisions, 'g-', label='Precision')
        plt.title('Precision Evolution')
        plt.legend()
        
        plt.subplot(3, 1, 2)
        plt.plot(confidences, 'b-', label='Bayesian Confidence')
        plt.title('Confidence Evolution')
        plt.legend()
        
        plt.subplot(3, 1, 3)
        plt.plot(errors, 'r-', label='Prediction Errors')
        plt.title('Prediction Errors')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('precision_weighted_simulation.png')
        logger.info("Saved visualization to precision_weighted_simulation.png")
    except ImportError:
        logger.info("Matplotlib not available for visualization")
    
    return {
        "precisions": precisions,
        "confidences": confidences,
        "errors": errors
    }

async def main():
    """Main demo function."""
    entity_id = "demo_entity_001"
    coupling_id = "demo_coupling_001"
    
    # Run simulation
    results = await simulate_precision_weighted_interactions(entity_id, coupling_id, count=15)
    
    logger.info("Demo completed successfully")

if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())



# === FILE: asf\__examples\prediction_demo.py ===

# === FILE: asf/examples/prediction_demo.py ===

import asyncio
import time
import logging
import random
from asf.environmental_coupling.components.predictive_modeler import PredictiveEnvironmentalModeler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PredictionDemo")

async def simulate_interactions(entity_id, count=10, modeler=None):
    """Simulate a sequence of interactions with an entity."""
    if modeler is None:
        modeler = PredictiveEnvironmentalModeler()
        
    interaction_types = ['query', 'update', 'notification', 'request']
    
    logger.info(f"Starting simulation with entity {entity_id} for {count} interactions")
    
    for i in range(count):
        # Generate a prediction
        context = {
            "last_interaction_type": interaction_types[i % len(interaction_types)],
            "iteration": i
        }
        prediction = await modeler.predict_interaction(entity_id, context)
        
        logger.info(f"Generated prediction {prediction.id} with confidence {prediction.confidence:.2f}")
        logger.info(f"Predicted interaction: {prediction.predicted_data['predicted_interaction_type']}")
        logger.info(f"Predicted timing: {time.ctime(prediction.predicted_data['predicted_timing'])}")
        
        # Simulate delay before actual interaction
        wait_time = random.uniform(0.5, 2.0)
        logger.info(f"Waiting {wait_time:.2f} seconds for actual interaction...")
        await asyncio.sleep(wait_time)
        
        # Generate actual interaction data (sometimes matching, sometimes not)
        prediction_correct = random.random() > 0.3  # 70% chance of correct prediction
        
        actual_data = {
            "interaction_type": prediction.predicted_data['predicted_interaction_type'] if prediction_correct else random.choice(interaction_types),
            "content_type": prediction.predicted_data['predicted_content'].get('type', 'text'),
            "timestamp": time.time(),
            "value": f"test data {i}"
        }
        
        # Evaluate prediction
        evaluation = await modeler.evaluate_prediction(prediction.id, actual_data)
        
        logger.info(f"Evaluated prediction - Error: {evaluation['error']:.4f}, Precision: {evaluation['precision']:.4f}")
        logger.info("-" * 60)
        
    # Show final statistics
    logger.info(f"Simulation completed for entity {entity_id}")
    logger.info(f"Final precision: {modeler.precision.get(entity_id, 0):.4f}")
    logger.info(f"Total predictions: {len(modeler.entity_predictions.get(entity_id, []))}")
    
async def main():
    """Main demo function."""
    # Create the modeler
    modeler = PredictiveEnvironmentalModeler()
    
    # Simulate interactions for a few entities
    await simulate_interactions("entity_001", count=5, modeler=modeler)
    await simulate_interactions("entity_002", count=10, modeler=modeler)
    
    # Verify we can retrieve predictions for specific entities
    entity_predictions = await modeler.get_predictions_for_entity("entity_001", limit=10)
    logger.info(f"Retrieved {len(entity_predictions)} predictions for entity_001")
    
    entity_predictions = await modeler.get_predictions_for_entity("entity_002", limit=10)
    logger.info(f"Retrieved {len(entity_predictions)} predictions for entity_002")

if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())



# === FILE: asf\__tests\test_active_inference.py ===


import asyncio
import time
import unittest
import logging
import random
import uuid
from unittest.mock import MagicMock, patch

from asf.environmental_coupling.components.active_inference_controller import ActiveInferenceController
from asf.environmental_coupling.models import EnvironmentalCoupling, ActiveInferenceTest
from asf.environmental_coupling.enums import CouplingType, CouplingStrength, CouplingState

# Configure logging
logging.basicConfig(level=logging.INFO)

class TestActiveInference(unittest.TestCase):
    
    def setUp(self):
        # Create mock knowledge substrate
        self.knowledge_substrate = MagicMock()
        
        # Create mock entity
        self.mock_entity = {
            'id': 'test_entity_001',
            'type': 'service',
            'properties': {},
            'created_at': time.time()
        }
        
        # Set up mock for get_entity method
        self.knowledge_substrate.get_entity = MagicMock(return_value=self.mock_entity)
        
        # Create active inference controller with mock substrate
        self.active_inference = ActiveInferenceController(self.knowledge_substrate)
        
        # Create mock coupling registry
        self.mock_registry = MagicMock()
        
        # Set up mock for get_coupling method
        self.mock_coupling = EnvironmentalCoupling(
            id='test_coupling_001',
            internal_entity_id='test_entity_001',
            environmental_entity_id='env_entity_001',
            coupling_type=CouplingType.INFORMATIONAL,
            coupling_strength=0.7,
            coupling_state=CouplingState.ACTIVE,
            bayesian_confidence=0.6,
            interaction_count=5,
            prediction_precision=1.0
        )
        
        self.mock_registry.get_coupling = MagicMock(return_value=self.mock_coupling)
        self.mock_registry.update_coupling = MagicMock(return_value=True)
        
        # Connect registry to controller
        self.active_inference.set_coupling_registry(self.mock_registry)
    
    def test_active_inference_test_generation(self):
        # Run the test in event loop
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self._async_test_inference_generation())
        
    async def _async_test_inference_generation(self):
        """Test the generation of active inference tests."""
        # Generate a test with uncertainty focus
        test = await self.active_inference.generate_test_interaction(
            'test_coupling_001',
            uncertainty_focus=True
        )
        
        # Verify test structure
        self.assertIsNotNone(test)
        self.assertIsInstance(test, ActiveInferenceTest)
        self.assertEqual(test.coupling_id, 'test_coupling_001')
        self.assertIsNotNone(test.test_parameters)
        
        # Verify test parameters
        params = test.test_parameters
        self.assertIn('interaction_type', params)
        self.assertEqual(params['interaction_type'], 'active_inference_test')
        self.assertIn('uncertainty_area', params)
        self.assertIn('test_content', params)
        self.assertIn('prediction', params)
        
        # Check that test info is recorded in history
        self.assertIn('test_coupling_001', self.active_inference.inference_history)
        self.assertEqual(len(self.active_inference.inference_history['test_coupling_001']), 1)
        
        # Generate a standard test
        test2 = await self.active_inference.generate_test_interaction(
            'test_coupling_001',
            uncertainty_focus=False
        )
        
        # Verify standard test
        self.assertIsNotNone(test2)
        self.assertEqual(test2.test_parameters['uncertainty_area'], 'general')
        
        # Verify history is updated
        self.assertEqual(len(self.active_inference.inference_history['test_coupling_001']), 2)
        
        return test
    
    def test_test_result_evaluation(self):
        # Run the test in event loop
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self._async_test_result_evaluation())
    
    async def _async_test_result_evaluation(self):
        """Test the evaluation of active inference test results."""
        # First generate a test
        test = await self.active_inference.generate_test_interaction(
            'test_coupling_001',
            uncertainty_focus=True
        )
        
        # Get the test type and area
        test_content = test.test_parameters.get('test_content', {})
        test_type = test_content.get('test_type', 'unknown')
        uncertainty_area = test.test_parameters.get('uncertainty_area', 'general')
        
        # Create a mock result based on test type
        if test_type == 'timing_test':
            actual_result = {
                'response_time': 1.2,  # Good response time
                'timestamp': time.time(),
                'status': 'success'
            }
        elif test_type == 'pattern_test':
            expected_next = test_content.get('expected_next')
            actual_result = {
                'next_value': expected_next,  # Correct pattern continuation
                'pattern_identified': True,
                'status': 'success'
            }
        elif test_type == 'reliability_test':
            verification_code = test_content.get('verification_code')
            actual_result = {
                'verification_code': verification_code,  # Perfect echo
                'status': 'success'
            }
        elif test_type == 'structure_test':
            # Create result with required fields
            actual_result = {
                'id': 'response_123',
                'timestamp': time.time(),
                'value': 42,
                'metadata': {'source': 'test'},
                'status': 'success'
            }
        elif test_type == 'context_test':
            context = test_content.get('context')
            expected_behavior = test_content.get('expected_behavior', {})
            actual_result = {
                'context_acknowledged': context,
                'behavior': expected_behavior.copy(),  # Perfect behavior match
                'status': 'success'
            }
        else:  # general test
            echo_data = test_content.get('echo_data')
            actual_result = {
                'echo': echo_data,
                'response_time': 1.5,
                'status': 'success'
            }
            
        # Evaluate the test result
        evaluation = await self.active_inference.evaluate_test_result(test.id, actual_result)
        
        # Verify evaluation structure
        self.assertTrue(evaluation.get('success', False))
        self.assertEqual(evaluation.get('test_id'), test.id)
        self.assertIn('information_gain', evaluation)
        self.assertIn('target_area', evaluation)
        self.assertIn('coupling_updates', evaluation)
        
        # Verify high information gain for perfect results
        self.assertGreater(evaluation['information_gain'], 0.7)
        
        # Check that the uncertainty profile was updated
        self.assertIn('test_coupling_001', self.active_inference.uncertainty_profiles)
        profile = self.active_inference.uncertainty_profiles['test_coupling_001']
        self.assertIn(uncertainty_area, profile)
        
        # Verify that coupling was updated
        self.mock_registry.update_coupling.assert_called()
        
        # Test with imperfect results
        if test_type == 'timing_test':
            bad_result = {
                'response_time': 10.0,  # Much slower than expected
                'status': 'delayed'
            }
        elif test_type == 'pattern_test':
            bad_result = {
                'next_value': 999,  # Wrong pattern continuation
                'status': 'error'
            }
        elif test_type == 'reliability_test':
            bad_result = {
                'verification_code': 'wrong_code',  # Incorrect echo
                'status': 'error'
            }
        else:
            bad_result = {
                'status': 'error',
                'message': 'Test failed'
            }
        
        # Evaluate with bad result
        bad_evaluation = await self.active_inference.evaluate_test_result(test.id, bad_result)
        
        # Verify lower information gain for bad results
        self.assertLess(bad_evaluation['information_gain'], evaluation['information_gain'])
        
        return evaluation
    
    def test_uncertainty_profile_creation(self):
        # Run the test in event loop
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self._async_test_uncertainty_profile())
    
    async def _async_test_uncertainty_profile(self):
        """Test the creation and updating of uncertainty profiles."""
        # Create initial uncertainty profile
        await self.active_inference._create_uncertainty_profile('test_coupling_001', self.mock_coupling)
        
        # Verify profile exists
        self.assertIn('test_coupling_001', self.active_inference.uncertainty_profiles)
        
        # Verify profile structure
        profile = self.active_inference.uncertainty_profiles['test_coupling_001']
        self.assertIn('response_timing', profile)
        self.assertIn('interaction_pattern', profile)
        self.assertIn('reliability', profile)
        self.assertIn('content_structure', profile)
        self.assertIn('contextual_behavior', profile)
        
        # Verify initial values are reasonable
        for area in ['response_timing', 'interaction_pattern', 'reliability', 'content_structure', 'contextual_behavior']:
            self.assertGreaterEqual(profile[area], 0.1)
            self.assertLessEqual(profile[area], 1.0)
        
        # Test analyzing uncertainty
        uncertainty_areas = await self.active_inference._analyze_coupling_uncertainty(self.mock_coupling)
        
        # Verify analysis returns a sorted list of areas
        self.assertIsInstance(uncertainty_areas, list)
        self.assertEqual(len(uncertainty_areas), 5)  # All 5 areas should be included
        
        # Verify that uncertainty updates after tests
        original_uncertainty = profile['response_timing']
        
        # Simulate a test that reduced uncertainty
        await self.active_inference._update_uncertainty_profile(
            'test_coupling_001',
            'response_timing',
            0.9,  # High information gain
            {'response_time': 1.5}
        )
        
        # Verify uncertainty decreased
        self.assertLess(profile['response_timing'], original_uncertainty)
        
        # Simulate a test that increased uncertainty
        original_uncertainty = profile['reliability']
        
        await self.active_inference._update_uncertainty_profile(
            'test_coupling_001',
            'reliability',
            0.2,  # Low information gain
            {'verification_failed': True}
        )
        
        # Verify uncertainty increased or stayed high
        self.assertGreaterEqual(profile['reliability'], original_uncertainty)
        
        return profile

if __name__ == "__main__":
    unittest.main()



# === FILE: asf\__tests\test_counterfactual_simulation.py ===


import asyncio
import time
import unittest
import logging
import random
from unittest.mock import MagicMock

from asf.environmental_coupling.components.counterfactual_simulator import CounterfactualSimulator
from asf.environmental_coupling.models import EnvironmentalCoupling
from asf.environmental_coupling.enums import CouplingType, CouplingStrength, CouplingState

# Configure logging
logging.basicConfig(level=logging.INFO)

class TestCounterfactualSimulation(unittest.TestCase):
    
    def setUp(self):
        self.simulator = CounterfactualSimulator()
        
        # Create test coupling
        self.test_coupling = EnvironmentalCoupling(
            id='test_coupling_001',
            internal_entity_id='test_entity_001',
            environmental_entity_id='env_entity_001',
            coupling_type=CouplingType.INFORMATIONAL,
            coupling_strength=0.7,
            coupling_state=CouplingState.ACTIVE,
            bayesian_confidence=0.6
        )
        self.test_coupling.properties = {
            'response_time': 1.5,
            'reliability': 0.8
        }
    
    def test_variation_generation(self):
        # Run the test in event loop
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self._async_test_variation_generation())
    
    async def _async_test_variation_generation(self):
        """Test the generation of counterfactual variations."""
        # Initialize variation templates
        self.simulator._initialize_variation_templates()
        
        # Generate variations
        variations = await self.simulator.generate_coupling_variations(self.test_coupling, 3)
        
        # Verify variations were generated
        self.assertEqual(len(variations), 3, "Should generate 3 variations")
        
        # Check variation structure
        for i, variation in enumerate(variations):
            self.assertIn('id', variation)
            self.assertEqual(variation['base_coupling_id'], self.test_coupling.id)
            self.assertIn('variation_type', variation)
            self.assertIn('description', variation)
            self.assertEqual(variation['variation_index'], i)
        
        # Check that variations are different from each other
        variation_types = [v['variation_type'] for v in variations]
        self.assertGreater(len(set(variation_types)), 1, "Should generate different variation types")
        
        return variations
    
    def test_outcome_simulation(self):
        # Run the test in event loop
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self._async_test_outcome_simulation())
    
    async def _async_test_outcome_simulation(self):
        """Test the simulation of outcomes for variations."""
        # Generate variations
        variations = await self.simulator.generate_coupling_variations(self.test_coupling, 3)
        
        # Simulate outcomes
        simulation_results = await self.simulator.simulate_outcomes(variations)
        
        # Verify outcomes were generated
        self.assertEqual(len(simulation_results), 3, "Should generate 3 outcomes")
        
        # Check outcome structure
        for result in simulation_results:
            self.assertIn('variation', result)
            self.assertIn('outcome', result)
            
            outcome = result['outcome']
            self.assertIn('success_rate', outcome)
            self.assertIn('efficiency', outcome)
            self.assertIn('response_time', outcome)
            self.assertIn('prediction_precision', outcome)
            self.assertIn('reliability', outcome)
            self.assertIn('adaptability', outcome)
            
            # Verify values are in valid ranges
            self.assertGreaterEqual(outcome['success_rate'], 0.0)
            self.assertLessEqual(outcome['success_rate'], 1.0)
            self.assertGreaterEqual(outcome['response_time'], 0.1)
        
        return simulation_results
    
    def test_optimal_configuration(self):
        # Run the test in event loop
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self._async_test_optimal_configuration())
    
    async def _async_test_optimal_configuration(self):
        """Test the identification of optimal configuration."""
        # Generate variations and outcomes
        variations = await self.simulator.generate_coupling_variations(self.test_coupling, 5)
        simulation_results = await self.simulator.simulate_outcomes(variations)
        
        # Identify optimal configuration
        optimal = await self.simulator.identify_optimal_configuration(simulation_results)
        
        # Verify optimal configuration
        self.assertIsNotNone(optimal)
        self.assertIn('optimal_configuration', optimal)
        self.assertIn('predicted_outcome', optimal)
        self.assertIn('improvement', optimal)
        
        # Check that the optimal configuration has the best score
        optimal_score = self.simulator._calculate_simulation_score(optimal['predicted_outcome'])
        
        for result in simulation_results:
            score = self.simulator._calculate_simulation_score(result['outcome'])
            self.assertLessEqual(score, optimal_score, "Optimal score should be highest")
        
        return optimal
    
    def test_actual_outcome_recording(self):
        # Run the test in event loop
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self._async_test_actual_outcome_recording())
    
    async def _async_test_actual_outcome_recording(self):
        """Test recording of actual outcomes to improve simulation accuracy."""
        # Generate a variation
        variations = await self.simulator.generate_coupling_variations(self.test_coupling, 1)
        variation = variations[0]
        
        # Simulate outcome
        simulation_results = await self.simulator.simulate_outcomes(variations)
        predicted_outcome = simulation_results[0]['outcome']
        
        # Create mock actual outcome (slightly different from predicted)
        actual_outcome = dict(predicted_outcome)
        actual_outcome['success_rate'] = predicted_outcome['success_rate'] * 0.9  # 10% lower
        actual_outcome['response_time'] = predicted_outcome['response_time'] * 1.1  # 10% higher
        
        # Record actual outcome
        result = await self.simulator.record_actual_outcome(
            self.test_coupling.id,
            {'predicted_outcome': predicted_outcome},
            actual_outcome
        )
        
        # Verify recording
        self.assertIn('prediction_error', result)
        self.assertIn('model_accuracy', result)
        self.assertEqual(result['outcomes_recorded'], 1)
        
        # Check that model was updated
        self.assertIn(self.test_coupling.id, self.simulator.simulation_models)
        model = self.simulator.simulation_models[self.test_coupling.id]
        self.assertEqual(len(model['outcomes']), 1)
        
        # Record another outcome
        result2 = await self.simulator.record_actual_outcome(
            self.test_coupling.id,
            {'predicted_outcome': predicted_outcome},
            actual_outcome
        )
        
        # Verify second recording
        self.assertEqual(result2['outcomes_recorded'], 2)
        
        return result

if __name__ == "__main__":
    unittest.main()



# === FILE: asf\__tests\test_precision_weighting.py ===


import asyncio
import time
import unittest
import logging
import numpy as np
from asf.environmental_coupling.components.predictive_modeler import PredictiveEnvironmentalModeler
from asf.environmental_coupling.components.enhanced_bayesian_updater import EnhancedBayesianUpdater

# Configure logging
logging.basicConfig(level=logging.INFO)

class TestPrecisionWeighting(unittest.TestCase):
    
    def setUp(self):
        self.modeler = PredictiveEnvironmentalModeler()
        self.bayesian_updater = EnhancedBayesianUpdater()
        
    def test_precision_calculation(self):
        # Run async test in event loop
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self._async_test_precision_calculation())
        
    async def _async_test_precision_calculation(self):
        # Test entity and coupling
        entity_id = "test_entity_precision"
        coupling_id = "test_coupling_precision"
        
        # Initialize Bayesian model
        await self.bayesian_updater.initialize_coupling_model(coupling_id)
        
        # Generate several predictions with varying accuracy
        initial_precision = self.modeler.precision.get(entity_id, 1.0)
        print(f"Initial precision: {initial_precision}")
        
        # First set of accurate predictions
        for i in range(5):
            prediction = await self.modeler.predict_interaction(entity_id)
            
            # Create actual data with low error (high accuracy)
            actual_data = {
                "interaction_type": prediction.predicted_data['predicted_interaction_type'],
                "content_type": "text",
                "timestamp": prediction.predicted_data['predicted_timing'] + 5,  # Only 5 seconds off
                "value": f"test data {i}"
            }
            
            # Evaluate prediction
            evaluation = await self.modeler.evaluate_prediction(prediction.id, actual_data)
            
            # Update Bayesian model with precision
            await self.bayesian_updater.update_precision(coupling_id, evaluation['error'])
            
            print(f"Iteration {i} (accurate): Error={evaluation['error']:.3f}, Precision={evaluation['precision']:.3f}")
            
        # Get precision after accurate predictions
        mid_precision = self.modeler.precision.get(entity_id, 1.0)
        print(f"Precision after accurate predictions: {mid_precision}")
        self.assertGreater(mid_precision, initial_precision, "Precision should increase after accurate predictions")
        
        # Second set of inaccurate predictions
        for i in range(5):
            prediction = await self.modeler.predict_interaction(entity_id)
            
            # Create actual data with high error (low accuracy)
            actual_data = {
                "interaction_type": "unexpected_type",  # Wrong type
                "content_type": "binary",  # Wrong content
                "timestamp": prediction.predicted_data['predicted_timing'] + 60,  # 60 seconds off
                "value": f"unexpected data {i}"
            }
            
            # Evaluate prediction
            evaluation = await self.modeler.evaluate_prediction(prediction.id, actual_data)
            
            # Update Bayesian model with precision
            await self.bayesian_updater.update_precision(coupling_id, evaluation['error'])
            
            print(f"Iteration {i} (inaccurate): Error={evaluation['error']:.3f}, Precision={evaluation['precision']:.3f}")
            
        # Get final precision after inaccurate predictions
        final_precision = self.modeler.precision.get(entity_id, 1.0)
        print(f"Final precision: {final_precision}")
        self.assertLess(final_precision, mid_precision, "Precision should decrease after inaccurate predictions")
        
    def test_precision_weighted_bayesian_updating(self):
        # Run async test in event loop
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self._async_test_precision_weighted_bayesian_updating())
        
    async def _async_test_precision_weighted_bayesian_updating(self):
        # Test coupling
        coupling_id = "test_coupling_bayes"
        
        # Initialize Bayesian model
        await self.bayesian_updater.initialize_coupling_model(coupling_id)
        
        # Test multiple interactions with varying precision
        precisions = [0.5, 1.0, 2.0, 5.0]
        
        for precision in precisions:
            # Manually set precision
            self.bayesian_updater.precision_values[coupling_id] = precision
            
            # Create interaction data
            interaction_data = {
                "interaction_type": "query",
                "content": "test content",
                "timestamp": time.time()
            }
            
            # Standard confidence
            confidence = 0.7
            
            # Update Bayesian model
            result = await self.bayesian_updater.update_from_interaction(
                coupling_id, interaction_data, "query", confidence
            )
            
            # Print results
            print(f"Precision {precision}:")
            print(f"  Prior: {result['prior_confidence']:.4f}")
            print(f"  Weighted confidence: {result['weighted_confidence']:.4f}")
            print(f"  Posterior: {result['new_confidence']:.4f}")
            print(f"  Strength delta: {result['strength_delta']:.4f}")
            
            # Verify that higher precision leads to larger confidence changes
            if precision > 1.0:
                self.assertGreater(
                    abs(result['strength_delta']), 
                    0.05, 
                    "Higher precision should lead to larger confidence changes"
                )

if __name__ == "__main__":
    unittest.main()



# === FILE: asf\__tests\test_prediction_framework.py ===

# === FILE: asf/tests/test_prediction_framework.py ===

import asyncio
import time
import unittest
import logging
from asf.environmental_coupling.components.predictive_modeler import PredictiveEnvironmentalModeler

# Configure logging
logging.basicConfig(level=logging.INFO)

class TestPredictionFramework(unittest.TestCase):
    
    def setUp(self):
        self.modeler = PredictiveEnvironmentalModeler()
        
    def test_prediction_generation(self):
        # Run async test in event loop
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self._async_test_prediction_generation())
        
    async def _async_test_prediction_generation(self):
        # Test entity
        entity_id = "test_entity_1"
        
        # Generate a prediction
        context = {"last_interaction_type": "query"}
        prediction = await self.modeler.predict_interaction(entity_id, context)
        
        # Validate prediction
        self.assertIsNotNone(prediction)
        self.assertEqual(prediction.environmental_entity_id, entity_id)
        self.assertGreaterEqual(prediction.confidence, 0.3)
        self.assertLessEqual(prediction.confidence, 0.9)
        
        # Check prediction structure
        self.assertIn('predicted_interaction_type', prediction.predicted_data)
        self.assertIn('predicted_content', prediction.predicted_data)
        self.assertIn('predicted_timing', prediction.predicted_data)
        
        # Validate it's stored properly
        self.assertIn(prediction.id, self.modeler.predictions)
        self.assertIn(prediction.id, self.modeler.entity_predictions[entity_id])
        
    def test_prediction_evaluation(self):
        # Run async test in event loop
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self._async_test_prediction_evaluation())
        
    async def _async_test_prediction_evaluation(self):
        # Test entity
        entity_id = "test_entity_2"
        
        # Generate a prediction
        prediction = await self.modeler.predict_interaction(entity_id)
        
        # Create some actual data
        actual_data = {
            "interaction_type": prediction.predicted_data['predicted_interaction_type'],
            "content_type": "text",
            "timestamp": time.time(),
            "value": "test data"
        }
        
        # Evaluate prediction
        evaluation = await self.modeler.evaluate_prediction(prediction.id, actual_data)
        
        # Validate evaluation
        self.assertIsNotNone(evaluation)
        self.assertEqual(evaluation['prediction_id'], prediction.id)
        self.assertIn('error', evaluation)
        self.assertIn('precision', evaluation)
        
        # Check prediction was updated
        updated_prediction = self.modeler.predictions[prediction.id]
        self.assertIsNotNone(updated_prediction.verification_time)
        self.assertIsNotNone(updated_prediction.prediction_error)
        
        # Check precision updating
        self.assertIn(entity_id, self.modeler.precision)
        
    def test_precision_calculation(self):
        # Run async test in event loop
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self._async_test_precision_calculation())
        
    async def _async_test_precision_calculation(self):
        # Test entity
        entity_id = "test_entity_3"
        
        # Generate multiple predictions with varying accuracy
        for i in range(10):
            prediction = await self.modeler.predict_interaction(entity_id)
            
            # Create actual data with controlled error
            error_level = 0.2 if i < 5 else 0.8  # First 5 accurate, next 5 inaccurate
            
            # Interaction type matches prediction half the time
            type_match = i % 2 == 0
            
            actual_data = {
                "interaction_type": prediction.predicted_data['predicted_interaction_type'] if type_match else "unexpected_type",
                "content_type": "text",
                "timestamp": prediction.predicted_data['predicted_timing'] + (5 if error_level < 0.5 else 30),
                "value": f"test data {i}"
            }
            
            # Evaluate prediction
            evaluation = await self.modeler.evaluate_prediction(prediction.id, actual_data)
            
            # Print current precision for debugging
            print(f"Iteration {i}: Error={evaluation['error']:.3f}, Precision={evaluation['precision']:.3f}")
            
        # Final precision should reflect the pattern of errors
        final_precision = self.modeler.precision.get(entity_id, 0)
        print(f"Final precision: {final_precision:.4f}")
        
        # Precision should be meaningful (not too high or too low)
        self.assertGreater(final_precision, 0.1)
        self.assertLess(final_precision, 10.0)

if __name__ == "__main__":
    unittest.main()

