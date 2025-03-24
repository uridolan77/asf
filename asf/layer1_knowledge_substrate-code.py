

# === FILE: layer1_knowledge_substrate\__init__.py ===

from asf.knowledge_substrate.knowledge_substrate import KnowledgeSubstrateLayer
from asf.knowledge_substrate.predictive_processor import PredictiveProcessor
from asf.knowledge_substrate.confidence.predictive_updater import PredictiveConfidenceUpdater


# === FILE: layer1_knowledge_substrate\chronograph_gnosis_layer.py ===




# === FILE: layer1_knowledge_substrate\chronograph_middleware_layer.py ===




# === FILE: layer1_knowledge_substrate\knowledge_substrate_layer.py ===

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



# === FILE: layer1_knowledge_substrate\predictive_processor.py ===

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



# === FILE: layer1_knowledge_substrate\causal\graph.py ===

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



# === FILE: layer1_knowledge_substrate\causal\representation.py ===

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



# === FILE: layer1_knowledge_substrate\causal\variable.py ===

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



# === FILE: layer1_knowledge_substrate\confidence\__init__.py ===

from asf.knowledge_substrate.confidence.bayesian_updater import BayesianConfidenceUpdater


# === FILE: layer1_knowledge_substrate\confidence\bayesian_updater.py ===

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



# === FILE: layer1_knowledge_substrate\confidence\predictive_updater.py ===

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



# === FILE: layer1_knowledge_substrate\extraction\base.py ===

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



# === FILE: layer1_knowledge_substrate\extraction\image.py ===

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



# === FILE: layer1_knowledge_substrate\extraction\text.py ===

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



# === FILE: layer1_knowledge_substrate\linking\__init__.py ===




# === FILE: layer1_knowledge_substrate\linking\entity_gnn.py ===

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



# === FILE: layer1_knowledge_substrate\memory\energy_based.py ===

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



# === FILE: layer1_knowledge_substrate\perception\__init__.py ===

from asf.knowledge_substrate.perception.feature import PerceptualFeature
from asf.knowledge_substrate.perception.entity import PerceptualEntity



# === FILE: layer1_knowledge_substrate\perception\entity.py ===

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



# === FILE: layer1_knowledge_substrate\perception\feature.py ===

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



# === FILE: layer1_knowledge_substrate\semiotic\feature_weighter.py ===

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



# === FILE: layer1_knowledge_substrate\temporal\__init__.py ===

from asf.knowledge_substrate.temporal.adaptive_metadata import AdaptiveTemporalMetadata
from asf.knowledge_substrate.temporal.sequence import TemporalSequence
from asf.knowledge_substrate.temporal.pattern_detector import TemporalPatternDetector
from asf.knowledge_substrate.temporal.processing_engine import TemporalProcessingEngine
from asf.knowledge_substrate.temporal.predictive_engine import PredictiveTemporalEngine



# === FILE: layer1_knowledge_substrate\temporal\adaptive_metadata.py ===

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



# === FILE: layer1_knowledge_substrate\temporal\pattern_detector.py ===

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



# === FILE: layer1_knowledge_substrate\temporal\predictive_engine.py ===

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



# === FILE: layer1_knowledge_substrate\temporal\processing_engine.py ===

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



# === FILE: layer1_knowledge_substrate\temporal\sequence.py ===

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

