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
