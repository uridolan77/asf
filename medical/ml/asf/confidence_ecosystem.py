"""
Confidence Ecosystem Module

This module implements the Dynamic Confidence Ecosystem component of the ASF framework,
which manages confidence scores for knowledge elements and their evolution over time.
"""

import math
import time
from typing import Dict, List, Optional, Tuple, Union, Any


class ConfidenceEcosystem:
    """
    Manages dynamic confidence scores for knowledge elements.
    
    The Confidence Ecosystem tracks, updates, and utilizes confidence scores for knowledge
    elements, allowing them to evolve over time based on new evidence and temporal decay.
    This is a core component of the ASF framework that enables dynamic knowledge representation.
    """
    
    def __init__(
        self, 
        decay_rate: float = 0.01, 
        min_confidence: float = 0.1, 
        max_confidence: float = 1.0
    ):
        """
        Initialize the Confidence Ecosystem.
        
        Args:
            decay_rate: Rate at which confidence decays over time (default: 0.01)
            min_confidence: Minimum confidence value (default: 0.1)
            max_confidence: Maximum confidence value (default: 1.0)
        """
        self.confidence_store: Dict[str, float] = {}  # Map of entity_id -> confidence_score
        self.metadata_store: Dict[str, Dict[str, Any]] = {}  # Map of entity_id -> metadata
        self.update_history: Dict[str, List[Tuple[float, float]]] = {}  # Track confidence updates
        
        self.decay_rate = decay_rate
        self.min_confidence = min_confidence
        self.max_confidence = max_confidence
    
    def initialize_confidence(self, entity_id: str, initial_confidence: float = 0.5) -> float:
        """
        Initialize confidence for a new entity.
        
        Args:
            entity_id: Entity ID
            initial_confidence: Initial confidence score (default: 0.5)
            
        Returns:
            Initial confidence value
        """
        # Ensure confidence is within bounds
        initial_confidence = min(self.max_confidence, max(self.min_confidence, initial_confidence))
        
        # Store confidence
        self.confidence_store[entity_id] = initial_confidence
        
        # Initialize metadata
        self.metadata_store[entity_id] = {
            "created_at": time.time(),
            "last_updated": time.time(),
            "update_count": 0,
            "source": "initialization"
        }
        
        # Initialize update history
        self.update_history[entity_id] = [(time.time(), initial_confidence)]
        
        return initial_confidence
    
    def get_confidence(self, entity_id: str, current_time: Optional[float] = None) -> float:
        """
        Get current confidence score with temporal decay applied.
        
        Args:
            entity_id: Entity ID
            current_time: Current time (defaults to now)
            
        Returns:
            Current confidence score
        """
        if entity_id not in self.confidence_store:
            return 0.0
            
        base_confidence = self.confidence_store[entity_id]
        
        if current_time is None:
            current_time = time.time()
            
        # Apply temporal decay
        time_delta = current_time - self.metadata_store[entity_id]["last_updated"]
        decayed_confidence = base_confidence * math.exp(-self.decay_rate * time_delta)
        
        return max(self.min_confidence, decayed_confidence)
    
    def update_confidence(
        self, 
        entity_id: str, 
        new_evidence: Union[float, Dict[str, Any]], 
        current_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Update confidence based on new evidence.
        
        Args:
            entity_id: Entity ID
            new_evidence: New evidence (features or score)
            current_time: Current time (defaults to now)
            
        Returns:
            Update information
        """
        if current_time is None:
            current_time = time.time()
            
        if entity_id not in self.confidence_store:
            return {"status": "entity_not_found", "entity_id": entity_id}
        
        # Get current confidence with decay
        current_confidence = self.get_confidence(entity_id, current_time)
        
        # Calculate evidence score
        if isinstance(new_evidence, dict):
            # If evidence is features, calculate a score
            evidence_score = self._calculate_evidence_score(new_evidence)
        else:
            # Otherwise use the provided score
            evidence_score = float(new_evidence)
        
        # Ensure evidence score is within bounds
        evidence_score = min(self.max_confidence, max(self.min_confidence, evidence_score))
        
        # Bayesian-inspired update
        # Weight the update based on current confidence and evidence quality
        update_weight = 0.3  # How much to weight the new evidence
        updated_confidence = (
            (1 - update_weight) * current_confidence + 
            update_weight * evidence_score
        )
        
        # Ensure confidence is within bounds
        updated_confidence = min(self.max_confidence, max(self.min_confidence, updated_confidence))
        
        # Store updated confidence
        self.confidence_store[entity_id] = updated_confidence
        
        # Update metadata
        self.metadata_store[entity_id]["last_updated"] = current_time
        self.metadata_store[entity_id]["update_count"] += 1
        self.metadata_store[entity_id]["last_evidence_score"] = evidence_score
        
        # Record in history
        self.update_history[entity_id].append((current_time, updated_confidence))
        
        # Trim history if it gets too long
        if len(self.update_history[entity_id]) > 100:
            self.update_history[entity_id] = self.update_history[entity_id][-100:]
        
        return {
            "status": "updated",
            "entity_id": entity_id,
            "prior_confidence": current_confidence,
            "new_confidence": updated_confidence,
            "evidence_score": evidence_score,
            "timestamp": current_time
        }
    
    def batch_decay(self, current_time: Optional[float] = None) -> int:
        """
        Apply temporal decay to all entities.
        
        Args:
            current_time: Current time (defaults to now)
            
        Returns:
            Number of entities updated
        """
        if current_time is None:
            current_time = time.time()
            
        updated_count = 0
        for entity_id in list(self.confidence_store.keys()):
            decayed_confidence = self.get_confidence(entity_id, current_time)
            self.confidence_store[entity_id] = decayed_confidence
            self.metadata_store[entity_id]["last_updated"] = current_time
            updated_count += 1
            
        return updated_count
    
    def get_confidence_history(self, entity_id: str) -> List[Tuple[float, float]]:
        """
        Get confidence history for an entity.
        
        Args:
            entity_id: Entity ID
            
        Returns:
            List of (timestamp, confidence) tuples
        """
        if entity_id not in self.update_history:
            return []
            
        return self.update_history[entity_id]
    
    def get_entities_by_confidence(
        self, 
        min_confidence: float = 0.0, 
        max_confidence: float = 1.0,
        limit: int = 100,
        current_time: Optional[float] = None
    ) -> List[Tuple[str, float]]:
        """
        Get entities filtered by confidence range.
        
        Args:
            min_confidence: Minimum confidence threshold
            max_confidence: Maximum confidence threshold
            limit: Maximum number of entities to return
            current_time: Current time for decay calculation
            
        Returns:
            List of (entity_id, confidence) tuples
        """
        if current_time is None:
            current_time = time.time()
            
        # Get all entities with their current confidence
        entities_with_confidence = [
            (entity_id, self.get_confidence(entity_id, current_time))
            for entity_id in self.confidence_store
        ]
        
        # Filter by confidence range
        filtered_entities = [
            (entity_id, confidence)
            for entity_id, confidence in entities_with_confidence
            if min_confidence <= confidence <= max_confidence
        ]
        
        # Sort by confidence (highest first)
        filtered_entities.sort(key=lambda x: x[1], reverse=True)
        
        return filtered_entities[:limit]
    
    def _calculate_evidence_score(self, evidence: Dict[str, Any]) -> float:
        """
        Calculate a score from evidence features.
        
        Args:
            evidence: Evidence features
            
        Returns:
            Evidence score (0.0 to 1.0)
        """
        # This is a placeholder - in a real implementation, this would
        # evaluate the quality of the evidence based on its features
        
        # Simple heuristic: average of feature values if they're numeric
        numeric_values = [v for v in evidence.values() if isinstance(v, (int, float))]
        if numeric_values:
            return min(1.0, max(0.0, sum(numeric_values) / len(numeric_values)))
        
        # If no numeric values, check for specific quality indicators
        if "quality" in evidence:
            return min(1.0, max(0.0, float(evidence["quality"])))
        
        if "confidence" in evidence:
            return min(1.0, max(0.0, float(evidence["confidence"])))
        
        if "source_reliability" in evidence:
            return min(1.0, max(0.0, float(evidence["source_reliability"])))
        
        # Default score
        return 0.5
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about confidence ecosystem performance.
        
        Returns:
            Dictionary of metrics
        """
        if not self.confidence_store:
            return {
                "entity_count": 0,
                "average_confidence": 0.0,
                "confidence_distribution": {}
            }
        
        current_time = time.time()
        
        # Get current confidence for all entities
        current_confidences = [
            self.get_confidence(entity_id, current_time)
            for entity_id in self.confidence_store
        ]
        
        # Calculate average confidence
        avg_confidence = sum(current_confidences) / len(current_confidences)
        
        # Calculate confidence distribution
        confidence_distribution = {}
        for conf in current_confidences:
            # Round to nearest 0.1
            bucket = round(conf * 10) / 10
            if bucket not in confidence_distribution:
                confidence_distribution[bucket] = 0
            confidence_distribution[bucket] += 1
        
        # Convert counts to percentages
        total_entities = len(current_confidences)
        confidence_distribution = {
            bucket: count / total_entities
            for bucket, count in confidence_distribution.items()
        }
        
        return {
            "entity_count": len(self.confidence_store),
            "average_confidence": avg_confidence,
            "confidence_distribution": confidence_distribution,
            "decay_rate": self.decay_rate,
            "min_confidence": self.min_confidence,
            "max_confidence": self.max_confidence
        }
