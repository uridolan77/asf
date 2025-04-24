"""
Hybrid Memory Engine Module

This module implements the Hybrid Memory Engine component of the ASF framework,
which manages entity information, caching, and retrieval.
"""

import time
import uuid
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from collections import OrderedDict


class HybridMemoryEngine:
    """
    Manages entity information, caching, and retrieval.
    
    The Hybrid Memory Engine stores and retrieves entity information efficiently,
    using a combination of in-memory storage and caching mechanisms to optimize
    performance while maintaining flexibility.
    """
    
    def __init__(self, cache_size: int = 10000):
        """
        Initialize the Hybrid Memory Engine.
        
        Args:
            cache_size: Maximum number of entities to cache (default: 10000)
        """
        self.entity_store: Dict[str, Dict[str, Any]] = {}  # Main storage for entities
        self.feature_store: Dict[str, Dict[str, Any]] = {}  # Storage for entity features
        self.cache: OrderedDict = OrderedDict()  # LRU cache for frequently accessed entities
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
        self.access_history: Dict[str, List[float]] = {}  # Track entity access times
        self.entity_relationships: Dict[str, Dict[str, List[str]]] = {}  # Track entity relationships
    
    async def store_entity(
        self, 
        entity_id: Optional[str] = None, 
        entity_data: Any = None, 
        features: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store an entity in the memory engine.
        
        Args:
            entity_id: Entity ID (optional, will be generated if not provided)
            entity_data: Entity data
            features: Entity features (optional)
            
        Returns:
            Entity ID
        """
        # Generate ID if not provided
        if entity_id is None:
            entity_id = str(uuid.uuid4())
            
        # Store entity data
        self.entity_store[entity_id] = {
            "data": entity_data,
            "created_at": time.time(),
            "updated_at": time.time(),
            "access_count": 0
        }
        
        # Store features if provided
        if features is not None:
            self.feature_store[entity_id] = features
        
        # Add to cache
        self._update_cache(entity_id)
        
        return entity_id
    
    async def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an entity from the memory engine.
        
        Args:
            entity_id: Entity ID
            
        Returns:
            Entity data or None if not found
        """
        # Check cache first
        if entity_id in self.cache:
            self.cache_hits += 1
            self._update_cache(entity_id)  # Update access history
            
            # Update access count
            if entity_id in self.entity_store:
                self.entity_store[entity_id]["access_count"] += 1
                
            return self.entity_store[entity_id]
        
        # Check main store
        if entity_id in self.entity_store:
            self.cache_misses += 1
            self._update_cache(entity_id)  # Add to cache
            
            # Update access count
            self.entity_store[entity_id]["access_count"] += 1
            
            return self.entity_store[entity_id]
        
        return None
    
    async def update_entity(
        self, 
        entity_id: str, 
        update_data: Any, 
        update_features: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update an existing entity.
        
        Args:
            entity_id: Entity ID
            update_data: New data for the entity
            update_features: New features for the entity (optional)
            
        Returns:
            Success flag
        """
        if entity_id not in self.entity_store:
            return False
        
        # Update entity data
        if isinstance(update_data, dict) and isinstance(self.entity_store[entity_id]["data"], dict):
            # Merge dictionaries
            self.entity_store[entity_id]["data"].update(update_data)
        else:
            # Replace data
            self.entity_store[entity_id]["data"] = update_data
            
        # Update timestamp
        self.entity_store[entity_id]["updated_at"] = time.time()
        
        # Update features if provided
        if update_features is not None:
            if entity_id in self.feature_store and isinstance(self.feature_store[entity_id], dict) and isinstance(update_features, dict):
                # Merge features
                self.feature_store[entity_id].update(update_features)
            else:
                self.feature_store[entity_id] = update_features
        
        # Update cache
        self._update_cache(entity_id)
        
        return True
    
    async def delete_entity(self, entity_id: str) -> bool:
        """
        Delete an entity from the memory engine.
        
        Args:
            entity_id: Entity ID
            
        Returns:
            Success flag
        """
        if entity_id not in self.entity_store:
            return False
            
        # Remove from entity store
        del self.entity_store[entity_id]
        
        # Remove from feature store
        if entity_id in self.feature_store:
            del self.feature_store[entity_id]
            
        # Remove from cache
        if entity_id in self.cache:
            del self.cache[entity_id]
            
        # Remove from access history
        if entity_id in self.access_history:
            del self.access_history[entity_id]
            
        # Remove from relationships
        if entity_id in self.entity_relationships:
            del self.entity_relationships[entity_id]
            
        # Remove references to this entity in other entities' relationships
        for other_id, relationships in self.entity_relationships.items():
            for rel_type, related_ids in list(relationships.items()):
                if entity_id in related_ids:
                    relationships[rel_type] = [rid for rid in related_ids if rid != entity_id]
        
        return True
    
    async def add_relationship(
        self, 
        source_id: str, 
        target_id: str, 
        relationship_type: str
    ) -> bool:
        """
        Add a relationship between entities.
        
        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            relationship_type: Type of relationship
            
        Returns:
            Success flag
        """
        if source_id not in self.entity_store or target_id not in self.entity_store:
            return False
            
        # Initialize relationships dict if needed
        if source_id not in self.entity_relationships:
            self.entity_relationships[source_id] = {}
            
        if relationship_type not in self.entity_relationships[source_id]:
            self.entity_relationships[source_id][relationship_type] = []
            
        # Add relationship if not already present
        if target_id not in self.entity_relationships[source_id][relationship_type]:
            self.entity_relationships[source_id][relationship_type].append(target_id)
            
        return True
    
    async def get_related_entities(
        self, 
        entity_id: str, 
        relationship_type: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """
        Get entities related to the specified entity.
        
        Args:
            entity_id: Entity ID
            relationship_type: Type of relationship (optional)
            
        Returns:
            Dictionary of relationship types to lists of related entity IDs
        """
        if entity_id not in self.entity_relationships:
            return {}
            
        if relationship_type:
            # Return specific relationship type
            if relationship_type in self.entity_relationships[entity_id]:
                return {relationship_type: self.entity_relationships[entity_id][relationship_type]}
            return {}
        else:
            # Return all relationships
            return self.entity_relationships[entity_id]
    
    async def query(
        self, 
        query_features: Dict[str, Any], 
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Query entities based on feature similarity.
        
        Args:
            query_features: Query features
            top_k: Maximum number of results
            
        Returns:
            List of (entity_id, score) tuples
        """
        if not self.feature_store:
            return []
        
        # Calculate similarity scores
        scores = []
        for entity_id, features in self.feature_store.items():
            similarity = self._calculate_similarity(query_features, features)
            scores.append((entity_id, similarity))
        
        # Sort by similarity (highest first)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:top_k]
    
    async def query_by_field(
        self, 
        field: str, 
        value: Any, 
        exact_match: bool = True
    ) -> List[str]:
        """
        Query entities by field value.
        
        Args:
            field: Field to query
            value: Value to match
            exact_match: Whether to require exact match
            
        Returns:
            List of matching entity IDs
        """
        matching_ids = []
        
        for entity_id, entity_info in self.entity_store.items():
            entity_data = entity_info["data"]
            
            if not isinstance(entity_data, dict):
                continue
                
            if field not in entity_data:
                continue
                
            field_value = entity_data[field]
            
            if exact_match:
                if field_value == value:
                    matching_ids.append(entity_id)
            else:
                # For non-exact match, handle different types
                if isinstance(field_value, str) and isinstance(value, str):
                    if value.lower() in field_value.lower():
                        matching_ids.append(entity_id)
                elif isinstance(field_value, (int, float)) and isinstance(value, (int, float)):
                    if abs(field_value - value) < 0.0001:  # Small epsilon for float comparison
                        matching_ids.append(entity_id)
                elif field_value == value:
                    matching_ids.append(entity_id)
        
        return matching_ids
    
    def _update_cache(self, entity_id: str) -> None:
        """
        Update the cache with an entity.
        
        Args:
            entity_id: Entity ID
        """
        # Add to cache
        self.cache[entity_id] = time.time()
        
        # Update access history
        if entity_id not in self.access_history:
            self.access_history[entity_id] = []
        self.access_history[entity_id].append(time.time())
        
        # Trim access history
        if len(self.access_history[entity_id]) > 100:
            self.access_history[entity_id] = self.access_history[entity_id][-100:]
        
        # If cache is full, remove least recently used entity
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
    
    def _calculate_similarity(
        self, 
        features_a: Dict[str, Any], 
        features_b: Dict[str, Any]
    ) -> float:
        """
        Calculate similarity between two feature sets.
        
        Args:
            features_a: First feature set
            features_b: Second feature set
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Get common keys
        common_keys = set(features_a.keys()).intersection(set(features_b.keys()))
        
        if not common_keys:
            return 0.0
        
        # Calculate similarity for each common key
        similarities = []
        for key in common_keys:
            value_a = features_a[key]
            value_b = features_b[key]
            
            # Handle different value types
            if isinstance(value_a, (int, float)) and isinstance(value_b, (int, float)):
                # Numeric similarity
                max_val = max(abs(value_a), abs(value_b))
                if max_val == 0:
                    similarities.append(1.0)  # Both are zero
                else:
                    similarities.append(1.0 - min(1.0, abs(value_a - value_b) / max_val))
            elif isinstance(value_a, str) and isinstance(value_b, str):
                # String similarity (simple word overlap)
                words_a = set(value_a.lower().split())
                words_b = set(value_b.lower().split())
                overlap = len(words_a.intersection(words_b))
                union = len(words_a.union(words_b))
                similarities.append(overlap / union if union > 0 else 0.0)
            elif isinstance(value_a, list) and isinstance(value_b, list):
                # List similarity (Jaccard similarity)
                set_a = set(str(item) for item in value_a)
                set_b = set(str(item) for item in value_b)
                overlap = len(set_a.intersection(set_b))
                union = len(set_a.union(set_b))
                similarities.append(overlap / union if union > 0 else 0.0)
            else:
                # Default similarity (equality check)
                similarities.append(1.0 if value_a == value_b else 0.0)
        
        # Average similarity across all common keys
        return sum(similarities) / len(similarities)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about memory engine performance.
        
        Returns:
            Dictionary of metrics
        """
        total_accesses = self.cache_hits + self.cache_misses
        
        # Calculate average access count
        if self.entity_store:
            avg_access_count = sum(
                entity_info["access_count"] for entity_info in self.entity_store.values()
            ) / len(self.entity_store)
        else:
            avg_access_count = 0
            
        # Calculate relationship metrics
        total_relationships = sum(
            sum(len(related_ids) for related_ids in relationships.values())
            for relationships in self.entity_relationships.values()
        )
        
        relationship_types = set()
        for relationships in self.entity_relationships.values():
            relationship_types.update(relationships.keys())
        
        return {
            "entity_count": len(self.entity_store),
            "feature_count": len(self.feature_store),
            "cache_size": len(self.cache),
            "cache_hit_rate": self.cache_hits / total_accesses if total_accesses > 0 else 0,
            "total_accesses": total_accesses,
            "avg_access_count": avg_access_count,
            "total_relationships": total_relationships,
            "relationship_types": list(relationship_types),
            "relationship_type_count": len(relationship_types)
        }
