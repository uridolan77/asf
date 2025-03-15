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
