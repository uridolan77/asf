import uuid
import time
import numpy as np
import logging
from typing import Dict, Any, Optional, List
from collections import defaultdict

from asf.layer3_cognitive_boundary.enums import SemanticNodeType, SemanticConfidenceState
from asf.layer3_cognitive_boundary.core.semantic_node import SemanticNode
from asf.layer3_cognitive_boundary.core.semantic_relation import SemanticRelation

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
        self.hierarchical_enabled = True
        self.adaptive_thresholds = True
        self.formation_confidence = 0.7
        
        self.partial_concept_cache = {}  # Partial features -> anticipated concept
        self.concept_formation_errors = defaultdict(list)  # Feature set hash -> prediction errors
        self.precision_values = {}  # Feature set hash -> precision
        
    async def anticipate_concept(self, partial_features, context=None):
        context = context or {}
        features_hash = self._hash_features(partial_features)
        
        if features_hash in self.partial_concept_cache:
            return self.partial_concept_cache[features_hash]
        
        partial_embeddings = self._generate_embeddings(partial_features)
        
        similar_concepts = []
        
        temp_id = f"temp_{uuid.uuid4().hex[:8]}"
        temp_node = SemanticNode(
            id=temp_id,
            label="Temporary anticipation node",
            node_type=SemanticNodeType.CONCEPT.value,
            properties=partial_features,
            embeddings=partial_embeddings
        )
        
        await self.semantic_network.add_node(temp_node, update_tensors=False)
        
        similarity_results = await self.semantic_network.get_similar_nodes(
            temp_id, k=5, threshold=0.6
        )
        
        
        anticipated_properties = dict(partial_features)  # Start with known properties
        
        if similarity_results:
            property_candidates = defaultdict(list)
            
            for concept_id, similarity in similarity_results:
                concept = await self.semantic_network.get_node(concept_id)
                if not concept:
                    continue
                
                for prop_name, value in concept.properties.items():
                    if prop_name not in partial_features:
                        property_candidates[prop_name].append((value, similarity))
            
            for prop_name, candidates in property_candidates.items():
                if not candidates:
                    continue
                
                if all(isinstance(v[0], (int, float)) for v in candidates):
                    total_weight = sum(weight for _, weight in candidates)
                    if total_weight > 0:
                        weighted_sum = sum(value * weight for value, weight in candidates)
                        anticipated_properties[prop_name] = weighted_sum / total_weight
                else:
                    best_candidate = max(candidates, key=lambda x: x[1])
                    anticipated_properties[prop_name] = best_candidate[0]
        
        anticipated_embeddings = self._generate_embeddings(anticipated_properties)
        
        confidence_factors = [
            len(partial_features) / max(1, len(anticipated_properties)),  # Ratio of known to total
            0.7,  # Base confidence
        ]
        if similarity_results:
            confidence_factors.append(similarity_results[0][1])  # Similarity to best match
        
        anticipated_confidence = min(0.9, sum(confidence_factors) / len(confidence_factors))
        
        anticipated_concept = {
            'properties': anticipated_properties,
            'embeddings': anticipated_embeddings,
            'confidence': anticipated_confidence,
            'similarity_results': similarity_results,
            'partial_features': partial_features,
        }
        
        self.partial_concept_cache[features_hash] = anticipated_concept
        
        return anticipated_concept
    
    async def form_concept(self, features, source_id=None, context=None):
        context = context or {}
        features_hash = self._hash_features(features)
        
        anticipated = None
        if 'anticipated_id' in context:
            anticipated_id = context['anticipated_id']
            if anticipated_id in self.partial_concept_cache:
                anticipated = self.partial_concept_cache[anticipated_id]
        
        if anticipated:
            anticipated_properties = anticipated['properties']
            
            prediction_errors = []
            
            for prop_name, actual_value in features.items():
                if prop_name in anticipated_properties:
                    anticipated_value = anticipated_properties[prop_name]
                    
                    if isinstance(actual_value, (int, float)) and isinstance(anticipated_value, (int, float)):
                        error = abs(actual_value - anticipated_value) / (1.0 + abs(actual_value))
                        prediction_errors.append(error)
                    elif isinstance(actual_value, str) and isinstance(anticipated_value, str):
                        error = 0.0 if actual_value == anticipated_value else 1.0
                        prediction_errors.append(error)
            
            if prediction_errors:
                avg_error = sum(prediction_errors) / len(prediction_errors)
                
                self.concept_formation_errors[features_hash].append(avg_error)
                
                if len(self.concept_formation_errors[features_hash]) > 20:
                    self.concept_formation_errors[features_hash] = self.concept_formation_errors[features_hash][-20:]
                
                if len(self.concept_formation_errors[features_hash]) > 1:
                    variance = np.var(self.concept_formation_errors[features_hash])
                    precision = 1.0 / (variance + 1e-6)  # Avoid division by zero
                    self.precision_values[features_hash] = min(10.0, precision)  # Cap very high precision
        
        
        if anticipated and 'confidence' in anticipated:
            anticipated_confidence = anticipated['confidence']
            context['anticipated_confidence'] = anticipated_confidence
        
        embeddings = self._generate_embeddings(features)
        
        properties = {}
        for name, value in features.items():
            if isinstance(value, (int, float, str, bool)):
                properties[name] = value
                
        source_confidence = context.get('source_confidence', 0.7)
        anticipated_confidence = context.get('anticipated_confidence', 0.0)
        
        precision = self.precision_values.get(features_hash, 1.0)
        confidence_weight = min(0.8, precision / (precision + 1.0))
        
        confidence = (
            self.formation_confidence * (1.0 - confidence_weight) +
            source_confidence * (1.0 - confidence_weight) +
            anticipated_confidence * confidence_weight * 2
        ) / (2.0 + confidence_weight)
        
        confidence = max(self.formation_confidence, min(0.95, confidence))
        
        
        metadata = {
            'created_from': context.get('created_from', 'direct'),
            'formation_time': time.time()
        }
        
        if anticipated:
            metadata['anticipated'] = True
            metadata['anticipation_error'] = sum(prediction_errors) / len(prediction_errors) if prediction_errors else 0
            metadata['anticipation_precision'] = self.precision_values.get(features_hash, 1.0)
        
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
        
        await self.semantic_network.add_node(concept_node)
        
        self.formation_history.append({
            'action': 'created',
            'concept_id': concept_id,
            'source_id': source_id,
            'anticipated': anticipated is not None,
            'timestamp': time.time()
        })
        
        return concept_id
    
    async def generate_counterfactual_concepts(self, features, modifications, context=None):
        context = context or {}
        counterfactual_concepts = []
        
        for modification in modifications:
            modified_features = self._apply_feature_modification(features, modification)
            
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
        sorted_items = sorted((str(k), str(v)) for k, v in features.items())
        return hash(tuple(sorted_items))
