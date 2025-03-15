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
