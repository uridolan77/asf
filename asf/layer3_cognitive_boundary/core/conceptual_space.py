import numpy as np
import torch
import torch.nn.functional as F
import uuid
import time
import logging
import asyncio
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict

class ConceptualSpace:
    """
    Represents a conceptual space with dimensional semantics.
    Provides operations for mapping, projecting, and transforming concepts.
    """
    def __init__(self, name, dimensions, dimension_weights=None, device=None):
        """
        Initialize a conceptual space with specific semantic dimensions.
        
        Args:
            name: Name of the conceptual space
            dimensions: List of dimension names
            dimension_weights: Optional weights for each dimension (or None for equal weights)
            device: Optional torch device
        """
        self.name = name
        self.dimensions = dimensions
        self.dim_count = len(dimensions)
        
        if dimension_weights is None:
            self.dimension_weights = torch.ones(self.dim_count)
        else:
            self.dimension_weights = torch.tensor(dimension_weights)
            
        self.dimension_weights = self.dimension_weights / torch.sum(self.dimension_weights)
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dimension_weights = self.dimension_weights.to(self.device)
        
        self.dim_to_idx = {dim: i for i, dim in enumerate(dimensions)}
        
        self.concepts = {}
    
    def add_concept(self, concept_id, coordinates, properties=None, metadata=None):
        """
        Add a concept to the space at the specified coordinates.
        
        Args:
            concept_id: Unique identifier for the concept
            coordinates: Tensor or array of coordinates in this space
            properties: Optional dictionary of concept properties
            metadata: Optional dictionary of metadata
            
        Returns:
            The concept_id
        """
        if isinstance(coordinates, np.ndarray):
            coordinates = torch.tensor(coordinates, dtype=torch.float32)
            
        if coordinates.dim() == 1:
            if coordinates.shape[0] != self.dim_count:
                raise ValueError(f"Coordinates must have {self.dim_count} dimensions")
        else:
            raise ValueError("Coordinates must be a 1D tensor")
            
        self.concepts[concept_id] = {
            'coordinates': coordinates.to(self.device),
            'properties': properties or {},
            'metadata': metadata or {}
        }
        
        return concept_id
        
    def get_concept(self, concept_id):
        """Get a concept by ID."""
        return self.concepts.get(concept_id)
        
    def distance(self, concept_id1, concept_id2, weighted=True):
        """
        Calculate distance between two concepts.
        
        Args:
            concept_id1: First concept ID
            concept_id2: Second concept ID
            weighted: Whether to use dimension weights
            
        Returns:
            Distance value
        Find k nearest concepts to the given coordinates.
        
        Args:
            coordinates: Query coordinates
            k: Number of neighbors to retrieve
            excluded_ids: Optional set of concept IDs to exclude
            
        Returns:
            List of (concept_id, distance) tuples
        Calculate centroid of multiple concepts.
        
        Args:
            concept_ids: List of concept IDs
            
        Returns:
            Centroid coordinates
        Interpolate between two concepts.
        
        Args:
            concept_id1: First concept ID
            concept_id2: Second concept ID
            alpha: Interpolation factor (0 = first concept, 1 = second concept)
            
        Returns:
            Interpolated coordinates
        Extrapolate beyond two concepts.
        
        Args:
            concept_id1: First concept ID (starting point)
            concept_id2: Second concept ID (direction)
            alpha: Extrapolation factor (positive = beyond second, negative = behind first)
            
        Returns:
            Extrapolated coordinates
        Project coordinates to a subspace defined by specific dimensions.
        
        Args:
            coordinates: Full coordinates
            dimensions: List of dimension names for the subspace
            
        Returns:
            Projected coordinates in the subspace
        Blend properties from multiple concepts.
        
        Args:
            concept_ids: List of concept IDs
            strategy: Blending strategy ('selective', 'union', 'intersection')
            
        Returns:
            Blended properties
    Implements conceptual blending using integration networks.
    Based on Fauconnier and Turner's theory with computational extensions.
        Create an input space for the integration network.
        
        Args:
            space_id: Unique identifier for the space
            dimensions: List of dimension names
            dimension_weights: Optional weights for each dimension
            
        Returns:
            The space_id
        Create a generic space (shared structure) for the integration network.
        
        Args:
            space_id: Unique identifier for the space
            dimensions: List of dimension names
            dimension_weights: Optional weights for each dimension
            
        Returns:
            The space_id
        Create a blend space for the integration network.
        
        Args:
            blend_id: Unique identifier for the blend space
            dimensions: List of dimension names
            dimension_weights: Optional weights for each dimension
            
        Returns:
            The blend_id
        Add a concept to a space.
        
        Args:
            space_id: ID of the space
            concept_id: Unique identifier for the concept
            coordinates: Tensor or array of coordinates in this space
            properties: Optional dictionary of concept properties
            metadata: Optional dictionary of metadata
            
        Returns:
            The concept_id or None if space not found
        Create a mapping between two input spaces.
        
        Args:
            space1_id: ID of the first space
            space2_id: ID of the second space
            concept_mappings: Dict mapping concept IDs from space1 to space2
            dimension_mappings: Optional dict mapping dimension names from space1 to space2
            
        Returns:
            True if successful, False otherwise
        Create a generic space by finding commonalities between input spaces.
        
        Args:
            generic_space_id: ID for the new generic space
            input_space_ids: List of input space IDs to derive generic space from
            
        Returns:
            ID of the created generic space
        Create a blend from input spaces and a generic space.
        
        Args:
            blend_id: ID for the new blend space
            input_space_ids: List of input space IDs
            generic_space_id: Optional generic space ID (will be created if not provided)
            blend_strategy: Strategy for blending ('selective', 'union', 'intersection')
            
        Returns:
            ID of the created blend space
        Project concepts from input and generic spaces to blend space.
        
        Args:
            blend_id: ID of the blend space
            input_space_ids: List of input space IDs
            generic_space_id: ID of the generic space
            blend_strategy: Strategy for blending
            
        Returns:
            True if successful, False otherwise
        Run composition phase on the blend.
        Combines elements from the inputs in the blended space.
        
        Args:
            blend_id: ID of the blend space
            
        Returns:
            True if successful, False otherwise
        Run completion phase on the blend.
        Fills in missing structure and infers additional properties.
        
        Args:
            blend_id: ID of the blend space
            
        Returns:
            True if successful, False otherwise
        Run elaboration phase on the blend.
        Runs the blend as an independent space to derive emergent structure.
        
        Args:
            blend_id: ID of the blend space
            
        Returns:
            True if successful, False otherwise
        Blend multiple property dictionaries.
        
        Args:
            property_dicts: List of property dictionaries
            strategy: Blending strategy ('selective', 'union', 'intersection')
            
        Returns:
            Blended properties
        Calculate alignment score between two spaces based on concept mappings.
        
        Args:
            space1: First conceptual space
            space2: Second conceptual space
            concept_mappings: Dict mapping concept IDs from space1 to space2
            
        Returns:
            Alignment score (0-1)
    Enhanced conceptual blending engine with counterfactual reasoning.
    Generates creative blends by exploring "what if" scenarios.
        Create a conceptual blend from input concepts.
        
        Args:
            input_concept_ids: List of concept IDs to blend
            blend_strategy: Strategy for blending ('selective', 'union', 'intersection')
            explore_counterfactuals: Whether to explore counterfactual scenarios
            
        Returns:
            Created blend details
        Evaluate a blend based on various criteria.
        
        Args:
            blend_id: ID of the blend to evaluate
            evaluation_criteria: Optional custom evaluation criteria
            
        Returns:
            Evaluation results
        Explore counterfactual variations of the blend.
        
        Args:
            blend_id: ID of the base blend
            input_concepts: Dict of input concepts
            input_spaces: Dict of input spaces
            count: Number of counterfactuals to generate
            
        Returns:
            List of counterfactual blends
        Create a counterfactual by varying properties of input concepts.
        
        Args:
            cf_id: ID for the counterfactual blend
            blend_id: ID of the base blend
            input_concepts: Dict of input concepts
            input_spaces: Dict of input spaces
            
        Returns:
            Counterfactual blend result
        Create a counterfactual by replacing one input concept with a similar one.
        
        Args:
            cf_id: ID for the counterfactual blend
            blend_id: ID of the base blend
            input_concepts: Dict of input concepts
            input_spaces: Dict of input spaces
            
        Returns:
            Counterfactual blend result
        Create a counterfactual by shifting dimensional focus.
        
        Args:
            cf_id: ID for the counterfactual blend
            blend_id: ID of the base blend
            input_concepts: Dict of input concepts
            input_spaces: Dict of input spaces
            
        Returns:
            Counterfactual blend result
        Create a counterfactual by adding emergent properties.
        
        Args:
            cf_id: ID for the counterfactual blend
            blend_id: ID of the base blend
            input_concepts: Dict of input concepts
            input_spaces: Dict of input spaces
            
        Returns:
            Counterfactual blend result
        Create cross-space mappings between input spaces.
        
        Args:
            input_spaces: Dict of input spaces
            
        Returns:
            True if successful, False otherwise
        Extract coordinates for dimensions from a concept.
        
        Args:
            concept: Concept node
            dimensions: List of dimension names
            
        Returns:
            Coordinates tensor
        Extract concept data from a blend space.
        
        Args:
            blend_space: ConceptualSpace for the blend
            
        Returns:
            List of blend concept data
        Extract emergent properties from a blend space.
        
        Args:
            blend_space: ConceptualSpace for the blend
            
        Returns:
            List of emergent properties
        Evaluate a blend and return overall score.
        
        Args:
            blend_id: ID of the blend to evaluate
            
        Returns:
            Overall evaluation score
        Evaluate how novel the blend is compared to inputs.
        
        Args:
            blend_id: ID of the blend to evaluate
            
        Returns:
            Novelty score (0-1)
        Evaluate how coherent the blend is.
        
        Args:
            blend_space: Blend space to evaluate
            
        Returns:
            Coherence score (0-1)
        Evaluate potential utility of the blend.
        
        Args:
            blend_space: Blend space to evaluate
            
        Returns:
            Utility score (0-1)
        Evaluate amount of emergent structure in the blend.
        
        Args:
            blend_space: Blend space to evaluate
            
        Returns:
            Emergent structure score (0-1)