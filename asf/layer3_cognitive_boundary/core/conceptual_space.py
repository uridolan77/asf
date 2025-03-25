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
        
        # Initialize dimension weights
        if dimension_weights is None:
            self.dimension_weights = torch.ones(self.dim_count)
        else:
            self.dimension_weights = torch.tensor(dimension_weights)
            
        # Normalize weights
        self.dimension_weights = self.dimension_weights / torch.sum(self.dimension_weights)
        
        # Set device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dimension_weights = self.dimension_weights.to(self.device)
        
        # Create dimension mapping
        self.dim_to_idx = {dim: i for i, dim in enumerate(dimensions)}
        
        # Initialize concept storage
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
        # Convert coordinates to tensor
        if isinstance(coordinates, np.ndarray):
            coordinates = torch.tensor(coordinates, dtype=torch.float32)
            
        # Ensure correct shape
        if coordinates.dim() == 1:
            if coordinates.shape[0] != self.dim_count:
                raise ValueError(f"Coordinates must have {self.dim_count} dimensions")
        else:
            raise ValueError("Coordinates must be a 1D tensor")
            
        # Add concept
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
        """
        concept1 = self.get_concept(concept_id1)
        concept2 = self.get_concept(concept_id2)
        
        if not concept1 or not concept2:
            return float('inf')
            
        coords1 = concept1['coordinates']
        coords2 = concept2['coordinates']
        
        if weighted:
            # Weighted Euclidean distance
            weighted_diff = (coords1 - coords2) * torch.sqrt(self.dimension_weights)
            return torch.norm(weighted_diff).item()
        else:
            # Regular Euclidean distance
            return torch.norm(coords1 - coords2).item()
            
    def nearest_concepts(self, coordinates, k=5, excluded_ids=None):
        """
        Find k nearest concepts to the given coordinates.
        
        Args:
            coordinates: Query coordinates
            k: Number of neighbors to retrieve
            excluded_ids: Optional set of concept IDs to exclude
            
        Returns:
            List of (concept_id, distance) tuples
        """
        if not self.concepts:
            return []
            
        excluded_ids = excluded_ids or set()
        
        # Convert coordinates to tensor
        if isinstance(coordinates, np.ndarray):
            coordinates = torch.tensor(coordinates, dtype=torch.float32)
            
        coordinates = coordinates.to(self.device)
        
        # Calculate distances
        distances = []
        for concept_id, concept in self.concepts.items():
            if concept_id in excluded_ids:
                continue
                
            # Calculate weighted distance
            weighted_diff = (concept['coordinates'] - coordinates) * torch.sqrt(self.dimension_weights)
            distance = torch.norm(weighted_diff).item()
            
            distances.append((concept_id, distance))
            
        # Sort by distance
        distances.sort(key=lambda x: x[1])
        
        return distances[:k]
        
    def centroid(self, concept_ids):
        """
        Calculate centroid of multiple concepts.
        
        Args:
            concept_ids: List of concept IDs
            
        Returns:
            Centroid coordinates
        """
        if not concept_ids:
            return None
            
        # Get coordinates
        coords_list = []
        for concept_id in concept_ids:
            concept = self.get_concept(concept_id)
            if concept:
                coords_list.append(concept['coordinates'])
                
        if not coords_list:
            return None
            
        # Calculate centroid
        return torch.mean(torch.stack(coords_list), dim=0)
        
    def interpolate(self, concept_id1, concept_id2, alpha=0.5):
        """
        Interpolate between two concepts.
        
        Args:
            concept_id1: First concept ID
            concept_id2: Second concept ID
            alpha: Interpolation factor (0 = first concept, 1 = second concept)
            
        Returns:
            Interpolated coordinates
        """
        concept1 = self.get_concept(concept_id1)
        concept2 = self.get_concept(concept_id2)
        
        if not concept1 or not concept2:
            return None
            
        coords1 = concept1['coordinates']
        coords2 = concept2['coordinates']
        
        return (1 - alpha) * coords1 + alpha * coords2
        
    def extrapolate(self, concept_id1, concept_id2, alpha=0.5):
        """
        Extrapolate beyond two concepts.
        
        Args:
            concept_id1: First concept ID (starting point)
            concept_id2: Second concept ID (direction)
            alpha: Extrapolation factor (positive = beyond second, negative = behind first)
            
        Returns:
            Extrapolated coordinates
        """
        concept1 = self.get_concept(concept_id1)
        concept2 = self.get_concept(concept_id2)
        
        if not concept1 or not concept2:
            return None
            
        coords1 = concept1['coordinates']
        coords2 = concept2['coordinates']
        
        # Direction vector from concept1 to concept2
        direction = coords2 - coords1
        
        # Extrapolate
        return coords1 + (1 + alpha) * direction
        
    def project_to_subspace(self, coordinates, dimensions):
        """
        Project coordinates to a subspace defined by specific dimensions.
        
        Args:
            coordinates: Full coordinates
            dimensions: List of dimension names for the subspace
            
        Returns:
            Projected coordinates in the subspace
        """
        if isinstance(coordinates, np.ndarray):
            coordinates = torch.tensor(coordinates, dtype=torch.float32).to(self.device)
            
        # Get indices of dimensions
        indices = [self.dim_to_idx[dim] for dim in dimensions if dim in self.dim_to_idx]
        
        if not indices:
            return None
            
        # Project to subspace
        return coordinates[indices]
        
    def blend_properties(self, concept_ids, strategy='selective'):
        """
        Blend properties from multiple concepts.
        
        Args:
            concept_ids: List of concept IDs
            strategy: Blending strategy ('selective', 'union', 'intersection')
            
        Returns:
            Blended properties
        """
        if not concept_ids:
            return {}
            
        # Collect properties
        all_properties = {}
        for concept_id in concept_ids:
            concept = self.get_concept(concept_id)
            if concept and 'properties' in concept:
                for prop, value in concept['properties'].items():
                    if prop not in all_properties:
                        all_properties[prop] = []
                    all_properties[prop].append(value)
                    
        # Blend properties according to strategy
        blended_properties = {}
        
        if strategy == 'union':
            # Take all properties
            for prop, values in all_properties.items():
                if all(isinstance(v, (int, float)) for v in values):
                    # For numeric properties, take mean
                    blended_properties[prop] = sum(values) / len(values)
                else:
                    # For non-numeric, take most common
                    value_counts = defaultdict(int)
                    for value in values:
                        value_counts[str(value)] += 1
                    most_common = max(value_counts.items(), key=lambda x: x[1])[0]
                    blended_properties[prop] = most_common
                    
        elif strategy == 'intersection':
            # Take only properties that all concepts have
            for prop, values in all_properties.items():
                if len(values) == len(concept_ids):
                    if all(isinstance(v, (int, float)) for v in values):
                        # For numeric properties, take mean
                        blended_properties[prop] = sum(values) / len(values)
                    else:
                        # For non-numeric, take most common
                        value_counts = defaultdict(int)
                        for value in values:
                            value_counts[str(value)] += 1
                        most_common = max(value_counts.items(), key=lambda x: x[1])[0]
                        blended_properties[prop] = most_common
                        
        else:  # selective
            # Take properties selectively based on relevance
            for prop, values in all_properties.items():
                # Determine relevance
                relevance = len(values) / len(concept_ids)
                
                if relevance >= 0.5:  # At least half the concepts have this property
                    if all(isinstance(v, (int, float)) for v in values):
                        # For numeric properties, take mean
                        blended_properties[prop] = sum(values) / len(values)
                    else:
                        # For non-numeric, take most common
                        value_counts = defaultdict(int)
                        for value in values:
                            value_counts[str(value)] += 1
                        most_common = max(value_counts.items(), key=lambda x: x[1])[0]
                        blended_properties[prop] = most_common
                        
        return blended_properties


class BlendingIntegrationNetwork:
    """
    Implements conceptual blending using integration networks.
    Based on Fauconnier and Turner's theory with computational extensions.
    """
    def __init__(self, device=None):
        self.logger = logging.getLogger("ASF.Layer3.BlendingNetwork")
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize spaces
        self.input_spaces = {}  # Maps space_id to ConceptualSpace
        self.generic_spaces = {}  # Maps space_id to ConceptualSpace
        self.blend_spaces = {}  # Maps blend_id to ConceptualSpace
        
        # Integration network components
        self.cross_space_mappings = {}  # Maps (space1_id, space2_id) to mapping dict
        self.projections = {}  # Maps (space_id, blend_id) to projection dict
        
    def create_input_space(self, space_id, dimensions, dimension_weights=None):
        """
        Create an input space for the integration network.
        
        Args:
            space_id: Unique identifier for the space
            dimensions: List of dimension names
            dimension_weights: Optional weights for each dimension
            
        Returns:
            The space_id
        """
        self.input_spaces[space_id] = ConceptualSpace(
            name=space_id,
            dimensions=dimensions,
            dimension_weights=dimension_weights,
            device=self.device
        )
        
        return space_id
        
    def create_generic_space(self, space_id, dimensions, dimension_weights=None):
        """
        Create a generic space (shared structure) for the integration network.
        
        Args:
            space_id: Unique identifier for the space
            dimensions: List of dimension names
            dimension_weights: Optional weights for each dimension
            
        Returns:
            The space_id
        """
        self.generic_spaces[space_id] = ConceptualSpace(
            name=space_id,
            dimensions=dimensions,
            dimension_weights=dimension_weights,
            device=self.device
        )
        
        return space_id
        
    def create_blend_space(self, blend_id, dimensions, dimension_weights=None):
        """
        Create a blend space for the integration network.
        
        Args:
            blend_id: Unique identifier for the blend space
            dimensions: List of dimension names
            dimension_weights: Optional weights for each dimension
            
        Returns:
            The blend_id
        """
        self.blend_spaces[blend_id] = ConceptualSpace(
            name=blend_id,
            dimensions=dimensions,
            dimension_weights=dimension_weights,
            device=self.device
        )
        
        return blend_id
        
    def add_concept_to_space(self, space_id, concept_id, coordinates, properties=None, metadata=None):
        """
        Add a concept to a space.
        
        Args:
            space_id: ID of the space
            concept_id: Unique identifier for the concept
            coordinates: Tensor or array of coordinates in this space
            properties: Optional dictionary of concept properties
            metadata: Optional dictionary of metadata
            
        Returns:
            The concept_id or None if space not found
        """
        # Find the space
        space = None
        if space_id in self.input_spaces:
            space = self.input_spaces[space_id]
        elif space_id in self.generic_spaces:
            space = self.generic_spaces[space_id]
        elif space_id in self.blend_spaces:
            space = self.blend_spaces[space_id]
            
        if space is None:
            self.logger.warning(f"Space {space_id} not found")
            return None
            
        # Add concept to space
        return space.add_concept(
            concept_id=concept_id,
            coordinates=coordinates,
            properties=properties,
            metadata=metadata
        )
        
    def create_cross_space_mapping(self, space1_id, space2_id, concept_mappings, dimension_mappings=None):
        """
        Create a mapping between two input spaces.
        
        Args:
            space1_id: ID of the first space
            space2_id: ID of the second space
            concept_mappings: Dict mapping concept IDs from space1 to space2
            dimension_mappings: Optional dict mapping dimension names from space1 to space2
            
        Returns:
            True if successful, False otherwise
        """
        # Check that spaces exist
        if space1_id not in self.input_spaces or space2_id not in self.input_spaces:
            self.logger.warning(f"One or both spaces not found: {space1_id}, {space2_id}")
            return False
            
        space1 = self.input_spaces[space1_id]
        space2 = self.input_spaces[space2_id]
        
        # Create mapping entry
        self.cross_space_mappings[(space1_id, space2_id)] = {
            'concept_mappings': concept_mappings,
            'dimension_mappings': dimension_mappings or {},
            'alignment_score': self._calculate_alignment_score(space1, space2, concept_mappings)
        }
        
        return True
        
    def create_generic_space_from_inputs(self, generic_space_id, input_space_ids):
        """
        Create a generic space by finding commonalities between input spaces.
        
        Args:
            generic_space_id: ID for the new generic space
            input_space_ids: List of input space IDs to derive generic space from
            
        Returns:
            ID of the created generic space
        """
        if not input_space_ids:
            return None
            
        # Get input spaces
        input_spaces = [self.input_spaces[space_id] for space_id in input_space_ids 
                      if space_id in self.input_spaces]
        
        if not input_spaces:
            return None
            
        # Find common dimensions across input spaces
        all_dimensions = set(input_spaces[0].dimensions)
        for space in input_spaces[1:]:
            all_dimensions.intersection_update(space.dimensions)
            
        common_dimensions = list(all_dimensions)
        
        if not common_dimensions:
            # No common dimensions found
            self.logger.warning("No common dimensions found across input spaces")
            # Use minimal set of dimensions
            common_dimensions = input_spaces[0].dimensions[:3]
            
        # Create generic space
        self.create_generic_space(generic_space_id, common_dimensions)
        
        # Find common concepts
        # For each input space, collect concepts that have counterparts in all other spaces
        common_concepts = []
        
        # Look through cross-space mappings to find common concepts
        for i, space1_id in enumerate(input_space_ids):
            for j, space2_id in enumerate(input_space_ids[i+1:], i+1):
                # Check both directions
                mapping1 = self.cross_space_mappings.get((space1_id, space2_id))
                mapping2 = self.cross_space_mappings.get((space2_id, space1_id))
                
                if mapping1:
                    # Collect mappings
                    for concept1_id, concept2_id in mapping1['concept_mappings'].items():
                        common_concepts.append((space1_id, concept1_id, space2_id, concept2_id))
                        
                if mapping2:
                    # Collect mappings (reversed)
                    for concept2_id, concept1_id in mapping2['concept_mappings'].items():
                        common_concepts.append((space2_id, concept2_id, space1_id, concept1_id))
        
        # Create generic concepts from common concepts
        generic_space = self.generic_spaces[generic_space_id]
        
        for i, (space1_id, concept1_id, space2_id, concept2_id) in enumerate(common_concepts):
            # Get concepts
            space1 = self.input_spaces[space1_id]
            space2 = self.input_spaces[space2_id]
            
            concept1 = space1.get_concept(concept1_id)
            concept2 = space2.get_concept(concept2_id)
            
            if not concept1 or not concept2:
                continue
                
            # Project both concepts to generic space dimensions
            coords1 = space1.project_to_subspace(concept1['coordinates'], common_dimensions)
            coords2 = space2.project_to_subspace(concept2['coordinates'], common_dimensions)
            
            if coords1 is None or coords2 is None:
                continue
                
            # Average coordinates for generic concept
            avg_coords = (coords1 + coords2) / 2
            
            # Find common properties
            common_props = {}
            for prop, value1 in concept1['properties'].items():
                if prop in concept2['properties']:
                    value2 = concept2['properties'][prop]
                    if value1 == value2:
                        common_props[prop] = value1
                    elif isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
                        # For numeric properties, take average
                        common_props[prop] = (value1 + value2) / 2
            
            # Add generic concept
            generic_concept_id = f"generic_{i}"
            generic_space.add_concept(
                concept_id=generic_concept_id,
                coordinates=avg_coords,
                properties=common_props
            )
            
        return generic_space_id
        
    async def create_blend_from_inputs(self, blend_id, input_space_ids, generic_space_id=None,
                              blend_strategy='selective'):
        """
        Create a blend from input spaces and a generic space.
        
        Args:
            blend_id: ID for the new blend space
            input_space_ids: List of input space IDs
            generic_space_id: Optional generic space ID (will be created if not provided)
            blend_strategy: Strategy for blending ('selective', 'union', 'intersection')
            
        Returns:
            ID of the created blend space
        """
        if not input_space_ids:
            return None
            
        # Get input spaces
        input_spaces = [self.input_spaces[space_id] for space_id in input_space_ids 
                      if space_id in self.input_spaces]
        
        if not input_spaces:
            return None
            
        # Create or get generic space
        if generic_space_id is None or generic_space_id not in self.generic_spaces:
            generic_space_id = f"generic_for_{blend_id}"
            self.create_generic_space_from_inputs(generic_space_id, input_space_ids)
            
        generic_space = self.generic_spaces.get(generic_space_id)
        if not generic_space:
            self.logger.warning(f"Generic space {generic_space_id} not found or created")
            return None
            
        # Determine blend dimensions (union of input space dimensions)
        blend_dimensions = set()
        for space in input_spaces:
            blend_dimensions.update(space.dimensions)
            
        # Create blend space
        self.create_blend_space(blend_id, list(blend_dimensions))
        blend_space = self.blend_spaces[blend_id]
        
        # Create projection mappings
        for i, space_id in enumerate(input_space_ids):
            if space_id in self.input_spaces:
                self.projections[(space_id, blend_id)] = {
                    'projection_type': 'input_to_blend',
                    'weight': 1.0 / len(input_space_ids)  # Equal weighting initially
                }
                
        self.projections[(generic_space_id, blend_id)] = {
            'projection_type': 'generic_to_blend',
            'weight': 0.5  # Generic space has significant influence
        }
        
        # Project concepts to blend space
        await self._project_to_blend(blend_id, input_space_ids, generic_space_id, blend_strategy)
        
        # Run composition, completion, and elaboration
        await self._compose_blend(blend_id)
        await self._complete_blend(blend_id)
        await self._elaborate_blend(blend_id)
        
        return blend_id
        
    async def _project_to_blend(self, blend_id, input_space_ids, generic_space_id, blend_strategy):
        """
        Project concepts from input and generic spaces to blend space.
        
        Args:
            blend_id: ID of the blend space
            input_space_ids: List of input space IDs
            generic_space_id: ID of the generic space
            blend_strategy: Strategy for blending
            
        Returns:
            True if successful, False otherwise
        """
        blend_space = self.blend_spaces.get(blend_id)
        if not blend_space:
            return False
            
        # Collect input and generic spaces
        spaces = {}
        for space_id in input_space_ids:
            if space_id in self.input_spaces:
                spaces[space_id] = self.input_spaces[space_id]
                
        generic_space = self.generic_spaces.get(generic_space_id)
        if generic_space:
            spaces[generic_space_id] = generic_space
            
        if not spaces:
            return False
            
        # Extract dimension mappings
        dimension_mappings = {}
        for (space1_id, space2_id), mapping in self.cross_space_mappings.items():
            if 'dimension_mappings' in mapping:
                dim_map = mapping['dimension_mappings']
                for dim1, dim2 in dim_map.items():
                    if dim1 not in dimension_mappings:
                        dimension_mappings[dim1] = {}
                    dimension_mappings[dim1][space2_id] = dim2
        
        # Extract concept mappings
        concept_mappings = defaultdict(dict)
        for (space1_id, space2_id), mapping in self.cross_space_mappings.items():
            for concept1_id, concept2_id in mapping['concept_mappings'].items():
                concept_mappings[concept1_id][space2_id] = concept2_id
                
        # Find concept clusters (concepts that are mapped to each other)
        concept_clusters = []
        visited_concepts = set()
        
        for space_id, space in spaces.items():
            for concept_id in space.concepts:
                if (space_id, concept_id) in visited_concepts:
                    continue
                    
                # Start a new cluster
                cluster = [(space_id, concept_id)]
                visited_concepts.add((space_id, concept_id))
                
                # Find all connected concepts through mappings
                i = 0
                while i < len(cluster):
                    current_space_id, current_concept_id = cluster[i]
                    
                    # Check if this concept is mapped to any others
                    if current_concept_id in concept_mappings:
                        for target_space_id, target_concept_id in concept_mappings[current_concept_id].items():
                            if (target_space_id, target_concept_id) not in visited_concepts:
                                cluster.append((target_space_id, target_concept_id))
                                visited_concepts.add((target_space_id, target_concept_id))
                    
                    i += 1
                    
                # Add cluster if it has multiple concepts
                if len(cluster) > 1:
                    concept_clusters.append(cluster)
                    
        # Project each cluster to blend space
        for cluster_idx, cluster in enumerate(concept_clusters):
            # Collect concept data
            cluster_concepts = {}
            for space_id, concept_id in cluster:
                space = spaces[space_id]
                concept = space.get_concept(concept_id)
                if concept:
                    if space_id not in cluster_concepts:
                        cluster_concepts[space_id] = []
                    cluster_concepts[space_id].append((concept_id, concept))
            
            # Create blended concept coordinates
            blended_coords = torch.zeros(len(blend_space.dimensions), device=self.device)
            coord_weights = torch.zeros(len(blend_space.dimensions), device=self.device)
            
            for space_id, concepts in cluster_concepts.items():
                space = spaces[space_id]
                projection_weight = self.projections.get((space_id, blend_id), {}).get('weight', 1.0)
                
                for concept_id, concept in concepts:
                    # Map each dimension from source to blend
                    for i, blend_dim in enumerate(blend_space.dimensions):
                        # Find matching dimension in source space
                        src_dim = None
                        
                        # Check direct match
                        if blend_dim in space.dimensions:
                            src_dim = blend_dim
                        # Check dimension mappings
                        elif blend_dim in dimension_mappings:
                            src_dim = dimension_mappings[blend_dim].get(space_id)
                            
                        if src_dim and src_dim in space.dim_to_idx:
                            src_idx = space.dim_to_idx[src_dim]
                            src_coord = concept['coordinates'][src_idx]
                            
                            # Add to blended coordinates with weight
                            blended_coords[i] += src_coord * projection_weight
                            coord_weights[i] += projection_weight
            
            # Normalize coordinates
            valid_dims = coord_weights > 0
            if torch.any(valid_dims):
                blended_coords[valid_dims] /= coord_weights[valid_dims]
                
            # Blend properties
            all_properties = []
            for space_id, concepts in cluster_concepts.items():
                for concept_id, concept in concepts:
                    all_properties.append(concept['properties'])
                    
            blended_properties = self._blend_properties(all_properties, blend_strategy)
            
            # Create blended concept ID and metadata
            blended_concept_id = f"blend_{cluster_idx}"
            
            # Record which concepts contributed to this blend
            source_concepts = {space_id: [concept_id for concept_id, _ in concepts]
                             for space_id, concepts in cluster_concepts.items()}
            
            # Add blended concept to blend space
            blend_space.add_concept(
                concept_id=blended_concept_id,
                coordinates=blended_coords,
                properties=blended_properties,
                metadata={'source_concepts': source_concepts}
            )
            
        return True
        
    async def _compose_blend(self, blend_id):
        """
        Run composition phase on the blend.
        Combines elements from the inputs in the blended space.
        
        Args:
            blend_id: ID of the blend space
            
        Returns:
            True if successful, False otherwise
        """
        blend_space = self.blend_spaces.get(blend_id)
        if not blend_space:
            return False
            
        # In a more complex implementation, this would detect and resolve property clashes,
        # perform frame blending, etc. For now, we'll just do simple property composition.
        
        # Get all concepts in the blend
        blend_concepts = list(blend_space.concepts.items())
        
        for concept_id, concept in blend_concepts:
            properties = concept['properties']
            
            # Check for numeric property interactions
            numeric_props = [(k, v) for k, v in properties.items() if isinstance(v, (int, float))]
            
            for i, (prop1, val1) in enumerate(numeric_props):
                for prop2, val2 in numeric_props[i+1:]:
                    # Look for patterns like "size" and "weight" that might correlate
                    if any(x in prop1 and x in prop2 for x in ["size", "weight", "height", "length"]):
                        # Create a derived property capturing the relationship
                        derived_prop = f"{prop1}_to_{prop2}_ratio"
                        if val2 != 0:  # Avoid division by zero
                            properties[derived_prop] = val1 / val2
            
            # Update the concept properties
            concept['properties'] = properties
            
        return True
        
    async def _complete_blend(self, blend_id):
        """
        Run completion phase on the blend.
        Fills in missing structure and infers additional properties.
        
        Args:
            blend_id: ID of the blend space
            
        Returns:
            True if successful, False otherwise
        """
        blend_space = self.blend_spaces.get(blend_id)
        if not blend_space:
            return False
            
        # In a more complex implementation, this would apply frames and schemas
        # to complete the blend with inferred structure. For now, we'll do a simple version.
        
        # Get all concepts in the blend
        blend_concepts = list(blend_space.concepts.items())
        
        for concept_id, concept in blend_concepts:
            properties = concept['properties']
            
            # Simple pattern completion
            # If we have both X and Y properties, check if we can infer Z
            property_patterns = [
                # If we have height and width, infer area
                (["height", "width"], "area", lambda h, w: h * w),
                # If we have weight and volume, infer density
                (["weight", "volume"], "density", lambda w, v: w / v if v != 0 else 0),
                # If we have speed and time, infer distance
                (["speed", "time"], "distance", lambda s, t: s * t),
            ]
            
            for required_props, inferred_prop, inference_func in property_patterns:
                # Check if we have all required properties and not the inferred one yet
                if all(any(req in p for p in properties.keys()) for req in required_props) and \
                   not any(inferred_prop in p for p in properties.keys()):
                    # Find the actual property names
                    actual_props = []
                    for req in required_props:
                        matching = [p for p in properties.keys() if req in p]
                        if matching:
                            actual_props.append(matching[0])
                    
                    if len(actual_props) == len(required_props):
                        # Extract values
                        values = [properties[p] for p in actual_props]
                        # Check if all values are numeric
                        if all(isinstance(v, (int, float)) for v in values):
                            # Infer the new property
                            try:
                                inferred_value = inference_func(*values)
                                properties[inferred_prop] = inferred_value
                            except:
                                pass  # Ignore errors in inference
            
            # Update the concept properties
            concept['properties'] = properties
            
        return True
        
    async def _elaborate_blend(self, blend_id):
        """
        Run elaboration phase on the blend.
        Runs the blend as an independent space to derive emergent structure.
        
        Args:
            blend_id: ID of the blend space
            
        Returns:
            True if successful, False otherwise
        """
        blend_space = self.blend_spaces.get(blend_id)
        if not blend_space:
            return False
            
        # In a full implementation, this would simulate the blend to discover emergent properties
        # For now, we'll implement a simple version that creates a few emergent properties
        
        # Get all concepts in the blend
        blend_concepts = list(blend_space.concepts.items())
        
        # Create emergent connections between concepts
        if len(blend_concepts) >= 2:
            # Calculate distances between all pairs
            distances = []
            for i, (id1, concept1) in enumerate(blend_concepts):
                for j, (id2, concept2) in enumerate(blend_concepts[i+1:], i+1):
                    distance = blend_space.distance(id1, id2)
                    distances.append((id1, id2, distance))
            
            # Find closest pairs
            distances.sort(key=lambda x: x[2])
            
            # Create emergent structure for closest pairs
            for id1, id2, distance in distances[:min(3, len(distances))]:
                concept1 = blend_space.get_concept(id1)
                concept2 = blend_space.get_concept(id2)
                
                if distance < 2.0:  # Close concepts
                    # Create emergent concept at midpoint
                    emergent_id = f"emergent_{id1}_{id2}"
                    
                    # Interpolate coordinates
                    coords = blend_space.interpolate(id1, id2)
                    
                    # Blend properties
                    properties = self._blend_properties(
                        [concept1['properties'], concept2['properties']], 
                        'selective'
                    )
                    
                    # Add emergent property
                    properties['emergent'] = True
                    properties['parent_concepts'] = [id1, id2]
                    
                    # Add to blend space
                    blend_space.add_concept(
                        concept_id=emergent_id,
                        coordinates=coords,
                        properties=properties,
                        metadata={'emergent': True, 'derived_from': [id1, id2]}
                    )
        
        return True
        
    def _blend_properties(self, property_dicts, strategy='selective'):
        """
        Blend multiple property dictionaries.
        
        Args:
            property_dicts: List of property dictionaries
            strategy: Blending strategy ('selective', 'union', 'intersection')
            
        Returns:
            Blended properties
        """
        if not property_dicts:
            return {}
            
        # Collect all properties
        all_properties = {}
        for props in property_dicts:
            for prop, value in props.items():
                if prop not in all_properties:
                    all_properties[prop] = []
                all_properties[prop].append(value)
                    
        # Blend properties according to strategy
        blended_properties = {}
        
        if strategy == 'union':
            # Take all properties
            for prop, values in all_properties.items():
                if all(isinstance(v, (int, float)) for v in values):
                    # For numeric properties, take mean
                    blended_properties[prop] = sum(values) / len(values)
                else:
                    # For non-numeric, take most common
                    value_counts = defaultdict(int)
                    for value in values:
                        value_counts[str(value)] += 1
                    most_common = max(value_counts.items(), key=lambda x: x[1])[0]
                    # Convert back to original type if possible
                    try:
                        if all(isinstance(v, bool) for v in values):
                            blended_properties[prop] = most_common.lower() == 'true'
                        elif all(isinstance(v, int) for v in values):
                            blended_properties[prop] = int(most_common)
                        elif all(isinstance(v, float) for v in values):
                            blended_properties[prop] = float(most_common)
                        else:
                            blended_properties[prop] = most_common
                    except:
                        blended_properties[prop] = most_common
                    
        elif strategy == 'intersection':
            # Take only properties that all concepts have
            for prop, values in all_properties.items():
                if len(values) == len(property_dicts):
                    if all(isinstance(v, (int, float)) for v in values):
                        # For numeric properties, take mean
                        blended_properties[prop] = sum(values) / len(values)
                    else:
                        # For non-numeric, take most common
                        value_counts = defaultdict(int)
                        for value in values:
                            value_counts[str(value)] += 1
                        most_common = max(value_counts.items(), key=lambda x: x[1])[0]
                        # Convert back to original type if possible
                        try:
                            if all(isinstance(v, bool) for v in values):
                                blended_properties[prop] = most_common.lower() == 'true'
                            elif all(isinstance(v, int) for v in values):
                                blended_properties[prop] = int(most_common)
                            elif all(isinstance(v, float) for v in values):
                                blended_properties[prop] = float(most_common)
                            else:
                                blended_properties[prop] = most_common
                        except:
                            blended_properties[prop] = most_common
                        
        else:  # selective
            # Take properties selectively based on relevance
            for prop, values in all_properties.items():
                # Determine relevance
                relevance = len(values) / len(property_dicts)
                
                if relevance >= 0.5:  # At least half the concepts have this property
                    if all(isinstance(v, (int, float)) for v in values):
                        # For numeric properties, take mean
                        blended_properties[prop] = sum(values) / len(values)
                    else:
                        # For non-numeric, take most common
                        value_counts = defaultdict(int)
                        for value in values:
                            value_counts[str(value)] += 1
                        most_common = max(value_counts.items(), key=lambda x: x[1])[0]
                        # Convert back to original type if possible
                        try:
                            if all(isinstance(v, bool) for v in values):
                                blended_properties[prop] = most_common.lower() == 'true'
                            elif all(isinstance(v, int) for v in values):
                                blended_properties[prop] = int(most_common)
                            elif all(isinstance(v, float) for v in values):
                                blended_properties[prop] = float(most_common)
                            else:
                                blended_properties[prop] = most_common
                        except:
                            blended_properties[prop] = most_common
                        
        return blended_properties
        
    def _calculate_alignment_score(self, space1, space2, concept_mappings):
        """
        Calculate alignment score between two spaces based on concept mappings.
        
        Args:
            space1: First conceptual space
            space2: Second conceptual space
            concept_mappings: Dict mapping concept IDs from space1 to space2
            
        Returns:
            Alignment score (0-1)
        """
        if not concept_mappings:
            return 0.0
            
        # Find common dimensions
        common_dims = set(space1.dimensions).intersection(space2.dimensions)
        
        if not common_dims:
            return 0.0
            
        # Calculate coordinate similarities for mapped concepts
        similarities = []
        for concept1_id, concept2_id in concept_mappings.items():
            concept1 = space1.get_concept(concept1_id)
            concept2 = space2.get_concept(concept2_id)
            
            if not concept1 or not concept2:
                continue
                
            # Project to common dimensions
            coords1 = space1.project_to_subspace(concept1['coordinates'], common_dims)
            coords2 = space2.project_to_subspace(concept2['coordinates'], common_dims)
            
            if coords1 is None or coords2 is None:
                continue
                
            # Calculate similarity (inverse of distance)
            distance = torch.norm(coords1 - coords2).item()
            similarity = 1.0 / (1.0 + distance)
            similarities.append(similarity)
            
            # Also check property alignment
            props1 = set(concept1['properties'].keys())
            props2 = set(concept2['properties'].keys())
            
            shared_props = props1.intersection(props2)
            prop_similarity = len(shared_props) / max(1, len(props1.union(props2)))
            similarities.append(prop_similarity)
            
        if not similarities:
            return 0.0
            
        return sum(similarities) / len(similarities)


class CounterfactualBlendingEngine:
    """
    Enhanced conceptual blending engine with counterfactual reasoning.
    Generates creative blends by exploring "what if" scenarios.
    """
    def __init__(self, semantic_network, concept_formation_engine, device=None):
        self.semantic_network = semantic_network
        self.concept_formation_engine = concept_formation_engine
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger("ASF.Layer3.CounterfactualBlending")
        
        # Integration network for blending
        self.blending_network = BlendingIntegrationNetwork(device=self.device)
        
        # Cache of generated blends
        self.blend_cache = {}
        
        # Tracking for blend evaluation
        self.blend_history = []
        self.blend_evaluations = {}
        
    async def create_blend(self, input_concept_ids, blend_strategy='selective', 
                         explore_counterfactuals=True):
        """
        Create a conceptual blend from input concepts.
        
        Args:
            input_concept_ids: List of concept IDs to blend
            blend_strategy: Strategy for blending ('selective', 'union', 'intersection')
            explore_counterfactuals: Whether to explore counterfactual scenarios
            
        Returns:
            Created blend details
        """
        blend_id = f"blend_{uuid.uuid4().hex[:8]}"
        
        # Check cache for similar blend request
        cache_key = f"{'-'.join(sorted(input_concept_ids))}_{blend_strategy}"
        if cache_key in self.blend_cache:
            # Return cached result with new ID
            cached_blend = dict(self.blend_cache[cache_key])
            cached_blend['blend_id'] = blend_id
            cached_blend['cached'] = True
            return cached_blend
        
        # Load input concepts from semantic network
        input_concepts = {}
        for concept_id in input_concept_ids:
            concept = await self.semantic_network.get_node(concept_id)
            if concept:
                input_concepts[concept_id] = concept
                
        if len(input_concepts) < 2:
            return {
                'status': 'error',
                'message': 'Need at least 2 valid input concepts',
                'blend_id': None
            }
            
        # Extract concept features and create input spaces
        input_spaces = {}
        
        for concept_id, concept in input_concepts.items():
            # Extract dimensions from properties
            properties = concept.properties if hasattr(concept, 'properties') else {}
            
            # Extract both structural and semantic dimensions
            dimensions = []
            
            # Add semantic dimensions from embeddings
            if hasattr(concept, 'embeddings') and concept.embeddings is not None:
                dimensions.extend([f"semantic_{i}" for i in range(min(10, len(concept.embeddings)))])
                
            # Add property dimensions
            for prop in properties:
                if prop not in dimensions:
                    dimensions.append(prop)
                    
            # Create input space if dimensions available
            if dimensions:
                space_id = f"space_{concept_id}"
                input_spaces[concept_id] = {
                    'space_id': space_id,
                    'dimensions': dimensions,
                    'concept': concept
                }
                
                # Create the space in the blending network
                self.blending_network.create_input_space(space_id, dimensions)
                
                # Add concept to space
                coordinates = self._extract_coordinates(concept, dimensions)
                self.blending_network.add_concept_to_space(
                    space_id=space_id,
                    concept_id=concept_id,
                    coordinates=coordinates,
                    properties=properties
                )
                
        # Create cross-space mappings
        self._create_cross_space_mappings(input_spaces)
        
        # Create generic space
        generic_space_id = f"generic_{blend_id}"
        self.blending_network.create_generic_space_from_inputs(
            generic_space_id=generic_space_id,
            input_space_ids=[space_data['space_id'] for space_data in input_spaces.values()]
        )
        
        # Create basic blend
        await self.blending_network.create_blend_from_inputs(
            blend_id=blend_id,
            input_space_ids=[space_data['space_id'] for space_data in input_spaces.values()],
            generic_space_id=generic_space_id,
            blend_strategy=blend_strategy
        )
        
        # Explore counterfactual variations if requested
        counterfactual_blends = []
        if explore_counterfactuals:
            counterfactual_blends = await self._explore_counterfactuals(
                blend_id=blend_id,
                input_concepts=input_concepts,
                input_spaces=input_spaces,
                count=3
            )
            
        # Evaluate blends
        base_score = await self._evaluate_blend(blend_id)
        blend_space = self.blending_network.blend_spaces.get(blend_id)
        
        # Compile blend result
        blend_result = {
            'status': 'success',
            'blend_id': blend_id,
            'input_concept_ids': input_concept_ids,
            'blend_concepts': self._extract_blend_concepts(blend_space) if blend_space else [],
            'emergent_properties': self._extract_emergent_properties(blend_space) if blend_space else [],
            'blend_strategy': blend_strategy,
            'blend_score': base_score,
            'counterfactual_blends': counterfactual_blends,
            'cached': False
        }
        
        # Cache the result
        self.blend_cache[cache_key] = blend_result
        
        # Add to history
        self.blend_history.append({
            'blend_id': blend_id,
            'input_concept_ids': input_concept_ids,
            'timestamp': time.time(),
            'blend_score': base_score,
            'counterfactual_count': len(counterfactual_blends)
        })
        
        return blend_result
        
    async def evaluate_blend(self, blend_id, evaluation_criteria=None):
        """
        Evaluate a blend based on various criteria.
        
        Args:
            blend_id: ID of the blend to evaluate
            evaluation_criteria: Optional custom evaluation criteria
            
        Returns:
            Evaluation results
        """
        if blend_id not in self.blending_network.blend_spaces:
            return {
                'status': 'error',
                'message': f'Blend {blend_id} not found',
                'score': 0.0
            }
            
        # Define default criteria if not provided
        if evaluation_criteria is None:
            evaluation_criteria = {
                'novelty': 0.3,
                'coherence': 0.3,
                'utility': 0.2,
                'emergent_structure': 0.2
            }
            
        blend_space = self.blending_network.blend_spaces[blend_id]
        
        # Calculate scores for each criterion
        scores = {}
        
        # Novelty: how different is the blend from the inputs
        scores['novelty'] = await self._evaluate_novelty(blend_id)
        
        # Coherence: how well do the elements fit together
        scores['coherence'] = self._evaluate_coherence(blend_space)
        
        # Utility: potential usefulness of the blend
        scores['utility'] = self._evaluate_utility(blend_space)
        
        # Emergent structure: amount of new structure that emerges
        scores['emergent_structure'] = self._evaluate_emergent_structure(blend_space)
        
        # Calculate overall score
        overall_score = sum(scores[criterion] * weight 
                          for criterion, weight in evaluation_criteria.items())
        
        # Store evaluation
        self.blend_evaluations[blend_id] = {
            'scores': scores,
            'overall_score': overall_score,
            'criteria': evaluation_criteria,
            'timestamp': time.time()
        }
        
        return {
            'status': 'success',
            'blend_id': blend_id,
            'scores': scores,
            'overall_score': overall_score
        }
        
    async def _explore_counterfactuals(self, blend_id, input_concepts, input_spaces, count=3):
        """
        Explore counterfactual variations of the blend.
        
        Args:
            blend_id: ID of the base blend
            input_concepts: Dict of input concepts
            input_spaces: Dict of input spaces
            count: Number of counterfactuals to generate
            
        Returns:
            List of counterfactual blends
        """
        counterfactuals = []
        
        # Define counterfactual strategies
        strategies = [
            self._counterfactual_property_variation,  # Vary properties
            self._counterfactual_concept_replacement,  # Replace a concept
            self._counterfactual_dimensional_shift,    # Shift dimensional focus
            self._counterfactual_emergent_properties   # Add emergent properties
        ]
        
        # Generate counterfactuals using different strategies
        for i, strategy in enumerate(strategies[:count]):
            cf_id = f"{blend_id}_cf_{i+1}"
            
            cf_result = await strategy(
                cf_id=cf_id,
                blend_id=blend_id,
                input_concepts=input_concepts,
                input_spaces=input_spaces
            )
            
            if cf_result and cf_result.get('status') == 'success':
                # Evaluate the counterfactual
                cf_score = await self._evaluate_blend(cf_id)
                cf_result['blend_score'] = cf_score
                
                counterfactuals.append(cf_result)
                
            if len(counterfactuals) >= count:
                break
                
        return counterfactuals
        
    async def _counterfactual_property_variation(self, cf_id, blend_id, input_concepts, input_spaces):
        """
        Create a counterfactual by varying properties of input concepts.
        
        Args:
            cf_id: ID for the counterfactual blend
            blend_id: ID of the base blend
            input_concepts: Dict of input concepts
            input_spaces: Dict of input spaces
            
        Returns:
            Counterfactual blend result
        """
        # Clone input spaces for the counterfactual
        cf_input_spaces = {}
        
        for concept_id, space_data in input_spaces.items():
            space_id = f"{space_data['space_id']}_cf"
            dimensions = space_data['dimensions'].copy()
            concept = input_concepts[concept_id]
            
            # Create modified space
            cf_input_spaces[concept_id] = {
                'space_id': space_id,
                'dimensions': dimensions,
                'concept': concept
            }
            
            # Create the space in the blending network
            self.blending_network.create_input_space(space_id, dimensions)
            
            # Create modified concept with varied properties
            properties = dict(concept.properties) if hasattr(concept, 'properties') else {}
            
            # Find numeric properties to vary
            numeric_props = [(k, v) for k, v in properties.items() if isinstance(v, (int, float))]
            
            if numeric_props:
                # Randomly select a property to modify
                prop_key, prop_val = numeric_props[np.random.randint(0, len(numeric_props))]
                
                # Modify the property value (increase or decrease by 20-50%)
                modifier = np.random.uniform(0.5, 1.5)
                properties[prop_key] = prop_val * modifier
                
                # Add counterfactual marker
                properties['_counterfactual'] = True
                properties['_cf_modified_property'] = prop_key
                properties['_cf_original_value'] = prop_val
                
            # Extract and potentially modify coordinates
            coordinates = self._extract_coordinates(concept, dimensions)
            
            # Add the modified concept to the space
            self.blending_network.add_concept_to_space(
                space_id=space_id,
                concept_id=concept_id,
                coordinates=coordinates,
                properties=properties
            )
            
        # Create cross-space mappings
        self._create_cross_space_mappings(cf_input_spaces)
        
        # Create generic space
        generic_space_id = f"generic_{cf_id}"
        self.blending_network.create_generic_space_from_inputs(
            generic_space_id=generic_space_id,
            input_space_ids=[space_data['space_id'] for space_data in cf_input_spaces.values()]
        )
        
        # Create counterfactual blend
        await self.blending_network.create_blend_from_inputs(
            blend_id=cf_id,
            input_space_ids=[space_data['space_id'] for space_data in cf_input_spaces.values()],
            generic_space_id=generic_space_id,
            blend_strategy='selective'
        )
        
        # Extract blend concepts
        blend_space = self.blending_network.blend_spaces.get(cf_id)
        if not blend_space:
            return {
                'status': 'error',
                'message': 'Failed to create counterfactual blend',
                'blend_id': cf_id
            }
            
        # Compile result
        return {
            'status': 'success',
            'blend_id': cf_id,
            'input_concept_ids': list(input_concepts.keys()),
            'blend_concepts': self._extract_blend_concepts(blend_space),
            'emergent_properties': self._extract_emergent_properties(blend_space),
            'counterfactual_type': 'property_variation',
            'parent_blend_id': blend_id
        }
        
    async def _counterfactual_concept_replacement(self, cf_id, blend_id, input_concepts, input_spaces):
        """
        Create a counterfactual by replacing one input concept with a similar one.
        
        Args:
            cf_id: ID for the counterfactual blend
            blend_id: ID of the base blend
            input_concepts: Dict of input concepts
            input_spaces: Dict of input spaces
            
        Returns:
            Counterfactual blend result
        """
        if len(input_concepts) < 2:
            return None
            
        # Choose a concept to replace
        replace_id = list(input_concepts.keys())[np.random.randint(0, len(input_concepts))]
        concept_to_replace = input_concepts[replace_id]
        
        # Find a similar concept to use as replacement
        similar_nodes = await self.semantic_network.get_similar_nodes(
            node_id=replace_id,
            k=3,
            threshold=0.6
        )
        
        if not similar_nodes:
            return None
            
        # Choose a replacement that's not already in the input set
        replacement_id = None
        for node_id, similarity in similar_nodes:
            if node_id not in input_concepts:
                replacement_id = node_id
                break
                
        if not replacement_id:
            # No suitable replacement found
            return None
            
        # Get the replacement concept
        replacement_concept = await self.semantic_network.get_node(replacement_id)
        if not replacement_concept:
            return None
            
        # Create new input concept set with replacement
        cf_input_concepts = {k: v for k, v in input_concepts.items() if k != replace_id}
        cf_input_concepts[replacement_id] = replacement_concept
        
        # Create clone of input spaces with replacement
        cf_input_spaces = {}
        
        for concept_id, space_data in input_spaces.items():
            if concept_id != replace_id:
                # Keep original space data
                space_id = f"{space_data['space_id']}_cf"
                dimensions = space_data['dimensions'].copy()
                concept = input_concepts[concept_id]
                
                cf_input_spaces[concept_id] = {
                    'space_id': space_id,
                    'dimensions': dimensions,
                    'concept': concept
                }
                
                # Create the space in the blending network
                self.blending_network.create_input_space(space_id, dimensions)
                
                # Add concept to space
                coordinates = self._extract_coordinates(concept, dimensions)
                self.blending_network.add_concept_to_space(
                    space_id=space_id,
                    concept_id=concept_id,
                    coordinates=coordinates,
                    properties=concept.properties if hasattr(concept, 'properties') else {}
                )
        
        # Add replacement concept space
        replacement_space_id = f"space_{replacement_id}_cf"
        
        # Extract dimensions from properties
        replacement_properties = replacement_concept.properties if hasattr(replacement_concept, 'properties') else {}
        
        # Extract both structural and semantic dimensions
        replacement_dimensions = []
        
        # Add semantic dimensions from embeddings
        if hasattr(replacement_concept, 'embeddings') and replacement_concept.embeddings is not None:
            replacement_dimensions.extend([f"semantic_{i}" for i in range(min(10, len(replacement_concept.embeddings)))])
            
        # Add property dimensions
        for prop in replacement_properties:
            if prop not in replacement_dimensions:
                replacement_dimensions.append(prop)
                
        # Create input space for replacement
        cf_input_spaces[replacement_id] = {
            'space_id': replacement_space_id,
            'dimensions': replacement_dimensions,
            'concept': replacement_concept
        }
        
        # Create the space in the blending network
        self.blending_network.create_input_space(replacement_space_id, replacement_dimensions)
        
        # Add replacement concept to space
        replacement_coordinates = self._extract_coordinates(replacement_concept, replacement_dimensions)
        self.blending_network.add_concept_to_space(
            space_id=replacement_space_id,
            concept_id=replacement_id,
            coordinates=replacement_coordinates,
            properties=replacement_properties
        )
        
        # Create cross-space mappings
        self._create_cross_space_mappings(cf_input_spaces)
        
        # Create generic space
        generic_space_id = f"generic_{cf_id}"
        self.blending_network.create_generic_space_from_inputs(
            generic_space_id=generic_space_id,
            input_space_ids=[space_data['space_id'] for space_data in cf_input_spaces.values()]
        )
        
        # Create counterfactual blend
        await self.blending_network.create_blend_from_inputs(
            blend_id=cf_id,
            input_space_ids=[space_data['space_id'] for space_data in cf_input_spaces.values()],
            generic_space_id=generic_space_id,
            blend_strategy='selective'
        )
        
        # Extract blend concepts
        blend_space = self.blending_network.blend_spaces.get(cf_id)
        if not blend_space:
            return {
                'status': 'error',
                'message': 'Failed to create counterfactual blend',
                'blend_id': cf_id
            }
            
        # Compile result
        return {
            'status': 'success',
            'blend_id': cf_id,
            'input_concept_ids': list(cf_input_concepts.keys()),
            'replaced_concept_id': replace_id,
            'replacement_concept_id': replacement_id,
            'blend_concepts': self._extract_blend_concepts(blend_space),
            'emergent_properties': self._extract_emergent_properties(blend_space),
            'counterfactual_type': 'concept_replacement',
            'parent_blend_id': blend_id
        }
        
    async def _counterfactual_dimensional_shift(self, cf_id, blend_id, input_concepts, input_spaces):
        """
        Create a counterfactual by shifting dimensional focus.
        
        Args:
            cf_id: ID for the counterfactual blend
            blend_id: ID of the base blend
            input_concepts: Dict of input concepts
            input_spaces: Dict of input spaces
            
        Returns:
            Counterfactual blend result
        """
        # Clone input spaces for the counterfactual
        cf_input_spaces = {}
        
        for concept_id, space_data in input_spaces.items():
            space_id = f"{space_data['space_id']}_cf"
            dimensions = space_data['dimensions'].copy()
            concept = input_concepts[concept_id]
            
            # Create modified space with adjusted dimension weights
            # This influences which aspects of the concepts are emphasized in the blend
            
            # Create the space in the blending network
            # Boost some dimensions and reduce others
            dimension_weights = None
            if len(dimensions) > 3:
                # Create unequal weights to shift focus
                weights = np.ones(len(dimensions))
                
                # Boost a few random dimensions
                boost_indices = np.random.choice(
                    len(dimensions), 
                    size=min(3, len(dimensions) // 2), 
                    replace=False
                )
                weights[boost_indices] = 3.0  # Triple importance
                
                # Reduce others
                reduce_indices = np.random.choice(
                    [i for i in range(len(dimensions)) if i not in boost_indices],
                    size=min(3, len(dimensions) // 2),
                    replace=False
                )
                weights[reduce_indices] = 0.3  # Reduce importance
                
                # Normalize
                weights = weights / np.sum(weights)
                dimension_weights = weights.tolist()
            
            self.blending_network.create_input_space(space_id, dimensions, dimension_weights)
            
            cf_input_spaces[concept_id] = {
                'space_id': space_id,
                'dimensions': dimensions,
                'concept': concept,
                'dimension_weights': dimension_weights
            }
            
            # Add concept to space
            coordinates = self._extract_coordinates(concept, dimensions)
            self.blending_network.add_concept_to_space(
                space_id=space_id,
                concept_id=concept_id,
                coordinates=coordinates,
                properties=concept.properties if hasattr(concept, 'properties') else {}
            )
            
        # Create cross-space mappings
        self._create_cross_space_mappings(cf_input_spaces)
        
        # Create generic space
        generic_space_id = f"generic_{cf_id}"
        self.blending_network.create_generic_space_from_inputs(
            generic_space_id=generic_space_id,
            input_space_ids=[space_data['space_id'] for space_data in cf_input_spaces.values()]
        )
        
        # Create counterfactual blend
        await self.blending_network.create_blend_from_inputs(
            blend_id=cf_id,
            input_space_ids=[space_data['space_id'] for space_data in cf_input_spaces.values()],
            generic_space_id=generic_space_id,
            blend_strategy='selective'
        )
        
        # Extract blend concepts
        blend_space = self.blending_network.blend_spaces.get(cf_id)
        if not blend_space:
            return {
                'status': 'error',
                'message': 'Failed to create counterfactual blend',
                'blend_id': cf_id
            }
            
        # Compile result
        # Extract which dimensions were emphasized
        emphasized_dimensions = {}
        for concept_id, space_data in cf_input_spaces.items():
            if space_data.get('dimension_weights'):
                dims = space_data['dimensions']
                weights = space_data['dimension_weights']
                top_dims = [dims[i] for i in np.argsort(weights)[-3:]]
                emphasized_dimensions[concept_id] = top_dims
        
        return {
            'status': 'success',
            'blend_id': cf_id,
            'input_concept_ids': list(input_concepts.keys()),
            'blend_concepts': self._extract_blend_concepts(blend_space),
            'emergent_properties': self._extract_emergent_properties(blend_space),
            'counterfactual_type': 'dimensional_shift',
            'emphasized_dimensions': emphasized_dimensions,
            'parent_blend_id': blend_id
        }
        
    async def _counterfactual_emergent_properties(self, cf_id, blend_id, input_concepts, input_spaces):
        """
        Create a counterfactual by adding emergent properties.
        
        Args:
            cf_id: ID for the counterfactual blend
            blend_id: ID of the base blend
            input_concepts: Dict of input concepts
            input_spaces: Dict of input spaces
            
        Returns:
            Counterfactual blend result
        """
        # Start by creating a normal blend
        original_blend_space = self.blending_network.blend_spaces.get(blend_id)
        if not original_blend_space:
            return None
            
        # Create clone of input spaces
        cf_input_spaces = {}
        
        for concept_id, space_data in input_spaces.items():
            space_id = f"{space_data['space_id']}_cf"
            dimensions = space_data['dimensions'].copy()
            concept = input_concepts[concept_id]
            
            cf_input_spaces[concept_id] = {
                'space_id': space_id,
                'dimensions': dimensions,
                'concept': concept
            }
            
            # Create the space in the blending network
            self.blending_network.create_input_space(space_id, dimensions)
            
            # Add concept to space
            coordinates = self._extract_coordinates(concept, dimensions)
            self.blending_network.add_concept_to_space(
                space_id=space_id,
                concept_id=concept_id,
                coordinates=coordinates,
                properties=concept.properties if hasattr(concept, 'properties') else {}
            )
            
        # Create cross-space mappings
        self._create_cross_space_mappings(cf_input_spaces)
        
        # Create generic space
        generic_space_id = f"generic_{cf_id}"
        self.blending_network.create_generic_space_from_inputs(
            generic_space_id=generic_space_id,
            input_space_ids=[space_data['space_id'] for space_data in cf_input_spaces.values()]
        )
        
        # Create counterfactual blend
        await self.blending_network.create_blend_from_inputs(
            blend_id=cf_id,
            input_space_ids=[space_data['space_id'] for space_data in cf_input_spaces.values()],
            generic_space_id=generic_space_id,
            blend_strategy='selective'
        )
        
        # Now add additional emergent properties that wouldn't normally arise
        blend_space = self.blending_network.blend_spaces.get(cf_id)
        if not blend_space:
            return None
            
        # Collect numeric properties from all concepts
        numeric_properties = {}
        
        for concept_id, concept in input_concepts.items():
            properties = concept.properties if hasattr(concept, 'properties') else {}
            for prop_name, prop_val in properties.items():
                if isinstance(prop_val, (int, float)):
                    if prop_name not in numeric_properties:
                        numeric_properties[prop_name] = []
                    numeric_properties[prop_name].append(prop_val)
                    
        # Generate emergent property patterns
        emergent_patterns = [
            # Opposite property (inversion)
            lambda name, vals: (f"inverse_{name}", -np.mean(vals)),
            
            # Squared property
            lambda name, vals: (f"{name}_squared", np.mean(vals) ** 2),
            
            # Order of magnitude shift
            lambda name, vals: (f"{name}_magnified", np.mean(vals) * 10),
            
            # Combination of two properties
            lambda name, vals: (f"{name}_combined", np.mean(vals) * np.std(vals))
        ]
        
        # Apply emergent patterns to blend concepts
        for concept_id in blend_space.concepts:
            concept = blend_space.get_concept(concept_id)
            if not concept:
                continue
                
            # Add emergent properties based on patterns
            for prop_name, prop_vals in numeric_properties.items():
                if len(prop_vals) >= 2:  # Need multiple values for meaningful emergence
                    # Randomly select a pattern
                    pattern = emergent_patterns[np.random.randint(0, len(emergent_patterns))]
                    
                    # Generate emergent property
                    try:
                        emergent_name, emergent_val = pattern(prop_name, prop_vals)
                        concept['properties'][emergent_name] = emergent_val
                        # Mark as counterfactual
                        concept['properties']['_counterfactual_emergence'] = True
                        concept['properties']['_cf_emergent_prop'] = emergent_name
                    except:
                        pass  # Ignore errors in pattern application
        
        # Extract blend concepts after adding emergent properties
        return {
            'status': 'success',
            'blend_id': cf_id,
            'input_concept_ids': list(input_concepts.keys()),
            'blend_concepts': self._extract_blend_concepts(blend_space),
            'emergent_properties': self._extract_emergent_properties(blend_space),
            'counterfactual_type': 'emergent_properties',
            'parent_blend_id': blend_id
        }
        
    def _create_cross_space_mappings(self, input_spaces):
        """
        Create cross-space mappings between input spaces.
        
        Args:
            input_spaces: Dict of input spaces
            
        Returns:
            True if successful, False otherwise
        """
        space_ids = [data['space_id'] for data in input_spaces.values()]
        
        # Create mappings between all pairs of spaces
        for i, (concept_id1, space_data1) in enumerate(input_spaces.items()):
            space_id1 = space_data1['space_id']
            
            for j, (concept_id2, space_data2) in enumerate(input_spaces.items()):
                if i >= j:  # Skip self and duplicates
                    continue
                    
                space_id2 = space_data2['space_id']
                
                # Create concept mapping (1:1 for now)
                concept_mappings = {concept_id1: concept_id2}
                
                # Create dimension mappings
                dimension_mappings = {}
                
                # Map identical dimensions
                for dim1 in space_data1['dimensions']:
                    if dim1 in space_data2['dimensions']:
                        dimension_mappings[dim1] = dim1
                        
                # Create the mapping
                self.blending_network.create_cross_space_mapping(
                    space1_id=space_id1,
                    space2_id=space_id2,
                    concept_mappings=concept_mappings,
                    dimension_mappings=dimension_mappings
                )
                
        return True
        
    def _extract_coordinates(self, concept, dimensions):
        """
        Extract coordinates for dimensions from a concept.
        
        Args:
            concept: Concept node
            dimensions: List of dimension names
            
        Returns:
            Coordinates tensor
        """
        coordinates = torch.zeros(len(dimensions), device=self.device)
        
        for i, dim in enumerate(dimensions):
            if dim.startswith("semantic_") and hasattr(concept, 'embeddings'):
                # Extract from embeddings
                semantic_idx = int(dim.split('_')[1])
                if semantic_idx < len(concept.embeddings):
                    if isinstance(concept.embeddings, np.ndarray):
                        coordinates[i] = float(concept.embeddings[semantic_idx])
                    elif isinstance(concept.embeddings, torch.Tensor):
                        coordinates[i] = concept.embeddings[semantic_idx]
            elif hasattr(concept, 'properties') and dim in concept.properties:
                # Extract from properties
                prop_val = concept.properties[dim]
                if isinstance(prop_val, (int, float)):
                    coordinates[i] = float(prop_val)
                    
        return coordinates
        
    def _extract_blend_concepts(self, blend_space):
        """
        Extract concept data from a blend space.
        
        Args:
            blend_space: ConceptualSpace for the blend
            
        Returns:
            List of blend concept data
        """
        if not blend_space:
            return []
            
        blend_concepts = []
        
        for concept_id, concept in blend_space.concepts.items():
            # Extract properties
            properties = concept['properties']
            
            # Extract source concepts if available
            source_concepts = concept['metadata'].get('source_concepts', {})
            
            # Add to results
            blend_concepts.append({
                'concept_id': concept_id,
                'properties': properties,
                'source_concepts': source_concepts,
                'emergent': concept['metadata'].get('emergent', False)
            })
            
        return blend_concepts
        
    def _extract_emergent_properties(self, blend_space):
        """
        Extract emergent properties from a blend space.
        
        Args:
            blend_space: ConceptualSpace for the blend
            
        Returns:
            List of emergent properties
        """
        if not blend_space:
            return []
            
        emergent_properties = []
        
        for concept_id, concept in blend_space.concepts.items():
            # Look for emergent properties
            for prop_name, prop_val in concept['properties'].items():
                # Check if property is emergent
                is_emergent = False
                
                # Check explicit marking
                if concept['metadata'].get('emergent', False):
                    is_emergent = True
                    
                # Check property naming patterns
                elif any(marker in prop_name for marker in ["emergent", "combined", "derived", 
                                                         "ratio", "inverse", "squared",
                                                         "magnified"]):
                    is_emergent = True
                    
                # Check counterfactual markings
                elif "_counterfactual_emergence" in concept['properties']:
                    is_emergent = True
                    
                if is_emergent:
                    emergent_properties.append({
                        'property_name': prop_name,
                        'property_value': prop_val,
                        'concept_id': concept_id,
                        'counterfactual': "_counterfactual" in concept['properties']
                    })
                    
        return emergent_properties
        
    async def _evaluate_blend(self, blend_id):
        """
        Evaluate a blend and return overall score.
        
        Args:
            blend_id: ID of the blend to evaluate
            
        Returns:
            Overall evaluation score
        """
        evaluation = await self.evaluate_blend(blend_id)
        return evaluation.get('overall_score', 0.0)
        
    async def _evaluate_novelty(self, blend_id):
        """
        Evaluate how novel the blend is compared to inputs.
        
        Args:
            blend_id: ID of the blend to evaluate
            
        Returns:
            Novelty score (0-1)
        """
        blend_space = self.blending_network.blend_spaces.get(blend_id)
        if not blend_space:
            return 0.0
            
        # Get projections to find input spaces
        input_space_ids = []
        for (space_id, target_id), projection in self.blending_network.projections.items():
            if target_id == blend_id and projection['projection_type'] == 'input_to_blend':
                input_space_ids.append(space_id)
                
        if not input_space_ids:
            return 0.5  # Default if no inputs found
            
        # Compare blend concepts to input concepts
        novelty_scores = []
        
        for blend_concept_id, blend_concept in blend_space.concepts.items():
            # Skip concepts explicitly marked as non-emergent
            if blend_concept['metadata'].get('emergent', False):
                continue
                
            # Get properties that this concept has
            blend_props = set(blend_concept['properties'].keys())
            
            # Find source concepts for this blend
            source_concepts = blend_concept['metadata'].get('source_concepts', {})
            
            # For each input space
            for space_id in input_space_ids:
                input_space = self.blending_network.input_spaces.get(space_id)
                if not input_space:
                    continue
                    
                # Get concept IDs for this space
                concept_ids = source_concepts.get(space_id, [])
                if not concept_ids:
                    continue
                    
                for concept_id in concept_ids:
                    input_concept = input_space.get_concept(concept_id)
                    if not input_concept:
                        continue
                        
                    # Compare property sets
                    input_props = set(input_concept['properties'].keys())
                    
                    # Property overlap
                    shared_props = blend_props.intersection(input_props)
                    all_props = blend_props.union(input_props)
                    
                    if all_props:
                        # Lower overlap = higher novelty
                        novelty = 1.0 - (len(shared_props) / len(all_props))
                        novelty_scores.append(novelty)
        
        if not novelty_scores:
            return 0.5  # Default if no comparisons made
            
        return sum(novelty_scores) / len(novelty_scores)
        
    def _evaluate_coherence(self, blend_space):
        """
        Evaluate how coherent the blend is.
        
        Args:
            blend_space: Blend space to evaluate
            
        Returns:
            Coherence score (0-1)
        """
        if not blend_space or not blend_space.concepts:
            return 0.0
            
        # Calculate average pairwise distance between blend concepts
        distances = []
        
        concept_ids = list(blend_space.concepts.keys())
        
        for i, id1 in enumerate(concept_ids):
            for j, id2 in enumerate(concept_ids[i+1:], i+1):
                distance = blend_space.distance(id1, id2)
                distances.append(distance)
                
        if not distances:
            return 0.5  # Default if no distances
            
        # Convert to coherence score (lower distance = higher coherence)
        avg_distance = sum(distances) / len(distances)
        
        # Normalize and invert (0 = low coherence, 1 = high coherence)
        coherence = 1.0 / (1.0 + avg_distance)
        
        return coherence
        
    def _evaluate_utility(self, blend_space):
        """
        Evaluate potential utility of the blend.
        
        Args:
            blend_space: Blend space to evaluate
            
        Returns:
            Utility score (0-1)
        """
        if not blend_space or not blend_space.concepts:
            return 0.0
            
        # Count number of properties per concept
        property_counts = []
        
        for concept_id, concept in blend_space.concepts.items():
            property_counts.append(len(concept['properties']))
            
        if not property_counts:
            return 0.0
            
        # More properties generally indicates higher utility
        avg_properties = sum(property_counts) / len(property_counts)
        
        # Normalize (assuming typical concepts have 5-10 properties)
        utility = min(1.0, avg_properties / 10.0)
        
        return utility
        
    def _evaluate_emergent_structure(self, blend_space):
        """
        Evaluate amount of emergent structure in the blend.
        
        Args:
            blend_space: Blend space to evaluate
            
        Returns:
            Emergent structure score (0-1)
        """
        if not blend_space or not blend_space.concepts:
            return 0.0
            
        # Count emergent properties and concepts
        emergent_concept_count = 0
        emergent_property_count = 0
        total_concepts = len(blend_space.concepts)
        
        for concept_id, concept in blend_space.concepts.items():
            # Count emergent concepts
            if concept['metadata'].get('emergent', False):
                emergent_concept_count += 1
                
            # Count emergent properties
            for prop_name in concept['properties']:
                if any(marker in prop_name for marker in ["emergent", "combined", "derived", 
                                                         "ratio", "inverse", "squared",
                                                         "magnified"]):
                    emergent_property_count += 1
                    
        # Calculate score based on emergence ratio
        concept_ratio = emergent_concept_count / max(1, total_concepts)
        
        # Assume average of 5 properties per concept
        property_ratio = emergent_property_count / max(1, total_concepts * 5)
        
        # Combine scores
        emergence_score = 0.4 * concept_ratio + 0.6 * property_ratio
        
        return emergence_score