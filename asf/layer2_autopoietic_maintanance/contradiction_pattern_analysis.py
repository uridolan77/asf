"""
Contradiction Pattern Analysis Module for ASF Maintenance Engine

This module implements eigenvalue-based pattern detection for contradiction analysis,
a key enhancement that helps the system identify underlying patterns in contradictions.
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Set, Union
import uuid

class PatternAnalyzer:
    """
    Implements eigenvalue-based pattern detection for contradiction analysis.
    
    This class analyzes contradictions across a domain to identify underlying patterns,
    mapping them to knowledge evolution patterns from the Knowledge Substrate Layer.
    """
    
    def __init__(self):
        """Initialize the pattern analyzer."""
        # Store known patterns by domain
        self.known_patterns = {}
        
        # Configuration
        self.similarity_threshold = 0.6  # Threshold for pattern similarity
        self.entity_match_weight = 0.4   # Weight for entity matching in similarity
        self.attribute_match_weight = 0.3  # Weight for attribute matching
        self.type_match_weight = 0.3     # Weight for contradiction type matching
    
    def construct_contradiction_matrix(self, domain_contradictions: List[Dict]) -> np.ndarray:
        """
        Construct a matrix representation of contradictions for eigenvalue analysis.
        
        Args:
            domain_contradictions: List of contradictions in a domain
            
        Returns:
            Contradiction matrix suitable for eigenvalue decomposition
        """
        n_contradictions = len(domain_contradictions)
        matrix = np.zeros((n_contradictions, n_contradictions))
        
        # Populate matrix based on similarity/relationship between contradictions
        for i in range(n_contradictions):
            for j in range(n_contradictions):
                if i == j:
                    matrix[i, j] = 1.0  # Self-similarity
                else:
                    # Calculate similarity between contradictions
                    similarity = self.calculate_contradiction_similarity(
                        domain_contradictions[i], 
                        domain_contradictions[j]
                    )
                    matrix[i, j] = similarity
        
        return matrix
    
    def identify_contradiction_patterns(self, 
                                      contradiction_matrix: np.ndarray,
                                      domain_contradictions: List[Dict],
                                      significance_threshold: float = 0.4) -> List[Dict]:
        """
        Uses eigenvalue decomposition to identify patterns in domain contradictions.
        
        Args:
            contradiction_matrix: Matrix of contradiction similarities
            domain_contradictions: List of contradictions in a domain
            significance_threshold: Threshold for pattern significance
            
        Returns:
            List of significant patterns with resolution strategies
        """
        # Apply eigenvalue decomposition
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(contradiction_matrix)
            
            # Sort by eigenvalue magnitude
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Identify significant patterns
            significant_patterns = []
            for i in range(min(5, len(eigenvalues))):
                if eigenvalues[i] > significance_threshold:
                    component = eigenvectors[:, i]
                    
                    # Find contradiction types most associated with this pattern
                    top_indices = np.argsort(np.abs(component))[-5:][::-1]
                    pattern_contradictions = [domain_contradictions[idx] for idx in top_indices if idx < len(domain_contradictions)]
                    
                    # Generate a unique pattern ID
                    pattern_id = f"pattern_{uuid.uuid4().hex[:8]}"
                    
                    # Extract key attributes from top contradictions
                    pattern_attributes = self._extract_pattern_attributes(pattern_contradictions)
                    
                    # Determine resolution strategy based on pattern
                    resolution_strategy = self._determine_pattern_resolution(
                        component, pattern_contradictions, pattern_attributes)
                    
                    significant_patterns.append({
                        'pattern_id': pattern_id,
                        'eigenvalue': float(eigenvalues[i]),
                        'explained_variance': float(eigenvalues[i] / sum(eigenvalues)),
                        'top_contradictions': pattern_contradictions,
                        'pattern_attributes': pattern_attributes,
                        'resolution_strategy': resolution_strategy,
                        'component': component.tolist()
                    })
            
            return significant_patterns
        except Exception as e:
            print(f"Error in eigenvalue decomposition: {str(e)}")
            return []
    
    def calculate_contradiction_similarity(self, 
                                         contradiction1: Dict, 
                                         contradiction2: Dict) -> float:
        """
        Calculate similarity between two contradictions based on multiple factors.
        
        Args:
            contradiction1: First contradiction
            contradiction2: Second contradiction
            
        Returns:
            Similarity score (0.0-1.0)
        """
        similarity = 0.0
        
        # Get the actual contradiction lists
        c1_list = contradiction1.get('contradictions', [])
        c2_list = contradiction2.get('contradictions', [])
        
        if not c1_list or not c2_list:
            return 0.0
        
        # Entity similarity
        entity_similarity = 0.0
        if contradiction1.get('entity_id') == contradiction2.get('entity_id'):
            entity_similarity = 1.0
        similarity += self.entity_match_weight * entity_similarity
        
        # Contradiction type similarity
        type_similarity = self._calculate_type_similarity(c1_list, c2_list)
        similarity += self.type_match_weight * type_similarity
        
        # Attribute similarity
        attribute_similarity = self._calculate_attribute_similarity(c1_list, c2_list)
        similarity += self.attribute_match_weight * attribute_similarity
        
        # Temporal proximity
        if 'timestamp' in contradiction1 and 'timestamp' in contradiction2:
            temporal_similarity = self._calculate_temporal_proximity(
                contradiction1['timestamp'],
                contradiction2['timestamp']
            )
            # Add a small weight for temporal proximity
            similarity += 0.1 * temporal_similarity
        
        return min(1.0, similarity)
    
    def _calculate_type_similarity(self, c1_list: List[Dict], c2_list: List[Dict]) -> float:
        """Calculate similarity based on contradiction types."""
        # Extract types and subtypes
        c1_types = [(c.get('type'), c.get('subtype')) for c in c1_list]
        c2_types = [(c.get('type'), c.get('subtype')) for c in c2_list]
        
        # Count exact matches (type and subtype)
        exact_matches = len(set(c1_types).intersection(set(c2_types)))
        
        # Count type-only matches
        c1_main_types = [t[0] for t in c1_types]
        c2_main_types = [t[0] for t in c2_types]
        type_matches = len(set(c1_main_types).intersection(set(c2_main_types)))
        
        # Calculate similarity score
        max_possible = max(len(c1_types), len(c2_types))
        if max_possible == 0:
            return 0.0
            
        # Exact matches count fully, type-only matches count partially
        score = (exact_matches + 0.5 * (type_matches - exact_matches)) / max_possible
        return score
    
    def _calculate_attribute_similarity(self, c1_list: List[Dict], c2_list: List[Dict]) -> float:
        """Calculate similarity based on affected attributes."""
        # Extract attributes
        c1_attrs = [c.get('attribute') for c in c1_list if 'attribute' in c]
        c2_attrs = [c.get('attribute') for c in c2_list if 'attribute' in c]
        
        if not c1_attrs or not c2_attrs:
            return 0.0
            
        # Count exact matches
        exact_matches = len(set(c1_attrs).intersection(set(c2_attrs)))
        
        # Calculate similarity score
        max_possible = max(len(c1_attrs), len(c2_attrs))
        return exact_matches / max_possible
    
    def _calculate_temporal_proximity(self, timestamp1: str, timestamp2: str) -> float:
        """Calculate temporal proximity between contradictions."""
        try:
            time1 = datetime.fromisoformat(timestamp1)
            time2 = datetime.fromisoformat(timestamp2)
            
            # Calculate time difference in hours
            time_diff = abs((time1 - time2).total_seconds()) / 3600
            
            # Convert to similarity score (1.0 for same time, decaying with difference)
            # Using exponential decay with 24-hour half-life
            return np.exp(-time_diff / 24)
        except:
            return 0.0
    
    def _extract_pattern_attributes(self, pattern_contradictions: List[Dict]) -> Dict:
        """Extract key attributes that characterize a contradiction pattern."""
        # Count contradiction types
        type_counts = {}
        subtype_counts = {}
        attributes = set()
        affected_entities = set()
        
        for contradiction in pattern_contradictions:
            entity_id = contradiction.get('entity_id')
            if entity_id:
                affected_entities.add(entity_id)
                
            # Process individual contradictions
            c_list = contradiction.get('contradictions', [])
            for c in c_list:
                # Count types and subtypes
                c_type = c.get('type')
                if c_type:
                    type_counts[c_type] = type_counts.get(c_type, 0) + 1
                    
                c_subtype = c.get('subtype')
                if c_subtype:
                    subtype_counts[c_subtype] = subtype_counts.get(c_subtype, 0) + 1
                    
                # Collect affected attributes
                if 'attribute' in c:
                    attributes.add(c['attribute'])
        
        # Determine dominant type and subtype
        dominant_type = max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else None
        dominant_subtype = max(subtype_counts.items(), key=lambda x: x[1])[0] if subtype_counts else None
        
        return {
            'dominant_type': dominant_type,
            'dominant_subtype': dominant_subtype,
            'type_distribution': type_counts,
            'subtype_distribution': subtype_counts,
            'affected_attributes': list(attributes),
            'affected_entities': list(affected_entities)
        }
    
    def _determine_pattern_resolution(self, component: np.ndarray,
                                     pattern_contradictions: List[Dict],
                                     pattern_attributes: Dict) -> Dict:
        """
        Determine the appropriate resolution strategy based on eigenpattern.
        
        Args:
            component: Eigenvector representing the contradiction pattern
            pattern_contradictions: Contradictions in this pattern
            pattern_attributes: Key attributes of the pattern
            
        Returns:
            Resolution strategy dictionary
        """
        # Analyze component characteristics
        component_magnitude = np.linalg.norm(component)
        component_variance = np.var(component)
        positive_weight = np.sum(component > 0)
        negative_weight = np.sum(component < 0)
        
        # Get dominant contradiction type
        dominant_type = pattern_attributes.get('dominant_type')
        dominant_subtype = pattern_attributes.get('dominant_subtype')
        
        # Determine pattern type based on component characteristics and contradiction types
        if dominant_type == 'attribute_value':
            if dominant_subtype == 'numeric_divergence':
                if component_variance < 0.1 and component_magnitude > 0.8:
                    # Systematic numeric shifts suggest contextual boundary
                    strategy_type = "contextual_boundary_refinement"
                    confidence_modifier = 0.8
                else:
                    # Variable numeric changes suggest temporal evolution
                    strategy_type = "temporal_evolution_tracking"
                    confidence_modifier = 0.7
            elif dominant_subtype == 'text_replacement':
                # Text replacements often indicate definitional changes
                strategy_type = "definitional_harmonization"
                confidence_modifier = 0.6
            else:
                strategy_type = "attribute_refinement"
                confidence_modifier = 0.5
                
        elif dominant_type == 'relationship':
            # Relationship contradictions often benefit from perspective integration
            strategy_type = "perspective_integration"
            confidence_modifier = 0.6
            
        elif dominant_type == 'temporal_pattern':
            # Temporal pattern contradictions typically indicate evolution
            strategy_type = "temporal_evolution_tracking"
            confidence_modifier = 0.8
            
        elif dominant_type == 'expectation_violation':
            if component_variance < 0.1:
                # Consistent violations suggest boundary issues
                strategy_type = "contextual_boundary_refinement"
                confidence_modifier = 0.7
            else:
                # Variable violations suggest multiple perspectives
                strategy_type = "perspective_integration"
                confidence_modifier = 0.6
        else:
            # Default approach for unknown patterns
            strategy_type = "graduated_confidence_adjustment"
            confidence_modifier = 0.5
        
        return {
            'strategy_type': strategy_type,
            'confidence_modifier': confidence_modifier,
            'confidence': 0.7,  # Base confidence in this strategy
            'justification': f"Pattern analysis detected {dominant_type}/{dominant_subtype} pattern with eigenvalue magnitude {component_magnitude:.2f}"
        }
    
    def update_known_patterns(self, domain: str, new_patterns: List[Dict]) -> None:
        """
        Update the catalog of known patterns for a domain.
        
        Args:
            domain: Knowledge domain
            new_patterns: New patterns detected for this domain
        """
        if domain not in self.known_patterns:
            self.known_patterns[domain] = []
        
        # For each new pattern, check if it matches existing ones
        for new_pattern in new_patterns:
            matched = False
            
            # Check for matches with existing patterns
            for existing_pattern in self.known_patterns[domain]:
                similarity = self._calculate_pattern_similarity(new_pattern, existing_pattern)
                
                if similarity > self.similarity_threshold:
                    # Update existing pattern with new information
                    self._merge_patterns(existing_pattern, new_pattern)
                    matched = True
                    break
            
            # If no match found, add as new pattern
            if not matched:
                # Generate a new ID if not present
                if 'pattern_id' not in new_pattern:
                    new_pattern['pattern_id'] = f"pattern_{uuid.uuid4().hex[:8]}"
                
                # Add creation timestamp
                if 'created' not in new_pattern:
                    new_pattern['created'] = datetime.now().isoformat()
                
                # Add to known patterns
                self.known_patterns[domain].append(new_pattern)
    
    def _calculate_pattern_similarity(self, pattern1: Dict, pattern2: Dict) -> float:
        """Calculate similarity between two patterns."""
        similarity_score = 0.0
        
        # Compare pattern attributes
        attrs1 = pattern1.get('pattern_attributes', {})
        attrs2 = pattern2.get('pattern_attributes', {})
        
        # Compare dominant type/subtype
        if attrs1.get('dominant_type') == attrs2.get('dominant_type'):
            similarity_score += 0.3
            
        if attrs1.get('dominant_subtype') == attrs2.get('dominant_subtype'):
            similarity_score += 0.2
        
        # Compare affected attributes
        attr_set1 = set(attrs1.get('affected_attributes', []))
        attr_set2 = set(attrs2.get('affected_attributes', []))
        
        if attr_set1 and attr_set2:
            jaccard_attrs = len(attr_set1.intersection(attr_set2)) / len(attr_set1.union(attr_set2))
            similarity_score += 0.25 * jaccard_attrs
        
        # Compare affected entities
        entity_set1 = set(attrs1.get('affected_entities', []))
        entity_set2 = set(attrs2.get('affected_entities', []))
        
        if entity_set1 and entity_set2:
            jaccard_entities = len(entity_set1.intersection(entity_set2)) / len(entity_set1.union(entity_set2))
            similarity_score += 0.25 * jaccard_entities
        
        return similarity_score
    
    def _merge_patterns(self, existing_pattern: Dict, new_pattern: Dict) -> None:
        """Merge a new pattern into an existing one, updating relevant fields."""
        # Update last seen timestamp
        existing_pattern['last_seen'] = datetime.now().isoformat()
        
        # Increment observation count
        existing_pattern['observation_count'] = existing_pattern.get('observation_count', 1) + 1
        
        # Update confidence based on repeated observations
        confidence_boost = min(0.1, 1.0 / existing_pattern.get('observation_count', 1))
        existing_resolution = existing_pattern.get('resolution_strategy', {})
        existing_resolution['confidence'] = min(
            0.95, 
            existing_resolution.get('confidence', 0.7) + confidence_boost
        )
        
        # Update affected entities
        existing_attrs = existing_pattern.get('pattern_attributes', {})
        new_attrs = new_pattern.get('pattern_attributes', {})
        
        existing_entities = set(existing_attrs.get('affected_entities', []))
        new_entities = set(new_attrs.get('affected_entities', []))
        
        existing_attrs['affected_entities'] = list(existing_entities.union(new_entities))
    
    def match_to_known_patterns(self, contradictions: List[Dict], 
                              domain: str, 
                              recent_history: List[Dict] = None) -> Optional[Dict]:
        """
        Match a set of contradictions to known patterns.
        
        Args:
            contradictions: List of contradictions to match
            domain: Knowledge domain
            recent_history: Recent contradiction history (optional)
            
        Returns:
            Matched pattern or None if no match found
        """
        if domain not in self.known_patterns or not self.known_patterns[domain]:
            return None
        
        # Create a temporary pattern from the contradictions
        temp_pattern = {
            'contradictions': contradictions,
            'pattern_attributes': self._extract_pattern_attributes([
                {'contradictions': contradictions}
            ])
        }
        
        # Find best matching pattern
        best_match = None
        best_similarity = 0.0
        
        for pattern in self.known_patterns[domain]:
            similarity = self._calculate_pattern_similarity(temp_pattern, pattern)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = pattern
        
        # If similarity exceeds threshold, return the match
        if best_similarity > self.similarity_threshold:
            return {
                'pattern_id': best_match['pattern_id'],
                'similarity': best_similarity,
                'confidence': best_similarity,
                'pattern': best_match
            }
        
        return None
    
    def map_contradiction_to_knowledge_patterns(self, 
                                              contradiction_patterns: List[Dict],
                                              compressed_history) -> List[Dict]:
        """
        Map contradiction patterns to knowledge evolution patterns from Knowledge Substrate.
        
        Args:
            contradiction_patterns: Identified contradiction patterns
            compressed_history: Compressed knowledge history from Knowledge Substrate
            
        Returns:
            Mapping between contradiction and knowledge patterns
        """
        mappings = []
        
        # For each contradiction pattern
        for c_pattern in contradiction_patterns:
            best_match = None
            highest_correlation = 0
            
            # Compare against knowledge evolution patterns
            k_shape = compressed_history.eigenvectors.shape
            for k_idx in range(min(k_shape[1], 5)):  # Check first 5 components
                knowledge_pattern = compressed_history.eigenvectors[:, k_idx]
                
                # Extract entities involved in this contradiction pattern
                contradiction_entities = c_pattern.get('pattern_attributes', {}).get('affected_entities', [])
                
                # Find these entities in the knowledge pattern
                entity_indices = []
                for entity in contradiction_entities:
                    try:
                        idx = compressed_history.entities.index(entity)
                        entity_indices.append(idx)
                    except ValueError:
                        continue
                
                if not entity_indices:
                    continue
                    
                # Calculate correlation between patterns for these entities
                entity_weights = np.array([abs(knowledge_pattern[idx]) for idx in entity_indices])
                
                # Get component values for comparison
                component = np.array(c_pattern.get('component', []))
                if len(component) == 0:
                    continue
                
                # Extract the values for common entities
                component_weights = []
                for entity_idx in entity_indices:
                    if entity_idx < len(component):
                        component_weights.append(abs(component[entity_idx]))
                    else:
                        component_weights.append(0)
                
                if not component_weights:
                    continue
                
                component_weights = np.array(component_weights)
                
                # Calculate correlation
                if len(entity_weights) > 1 and len(component_weights) > 1:
                    try:
                        correlation_matrix = np.corrcoef(entity_weights, component_weights)
                        if correlation_matrix.shape == (2, 2):
                            correlation = correlation_matrix[0, 1]
                        else:
                            correlation = 0
                    except:
                        correlation = 0
                else:
                    # Simple product for single entity case
                    correlation = float(entity_weights[0] * component_weights[0])
                
                if abs(correlation) > highest_correlation:
                    highest_correlation = abs(correlation)
                    
                    # Extract temporal pattern from knowledge component
                    temporal_data = compressed_history.compressed_data[k_idx, :].copy()
                    
                    # Calculate trend using polynomial fit
                    x = np.arange(len(temporal_data))
                    trend_coeffs = np.polyfit(x, temporal_data, 1)
                    trend = float(trend_coeffs[0])
                    
                    best_match = {
                        'knowledge_component_idx': k_idx,
                        'correlation': float(correlation),
                        'explained_variance': float(compressed_history.eigenvalues[k_idx] / 
                                                  sum(compressed_history.eigenvalues)),
                        'temporal_trend': trend,
                        'shared_entities': [compressed_history.entities[idx] for idx in entity_indices]
                    }
            
            if best_match:
                mappings.append({
                    'contradiction_pattern_id': c_pattern.get('pattern_id'),
                    'knowledge_pattern': best_match,
                    'mapping_confidence': float(highest_correlation)
                })
        
        return mappings
