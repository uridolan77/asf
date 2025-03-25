import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Set, Union
import uuid
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class PatternAnalyzer:
    """
    Implements eigenvalue-based pattern detection for contradiction analysis,
    with improvements for dynamic pattern learning and strategy selection.

    This class analyzes contradictions across a domain to identify underlying patterns,
    mapping them to knowledge evolution patterns from the Knowledge Substrate Layer.
    It uses clustering to improve pattern identification and a more data-driven
    approach to resolution strategy selection.
    """

    def __init__(self):
        """Initialize the pattern analyzer."""
        # Store known patterns by domain
        self.known_patterns = {}

        # Configuration (adjustable)
        self.similarity_threshold = 0.6  # Threshold for pattern similarity
        self.entity_match_weight = 0.4  # Weight for entity matching
        self.attribute_match_weight = 0.3  # Weight for attribute matching
        self.type_match_weight = 0.3  # Weight for contradiction type
        self.min_pattern_size = 3  # Minimum contradictions for a pattern
        self.max_patterns = 10      # Maximum number of patterns to retain
        self.dynamic_resolution_rules = {} #For data driven resolution determination

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

        # Populate matrix based on similarity/relationship
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
        Uses clustering and eigenvalue decomposition to identify patterns.

        Args:
            contradiction_matrix: Matrix of contradiction similarities.
            domain_contradictions: List of contradictions.
            significance_threshold: Threshold for pattern significance.

        Returns:
            List of significant patterns with resolution strategies.
        """
        if len(domain_contradictions) < self.min_pattern_size:
            return []

        # --- 1. Clustering ---
        patterns = self._cluster_contradictions(contradiction_matrix, domain_contradictions)

        # --- 2. Eigenvalue Analysis (within each cluster) ---
        significant_patterns = []
        for pattern in patterns:
            cluster_indices = pattern['contradiction_indices']
            if len(cluster_indices) < self.min_pattern_size:
                continue  # Skip small clusters

            # Submatrix for the cluster
            submatrix = contradiction_matrix[np.ix_(cluster_indices, cluster_indices)]
            cluster_contradictions = [domain_contradictions[i] for i in cluster_indices]

            try:
                eigenvalues, eigenvectors = np.linalg.eigh(submatrix)
                idx = np.argsort(eigenvalues)[::-1]
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]


                # --- 3. Pattern Characterization & Resolution ---
                # Consider only the top eigenvector (principal component)
                if eigenvalues[0] > significance_threshold:
                   component = eigenvectors[:, 0]

                   # Top contradictions in *this* cluster
                   top_indices = np.argsort(np.abs(component))[-5:][::-1]
                   # Map back to the *global* indices of the contradictions.
                   global_top_indices = [cluster_indices[i] for i in top_indices]
                   pattern_contradictions = [domain_contradictions[i] for i in global_top_indices if i < len(domain_contradictions)]

                   pattern_id = f"pattern_{uuid.uuid4().hex[:8]}"
                   pattern_attributes = self._extract_pattern_attributes(pattern_contradictions)
                   resolution_strategy = self._determine_pattern_resolution(component, pattern_contradictions, pattern_attributes)

                   significant_patterns.append({
                       'pattern_id': pattern_id,
                       'eigenvalue': float(eigenvalues[0]),
                       'explained_variance': float(eigenvalues[0] / sum(eigenvalues)),
                       'top_contradictions': pattern_contradictions,
                       'pattern_attributes': pattern_attributes,
                       'resolution_strategy': resolution_strategy,
                       'component': component.tolist(),  # For the *subspace*
                       'contradiction_indices': cluster_indices #Indices into the main list.
                   })

            except Exception as e:
                print(f"Error in eigenvalue decomposition: {str(e)}")
                continue

        return significant_patterns

    def _cluster_contradictions(self, contradiction_matrix: np.ndarray, domain_contradictions: List[Dict]) -> List[Dict]:
      """Clusters contradictions using K-Means."""
      if len(domain_contradictions) < 2: #Need at least 2 for K-Means
        if len(domain_contradictions) > 0:
          return [{'pattern_id': f"pattern_{uuid.uuid4().hex[:8]}",
                   'contradiction_indices': [0]}]
        else:
          return [] # No data.

      # Determine optimal k (number of clusters) using Silhouette Score
      best_k = 2
      best_score = -1

      #Limit Range: Don't try clustering if you have a tiny number of contradictions.
      max_k = min(len(domain_contradictions) -1, 5)
      if max_k < 2:
        max_k = 2

      for k in range(2, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10) #Explicitly set n_init.
        cluster_labels = kmeans.fit_predict(contradiction_matrix)
        score = silhouette_score(contradiction_matrix, cluster_labels)
        if score > best_score:
          best_score = score
          best_k = k

      #Final Clustering:
      kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
      cluster_labels = kmeans.fit_predict(contradiction_matrix)

      #Organize patterns:
      patterns = []
      for k in range(best_k):
        cluster_indices = [i for i, label in enumerate(cluster_labels) if label == k]
        pattern_id = f"pattern_{uuid.uuid4().hex[:8]}"
        patterns.append({
            'pattern_id': pattern_id,
            'contradiction_indices': cluster_indices
        })

      return patterns

    def calculate_contradiction_similarity(self,
                                            contradiction1: Dict,
                                            contradiction2: Dict) -> float:
        """
        Calculate similarity between two contradictions (same as before).
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

          # Convert to similarity score (1.0 for same time, decaying)
          return np.exp(-time_diff / 24)  # Exponential decay
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
      """Determine the appropriate resolution strategy based on the pattern."""

      # --- 1. Check Dynamic Rules ---
      strategy_type = self._check_dynamic_resolution_rules(pattern_attributes)
      if strategy_type:
          return {
            'strategy_type': strategy_type,
            'confidence': 0.9,  # High confidence for learned rules
            'justification': f"Dynamic rule matched for pattern attributes."
          }

      # --- 2. Fallback to Heuristics (simplified) ---

      dominant_type = pattern_attributes.get('dominant_type')

      if dominant_type == 'attribute_value':
          strategy_type = "contextual_boundary_refinement"
      elif dominant_type == 'relationship':
          strategy_type = "perspective_integration"
      elif dominant_type == 'temporal_pattern':
          strategy_type = "temporal_evolution_tracking"
      else:
          strategy_type = "graduated_confidence_adjustment"  # Default

      return {
          'strategy_type': strategy_type,
          'confidence': 0.7,  # Base confidence
          'justification': f"Heuristic rule for {dominant_type} pattern."
      }

    def _check_dynamic_resolution_rules(self, pattern_attributes:Dict) -> Optional[str]:
      """Checks if a dynamic resolution rule applies."""
      for rule_key, rule_details in self.dynamic_resolution_rules.items():
        if self._rule_matches(pattern_attributes, rule_details['conditions']):
          return rule_details['strategy']
      return None
    def _rule_matches(self, pattern_attributes: Dict, rule_conditions:Dict) -> bool:
      """Checks if pattern attributes satisfy rule conditions"""
      for condition_key, condition_value in rule_conditions.items():
        if condition_key not in pattern_attributes:
          return False #Missing condition

        if isinstance(condition_value, list):
          #Handle list based conditions (e.g. "affected_entities")
          if not isinstance(pattern_attributes[condition_key], list):
            return False
          if not set(condition_value).issubset(set(pattern_attributes[condition_key])):
            return False

        elif pattern_attributes[condition_key] != condition_value:
          return False #Value mismatch.

      return True

    def add_dynamic_resolution_rule(self, rule_name:str, conditions: Dict, strategy: str):
      """Adds or Updates a dynamic resolution rule."""
      self.dynamic_resolution_rules[rule_name] = {
        'conditions': conditions,
        'strategy': strategy
      }

    def remove_dynamic_resolution_rule(self, rule_name: str):
      """Removes a dynamic resolution rule"""
      if rule_name in self.dynamic_resolution_rules:
        del self.dynamic_resolution_rules[rule_name]

    def update_known_patterns(self, domain: str, new_patterns: List[Dict]) -> None:
        """
        Update the catalog of known patterns for a domain (simplified merging).
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
                      # Update existing pattern with new information (Simplified Merging).
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


        # --- Pattern Pruning (Keep only top N patterns) ---
        self.known_patterns[domain] = sorted(
            self.known_patterns[domain],
            key=lambda x: x.get('eigenvalue', 0.0),  # Sort by significance
            reverse=True
        )[:self.max_patterns]


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
        """Merge a new pattern into an existing one, updating fields."""
        # Update last seen
        existing_pattern['last_seen'] = datetime.now().isoformat()

        # Increment observation count
        existing_pattern['observation_count'] = existing_pattern.get('observation_count', 1) + 1

        # Update confidence (simple average)
        existing_resolution = existing_pattern.get('resolution_strategy', {})
        new_resolution = new_pattern.get('resolution_strategy', {})
        existing_resolution['confidence'] = (
            existing_resolution.get('confidence', 0.7) +
            new_resolution.get('confidence', 0.7)
        ) / 2

        # Update affected entities (union)
        existing_attrs = existing_pattern.get('pattern_attributes', {})
        new_attrs = new_pattern.get('pattern_attributes', {})

        existing_entities = set(existing_attrs.get('affected_entities', []))
        new_entities = set(new_attrs.get('affected_entities', []))
        existing_attrs['affected_entities'] = list(existing_entities.union(new_entities))

        #Update Contradiction Indices:
        existing_indices = set(existing_pattern.get('contradiction_indices', []))
        new_indices = set(new_pattern.get('contradiction_indices', []))
        existing_pattern['contradiction_indices'] = list(existing_indices.union(new_indices))


    def match_to_known_patterns(self, contradictions: List[Dict],
                                domain: str,
                                recent_history: List[Dict] = None) -> Optional[Dict]:
        """
        Match a set of contradictions to known patterns.
        """
        if domain not in self.known_patterns or not self.known_patterns[domain]:
            return None

        # Create a temporary pattern from the contradictions
        temp_pattern = {
            'contradictions': contradictions,
            'pattern_attributes': self._extract_pattern_attributes([
                {'contradictions': contradictions}
            ]),
            #Include indices for matching.
            'contradiction_indices': list(range(len(contradictions)))
        }
        # Find best matching pattern
        best_match = None
        best_similarity = 0.0

        for pattern in self.known_patterns[domain]:
            similarity = self._calculate_pattern_similarity(temp_pattern, pattern)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = pattern

        # If similarity exceeds threshold, return
        if best_similarity > self.similarity_threshold:
            return {
                'pattern_id': best_match['pattern_id'],
                'similarity': best_similarity,
                'confidence': best_similarity,  # Confidence = similarity
                'pattern': best_match
            }

        return None

    def map_contradiction_to_knowledge_patterns(self,
                                                contradiction_patterns: List[Dict],
                                                compressed_history) -> List[Dict]:
        """
        Map contradiction patterns to knowledge evolution patterns.
        (Same as before, but using 'contradiction_indices' for mapping)
        """
        mappings = []

        # For each contradiction pattern
        for c_pattern in contradiction_patterns:
            best_match = None
            highest_correlation = 0

            # Compare against knowledge evolution patterns
            k_shape = compressed_history.eigenvectors.shape
            for k_idx in range(min(k_shape[1], 5)):  # Check top components
                knowledge_pattern = compressed_history.eigenvectors[:, k_idx]

                # --- Entity Mapping (using contradiction_indices) ---
                contradiction_indices = c_pattern.get('contradiction_indices', [])
                if not contradiction_indices:
                    continue

                # Map contradiction indices to entity indices
                entity_indices = []
                for c_idx in contradiction_indices:
                    # Assuming 'entity_id' is in the original contradiction data
                    if c_idx < len(compressed_history.entities): #Bounds Check
                        entity_id = compressed_history.entities[c_idx] #This now works.
                        try:
                            e_idx = compressed_history.entities.index(entity_id)
                            entity_indices.append(e_idx)
                        except ValueError:
                            continue
                if not entity_indices:
                    continue

                # Calculate correlation (using mapped entity indices)
                entity_weights = np.array([abs(knowledge_pattern[idx]) for idx in entity_indices])

                component = np.array(c_pattern.get('component', []))
                if len(component) == 0 :
                  continue

                component_weights = []

                #Use the indices from the *sub-space clustering*
                subspace_indices = c_pattern.get('contradiction_indices')
                for sub_idx in subspace_indices:
                  if sub_idx < len(component):
                    component_weights.append(abs(component[sub_idx]))
                  else:
                    component_weights.append(0)
                if not component_weights:
                  continue
                component_weights = np.array(component_weights)


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
                    # Simple product for the single-entity case
                    correlation = float(entity_weights[0] * component_weights[0])

                if abs(correlation) > highest_correlation:
                    highest_correlation = abs(correlation)
                    # Extract temporal pattern
                    temporal_data = compressed_history.compressed_data[k_idx, :].copy()
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