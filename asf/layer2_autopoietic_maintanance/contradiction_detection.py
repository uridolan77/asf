"""
Contradiction Detection Module for ASF Maintenance Engine

This module implements mechanisms for detecting different types of contradictions
in knowledge entities.
"""

from typing import Dict, List, Any, Optional, Set, Union
import numpy as np
from datetime import datetime

class ContradictionDetector:
    """
    Detects contradictions between knowledge states by implementing multiple
    detection strategies for different contradiction types.
    """
    
    def __init__(self):
        """Initialize the contradiction detector."""
        # Configuration for different contradiction types
        self.detection_thresholds = {
            "numeric_divergence": 0.2,  # Relative change threshold for numeric values
            "text_replacement": 0.7,    # String similarity threshold for text replacement
            "list_divergence": 0.5,     # List difference threshold
            "relationship_contradiction": 0.8,  # Relationship contradiction threshold
            "temporal_pattern": 0.6     # Temporal pattern contradiction threshold
        }
    
    def detect_contradictions(self, current_entity: Dict, update_data: Dict) -> List[Dict]:
        """
        Detect contradictions between current entity state and update data.
        
        Args:
            current_entity: Current entity state
            update_data: New data to be applied
            
        Returns:
            List of detected contradictions
        """
        contradictions = []
        
        # Check for attribute value contradictions
        attribute_contradictions = self._detect_attribute_contradictions(
            current_entity.get('data', {}), 
            update_data
        )
        contradictions.extend(attribute_contradictions)
        
        # Check for relationship contradictions if relationships exist
        if 'relationships' in current_entity:
            relationship_contradictions = self._detect_relationship_contradictions(
                current_entity.get('relationships', {}),
                update_data.get('relationships', {})
            )
            contradictions.extend(relationship_contradictions)
        
        # Check for temporal pattern contradictions
        if 'temporal_data' in current_entity:
            temporal_contradictions = self._detect_temporal_pattern_contradictions(
                current_entity.get('temporal_data', {}),
                update_data.get('temporal_data', {})
            )
            contradictions.extend(temporal_contradictions)
        
        # Check for expectation violations
        expectation_contradictions = self._detect_expectation_violations(
            current_entity, update_data
        )
        contradictions.extend(expectation_contradictions)
        
        return contradictions
    
    def _detect_attribute_contradictions(self, current_data: Dict, update_data: Dict, 
                                       prefix: str = "") -> List[Dict]:
        """
        Detect contradictions between attribute values.
        
        Args:
            current_data: Current attribute values
            update_data: New attribute values
            prefix: Prefix for nested attribute paths
            
        Returns:
            List of attribute contradictions
        """
        contradictions = []
        
        for key, new_value in update_data.items():
            # Build the full attribute path
            attr_path = f"{prefix}.{key}" if prefix else key
            
            # Skip metadata fields
            if key.startswith('_'):
                continue
                
            # Check if attribute exists in current data
            if key in current_data:
                current_value = current_data[key]
                
                # Different types might indicate contradiction
                if type(new_value) != type(current_value):
                    contradictions.append({
                        'type': 'attribute_value',
                        'subtype': 'type_mismatch',
                        'attribute': attr_path,
                        'current_value': current_value,
                        'new_value': new_value,
                        'severity': 0.8  # Type mismatches are severe contradictions
                    })
                    continue
                    
                # For numeric values, check for significant difference
                if isinstance(new_value, (int, float)) and isinstance(current_value, (int, float)):
                    # Calculate relative change
                    if current_value != 0:
                        relative_change = abs((new_value - current_value) / current_value)
                    else:
                        relative_change = 1.0 if new_value != 0 else 0.0
                        
                    if relative_change > self.detection_thresholds["numeric_divergence"]:
                        contradictions.append({
                            'type': 'attribute_value',
                            'subtype': 'numeric_divergence',
                            'attribute': attr_path,
                            'current_value': current_value,
                            'new_value': new_value,
                            'relative_change': relative_change,
                            'severity': min(relative_change, 1.0)  # Severity based on change magnitude
                        })
                
                # For strings, check for replacement (not just expansion)
                elif isinstance(new_value, str) and isinstance(current_value, str):
                    # Calculate string similarity
                    similarity = self._calculate_string_similarity(current_value, new_value)
                    
                    if similarity < self.detection_thresholds["text_replacement"]:
                        # Text has changed significantly
                        contradictions.append({
                            'type': 'attribute_value',
                            'subtype': 'text_replacement',
                            'attribute': attr_path,
                            'current_value': current_value,
                            'new_value': new_value,
                            'similarity': similarity,
                            'severity': 1.0 - similarity  # Severity based on dissimilarity
                        })
                
                # For lists, check for significant changes
                elif isinstance(new_value, list) and isinstance(current_value, list):
                    # Calculate list difference
                    difference = self._calculate_list_difference(current_value, new_value)
                    
                    if difference > self.detection_thresholds["list_divergence"]:
                        contradictions.append({
                            'type': 'attribute_value',
                            'subtype': 'list_divergence',
                            'attribute': attr_path,
                            'current_value': current_value,
                            'new_value': new_value,
                            'difference': difference,
                            'severity': difference  # Severity based on difference magnitude
                        })
                
                # For dictionaries, recurse to check nested attributes
                elif isinstance(new_value, dict) and isinstance(current_value, dict):
                    nested_contradictions = self._detect_attribute_contradictions(
                        current_value, new_value, attr_path)
                    contradictions.extend(nested_contradictions)
        
        return contradictions
    
    def _detect_relationship_contradictions(self, current_relationships: Dict, 
                                          update_relationships: Dict) -> List[Dict]:
        """
        Detect contradictions in entity relationships.
        
        Args:
            current_relationships: Current relationship data
            update_relationships: New relationship data
            
        Returns:
            List of relationship contradictions
        """
        contradictions = []
        
        for rel_type, new_relations in update_relationships.items():
            if rel_type in current_relationships:
                current_relations = current_relationships[rel_type]
                
                # Check for contradictory relationship targets
                for target_id, new_props in new_relations.items():
                    if target_id in current_relations:
                        current_props = current_relations[target_id]
                        
                        # Check for property contradictions
                        for prop, new_value in new_props.items():
                            if prop in current_props:
                                current_value = current_props[prop]
                                
                                # Skip metadata fields
                                if prop.startswith('_'):
                                    continue
                                
                                # Different values might indicate contradiction
                                if new_value != current_value:
                                    # For relationship strengths, check if change is significant
                                    if prop == 'strength' and isinstance(new_value, (int, float)) and \
                                       isinstance(current_value, (int, float)):
                                        change = abs(new_value - current_value)
                                        if change > 0.3:  # Significant strength change
                                            contradictions.append({
                                                'type': 'relationship',
                                                'subtype': 'strength_change',
                                                'relation_type': rel_type,
                                                'target_id': target_id,
                                                'current_strength': current_value,
                                                'new_strength': new_value,
                                                'change': change,
                                                'severity': change
                                            })
                                    else:
                                        # General property contradiction
                                        contradictions.append({
                                            'type': 'relationship',
                                            'subtype': 'property_contradiction',
                                            'relation_type': rel_type,
                                            'target_id': target_id,
                                            'property': prop,
                                            'current_value': current_value,
                                            'new_value': new_value,
                                            'severity': 0.7
                                        })
                
                # Check for relationship removals (may be contradictions)
                for target_id in current_relations:
                    if target_id not in new_relations:
                        # Check if this is a high-confidence relationship
                        if 'confidence' in current_relations[target_id] and \
                           current_relations[target_id]['confidence'] > 0.8:
                            contradictions.append({
                                'type': 'relationship',
                                'subtype': 'high_confidence_removal',
                                'relation_type': rel_type,
                                'target_id': target_id,
                                'current_confidence': current_relations[target_id].get('confidence', 0),
                                'severity': current_relations[target_id].get('confidence', 0) * 0.8
                            })
        
        return contradictions
    
    def _detect_temporal_pattern_contradictions(self, current_temporal: Dict, 
                                              update_temporal: Dict) -> List[Dict]:
        """
        Detect contradictions in temporal patterns.
        
        Args:
            current_temporal: Current temporal pattern data
            update_temporal: New temporal pattern data
            
        Returns:
            List of temporal pattern contradictions
        """
        contradictions = []
        
        # Check for trend reversals
        if 'trend' in current_temporal and 'trend' in update_temporal:
            current_trend = current_temporal['trend']
            new_trend = update_temporal['trend']
            
            # Check if trends have opposite signs (reversal)
            if (current_trend * new_trend < 0) and (abs(current_trend) > 0.1) and (abs(new_trend) > 0.1):
                contradictions.append({
                    'type': 'temporal_pattern',
                    'subtype': 'trend_reversal',
                    'current_trend': current_trend,
                    'new_trend': new_trend,
                    'severity': min(abs(current_trend) + abs(new_trend), 1.0)
                })
        
        # Check for seasonality contradictions
        if 'seasonality' in current_temporal and 'seasonality' in update_temporal:
            current_seasonality = current_temporal['seasonality']
            new_seasonality = update_temporal['seasonality']
            
            # If period changes significantly
            if 'period' in current_seasonality and 'period' in new_seasonality:
                current_period = current_seasonality['period']
                new_period = new_seasonality['period']
                
                # Calculate relative change in period
                if current_period != 0:
                    period_change = abs((new_period - current_period) / current_period)
                    
                    if period_change > 0.3:  # Significant period change
                        contradictions.append({
                            'type': 'temporal_pattern',
                            'subtype': 'seasonality_period_change',
                            'current_period': current_period,
                            'new_period': new_period,
                            'relative_change': period_change,
                            'severity': min(period_change, 1.0)
                        })
        
        # Check for volatility contradictions
        if 'volatility' in current_temporal and 'volatility' in update_temporal:
            current_volatility = current_temporal['volatility']
            new_volatility = update_temporal['volatility']
            
            # Calculate relative change in volatility
            if current_volatility != 0:
                volatility_change = (new_volatility - current_volatility) / current_volatility
                
                if abs(volatility_change) > 0.5:  # Significant volatility change
                    contradictions.append({
                        'type': 'temporal_pattern',
                        'subtype': 'volatility_change',
                        'current_volatility': current_volatility,
                        'new_volatility': new_volatility,
                        'relative_change': volatility_change,
                        'severity': min(abs(volatility_change), 1.0)
                    })
        
        return contradictions
    
    def _detect_expectation_violations(self, current_entity: Dict, 
                                     update_data: Dict) -> List[Dict]:
        """
        Detect violations of expected patterns based on entity type and domain.
        
        Args:
            current_entity: Current entity state
            update_data: New data to be applied
            
        Returns:
            List of expectation violations
        """
        contradictions = []
        
        # Get entity type and domain for domain-specific expectations
        entity_type = current_entity.get('type')
        domain = current_entity.get('domain')
        
        if not entity_type or not domain:
            return []
        
        # Check for domain-specific expectation violations
        if domain == 'medical':
            medical_contradictions = self._check_medical_expectations(
                entity_type, current_entity, update_data)
            contradictions.extend(medical_contradictions)
            
        elif domain == 'financial':
            financial_contradictions = self._check_financial_expectations(
                entity_type, current_entity, update_data)
            contradictions.extend(financial_contradictions)
            
        elif domain == 'legal':
            legal_contradictions = self._check_legal_expectations(
                entity_type, current_entity, update_data)
            contradictions.extend(legal_contradictions)
        
        return contradictions
    
    def _check_medical_expectations(self, entity_type: str, current_entity: Dict, 
                                  update_data: Dict) -> List[Dict]:
        """Check medical domain-specific expectations."""
        contradictions = []
        
        # Example: Check for treatment efficacy reversals
        if entity_type == 'treatment':
            current_efficacy = current_entity.get('data', {}).get('efficacy')
            new_efficacy = update_data.get('efficacy')
            
            if current_efficacy is not None and new_efficacy is not None:
                # Check if treatment went from effective to ineffective or vice versa
                if (current_efficacy > 0.7 and new_efficacy < 0.3) or \
                   (current_efficacy < 0.3 and new_efficacy > 0.7):
                    contradictions.append({
                        'type': 'expectation_violation',
                        'subtype': 'treatment_efficacy_reversal',
                        'current_efficacy': current_efficacy,
                        'new_efficacy': new_efficacy,
                        'severity': abs(current_efficacy - new_efficacy)
                    })
        
        # Example: Check for diagnosis criteria contradictions
        elif entity_type == 'diagnosis':
            current_criteria = current_entity.get('data', {}).get('diagnostic_criteria', [])
            new_criteria = update_data.get('diagnostic_criteria', [])
            
            if current_criteria and new_criteria:
                # Check for complete replacement of criteria
                common_criteria = set(current_criteria).intersection(set(new_criteria))
                if not common_criteria and len(current_criteria) > 2 and len(new_criteria) > 2:
                    contradictions.append({
                        'type': 'expectation_violation',
                        'subtype': 'diagnostic_criteria_replacement',
                        'current_criteria': current_criteria,
                        'new_criteria': new_criteria,
                        'severity': 0.9
                    })
        
        return contradictions
    
    def _check_financial_expectations(self, entity_type: str, current_entity: Dict, 
                                    update_data: Dict) -> List[Dict]:
        """Check financial domain-specific expectations."""
        contradictions = []
        
        # Example: Check for correlation sign reversals
        if entity_type == 'correlation':
            current_value = current_entity.get('data', {}).get('correlation_value')
            new_value = update_data.get('correlation_value')
            
            if current_value is not None and new_value is not None:
                # Check if correlation flipped from positive to negative or vice versa
                if (current_value > 0.3 and new_value < -0.3) or \
                   (current_value < -0.3 and new_value > 0.3):
                    contradictions.append({
                        'type': 'expectation_violation',
                        'subtype': 'correlation_sign_reversal',
                        'current_value': current_value,
                        'new_value': new_value,
                        'severity': abs(current_value - new_value) / 2  # Scale to 0-1
                    })
        
        # Example: Check for risk rating expectation violations
        elif entity_type == 'asset':
            current_risk = current_entity.get('data', {}).get('risk_rating')
            new_risk = update_data.get('risk_rating')
            
            if current_risk is not None and new_risk is not None:
                # Check for dramatic risk rating changes
                risk_change = abs(new_risk - current_risk)
                if risk_change > 2:  # More than 2 levels of change (e.g., from low to high)
                    contradictions.append({
                        'type': 'expectation_violation',
                        'subtype': 'risk_rating_jump',
                        'current_risk': current_risk,
                        'new_risk': new_risk,
                        'change': risk_change,
                        'severity': min(risk_change / 4, 1.0)  # Scale to 0-1
                    })
        
        return contradictions
    
    def _check_legal_expectations(self, entity_type: str, current_entity: Dict, 
                                update_data: Dict) -> List[Dict]:
        """Check legal domain-specific expectations."""
        contradictions = []
        
        # Example: Check for precedent status reversals
        if entity_type == 'precedent':
            current_status = current_entity.get('data', {}).get('status')
            new_status = update_data.get('status')
            
            if current_status and new_status:
                # Check for severe status changes (e.g., "upheld" to "overturned")
                severe_transitions = [
                    ('upheld', 'overturned'),
                    ('controlling', 'superseded'),
                    ('good_law', 'bad_law')
                ]
                
                for from_status, to_status in severe_transitions:
                    if current_status == from_status and new_status == to_status:
                        contradictions.append({
                            'type': 'expectation_violation',
                            'subtype': 'precedent_status_reversal',
                            'current_status': current_status,
                            'new_status': new_status,
                            'severity': 0.9
                        })
        
        # Example: Check for jurisdiction contradictions
        elif entity_type == 'statute':
            current_jurisdictions = current_entity.get('data', {}).get('jurisdictions', [])
            new_jurisdictions = update_data.get('jurisdictions', [])
            
            if current_jurisdictions and new_jurisdictions:
                # Check for complete replacement of jurisdictions
                if set(current_jurisdictions) and set(new_jurisdictions) and \
                   not set(current_jurisdictions).intersection(set(new_jurisdictions)):
                    contradictions.append({
                        'type': 'expectation_violation',
                        'subtype': 'jurisdiction_replacement',
                        'current_jurisdictions': current_jurisdictions,
                        'new_jurisdictions': new_jurisdictions,
                        'severity': 0.8
                    })
        
        return contradictions
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate similarity between two strings.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Similarity score between 0 and 1
        """
        # Simple implementation - can be replaced with more sophisticated algorithms
        if not str1 and not str2:
            return 1.0
        if not str1 or not str2:
            return 0.0
            
        # Check if one is a substring of the other
        if str1 in str2 or str2 in str1:
            # Calculate ratio of shorter to longer
            shorter = min(len(str1), len(str2))
            longer = max(len(str1), len(str2))
            return shorter / longer
        
        # Calculate Levenshtein distance (simplified implementation)
        # For a full implementation, you would use a library like python-Levenshtein
        m, n = len(str1), len(str2)
        if m == 0:
            return 0.0
        if n == 0:
            return 0.0
            
        # Initialize distance matrix
        matrix = [[0 for x in range(n + 1)] for y in range(m + 1)]
        
        for i in range(m + 1):
            matrix[i][0] = i
        for j in range(n + 1):
            matrix[0][j] = j
            
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i-1] == str2[j-1]:
                    matrix[i][j] = matrix[i-1][j-1]
                else:
                    matrix[i][j] = min(
                        matrix[i-1][j] + 1,      # deletion
                        matrix[i][j-1] + 1,      # insertion
                        matrix[i-1][j-1] + 1     # substitution
                    )
        
        distance = matrix[m][n]
        max_len = max(m, n)
        
        return 1.0 - (distance / max_len)
    
    def _calculate_list_difference(self, list1: List, list2: List) -> float:
        """
        Calculate difference between two lists.
        
        Args:
            list1: First list
            list2: Second list
            
        Returns:
            Difference score between 0 and 1
        """
        if not list1 and not list2:
            return 0.0
            
        # Convert to sets for easier comparison
        set1 = set(str(item) for item in list1)
        set2 = set(str(item) for item in list2)
        
        # Calculate Jaccard distance
        union_size = len(set1.union(set2))
        
        if union_size == 0:
            return 0.0
            
        intersection_size = len(set1.intersection(set2))
        return 1.0 - (intersection_size / union_size)
