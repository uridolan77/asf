"""
Autopoietic Maintenance Engine - ASF Layer 2

This module implements the central engine that detects contradictions and transforms
them into opportunities for knowledge evolution through sophisticated pattern analysis
and resolution strategies.
"""

import numpy as np
from datetime import datetime
import uuid
from typing import Dict, List, Optional, Any, Tuple, Set, Union

from contradiction_detection import ContradictionDetector
from contradiction_pattern_analysis import PatternAnalyzer
from resolution_strategies import ResolutionStrategySelector
from bayesian_confidence import BayesianConfidenceUpdater
from multi_resolution_modeling import MultiResolutionModelManager

class AutopoieticMaintenanceEngine:
    """
    The Autopoietic Maintenance Engine transforms contradictions from errors
    into opportunities for knowledge evolution.
    
    This engine implements Layer 2 of the ASF framework, continuously monitoring
    knowledge for contradictions and applying sophisticated resolution strategies
    to maintain coherence while enabling evolution.
    """
    
    def __init__(self, knowledge_substrate):
        """
        Initialize the Autopoietic Maintenance Engine.
        
        Args:
            knowledge_substrate: Reference to the Knowledge Substrate Layer (Layer 1)
        """
        self.knowledge_substrate = knowledge_substrate
        
        # Initialize components
        self.contradiction_detector = ContradictionDetector()
        self.pattern_analyzer = PatternAnalyzer()
        self.strategy_selector = ResolutionStrategySelector()
        self.confidence_updater = BayesianConfidenceUpdater()
        self.resolution_model_manager = MultiResolutionModelManager()
        
        # Track historical contradictions for pattern analysis
        self.contradiction_history = {}
        self.resolution_history = {}
        
        # State tracking
        self.active_resolution_processes = {}
        self.periodic_checks_scheduled = False
        
        # Configuration
        self.detection_sensitivity = 0.7  # Default sensitivity
        self.min_confidence_threshold = 0.2  # Minimum confidence for consideration
        self.pattern_significance_threshold = 0.4  # Threshold for pattern significance
        self.automatic_resolution_threshold = 0.85  # Confidence needed for auto-resolution
        
    def monitor_knowledge_update(self, entity_id: str, update_data: Dict, 
                                source: Optional[str] = None) -> Dict:
        """
        Monitor knowledge updates for potential contradictions.
        
        This method is called by the Knowledge Substrate Layer when entities are updated,
        allowing the Maintenance Engine to detect and resolve contradictions.
        
        Args:
            entity_id: Identifier of the entity being updated
            update_data: New data that will be applied to the entity
            source: Source of the update (for trust weighting)
            
        Returns:
            Dictionary with contradiction analysis results
        """
        # Get current entity state
        try:
            current_entity = self.knowledge_substrate.get_entity(entity_id)
        except ValueError:
            # New entity, no contradiction possible
            return {"contradiction_detected": False, "entity_id": entity_id}
        
        # Detect contradictions between current state and update
        contradictions = self.contradiction_detector.detect_contradictions(
            current_entity, update_data)
        
        if not contradictions:
            return {"contradiction_detected": False, "entity_id": entity_id}
            
        # Log contradictions for pattern analysis
        self._log_contradiction(entity_id, contradictions, current_entity, update_data, source)
        
        # Determine domain for domain-specific patterns
        domain = current_entity.get('domain')
        
        # Check if this matches known patterns
        pattern_match = self.pattern_analyzer.match_to_known_patterns(
            contradictions, domain, self.contradiction_history.get(domain, []))
        
        if pattern_match and pattern_match['confidence'] > self.automatic_resolution_threshold:
            # Apply pre-determined resolution strategy for known pattern
            resolution_result = self.strategy_selector.apply_pattern_resolution(
                pattern_match, current_entity, update_data, contradictions)
            
            # Record resolution in history
            self._log_resolution(entity_id, pattern_match['pattern_id'], resolution_result)
            
            return {
                "contradiction_detected": True,
                "entity_id": entity_id,
                "contradictions": contradictions,
                "pattern_matched": pattern_match['pattern_id'],
                "resolution": resolution_result,
                "automatic_resolution": True
            }
            
        # For unmatched or low-confidence matches, start a resolution process
        resolution_process_id = self._start_resolution_process(
            entity_id, contradictions, current_entity, update_data, source)
            
        return {
            "contradiction_detected": True,
            "entity_id": entity_id,
            "contradictions": contradictions,
            "resolution_process_id": resolution_process_id,
            "automatic_resolution": False
        }
    
    def analyze_contradiction_patterns(self, domain: str) -> Dict:
        """
        Apply eigenvalue-based pattern detection to identify underlying patterns
        in domain contradictions.
        
        Args:
            domain: Knowledge domain to analyze
            
        Returns:
            Dictionary of detected contradiction patterns
        """
        # Get domain-specific contradiction history
        domain_contradictions = self.contradiction_history.get(domain, [])
        
        if len(domain_contradictions) < 5:
            return {
                "domain": domain,
                "pattern_detection_status": "insufficient_data",
                "patterns": []
            }
            
        # Construct contradiction similarity matrix for eigenvalue decomposition
        contradiction_matrix = self.pattern_analyzer.construct_contradiction_matrix(
            domain_contradictions)
            
        # Apply eigenvalue decomposition
        significant_patterns = self.pattern_analyzer.identify_contradiction_patterns(
            contradiction_matrix, domain_contradictions, self.pattern_significance_threshold)
            
        # Get compressed history from Knowledge Substrate if available
        try:
            compressed_history = self.knowledge_substrate.compress_domain_history(domain)
            
            # Map contradiction patterns to knowledge evolution patterns
            if compressed_history:
                knowledge_pattern_mappings = self.pattern_analyzer.map_contradiction_to_knowledge_patterns(
                    significant_patterns, compressed_history)
                    
                # Enhance resolution strategies based on knowledge patterns
                for i, pattern in enumerate(significant_patterns):
                    if i < len(knowledge_pattern_mappings) and knowledge_pattern_mappings[i]:
                        mapping = knowledge_pattern_mappings[i]
                        pattern['knowledge_pattern_mapping'] = mapping
                        
                        # Update resolution strategy based on knowledge pattern
                        if mapping['mapping_confidence'] > 0.6:
                            pattern['resolution_strategy'] = self.strategy_selector.derive_strategy_from_knowledge_pattern(
                                pattern['resolution_strategy'], mapping, compressed_history)
        except Exception as e:
            # No compressed history available, continue without mapping
            print(f"Unable to map to knowledge patterns: {str(e)}")
            
        # Save the patterns for future contradiction matching
        self.pattern_analyzer.update_known_patterns(domain, significant_patterns)
        
        return {
            "domain": domain,
            "pattern_detection_status": "success",
            "patterns": significant_patterns,
            "total_contradictions_analyzed": len(domain_contradictions),
            "eigenvalues": [p.get('eigenvalue', 0) for p in significant_patterns]
        }
    
    def resolve_contradiction(self, 
                            resolution_process_id: str, 
                            selected_strategy: Optional[str] = None) -> Dict:
        """
        Resolve a contradiction using either a specified strategy or the optimal
        strategy determined by the engine.
        
        Args:
            resolution_process_id: ID of the resolution process
            selected_strategy: Optional strategy name to use (if None, optimal is selected)
            
        Returns:
            Resolution results
        """
        if resolution_process_id not in self.active_resolution_processes:
            raise ValueError(f"Resolution process {resolution_process_id} not found")
            
        resolution_data = self.active_resolution_processes[resolution_process_id]
        entity_id = resolution_data['entity_id']
        contradictions = resolution_data['contradictions']
        current_entity = resolution_data['current_entity']
        update_data = resolution_data['update_data']
        
        # Get domain for domain-specific handling
        domain = current_entity.get('domain')
        
        # If strategy not specified, determine optimal strategy using eigenpatterns if available
        if not selected_strategy:
            # Check if we have pattern matches from compressed knowledge
            if domain in self.contradiction_history and self.knowledge_substrate.compressed_histories.get(domain) is not None:
                try:
                    # Use eigenvalue-based pattern detection to find optimal strategy
                    strategy_info = self.resolve_contradiction_with_eigenpatterns(
                        domain, contradictions, current_entity, update_data)
                    selected_strategy = strategy_info['strategy_name']
                except Exception as e:
                    print(f"Error in eigenpattern resolution: {str(e)}")
                    # Fall back to standard selection if error occurs
                    strategy_info = self.strategy_selector.select_optimal_strategy(
                        contradictions, current_entity, update_data, domain)
                    selected_strategy = strategy_info['strategy_name']
            else:
                # Use standard strategy selection
                strategy_info = self.strategy_selector.select_optimal_strategy(
                    contradictions, current_entity, update_data, domain)
                selected_strategy = strategy_info['strategy_name']
                
        # Apply the selected strategy
        resolution_result = self.strategy_selector.apply_strategy(
            selected_strategy, current_entity, update_data, contradictions)
            
        # Update confidence scores based on resolution
        confidence_updates = self.confidence_updater.update_confidence_after_resolution(
            resolution_result, current_entity, domain)
            
        # Record resolution in history
        self._log_resolution(entity_id, selected_strategy, resolution_result)
        
        # Clean up resolution process
        del self.active_resolution_processes[resolution_process_id]
        
        return {
            "entity_id": entity_id,
            "resolution_strategy": selected_strategy,
            "resolution_result": resolution_result,
            "confidence_updates": confidence_updates,
            "status": "completed"
        }
    
    def resolve_contradiction_with_eigenpatterns(self, 
                                               domain: str,
                                               contradictions: List[Dict],
                                               current_entity: Dict,
                                               update_data: Dict) -> Dict:
        """
        Resolve contradictions using eigenpattern analysis from Knowledge Substrate.
        
        Args:
            domain: Knowledge domain
            contradictions: List of contradiction dictionaries
            current_entity: Current entity state
            update_data: New data causing contradiction
            
        Returns:
            Resolution strategy information
        """
        # Get historical contradictions from this domain
        domain_contradictions = self.contradiction_history.get(domain, [])
        
        # Create temporary contradiction record for current case
        current_contradiction = {
            'entity_id': current_entity['entity_id'],
            'timestamp': datetime.now().isoformat(),
            'contradictions': contradictions,
            'current_data': current_entity,
            'update_data': update_data
        }
        
        # Add to list for pattern analysis
        all_contradictions = domain_contradictions + [current_contradiction]
        
        # Identify patterns in contradictions
        contradiction_matrix = self.pattern_analyzer.construct_contradiction_matrix(all_contradictions)
        patterns = self.pattern_analyzer.identify_contradiction_patterns(
            contradiction_matrix, all_contradictions, self.pattern_significance_threshold)
        
        # Get compressed history from Knowledge Substrate
        compressed_history = self.knowledge_substrate.compressed_histories.get(domain)
        
        if compressed_history and patterns:
            # Find which compressed knowledge patterns relate to these contradiction patterns
            related_knowledge_patterns = self.pattern_analyzer.map_contradiction_to_knowledge_patterns(
                patterns, compressed_history)
            
            # Use eigenvectors from Knowledge Substrate to guide resolution
            resolution = self.strategy_selector.resolve_using_knowledge_patterns(
                current_entity,
                contradictions,
                related_knowledge_patterns,
                compressed_history
            )
            
            return resolution
        else:
            # Fall back to standard resolution if no patterns or compressed history
            return self.strategy_selector.select_optimal_strategy(
                contradictions, current_entity, update_data, domain)
    
    def perform_maintenance_check(self, domain: Optional[str] = None) -> Dict:
        """
        Perform a proactive maintenance check to identify and address potential 
        contradictions before they cause problems.
        
        Args:
            domain: Optional domain to limit the check (if None, all domains are checked)
            
        Returns:
            Maintenance check results
        """
        check_results = {
            "timestamp": datetime.now().isoformat(),
            "domains_checked": [],
            "patterns_detected": 0,
            "contradictions_identified": 0,
            "automatic_resolutions": 0,
            "actions": []
        }
        
        # Get domains to check
        domains = [domain] if domain else list(self.knowledge_substrate.domains.keys())
        check_results["domains_checked"] = domains
        
        for domain_name in domains:
            # Analyze for patterns in this domain
            pattern_results = self.analyze_contradiction_patterns(domain_name)
            
            if pattern_results["pattern_detection_status"] == "success":
                check_results["patterns_detected"] += len(pattern_results["patterns"])
                
                # For each significant pattern, check if it indicates a systemic issue
                for pattern in pattern_results["patterns"]:
                    if pattern.get("explained_variance", 0) > 0.3:  # Significant pattern
                        # Check if this pattern suggests proactive resolution
                        if self._should_proactively_resolve(pattern):
                            # Get affected entities
                            affected_entities = self._identify_affected_entities(pattern, domain_name)
                            
                            # Apply resolution to affected entities
                            for entity_id in affected_entities:
                                resolution_result = self._apply_proactive_resolution(
                                    entity_id, pattern, domain_name)
                                
                                if resolution_result.get("applied", False):
                                    check_results["automatic_resolutions"] += 1
                                    check_results["actions"].append({
                                        "type": "proactive_resolution",
                                        "entity_id": entity_id,
                                        "pattern_id": pattern.get("pattern_id", "unknown"),
                                        "resolution_strategy": resolution_result.get("strategy")
                                    })
        
        return check_results
    
    def _log_contradiction(self, entity_id: str, contradictions: List[Dict],
                          current_entity: Dict, update_data: Dict, source: Optional[str]) -> None:
        """Log a contradiction in history for pattern analysis."""
        domain = current_entity.get('domain', 'unknown')
        
        if domain not in self.contradiction_history:
            self.contradiction_history[domain] = []
            
        self.contradiction_history[domain].append({
            'entity_id': entity_id,
            'timestamp': datetime.now().isoformat(),
            'contradictions': contradictions,
            'current_data': current_entity,
            'update_data': update_data,
            'source': source
        })
        
        # Limit history size to prevent unbounded growth
        max_history = 1000
        if len(self.contradiction_history[domain]) > max_history:
            self.contradiction_history[domain] = self.contradiction_history[domain][-max_history:]
    
    def _log_resolution(self, entity_id: str, strategy: str, result: Dict) -> None:
        """Log a resolution in history for effectiveness tracking."""
        domain = result.get('domain', 'unknown')
        
        if domain not in self.resolution_history:
            self.resolution_history[domain] = []
            
        self.resolution_history[domain].append({
            'entity_id': entity_id,
            'timestamp': datetime.now().isoformat(),
            'strategy': strategy,
            'result': result
        })
        
        # Limit history size to prevent unbounded growth
        max_history = 1000
        if len(self.resolution_history[domain]) > max_history:
            self.resolution_history[domain] = self.resolution_history[domain][-max_history:]
    
    def _start_resolution_process(self, entity_id: str, contradictions: List[Dict],
                                current_entity: Dict, update_data: Dict, source: Optional[str]) -> str:
        """Start a new resolution process for manual or deferred resolution."""
        process_id = f"resolution_{entity_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        self.active_resolution_processes[process_id] = {
            'entity_id': entity_id,
            'contradictions': contradictions,
            'current_entity': current_entity,
            'update_data': update_data,
            'source': source,
            'status': 'pending',
            'created': datetime.now().isoformat()
        }
        
        return process_id
    
    def _should_proactively_resolve(self, pattern: Dict) -> bool:
        """Determine if a pattern warrants proactive resolution."""
        # Patterns with high eigenvalues and clear resolution strategies
        if (pattern.get("eigenvalue", 0) > 0.7 and 
            pattern.get("resolution_strategy", {}).get("confidence", 0) > 0.8):
            return True
            
        # Patterns mapped to knowledge evolution with high confidence
        if (pattern.get("knowledge_pattern_mapping", {}).get("mapping_confidence", 0) > 0.8 and
            pattern.get("eigenvalue", 0) > 0.5):
            return True
            
        return False
    
    def _identify_affected_entities(self, pattern: Dict, domain: str) -> List[str]:
        """Identify entities affected by a contradiction pattern."""
        # Extract entities directly involved in pattern
        pattern_entities = []
        for contradiction in pattern.get("top_contradictions", []):
            entity_id = contradiction.get("entity_id")
            if entity_id:
                pattern_entities.append(entity_id)
        
        # For patterns with knowledge mappings, use eigenvectors to find similar entities
        if "knowledge_pattern_mapping" in pattern and pattern["knowledge_pattern_mapping"]:
            mapping = pattern["knowledge_pattern_mapping"]
            
            if mapping.get("knowledge_component_idx") is not None:
                try:
                    # Get entities with similar characteristics from the knowledge substrate
                    compressed_history = self.knowledge_substrate.compressed_histories.get(domain)
                    if compressed_history:
                        k_idx = mapping.get("knowledge_component_idx")
                        component = compressed_history.eigenvectors[:, k_idx]
                        
                        # Find entities with significant weights on this component
                        similar_entities = []
                        for i, weight in enumerate(component):
                            if abs(weight) > 0.1 and i < len(compressed_history.entities):
                                similar_entities.append(compressed_history.entities[i])
                        
                        # Combine with direct pattern entities
                        return list(set(pattern_entities + similar_entities))
                except Exception as e:
                    print(f"Error finding similar entities: {str(e)}")
        
        return pattern_entities
    
    def _apply_proactive_resolution(self, entity_id: str, pattern: Dict, domain: str) -> Dict:
        """Apply proactive resolution to an entity based on a pattern."""
        try:
            # Get entity state
            entity = self.knowledge_substrate.get_entity(entity_id)
            
            # Check if entity still exists and belongs to correct domain
            if not entity or entity.get('domain') != domain:
                return {"applied": False, "reason": "entity_not_found"}
            
            # Extract resolution strategy from pattern
            strategy_info = pattern.get("resolution_strategy", {})
            strategy_name = strategy_info.get("strategy_type")
            if not strategy_name or strategy_name not in self.strategy_selector.strategies:
                return {"applied": False, "reason": "no_strategy"}
            
            # Apply strategy with minimal transformation
            resolution_result = self.strategy_selector.apply_preventative_strategy(
                strategy_name, entity, pattern)
            
            if resolution_result.get("changes_made", False):
                # Apply updates to entity
                self.knowledge_substrate.update_entity(
                    entity_id,
                    resolution_result.get("updated_data", {}),
                    resolution_result.get("confidence_modifier", 0),
                    source="maintenance_engine"
                )
                
                return {
                    "applied": True,
                    "strategy": strategy_name,
                    "entity_id": entity_id,
                    "confidence_modifier": resolution_result.get("confidence_modifier", 0)
                }
            
            return {"applied": False, "reason": "no_changes_needed"}
        except Exception as e:
            return {"applied": False, "reason": f"error: {str(e)}"}
