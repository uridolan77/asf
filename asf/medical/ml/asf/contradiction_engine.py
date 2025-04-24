"""
Contradiction Assimilation Engine Module

This module implements the Contradiction Assimilation Engine component of the ASF framework,
which detects and processes contradictions between knowledge elements.
"""

import time
import re
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, Set


class ContradictionAssimilationEngine:
    """
    Detects and processes contradictions between knowledge elements.
    
    The Contradiction Assimilation Engine identifies conflicts between knowledge elements
    and applies resolution strategies to handle them constructively, treating contradictions
    as opportunities for knowledge refinement rather than errors.
    """
    
    def __init__(self, confidence_ecosystem=None):
        """
        Initialize the Contradiction Assimilation Engine.
        
        Args:
            confidence_ecosystem: Reference to the confidence ecosystem (optional)
        """
        self.confidence_ecosystem = confidence_ecosystem
        self.resolution_strategies: Dict[str, Callable] = {}  # Map of strategy_id -> resolution_function
        self.contradiction_history: List[Dict[str, Any]] = []  # List of processed contradictions
        self.contradiction_patterns: Dict[str, Dict[str, Any]] = {}  # Patterns for detecting contradictions
    
    def register_resolution_strategy(self, strategy_id: str, resolution_function: Callable) -> bool:
        """
        Register a resolution strategy.
        
        Args:
            strategy_id: Strategy ID
            resolution_function: Function to resolve contradictions
            
        Returns:
            Success flag
        """
        self.resolution_strategies[strategy_id] = resolution_function
        return True
    
    def register_contradiction_pattern(
        self, 
        pattern_id: str, 
        pattern_type: str, 
        pattern_data: Any,
        description: str = ""
    ) -> bool:
        """
        Register a pattern for detecting contradictions.
        
        Args:
            pattern_id: Pattern ID
            pattern_type: Type of pattern ('regex', 'semantic', 'logical', etc.)
            pattern_data: Pattern data (depends on pattern_type)
            description: Description of the pattern
            
        Returns:
            Success flag
        """
        self.contradiction_patterns[pattern_id] = {
            "type": pattern_type,
            "data": pattern_data,
            "description": description
        }
        return True
    
    def detect_contradiction(
        self, 
        knowledge_a: Any, 
        knowledge_b: Any, 
        threshold: float = 0.5,
        pattern_ids: Optional[List[str]] = None
    ) -> bool:
        """
        Detect if two knowledge elements contradict each other.
        
        Args:
            knowledge_a: First knowledge element
            knowledge_b: Second knowledge element
            threshold: Contradiction threshold
            pattern_ids: List of pattern IDs to use (optional)
            
        Returns:
            Whether a contradiction is detected
        """
        # Calculate contradiction score
        contradiction_score = self._compute_contradiction_score(knowledge_a, knowledge_b, pattern_ids)
        
        # Return whether contradiction score exceeds threshold
        return contradiction_score > threshold
    
    def _compute_contradiction_score(
        self, 
        knowledge_a: Any, 
        knowledge_b: Any,
        pattern_ids: Optional[List[str]] = None
    ) -> float:
        """
        Compute a contradiction score between two knowledge elements.
        
        Args:
            knowledge_a: First knowledge element
            knowledge_b: Second knowledge element
            pattern_ids: List of pattern IDs to use (optional)
            
        Returns:
            Contradiction score (0.0 to 1.0)
        """
        # Extract text content from knowledge elements
        text_a = self._extract_text(knowledge_a)
        text_b = self._extract_text(knowledge_b)
        
        # If either text is empty, no contradiction
        if not text_a or not text_b:
            return 0.0
        
        # If texts are identical, no contradiction
        if text_a == text_b:
            return 0.0
        
        # Apply contradiction patterns
        pattern_scores = []
        
        # If specific patterns are provided, use only those
        if pattern_ids:
            patterns_to_check = {
                pattern_id: pattern
                for pattern_id, pattern in self.contradiction_patterns.items()
                if pattern_id in pattern_ids
            }
        else:
            patterns_to_check = self.contradiction_patterns
        
        # If no patterns are available, use basic heuristics
        if not patterns_to_check:
            return self._basic_contradiction_heuristic(text_a, text_b)
        
        # Apply each pattern
        for pattern_id, pattern in patterns_to_check.items():
            pattern_type = pattern["type"]
            pattern_data = pattern["data"]
            
            if pattern_type == "regex":
                score = self._apply_regex_pattern(text_a, text_b, pattern_data)
            elif pattern_type == "semantic":
                score = self._apply_semantic_pattern(text_a, text_b, pattern_data)
            elif pattern_type == "logical":
                score = self._apply_logical_pattern(text_a, text_b, pattern_data)
            else:
                # Unknown pattern type, skip
                continue
                
            pattern_scores.append(score)
        
        # If no pattern scores were computed, use basic heuristic
        if not pattern_scores:
            return self._basic_contradiction_heuristic(text_a, text_b)
            
        # Return maximum pattern score
        return max(pattern_scores)
    
    def _extract_text(self, knowledge: Any) -> str:
        """
        Extract text content from a knowledge element.
        
        Args:
            knowledge: Knowledge element
            
        Returns:
            Text content
        """
        if isinstance(knowledge, str):
            return knowledge
        
        if isinstance(knowledge, dict):
            # Try common text fields
            for field in ["text", "content", "statement", "claim"]:
                if field in knowledge and isinstance(knowledge[field], str):
                    return knowledge[field]
            
            # If no text field found, convert the whole dict to string
            return str(knowledge)
        
        # Default: convert to string
        return str(knowledge)
    
    def _basic_contradiction_heuristic(self, text_a: str, text_b: str) -> float:
        """
        Apply a basic heuristic to detect contradictions.
        
        Args:
            text_a: First text
            text_b: Second text
            
        Returns:
            Contradiction score (0.0 to 1.0)
        """
        # Tokenize texts
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        
        # Calculate overlap
        overlap = len(words_a.intersection(words_b))
        union = len(words_a.union(words_b))
        
        if union == 0:
            return 0.0
            
        similarity = overlap / union
        
        # Check for negation words
        negation_words = {"not", "no", "never", "isn't", "aren't", "wasn't", "weren't", 
                          "doesn't", "don't", "didn't", "cannot", "can't", "won't"}
        
        has_negation_a = any(word in negation_words for word in words_a)
        has_negation_b = any(word in negation_words for word in words_b)
        
        # If one text has negation and the other doesn't, and they're similar,
        # they're likely contradictory
        if (has_negation_a != has_negation_b) and similarity > 0.3:
            return 0.7 + (similarity * 0.3)  # Higher score for more similar texts
        
        # Contradiction is highest when similarity is around 0.5
        # (very similar but not identical)
        return 1.0 - abs(similarity - 0.5) * 2
    
    def _apply_regex_pattern(self, text_a: str, text_b: str, pattern_data: Any) -> float:
        """
        Apply a regex pattern to detect contradictions.
        
        Args:
            text_a: First text
            text_b: Second text
            pattern_data: Regex pattern data
            
        Returns:
            Contradiction score (0.0 to 1.0)
        """
        if not isinstance(pattern_data, dict) or "pattern_a" not in pattern_data or "pattern_b" not in pattern_data:
            return 0.0
            
        pattern_a = pattern_data["pattern_a"]
        pattern_b = pattern_data["pattern_b"]
        
        try:
            matches_a = re.search(pattern_a, text_a, re.IGNORECASE)
            matches_b = re.search(pattern_b, text_b, re.IGNORECASE)
            
            if matches_a and matches_b:
                return 1.0
        except:
            # If regex fails, return 0
            pass
            
        return 0.0
    
    def _apply_semantic_pattern(self, text_a: str, text_b: str, pattern_data: Any) -> float:
        """
        Apply a semantic pattern to detect contradictions.
        
        Args:
            text_a: First text
            text_b: Second text
            pattern_data: Semantic pattern data
            
        Returns:
            Contradiction score (0.0 to 1.0)
        """
        # This is a placeholder - in a real implementation, this might use
        # semantic similarity or entailment models
        
        # For now, return 0
        return 0.0
    
    def _apply_logical_pattern(self, text_a: str, text_b: str, pattern_data: Any) -> float:
        """
        Apply a logical pattern to detect contradictions.
        
        Args:
            text_a: First text
            text_b: Second text
            pattern_data: Logical pattern data
            
        Returns:
            Contradiction score (0.0 to 1.0)
        """
        # This is a placeholder - in a real implementation, this might use
        # logical reasoning or rule-based systems
        
        # For now, return 0
        return 0.0
    
    def resolve_contradiction(
        self, 
        knowledge_a: Any, 
        knowledge_b: Any, 
        strategy_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Resolve contradiction between two knowledge elements.
        
        Args:
            knowledge_a: First knowledge element
            knowledge_b: Second knowledge element
            strategy_id: Strategy ID to use (optional)
            
        Returns:
            Resolution result or None if no contradiction or no suitable strategy
        """
        # Check if there's a contradiction
        if not self.detect_contradiction(knowledge_a, knowledge_b):
            return None  # No contradiction to resolve
            
        # If no specific strategy provided, use the first registered strategy
        if strategy_id is None and self.resolution_strategies:
            strategy_id = list(self.resolution_strategies.keys())[0]
            
        if strategy_id not in self.resolution_strategies:
            return None  # No suitable resolution strategy found
            
        # Get confidence scores if confidence ecosystem is available
        confidence_a = 0.5
        confidence_b = 0.5
        
        if self.confidence_ecosystem:
            # Extract IDs from knowledge elements
            id_a = knowledge_a.get("id") if isinstance(knowledge_a, dict) else None
            id_b = knowledge_b.get("id") if isinstance(knowledge_b, dict) else None
            
            if id_a:
                confidence_a = self.confidence_ecosystem.get_confidence(id_a)
            
            if id_b:
                confidence_b = self.confidence_ecosystem.get_confidence(id_b)
            
        # Apply resolution strategy
        resolution = self.resolution_strategies[strategy_id](
            knowledge_a, 
            knowledge_b,
            confidence_a,
            confidence_b
        )
        
        # Record the contradiction and its resolution
        contradiction_record = {
            "timestamp": time.time(),
            "knowledge_a": knowledge_a.get("id") if isinstance(knowledge_a, dict) else str(knowledge_a)[:50],
            "knowledge_b": knowledge_b.get("id") if isinstance(knowledge_b, dict) else str(knowledge_b)[:50],
            "strategy": strategy_id,
            "resolution": resolution,
            "confidence_a": confidence_a,
            "confidence_b": confidence_b
        }
        
        self.contradiction_history.append(contradiction_record)
        
        # Trim history if it gets too long
        if len(self.contradiction_history) > 1000:
            self.contradiction_history = self.contradiction_history[-1000:]
        
        return resolution
    
    def get_contradiction_history(
        self, 
        limit: int = 100, 
        strategy_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get contradiction resolution history.
        
        Args:
            limit: Maximum number of records to return
            strategy_id: Filter by strategy ID (optional)
            
        Returns:
            List of contradiction records
        """
        if strategy_id:
            filtered_history = [
                record for record in self.contradiction_history
                if record["strategy"] == strategy_id
            ]
        else:
            filtered_history = self.contradiction_history
            
        # Return most recent records first
        return sorted(filtered_history, key=lambda x: x["timestamp"], reverse=True)[:limit]
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about contradiction handling.
        
        Returns:
            Dictionary of metrics
        """
        if not self.contradiction_history:
            return {
                "total_contradictions": 0,
                "resolution_rate": 0.0,
                "strategy_distribution": {}
            }
            
        total = len(self.contradiction_history)
        resolved = sum(1 for entry in self.contradiction_history if entry["resolution"] is not None)
        
        # Group by strategy
        strategy_counts = {}
        for entry in self.contradiction_history:
            strategy = entry["strategy"]
            if strategy not in strategy_counts:
                strategy_counts[strategy] = 0
            strategy_counts[strategy] += 1
        
        # Convert counts to percentages
        strategy_distribution = {
            strategy: count / total
            for strategy, count in strategy_counts.items()
        }
        
        return {
            "total_contradictions": total,
            "resolution_rate": resolved / total if total > 0 else 0.0,
            "strategy_distribution": strategy_distribution,
            "registered_strategies_count": len(self.resolution_strategies),
            "registered_patterns_count": len(self.contradiction_patterns)
        }
