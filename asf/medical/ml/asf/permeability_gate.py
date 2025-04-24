"""
Permeability Gate Module

This module implements the Contextual Permeability Gates component of the ASF framework,
which controls information flow based on context-dependent validation thresholds.
"""

import time
from typing import Dict, List, Optional, Tuple, Union, Any, Callable


class PermeabilityGate:
    """
    Controls information flow based on context-dependent validation thresholds.
    
    The Permeability Gate evaluates incoming information against validation criteria
    in a context-dependent manner, determining whether the information should pass
    through the gate and influence the system's state or output.
    """
    
    def __init__(self, default_threshold: float = 0.5):
        """
        Initialize the Permeability Gate.
        
        Args:
            default_threshold: Default threshold for validation (default: 0.5)
        """
        self.default_threshold = default_threshold
        self.context_thresholds: Dict[str, float] = {}  # Map of context_id -> threshold
        self.validation_criteria: Dict[str, Callable] = {}  # Map of criteria_id -> validation_function
        self.gate_history: List[Dict[str, Any]] = []  # Track gate decisions for analysis
    
    def register_context(self, context_id: str, threshold: float) -> bool:
        """
        Register a context with its threshold.
        
        Args:
            context_id: Context ID
            threshold: Threshold for this context
            
        Returns:
            Success flag
        """
        self.context_thresholds[context_id] = threshold
        return True
    
    def register_validation_criterion(self, criteria_id: str, validation_function: Callable) -> bool:
        """
        Register a validation criterion with its evaluation function.
        
        Args:
            criteria_id: Criterion ID
            validation_function: Function to evaluate information against this criterion
            
        Returns:
            Success flag
        """
        self.validation_criteria[criteria_id] = validation_function
        return True
    
    def get_threshold(self, context_id: str) -> float:
        """
        Get threshold for a specific context.
        
        Args:
            context_id: Context ID
            
        Returns:
            Threshold value
        """
        return self.context_thresholds.get(context_id, self.default_threshold)
    
    def evaluate(
        self, 
        information: Any, 
        context_id: Optional[str] = None, 
        criteria_ids: Optional[List[str]] = None
    ) -> bool:
        """
        Evaluate information against validation criteria in the given context.
        
        Args:
            information: Information to evaluate
            context_id: Context ID (optional)
            criteria_ids: List of criteria IDs to use (optional)
            
        Returns:
            Whether information passes the gate
        """
        threshold = self.get_threshold(context_id) if context_id else self.default_threshold
        
        # If no specific criteria provided, use all registered criteria
        if criteria_ids is None:
            criteria_ids = list(self.validation_criteria.keys())
            
        # If no criteria available, default to passing
        if not criteria_ids or not self.validation_criteria:
            return True
            
        # Evaluate each criterion
        scores = []
        criterion_scores = {}
        
        for criteria_id in criteria_ids:
            if criteria_id in self.validation_criteria:
                try:
                    score = self.validation_criteria[criteria_id](information)
                    scores.append(score)
                    criterion_scores[criteria_id] = score
                except Exception as e:
                    # If criterion evaluation fails, log it but continue
                    criterion_scores[criteria_id] = 0.0
                
        # If no scores were computed, return True (pass by default)
        if not scores:
            return True
            
        # Compute average score
        avg_score = sum(scores) / len(scores)
        
        # Determine if information passes the gate
        passes = avg_score >= threshold
        
        # Record decision in history
        self.gate_history.append({
            "timestamp": time.time(),
            "context_id": context_id,
            "threshold": threshold,
            "average_score": avg_score,
            "criterion_scores": criterion_scores,
            "passed": passes
        })
        
        # Trim history if it gets too long
        if len(self.gate_history) > 1000:
            self.gate_history = self.gate_history[-1000:]
        
        # Return whether information passes the gate
        return passes
    
    def get_context_performance(self, context_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance metrics for a specific context.
        
        Args:
            context_id: Context ID (optional)
            
        Returns:
            Dictionary of performance metrics
        """
        if not self.gate_history:
            return {
                "total_evaluations": 0,
                "pass_rate": 0.0,
                "average_score": 0.0
            }
        
        # Filter history by context if provided
        if context_id:
            relevant_history = [
                entry for entry in self.gate_history
                if entry.get("context_id") == context_id
            ]
        else:
            relevant_history = self.gate_history
            
        if not relevant_history:
            return {
                "total_evaluations": 0,
                "pass_rate": 0.0,
                "average_score": 0.0
            }
            
        total = len(relevant_history)
        passed = sum(1 for entry in relevant_history if entry["passed"])
        avg_score = sum(entry["average_score"] for entry in relevant_history) / total
        
        return {
            "total_evaluations": total,
            "pass_rate": passed / total,
            "average_score": avg_score
        }
    
    def get_criterion_performance(self, criteria_id: str) -> Dict[str, Any]:
        """
        Get performance metrics for a specific criterion.
        
        Args:
            criteria_id: Criterion ID
            
        Returns:
            Dictionary of performance metrics
        """
        if not self.gate_history or criteria_id not in self.validation_criteria:
            return {
                "total_evaluations": 0,
                "average_score": 0.0
            }
        
        # Filter history entries that include this criterion
        relevant_entries = [
            entry for entry in self.gate_history
            if criteria_id in entry.get("criterion_scores", {})
        ]
        
        if not relevant_entries:
            return {
                "total_evaluations": 0,
                "average_score": 0.0
            }
            
        total = len(relevant_entries)
        avg_score = sum(
            entry["criterion_scores"][criteria_id]
            for entry in relevant_entries
        ) / total
        
        return {
            "total_evaluations": total,
            "average_score": avg_score
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about gate performance.
        
        Returns:
            Dictionary of metrics
        """
        if not self.gate_history:
            return {
                "total_evaluations": 0,
                "pass_rate": 0,
                "average_score": 0,
                "contexts": {},
                "criteria": {}
            }
            
        total = len(self.gate_history)
        passed = sum(1 for entry in self.gate_history if entry["passed"])
        avg_score = sum(entry["average_score"] for entry in self.gate_history) / total
        
        # Get metrics by context
        context_metrics = {}
        contexts = set(entry.get("context_id") for entry in self.gate_history if entry.get("context_id"))
        for context_id in contexts:
            context_metrics[context_id] = self.get_context_performance(context_id)
            
        # Get metrics by criterion
        criterion_metrics = {}
        for criteria_id in self.validation_criteria:
            criterion_metrics[criteria_id] = self.get_criterion_performance(criteria_id)
        
        return {
            "total_evaluations": total,
            "pass_rate": passed / total,
            "average_score": avg_score,
            "contexts": context_metrics,
            "criteria": criterion_metrics,
            "default_threshold": self.default_threshold,
            "registered_criteria_count": len(self.validation_criteria),
            "registered_contexts_count": len(self.context_thresholds)
        }
