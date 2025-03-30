import logging
from collections import defaultdict, Counter
import uuid
import re
from typing import Dict, List, Any, Optional, Tuple, Set
import json
from dataclasses import dataclass
import time

class ComplianceConstraintOptimizer:
    """
    Optimizes compliance constraints to balance compliance requirements and user experience.
    
    This class analyzes compliance filtering results and identifies ways to optimize
    the filtering process to reduce false positives, minimize user friction, and
    maximize content throughput while maintaining compliance requirements.
    """
    
    def __init__(self, config):
        self.config = config
        
        # Configure optimizer settings
        self.optimization_level = config.get("optimization_level", "balanced")
        self.max_constraint_adjustments = config.get("max_constraint_adjustments", 3)
        self.adjustment_factor = config.get("adjustment_factor", 0.1)
        self.min_confidence_threshold = config.get("min_confidence_threshold", 0.5)
        
        # Load configurable constraints
        self.configurable_constraints = config.get("configurable_constraints", {})
        
        # Initialize constraint adjustment history
        self.adjustment_history = []
        
        # Initialize allowed modifications
        self.allowed_modifications = config.get("allowed_modifications", {
            "relaxation": True,
            "tightening": True,
            "contextualization": True,
            "rule_specific": True
        })
        
        # Load exemption patterns
        self.exemption_patterns = config.get("exemption_patterns", [])
        
        # Load category weights for balancing
        self.category_weights = config.get("category_weights", {})
        
        # Configure optimization strategies
        self.optimization_strategies = self._initialize_optimization_strategies()
        
    def optimize_constraints(self, filter_results, original_input, context=None):
        """
        Optimize compliance constraints based on filtering results.
        
        Args:
            filter_results: Results from compliance filtering
            original_input: Original input content before filtering
            context: Optional context information
            
        Returns:
            Dict with optimized constraints and modifications
        """
        # Skip optimization if no issues or if already compliant
        if not filter_results.get("issues", []) or filter_results.get("is_compliant", True):
            return {
                "optimized": False,
                "constraint_adjustments": [],
                "alternative_formulations": [],
                "exempt_patterns": [],
                "original_constraints": {}
            }
            
        # Extract issues and determine if optimization is possible
        issues = filter_results.get("issues", [])
        
        # Select optimization strategy based on configuration
        strategy = self.optimization_level
        optimizer = self.optimization_strategies.get(strategy, self.optimization_strategies["balanced"])
        
        # Apply the selected optimization strategy
        optimization_results = optimizer(issues, original_input, context)
        
        # Track adjustments in history for future reference
        self._track_adjustments(optimization_results.get("constraint_adjustments", []))
        
        return optimization_results
        
    def generate_alternative_formulations(self, text, issues, context=None):
        """
        Generate alternative formulations of content to address compliance issues.
        
        Args:
            text: Original text content
            issues: Compliance issues detected
            context: Optional context information
            
        Returns:
            List of alternative formulations
        """
        if not issues:
            return []
            
        alternatives = []
        
        # Group issues by type for targeted modifications
        issues_by_rule = defaultdict(list)
        for issue in issues:
            rule_id = issue.get("rule_id", "")
            if rule_id:
                issues_by_rule[rule_id].append(issue)
                
        # Generate alternatives for each rule violation
        for rule_id, rule_issues in issues_by_rule.items():
            # Get rule-specific alternatives
            rule_alternatives = self._generate_rule_specific_alternatives(text, rule_id, rule_issues, context)
            if rule_alternatives:
                alternatives.extend(rule_alternatives)
                
        # If no rule-specific alternatives, try general strategies
        if not alternatives:
            # Try general rephrasing strategies
            general_alternatives = self._apply_general_alternatives(text, issues, context)
            if general_alternatives:
                alternatives.extend(general_alternatives)
                
        # Deduplicate alternatives
        unique_alternatives = []
        seen = set()
        
        for alt in alternatives:
            alt_text = alt.get("text", "")
            if alt_text and alt_text not in seen:
                seen.add(alt_text)
                unique_alternatives.append(alt)
                
        return unique_alternatives
        
    def identify_exemption_patterns(self, issues, original_input, context=None):
        """
        Identify patterns that might qualify for compliance exemptions.
        
        Args:
            issues: Compliance issues detected
            original_input: Original input content
            context: Optional context information
            
        Returns:
            List of potential exemption patterns
        """
        if not issues or not original_input:
            return []
            
        potential_exemptions = []
        
        # Check for existing exemption patterns
        matching_exemptions = self._match_existing_exemptions(issues, original_input)
        if matching_exemptions:
            potential_exemptions.extend(matching_exemptions)
            
        # Look for new potential exemption patterns
        new_exemptions = self._identify_new_exemptions(issues, original_input, context)
        if new_exemptions:
            potential_exemptions.extend(new_exemptions)
            
        return potential_exemptions
        
    def recommend_constraint_adjustments(self, filter_results, historical_data=None):
        """
        Recommend adjustments to compliance constraints based on results.
        
        Args:
            filter_results: Results from compliance filtering
            historical_data: Optional historical filtering data
            
        Returns:
            Dict with recommended constraint adjustments
        """
        issues = filter_results.get("issues", [])
        
        if not issues:
            return {"adjustments": []}
            
        recommended_adjustments = []
        
        # Calculate false positive likelihood for each issue
        fp_likelihoods = self._estimate_false_positive_likelihoods(issues, historical_data)
        
        # Find constraints with high false positive rates
        for rule_id, likelihood in fp_likelihoods.items():
            if likelihood > 0.7:  # Threshold for high false positive likelihood
                if rule_id in self.configurable_constraints:
                    constraint = self.configurable_constraints[rule_id]
                    
                    # Calculate suggested adjustment
                    current_value = constraint.get("current_value", 0)
                    adjustment = current_value * self.adjustment_factor
                    
                    # Ensure the adjustment is within allowed bounds
                    min_value = constraint.get("min_value", 0)
                    max_value = constraint.get("max_value", 1)
                    
                    new_value = min(max(current_value + adjustment, min_value), max_value)
                    
                    if new_value != current_value:
                        recommended_adjustments.append({
                            "rule_id": rule_id,
                            "parameter": constraint.get("parameter", "threshold"),
                            "current_value": current_value,
                            "recommended_value": new_value,
                            "reason": f"High false positive likelihood ({likelihood:.2f})",
                            "confidence": likelihood
                        })
                        
        # Sort adjustments by confidence
        recommended_adjustments.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        
        # Limit to max allowed adjustments
        recommended_adjustments = recommended_adjustments[:self.max_constraint_adjustments]
        
        return {"adjustments": recommended_adjustments}
        
    def balance_compliance_requirements(self, compliance_config, risk_tolerance=None):
        """
        Balance compliance requirements based on risk tolerance.
        
        Args:
            compliance_config: Current compliance configuration
            risk_tolerance: Optional risk tolerance level (low, medium, high)
            
        Returns:
            Dict with balanced compliance configuration
        """
        if risk_tolerance is None:
            risk_tolerance = self.config.get("default_risk_tolerance", "medium")
            
        # Clone the original configuration to avoid modifying it
        balanced_config = {k: v for k, v in compliance_config.items()}
        
        # Adjust thresholds based on risk tolerance
        tolerance_factors = {
            "low": 0.8,     # Stricter thresholds
            "medium": 1.0,  # No change
            "high": 1.2     # More lenient thresholds
        }
        
        factor = tolerance_factors.get(risk_tolerance, 1.0)
        
        # Adjust configurable constraint thresholds
        if "configurable_constraints" in balanced_config:
            for rule_id, constraint in balanced_config["configurable_constraints"].items():
                if "current_value" in constraint and "parameter" in constraint:
                    parameter = constraint["parameter"]
                    
                    # Only adjust thresholds
                    if parameter == "threshold":
                        current_value = constraint["current_value"]
                        min_value = constraint.get("min_value", 0)
                        max_value = constraint.get("max_value", 1)
                        
                        # Adjust based on risk tolerance
                        new_value = current_value * factor
                        
                        # Ensure within bounds
                        new_value = min(max(new_value, min_value), max_value)
                        
                        balanced_config["configurable_constraints"][rule_id]["current_value"] = new_value
                        
        # Adjust category weights
        if "category_weights" in balanced_config:
            for category, weight in balanced_config["category_weights"].items():
                # Adjust category weights inversely to risk tolerance
                # Higher risk tolerance means lower weight for non-critical categories
                if category != "critical":
                    new_weight = weight / factor
                    balanced_config["category_weights"][category] = new_weight
                    
        return {
            "balanced_config": balanced_config,
            "risk_tolerance": risk_tolerance,
            "adjustment_factor": factor
        }
        
    def _initialize_optimization_strategies(self):
        """Initialize optimization strategies."""
        return {
            "strict": self._optimize_strict,
            "balanced": self._optimize_balanced,
            "lenient": self._optimize_lenient,
            "adaptive": self._optimize_adaptive
        }
        
    def _optimize_strict(self, issues, original_input, context):
        """
        Strict optimization strategy - prioritizes compliance.
        
        Makes minimal adjustments, focusing only on high-confidence false positives.
        """
        # Only consider high-confidence false positives
        constraint_adjustments = self._identify_constraint_adjustments(
            issues, 
            confidence_threshold=0.85,
            max_adjustments=1
        )
        
        # Look for exemption patterns with high confidence
        exemptions = self._match_existing_exemptions(issues, original_input)
        
        # Generate only minimal alternatives
        alternatives = self.generate_alternative_formulations(original_input, issues, context)
        alternatives = alternatives[:1]  # Only the best alternative
        
        return {
            "optimized": bool(constraint_adjustments or exemptions or alternatives),
            "optimization_level": "strict",
            "constraint_adjustments": constraint_adjustments,
            "alternative_formulations": alternatives,
            "exempt_patterns": exemptions,
            "original_constraints": self._get_original_constraints(issues)
        }
        
    def _optimize_balanced(self, issues, original_input, context):
        """
        Balanced optimization strategy - balances compliance and user experience.
        
        Makes moderate adjustments to reduce false positives while maintaining compliance.
        """
        # Consider moderate-confidence false positives
        constraint_adjustments = self._identify_constraint_adjustments(
            issues, 
            confidence_threshold=0.7,
            max_adjustments=2
        )
        
        # Look for exemption patterns
        exemptions = self.identify_exemption_patterns(issues, original_input, context)
        
        # Generate alternatives
        alternatives = self.generate_alternative_formulations(original_input, issues, context)
        alternatives = alternatives[:3]  # Top 3 alternatives
        
        return {
            "optimized": bool(constraint_adjustments or exemptions or alternatives),
            "optimization_level": "balanced",
            "constraint_adjustments": constraint_adjustments,
            "alternative_formulations": alternatives,
            "exempt_patterns": exemptions,
            "original_constraints": self._get_original_constraints(issues)
        }
        
    def _optimize_lenient(self, issues, original_input, context):
        """
        Lenient optimization strategy - prioritizes user experience.
        
        Makes more aggressive adjustments to reduce false positives, focusing on user experience.
        """
        # Consider lower-confidence false positives
        constraint_adjustments = self._identify_constraint_adjustments(
            issues, 
            confidence_threshold=0.6,
            max_adjustments=self.max_constraint_adjustments
        )
        
        # Look for exemption patterns more aggressively
        exemptions = self.identify_exemption_patterns(issues, original_input, context)
        
        # Generate more alternatives
        alternatives = self.generate_alternative_formulations(original_input, issues, context)
        
        return {
            "optimized": bool(constraint_adjustments or exemptions or alternatives),
            "optimization_level": "lenient",
            "constraint_adjustments": constraint_adjustments,
            "alternative_formulations": alternatives,
            "exempt_patterns": exemptions,
            "original_constraints": self._get_original_constraints(issues)
        }
        
    def _optimize_adaptive(self, issues, original_input, context):
        """
        Adaptive optimization strategy - adjusts based on context and history.
        
        Adapts optimization approach based on context, severity, and historical data.
        """
        # Determine appropriate strategy based on context and issue severity
        strategy = self._determine_adaptive_strategy(issues, context)
        
        # Use the selected strategy
        if strategy == "strict":
            return self._optimize_strict(issues, original_input, context)
        elif strategy == "lenient":
            return self._optimize_lenient(issues, original_input, context)
        else:
            return self._optimize_balanced(issues, original_input, context)
            
    def _determine_adaptive_strategy(self, issues, context):
        """Determine the appropriate strategy for adaptive optimization."""
        # Default to balanced
        if not issues:
            return "balanced"
            
        # Check for critical issues
        has_critical = any(issue.get("severity") == "critical" for issue in issues)
        if has_critical:
            return "strict"
            
        # Check context for risk factors
        if context:
            domain = context.get("domain", "general")
            
            # High-risk domains use strict strategy
            high_risk_domains = self.config.get("high_risk_domains", [])
            if domain in high_risk_domains:
                return "strict"
                
            # Low-risk domains can use lenient strategy
            low_risk_domains = self.config.get("low_risk_domains", [])
            if domain in low_risk_domains:
                return "lenient"
                
            # Check user risk level
            user_info = context.get("user_info", {})
            user_risk = user_info.get("risk_level", "medium")
            
            if user_risk == "low":
                return "lenient"
            elif user_risk == "high":
                return "strict"
                
        # Count issue severity
        severity_counts = Counter(issue.get("severity", "medium") for issue in issues)
        
        # Many high-severity issues, use strict
        if severity_counts.get("high", 0) > 3:
            return "strict"
            
        # Few low-severity issues, use lenient
        if severity_counts.get("low", 0) > 0 and not severity_counts.get("high", 0) and not severity_counts.get("medium", 0):
            return "lenient"
            
        # Default to balanced
        return "balanced"
        
    def _identify_constraint_adjustments(self, issues, confidence_threshold=0.7, max_adjustments=None):
        """Identify potential constraint adjustments to reduce false positives."""
        if not issues:
            return []
            
        adjustments = []
        
        # Group issues by rule ID
        issues_by_rule = defaultdict(list)
        for issue in issues:
            rule_id = issue.get("rule_id", "")
            if rule_id:
                issues_by_rule[rule_id].append(issue)
                
        # Estimate false positive likelihood for each rule
        fp_likelihoods = self._estimate_false_positive_likelihoods(issues)
        
        # Consider adjustments for rules with high FP likelihood
        for rule_id, likelihood in fp_likelihoods.items():
            if likelihood >= confidence_threshold and rule_id in self.configurable_constraints:
                constraint = self.configurable_constraints[rule_id]
                
                # Calculate adjustment based on FP likelihood
                parameter = constraint.get("parameter", "threshold")
                current_value = constraint.get("current_value", 0)
                
                # Adjustment size based on FP likelihood
                adjustment_size = likelihood * self.adjustment_factor
                
                # Determine direction based on parameter type
                if parameter in ["threshold", "sensitivity", "min_confidence"]:
                    # Increase thresholds to reduce false positives
                    new_value = current_value + adjustment_size
                elif parameter in ["max_distance", "window_size"]:
                    # Decrease distances to make rules more precise
                    new_value = current_value - (current_value * adjustment_size)
                else:
                    # Default behavior
                    new_value = current_value + adjustment_size
                    
                # Ensure within bounds
                min_value = constraint.get("min_value", 0)
                max_value = constraint.get("max_value", 1)
                new_value = min(max(new_value, min_value), max_value)
                
                if new_value != current_value:
                    adjustments.append({
                        "rule_id": rule_id,
                        "parameter": parameter,
                        "current_value": current_value,
                        "new_value": new_value,
                        "confidence": likelihood,
                        "affected_issues": len(issues_by_rule[rule_id])
                    })
                    
        # Sort by confidence and affected issues
        adjustments.sort(key=lambda x: (x["confidence"], x["affected_issues"]), reverse=True)
        
        # Limit to max adjustments if specified
        if max_adjustments is not None:
            adjustments = adjustments[:max_adjustments]
            
        return adjustments
        
    def _estimate_false_positive_likelihoods(self, issues, historical_data=None):
        """Estimate likelihood that each issue is a false positive."""
        # In a real system, this would use ML or heuristics based on historical data
        # Here, we'll use a simple heuristic approach
        
        fp_likelihoods = {}
        
        for issue in issues:
            rule_id = issue.get("rule_id", "")
            confidence = issue.get("confidence", 0.5)
            metadata = issue.get("metadata", {})
            
            # Simple heuristic: lower confidence issues are more likely to be false positives
            fp_likelihood = 1.0 - confidence
            
            # If rule has specific FP indicators, adjust likelihood
            if rule_id in self.config.get("false_positive_indicators", {}):
                indicators = self.config["false_positive_indicators"][rule_id]
                
                for indicator, weight in indicators.items():
                    if indicator in metadata and metadata[indicator]:
                        fp_likelihood += weight
                        
            # Bound between 0 and 1
            fp_likelihood = min(max(fp_likelihood, 0.0), 1.0)
            
            # Store the highest likelihood for each rule
            if rule_id not in fp_likelihoods or fp_likelihood > fp_likelihoods[rule_id]:
                fp_likelihoods[rule_id] = fp_likelihood
                
        return fp_likelihoods
        
    def _match_existing_exemptions(self, issues, original_input):
        """Match issues against existing exemption patterns."""
        matching_exemptions = []
        
        for issue in issues:
            rule_id = issue.get("rule_id", "")
            
            for exemption in self.exemption_patterns:
                if exemption.get("rule_id") == rule_id:
                    pattern = exemption.get("pattern", "")
                    
                    # Skip invalid patterns
                    if not pattern:
                        continue
                        
                    try:
                        if re.search(pattern, original_input, re.IGNORECASE):
                            matching_exemptions.append({
                                "rule_id": rule_id,
                                "pattern": pattern,
                                "description": exemption.get("description", ""),
                                "confidence": exemption.get("confidence", 0.8)
                            })
                    except re.error:
                        # Log invalid regex pattern
                        logging.error(f"Invalid exemption regex pattern: {pattern}")
                        
        return matching_exemptions
        
    def _identify_new_exemptions(self, issues, original_input, context):
        """Identify potential new exemption patterns."""
        # This is a placeholder for a more sophisticated implementation
        # In a real system, this would analyze patterns more deeply
        
        # Simple approach: look for contextual indicators that might justify exemptions
        potential_exemptions = []
        
        # Check if context includes exemption indicators
        if context and "metadata" in context:
            metadata = context["metadata"]
            
            # Educational or research context might justify exemptions
            if metadata.get("purpose") in ["educational", "research", "analysis"]:
                for issue in issues:
                    rule_id = issue.get("rule_id", "")
                    
                    # For now, just suggest the possibility
                    potential_exemptions.append({
                        "rule_id": rule_id,
                        "pattern": None,  # No specific pattern yet
                        "description": f"Potential exemption for {rule_id} in {metadata.get('purpose')} context",
                        "confidence": 0.6,
                        "is_suggestion": True
                    })
                    
        return potential_exemptions
        
    def _generate_rule_specific_alternatives(self, text, rule_id, issues, context):
        """Generate alternative formulations specific to a rule."""
        alternatives = []
        
        # In a real system, this would use more sophisticated techniques
        # based on rule-specific knowledge
        
        # Example: Handle specific rule types
        if rule_id.startswith("pii_"):
            # For PII rules, try redaction
            alt_text = text
            for issue in issues:
                metadata = issue.get("metadata", {})
                if "location" in metadata:
                    start = metadata["location"].get("start", 0)
                    end = metadata["location"].get("end", 0)
                    
                    if start < end and end <= len(alt_text):
                        # Replace with [REDACTED]
                        alt_text = alt_text[:start] + "[REDACTED]" + alt_text[end:]
                        
            if alt_text != text:
                alternatives.append({
                    "text": alt_text,
                    "rule_id": rule_id,
                    "confidence": 0.9,
                    "type": "redaction"
                })
                
        elif rule_id.startswith("keyword_"):
            # For keyword rules, try synonym replacement
            # In a real system, this would use a thesaurus or embedding model
            alt_text = text
            for issue in issues:
                metadata = issue.get("metadata", {})
                matched_keyword = metadata.get("matched_keyword", "")
                
                if matched_keyword and matched_keyword in alt_text:
                    # Simple demonstration - in a real system, use actual synonyms
                    placeholder = f"[alternative for '{matched_keyword}']"
                    alt_text = alt_text.replace(matched_keyword, placeholder)
                    
            if alt_text != text:
                alternatives.append({
                    "text": alt_text,
                    "rule_id": rule_id,
                    "confidence": 0.7,
                    "type": "synonym_replacement"
                })
                
        return alternatives
        
    def _apply_general_alternatives(self, text, issues, context):
        """Apply general strategies for generating alternatives."""
        alternatives = []
        
        # 1. Try removing problematic segments
        segments_to_remove = []
        
        for issue in issues:
            metadata = issue.get("metadata", {})
            
            if "location" in metadata:
                start = metadata["location"].get("start", 0)
                end = metadata["location"].get("end", 0)
                
                if start < end and end <= len(text):
                    segments_to_remove.append((start, end))
                    
        if segments_to_remove:
            # Sort segments in reverse order so removal doesn't affect indices
            segments_to_remove.sort(reverse=True)
            
            alt_text = text
            for start, end in segments_to_remove:
                alt_text = alt_text[:start] + alt_text[end:]
                
            if alt_text and alt_text != text:
                alternatives.append({
                    "text": alt_text,
                    "confidence": 0.6,
                    "type": "segment_removal"
                })
                
        # 2. Try rewording (placeholder - in a real system this would use NLP)
        alternatives.append({
            "text": f"[Alternative formulation requested: {len(text)} characters]",
            "confidence": 0.5,
            "type": "reword_suggestion"
        })
        
        return alternatives
        
    def _get_original_constraints(self, issues):
        """Get the original constraints that triggered the issues."""
        original_constraints = {}
        
        for issue in issues:
            rule_id = issue.get("rule_id", "")
            if rule_id in self.configurable_constraints:
                constraint = self.configurable_constraints[rule_id]
                original_constraints[rule_id] = {
                    "parameter": constraint.get("parameter", "threshold"),
                    "current_value": constraint.get("current_value", 0)
                }
                
        return original_constraints
        
    def _track_adjustments(self, adjustments):
        """Track constraint adjustments for history."""
        timestamp = time.time()
        
        for adjustment in adjustments:
            self.adjustment_history.append({
                "timestamp": timestamp,
                "rule_id": adjustment.get("rule_id", ""),
                "parameter": adjustment.get("parameter", ""),
                "old_value": adjustment.get("current_value", 0),
                "new_value": adjustment.get("new_value", 0),
                "confidence": adjustment.get("confidence", 0)
            })
            
        # Keep history limited
        if len(self.adjustment_history) > 100:
            self.adjustment_history = self.adjustment_history[-100:]