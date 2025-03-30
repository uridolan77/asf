import logging
from collections import defaultdict, Counter
import uuid
import re
from typing import Dict, List, Any, Optional, Tuple, Set
import json
from dataclasses import dataclass
import time


@dataclass
class ViolationSummary:
    """Summary of compliance violations found during filtering."""
    violation_count: int
    severity_counts: Dict[str, int]  # Counts by severity level
    categories: Dict[str, int]       # Counts by violation category
    top_rules: List[Dict[str, Any]]  # Most frequently triggered rules
    primary_severity: str            # Most severe violation level
    timestamp: float                 # When the analysis was performed
    suggestion: Optional[str] = None # Suggested remediation


class ViolationAnalyzer:
    """
    Analyzes compliance violations to provide insights and remediation suggestions.
    
    This class processes violation data from various filter components, categorizes them,
    identifies patterns, generates reports, and provides suggestions for addressing violations.
    """
    
    def __init__(self, config):
        self.config = config
        
        # Configure analyzer settings
        self.min_violations_for_pattern = config.get("min_violations_for_pattern", 3)
        self.severity_weights = config.get("severity_weights", {
            "critical": 100,
            "high": 50,
            "medium": 10,
            "low": 1
        })
        
        # Configure rule metadata
        self.rule_metadata = self._load_rule_metadata(config.get("rule_metadata", {}))
        
        # Configure remediation templates
        self.remediation_templates = config.get("remediation_templates", {})
        
        # Configure category mapping
        self.category_mapping = config.get("category_mapping", {})
        
        # Configure violation thresholds
        self.violation_thresholds = config.get("violation_thresholds", {
            "critical": 1,
            "high": 2,
            "medium": 5,
            "low": 10
        })
        
        # Historical tracking (could be connected to a database in a real system)
        self.historical_violations = defaultdict(list)
        self.violation_trends = defaultdict(list)
        
        # Pattern recognition settings
        self.pattern_recognition_enabled = config.get("pattern_recognition_enabled", True)
        self.pattern_matchers = self._initialize_pattern_matchers()
        
    def analyze_violations(self, filter_results, context=None):
        """
        Analyze compliance violations from filter results.
        
        Args:
            filter_results: Results from compliance filtering
            context: Optional context information
            
        Returns:
            Dict with analysis results
        """
        violations = filter_results.get("issues", [])
        
        if not violations:
            return {
                "violation_count": 0,
                "has_violations": False,
                "summary": None
            }
            
        # Extract violations by severity
        violations_by_severity = self._group_by_severity(violations)
        
        # Extract violations by category
        violations_by_category = self._group_by_category(violations)
        
        # Identify the most triggered rules
        top_rules = self._identify_top_rules(violations)
        
        # Determine primary severity
        primary_severity = self._determine_primary_severity(violations_by_severity)
        
        # Generate violation summary
        summary = ViolationSummary(
            violation_count=len(violations),
            severity_counts={severity: len(violations_list) for severity, violations_list in violations_by_severity.items()},
            categories={category: len(violations_list) for category, violations_list in violations_by_category.items()},
            top_rules=top_rules[:5],  # Top 5 rules
            primary_severity=primary_severity,
            timestamp=time.time()
        )
        
        # Generate remediation suggestion
        suggestion = self._generate_remediation_suggestion(violations, context, filter_results.get("filtered_input", ""))
        summary.suggestion = suggestion
        
        # Update historical tracking
        self._update_historical_data(violations, context)
        
        # Detect patterns if enabled
        patterns = {}
        if self.pattern_recognition_enabled:
            patterns = self._detect_violation_patterns(violations, context)
            
        # Create full analysis results
        analysis_results = {
            "violation_count": len(violations),
            "has_violations": True,
            "summary": summary.__dict__,
            "violations_by_severity": {k: [v.__dict__ if hasattr(v, '__dict__') else v for v in violations_by_severity[k]] for k in violations_by_severity},
            "violations_by_category": {k: [v.__dict__ if hasattr(v, '__dict__') else v for v in violations_by_category[k]] for k in violations_by_category},
            "patterns": patterns,
            "impact_assessment": self._assess_impact(violations, context)
        }
        
        return analysis_results
        
    def generate_violation_report(self, analysis_results, format="json"):
        """
        Generate a formatted report of violation analysis.
        
        Args:
            analysis_results: Results from analyze_violations
            format: Report format (json, text, html)
            
        Returns:
            Formatted report
        """
        if not analysis_results.get("has_violations", False):
            return "No compliance violations detected."
            
        summary = analysis_results.get("summary", {})
        
        if format == "json":
            # Return JSON format report
            return json.dumps(analysis_results, indent=2)
            
        elif format == "html":
            # Generate HTML report
            # In a real implementation, this would use a template engine
            html_parts = [
                "<html><head><title>Compliance Violation Report</title></head><body>",
                f"<h1>Compliance Violation Report</h1>",
                f"<h2>Summary</h2>",
                f"<p>Total violations: {summary.get('violation_count', 0)}</p>",
                f"<p>Primary severity: {summary.get('primary_severity', 'Unknown')}</p>",
                "<h3>Violations by Severity</h3>",
                "<ul>"
            ]
            
            for severity, count in summary.get("severity_counts", {}).items():
                html_parts.append(f"<li>{severity}: {count}</li>")
                
            html_parts.append("</ul>")
            html_parts.append("<h3>Violations by Category</h3>")
            html_parts.append("<ul>")
            
            for category, count in summary.get("categories", {}).items():
                html_parts.append(f"<li>{category}: {count}</li>")
                
            html_parts.append("</ul>")
            
            if summary.get("suggestion"):
                html_parts.append("<h3>Suggested Remediation</h3>")
                html_parts.append(f"<p>{summary.get('suggestion')}</p>")
                
            html_parts.append("</body></html>")
            
            return "".join(html_parts)
            
        else:  # Default to text format
            # Generate plain text report
            text_parts = [
                "COMPLIANCE VIOLATION REPORT",
                "===========================",
                f"Total violations: {summary.get('violation_count', 0)}",
                f"Primary severity: {summary.get('primary_severity', 'Unknown')}",
                "",
                "VIOLATIONS BY SEVERITY",
                "----------------------"
            ]
            
            for severity, count in summary.get("severity_counts", {}).items():
                text_parts.append(f"{severity}: {count}")
                
            text_parts.append("")
            text_parts.append("VIOLATIONS BY CATEGORY")
            text_parts.append("---------------------")
            
            for category, count in summary.get("categories", {}).items():
                text_parts.append(f"{category}: {count}")
                
            text_parts.append("")
            
            if "top_rules" in summary:
                text_parts.append("TOP TRIGGERED RULES")
                text_parts.append("-----------------")
                
                for rule in summary.get("top_rules", []):
                    text_parts.append(f"- {rule.get('rule_id', 'Unknown')}: {rule.get('count', 0)} violations")
                    
                text_parts.append("")
                
            if summary.get("suggestion"):
                text_parts.append("SUGGESTED REMEDIATION")
                text_parts.append("---------------------")
                text_parts.append(summary.get("suggestion"))
                
            return "\n".join(text_parts)
            
    def get_remediation_suggestions(self, violations, context=None):
        """
        Get specific remediation suggestions for the given violations.
        
        Args:
            violations: List of compliance violations
            context: Optional context information
            
        Returns:
            Dict with remediation suggestions
        """
        if not violations:
            return {"suggestions": []}
            
        violations_by_category = self._group_by_category(violations)
        suggestions = []
        
        # Generate category-specific suggestions
        for category, category_violations in violations_by_category.items():
            if category in self.remediation_templates:
                template = self.remediation_templates[category]
                
                # Extract all unique rule IDs in this category
                rule_ids = set(v.get("rule_id", "") for v in category_violations)
                rule_names = [self.rule_metadata.get(rule_id, {}).get("name", "Unknown rule") for rule_id in rule_ids if rule_id]
                
                # Format the suggestion
                suggestion = template.format(
                    count=len(category_violations),
                    rules=", ".join(rule_names),
                    category=category
                )
                
                suggestions.append({
                    "category": category,
                    "suggestion": suggestion,
                    "priority": self._get_category_priority(category),
                    "violation_count": len(category_violations)
                })
                
        # Generate rule-specific suggestions
        rule_violations = defaultdict(list)
        for violation in violations:
            rule_id = violation.get("rule_id", "")
            if rule_id:
                rule_violations[rule_id].append(violation)
                
        for rule_id, rule_violations_list in rule_violations.items():
            if rule_id in self.rule_metadata and "remediation_template" in self.rule_metadata[rule_id]:
                template = self.rule_metadata[rule_id]["remediation_template"]
                rule_name = self.rule_metadata[rule_id].get("name", "Unknown rule")
                
                # Extract example content from violations
                example_texts = []
                for v in rule_violations_list[:3]:  # Take up to 3 examples
                    if "metadata" in v and "matched_content" in v["metadata"]:
                        example_texts.append(v["metadata"]["matched_content"])
                        
                # Format the suggestion
                suggestion = template.format(
                    count=len(rule_violations_list),
                    rule=rule_name,
                    examples=", ".join(f'"{text}"' for text in example_texts) if example_texts else "N/A"
                )
                
                suggestions.append({
                    "rule_id": rule_id,
                    "suggestion": suggestion,
                    "priority": self._get_rule_priority(rule_id),
                    "violation_count": len(rule_violations_list)
                })
                
        # Sort suggestions by priority, then by violation count
        suggestions.sort(key=lambda x: (-x["priority"], -x["violation_count"]))
        
        return {"suggestions": suggestions}
        
    def assess_violation_severity(self, violations):
        """
        Assess the overall severity of the given violations.
        
        Args:
            violations: List of compliance violations
            
        Returns:
            Dict with severity assessment
        """
        if not violations:
            return {
                "overall_severity": "none",
                "severity_score": 0,
                "violation_count": 0
            }
            
        # Count violations by severity
        severity_counts = Counter(v.get("severity", "medium") for v in violations)
        
        # Calculate weighted severity score
        severity_score = sum(
            count * self.severity_weights.get(severity, 1)
            for severity, count in severity_counts.items()
        )
        
        # Determine overall severity based on thresholds
        overall_severity = "none"
        for severity, threshold in sorted(self.violation_thresholds.items(), key=lambda x: -self.severity_weights.get(x[0], 0)):
            if severity_counts.get(severity, 0) >= threshold:
                overall_severity = severity
                break
                
        return {
            "overall_severity": overall_severity,
            "severity_score": severity_score,
            "severity_counts": dict(severity_counts),
            "violation_count": len(violations)
        }
        
    def _load_rule_metadata(self, rule_metadata):
        """Load and process rule metadata."""
        processed_metadata = {}
        
        for rule_id, metadata in rule_metadata.items():
            processed_metadata[rule_id] = {
                "name": metadata.get("name", f"Rule {rule_id}"),
                "category": metadata.get("category", "general"),
                "description": metadata.get("description", ""),
                "severity": metadata.get("severity", "medium"),
                "remediation_template": metadata.get("remediation_template", ""),
                "priority": metadata.get("priority", 0),
                "tags": metadata.get("tags", [])
            }
            
        return processed_metadata
        
    def _initialize_pattern_matchers(self):
        """Initialize pattern matching functions."""
        return {
            "repeated_violations": self._match_repeated_violations,
            "sequential_violations": self._match_sequential_violations,
            "contextual_patterns": self._match_contextual_patterns
        }
        
    def _group_by_severity(self, violations):
        """Group violations by severity level."""
        result = defaultdict(list)
        
        for violation in violations:
            severity = violation.get("severity", "medium")
            result[severity].append(violation)
            
        return result
        
    def _group_by_category(self, violations):
        """Group violations by category."""
        result = defaultdict(list)
        
        for violation in violations:
            rule_id = violation.get("rule_id", "")
            
            # Look up category from rule metadata
            if rule_id in self.rule_metadata:
                category = self.rule_metadata[rule_id].get("category", "general")
            else:
                # Try to determine category from rule_id
                category = self._infer_category_from_rule_id(rule_id)
                
            result[category].append(violation)
            
        return result
        
    def _infer_category_from_rule_id(self, rule_id):
        """Infer violation category from rule ID if not in metadata."""
        # Check explicit mapping first
        if rule_id in self.category_mapping:
            return self.category_mapping[rule_id]
            
        # Try to infer from rule ID prefix
        for prefix, category in self.category_mapping.items():
            if rule_id.startswith(prefix):
                return category
                
        # Default category
        return "general"
        
    def _identify_top_rules(self, violations):
        """Identify the most frequently triggered rules."""
        rule_counts = Counter(v.get("rule_id", "") for v in violations if "rule_id" in v)
        
        top_rules = []
        for rule_id, count in rule_counts.most_common():
            if not rule_id:
                continue
                
            rule_info = {
                "rule_id": rule_id,
                "count": count
            }
            
            # Add metadata if available
            if rule_id in self.rule_metadata:
                rule_info.update({
                    "name": self.rule_metadata[rule_id].get("name", ""),
                    "category": self.rule_metadata[rule_id].get("category", ""),
                    "severity": self.rule_metadata[rule_id].get("severity", "")
                })
                
            top_rules.append(rule_info)
            
        return top_rules
        
    def _determine_primary_severity(self, violations_by_severity):
        """Determine the primary severity level of violations."""
        # Order of severity from highest to lowest
        severity_order = ["critical", "high", "medium", "low"]
        
        for severity in severity_order:
            if severity in violations_by_severity and violations_by_severity[severity]:
                return severity
                
        return "low"  # Default
        
    def _generate_remediation_suggestion(self, violations, context, input_text):
        """Generate an overall remediation suggestion based on violations."""
        if not violations:
            return None
            
        # Get detailed suggestions
        suggestions_result = self.get_remediation_suggestions(violations, context)
        suggestions = suggestions_result.get("suggestions", [])
        
        if not suggestions:
            # Fallback general suggestion
            return "Review and modify content to address compliance issues."
            
        # Take the highest priority suggestion
        top_suggestion = suggestions[0]["suggestion"]
        
        # If there are multiple high-priority suggestions, combine them
        if len(suggestions) > 1 and suggestions[1]["priority"] == suggestions[0]["priority"]:
            top_suggestions = [s["suggestion"] for s in suggestions[:3] if s["priority"] == suggestions[0]["priority"]]
            return " ".join(top_suggestions)
            
        return top_suggestion
        
    def _update_historical_data(self, violations, context):
        """Update historical violation data for trend analysis."""
        # In a real system, this might store data in a database
        timestamp = time.time()
        
        # Group violations by rule ID
        for violation in violations:
            rule_id = violation.get("rule_id", "")
            if rule_id:
                self.historical_violations[rule_id].append({
                    "timestamp": timestamp,
                    "severity": violation.get("severity", "medium"),
                    "context_info": self._extract_context_summary(context)
                })
                
        # Keep only recent history (last 100 entries per rule)
        for rule_id in self.historical_violations:
            if len(self.historical_violations[rule_id]) > 100:
                self.historical_violations[rule_id] = self.historical_violations[rule_id][-100:]
                
        # Update violation trends
        self._update_violation_trends()
        
    def _extract_context_summary(self, context):
        """Extract a summary of context information for historical tracking."""
        if not context:
            return {}
            
        # Extract relevant context info, avoiding storing sensitive data
        summary = {}
        
        if "domain" in context:
            summary["domain"] = context["domain"]
            
        if "user_info" in context:
            # Only store non-sensitive user info
            user_summary = {}
            user_info = context["user_info"]
            
            safe_user_fields = ["role", "access_level", "account_type"]
            for field in safe_user_fields:
                if field in user_info:
                    user_summary[field] = user_info[field]
                    
            summary["user_info"] = user_summary
            
        if "metadata" in context:
            # Only store select metadata
            metadata = context["metadata"]
            metadata_summary = {}
            
            safe_metadata_fields = ["source", "channel", "request_type"]
            for field in safe_metadata_fields:
                if field in metadata:
                    metadata_summary[field] = metadata[field]
                    
            summary["metadata"] = metadata_summary
            
        return summary
        
    def _update_violation_trends(self):
        """Update violation trend analysis."""
        # Calculate current violation distribution
        rule_counts = {
            rule_id: len(violations) 
            for rule_id, violations in self.historical_violations.items()
            if violations  # Only rules with violations
        }
        
        timestamp = time.time()
        
        # Add current snapshot to trends
        self.violation_trends["rule_distribution"].append({
            "timestamp": timestamp,
            "distribution": rule_counts
        })
        
        # Keep only recent trends (last 100 entries)
        if len(self.violation_trends["rule_distribution"]) > 100:
            self.violation_trends["rule_distribution"] = self.violation_trends["rule_distribution"][-100:]
            
    def _detect_violation_patterns(self, violations, context):
        """Detect patterns in violation data."""
        patterns = {}
        
        for pattern_type, matcher in self.pattern_matchers.items():
            pattern_results = matcher(violations, context)
            if pattern_results:
                patterns[pattern_type] = pattern_results
                
        return patterns
        
    def _match_repeated_violations(self, violations, context):
        """Match pattern: repeated violations of the same rule."""
        rule_counts = Counter(v.get("rule_id", "") for v in violations if "rule_id" in v)
        
        repeated_violations = [
            {
                "rule_id": rule_id,
                "count": count,
                "name": self.rule_metadata.get(rule_id, {}).get("name", "Unknown rule"),
                "pattern_type": "repeated_violation"
            }
            for rule_id, count in rule_counts.items()
            if count >= self.min_violations_for_pattern and rule_id
        ]
        
        return repeated_violations if repeated_violations else None
        
    def _match_sequential_violations(self, violations, context):
        """Match pattern: violations that follow a sequential pattern."""
        # This is a placeholder for more sophisticated sequential pattern detection
        # In a real system, this would analyze sequences of violations
        return None
        
    def _match_contextual_patterns(self, violations, context):
        """Match pattern: violations that occur in specific contexts."""
        # This is a placeholder for more sophisticated contextual pattern detection
        # In a real system, this would analyze patterns related to specific contexts
        return None
        
    def _assess_impact(self, violations, context):
        """Assess the potential impact of the violations."""
        severity_assessment = self.assess_violation_severity(violations)
        
        # Base impact on severity
        impact_level = severity_assessment["overall_severity"]
        
        # Consider context to adjust impact assessment
        if context:
            domain = context.get("domain", "general")
            
            # In some domains, certain violations have higher impact
            domain_sensitivities = self.config.get("domain_sensitivities", {}).get(domain, {})
            
            for violation in violations:
                rule_id = violation.get("rule_id", "")
                if rule_id in domain_sensitivities:
                    rule_impact = domain_sensitivities[rule_id]
                    # Escalate impact if domain-sensitive rule is violated
                    if rule_impact == "high" and impact_level in ["low", "medium"]:
                        impact_level = "high"
                    elif rule_impact == "critical" and impact_level != "critical":
                        impact_level = "critical"
        
        return {
            "impact_level": impact_level,
            "severity_assessment": severity_assessment
        }
        
    def _get_category_priority(self, category):
        """Get priority level for a violation category."""
        category_priorities = self.config.get("category_priorities", {})
        return category_priorities.get(category, 0)
        
    def _get_rule_priority(self, rule_id):
        """Get priority level for a rule."""
        if rule_id in self.rule_metadata:
            return self.rule_metadata[rule_id].get("priority", 0)
        return 0

