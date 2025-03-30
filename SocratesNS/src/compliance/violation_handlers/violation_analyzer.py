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

    def _match_sequential_violations(self, violations, context):
        """
        Match violations that follow a sequential pattern.
        
        Identifies patterns where violations occur in a specific order or sequence,
        which may indicate systematic issues or common violation paths.
        
        Args:
            violations: List of compliance violations
            context: Optional context information
            
        Returns:
            List of sequential violation patterns or None if none found
        """
        if len(violations) < 2:
            return None
            
        # Group violations by content or session if available
        grouped_violations = {}
        
        # First try to group by content
        if context and 'content_id' in context:
            content_id = context['content_id']
            grouped_violations[content_id] = violations
        # Then try by session
        elif context and 'session_id' in context:
            session_id = context['session_id']
            grouped_violations[session_id] = violations
        # Finally, just use a default group if no better grouping is available
        else:
            grouped_violations['default'] = violations
        
        sequential_patterns = []
        
        for group_id, group_violations in grouped_violations.items():
            # Sort violations by timestamp if available
            sorted_violations = sorted(
                group_violations,
                key=lambda v: v.get('timestamp', 0)
            )
            
            # Find sequence patterns using sliding window approach
            # Look for sequences of 2-4 violations
            for window_size in range(2, min(5, len(sorted_violations) + 1)):
                # Use frequency counting to identify repeated sequences
                sequence_counts = {}
                
                for i in range(len(sorted_violations) - window_size + 1):
                    # Create a sequence key using rule IDs
                    sequence = tuple(v.get('rule_id', str(i)) for v, i in 
                                zip(sorted_violations[i:i+window_size], range(window_size)))
                    
                    if sequence not in sequence_counts:
                        sequence_counts[sequence] = 0
                    sequence_counts[sequence] += 1
                
                # Filter sequences that occur multiple times
                for sequence, count in sequence_counts.items():
                    if count >= self.min_violations_for_pattern:
                        # Found a repeated sequence
                        rules = [rule_id for rule_id in sequence]
                        
                        # Get rule metadata for better description
                        rule_names = []
                        for rule_id in rules:
                            if rule_id in self.rule_metadata:
                                rule_names.append(self.rule_metadata[rule_id].get('name', rule_id))
                            else:
                                rule_names.append(f"Rule {rule_id}")
                        
                        sequential_patterns.append({
                            'rules': rules,
                            'rule_names': rule_names,
                            'count': count,
                            'group_id': group_id,
                            'pattern_type': 'sequential_violation',
                            'description': f"Sequential pattern: {' → '.join(rule_names)} (occurs {count} times)"
                        })
            
            # Look for time-based patterns - violations that consistently occur within timeframes
            if len(sorted_violations) >= 2 and all('timestamp' in v for v in sorted_violations):
                time_clusters = self._cluster_by_time_proximity(sorted_violations)
                
                for cluster in time_clusters:
                    if len(cluster) >= self.min_violations_for_pattern:
                        # Extract rule IDs and names
                        rules = [v.get('rule_id', 'unknown') for v in cluster]
                        rule_names = []
                        for rule_id in rules:
                            if rule_id in self.rule_metadata:
                                rule_names.append(self.rule_metadata[rule_id].get('name', rule_id))
                            else:
                                rule_names.append(f"Rule {rule_id}")
                        
                        sequential_patterns.append({
                            'rules': rules,
                            'rule_names': rule_names,
                            'count': len(cluster),
                            'group_id': group_id,
                            'pattern_type': 'time_proximity_pattern',
                            'description': f"Time-proximity pattern: {', '.join(rule_names)} occur together"
                        })
        
        return sequential_patterns if sequential_patterns else None

    def _cluster_by_time_proximity(self, violations, max_time_diff=60):
        """
        Helper function to cluster violations by time proximity.
        
        Args:
            violations: List of violations with timestamps
            max_time_diff: Maximum time difference (in seconds) to consider violations as clustered
            
        Returns:
            List of violation clusters
        """
        # Ensure violations are sorted by timestamp
        sorted_violations = sorted(violations, key=lambda v: v.get('timestamp', 0))
        
        clusters = []
        current_cluster = [sorted_violations[0]]
        
        for i in range(1, len(sorted_violations)):
            curr_time = sorted_violations[i].get('timestamp', 0)
            prev_time = sorted_violations[i-1].get('timestamp', 0)
            
            # Check if time difference is within threshold
            if curr_time - prev_time <= max_time_diff:
                # Add to current cluster
                current_cluster.append(sorted_violations[i])
            else:
                # Start a new cluster
                if len(current_cluster) >= 2:
                    clusters.append(current_cluster)
                current_cluster = [sorted_violations[i]]
        
        # Add the last cluster if it has at least 2 violations
        if len(current_cluster) >= 2:
            clusters.append(current_cluster)
            
        return clusters

    def _match_contextual_patterns(self, violations, context):
        """
        Match violations that occur in specific contexts.
        
        Identifies patterns related to specific contexts such as domain, user role,
        content type, etc., which can help identify context-specific compliance issues.
        
        Args:
            violations: List of compliance violations
            context: Optional context information
            
        Returns:
            List of contextual violation patterns or None if none found
        """
        if not context or not violations:
            return None
        
        contextual_patterns = []
        
        # Identify context-specific patterns
        
        # 1. Domain-specific patterns
        if 'domain' in context:
            domain = context['domain']
            domain_violations = self._analyze_domain_patterns(violations, domain)
            if domain_violations:
                contextual_patterns.extend(domain_violations)
        
        # 2. User role patterns
        if 'user_info' in context and 'role' in context['user_info']:
            user_role = context['user_info']['role']
            role_violations = self._analyze_role_patterns(violations, user_role)
            if role_violations:
                contextual_patterns.extend(role_violations)
        
        # 3. Content type patterns
        if 'content_type' in context:
            content_type = context['content_type']
            content_type_violations = self._analyze_content_type_patterns(violations, content_type)
            if content_type_violations:
                contextual_patterns.extend(content_type_violations)
        
        # 4. Device or platform patterns
        if 'platform' in context or 'device' in context:
            platform = context.get('platform', context.get('device'))
            platform_violations = self._analyze_platform_patterns(violations, platform)
            if platform_violations:
                contextual_patterns.extend(platform_violations)
        
        # 5. Time-based patterns (time of day, day of week)
        if 'timestamp' in context:
            import datetime
            try:
                dt = datetime.datetime.fromisoformat(context['timestamp'])
                time_patterns = self._analyze_time_patterns(violations, dt)
                if time_patterns:
                    contextual_patterns.extend(time_patterns)
            except (ValueError, TypeError):
                pass
        
        # 6. Location-based patterns
        if 'location' in context or 'country' in context or 'region' in context:
            location = context.get('location', context.get('country', context.get('region')))
            location_patterns = self._analyze_location_patterns(violations, location)
            if location_patterns:
                contextual_patterns.extend(location_patterns)
        
        # 7. Multi-dimensional context patterns (combinations of contexts)
        multi_context_patterns = self._analyze_multi_context_patterns(violations, context)
        if multi_context_patterns:
            contextual_patterns.extend(multi_context_patterns)
        
        return contextual_patterns if contextual_patterns else None

    def _analyze_domain_patterns(self, violations, domain):
        """Analyze domain-specific violation patterns"""
        # Group violations by rule ID
        rule_violations = {}
        for violation in violations:
            rule_id = violation.get('rule_id', 'unknown')
            if rule_id not in rule_violations:
                rule_violations[rule_id] = []
            rule_violations[rule_id].append(violation)
        
        # Check if any rules have a high frequency in this domain
        domain_patterns = []
        
        for rule_id, rule_viols in rule_violations.items():
            if len(rule_viols) >= self.min_violations_for_pattern:
                # Check if this rule has domain-specific sensitivity
                domain_sensitivity = self.config.get('domain_sensitivities', {}).get(domain, {}).get(rule_id)
                
                if domain_sensitivity:
                    # This rule has specific domain sensitivity
                    domain_patterns.append({
                        'rule_id': rule_id,
                        'count': len(rule_viols),
                        'domain': domain,
                        'pattern_type': 'domain_specific_pattern',
                        'name': self.rule_metadata.get(rule_id, {}).get('name', f"Rule {rule_id}"),
                        'description': f"Domain-specific pattern: {rule_id} frequently violated in {domain} domain"
                    })
        
        return domain_patterns

    def _analyze_role_patterns(self, violations, user_role):
        """Analyze user role specific violation patterns"""
        # Similar implementation to domain patterns but for user roles
        # Group violations by rule ID
        rule_violations = {}
        for violation in violations:
            rule_id = violation.get('rule_id', 'unknown')
            if rule_id not in rule_violations:
                rule_violations[rule_id] = []
            rule_violations[rule_id].append(violation)
        
        # Check if any rules have a high frequency for this user role
        role_patterns = []
        
        for rule_id, rule_viols in rule_violations.items():
            if len(rule_viols) >= self.min_violations_for_pattern:
                # Check if this rule has role-specific sensitivity
                role_sensitivity = self.config.get('role_sensitivities', {}).get(user_role, {}).get(rule_id)
                
                if role_sensitivity:
                    # This rule has specific role sensitivity
                    role_patterns.append({
                        'rule_id': rule_id,
                        'count': len(rule_viols),
                        'user_role': user_role,
                        'pattern_type': 'role_specific_pattern',
                        'name': self.rule_metadata.get(rule_id, {}).get('name', f"Rule {rule_id}"),
                        'description': f"Role-specific pattern: {rule_id} frequently violated by {user_role} role"
                    })
        
        return role_patterns

    def _analyze_content_type_patterns(self, violations, content_type):
        """Analyze content type specific violation patterns"""
        # Implementation for content type patterns
        # Follow similar pattern as domain patterns
        return []  # Placeholder - implement similar to domain patterns

    def _analyze_platform_patterns(self, violations, platform):
        """Analyze platform-specific violation patterns"""
        # Implementation for platform patterns
        # Follow similar pattern as domain patterns
        return []  # Placeholder - implement similar to domain patterns

    def _analyze_time_patterns(self, violations, datetime_obj):
        """Analyze time-based violation patterns"""
        # Implementation for time-based patterns
        # Look for patterns related to time of day, day of week, etc.
        return []  # Placeholder - implement similar to domain patterns

    def _analyze_location_patterns(self, violations, location):
        """Analyze location-based violation patterns"""
        # Implementation for location patterns
        # Follow similar pattern as domain patterns
        return []  # Placeholder - implement similar to domain patterns

    def _analyze_multi_context_patterns(self, violations, context):
        """Analyze patterns involving multiple context dimensions"""
        # Implementation for multi-dimensional context patterns
        # Look for patterns that involve combinations of contexts
        return []  # Placeholder - implement more sophisticated analysis

    def _identify_new_exemptions(self, violations, original_input, context):
        """
        Identify patterns that might qualify for compliance exemptions.
        
        Analyzes patterns to identify potential new exemptions to rules, which can
        help reduce false positives and improve rule precision over time.
        
        Args:
            violations: List of compliance violations
            original_input: Original input content
            context: Optional context information
            
        Returns:
            List of potential exemption patterns
        """
        potential_exemptions = []
        
        # Skip if no violations or input
        if not violations or not original_input:
            return potential_exemptions
        
        # 1. Educational context exemptions
        if context and context.get('purpose') == 'educational':
            educational_exemptions = self._identify_educational_exemptions(violations, original_input)
            potential_exemptions.extend(educational_exemptions)
        
        # 2. Scientific/research context exemptions
        if context and context.get('purpose') in ['research', 'scientific']:
            research_exemptions = self._identify_research_exemptions(violations, original_input)
            potential_exemptions.extend(research_exemptions)
        
        # 3. Quoted content exemptions
        quoted_exemptions = self._identify_quoted_content_exemptions(violations, original_input)
        potential_exemptions.extend(quoted_exemptions)
        
        # 4. Legal/compliance discussion exemptions
        legal_exemptions = self._identify_legal_discussion_exemptions(violations, original_input)
        potential_exemptions.extend(legal_exemptions)
        
        # 5. Statistical/aggregate data exemptions
        statistical_exemptions = self._identify_statistical_exemptions(violations, original_input)
        potential_exemptions.extend(statistical_exemptions)
        
        # 6. Fictional/creative content exemptions
        if context and context.get('content_type') in ['fiction', 'creative', 'artistic']:
            fictional_exemptions = self._identify_fictional_exemptions(violations, original_input)
            potential_exemptions.extend(fictional_exemptions)
        
        # 7. Historical content exemptions
        historical_exemptions = self._identify_historical_exemptions(violations, original_input)
        potential_exemptions.extend(historical_exemptions)
        
        # 8. Redacted/anonymized content exemptions
        anonymized_exemptions = self._identify_anonymized_exemptions(violations, original_input)
        potential_exemptions.extend(anonymized_exemptions)
        
        # 9. Rule-specific contextual exemptions
        for violation in violations:
            rule_id = violation.get('rule_id')
            if not rule_id:
                continue
                
            rule_exemptions = self._identify_rule_specific_exemptions(rule_id, violation, original_input, context)
            if rule_exemptions:
                potential_exemptions.extend(rule_exemptions)
        
        return potential_exemptions

    def _identify_educational_exemptions(self, violations, original_input):
        """Identify exemption patterns in educational contexts"""
        exemptions = []
        
        # Check for educational indicators
        educational_indicators = [
            r'for\s+educational\s+purposes',
            r'in\s+an?\s+educational\s+context',
            r'for\s+teaching\s+purposes',
            r'as\s+a\s+learning\s+example',
            r'for\s+instructional\s+purposes'
        ]
        
        import re
        for indicator in educational_indicators:
            if re.search(indicator, original_input, re.IGNORECASE):
                # Found educational context indicator
                for violation in violations:
                    rule_id = violation.get('rule_id')
                    if not rule_id:
                        continue
                    
                    # Check if this rule can have educational exemptions
                    if self._rule_allows_educational_exemption(rule_id):
                        exemptions.append({
                            'rule_id': rule_id,
                            'pattern': indicator,
                            'match': re.search(indicator, original_input, re.IGNORECASE).group(0),
                            'exemption_type': 'educational_context',
                            'description': f"Educational context exemption for rule {rule_id}",
                            'confidence': 0.8
                        })
        
        return exemptions

    def _rule_allows_educational_exemption(self, rule_id):
        """Check if a rule allows educational exemptions"""
        # This would check rule metadata to see if educational exemptions are allowed
        # For now, default to True for most rules except highly sensitive ones
        high_risk_rules = ['pii_disclosure', 'phi_disclosure', 'financial_account_numbers']
        return rule_id not in high_risk_rules

    def _identify_research_exemptions(self, violations, original_input):
        """Identify exemption patterns in research contexts"""
        # Similar implementation to educational exemptions
        return []  # Placeholder - implement similar to educational exemptions

    def _identify_quoted_content_exemptions(self, violations, original_input):
        """Identify exemption patterns for quoted content"""
        exemptions = []
        
        # Check for quotation patterns
        import re
        
        # Find all quoted segments
        quote_patterns = [
            r'"([^"]+)"',           # Double quotes
            r"'([^']+)'",           # Single quotes
            r'\["""([^"""]+)\]"""', # Smart quotes
            r"[«]([^»])+[»]",       # Guillemets
            r"[「]([^」])+[」]"      # CJK quotes
        ]
        
        for pattern in quote_patterns:
            for match in re.finditer(pattern, original_input):
                quoted_text = match.group(1)
                quote_span = (match.start(), match.end())
                
                # Check if any violations occur within this quote
                for violation in violations:
                    if 'location' in violation:
                        viol_start = violation['location'].get('start', 0)
                        viol_end = violation['location'].get('end', 0)
                        
                        # Check if violation is contained within quote
                        if quote_span[0] <= viol_start and viol_end <= quote_span[1]:
                            rule_id = violation.get('rule_id', 'unknown')
                            
                            # Check if this rule allows quote exemptions
                            if self._rule_allows_quote_exemption(rule_id):
                                exemptions.append({
                                    'rule_id': rule_id,
                                    'pattern': pattern,
                                    'match': quoted_text,
                                    'exemption_type': 'quoted_content',
                                    'description': f"Quoted content exemption for rule {rule_id}",
                                    'confidence': 0.9,
                                    'location': {'start': quote_span[0], 'end': quote_span[1]}
                                })
        
        return exemptions

    def _rule_allows_quote_exemption(self, rule_id):
        """Check if a rule allows quote exemptions"""
        # Similar to educational exemptions, check rule metadata
        return True  # Placeholder - implement rule-specific logic

    def _identify_legal_discussion_exemptions(self, violations, original_input):
        """Identify exemption patterns for legal/compliance discussions"""
        # Implementation for legal discussion exemptions
        return []  # Placeholder - implement similar to quoted content

    def _identify_statistical_exemptions(self, violations, original_input):
        """Identify exemption patterns for statistical/aggregate data"""
        # Implementation for statistical exemptions
        return []  # Placeholder - implement similar to quoted content

    def _identify_fictional_exemptions(self, violations, original_input):
        """Identify exemption patterns for fictional content"""
        # Implementation for fictional content exemptions
        return []  # Placeholder - implement similar to quoted content

    def _identify_historical_exemptions(self, violations, original_input):
        """Identify exemption patterns for historical content"""
        # Implementation for historical content exemptions
        return []  # Placeholder - implement similar to quoted content

    def _identify_anonymized_exemptions(self, violations, original_input):
        """Identify exemption patterns for anonymized content"""
        # Implementation for anonymized content exemptions
        return []  # Placeholder - implement similar to quoted content

    def _identify_rule_specific_exemptions(self, rule_id, violation, original_input, context):
        """Identify rule-specific exemption patterns"""
        # This would contain rule-specific logic for identifying exemptions
        # Different rules may have different kinds of valid exemptions
        
        # Check rule metadata for exemption patterns
        rule_meta = self.rule_metadata.get(rule_id, {})
        exemption_patterns = rule_meta.get('exemption_patterns', [])
        
        exemptions = []
        
        import re
        for pattern in exemption_patterns:
            pattern_regex = pattern.get('regex')
            if pattern_regex and re.search(pattern_regex, original_input, re.IGNORECASE):
                exemptions.append({
                    'rule_id': rule_id,
                    'pattern': pattern_regex,
                    'match': re.search(pattern_regex, original_input, re.IGNORECASE).group(0),
                    'exemption_type': pattern.get('type', 'rule_specific'),
                    'description': pattern.get('description', f"Rule-specific exemption for {rule_id}"),
                    'confidence': pattern.get('confidence', 0.7)
                })
        
        return exemptions

    def _generate_rule_specific_alternatives(self, text, rule_id, rule_issues, context):
        """
        Generate alternative formulations specific to a rule violation.
        
        Uses sophisticated techniques like synonym replacement, restructuring,
        or embeddings-based reformulation to address specific rule violations.
        
        Args:
            text: Original text content
            rule_id: Rule identifier
            rule_issues: Issues related to this rule
            context: Optional context information
            
        Returns:
            List of alternative formulations
        """
        alternatives = []
        
        # Get rule metadata
        rule_meta = self.rule_metadata.get(rule_id, {})
        rule_type = rule_meta.get('type', 'unknown')
        rule_name = rule_meta.get('name', f"Rule {rule_id}")
        
        # Different strategies based on rule type
        if rule_type == 'prohibited_term':
            # Replace prohibited terms with alternatives
            alternatives.extend(self._generate_term_replacement_alternatives(text, rule_id, rule_issues, context))
        
        elif rule_type == 'sensitive_data':
            # Anonymize or redact sensitive data
            alternatives.extend(self._generate_anonymization_alternatives(text, rule_id, rule_issues, context))
        
        elif rule_type == 'data_minimization':
            # Reduce unnecessary data
            alternatives.extend(self._generate_data_minimization_alternatives(text, rule_id, rule_issues, context))
        
        elif rule_type == 'disclaimer_required':
            # Add required disclaimers
            alternatives.extend(self._generate_disclaimer_alternatives(text, rule_id, rule_issues, context))
        
        elif rule_type == 'biased_language':
            # Replace biased language
            alternatives.extend(self._generate_bias_correction_alternatives(text, rule_id, rule_issues, context))
        
        # Additional rule-specific alternatives using embeddings if available
        embedding_alternatives = self._generate_embedding_based_alternatives(text, rule_id, rule_issues, context)
        alternatives.extend(embedding_alternatives)
        
        # Try transformer-based alternatives if available
        transformer_alternatives = self._generate_transformer_based_alternatives(text, rule_id, rule_issues, context)
        alternatives.extend(transformer_alternatives)
        
        return alternatives

    def _generate_term_replacement_alternatives(self, text, rule_id, rule_issues, context):
        """Generate alternatives by replacing prohibited terms with safer alternatives"""
        alternatives = []
        
        # Get prohibited terms for this rule
        rule_meta = self.rule_metadata.get(rule_id, {})
        prohibited_terms = rule_meta.get('prohibited_terms', [])
        
        # Get alternative terms for each prohibited term
        term_alternatives = rule_meta.get('term_alternatives', {})
        
        for issue in rule_issues:
            # Get the problematic text
            if 'metadata' in issue and 'matched_content' in issue['metadata']:
                matched_text = issue['metadata']['matched_content']
                
                # Find this text in the original content
                import re
                matches = list(re.finditer(re.escape(matched_text), text, re.IGNORECASE))
                
                for match in matches:
                    start, end = match.span()
                    
                    # Try term-specific alternatives
                    for term in prohibited_terms:
                        if term.lower() in matched_text.lower():
                            # Found a prohibited term, generate alternatives
                            
                            # Method 1: Use predefined alternatives if available
                            if term in term_alternatives:
                                for alt_term in term_alternatives[term]:
                                    # Replace the term
                                    alternative_text = text[:start] + text[start:end].replace(term, alt_term) + text[end:]
                                    
                                    alternatives.append({
                                        'text': alternative_text,
                                        'rule_id': rule_id,
                                        'confidence': 0.9,
                                        'type': 'term_replacement',
                                        'description': f"Replaced prohibited term '{term}' with '{alt_term}'"
                                    })
                            
                            # Method 2: Use thesaurus for synonyms
                            synonyms = self._get_synonyms(term)
                            for synonym in synonyms[:3]:  # Limit to top 3 synonyms
                                # Replace the term
                                alternative_text = text[:start] + text[start:end].replace(term, synonym) + text[end:]
                                
                                alternatives.append({
                                    'text': alternative_text,
                                    'rule_id': rule_id,
                                    'confidence': 0.7,
                                    'type': 'synonym_replacement',
                                    'description': f"Replaced prohibited term '{term}' with synonym '{synonym}'"
                                })
        
        return alternatives

    def _get_synonyms(self, term):
        """Get synonyms for a term using thesaurus or embeddings"""
        # This would use a thesaurus API or embedding model in a real implementation
        # For now, return some examples
        
        # Try using NLTK WordNet if available
        try:
            from nltk.corpus import wordnet
            synonyms = []
            
            # Get synonyms from WordNet
            for syn in wordnet.synsets(term):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym != term and synonym not in synonyms:
                        synonyms.append(synonym)
            
            return synonyms[:5]  # Return top 5 synonyms
        except:
            # Fallback to a simple dictionary for common terms
            synonym_dict = {
                'issue': ['problem', 'concern', 'matter'],
                'bad': ['poor', 'suboptimal', 'concerning'],
                'good': ['positive', 'beneficial', 'favorable'],
                'customer': ['client', 'user', 'consumer'],
                'data': ['information', 'details', 'records'],
                'money': ['funds', 'financial resources', 'capital'],
                'problem': ['issue', 'challenge', 'difficulty'],
                'sensitive': ['confidential', 'private', 'protected']
            }
            
            return synonym_dict.get(term.lower(), [])

    def _generate_anonymization_alternatives(self, text, rule_id, rule_issues, context):
        """Generate alternatives by anonymizing sensitive data"""
        alternatives = []
        
        for issue in rule_issues:
            # Get the sensitive data
            if 'metadata' in issue and 'matched_content' in issue['metadata']:
                sensitive_text = issue['metadata']['matched_content']
                
                # Find this text in the original content
                import re
                matches = list(re.finditer(re.escape(sensitive_text), text, re.IGNORECASE))
                
                for match in matches:
                    start, end = match.span()
                    
                    # Method 1: Simple redaction
                    redacted_text = text[:start] + "[REDACTED]" + text[end:]
                    alternatives.append({
                        'text': redacted_text,
                        'rule_id': rule_id,
                        'confidence': 0.9,
                        'type': 'redaction',
                        'description': f"Redacted sensitive data"
                    })
                    
                    # Method 2: Type-specific anonymization
                    # Determine the type of sensitive data
                    data_type = self._determine_sensitive_data_type(sensitive_text)
                    
                    if data_type == 'email':
                        # For emails, keep domain but anonymize local part
                        if '@' in sensitive_text:
                            local, domain = sensitive_text.split('@', 1)
                            anonymized = f"[email]@{domain}"
                            alternative_text = text[:start] + anonymized + text[end:]
                            
                            alternatives.append({
                                'text': alternative_text,
                                'rule_id': rule_id,
                                'confidence': 0.9,
                                'type': 'email_anonymization',
                                'description': f"Anonymized email address"
                            })
                    
                    elif data_type == 'phone':
                        # For phone numbers, keep area code but anonymize the rest
                        if len(sensitive_text) >= 10:
                            # Try to extract area code (first 3 digits for US numbers)
                            digits = ''.join(c for c in sensitive_text if c.isdigit())
                            if len(digits) >= 10:
                                area_code = digits[:3]
                                anonymized = f"({area_code}) XXX-XXXX"
                                alternative_text = text[:start] + anonymized + text[end:]
                                
                                alternatives.append({
                                    'text': alternative_text,
                                    'rule_id': rule_id,
                                    'confidence': 0.8,
                                    'type': 'phone_anonymization',
                                    'description': f"Anonymized phone number"
                                })
                    
                    elif data_type == 'name':
                        # For names, replace with placeholder
                        anonymized = "[Person's Name]"
                        alternative_text = text[:start] + anonymized + text[end:]
                        
                        alternatives.append({
                            'text': alternative_text,
                            'rule_id': rule_id,
                            'confidence': 0.7,
                            'type': 'name_anonymization',
                            'description': f"Anonymized personal name"
                        })
                    
                    # Method 3: Pseudonymization (fake but realistic data)
                    pseudonymized_text = self._generate_pseudonym(sensitive_text, data_type)
                    if pseudonymized_text:
                        pseudo_text = text[:start] + pseudonymized_text + text[end:]
                        
                        alternatives.append({
                            'text': pseudo_text,
                            'rule_id': rule_id,
                            'confidence': 0.7,
                            'type': 'pseudonymization',
                            'description': f"Replaced sensitive data with pseudonym"
                        })
        
        return alternatives

    def _determine_sensitive_data_type(self, text):
        """Determine the type of sensitive data"""
        import re
        
        # Check for email pattern
        if re.match(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text):
            return 'email'
        
        # Check for phone pattern
        if re.match(r'\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', text):
            return 'phone'
        
        # Check for SSN pattern
        if re.match(r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b', text):
            return 'ssn'
        
        # Check for credit card pattern
        if re.match(r'\b(?:\d{4}[-.\s]?){3}\d{4}\b', text):
            return 'credit_card'
        
        # Check for name pattern (capitalized words)
        if re.match(r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)+\b', text):
            return 'name'
        
        # Default to generic
        return 'generic'

    def _generate_pseudonym(self, text, data_type):
        """Generate a pseudonym for the sensitive data"""
        # This would use a more sophisticated approach in a real implementation
        
        if data_type == 'email':
            return "john.doe@example.com"
        elif data_type == 'phone':
            return "(555) 123-4567"
        elif data_type == 'name':
            return "Jane Smith"
        elif data_type == 'ssn':
            return "123-45-6789"
        elif data_type == 'credit_card':
            return "1234-5678-9012-3456"
        else:
            return "EXAMPLE_DATA"

    def _generate_data_minimization_alternatives(self, text, rule_id, rule_issues, context):
        """Generate alternatives for data minimization principles"""
        # Placeholder - would implement data reduction strategies
        return []

    def _generate_disclaimer_alternatives(self, text, rule_id, rule_issues, context):
        """Generate alternatives that add required disclaimers"""
        # Placeholder - would implement disclaimer addition
        return []

    def _generate_bias_correction_alternatives(self, text, rule_id, rule_issues, context):
        """Generate alternatives that correct biased language"""
        # Placeholder - would implement bias correction
        return []

    def _generate_embedding_based_alternatives(self, text, rule_id, rule_issues, context):
        """Generate alternatives using embedding-based similarity"""
        # This would use embeddings to find semantically similar alternatives
        # Requires a sentence embedding model
        
        alternatives = []
        
        # Try using sentence-transformers if available
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')  # or another suitable model
            
            for issue in rule_issues:
                # Get the problematic text
                if 'metadata' in issue and 'matched_content' in issue['metadata']:
                    matched_text = issue['metadata']['matched_content']
                    
                    # Find this text in the original content
                    import re
                    matches = list(re.finditer(re.escape(matched_text), text, re.IGNORECASE))
                    
                    if matches:
                        # Get alternative segments from a precomputed set
                        # In a real implementation, these would be generated dynamically
                        alternative_segments = self._get_compliant_alternatives_for_rule(rule_id)
                        
                        if alternative_segments:
                            # Encode the matched text
                            matched_embedding = model.encode(matched_text)
                            
                            # Encode all alternatives
                            alternative_embeddings = model.encode(alternative_segments)
                            
                            # Calculate similarities
                            similarities = []
                            for i, alt_embedding in enumerate(alternative_embeddings):
                                similarity = self._cosine_similarity(matched_embedding, alt_embedding)
                                similarities.append((i, similarity))
                            
                            # Sort by similarity
                            similarities.sort(key=lambda x: x[1], reverse=True)
                            
                            # Take top 3 most similar alternatives
                            for i, similarity in similarities[:3]:
                                alternative_segment = alternative_segments[i]
                                
                                # Replace in the original text
                                for match in matches:
                                    start, end = match.span()
                                    alternative_text = text[:start] + alternative_segment + text[end:]
                                    
                                    alternatives.append({
                                        'text': alternative_text,
                                        'rule_id': rule_id,
                                        'confidence': similarity * 0.9,  # Scale by similarity
                                        'type': 'embedding_based_replacement',
                                        'description': f"Replaced text with semantically similar compliant alternative"
                                    })
            
            return alternatives
        except:
            # Embedding model not available
            return []

    def _cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors"""
        import numpy as np
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def _get_compliant_alternatives_for_rule(self, rule_id):
        """Get a set of compliant alternatives for a specific rule"""
        # This would be populated from a database of pre-verified compliant alternatives
        # For now, return some examples
        
        alternatives_by_rule = {
            'prohibited_term_001': [
                "appropriate language",
                "suitable wording",
                "acceptable terminology"
            ],
            'bias_001': [
                "all people",
                "everyone",
                "all individuals"
            ],
            'compliance_001': [
                "in accordance with regulations",
                "following proper procedures",
                "in compliance with policies"
            ]
        }
        
        return alternatives_by_rule.get(rule_id, [])

    def _generate_transformer_based_alternatives(self, text, rule_id, rule_issues, context):
        """Generate alternatives using transformer models"""
        # This would use a language model to generate compliant alternatives
        # Requires a language model like GPT or T5
        
        alternatives = []
        
        # Try using transformers if available
        try:
            from transformers import pipeline
            
            # Initialize a text generation pipeline
            generator = pipeline('text2text-generation', model='t5-small')
            
            for issue in rule_issues:
                # Get the problematic text
                if 'metadata' in issue and 'matched_content' in issue['metadata']:
                    matched_text = issue['metadata']['matched_content']
                    
                    # Find this text in the original content
                    import re
                    matches = list(re.finditer(re.escape(matched_text), text, re.IGNORECASE))
                    
                    if matches:
                        # Create prompt for the model
                        rule_name = self.rule_metadata.get(rule_id, {}).get('name', f"Rule {rule_id}")
                        prompt = f"Rewrite the following text to comply with {rule_name}: {matched_text}"
                        
                        # Generate alternative
                        result = generator(prompt, max_length=100, num_return_sequences=1)
                        
                        if result and len(result) > 0:
                            alternative_segment = result[0]['generated_text']
                            
                            # Replace in the original text
                            for match in matches:
                                start, end = match.span()
                                alternative_text = text[:start] + alternative_segment + text[end:]
                                
                                alternatives.append({
                                    'text': alternative_text,
                                    'rule_id': rule_id,
                                    'confidence': 0.7,  # Lower confidence since it's automated
                                    'type': 'transformer_based_replacement',
                                    'description': f"Used language model to generate compliant alternative"
                                })
            
            return alternatives
        except:
            # Transformer model not available
            return []

    def _apply_general_alternatives(self, text, issues, context):
        """
        Apply general strategies for generating alternatives when rule-specific
        strategies are not available.
        
        Uses NLP techniques like text simplification, general rewording, or removal
        of problematic segments to address compliance issues.
        
        Args:
            text: Original text content
            issues: Compliance issues detected
            context: Optional context information
            
        Returns:
            List of alternative formulations
        """
        alternatives = []
        
        # 1. Remove problematic segments
        removal_alt = self._apply_segment_removal(text, issues)
        if removal_alt:
            alternatives.append(removal_alt)
        
        # 2. Apply hedging language
        hedging_alt = self._apply_hedging_language(text, issues)
        if hedging_alt:
            alternatives.append(hedging_alt)
        
        # 3. Text simplification
        simplification_alt = self._apply_text_simplification(text, issues)
        if simplification_alt:
            alternatives.append(simplification_alt)
        
        # 4. General paraphrasing
        paraphrase_alt = self._apply_paraphrasing(text, issues)
        if paraphrase_alt:
            alternatives.append(paraphrase_alt)
        
        # 5. Change tone (more formal/neutral)
        tone_alt = self._apply_tone_change(text, issues)
        if tone_alt:
            alternatives.append(tone_alt)
        
        # 6. Add clarifying context
        context_alt = self._add_clarifying_context(text, issues)
        if context_alt:
            alternatives.append(context_alt)
        
        # 7. Restructure sentences
        restructure_alt = self._restructure_sentences(text, issues)
        if restructure_alt:
            alternatives.append(restructure_alt)
        
        return alternatives

    def _apply_segment_removal(self, text, issues):
        """Apply strategy: Remove problematic segments"""
        # Identify segments to remove
        segments_to_remove = []
        
        for issue in issues:
            if 'metadata' in issue and 'location' in issue['metadata']:
                start = issue['metadata']['location'].get('start', -1)
                end = issue['metadata']['location'].get('end', -1)
                
                if start >= 0 and end > start and end <= len(text):
                    segments_to_remove.append((start, end))
        
        if not segments_to_remove:
            # Try to extract from matched_content if location not available
            for issue in issues:
                if 'metadata' in issue and 'matched_content' in issue['metadata']:
                    matched_content = issue['metadata']['matched_content']
                    
                    # Find in text
                    import re
                    for match in re.finditer(re.escape(matched_content), text):
                        segments_to_remove.append(match.span())
        
        if segments_to_remove:
            # Sort segments in reverse order so removal doesn't affect indices
            segments_to_remove.sort(reverse=True)
            
            # Apply removals
            modified_text = text
            for start, end in segments_to_remove:
                modified_text = modified_text[:start] + modified_text[end:]
            
            # Only return if text was actually modified
            if modified_text != text:
                return {
                    'text': modified_text,
                    'confidence': 0.7,
                    'type': 'segment_removal',
                    'description': f"Removed {len(segments_to_remove)} problematic segments"
                }
        
        return None

    def _apply_hedging_language(self, text, issues):
        """Apply strategy: Add hedging language to questionable statements"""
        # Hedging phrases to insert before problematic statements
        hedging_phrases = [
            "It is commonly suggested that ",
            "Some sources indicate that ",
            "According to certain perspectives, ",
            "It might be considered that ",
            "In some contexts, ",
            "From one point of view, "
        ]
        
        # Find problematic statements
        problematic_statements = []
        
        # Try to identify sentence boundaries around issues
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for issue in issues:
            if 'metadata' in issue and 'matched_content' in issue['metadata']:
                matched_content = issue['metadata']['matched_content']
                
                # Find which sentence contains this content
                for i, sentence in enumerate(sentences):
                    if matched_content in sentence:
                        problematic_statements.append((i, sentence))
                        break
        
        if problematic_statements:
            # Deduplicate
            problematic_statements = list(set(problematic_statements))
            
            # Apply hedging to sentences
            import random
            modified_sentences = sentences.copy()
            
            for idx, _ in problematic_statements:
                if idx < len(modified_sentences):
                    # Choose a random hedging phrase
                    hedge = random.choice(hedging_phrases)
                    
                    # Apply hedging to start of sentence if it doesn't already have hedging
                    if not any(h.lower() in modified_sentences[idx].lower() for h in hedging_phrases):
                        # Capitalize first letter after hedging
                        sentence = modified_sentences[idx]
                        if sentence and sentence[0].isupper():
                            sentence = sentence[0].lower() + sentence[1:]
                        modified_sentences[idx] = hedge + sentence
            
            # Reconstruct text
            modified_text = " ".join(modified_sentences)
            
            # Only return if text was actually modified
            if modified_text != text:
                return {
                    'text': modified_text,
                    'confidence': 0.6,
                    'type': 'hedging_language',
                    'description': f"Added hedging language to {len(problematic_statements)} statements"
                }
        
        return None

    def _apply_text_simplification(self, text, issues):
        """Apply strategy: Simplify text to reduce complexity and potential issues"""
        # Text simplification requires more sophisticated NLP
        # This is a placeholder that would use text simplification models
        
        # Try using a simple rule-based approach for demonstration
        complex_words = {
            'utilize': 'use',
            'implement': 'use',
            'leverage': 'use',
            'facilitate': 'help',
            'furthermore': 'also',
            'additionally': 'also',
            'consequently': 'so',
            'subsequently': 'later',
            'nevertheless': 'still',
            'accordingly': 'so',
            'furthermore': 'also',
            'notwithstanding': 'despite',
            'aforementioned': 'this',
            'heretofore': 'until now'
        }
        
        # Simplify by replacing complex words
        simplified_text = text
        for complex_word, simple_word in complex_words.items():
            # Use word boundaries to avoid partial replacements
            simplified_text = re.sub(
                r'\b' + re.escape(complex_word) + r'\b', 
                simple_word, 
                simplified_text, 
                flags=re.IGNORECASE
            )
        
        # Only return if text was actually modified
        if simplified_text != text:
            return {
                'text': simplified_text,
                'confidence': 0.5,
                'type': 'text_simplification',
                'description': "Simplified text by replacing complex words with simpler alternatives"
            }
        
        return None

    def _apply_paraphrasing(self, text, issues):
        """Apply strategy: Paraphrase content to address issues"""
        # Full paraphrasing requires advanced NLP models
        # This is a placeholder that would use paraphrasing models
        
        # In a real implementation, this would use a text-to-text model
        # to paraphrase the content while preserving meaning
        try:
            from transformers import pipeline
            generator = pipeline('text2text-generation', model='t5-small')
            
            # Only paraphrase smaller texts due to model limitations
            if len(text) <= 512:
                # Generate paraphrase
                result = generator(f"paraphrase: {text}", max_length=512, num_return_sequences=1)
                
                if result and len(result) > 0:
                    paraphrased_text = result[0]['generated_text']
                    
                    # Only return if text was meaningfully modified
                    if paraphrased_text != text and len(paraphrased_text) > len(text) * 0.5:
                        return {
                            'text': paraphrased_text,
                            'confidence': 0.5,
                            'type': 'paraphrasing',
                            'description': "Paraphrased content to address compliance issues"
                        }
        except:
            # Paraphrasing model not available
            pass
        
        return None

    def _apply_tone_change(self, text, issues):
        """Apply strategy: Change tone to be more formal/neutral"""
        # Tone change requires advanced NLP
        # This is a placeholder for a more sophisticated implementation
        
        # Return placeholder suggestion
        return {
            'text': "[Tone-adjusted version would be generated here - requires NLP model]",
            'confidence': 0.4,
            'type': 'tone_change',
            'description': "Suggested changing tone to be more formal and neutral"
        }

    def _add_clarifying_context(self, text, issues):
        """Apply strategy: Add clarifying context to address issues"""
        # This would add explanatory context based on the issues
        # For now, return a simple example
        
        # Create disclaimer based on issue types
        disclaimers = set()
        
        for issue in issues:
            rule_id = issue.get('rule_id', '')
            
            if 'pii' in rule_id.lower():
                disclaimers.add("Please note that any personal identifiers mentioned are examples only and should be replaced with appropriate data in real usage.")
            elif 'health' in rule_id.lower() or 'medical' in rule_id.lower():
                disclaimers.add("Note: This information is not medical advice. Consult with healthcare professionals for specific guidance.")
            elif 'financial' in rule_id.lower():
                disclaimers.add("Disclaimer: This information is not financial advice. Consult with financial professionals for specific guidance.")
        
        if disclaimers:
            # Add disclaimers at the end
            modified_text = text
            for disclaimer in disclaimers:
                if disclaimer not in modified_text:
                    modified_text += f"\n\n{disclaimer}"
            
            return {
                'text': modified_text,
                'confidence': 0.7,
                'type': 'clarifying_context',
                'description': f"Added {len(disclaimers)} clarifying disclaimer(s)"
            }
        
        return None

    def _restructure_sentences(self, text, issues):
        """Apply strategy: Restructure sentences to address issues"""
        # Sentence restructuring requires advanced NLP
        # This is a placeholder for a more sophisticated implementation
        
        # Return placeholder suggestion
        return {
            'text': "[Restructured version would be generated here - requires NLP model]",
            'confidence': 0.4,
            'type': 'sentence_restructuring',
            'description': "Suggested restructuring sentences to address compliance issues"
        }