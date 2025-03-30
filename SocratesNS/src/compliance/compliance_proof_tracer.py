import json
import datetime
import os
import logging
import random
import string
from typing import List, Optional, Dict, Any
from enum import Enum  # Add this import for defining enums like ConflictResolutionStrategy
from enum import Enum  # Add this import for defining ProofNodeStatus
from src.compliance.compliance_proof_trace import ComplianceProofNode, ComplianceProofTrace
from src.compliance.compliance_verifier import ComplianceVerifier
from src.rules.violation_analyzer import ViolationAnalyzer
from src.core.utils.utils import LRUCache
from src.core.contextual_rule_interpreter import ContextualRuleInterpreter
from src.compliance.compliance_proof_formatter import ComplianceProofFormatter
from dataclasses import dataclass, field

import uuid

class ConflictResolutionStrategy(Enum):
    """Enumeration for conflict resolution strategies."""
    PRIORITIZE_FRAMEWORK = "prioritize_framework"
    MOST_RESTRICTIVE = "most_restrictive"
    MERGE_CONSTRAINTS = "merge_constraints"
# Define ComplianceProofNode or import it if it exists in another module
# Define ProofTraceNodeType or import it if it exists in another module
class ProofTraceNodeType(Enum):
    """Enumeration for proof trace node types."""
    TYPE_A = "type_a"
    TYPE_B = "type_b"
    TYPE_C = "type_c"

class ProofNodeStatus(Enum):
    """Enumeration for proof node statuses."""
    SUCCESS = "success"
    FAILURE = "failure"
    WARNING = "warning"
    EXCEPTION = "exception"
    UNKNOWN = "unknown"
# Define ComplianceProofNode or import it if it exists in another module
class ComplianceProofNode:
    def __init__(self, id: str, node_type: str, status: str, description: str, children: Optional[List['ComplianceProofNode']] = None):
        self.id = id
        self.node_type = node_type
        self.status = status
        self.description = description
        self.children = children or []

    def add_child(self, child: 'ComplianceProofNode'):
        self.children.append(child)

class ComplianceProofTracer:
    """
    Enhances the ComplianceVerifier with formal proof traces for auditable verification
    """
    def __init__(self, ruleset, symbolic_engine):
        self.ruleset = ruleset
        self.engine = symbolic_engine
        self.proof_history = []
        
    def verify_token(self, token, context_repr):
        """Verify token compliance with proof trace generation"""
        is_compliant, rule_chain = self.engine.evaluate(token, context_repr)
        proof_trace = self._construct_trace(token, rule_chain, context_repr)
        self.proof_history.append(proof_trace)
        return is_compliant, proof_trace

    def _construct_trace(self, token, rule_chain, context):
        """Construct a formal proof trace with full derivation chain"""
        trace = {
            "token": token,
            "context_snapshot": self._create_context_snapshot(context),
            "steps": [],
            "conclusion": {"is_compliant": bool(rule_chain)}
        }
        
        # Construct stepwise proof
        for i, rule_id in enumerate(rule_chain):
            rule = self.ruleset.get_rule(rule_id)
            premises = self._extract_premises(rule, context)
            inference = self._apply_inference_rule(rule, premises)
            
            trace["steps"].append({
                "step_id": i,
                "rule_id": rule_id,
                "rule_text": rule.text,
                "premises": premises,
                "inference": inference,
                "intermediate_conclusion": self._format_conclusion(inference)
            })
            
        # Add final derivation justification
        if trace["steps"]:
            trace["conclusion"]["justification"] = self._generate_justification(trace["steps"])
            trace["conclusion"]["formal_proof"] = self._generate_formal_notation(trace["steps"])
        
        return trace
    
    def _create_context_snapshot(self, context):
        """Create a minimal snapshot of relevant context"""
        return {k: v for k, v in context.items() if k in self._get_relevant_context_keys()}
    
    def _extract_premises(self, rule, context):
        """Extract premises that activated this rule"""
        # Implementation would identify which specific conditions in context
        # triggered this rule to apply
        return ["premise1", "premise2"]  # Placeholder
    
    def _apply_inference_rule(self, rule, premises):
        """Apply the rule's inference logic to premises"""
        # Implementation would show how conclusion follows from premises
        return {"conclusion": "token_compliant", "confidence": 0.95}  # Placeholder
    
    def _format_conclusion(self, inference):
        """Format intermediate conclusion in human-readable form"""
        return f"Token satisfies {inference['conclusion']} with confidence {inference['confidence']}"
    
    def _generate_justification(self, steps):
        """Generate natural language justification from proof steps"""
        if not steps:
            return "No applicable rules found."
        
        step_count = len(steps)
        last_step = steps[-1]
        return (f"Compliance verified through {step_count} rule application steps, "
                f"concluding with rule {last_step['rule_id']}: {last_step['rule_text']}")
    
    def _generate_formal_notation(self, steps):
        """Generate formal logical notation of the proof"""
        if not steps:
            return "∅ ⊬ compliant(token)"
        
        # Format as logical proof with proper notation
        proof_str = ""
        for step in steps:
            premises_str = " ∧ ".join([f"premise({p})" for p in step["premises"]])
            proof_str += f"{premises_str} ⊢ {step['intermediate_conclusion']}\n"
        
        return proof_str
    
    def export_proof_trace(self, format="json"):
        """Export the full proof trace history in specified format"""
        if format == "json":
            return json.dumps(self.proof_history)
        elif format == "pdf":
            return self._generate_pdf_report()
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _get_relevant_context_keys(self):
        """Get context keys relevant for proof construction"""
        return ["entities", "concepts", "sentiment", "previous_tokens"]
    
    def _generate_pdf_report(self):
        """Generate formal PDF compliance report with proof traces"""
        # Implementation would create formal documentation for auditors
        return "PDF_REPORT_BYTES"  # Placeholder

    def get_node_by_id(self, node_id: str) -> Optional[ComplianceProofNode]:
        """Get a node by its ID"""
        # Start the search from the root node
        return self._find_node_by_id(self.root_node, node_id)
        
    def _find_node_by_id(self, node: ComplianceProofNode, node_id: str) -> Optional[ComplianceProofNode]:
        """Recursively search for a node with the given ID"""
        if node.id == node_id:
            return node
            
        # Search in children
        for child in node.children:
            found = self._find_node_by_id(child, node_id)
            if found:
                return found
                
        # Not found in this branch
        return None
        
    def add_node(self, parent_id: str, node: ComplianceProofNode) -> bool:
        """
        Add a node as a child of the specified parent node.
        
        Args:
            parent_id: ID of the parent node
            node: Node to add
            
        Returns:
            True if added successfully, False otherwise
        """
        parent = self.get_node_by_id(parent_id)
        if parent:
            parent.add_child(node)
            return True
        return False
        
    def get_leaf_nodes(self) -> List[ComplianceProofNode]:
        """Get all leaf nodes in the proof trace"""
        leaves = []
        self._collect_leaf_nodes(self.root_node, leaves)
        return leaves
        
    def _collect_leaf_nodes(self, node: ComplianceProofNode, leaves: List[ComplianceProofNode]):
        """Recursively collect leaf nodes"""
        if not node.children:
            leaves.append(node)
        else:
            for child in node.children:
                self._collect_leaf_nodes(child, leaves)
                
    def get_nodes_by_type(self, node_type: str) -> List[ComplianceProofNode]:
        """Get all nodes of a specific type"""
        nodes = []
        self._collect_nodes_by_type(self.root_node, node_type, nodes)
        return nodes
        
    def _collect_nodes_by_type(self, node: ComplianceProofNode, node_type: str, 
                            nodes: List[ComplianceProofNode]):
        """Recursively collect nodes of a specific type"""
        if node.node_type == node_type:
            nodes.append(node)
            
        for child in node.children:
            self._collect_nodes_by_type(child, node_type, nodes)
            
    def get_nodes_by_status(self, status: ProofNodeStatus) -> List[ComplianceProofNode]:
        """Get all nodes with a specific status"""
        nodes = []
        self._collect_nodes_by_status(self.root_node, status, nodes)
        return nodes
        
    def _collect_nodes_by_status(self, node: ComplianceProofNode, status: ProofNodeStatus, 
                                nodes: List[ComplianceProofNode]):
        """Recursively collect nodes with a specific status"""
        if node.status == status:
            nodes.append(node)
            
        for child in node.children:
            self._collect_nodes_by_status(child, status, nodes)
            
    def get_path_to_node(self, node_id: str) -> List[ComplianceProofNode]:
        """
        Get the path from root to the specified node.
        
        Args:
            node_id: ID of the target node
            
        Returns:
            List of nodes from root to target, or empty list if not found
        """
        path = []
        self._find_path_to_node(self.root_node, node_id, path)
        return path
        
    def _find_path_to_node(self, node: ComplianceProofNode, node_id: str, 
                        path: List[ComplianceProofNode]) -> bool:
        """
        Recursively find path to node with the given ID.
        
        Returns:
            True if path was found, False otherwise
        """
        # Add current node to path
        path.append(node)
        
        # Check if this is the target
        if node.id == node_id:
            return True
            
        # Check children
        for child in node.children:
            if self._find_path_to_node(child, node_id, path):
                return True
                
        # If we reach here, this node is not on the path to target
        path.pop()
        return False
        
    def calculate_compliance_score(self) -> float:
        """
        Calculate overall compliance score based on the proof tree.
        
        Returns:
            Compliance score between 0.0 and 1.0
        """
        # Get all failure and success nodes
        failure_nodes = self.get_nodes_by_status(ProofNodeStatus.FAILURE)
        success_nodes = self.get_nodes_by_status(ProofNodeStatus.SUCCESS)
        warning_nodes = self.get_nodes_by_status(ProofNodeStatus.WARNING)
        
        # Calculate base score
        total_nodes = len(failure_nodes) + len(success_nodes) + len(warning_nodes)
        if total_nodes == 0:
            return 0.0
            
        # Calculate weighted score
        # Failures have full weight, warnings have half weight
        weighted_success = len(success_nodes)
        weighted_total = len(failure_nodes) + len(success_nodes) + (len(warning_nodes) * 0.5)
        
        # Update compliance score
        self.compliance_score = weighted_success / weighted_total if weighted_total > 0 else 0.0
        
        # Update compliance status
        self.is_compliant = len(failure_nodes) == 0
        
        return self.compliance_score
        
    def get_violations(self) -> List[Dict[str, Any]]:
        """
        Get list of compliance violations from the proof trace.
        
        Returns:
            List of violation details
        """
        violations = []
        
        # Get all failure nodes
        failure_nodes = self.get_nodes_by_status(ProofNodeStatus.FAILURE)
        
        for node in failure_nodes:
            violation = {
                "node_id": node.id,
                "description": node.description,
                "type": node.node_type.value,
            }
            
            # Add specific details based on node type
            if node.entity_id:
                violation["entity_id"] = node.entity_id
            if node.rule_id:
                violation["rule_id"] = node.rule_id
            if node.framework_id:
                violation["framework_id"] = node.framework_id
            if node.constraint_id:
                violation["constraint_id"] = node.constraint_id
                
            # Add node details
            if node.details:
                violation["details"] = node.details
                
            violations.append(violation)
            
        return violations
        
    def summarize(self) -> Dict[str, Any]:
        """
        Summarize the compliance proof trace.
        
        Returns:
            Summary of the proof trace
        """
        # Count nodes by type
        type_counts = {}
        for node_type in ProofTraceNodeType:
            nodes = self.get_nodes_by_type(node_type)
            if nodes:
                type_counts[node_type.value] = len(nodes)
                
        # Count nodes by status
        status_counts = {}
        for status in ProofNodeStatus:
            nodes = self.get_nodes_by_status(status)
            if nodes:
                status_counts[status.value] = len(nodes)
        
        # Calculate compliance score
        self.calculate_compliance_score()
        
        # Create summary
        return {
            "id": self.id,
            "is_compliant": self.is_compliant,
            "compliance_score": self.compliance_score,
            "created_at": self.created_at.isoformat(),
            "node_count": self._count_nodes(self.root_node),
            "node_types": type_counts,
            "node_statuses": status_counts,
            "violation_count": len(self.get_violations()),
            "metadata": self.metadata
        }
        
    def _count_nodes(self, node: ComplianceProofNode) -> int:
        """Recursively count nodes in the tree"""
        count = 1  # Count this node
        for child in node.children:
            count += self._count_nodes(child)
        return count
        
    def merge(self, other_trace: 'ComplianceProofTrace') -> 'ComplianceProofTrace':
        """
        Merge another proof trace into this one.
        
        Args:
            other_trace: Another proof trace to merge
            
        Returns:
            New merged proof trace
        """
        # Create a new trace
        # Ensure ComplianceProofTrace is defined or imported
        from compliance_proof_trace import ComplianceProofTrace  # Import if defined in another module
        merged_trace = ComplianceProofTrace()
        
        # Copy metadata
        merged_trace.metadata = {
            **self.metadata,
            **other_trace.metadata,
            "merged_from": [self.id, other_trace.id],
            "merge_time": datetime.datetime.now().isoformat()
        }
        
        # Add children from both root nodes to the new root
        for child in self.root_node.children:
            # Create a deep copy to avoid modifying the original
            child_copy = ComplianceProofNode.from_dict(child.to_dict())
            merged_trace.root_node.add_child(child_copy)
            
        for child in other_trace.root_node.children:
            # Create a deep copy to avoid modifying the original
            child_copy = ComplianceProofNode.from_dict(child.to_dict())
            merged_trace.root_node.add_child(child_copy)
        
        # Update compliance status and score
        merged_trace.calculate_compliance_score()
        
        return merged_trace
        
    def prune(self, max_depth: Optional[int] = None, 
            include_types: Optional[List[ProofTraceNodeType]] = None,
            exclude_types: Optional[List[ProofTraceNodeType]] = None) -> 'ComplianceProofTrace':
        """
        Create a pruned version of the proof trace.
        
        Args:
            max_depth: Maximum depth to include (None for no limit)
            include_types: Only include these node types (None for all)
            exclude_types: Exclude these node types
            
        Returns:
            New pruned proof trace
        """
        # Create a new trace
        pruned_trace = ComplianceProofTrace()
        
        # Copy metadata and add pruning info
        pruned_trace.metadata = {
            **self.metadata,
            "pruned_from": self.id,
            "prune_time": datetime.datetime.now().isoformat(),
            "prune_params": {
                "max_depth": max_depth,
                "include_types": [t.value for t in include_types] if include_types else None,
                "exclude_types": [t.value for t in exclude_types] if exclude_types else None
            }
        }
        
        # Clone the root node
        pruned_trace.root_node = ComplianceProofNode.from_dict(self.root_node.to_dict())
        
        # Prune children recursively
        self._prune_children(
            pruned_trace.root_node,
            max_depth=max_depth,
            include_types=include_types,
            exclude_types=exclude_types,
            current_depth=0
        )
        
        # Update compliance status and score
        pruned_trace.calculate_compliance_score()
        
        return pruned_trace
        
    def _prune_children(self, node: ComplianceProofNode, max_depth: Optional[int],
                    include_types: Optional[List[ProofTraceNodeType]],
                    exclude_types: Optional[List[ProofTraceNodeType]],
                    current_depth: int):
        """Recursively prune children based on criteria"""
        # Check if we've reached max depth
        if max_depth is not None and current_depth >= max_depth:
            node.children = []
            return
            
        # Process children
        new_children = []
        for child in node.children:
            # Check type inclusion/exclusion
            if include_types and child.node_type not in include_types:
                continue
                
            if exclude_types and child.node_type in exclude_types:
                continue
                
            # Keep this child
            new_children.append(child)
            
            # Recursively prune its children
            self._prune_children(
                child,
                max_depth=max_depth,
                include_types=include_types,
                exclude_types=exclude_types,
                current_depth=current_depth + 1
            )
            
        # Update node's children
        node.children = new_children
        
    def export_to_format(self, format_type: str) -> str:
        """
        Export the proof trace to a specific format.
        
        Args:
            format_type: "json", "xml", "dot" (for GraphViz), or "text"
            
        Returns:
            Formatted string representation
        """
        if format_type == "json":
            return json.dumps(self.to_dict(), indent=2)
        elif format_type == "xml":
            return self._to_xml()
        elif format_type == "dot":
            return self._to_dot()
        elif format_type == "text":
            return self._to_text()
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
            
    def _to_xml(self) -> str:
        """Convert to XML representation"""
        lines = ['<?xml version="1.0" encoding="UTF-8"?>']
        lines.append(f'<complianceProofTrace id="{self.id}" isCompliant="{str(self.is_compliant).lower()}" score="{self.compliance_score:.4f}">')
        
        # Add metadata
        if self.metadata:
            lines.append('  <metadata>')
            for key, value in self.metadata.items():
                # Simple conversion for basic types
                if isinstance(value, (str, int, float, bool)):
                    lines.append(f'    <{key}>{value}</{key}>')
            lines.append('  </metadata>')
        
        # Add root node and its children
        lines.append(self._node_to_xml(self.root_node, indent=2))
        
        lines.append('</complianceProofTrace>')
        return '\n'.join(lines)
        
    def _node_to_xml(self, node: ComplianceProofNode, indent: int) -> str:
        """Convert node to XML representation with proper indentation"""
        indent_str = ' ' * indent
        lines = []
        
        # Start node tag with attributes
        lines.append(f'{indent_str}<node id="{node.id}" type="{node.node_type.value}" status="{node.status.value}">')
        
        # Add node fields
        lines.append(f'{indent_str}  <description>{node.description}</description>')
        
        if node.entity_id:
            lines.append(f'{indent_str}  <entityId>{node.entity_id}</entityId>')
        if node.rule_id:
            lines.append(f'{indent_str}  <ruleId>{node.rule_id}</ruleId>')
        if node.framework_id:
            lines.append(f'{indent_str}  <frameworkId>{node.framework_id}</frameworkId>')
        if node.constraint_id:
            lines.append(f'{indent_str}  <constraintId>{node.constraint_id}</constraintId>')
        
        # Add details if present
        if node.details:
            lines.append(f'{indent_str}  <details>')
            for key, value in node.details.items():
                if isinstance(value, (str, int, float, bool)):
                    lines.append(f'{indent_str}    <{key}>{value}</{key}>')
            lines.append(f'{indent_str}  </details>')
        
        # Add children
        if node.children:
            lines.append(f'{indent_str}  <children>')
            for child in node.children:
                lines.append(self._node_to_xml(child, indent + 4))
            lines.append(f'{indent_str}  </children>')
        
        # Close node tag
        lines.append(f'{indent_str}</node>')
        
        return '\n'.join(lines)
        
    def _to_dot(self) -> str:
        """Convert to GraphViz DOT format for visualization"""
        lines = ['digraph ComplianceProofTrace {']
        lines.append('  rankdir=TB;')
        lines.append('  node [shape=box, style=filled];')
        
        # Add nodes
        node_count = self._add_nodes_to_dot(self.root_node, lines)
        
        # Add edges
        self._add_edges_to_dot(self.root_node, lines)
        
        lines.append('}')
        return '\n'.join(lines)
        
    def _add_nodes_to_dot(self, node: ComplianceProofNode, lines: List[str]) -> int:
        """Add nodes to DOT representation"""
        # Determine color based on status
        color_map = {
            ProofNodeStatus.SUCCESS: "green",
            ProofNodeStatus.FAILURE: "red",
            ProofNodeStatus.WARNING: "orange",
            ProofNodeStatus.EXCEPTION: "purple",
            ProofNodeStatus.UNKNOWN: "gray"
        }
        color = color_map.get(node.status, "blue")
        
        # Add node
        label = f"{node.node_type.value}\\n{node.description}"
        if node.rule_id:
            label += f"\\nRule: {node.rule_id}"
        if node.entity_id:
            label += f"\\nEntity: {node.entity_id}"
            
        lines.append(f'  node_{node.id} [label="{label}", fillcolor="{color}"];')
        
        # Add children recursively
        count = 1
        for child in node.children:
            count += self._add_nodes_to_dot(child, lines)
            
        return count
        
    def _add_edges_to_dot(self, node: ComplianceProofNode, lines: List[str]):
        """Add edges to DOT representation"""
        for child in node.children:
            lines.append(f'  node_{node.id} -> node_{child.id};')
            self._add_edges_to_dot(child, lines)
        
    def _to_text(self) -> str:
        """Convert to text representation"""
        lines = [f"Compliance Proof Trace: {self.id}"]
        lines.append(f"Created: {self.created_at}")
        lines.append(f"Compliance Status: {'Compliant' if self.is_compliant else 'Non-Compliant'}")
        lines.append(f"Compliance Score: {self.compliance_score:.4f}")
        lines.append("")
        
        lines.append("Proof Tree:")
        self._append_node_text(self.root_node, lines, indent=0)
        
        if self.metadata:
            lines.append("")
            lines.append("Metadata:")
            for key, value in self.metadata.items():
                lines.append(f"  {key}: {value}")
        
        return '\n'.join(lines)
        
    def _append_node_text(self, node: ComplianceProofNode, lines: List[str], indent: int):
        """Append text representation of node to lines"""
        indent_str = '  ' * indent
        status_symbol = {
            ProofNodeStatus.SUCCESS: "✓",
            ProofNodeStatus.FAILURE: "✗",
            ProofNodeStatus.WARNING: "⚠",
            ProofNodeStatus.EXCEPTION: "!",
            ProofNodeStatus.UNKNOWN: "?"
        }.get(node.status, " ")
        
        lines.append(f"{indent_str}{status_symbol} {node.node_type.value}: {node.description}")
        
        # Add additional details with more indent
        detail_indent = indent_str + "    "
        if node.entity_id:
            lines.append(f"{detail_indent}Entity: {node.entity_id}")
        if node.rule_id:
            lines.append(f"{detail_indent}Rule: {node.rule_id}")
        if node.framework_id:
            lines.append(f"{detail_indent}Framework: {node.framework_id}")
        if node.constraint_id:
            lines.append(f"{detail_indent}Constraint: {node.constraint_id}")
            
        # Add children
        for child in node.children:
            self._append_node_text(child, lines, indent + 1)

    def explain_resolution(self, constraints, resolution_result, detail_level="medium"):
        """
        Generate human-readable explanation of conflict resolution.
        
        Args:
            constraints: List of constraints that were resolved
            resolution_result: Result of conflict resolution
            detail_level: "low", "medium", or "high"
            
        Returns:
            Human-readable explanation
        """
        if not resolution_result.is_conflict_resolved:
            return f"Unable to resolve conflict: {resolution_result.reason}"
            
        # Start with a basic explanation
        explanation = [f"Conflict resolution used strategy: {resolution_result.resolution_strategy.value}"]
        explanation.append(f"Reason: {resolution_result.reason}")
        
        # Add constraint details based on detail level
        if detail_level in ["medium", "high"]:
            # Add information about constraints
            constraint_ids = [c.id for c in constraints]
            explanation.append(f"\nResolving conflicts between {len(constraints)} constraints: {', '.join(constraint_ids)}")
            
            if resolution_result.winning_constraint_id:
                winner = next((c for c in constraints if c.id == resolution_result.winning_constraint_id), None)
                if winner:
                    explanation.append(f"\nWinning constraint: {winner.id} from framework {winner.framework_id}")
                    explanation.append(f"Description: {winner.description}")
                    
                    if detail_level == "high":
                        explanation.append(f"Severity: {winner.severity}")
                        explanation.append(f"Restriction level: {winner.restriction_level}")
                        explanation.append(f"Applicable entities: {', '.join(winner.applicable_entities)}")
            
            if resolution_result.merged_constraint:
                merged = resolution_result.merged_constraint
                explanation.append(f"\nConstraints were merged into a new constraint: {merged.id}")
                explanation.append(f"Description: {merged.description}")
                
                if detail_level == "high":
                    explanation.append(f"Merged from frameworks: {merged.framework_id}")
                    explanation.append(f"Severity: {merged.severity}")
                    explanation.append(f"Restriction level: {merged.restriction_level}")
                    explanation.append(f"Combined scope: {', '.join(merged.scope)}")
                    explanation.append(f"Combined applicable entities: {', '.join(merged.applicable_entities)}")
        
        # For high detail, add the decision process
        if detail_level == "high":
            explanation.append("\nDecision process:")
            if resolution_result.resolution_strategy == ConflictResolutionStrategy.PRIORITIZE_FRAMEWORK:
                # Explain framework priority decision
                explanation.append("Framework priority comparison:")
                for i, constraint_a in enumerate(constraints):
                    for constraint_b in constraints[i+1:]:
                        fw_a, fw_b = constraint_a.framework_id, constraint_b.framework_id
                        winning_id, precedence, reason = self.get_framework_precedence(fw_a, fw_b)
                        if winning_id:
                            explanation.append(f"  - {winning_id} takes precedence over {fw_b if winning_id == fw_a else fw_a}: {reason}")
            
            elif resolution_result.resolution_strategy == ConflictResolutionStrategy.MOST_RESTRICTIVE:
                # Explain restriction level comparison
                explanation.append("Restriction level comparison:")
                for constraint in constraints:
                    explanation.append(f"  - {constraint.id}: {constraint.restriction_level}")
                    
            elif resolution_result.resolution_strategy == ConflictResolutionStrategy.MERGE_CONSTRAINTS:
                # Explain merge process
                explanation.append("Constraint merging process:")
                explanation.append(f"  - Combined scope from {len(constraints)} constraints")
                explanation.append(f"  - Used highest restriction level: {resolution_result.merged_constraint.restriction_level}")
                explanation.append(f"  - Used highest severity: {resolution_result.merged_constraint.severity}")
        
        return "\n".join(explanation)


class ComplianceProofTracer:
    """
    Verifier that produces detailed compliance proofs for auditability and
    explanation generation.
    """
    def __init__(self, ruleset, symbolic_engine):
        self.ruleset = ruleset
        self.symbolic_engine = symbolic_engine
        self.proof_formatter = ComplianceProofFormatter()
        
    def verify_with_proof(self, text, frameworks, compliance_mode="strict"):
        """
        Verify compliance with detailed proof tracing
        
        Args:
            text: Text to verify
            frameworks: Applicable regulatory frameworks
            compliance_mode: Compliance strictness mode
            
        Returns:
            Tuple of (verification_result, proof_trace)
        """
        # Start proof trace
        proof_trace = {
            "input": self._format_input_summary(text),
            "frameworks": [f.id for f in frameworks],
            "mode": compliance_mode,
            "steps": [],
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Convert text to symbolic representation
        symbolic_repr = self.symbolic_engine.text_to_symbolic(text)
        
        # Add symbolic representation step
        proof_trace["steps"].append({
            "step_type": "representation",
            "description": "Convert text to symbolic representation",
            "output_type": "symbolic_representation"
        })
        
        # Verify against each framework
        framework_results = []
        for framework in frameworks:
            # Verify compliance for this framework
            framework_result = self._verify_framework_with_proof(
                symbolic_repr,
                framework,
                compliance_mode
            )
            
            # Add framework verification to proof steps
            proof_trace["steps"].extend(framework_result["proof_steps"])
            
            # Save framework result
            framework_results.append({
                "framework_id": framework.id,
                "is_compliant": framework_result["is_compliant"],
                "compliance_score": framework_result["compliance_score"],
                "violations": framework_result["violations"]
            })
        
        # Aggregate framework results
        aggregated_result = self._aggregate_framework_results(framework_results)
        
        # Add aggregation step to proof
        proof_trace["steps"].append({
            "step_type": "aggregation",
            "description": "Aggregate results across frameworks",
            "intermediate_conclusion": f"Content {'complies with' if aggregated_result['is_compliant'] else 'violates'} requirements",
            "compliance_score": aggregated_result["compliance_score"]
        })
        
        # Generate final conclusion
        conclusion = self._generate_conclusion(aggregated_result, frameworks)
        proof_trace["conclusion"] = conclusion
        
        return aggregated_result, proof_trace
        
    def _verify_framework_with_proof(self, symbolic_repr, framework, compliance_mode):
        """Verify compliance against framework with proof tracing"""
        # Start framework verification
        proof_steps = [{
            "step_type": "framework_start",
            "framework_id": framework.id,
            "description": f"Begin verification against {framework.id}"
        }]
        
        # Get framework rules
        rules = framework.get_rules(compliance_mode)
        
        # Track violations and rule results
        violations = []
        rule_results = []
        
        # Verify each rule
        for rule in rules:
            # Apply rule to symbolic representation
            rule_result = self._apply_rule_with_proof(symbolic_repr, rule)
            
            # Add rule verification step
            proof_steps.append({
                "step_type": "rule_verification",
                "rule_id": rule.get("id", "unknown"),
                "rule_text": rule.get("description", ""),
                "intermediate_conclusion": rule_result["conclusion"],
                "justification": rule_result["justification"]
            })
            
            # Track result
            rule_results.append(rule_result)
            
            # Track violation if rule failed
            if not rule_result["is_compliant"]:
                violations.append({
                    "rule_id": rule.get("id", "unknown"),
                    "rule_text": rule.get("description", ""),
                    "severity": rule.get("severity", "medium"),
                    "justification": rule_result["justification"]
                })
        
        # Calculate framework compliance score
        rule_scores = [r["compliance_score"] for r in rule_results]
        avg_score = sum(rule_scores) / len(rule_scores) if rule_scores else 1.0
        
        # Determine if framework is compliant (all rules must pass in strict mode)
        if compliance_mode == "strict":
            is_compliant = len(violations) == 0
        else:
            # In relaxed mode, allow minor violations
            is_compliant = all(v["severity"] != "high" for v in violations)
            
        # Add framework conclusion step
        proof_steps.append({
            "step_type": "framework_conclusion",
            "framework_id": framework.id,
            "intermediate_conclusion": f"Content {'complies with' if is_compliant else 'violates'} framework {framework.id}",
            "violation_count": len(violations),
            "compliance_score": avg_score
        })
        
        return {
            "is_compliant": is_compliant,
            "compliance_score": avg_score,
            "violations": violations,
            "proof_steps": proof_steps
        }
        
    def _apply_rule_with_proof(self, symbolic_repr, rule):
        """Apply rule to symbolic representation with proof tracing"""
        # This is a placeholder implementation
        # In a real system, this would apply formal verification logic
        
        # Simulate rule application (random outcome for demonstration)
        is_compliant = random.random() > 0.2  # 80% chance of compliance
        compliance_score = random.uniform(0.7, 1.0) if is_compliant else random.uniform(0.3, 0.7)
        
        # Generate justification
        if is_compliant:
            justification = f"Content satisfies rule constraints"
            conclusion = f"Rule {rule.get('id', 'unknown')} is satisfied"
        else:
            justification = f"Content contains elements that violate rule constraints"
            conclusion = f"Content violates rule {rule.get('id', 'unknown')}"
            
        return {
            "is_compliant": is_compliant,
            "compliance_score": compliance_score,
            "justification": justification,
            "conclusion": conclusion
        }
        
    def _aggregate_framework_results(self, framework_results):
        """Aggregate results from multiple frameworks"""
        if not framework_results:
            return {"is_compliant": True, "compliance_score": 1.0, "violations": []}
        
        # Determine overall compliance (all frameworks must pass)
        is_compliant = all(result["is_compliant"] for result in framework_results)
        
        # Calculate overall compliance score (weighted by violation severity)
        compliance_scores = [result["compliance_score"] for result in framework_results]
        overall_score = min(compliance_scores) if compliance_scores else 1.0
        
        # Collect all violations
        all_violations = []
        for result in framework_results:
            for violation in result["violations"]:
                violation["framework_id"] = result["framework_id"]
                all_violations.append(violation)
        
        # Remove duplicates (same rule across frameworks)
        unique_violations = self._deduplicate_violations(all_violations)
        
        return {
            "is_compliant": is_compliant,
            "compliance_score": overall_score,
            "violations": unique_violations,
            "framework_results": framework_results
        }

    def _generate_conclusion(self, aggregated_result, frameworks):
        """Generate final conclusion with justification"""
        is_compliant = aggregated_result["is_compliant"]
        score = aggregated_result["compliance_score"]
        violations = aggregated_result["violations"]
        
        # Framework summary text
        framework_names = [f.id for f in frameworks]
        frameworks_text = ", ".join(framework_names)
        
        if is_compliant:
            if score > 0.95:
                justification = f"Content fully complies with all requirements from {frameworks_text}."
            else:
                justification = f"Content satisfies requirements from {frameworks_text}, although some areas could be improved."
        else:
            if len(violations) == 1:
                violation = violations[0]
                justification = f"Content violates 1 rule from {violation['framework_id']}: {violation['rule_text']}"
            else:
                justification = f"Content violates {len(violations)} rules across {len(set(v['framework_id'] for v in violations))} frameworks."
        
        return {
            "is_compliant": is_compliant,
            "compliance_score": score,
            "violation_count": len(violations),
            "justification": justification,
            "timestamp": datetime.datetime.now().isoformat()
        }

    def _format_input_summary(self, text):
        """Format input text summary for proof trace"""
        # Truncate and summarize for the proof
        if len(text) > 500:
            return text[:250] + "..." + text[-250:]
        return text

    def _deduplicate_violations(self, violations):
        """Remove duplicate violations based on rule ID"""
        unique_violations = {}
        
        for violation in violations:
            rule_id = violation["rule_id"]
            
            # If rule not seen yet or this instance has higher severity, keep this one
            if rule_id not in unique_violations or self._severity_rank(violation["severity"]) > self._severity_rank(unique_violations[rule_id]["severity"]):
                unique_violations[rule_id] = violation
        
        return list(unique_violations.values())

    def _severity_rank(self, severity):
        """Convert severity string to numeric rank"""
        ranks = {"low": 1, "medium": 2, "high": 3}
        return ranks.get(severity, 1)