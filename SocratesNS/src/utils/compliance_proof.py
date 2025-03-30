from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import uuid
import json
import datetime
import logging
import copy


class ProofTraceNodeType(Enum):
    """Types of nodes in a compliance proof trace"""
    TOKEN_CHECK = "token_check"           # Verification of a specific token
    ENTITY_VERIFICATION = "entity"        # Verification of an entity
    CONCEPT_VERIFICATION = "concept"      # Verification of a semantic concept
    PATTERN_MATCH = "pattern_match"       # Verification of a text pattern
    CONSTRAINT_APPLICATION = "constraint" # Application of a constraint
    RULE_EVALUATION = "rule"              # Evaluation of a compliance rule
    FRAMEWORK_VERIFICATION = "framework"  # Verification against a framework
    EXCEPTION_CHECK = "exception"         # Check for exception conditions
    COMPOSITE_CHECK = "composite"         # Composite of multiple checks


class ProofNodeStatus(Enum):
    """Status of a node in the proof trace"""
    SUCCESS = "success"       # Verification succeeded
    FAILURE = "failure"       # Verification failed
    WARNING = "warning"       # Succeeded with warnings
    EXCEPTION = "exception"   # Exception applied
    UNKNOWN = "unknown"       # Status unknown or not applicable


@dataclass
class ComplianceProofNode:
    """
    Node in a compliance proof trace tree, representing a step in the
    verification process.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    node_type: ProofTraceNodeType = ProofTraceNodeType.COMPOSITE_CHECK
    status: ProofNodeStatus = ProofNodeStatus.UNKNOWN
    description: str = ""
    entity_id: Optional[str] = None
    rule_id: Optional[str] = None
    framework_id: Optional[str] = None
    constraint_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    children: List['ComplianceProofNode'] = field(default_factory=list)
    parent_id: Optional[str] = None
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    
    def add_child(self, child: 'ComplianceProofNode') -> 'ComplianceProofNode':
        """Add a child node and return it"""
        child.parent_id = self.id
        self.children.append(child)
        return child
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": self.id,
            "node_type": self.node_type.value,
            "status": self.status.value,
            "description": self.description,
            "entity_id": self.entity_id,
            "rule_id": self.rule_id,
            "framework_id": self.framework_id,
            "constraint_id": self.constraint_id,
            "details": self.details,
            "children": [child.to_dict() for child in self.children],
            "parent_id": self.parent_id,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComplianceProofNode':
        """Create from dictionary representation"""
        node = cls(
            id=data.get("id", str(uuid.uuid4())),
            node_type=ProofTraceNodeType(data.get("node_type", "composite")),
            status=ProofNodeStatus(data.get("status", "unknown")),
            description=data.get("description", ""),
            entity_id=data.get("entity_id"),
            rule_id=data.get("rule_id"),
            framework_id=data.get("framework_id"),
            constraint_id=data.get("constraint_id"),
            details=data.get("details", {}),
            parent_id=data.get("parent_id"),
        )
        
        if "created_at" in data:
            node.created_at = datetime.datetime.fromisoformat(data["created_at"])
            
        # Recursively create children
        for child_data in data.get("children", []):
            child = cls.from_dict(child_data)
            child.parent_id = node.id
            node.children.append(child)
            
        return node


@dataclass
class ComplianceProofTrace:
    """
    Complete trace of the compliance verification process, containing
    a tree of proof nodes and metadata.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    root_node: ComplianceProofNode = field(default_factory=lambda: ComplianceProofNode(
        node_type=ProofTraceNodeType.COMPOSITE_CHECK,
        description="Root of compliance proof trace"
    ))
    is_compliant: bool = False
    compliance_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": self.id,
            "root_node": self.root_node.to_dict(),
            "is_compliant": self.is_compliant,
            "compliance_score": self.compliance_score,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComplianceProofTrace':
        """Create from dictionary representation"""
        trace = cls(
            id=data.get("id", str(uuid.uuid4())),
            is_compliant=data.get("is_compliant", False),
            compliance_score=data.get("compliance_score", 0.0),
            metadata=data.get("metadata", {})
        )
        
        if "root_node" in data:
            trace.root_node = ComplianceProofNode.from_dict(data["root_node"])
            
        if "created_at" in data:
            trace.created_at = datetime.datetime.fromisoformat(data["created_at"])
            
        return trace
