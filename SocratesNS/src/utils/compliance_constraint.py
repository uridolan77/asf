from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import re
import logging
import datetime
import json
import hashlib


class RegulationPrecedence(Enum):
    """Precedence types for regulation conflict resolution"""
    HIGHER = 1   # Higher precedence regulation overrides lower
    STRICTER = 2 # Stricter regulation overrides less strict
    NEWER = 3    # Newer regulation overrides older
    DOMAIN = 4   # Domain-specific overrides general
    CUSTOM = 5   # Custom precedence rules


class ConflictResolutionStrategy(Enum):
    """Strategies for resolving conflicts between regulations"""
    PRIORITIZE_FRAMEWORK = "prioritize_framework"  # Use framework priority ordering
    PRIORITIZE_RULE_TYPE = "prioritize_rule_type"  # Use rule type priority
    MERGE_CONSTRAINTS = "merge_constraints"        # Combine constraints where possible
    MOST_RESTRICTIVE = "most_restrictive"          # Apply the most restrictive constraint
    LEAST_RESTRICTIVE = "least_restrictive"        # Apply the least restrictive constraint
    CONTEXT_DEPENDENT = "context_dependent"        # Resolve based on context


@dataclass
class RegulationMetadata:
    """Metadata about a regulatory framework"""
    id: str
    name: str
    description: str
    version: str = "1.0"
    issued_date: str = ""
    jurisdiction: List[str] = field(default_factory=list)
    sector: List[str] = field(default_factory=list)
    precedence_level: int = 5  # Lower number = higher precedence
    is_domain_specific: bool = False
    domain: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "issued_date": self.issued_date,
            "jurisdiction": self.jurisdiction,
            "sector": self.sector,
            "precedence_level": self.precedence_level,
            "is_domain_specific": self.is_domain_specific,
            "domain": self.domain
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RegulationMetadata':
        """Create from dictionary"""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            version=data.get("version", "1.0"),
            issued_date=data.get("issued_date", ""),
            jurisdiction=data.get("jurisdiction", []),
            sector=data.get("sector", []),
            precedence_level=data.get("precedence_level", 5),
            is_domain_specific=data.get("is_domain_specific", False),
            domain=data.get("domain")
        )


@dataclass
class ComplianceConstraint:
    """Representation of a compliance constraint from a regulation"""
    id: str
    framework_id: str
    description: str
    constraint_type: str
    severity: str = "medium"
    restriction_level: float = 0.5  # 0.0 = no restriction, 1.0 = complete restriction
    scope: List[str] = field(default_factory=list)
    applicable_entities: List[str] = field(default_factory=list)
    condition: Optional[str] = None
    exception_conditions: List[str] = field(default_factory=list)
    implementation: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "framework_id": self.framework_id,
            "description": self.description,
            "constraint_type": self.constraint_type,
            "severity": self.severity,
            "restriction_level": self.restriction_level,
            "scope": self.scope,
            "applicable_entities": self.applicable_entities,
            "condition": self.condition,
            "exception_conditions": self.exception_conditions,
            "implementation": self.implementation
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComplianceConstraint':
        """Create from dictionary"""
        return cls(
            id=data["id"],
            framework_id=data["framework_id"],
            description=data["description"],
            constraint_type=data["constraint_type"],
            severity=data.get("severity", "medium"),
            restriction_level=data.get("restriction_level", 0.5),
            scope=data.get("scope", []),
            applicable_entities=data.get("applicable_entities", []),
            condition=data.get("condition"),
            exception_conditions=data.get("exception_conditions", []),
            implementation=data.get("implementation", {})
        )


@dataclass
class ConstraintRelationship:
    """Relationship between two constraints"""
    constraint_a_id: str
    constraint_b_id: str
    relationship_type: str  # e.g., "contradicts", "subsumes", "overlaps"
    description: str
    resolution_strategy: ConflictResolutionStrategy
    framework_a_id: str
    framework_b_id: str


@dataclass
class ConflictResolutionResult:
    """Result of conflict resolution process"""
    is_conflict_resolved: bool
    resolution_strategy: ConflictResolutionStrategy
    winning_constraint_id: Optional[str] = None
    merged_constraint: Optional[ComplianceConstraint] = None
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class RegulationConflictResolver:
    """
    Advanced conflict resolution for regulatory frameworks,
    handling overlapping and contradictory compliance requirements.
    """
    
    def __init__(self,
                 framework_metadata: Dict[str, RegulationMetadata] = None,
                 precedence_rules: Dict[str, Dict[str, RegulationPrecedence]] = None,
                 default_resolution_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.MOST_RESTRICTIVE,
                 constraint_relationships: List[ConstraintRelationship] = None,
                 logger=None):
        """
        Initialize the conflict resolver.
        
        Args:
            framework_metadata: Metadata about regulatory frameworks
            precedence_rules: Rules for precedence between frameworks
            default_resolution_strategy: Default strategy for resolving conflicts
            constraint_relationships: Known relationships between constraints
            logger: Logger for debug information
        """
        self.framework_metadata = framework_metadata or {}
        self.precedence_rules = precedence_rules or {}
        self.default_resolution_strategy = default_resolution_strategy
        self.constraint_relationships = constraint_relationships or []
        self.logger = logger or logging.getLogger("regulation_conflict_resolver")
        
        # Build index of constraint relationships for faster lookup
        self.constraint_relationship_index = {}
        self._build_constraint_relationship_index()
        
        # Cache for conflict resolution results
        self.resolution_cache = {}
        
    def resolve_conflicts(self, 
                         constraints: List[ComplianceConstraint], 
                         context: Dict[str, Any] = None) -> List[ComplianceConstraint]:
        """
        Resolve conflicts between compliance constraints.
        
        Args:
            constraints: List of constraints to resolve conflicts between
            context: Context information for resolving conflicts
            
        Returns:
            List of constraints with conflicts resolved
        """
        if not constraints:
            return []
            
        if len(constraints) == 1:
            return constraints
            
        self.logger.info(f"Resolving conflicts between {len(constraints)} constraints")
        resolved_constraints = []
        
        # Group constraints by type to identify potential conflicts
        constraints_by_type = self._group_constraints_by_type(constraints)
        
        # Process each constraint type
        for constraint_type, type_constraints in constraints_by_type.items():
            if len(type_constraints) == 1:
                # No conflicts for this type
                resolved_constraints.extend(type_constraints)
                continue
                
            self.logger.debug(f"Resolving conflicts for type {constraint_type} "
                              f"with {len(type_constraints)} constraints")
                
            # Identify conflicts within this type
            conflict_groups = self._identify_conflict_groups(type_constraints)
            
            # Resolve each conflict group
            for group in conflict_groups:
                if len(group) == 1:
                    # No conflict in this group
                    resolved_constraints.extend(group)
                else:
                    # Resolve conflicts in this group
                    resolution = self._resolve_constraint_group(group, context)
                    if resolution.merged_constraint:
                        resolved_constraints.append(resolution.merged_constraint)
                    elif resolution.winning_constraint_id:
                        # Find the winning constraint
                        winning_constraint = next(
                            (c for c in group if c.id == resolution.winning_constraint_id),
                            None
                        )
                        if winning_constraint:
                            resolved_constraints.append(winning_constraint)
                    else:
                        # Failed to resolve, apply all constraints (conservative approach)
                        resolved_constraints.extend(group)
                        
        return resolved_constraints
    
    def get_framework_precedence(self, 
                               framework_a_id: str, 
                               framework_b_id: str) -> Tuple[str, RegulationPrecedence, str]:
        """
        Determine which framework takes precedence when there is a conflict.
        
        Args:
            framework_a_id: ID of first framework
            framework_b_id: ID of second framework
            
        Returns:
            Tuple of (winning_framework_id, precedence_type, reason)
        """
        # Check direct precedence rules
        if framework_a_id in self.precedence_rules:
            if framework_b_id in self.precedence_rules[framework_a_id]:
                precedence = self.precedence_rules[framework_a_id][framework_b_id]
                return framework_a_id, precedence, f"Direct precedence rule: {precedence.name}"
                
        if framework_b_id in self.precedence_rules:
            if framework_a_id in self.precedence_rules[framework_b_id]:
                precedence = self.precedence_rules[framework_b_id][framework_a_id]
                # Reverse the result since the rule is defined for B over A
                return framework_b_id, precedence, f"Direct precedence rule: {precedence.name}"
                
        # Get metadata for both frameworks
        meta_a = self.framework_metadata.get(framework_a_id)
        meta_b = self.framework_metadata.get(framework_b_id)
        
        if not meta_a or not meta_b:
            # Missing metadata, can't determine precedence
            return None, None, "Missing framework metadata"
            
        # Check precedence level
        if meta_a.precedence_level != meta_b.precedence_level:
            if meta_a.precedence_level < meta_b.precedence_level:
                return framework_a_id, RegulationPrecedence.HIGHER, "Higher precedence level"
            else:
                return framework_b_id, RegulationPrecedence.HIGHER, "Higher precedence level"
                
        # Check domain specificity
        if meta_a.is_domain_specific != meta_b.is_domain_specific:
            if meta_a.is_domain_specific:
                return framework_a_id, RegulationPrecedence.DOMAIN, "Domain-specific regulation"
            else:
                return framework_b_id, RegulationPrecedence.DOMAIN, "Domain-specific regulation"
                
        # Check issuance date if available
        if meta_a.issued_date and meta_b.issued_date:
            try:
                date_a = datetime.datetime.fromisoformat(meta_a.issued_date)
                date_b = datetime.datetime.fromisoformat(meta_b.issued_date)
                
                if date_a != date_b:
                    if date_a > date_b:
                        return framework_a_id, RegulationPrecedence.NEWER, "More recent regulation"
                    else:
                        return framework_b_id, RegulationPrecedence.NEWER, "More recent regulation"
            except (ValueError, TypeError):
                # Invalid date format, ignore
                pass
                
        # No clear precedence
        return None, None, "No precedence rule applies"
    
    def add_precedence_rule(self, 
                          framework_a_id: str, 
                          framework_b_id: str, 
                          precedence: RegulationPrecedence):
        """
        Add a precedence rule between two frameworks.
        
        Args:
            framework_a_id: ID of first framework
            framework_b_id: ID of second framework
            precedence: Type of precedence
        """
        if framework_a_id not in self.precedence_rules:
            self.precedence_rules[framework_a_id] = {}
            
        self.precedence_rules[framework_a_id][framework_b_id] = precedence
        
    def add_constraint_relationship(self, relationship: ConstraintRelationship):
        """
        Add a known relationship between constraints.
        
        Args:
            relationship: Relationship between constraints
        """
        self.constraint_relationships.append(relationship)
        self._build_constraint_relationship_index()
        
    def _build_constraint_relationship_index(self):
        """Build index of constraint relationships for faster lookup"""
        self.constraint_relationship_index = {}
        
        for relationship in self.constraint_relationships:
            # Index by both constraints
            key_a_b = self._make_relationship_key(
                relationship.constraint_a_id, 
                relationship.constraint_b_id
            )
            key_b_a = self._make_relationship_key(
                relationship.constraint_b_id, 
                relationship.constraint_a_id
            )
            
            self.constraint_relationship_index[key_a_b] = relationship
            self.constraint_relationship_index[key_b_a] = relationship
    
    def _make_relationship_key(self, constraint_a_id: str, constraint_b_id: str) -> str:
        """Make a key for constraint relationship lookup"""
        return f"{constraint_a_id}|{constraint_b_id}"
    
    def _group_constraints_by_type(self, 
                                 constraints: List[ComplianceConstraint]) -> Dict[str, List[ComplianceConstraint]]:
        """Group constraints by type"""
        result = {}
        
        for constraint in constraints:
            if constraint.constraint_type not in result:
                result[constraint.constraint_type] = []
            result[constraint.constraint_type].append(constraint)
            
        return result
    
    def _identify_conflict_groups(self, 
                                constraints: List[ComplianceConstraint]) -> List[List[ComplianceConstraint]]:
        """
        Identify groups of constraints that potentially conflict with each other.
        
        Args:
            constraints: List of constraints of the same type
            
        Returns:
            List of constraint groups that need conflict resolution
        """
        # Group by entity scope for more precise conflict detection
        # A conflict can only occur if constraints apply to the same entities
        entity_groups = {}
        
        for constraint in constraints:
            # Create a hash based on applicable entities
            entities = sorted(constraint.applicable_entities)
            entity_key = "|".join(entities) if entities else "GLOBAL"
            
            if entity_key not in entity_groups:
                entity_groups[entity_key] = []
            entity_groups[entity_key].append(constraint)
            
        # For each entity group, identify conflicts
        conflict_groups = []
        
        for entity_key, entity_constraints in entity_groups.items():
            if len(entity_constraints) == 1:
                # No conflicts within this entity scope
                conflict_groups.append(entity_constraints)
                continue
                
            # Check for known conflicts using constraint relationships
            conflict_subgroups = self._identify_subgroups_by_relationships(entity_constraints)
            conflict_groups.extend(conflict_subgroups)
            
        return conflict_groups
    
    def _identify_subgroups_by_relationships(self, 
                                          constraints: List[ComplianceConstraint]) -> List[List[ComplianceConstraint]]:
        """
        Identify subgroups of constraints based on known relationships.
        
        Args:
            constraints: List of constraints with same entity scope
            
        Returns:
            List of constraint subgroups
        """
        # If no relationships defined, consider all constraints as a single group
        if not self.constraint_relationships:
            return [constraints]
            
        # Build a graph of constraint relationships
        # Each constraint is a node, and an edge exists if there's a relationship
        relationship_graph = {}
        
        for constraint in constraints:
            relationship_graph[constraint.id] = set()
            
        # Add edges between related constraints
        for i, constraint_a in enumerate(constraints):
            for constraint_b in constraints[i+1:]:
                key = self._make_relationship_key(constraint_a.id, constraint_b.id)
                
                if key in self.constraint_relationship_index:
                    # There's a relationship between these constraints
                    relationship_graph[constraint_a.id].add(constraint_b.id)
                    relationship_graph[constraint_b.id].add(constraint_a.id)
                    
        # Identify connected components in the graph
        subgroups = self._find_connected_components(relationship_graph)
        
        # Convert constraint IDs back to constraints
        constraint_map = {c.id: c for c in constraints}
        result = []
        
        for subgroup in subgroups:
            result.append([constraint_map[constraint_id] for constraint_id in subgroup])
            
        return result
    
    def _find_connected_components(self, graph: Dict[str, Set[str]]) -> List[List[str]]:
        """
        Find connected components in a graph using DFS.
        
        Args:
            graph: Adjacency list representation of graph
            
        Returns:
            List of connected components (lists of node IDs)
        """
        components = []
        visited = set()
        
        def dfs(node, component):
            visited.add(node)
            component.append(node)
            
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, component)
        
        # Find connected components
        for node in graph:
            if node not in visited:
                component = []
                dfs(node, component)
                components.append(component)
                
        return components
    
    def _resolve_constraint_group(self, 
                                constraints: List[ComplianceConstraint], 
                                context: Dict[str, Any] = None) -> ConflictResolutionResult:
        """
        Resolve conflicts within a group of constraints.
        
        Args:
            constraints: Group of potentially conflicting constraints
            context: Context information for resolving conflicts
            
        Returns:
            Result of conflict resolution
        """
        if len(constraints) == 1:
            # No conflicts to resolve
            return ConflictResolutionResult(
                is_conflict_resolved=True,
                resolution_strategy=ConflictResolutionStrategy.PRIORITIZE_FRAMEWORK,
                winning_constraint_id=constraints[0].id,
                reason="Single constraint, no conflicts to resolve"
            )
            
        # Check cache first
        cache_key = self._make_cache_key(constraints, context)
        if cache_key in self.resolution_cache:
            return self.resolution_cache[cache_key]
            
        # Get constraint relationships
        relationships = self._get_constraint_relationships(constraints)
        
        # Choose resolution strategy
        strategy = self._determine_resolution_strategy(constraints, relationships, context)
        
        # Resolve based on strategy
        if strategy == ConflictResolutionStrategy.PRIORITIZE_FRAMEWORK:
            result = self._resolve_by_framework_priority(constraints, context)
        elif strategy == ConflictResolutionStrategy.MOST_RESTRICTIVE:
            result = self._resolve_by_restriction_level(constraints, True, context)
        elif strategy == ConflictResolutionStrategy.LEAST_RESTRICTIVE:
            result = self._resolve_by_restriction_level(constraints, False, context)
        elif strategy == ConflictResolutionStrategy.MERGE_CONSTRAINTS:
            result = self._resolve_by_merging(constraints, context)
        elif strategy == ConflictResolutionStrategy.CONTEXT_DEPENDENT:
            result = self._resolve_by_context(constraints, context)
        else:
            # Default strategy
            result = self._resolve_by_framework_priority(constraints, context)
            
        # Cache result
        self.resolution_cache[cache_key] = result
        return result
    
    def _make_cache_key(self, 
                      constraints: List[ComplianceConstraint], 
                      context: Dict[str, Any] = None) -> str:
        """Make a cache key for constraint resolution"""
        # Sort constraint IDs for consistent key
        constraint_ids = sorted([c.id for c in constraints])
        key_parts = ["|".join(constraint_ids)]
        
        # Add relevant context to key if available
        if context:
            # Only include context keys that affect resolution
            context_items = []
            for key in sorted(context.keys()):
                if key in ["domain", "user_type", "data_category", "operation"]:
                    value = context[key]
                    if isinstance(value, (str, int, float, bool, type(None))):
                        context_items.append(f"{key}:{value}")
            
            if context_items:
                key_parts.append("|".join(context_items))
                
        # Create hash of key parts for shorter key
        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_constraint_relationships(self, 
                                   constraints: List[ComplianceConstraint]) -> List[ConstraintRelationship]:
        """Get known relationships between constraints in a group"""
        relationships = []
        
        for i, constraint_a in enumerate(constraints):
            for constraint_b in constraints[i+1:]:
                key = self._make_relationship_key(constraint_a.id, constraint_b.id)
                
                if key in self.constraint_relationship_index:
                    relationships.append(self.constraint_relationship_index[key])
                    
        return relationships
    
    def _determine_resolution_strategy(self, 
                                     constraints: List[ComplianceConstraint],
                                     relationships: List[ConstraintRelationship],
                                     context: Dict[str, Any] = None) -> ConflictResolutionStrategy:
        """
        Determine the best strategy for resolving conflicts between constraints.
        
        Args:
            constraints: Group of potentially conflicting constraints
            relationships: Known relationships between constraints
            context: Context information for resolving conflicts
            
        Returns:
            Resolution strategy to use
        """
        # If there are explicit relationships with resolution strategies, use the most common
        if relationships:
            strategy_counts = {}
            for relationship in relationships:
                strategy = relationship.resolution_strategy
                if strategy not in strategy_counts:
                    strategy_counts[strategy] = 0
                strategy_counts[strategy] += 1
                
            if strategy_counts:
                # Use most common strategy
                return max(strategy_counts.items(), key=lambda x: x[1])[0]
                
        # If constraints are from different frameworks, use framework priority
        framework_ids = set(c.framework_id for c in constraints)
        if len(framework_ids) > 1:
            return ConflictResolutionStrategy.PRIORITIZE_FRAMEWORK
            
        # Check if constraints can be merged (same framework, same type)
        constraint_type = constraints[0].constraint_type
        if all(c.constraint_type == constraint_type for c in constraints):
            if constraint_type in ["data_minimization", "purpose_limitation", "storage_limitation"]:
                return ConflictResolutionStrategy.MERGE_CONSTRAINTS
                
        # Check for constraints where restriction level should be considered
        if constraint_type in ["access_control", "data_protection", "data_transfer"]:
            return ConflictResolutionStrategy.MOST_RESTRICTIVE
            
        # Check if context-dependent resolution is appropriate
        if context and context.get("domain"):
            return ConflictResolutionStrategy.CONTEXT_DEPENDENT
            
        # Default to framework priority
        return self.default_resolution_strategy
    
    def _resolve_by_framework_priority(self, 
                                     constraints: List[ComplianceConstraint],
                                     context: Dict[str, Any] = None) -> ConflictResolutionResult:
        """
        Resolve conflicts by determining which framework has higher priority.
        
        Args:
            constraints: Group of potentially conflicting constraints
            context: Context information for resolving conflicts
            
        Returns:
            Result of conflict resolution
        """
        # Get all unique framework pairs
        framework_ids = set(c.framework_id for c in constraints)
        if len(framework_ids) == 1:
            # All constraints from same framework, can't use framework priority
            return ConflictResolutionResult(
                is_conflict_resolved=False,
                resolution_strategy=ConflictResolutionStrategy.PRIORITIZE_FRAMEWORK,
                reason="All constraints from same framework"
            )
            
        # Find the winning framework
        framework_winners = {}
        for framework_a in framework_ids:
            framework_winners[framework_a] = 0
            
        # Compare each pair of frameworks
        for i, framework_a in enumerate(framework_ids):
            for framework_b in list(framework_ids)[i+1:]:
                winning_id, precedence, reason = self.get_framework_precedence(framework_a, framework_b)
                
                if winning_id:
                    framework_winners[winning_id] += 1
                    
        # Find framework with most wins
        if framework_winners:
            winning_framework = max(framework_winners.items(), key=lambda x: x[1])[0]
            
            # Find constraint from winning framework
            for constraint in constraints:
                if constraint.framework_id == winning_framework:
                    return ConflictResolutionResult(
                        is_conflict_resolved=True,
                        resolution_strategy=ConflictResolutionStrategy.PRIORITIZE_FRAMEWORK,
                        winning_constraint_id=constraint.id,
                        reason=f"Framework {winning_framework} has highest precedence"
                    )
        
        # No clear winner by framework precedence
        return ConflictResolutionResult(
            is_conflict_resolved=False,
            resolution_strategy=ConflictResolutionStrategy.PRIORITIZE_FRAMEWORK,
            reason="No clear framework precedence"
        )
        
    def _resolve_by_restriction_level(self, 
                                    constraints: List[ComplianceConstraint],
                                    most_restrictive: bool,
                                    context: Dict[str, Any] = None) -> ConflictResolutionResult:
        """
        Resolve conflicts by selecting the most or least restrictive constraint.
        
        Args:
            constraints: Group of potentially conflicting constraints
            most_restrictive: Whether to select most restrictive (True) or least (False)
            context: Context information for resolving conflicts
            
        Returns:
            Result of conflict resolution
        """
        if most_restrictive:
            # Find constraint with highest restriction level
            comparison_func = lambda x: x.restriction_level
            comparison_desc = "most restrictive"
        else:
            # Find constraint with lowest restriction level
            comparison_func = lambda x: -x.restriction_level
            comparison_desc = "least restrictive"
            
        # Find constraint with appropriate restriction level
        winning_constraint = max(constraints, key=comparison_func)
        
        return ConflictResolutionResult(
            is_conflict_resolved=True,
            resolution_strategy=(ConflictResolutionStrategy.MOST_RESTRICTIVE if most_restrictive 
                              else ConflictResolutionStrategy.LEAST_RESTRICTIVE),
            winning_constraint_id=winning_constraint.id,
            reason=f"Selected {comparison_desc} constraint with level {winning_constraint.restriction_level}"
        )
    
    def _resolve_by_merging(self, 
                          constraints: List[ComplianceConstraint],
                          context: Dict[str, Any] = None) -> ConflictResolutionResult:
        """
        Resolve conflicts by merging compatible constraints.
        
        Args:
            constraints: Group of potentially conflicting constraints
            context: Context information for resolving conflicts
            
        Returns:
            Result of conflict resolution
        """
        # Ensure all constraints are of same type
        constraint_type = constraints[0].constraint_type
        if not all(c.constraint_type == constraint_type for c in constraints):
            return ConflictResolutionResult(
                is_conflict_resolved=False,
                resolution_strategy=ConflictResolutionStrategy.MERGE_CONSTRAINTS,
                reason="Cannot merge constraints of different types"
            )
            
        # Extract scope from all constraints
        all_scope = set()
        for constraint in constraints:
            all_scope.update(constraint.scope)
            
        # Extract applicable entities from all constraints
        all_entities = set()
        for constraint in constraints:
            all_entities.update(constraint.applicable_entities)
            
        # Extract exception conditions from all constraints
        all_exceptions = []
        for constraint in constraints:
            all_exceptions.extend(constraint.exception_conditions)
            
        # Create merged constraint
        framework_ids = [c.framework_id for c in constraints]
        constraint_ids = [c.id for c in constraints]
        
        # Generate merged ID
        merged_id = f"merged_{'_'.join(sorted(constraint_ids))}"
        
        # Determine restriction level for merged constraint
        # Usually take the most restrictive level
        restriction_level = max(c.restriction_level for c in constraints)
        
        # Determine severity for merged constraint
        # Usually take the highest severity
        severity_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        severity = max(constraints, key=lambda c: severity_order.get(c.severity, 0)).severity
        
        # Merge implementation details if possible
        implementation = {}
        for constraint in constraints:
            implementation.update(constraint.implementation)
            
        # Create merged constraint
        merged_constraint = ComplianceConstraint(
            id=merged_id,
            framework_id="+".join(sorted(set(framework_ids))),
            description=f"Merged constraint from {len(constraints)} source constraints",
            constraint_type=constraint_type,
            severity=severity,
            restriction_level=restriction_level,
            scope=list(all_scope),
            applicable_entities=list(all_entities),
            exception_conditions=all_exceptions,
            implementation=implementation
        )
        
        return ConflictResolutionResult(
            is_conflict_resolved=True,
            resolution_strategy=ConflictResolutionStrategy.MERGE_CONSTRAINTS,
            merged_constraint=merged_constraint,
            reason=f"Merged {len(constraints)} constraints into unified constraint"
        )
    
    def _resolve_by_context(self, 
                          constraints: List[ComplianceConstraint],
                          context: Dict[str, Any] = None) -> ConflictResolutionResult:
        """
        Resolve conflicts based on context information.
        
        Args:
            constraints: Group of potentially conflicting constraints
            context: Context information for resolving conflicts
            
        Returns:
            Result of conflict resolution
        """
        if not context:
            return ConflictResolutionResult(
                is_conflict_resolved=False,
                resolution_strategy=ConflictResolutionStrategy.CONTEXT_DEPENDENT,
                reason="No context information provided"
            )
            
        # Check domain-specific applicability
        domain = context.get("domain")
        if domain:
            # Find constraint whose framework is specific to this domain
            domain_constraint = None
            for constraint in constraints:
                framework_id = constraint.framework_id
                if framework_id in self.framework_metadata:
                    metadata = self.framework_metadata[framework_id]
                    if metadata.is_domain_specific and metadata.domain == domain:
                        domain_constraint = constraint
                        break
                        
            if domain_constraint:
                return ConflictResolutionResult(
                    is_conflict_resolved=True,
                    resolution_strategy=ConflictResolutionStrategy.CONTEXT_DEPENDENT,
                    winning_constraint_id=domain_constraint.id,
                    reason=f"Selected domain-specific constraint for {domain}"
                )
                
        # Look for constraints with matching scope
        for key, value in context.items():
            for constraint in constraints:
                if key in constraint.scope:
                    return ConflictResolutionResult(
                        is_conflict_resolved=True,
                        resolution_strategy=ConflictResolutionStrategy.CONTEXT_DEPENDENT,
                        winning_constraint_id=constraint.id,
                        reason=f"Selected constraint with matching scope for {key}"
                    )
                    
        # No context-based resolution possible
        return ConflictResolutionResult(
            is_conflict_resolved=False,
            resolution_strategy=ConflictResolutionStrategy.CONTEXT_DEPENDENT,
            reason="No context-based resolution found"
        )
    
    def visualize_conflict_resolution(self, constraints, resolution_result, output_format="html"):
        """
        Visualize the conflict resolution process for easier understanding.
        
        Args:
            constraints: List of constraints that were resolved
            resolution_result: Result of conflict resolution
            output_format: "html", "svg", or "json"
            
        Returns:
            Visualization in the specified format
        """
        if output_format == "html":
            return self._generate_html_visualization(constraints, resolution_result)
        elif output_format == "svg":
            return self._generate_svg_visualization(constraints, resolution_result)
        elif output_format == "json":
            return self._generate_json_visualization(constraints, resolution_result)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
            
    def _generate_html_visualization(self, constraints, resolution_result):
        """Generate HTML visualization of conflict resolution"""
        html = ["<!DOCTYPE html>", "<html>", "<head>"]
        html.append("<style>")
        html.append("body { font-family: Arial, sans-serif; }")
        html.append(".constraint { border: 1px solid #ccc; margin: 10px; padding: 10px; border-radius: 5px; }")
        html.append(".constraint.winning { border-color: green; background-color: #e8f5e9; }")
        html.append(".constraint.merged { border-color: blue; background-color: #e3f2fd; }")
        html.append(".constraint.conflicting { border-color: orange; background-color: #fff3e0; }")
        html.append(".resolution { margin: 20px; padding: 15px; background-color: #f5f5f5; border-radius: 5px; }")
        html.append("</style>")
        html.append("</head>")
        html.append("<body>")
        
        # Add header
        html.append("<h1>Constraint Conflict Resolution</h1>")
        
        # Add original constraints
        html.append("<h2>Original Constraints</h2>")
        for constraint in constraints:
            css_class = "constraint"
            if resolution_result.winning_constraint_id == constraint.id:
                css_class += " winning"
            elif resolution_result.merged_constraint and f"merged_{constraint.id}" in resolution_result.merged_constraint.id:
                css_class += " merged"
            else:
                css_class += " conflicting"
                
            html.append(f'<div class="{css_class}">')
            html.append(f'<h3>{constraint.id}</h3>')
            html.append(f'<p><strong>Framework:</strong> {constraint.framework_id}</p>')
            html.append(f'<p><strong>Description:</strong> {constraint.description}</p>')
            html.append(f'<p><strong>Type:</strong> {constraint.constraint_type}</p>')
            html.append(f'<p><strong>Severity:</strong> {constraint.severity}</p>')
            html.append(f'<p><strong>Restriction Level:</strong> {constraint.restriction_level}</p>')
            html.append('</div>')
        
        # Add resolution result
        html.append("<h2>Resolution Result</h2>")
        html.append('<div class="resolution">')
        html.append(f'<p><strong>Resolution Strategy:</strong> {resolution_result.resolution_strategy.value}</p>')
        html.append(f'<p><strong>Reason:</strong> {resolution_result.reason}</p>')
        
        if resolution_result.winning_constraint_id:
            html.append(f'<p><strong>Winning Constraint:</strong> {resolution_result.winning_constraint_id}</p>')
        
        if resolution_result.merged_constraint:
            merged = resolution_result.merged_constraint
            html.append('<div class="constraint merged">')
            html.append(f'<h3>Merged Constraint: {merged.id}</h3>')
            html.append(f'<p><strong>Description:</strong> {merged.description}</p>')
            html.append(f'<p><strong>Type:</strong> {merged.constraint_type}</p>')
            html.append(f'<p><strong>Severity:</strong> {merged.severity}</p>')
            html.append(f'<p><strong>Restriction Level:</strong> {merged.restriction_level}</p>')
            html.append('</div>')
        
        html.append('</div>')
        html.append("</body>")
        html.append("</html>")
        
        return "\n".join(html)

    def _generate_svg_visualization(self, constraints, resolution_result):
        """Generate SVG visualization of conflict resolution"""
        # Implementation would create an SVG diagram showing relationships
        # between constraints and how they were resolved
        return "<svg>...</svg>"  # Placeholder
        
    def _generate_json_visualization(self, constraints, resolution_result):
        """Generate JSON visualization data for conflict resolution"""
        visualization_data = {
            "original_constraints": [c.to_dict() for c in constraints],
            "resolution_result": {
                "strategy": resolution_result.resolution_strategy.value,
                "reason": resolution_result.reason,
                "winning_constraint_id": resolution_result.winning_constraint_id,
                "is_resolved": resolution_result.is_conflict_resolved
            }
        }
        
        if resolution_result.merged_constraint:
            visualization_data["merged_constraint"] = resolution_result.merged_constraint.to_dict()
            
        return json.dumps(visualization_data, indent=2)
    


# Example initialization with predefined regulatory framework metadata
def initialize_regulation_conflict_resolver():
    """Initialize a conflict resolver with predefined regulatory frameworks"""
    # Define metadata for common regulatory frameworks
    framework_metadata = {
        "GDPR": RegulationMetadata(
            id="GDPR",
            name="General Data Protection Regulation",
            description="EU regulation on data protection and privacy",
            version="2016/679",
            issued_date="2016-04-27",
            jurisdiction=["EU", "EEA"],
            sector=["all"],
            precedence_level=2,
            is_domain_specific=False
        ),
        "HIPAA": RegulationMetadata(
            id="HIPAA",
            name="Health Insurance Portability and Accountability Act",
            description="US regulation for medical information privacy",
            version="1996",
            issued_date="1996-08-21",
            jurisdiction=["US"],
            sector=["healthcare"],
            precedence_level=2,
            is_domain_specific=True,
            domain="healthcare"
        ),
        "CCPA": RegulationMetadata(
            id="CCPA",
            name="California Consumer Privacy Act",
            description="California law on consumer data privacy",
            version="2018",
            issued_date="2018-06-28",
            jurisdiction=["US-CA"],
            sector=["all"],
            precedence_level=3,
            is_domain_specific=False
        ),
        "GLBA": RegulationMetadata(
            id="GLBA",
            name="Gramm-Leach-Bliley Act",
            description="US regulation for financial data privacy",
            version="1999",
            issued_date="1999-11-12",
            jurisdiction=["US"],
            sector=["finance"],
            precedence_level=3,
            is_domain_specific=True,
            domain="finance"
        ),
        "PCI-DSS": RegulationMetadata(
            id="PCI-DSS",
            name="Payment Card Industry Data Security Standard",
            description="Security standard for payment card processing",
            version="3.2.1",
            issued_date="2018-05-01",
            jurisdiction=["global"],
            sector=["finance", "retail"],
            precedence_level=4,
            is_domain_specific=True,
            domain="payment_processing"
        )
    }
    
    # Define precedence rules
    precedence_rules = {
        "HIPAA": {
            "GDPR": RegulationPrecedence.DOMAIN  # HIPAA takes precedence over GDPR for healthcare data
        },
        "GDPR": {
            "CCPA": RegulationPrecedence.STRICTER  # GDPR is stricter than CCPA
        },
        "GLBA": {
            "CCPA": RegulationPrecedence.DOMAIN  # GLBA takes precedence for financial institutions
        }
    }
    
    # Define some example constraint relationships
    constraint_relationships = [
        ConstraintRelationship(
            constraint_a_id="GDPR:data_minimization:1",
            constraint_b_id="CCPA:data_collection:1",
            relationship_type="overlaps",
            description="Overlapping requirements for data collection and minimization",
            resolution_strategy=ConflictResolutionStrategy.MERGE_CONSTRAINTS,
            framework_a_id="GDPR",
            framework_b_id="CCPA"
        ),
        ConstraintRelationship(
            constraint_a_id="HIPAA:phi_protection:1",
            constraint_b_id="GDPR:sensitive_data:1",
            relationship_type="contradicts",
            description="Different requirements for health data protection",
            resolution_strategy=ConflictResolutionStrategy.PRIORITIZE_FRAMEWORK,
            framework_a_id="HIPAA",
            framework_b_id="GDPR"
        )
    ]
    
    # Create and return the resolver
    return RegulationConflictResolver(
        framework_metadata=framework_metadata,
        precedence_rules=precedence_rules,
        default_resolution_strategy=ConflictResolutionStrategy.MOST_RESTRICTIVE,
        constraint_relationships=constraint_relationships
    )


# Example usage:
# resolver = initialize_regulation_conflict_resolver()
#
# # Define some constraints
# constraint1 = ComplianceConstraint(
#     id="GDPR:data_retention:1",
#     framework_id="GDPR",
#     description="GDPR data retention limitation",
#     constraint_type="storage_limitation",
#     severity="medium",
#     restriction_level=0.7,
#     scope=["data_storage", "retention_period"],
#     applicable_entities=["PII"]
# )
#
# constraint2 = ComplianceConstraint(
#     id="CCPA:data_deletion:1",
#     framework_id="CCPA",
#     description="CCPA data deletion requirement",
#     constraint_type="storage_limitation",
#     severity="medium",
#     restriction_level=0.5,
#     scope=["data_deletion", "retention_period"],
#     applicable_entities=["PII"]
# )
#
# # Resolve conflicts
# resolved = resolver.resolve_conflicts([constraint1, constraint2])
# print(f"Resolved {len(resolved)} constraints")
