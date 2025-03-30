
from src.rules import TextPatternRules
from src.rules import SemanticRules
from src.rules import EntityProtectionRules
from src.rules import DomainSpecificRules
from src.compliance import ComplianceImpactAnalyzer
from src.compliance import RegulatoryKnowledgeBase
from src.compliance import ComplianceFramework
from src.compliance import ComplianceImpactAnalysis

class LanguageModelRegulatoryExtensions:
    """
    Extensions to the Regulatory Knowledge Base for language model content,
    including specialized rules for text generation, natural language patterns,
    and context-aware compliance enforcement.
    """
    def __init__(self, base_regulatory_kb):
        """
        Initialize the language model regulatory extensions.
        
        Args:
            base_regulatory_kb: Base regulatory knowledge base from SocratesNS
        """
        self.base_kb = base_regulatory_kb
        
        # Initialize specialized components
        self.pattern_rules = TextPatternRules()
        self.semantic_rules = SemanticRules()
        self.entity_rules = EntityProtectionRules()
        self.domain_specific_rules = DomainSpecificRules()
        self.compliance_impact_analyzer = ComplianceImpactAnalyzer()
        
    def integrate_language_extensions(self):
        """
        Integrate language-specific extensions into the base regulatory knowledge base.
        
        Returns:
            Enhanced regulatory knowledge base
        """
        # For each framework in the base KB, add language-specific extensions
        for framework_id in self.base_kb.get_all_framework_ids():
            framework = self.base_kb.get_framework(framework_id)
            
            # Add text pattern rules
            pattern_extensions = self.pattern_rules.get_rules_for_framework(framework_id)
            framework.add_rules(pattern_extensions, category="text_patterns")
            
            # Add semantic rules
            semantic_extensions = self.semantic_rules.get_rules_for_framework(framework_id)
            framework.add_rules(semantic_extensions, category="semantic")
            
            # Add entity protection rules
            entity_extensions = self.entity_rules.get_rules_for_framework(framework_id)
            framework.add_rules(entity_extensions, category="entity_protection")
            
            # Add domain-specific language rules
            domain_extensions = self.domain_specific_rules.get_rules_for_framework(framework_id)
            framework.add_rules(domain_extensions, category="domain_specific")
        
        # Update conflict resolution to handle language-specific rule conflicts
        self.update_conflict_resolution()
        
        return self.base_kb
    
    def update_conflict_resolution(self):
        """
        Update conflict resolution strategies to handle language-specific rule conflicts.
        """
        # Add specialized resolution strategies for text pattern conflicts
        pattern_resolution = self.pattern_rules.get_conflict_resolution_strategy()
        self.base_kb.add_conflict_resolution_strategy("text_pattern", pattern_resolution)
        
        # Add specialized resolution strategies for semantic conflicts
        semantic_resolution = self.semantic_rules.get_conflict_resolution_strategy()
        self.base_kb.add_conflict_resolution_strategy("semantic", semantic_resolution)
        
        # Add specialized resolution strategies for entity protection conflicts
        entity_resolution = self.entity_rules.get_conflict_resolution_strategy()
        self.base_kb.add_conflict_resolution_strategy("entity_protection", entity_resolution)
    
    def get_applicable_text_rules(self, text, context, compliance_mode='strict'):
        """
        Get applicable regulatory rules for a specific text and context.
        
        Args:
            text: The text to evaluate
            context: Additional context (e.g., domain, user role)
            compliance_mode: 'strict' or 'soft' enforcement
            
        Returns:
            List of applicable regulatory rules
        """
        # Get applicable frameworks
        applicable_frameworks = self.base_kb.get_applicable_frameworks(text, context, compliance_mode)
        
        # Get applicable rules from each framework
        applicable_rules = []
        for framework in applicable_frameworks:
            # Get text pattern rules
            pattern_rules = framework.get_rules_by_category("text_patterns")
            applicable_pattern_rules = self.filter_applicable_rules(pattern_rules, text, context, 
                                                                   compliance_mode)
            applicable_rules.extend(applicable_pattern_rules)
            
            # Get semantic rules
            semantic_rules = framework.get_rules_by_category("semantic")
            applicable_semantic_rules = self.filter_applicable_rules(semantic_rules, text, context, 
                                                                    compliance_mode)
            applicable_rules.extend(applicable_semantic_rules)
            
            # Get entity protection rules
            entity_rules = framework.get_rules_by_category("entity_protection")
            applicable_entity_rules = self.filter_applicable_rules(entity_rules, text, context, 
                                                                 compliance_mode)
            applicable_rules.extend(applicable_entity_rules)
            
            # Get domain-specific rules
            domain_rules = framework.get_rules_by_category("domain_specific")
            applicable_domain_rules = self.filter_applicable_rules(domain_rules, text, context, 
                                                                 compliance_mode)
            applicable_rules.extend(applicable_domain_rules)
        
        # Resolve any conflicts between the rules
        resolved_rules = self.base_kb.resolve_conflicts(applicable_rules)
        
        return resolved_rules
    
    def filter_applicable_rules(self, rules, text, context, compliance_mode):
        """
        Filter rules to those applicable to the given text and context.
        
        Args:
            rules: List of regulatory rules
            text: The text to evaluate
            context: Additional context
            compliance_mode: 'strict' or 'soft' enforcement
            
        Returns:
            List of applicable rules
        """
        applicable_rules = []
        
        for rule in rules:
            # Check if rule applies to this text and context
            if self.rule_applies(rule, text, context, compliance_mode):
                applicable_rules.append(rule)
        
        return applicable_rules
    
    def rule_applies(self, rule, text, context, compliance_mode):
        """
        Determine if a rule applies to the given text and context.
        
        Args:
            rule: Regulatory rule
            text: The text to evaluate
            context: Additional context
            compliance_mode: 'strict' or 'soft' enforcement
            
        Returns:
            Boolean indicating if rule applies
        """
        # Check rule conditions
        for condition in rule.get('conditions', []):
            if not self.condition_applies(condition, text, context):
                return False
        
        # Check rule applicability based on compliance mode
        if compliance_mode == 'soft' and rule.get('enforcement_level', 'mandatory') == 'optional':
            return False
        
        return True
    
    def condition_applies(self, condition, text, context):
        """
        Check if a condition applies to the text and context.
        
        Args:
            condition: Rule condition
            text: The text to evaluate
            context: Additional context
            
        Returns:
            Boolean indicating if condition applies
        """
        condition_type = condition.get('type')
        
        if condition_type == 'text_contains':
            pattern = condition.get('pattern')
            return pattern in text
        
        elif condition_type == 'text_matches_regex':
            regex = condition.get('regex')
            # Would use regex matching here
            return True  # Placeholder
        
        elif condition_type == 'domain_specific':
            domain = condition.get('domain')
            return domain == context.get('domain')
        
        elif condition_type == 'entity_present':
            entity_type = condition.get('entity_type')
            # Would check if entity of this type is in the text
            return False  # Placeholder
        
        elif condition_type == 'semantic_concept':
            concept = condition.get('concept')
            # Would check if semantic concept is present
            return False  # Placeholder
        
        # Default case
        return True
    
    def analyze_compliance_impact(self, text, context, compliance_mode='strict'):
        """
        Analyze the compliance impact of text against all relevant regulations.
        
        Args:
            text: The text to evaluate
            context: Additional context
            compliance_mode: 'strict' or 'soft' enforcement
            
        Returns:
            Compliance impact analysis
        """
        return self.compliance_impact_analyzer.analyze(text, context, self, compliance_mode)
