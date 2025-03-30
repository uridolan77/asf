import re
import importlib
import datetime
import logging
import src.compliance.frameworks.ccpa as CCPAComplianceProcessor
import src.compliance.frameworks.gdpr as GDPRComplianceProcessor
from src.compliance.violation_handlers import (AnonymizeViolationHandler, BlockViolationHandler, FlagViolationHandler, LogViolationHandler, ModifyContentViolationHandler)
from compliance.frameworks.hipaa import HIPAAComplianceProcessor
from src.compliance.compliance_audit_log import ComplianceAuditLog  # Import ComplianceAuditLog
# import src.compliance.frameworks.content_policy as ContentPolicyComplianceProcessor

class EnhancedComplianceVerifier:
    """
    Enhanced compliance verification with rule-based systems, contextual awareness,
    and support for regulatory frameworks.
    """
    def __init__(self, compliance_config):
        self.config = compliance_config
        self.rules = self._initialize_rules()
        self.framework_processors = self._initialize_framework_processors()
        self.violation_handlers = self._initialize_violation_handlers()
        self.compliance_history = ComplianceAuditLog()
        
    def _initialize_rules(self):
        """Initialize compliance rules from config or default set"""
        # Load rules from configuration
        config_rules = self.config.get("compliance_rules", {})
        
        # Default rules for common compliance requirements
        default_rules = {
            'pii_protection': {
                'id': 'rule001',
                'name': 'PII Protection Rule',
                'description': 'Verifies that PII is properly protected',
                'severity': 'high',
                'entity_types': ['PII'],
                'conditions': [
                    {'type': 'entity_present', 'entity_type': 'PII', 'action': 'check_protection'},
                    {'type': 'relation_check', 'relation_type': 'discloses', 'source_type': 'PII', 'action': 'block'}
                ],
                'exceptions': [
                    {'type': 'context', 'context_type': 'anonymized', 'value': True}
                ],
                'frameworks': ['GDPR', 'CCPA', 'HIPAA']
            },
            'phi_protection': {
                'id': 'rule002',
                'name': 'PHI Protection Rule',
                'description': 'Verifies that PHI is properly protected',
                'severity': 'high',
                'entity_types': ['PHI'],
                'conditions': [
                    {'type': 'entity_present', 'entity_type': 'PHI', 'action': 'check_protection'},
                    {'type': 'relation_check', 'relation_type': 'discloses', 'source_type': 'PHI', 'action': 'block'}
                ],
                'exceptions': [
                    {'type': 'context', 'context_type': 'anonymized', 'value': True},
                    {'type': 'context', 'context_type': 'authorized_healthcare_operation', 'value': True}
                ],
                'frameworks': ['HIPAA']
            },
            'consent_verification': {
                'id': 'rule003',
                'name': 'Consent Verification Rule',
                'description': 'Verifies that appropriate consent is obtained before data processing',
                'severity': 'high',
                'entity_types': ['PII', 'PHI'],
                'conditions': [
                    {'type': 'concept_check', 'concept': 'data_processing', 'action': 'check_consent'},
                    {'type': 'entity_present', 'entity_type': 'PII', 'action': 'check_consent'}
                ],
                'exceptions': [
                    {'type': 'context', 'context_type': 'legitimate_interest', 'value': True},
                    {'type': 'context', 'context_type': 'vital_interest', 'value': True},
                    {'type': 'context', 'context_type': 'public_interest', 'value': True}
                ],
                'frameworks': ['GDPR', 'CCPA']
            },
            'data_minimization': {
                'id': 'rule004',
                'name': 'Data Minimization Rule',
                'description': 'Verifies that only necessary data is collected and processed',
                'severity': 'medium',
                'entity_types': ['PII', 'PHI'],
                'conditions': [
                    {'type': 'entity_count', 'entity_type': 'PII', 'threshold': 5, 'action': 'flag'},
                    {'type': 'concept_check', 'concept': 'data_minimization', 'min_activation': 0.6}
                ],
                'exceptions': [
                    {'type': 'context', 'context_type': 'comprehensive_assessment', 'value': True}
                ],
                'frameworks': ['GDPR']
            },
            'harmful_content': {
                'id': 'rule005',
                'name': 'Harmful Content Rule',
                'description': 'Identifies and flags potentially harmful content',
                'severity': 'high',
                'entity_types': [],
                'conditions': [
                    {'type': 'concept_check', 'concept': 'harmful_offensive', 'threshold': 0.7, 'action': 'block'},
                    {'type': 'concept_check', 'concept': 'harmful_dangerous', 'threshold': 0.6, 'action': 'block'},
                    {'type': 'concept_check', 'concept': 'harmful_misleading', 'threshold': 0.8, 'action': 'block'}
                ],
                'exceptions': [
                    {'type': 'context', 'context_type': 'educational', 'value': True},
                    {'type': 'context', 'context_type': 'research', 'value': True}
                ],
                'frameworks': ['CONTENT_POLICY']
            }
        }
        
        # Merge provided rules with defaults, preserving user-defined rules
        rules = {}
        for rule_id, rule in default_rules.items():
            if rule_id not in config_rules:
                rules[rule_id] = rule
            else:
                # User-defined rule takes precedence
                rules[rule_id] = config_rules[rule_id]
        
        # Add any additional user-defined rules
        for rule_id, rule in config_rules.items():
            if rule_id not in rules:
                rules[rule_id] = rule
                
        return rules
    
    def _initialize_framework_processors(self):
        """Initialize specialized processors for regulatory frameworks"""
        processors = {}
        
        # Initialize GDPR processor if enabled
        if self.config.get("enable_gdpr", True):
            processors['GDPR'] = GDPRComplianceProcessor(self.config.get("gdpr_config", {}))
            
        # Initialize HIPAA processor if enabled
        if self.config.get("enable_hipaa", True):
            processors['HIPAA'] = HIPAAComplianceProcessor(self.config.get("hipaa_config", {}))
            
        # Initialize CCPA processor if enabled
        if self.config.get("enable_ccpa", True):
            processors['CCPA'] = CCPAComplianceProcessor(self.config.get("ccpa_config", {}))
            
        return processors
    
    def _initialize_violation_handlers(self):
        """Initialize handlers for compliance violations"""
        handlers = {
            'block': BlockViolationHandler(),
            'flag': FlagViolationHandler(),
            'log': LogViolationHandler(),
            'anonymize': AnonymizeViolationHandler(),
            'modify': ModifyContentViolationHandler()
        }
        
        # Add custom handlers from config
        custom_handlers = self.config.get("custom_violation_handlers", {})
        for handler_id, handler_class in custom_handlers.items():
            try:
                # Dynamically load and instantiate custom handler
                handler = self._load_custom_handler(handler_class)
                handlers[handler_id] = handler
            except Exception as e:
                logging.warning(f"Failed to load custom violation handler {handler_id}: {str(e)}")
                
        return handlers
    
    def _load_custom_handler(self, class_path):
        """Dynamically load a custom violation handler class"""
        try:
            module_path, class_name = class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            handler_class = getattr(module, class_name)
            return handler_class()
        except Exception as e:
            raise ValueError(f"Cannot load custom handler class {class_path}: {str(e)}")
    
    def verify_content(self, content, content_type="text", compliance_mode="strict", context=None):
        """
        Verify content compliance against applicable rules and frameworks
        
        Args:
            content: Content to verify (text, entities, concepts, etc.)
            content_type: Type of content ('text', 'entity', 'relation', etc.)
            compliance_mode: 'strict' or 'soft' enforcement
            context: Additional context information
            
        Returns:
            Compliance verification result with details
        """
        # Initialize verification context
        verification_context = self._create_verification_context(content, content_type, context)
        
        # Determine applicable rules based on content type and context
        applicable_rules = self._get_applicable_rules(content_type, context)
        
        # Determine applicable regulatory frameworks
        applicable_frameworks = self._get_applicable_frameworks(context)
        
        # Apply rules to verify compliance
        rule_results = self._apply_rules(applicable_rules, verification_context, compliance_mode)
        
        # Apply framework-specific verification if needed
        framework_results = self._apply_framework_verification(
            applicable_frameworks, verification_context, compliance_mode
        )
        
        # Aggregate results and determine overall compliance
        compliance_result = self._aggregate_compliance_results(
            rule_results, framework_results, compliance_mode
        )
        
        # Log the compliance check
        self.compliance_history.log_compliance_check(
            content=content,
            content_type=content_type,
            context=context,
            compliance_mode=compliance_mode,
            result=compliance_result
        )
        
        return compliance_result
    
    def _create_verification_context(self, content, content_type, context=None):
        """Create context object for verification"""
        # Initialize with provided context or empty dict
        verification_context = context or {}
        
        # Add content and type information
        verification_context['content'] = content
        verification_context['content_type'] = content_type
        
        # Extract additional context based on content type
        if content_type == "text":
            # Extract additional info from text if needed
            pass
        elif content_type == "entity":
            # Extract content from entity if needed
            verification_context['entity_type'] = content.get('type', 'UNKNOWN')
        elif content_type == "relation":
            # Extract content from relation if needed
            verification_context['relation_type'] = content.get('type', 'UNKNOWN')
        elif content_type == "concept":
            # Extract content from concept if needed
            verification_context['concept_activation'] = content.get('activation', 0.0)
        
        # Add timestamp
        verification_context['timestamp'] = datetime.datetime.now().isoformat()
        
        return verification_context
    
    def _get_applicable_rules(self, content_type, context=None):
        """Get rules applicable to current content and context"""
        applicable_rules = []
        
        for rule_id, rule in self.rules.items():
            # Check if rule applies to this content type
            if self._rule_applies_to_content_type(rule, content_type):
                # Check if rule applies to this context
                if self._rule_applies_to_context(rule, context):
                    applicable_rules.append(rule)
        
        return applicable_rules
    
    def _rule_applies_to_content_type(self, rule, content_type):
        """Check if rule applies to specific content type"""
        # If rule has no specific content_types, it applies to all
        if 'content_types' not in rule:
            return True
            
        # Check if this content type is in the rule's list
        return content_type in rule['content_types']
    
    def _rule_applies_to_context(self, rule, context):
        """Check if rule applies to specific context"""
        if not context:
            # No context provided, rule applies by default
            return True
            
        # Check if rule has context requirements
        if 'context_requirements' in rule:
            for req in rule['context_requirements']:
                # Get the required context key and value
                key = req.get('key')
                value = req.get('value')
                
                if key not in context:
                    # Required context key is missing
                    return False
                    
                if value is not None and context[key] != value:
                    # Context value doesn't match
                    return False
        
        return True
    
    def _get_applicable_frameworks(self, context=None):
        """Determine applicable regulatory frameworks based on context"""
        applicable_frameworks = []
        
        # Check for explicit framework specification in context
        if context and 'regulatory_frameworks' in context:
            specified_frameworks = context['regulatory_frameworks']
            if isinstance(specified_frameworks, list):
                # Add all specified frameworks that we have processors for
                for framework in specified_frameworks:
                    if framework in self.framework_processors:
                        applicable_frameworks.append(framework)
            elif isinstance(specified_frameworks, str) and specified_frameworks in self.framework_processors:
                # Single framework specified as string
                applicable_frameworks.append(specified_frameworks)
                
        # If no frameworks specified or found, apply all enabled frameworks
        if not applicable_frameworks:
            applicable_frameworks = list(self.framework_processors.keys())
            
        return applicable_frameworks
    
    def _apply_rules(self, rules, context, compliance_mode):
        """Apply compliance rules to verify content"""
        rule_results = []
        
        for rule in rules:
            rule_result = self._apply_rule(rule, context, compliance_mode)
            rule_results.append(rule_result)
        
        return rule_results
    
    def _apply_rule(self, rule, context, compliance_mode):
        """Apply a single compliance rule"""
        # Check for rule exceptions first
        if self._check_rule_exceptions(rule, context):
            # Rule doesn't apply due to exception
            return {
                'rule_id': rule['id'],
                'name': rule['name'],
                'is_compliant': True,
                'reason': 'exception_applied',
                'details': {
                    'exception': 'Rule exception applies to this context'
                }
            }
        
        # Check each condition in the rule
        condition_results = []
        for condition in rule.get('conditions', []):
            condition_result = self._check_condition(condition, context)
            condition_results.append(condition_result)
        
        # Determine if rule is satisfied based on condition results
        rule_compliance = self._evaluate_rule_compliance(rule, condition_results, compliance_mode)
        
        # Create detailed rule result
        rule_result = {
            'rule_id': rule['id'],
            'name': rule['name'],
            'is_compliant': rule_compliance['is_compliant'],
            'compliance_score': rule_compliance['compliance_score'],
            'severity': rule.get('severity', 'medium'),
            'condition_results': condition_results,
            'reason': rule_compliance['reason']
        }
        
        # Add violation details if not compliant
        if not rule_result['is_compliant']:
            rule_result['violation'] = {
                'type': rule_compliance.get('violation_type', 'rule_violation'),
                'description': rule_compliance.get('violation_description', f"Violation of rule: {rule['name']}"),
                'recommended_action': rule_compliance.get('recommended_action', 'review')
            }
        
        return rule_result
    
    def _check_rule_exceptions(self, rule, context):
        """Check if any rule exceptions apply to this context"""
        for exception in rule.get('exceptions', []):
            if self._exception_applies(exception, context):
                return True
        return False
    
    def _exception_applies(self, exception, context):
        """Check if an exception applies to this context"""
        exception_type = exception.get('type')
        
        if exception_type == 'context':
            # Context-based exception
            context_type = exception.get('context_type')
            expected_value = exception.get('value')
            
            if context_type in context and context[context_type] == expected_value:
                return True
        elif exception_type == 'entity_type':
            # Entity type exception
            if context.get('content_type') == 'entity':
                entity_type = context.get('entity_type')
                excepted_types = exception.get('entity_types', [])
                
                if entity_type in excepted_types:
                    return True
        elif exception_type == 'custom':
            # Custom exception logic
            custom_check = exception.get('check_function')
            if custom_check and callable(custom_check):
                return custom_check(context)
                
        return False
    
    def _check_condition(self, condition, context):
        """Check if a condition is satisfied"""
        condition_type = condition.get('type')
        
        # Initialize result structure
        condition_result = {
            'type': condition_type,
            'is_satisfied': False,
            'details': {}
        }
        
        if condition_type == 'entity_present':
            # Check if entity of specified type is present
            result = self._check_entity_present_condition(condition, context)
            condition_result.update(result)
        elif condition_type == 'entity_count':
            # Check entity count against threshold
            result = self._check_entity_count_condition(condition, context)
            condition_result.update(result)
        elif condition_type == 'relation_check':
            # Check for specific relations
            result = self._check_relation_condition(condition, context)
            condition_result.update(result)
        elif condition_type == 'concept_check':
            # Check concept activation
            result = self._check_concept_condition(condition, context)
            condition_result.update(result)
        elif condition_type == 'text_pattern':
            # Check for text patterns
            result = self._check_text_pattern_condition(condition, context)
            condition_result.update(result)
        elif condition_type == 'content_classification':
            # Check content classification
            result = self._check_classification_condition(condition, context)
            condition_result.update(result)
        elif condition_type == 'custom':
            # Custom condition check
            custom_check = condition.get('check_function')
            if custom_check and callable(custom_check):
                is_satisfied, details = custom_check(context)
                condition_result['is_satisfied'] = is_satisfied
                condition_result['details'] = details
                
        # Add recommended action if condition not satisfied
        if not condition_result['is_satisfied'] and 'action' in condition:
            condition_result['action'] = condition['action']
            
        return condition_result
    
    def _check_entity_present_condition(self, condition, context):
        """Check if entity of specified type is present"""
        if context['content_type'] == 'entity':
            # Direct entity check
            entity_type = context.get('entity_type')
            required_type = condition.get('entity_type')
            
            is_satisfied = entity_type == required_type
            return {
                'is_satisfied': is_satisfied,
                'details': {
                    'entity_type': entity_type,
                    'required_type': required_type
                }
            }
        elif 'entities' in context:
            # Check in list of entities
            entities = context['entities']
            required_type = condition.get('entity_type')
            
            matching_entities = [e for e in entities if e.get('type') == required_type]
            is_satisfied = len(matching_entities) > 0
            
            return {
                'is_satisfied': is_satisfied,
                'details': {
                    'matching_entities': len(matching_entities),
                    'required_type': required_type
                }
            }
            
        return {'is_satisfied': False, 'details': {'reason': 'no_entities_available'}}
    
    def _check_entity_count_condition(self, condition, context):
        """Check entity count against threshold"""
        if 'entities' not in context:
            return {'is_satisfied': True, 'details': {'reason': 'no_entities_available'}}
            
        entities = context['entities']
        entity_type = condition.get('entity_type')
        threshold = condition.get('threshold', 0)
        operator = condition.get('operator', '<=')  # Default: count should be <= threshold
        
        # Count entities of specified type
        if entity_type:
            count = len([e for e in entities if e.get('type') == entity_type])
        else:
            count = len(entities)
            
        # Apply operator
        if operator == '<=':
            is_satisfied = count <= threshold
        elif operator == '<':
            is_satisfied = count < threshold
        elif operator == '==':
            is_satisfied = count == threshold
        elif operator == '>=':
            is_satisfied = count >= threshold
        elif operator == '>':
            is_satisfied = count > threshold
        else:
            is_satisfied = False  # Unknown operator
            
        return {
            'is_satisfied': is_satisfied,
            'details': {
                'count': count,
                'threshold': threshold,
                'operator': operator,
                'entity_type': entity_type
            }
        }
    
    def _check_relation_condition(self, condition, context):
        """Check for specific relations"""
        if context['content_type'] == 'relation':
            # Direct relation check
            relation_type = context.get('relation_type')
            required_type = condition.get('relation_type')
            
            is_satisfied = relation_type == required_type
            
            # Check source/target types if specified
            if is_satisfied and 'source_type' in condition:
                source_type = context.get('source_type')
                is_satisfied = source_type == condition['source_type']
                
            if is_satisfied and 'target_type' in condition:
                target_type = context.get('target_type')
                is_satisfied = target_type == condition['target_type']
                
            return {
                'is_satisfied': is_satisfied,
                'details': {
                    'relation_type': relation_type,
                    'required_type': required_type
                }
            }
        elif 'relations' in context:
            # Check in list of relations
            relations = context['relations']
            required_type = condition.get('relation_type')
            
            # Filter relations by type
            matching_relations = [r for r in relations if r.get('type') == required_type]
            
            # Further filter by source/target types if specified
            if 'source_type' in condition:
                source_type = condition['source_type']
                matching_relations = [
                    r for r in matching_relations 
                    if self._get_entity_type(r.get('source'), context) == source_type
                ]
                
            if 'target_type' in condition:
                target_type = condition['target_type']
                matching_relations = [
                    r for r in matching_relations 
                    if self._get_entity_type(r.get('target'), context) == target_type
                ]
                
            is_satisfied = len(matching_relations) > 0
            
            return {
                'is_satisfied': is_satisfied,
                'details': {
                    'matching_relations': len(matching_relations),
                    'required_type': required_type
                }
            }
            
        return {'is_satisfied': True, 'details': {'reason': 'no_relations_available'}}
    
    def _get_entity_type(self, entity_id, context):
        """Get entity type from entity ID using context"""
        if 'entities' not in context:
            return 'UNKNOWN'
            
        for entity in context['entities']:
            if entity.get('id') == entity_id:
                return entity.get('type', 'UNKNOWN')
                
        return 'UNKNOWN'
    
    def _check_concept_condition(self, condition, context):
        """Check concept activation"""
        if context['content_type'] == 'concept':
            # Direct concept check
            concept_name = context.get('content', {}).get('name', '')
            required_concept = condition.get('concept')
            
            is_name_match = concept_name == required_concept
            
            # Check activation threshold if specified
            activation = context.get('concept_activation', 0.0)
            threshold = condition.get('threshold', 0.0)
            is_activation_sufficient = activation >= threshold
            
            is_satisfied = is_name_match and is_activation_sufficient
            
            return {
                'is_satisfied': is_satisfied,
                'details': {
                    'concept_name': concept_name,
                    'required_concept': required_concept,
                    'activation': activation,
                    'threshold': threshold
                }
            }
        elif 'concepts' in context:
            # Check in map of concepts
            concepts = context['concepts']
            required_concept = condition.get('concept')
            
            if required_concept in concepts:
                # Concept exists, check activation
                concept_data = concepts[required_concept]
                activation = concept_data.get('activation', 0.0)
                threshold = condition.get('threshold', 0.0)
                is_satisfied = activation >= threshold
                
                return {
                    'is_satisfied': is_satisfied,
                    'details': {
                        'concept': required_concept,
                        'activation': activation,
                        'threshold': threshold
                    }
                }
            else:
                # Concept not found
                return {
                    'is_satisfied': False,
                    'details': {
                        'reason': 'concept_not_found',
                        'required_concept': required_concept
                    }
                }
                
        return {'is_satisfied': True, 'details': {'reason': 'no_concepts_available'}}
    
    def _check_text_pattern_condition(self, condition, context):
        """Check for text patterns"""
        # Get text to check
        text = None
        if context['content_type'] == 'text':
            text = context['content']
        elif isinstance(context['content'], dict) and 'text' in context['content']:
            text = context['content']['text']
        elif isinstance(context['content'], str):
            text = context['content']
            
        if not text:
            return {'is_satisfied': True, 'details': {'reason': 'no_text_available'}}
            
        # Get pattern to check
        pattern = condition.get('pattern')
        if not pattern:
            return {'is_satisfied': True, 'details': {'reason': 'no_pattern_specified'}}
            
        # Compile pattern if it's a string
        if isinstance(pattern, str):
            try:
                pattern = re.compile(pattern, re.IGNORECASE)
            except re.error:
                return {
                    'is_satisfied': False, 
                    'details': {'reason': 'invalid_pattern', 'pattern': pattern}
                }
                
        # Check for matches
        matches = list(pattern.finditer(text))
        match_condition = condition.get('match_condition', 'any')  # 'any', 'none', 'exact'
        
        if match_condition == 'any':
            is_satisfied = len(matches) > 0
        elif match_condition == 'none':
            is_satisfied = len(matches) == 0
        elif match_condition == 'exact':
            exact_count = condition.get('match_count', 1)
            is_satisfied = len(matches) == exact_count
        else:
            is_satisfied = False  # Unknown condition
            
        return {
            'is_satisfied': is_satisfied,
            'details': {
                'matches': len(matches),
                'match_condition': match_condition,
                'pattern': pattern.pattern if hasattr(pattern, 'pattern') else str(pattern)
            }
        }
    
    def _check_classification_condition(self, condition, context):
        """Check content classification"""
        if 'classification' not in context:
            return {'is_satisfied': True, 'details': {'reason': 'no_classification_available'}}
            
        classification = context['classification']
        required_class = condition.get('class')
        threshold = condition.get('threshold', 0.5)
        
        if required_class in classification:
            # Class exists, check confidence
            confidence = classification[required_class]
            is_satisfied = confidence >= threshold
            
            return {
                'is_satisfied': is_satisfied,
                'details': {
                    'class': required_class,
                    'confidence': confidence,
                    'threshold': threshold
                }
            }
        else:
            # Class not found
            return {
                'is_satisfied': False,
                'details': {
                    'reason': 'class_not_found',
                    'required_class': required_class
                }
            }
    
    def _evaluate_rule_compliance(self, rule, condition_results, compliance_mode):
        """Evaluate overall rule compliance based on condition results"""
        # Get rule combination mode (default: all conditions must be satisfied)
        combination_mode = rule.get('combination_mode', 'all')
        
        if combination_mode == 'all':
            # All conditions must be satisfied
            is_compliant = all(result['is_satisfied'] for result in condition_results)
            failed_conditions = [result for result in condition_results if not result['is_satisfied']]
            
            if is_compliant:
                return {
                    'is_compliant': True,
                    'compliance_score': 1.0,
                    'reason': 'all_conditions_satisfied'
                }
            else:
                # Get the most severe failed condition
                most_severe_action = self._get_most_severe_action(failed_conditions)
                violation_type = f"{most_severe_action}_violation" if most_severe_action else "rule_violation"
                
                return {
                    'is_compliant': False,
                    'compliance_score': 0.0,
                    'reason': 'conditions_not_satisfied',
                    'violation_type': violation_type,
                    'violation_description': f"Rule violated: {len(failed_conditions)} condition(s) not satisfied",
                    'failed_conditions': failed_conditions,
                    'recommended_action': most_severe_action or 'review'
                }
        elif combination_mode == 'any':
            # At least one condition must be satisfied
            is_compliant = any(result['is_satisfied'] for result in condition_results)
            
            if is_compliant:
                return {
                    'is_compliant': True,
                    'compliance_score': 1.0,
                    'reason': 'some_conditions_satisfied'
                }
            else:
                return {
                    'is_compliant': False,
                    'compliance_score': 0.0,
                    'reason': 'no_conditions_satisfied',
                    'violation_type': 'rule_violation',
                    'violation_description': "Rule violated: No conditions satisfied",
                    'recommended_action': 'review'
                }
        elif combination_mode == 'weighted':
            # Calculate weighted compliance score
            weights = rule.get('condition_weights', [1.0] * len(condition_results))
            total_weight = sum(weights)
            
            if total_weight == 0:
                # No weights, treat as all mode
                is_compliant = all(result['is_satisfied'] for result in condition_results)
                compliance_score = 1.0 if is_compliant else 0.0
            else:
                # Calculate weighted score
                weighted_sum = sum(
                    weight if result['is_satisfied'] else 0.0
                    for weight, result in zip(weights, condition_results)
                )
                compliance_score = weighted_sum / total_weight
                
                # Determine compliance based on threshold
                threshold = rule.get('compliance_threshold', 0.8)
                is_compliant = compliance_score >= threshold
            
            if is_compliant:
                return {
                    'is_compliant': True,
                    'compliance_score': compliance_score,
                    'reason': 'weighted_compliance_sufficient'
                }
            else:
                return {
                    'is_compliant': False,
                    'compliance_score': compliance_score,
                    'reason': 'weighted_compliance_insufficient',
                    'violation_type': 'rule_violation',
                    'violation_description': f"Rule violated: Weighted compliance score {compliance_score:.2f} below threshold",
                    'recommended_action': 'review'
                }
                
        return {
            'is_compliant': False,
            'compliance_score': 0.0,
            'reason': 'unknown_combination_mode',
            'violation_type': 'rule_evaluation_error',
            'violation_description': f"Unknown rule combination mode: {combination_mode}",
            'recommended_action': 'review'
        }
    
    def _get_most_severe_action(self, failed_conditions):
        """Get the most severe action from failed conditions"""
        # Define action severity order (from most to least severe)
        action_severity = {
            'block': 5,
            'flag': 4,
            'anonymize': 3,
            'modify': 2,
            'log': 1
        }
        
        # Find actions in failed conditions
        actions = [
            result.get('action') for result in failed_conditions
            if 'action' in result
        ]
        
        if not actions:
            return None
            
        # Find most severe action
        return max(actions, key=lambda a: action_severity.get(a, 0))
    
    def _apply_framework_verification(self, frameworks, context, compliance_mode):
        """Apply framework-specific verification"""
        framework_results = []
        
        for framework in frameworks:
            if framework in self.framework_processors:
                processor = self.framework_processors[framework]
                framework_result = processor.verify_compliance(context, compliance_mode)
                framework_results.append({
                    'framework': framework,
                    'result': framework_result
                })
                
        return framework_results
    
    def _aggregate_compliance_results(self, rule_results, framework_results, compliance_mode):
        """Aggregate compliance results and determine overall compliance"""
        # Get rule violations
        rule_violations = [
            result for result in rule_results
            if not result['is_compliant']
        ]
        
        # Get framework violations
        framework_violations = [
            {
                'framework': fr['framework'],
                'violation': fr['result'].get('violations', [])[0] if not fr['result'].get('is_compliant') else None
            }
            for fr in framework_results
            if not fr['result'].get('is_compliant')
        ]
        
        # Determine if there are any high severity violations
        has_high_severity = any(
            violation['severity'] == 'high' for violation in rule_violations
        ) or any(
            violation.get('violation', {}).get('severity') == 'high' 
            for violation in framework_violations if violation.get('violation')
        )
        
        # Calculate overall compliance score
        if rule_results:
            rule_scores = [
                result.get('compliance_score', 1.0 if result['is_compliant'] else 0.0)
                for result in rule_results
            ]
            rule_compliance_score = sum(rule_scores) / len(rule_scores)
        else:
            rule_compliance_score = 1.0
            
        if framework_results:
            framework_scores = [
                fr['result'].get('compliance_score', 1.0 if fr['result'].get('is_compliant', False) else 0.0)
                for fr in framework_results
            ]
            framework_compliance_score = sum(framework_scores) / len(framework_scores)
        else:
            framework_compliance_score = 1.0
            
        # Weight rule and framework scores (slightly higher weight for rules)
        overall_compliance_score = 0.6 * rule_compliance_score + 0.4 * framework_compliance_score
        
        # Determine overall compliance
        # In strict mode, any violation means non-compliant
        # In soft mode, only high severity violations or very low score means non-compliant
        if compliance_mode == 'strict':
            is_compliant = len(rule_violations) == 0 and all(
                fr['result'].get('is_compliant', True) for fr in framework_results
            )
        else:  # soft mode
            is_compliant = (not has_high_severity) and (overall_compliance_score >= 0.5)
            
        # Prepare result object
        result = {
            'is_compliant': is_compliant,
            'compliance_score': overall_compliance_score,
            'rule_results': rule_results,
            'framework_results': [fr['result'] for fr in framework_results],
            'metadata': {
                'compliance_mode': compliance_mode,
                'rule_count': len(rule_results),
                'framework_count': len(framework_results),
                'rule_violation_count': len(rule_violations),
                'framework_violation_count': len([fr for fr in framework_results if not fr['result'].get('is_compliant', True)])
            }
        }
        
        # Add violations if not compliant
        if not is_compliant:
            result['violations'] = []
            
            # Add rule violations
            for violation in rule_violations:
                result['violations'].append({
                    'source': 'rule',
                    'rule_id': violation['rule_id'],
                    'rule_name': violation['name'],
                    'type': violation.get('violation', {}).get('type', 'rule_violation'),
                    'description': violation.get('violation', {}).get('description', f"Violation of rule: {violation['name']}"),
                    'severity': violation['severity'],
                    'recommended_action': violation.get('violation', {}).get('recommended_action', 'review')
                })
                
            # Add framework violations
            for fv in framework_violations:
                if fv.get('violation'):
                    result['violations'].append({
                        'source': 'framework',
                        'framework': fv['framework'],
                        'type': fv['violation'].get('type', 'framework_violation'),
                        'description': fv['violation'].get('description', f"Violation of framework: {fv['framework']}"),
                        'severity': fv['violation'].get('severity', 'medium'),
                        'recommended_action': fv['violation'].get('recommended_action', 'review')
                    })
        
        return result
    
    def handle_violations(self, compliance_result, content):
        """
        Handle compliance violations by applying appropriate actions
        
        Args:
            compliance_result: Result from verify_content
            content: Original content that was verified
            
        Returns:
            Handled content and handling metadata
        """
        if compliance_result['is_compliant']:
            # No violations to handle
            return {
                'handled_content': content,
                'is_modified': False,
                'applied_actions': []
            }
            
        # Get violations and their recommended actions
        violations = compliance_result.get('violations', [])
        
        # Group violations by recommended action
        action_violations = {}
        for violation in violations:
            action = violation.get('recommended_action', 'review')
            if action not in action_violations:
                action_violations[action] = []
            action_violations[action].append(violation)
            
        # Apply handlers in order of severity
        # Order: block, flag, anonymize, modify, log
        applied_actions = []
        current_content = content
        is_modified = False
        
        # If 'block' action exists, apply it first (content will be blocked)
        if 'block' in action_violations and self.violation_handlers.get('block'):
            handler = self.violation_handlers['block']
            result = handler.handle(current_content, action_violations['block'])
            applied_actions.append({
                'action': 'block',
                'violations': action_violations['block'],
                'result': result
            })
            # No need to process other actions if content is blocked
            return {
                'handled_content': result.get('content'),
                'is_modified': True,
                'applied_actions': applied_actions,
                'is_blocked': True
            }
            
        # Apply other handlers
        for action in ['flag', 'anonymize', 'modify', 'log']:
            if action in action_violations and action in self.violation_handlers:
                handler = self.violation_handlers[action]
                result = handler.handle(current_content, action_violations[action])
                
                if result.get('is_modified', False):
                    is_modified = True
                    current_content = result.get('content', current_content)
                
                applied_actions.append({
                    'action': action,
                    'violations': action_violations[action],
                    'result': result
                })
                
        # Apply custom handlers
        for action, violations in action_violations.items():
            if action not in ['block', 'flag', 'anonymize', 'modify', 'log'] and action in self.violation_handlers:
                handler = self.violation_handlers[action]
                result = handler.handle(current_content, violations)
                
                if result.get('is_modified', False):
                    is_modified = True
                    current_content = result.get('content', current_content)
                
                applied_actions.append({
                    'action': action,
                    'violations': violations,
                    'result': result
                })
                
        return {
            'handled_content': current_content,
            'is_modified': is_modified,
            'applied_actions': applied_actions
        }


