import json
import logging
import re
import time
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
from src.core.utils.utils import LRUCache

class NeuralSymbolicInterface:
    """
    Interface for bidirectional translation between neural and symbolic representations
    to support compliance verification and reasoning.
    
    This class provides methods to:
    1. Convert neural model outputs to symbolic representations
    2. Convert symbolic representations to natural language
    3. Generate explanations from symbolic reasoning traces
    """
    
    def __init__(self, config: Dict[str, Any], language_model=None):
        """
        Initialize the neural-symbolic interface.
        
        Args:
            config: Configuration dictionary
            language_model: Optional language model for neural processing
        """
        self.config = config
        self.language_model = language_model
        
        # Initialize translation components
        self.n2s_translator = NeuralToSymbolicTranslator(config, language_model)
        self.s2n_translator = SymbolicToNeuralTranslator(config, language_model)
        self.c2s_translator = ComplianceToSymbolicTranslator(config)
        
        # Initialize caches
        cache_size = config.get("cache_size", 1000)
        self.n2s_cache = LRUCache(maxsize=cache_size)
        self.s2n_cache = LRUCache(maxsize=cache_size)
        
        # Performance tracking
        self.stats = {
            "n2s_calls": 0,
            "s2n_calls": 0,
            "cache_hits": 0,
            "avg_translation_time": 0.0
        }
        
    def neural_to_symbolic_text(self, text: str, mode: str = "standard") -> Dict[str, Any]:
        """
        Convert natural language text to symbolic representation.
        
        Args:
            text: Input text to convert
            mode: Translation mode (standard, strict, relaxed)
            
        Returns:
            Dict with symbolic representation and metadata
        """
        # Update stats
        self.stats["n2s_calls"] += 1
        
        # Check cache
        cache_key = f"n2s:{mode}:{self._hash_text(text)}"
        cached_result = self.n2s_cache.get(cache_key)
        
        if cached_result:
            self.stats["cache_hits"] += 1
            return cached_result
        
        # Measure performance
        start_time = time.time()
        
        # Perform translation
        try:
            symbolic_repr = self.n2s_translator.translate_text(text, mode)
            
            # Create result with metadata
            result = {
                "is_compliant": True,
                "symbolic_repr": symbolic_repr,
                "translation_time": time.time() - start_time,
                "mode": mode
            }
            
        except Exception as e:
            # Handle translation errors
            logging.error(f"Error translating text to symbolic: {str(e)}")
            result = {
                "is_compliant": False,
                "compliance_error": f"Translation error: {str(e)}",
                "translation_time": time.time() - start_time,
                "mode": mode
            }
        
        # Update performance stats
        self._update_performance_stats(time.time() - start_time)
        
        # Cache result
        self.n2s_cache[cache_key] = result
        
        return result
    
    def neural_to_symbolic_embedding(self, embedding: np.ndarray, mode: str = "standard") -> Dict[str, Any]:
        """
        Convert neural embedding to symbolic representation.
        
        Args:
            embedding: Input embedding vector
            mode: Translation mode (standard, strict, relaxed)
            
        Returns:
            Dict with symbolic representation and metadata
        """
        # Update stats
        self.stats["n2s_calls"] += 1
        
        # Embedding cache key (hash of first few values)
        cache_key = f"n2s_emb:{mode}:{hash(tuple(embedding[:5].tolist()))}"
        cached_result = self.n2s_cache.get(cache_key)
        
        if cached_result:
            self.stats["cache_hits"] += 1
            return cached_result
        
        # Measure performance
        start_time = time.time()
        
        # Perform translation
        try:
            symbolic_repr = self.n2s_translator.translate_embedding(embedding, mode)
            
            # Create result with metadata
            result = {
                "is_compliant": True,
                "symbolic_repr": symbolic_repr,
                "translation_time": time.time() - start_time,
                "mode": mode
            }
            
        except Exception as e:
            # Handle translation errors
            logging.error(f"Error translating embedding to symbolic: {str(e)}")
            result = {
                "is_compliant": False,
                "compliance_error": f"Translation error: {str(e)}",
                "translation_time": time.time() - start_time,
                "mode": mode
            }
        
        # Update performance stats
        self._update_performance_stats(time.time() - start_time)
        
        # Cache result
        self.n2s_cache[cache_key] = result
        
        return result
    
    def symbolic_to_neural_text(self, symbolic_repr: Dict[str, Any], mode: str = "standard") -> Dict[str, Any]:
        """
        Convert symbolic representation to natural language text.
        
        Args:
            symbolic_repr: Symbolic representation to convert
            mode: Translation mode (standard, verbose, concise)
            
        Returns:
            Dict with natural language text and metadata
        """
        # Update stats
        self.stats["s2n_calls"] += 1
        
        # Check cache
        cache_key = f"s2n:{mode}:{self._hash_object(symbolic_repr)}"
        cached_result = self.s2n_cache.get(cache_key)
        
        if cached_result:
            self.stats["cache_hits"] += 1
            return cached_result
        
        # Measure performance
        start_time = time.time()
        
        # Perform translation
        try:
            natural_text = self.s2n_translator.translate_to_text(symbolic_repr, mode)
            
            # Create result with metadata
            result = {
                "is_compliant": True,
                "text": natural_text,
                "translation_time": time.time() - start_time,
                "mode": mode
            }
            
        except Exception as e:
            # Handle translation errors
            logging.error(f"Error translating symbolic to text: {str(e)}")
            result = {
                "is_compliant": False,
                "compliance_error": f"Translation error: {str(e)}",
                "translation_time": time.time() - start_time,
                "mode": mode
            }
        
        # Update performance stats
        self._update_performance_stats(time.time() - start_time)
        
        # Cache result
        self.s2n_cache[cache_key] = result
        
        return result
    
    def symbolic_to_neural_guidance(self, symbolic_result: Dict[str, Any], mode: str = "standard") -> Dict[str, Any]:
        """
        Convert symbolic reasoning result to neural guidance for LLM control.
        
        Args:
            symbolic_result: Symbolic reasoning result
            mode: Translation mode
            
        Returns:
            Dict with neural guidance and metadata
        """
        # Update stats
        self.stats["s2n_calls"] += 1
        
        # Check cache
        cache_key = f"s2n_guide:{mode}:{self._hash_object(symbolic_result)}"
        cached_result = self.s2n_cache.get(cache_key)
        
        if cached_result:
            self.stats["cache_hits"] += 1
            return cached_result
        
        # Measure performance
        start_time = time.time()
        
        # Perform translation
        try:
            guidance = self.s2n_translator.translate_to_guidance(symbolic_result, mode)
            
            # Create result with metadata
            result = {
                "is_compliant": True,
                "neural_guidance": guidance,
                "translation_time": time.time() - start_time,
                "mode": mode
            }
            
        except Exception as e:
            # Handle translation errors
            logging.error(f"Error translating symbolic to guidance: {str(e)}")
            result = {
                "is_compliant": False,
                "compliance_error": f"Translation error: {str(e)}",
                "translation_time": time.time() - start_time,
                "mode": mode
            }
        
        # Update performance stats
        self._update_performance_stats(time.time() - start_time)
        
        # Cache result
        self.s2n_cache[cache_key] = result
        
        return result
    
    def translate_compliance_result(self, verification_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate compliance verification result to symbolic explanation.
        
        Args:
            verification_result: Compliance verification result
            
        Returns:
            Dict with symbolic explanation
        """
        # Translate using compliance translator
        return self.c2s_translator.translate(verification_result)
    
    def generate_explanation(self, symbolic_explanation: Dict[str, Any], detail_level: str = "medium") -> str:
        """
        Generate human-readable explanation from symbolic explanation.
        
        Args:
            symbolic_explanation: Symbolic explanation
            detail_level: Detail level (brief, medium, detailed)
            
        Returns:
            Human-readable explanation text
        """
        return self.s2n_translator.translate_explanation(symbolic_explanation, detail_level)
    
    def _hash_text(self, text: str) -> str:
        """Create hash of text for caching."""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()
    
    def _hash_object(self, obj: Any) -> str:
        """Create hash of object for caching."""
        import hashlib
        return hashlib.md5(json.dumps(obj, sort_keys=True).encode()).hexdigest()
    
    def _update_performance_stats(self, elapsed_time: float):
        """Update performance statistics."""
        # Calculate running average of translation time
        if "avg_translation_time" in self.stats:
            # Use exponential moving average
            current_avg = self.stats["avg_translation_time"]
            self.stats["avg_translation_time"] = 0.9 * current_avg + 0.1 * elapsed_time
        else:
            self.stats["avg_translation_time"] = elapsed_time

class NeuralToSymbolicTranslator:
    """
    Translates neural model outputs to symbolic representations
    for compliance reasoning
    """
    
    def __init__(self, reasoning_engine, compliance_ontology):
        """
        Initialize translator with reasoning engine and compliance ontology
        
        Args:
            reasoning_engine: Engine for symbolic reasoning
            compliance_ontology: Ontology for compliance concepts
        """
        self.reasoning_engine = reasoning_engine
        self.ontology = compliance_ontology
        
        # Initialize concept mapping
        self.concept_mapping = self._initialize_concept_mapping()
    
    def _initialize_concept_mapping(self):
        """
        Initialize mapping between natural language concepts
        and formal symbolic representations
        
        Returns:
            Dictionary mapping concept phrases to symbolic terms
        """
        # Basic concept mapping from natural language to symbolic terms
        return {
            # GDPR concepts
            "personal data": "PersonalData",
            "data subject": "DataSubject",
            "data controller": "DataController",
            "data processor": "DataProcessor",
            "consent": "Consent",
            "explicit consent": "ExplicitConsent",
            "legitimate interest": "LegitimateInterest",
            "data protection": "DataProtection",
            "right to access": "RightToAccess",
            "right to erasure": "RightToErasure",
            
            # HIPAA concepts
            "protected health information": "PHI",
            "electronic health record": "EHR",
            "covered entity": "CoveredEntity",
            "business associate": "BusinessAssociate",
            "authorization": "Authorization",
            "minimum necessary": "MinimumNecessary",
            
            # Relationships
            "processes": "processes",
            "controls": "controls",
            "shares with": "sharesWith",
            "gives consent to": "givesConsentTo",
            "requires": "requires",
            "authorized by": "authorizedBy",
            
            # Actions
            "collect": "collect",
            "process": "process",
            "share": "share",
            "disclose": "disclose",
            "anonymize": "anonymize",
            "encrypt": "encrypt"
        }
    
    def translate(self, text):
        """
        Translate natural language text to symbolic representation
        
        Args:
            text: Natural language text to translate
            
        Returns:
            Symbolic representation of the text
        """
        # Extract key entities and relationships
        entities = self._extract_entities(text)
        relationships = self._extract_relationships(text, entities)
        actions = self._extract_actions(text, entities)
        
        # Construct symbolic representation
        symbolic_rep = self._build_symbolic_representation(entities, relationships, actions)
        
        # Run through reasoning engine to check consistency
        is_consistent = self.reasoning_engine.check_consistency(symbolic_rep)
        
        return {
            "symbolic_representation": symbolic_rep,
            "is_consistent": is_consistent,
            "entities": entities,
            "relationships": relationships,
            "actions": actions
        }
    
    def _extract_entities(self, text):
        """
        Extract entities from text and map to ontology concepts
        
        Args:
            text: Input text
            
        Returns:
            List of extracted entities with types
        """
        entities = []
        
        # Convert text to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Extract entities based on concept mapping
        for concept, symbol in self.concept_mapping.items():
            if concept in text_lower:
                # Find all occurrences
                start_pos = 0
                while True:
                    pos = text_lower.find(concept, start_pos)
                    if pos == -1:
                        break
                        
                    # Add entity
                    entities.append({
                        "text": concept,
                        "symbol": symbol,
                        "position": (pos, pos + len(concept)),
                        "type": self._get_entity_type(symbol)
                    })
                    
                    # Move to next potential occurrence
                    start_pos = pos + len(concept)
        
        # Extract specific entity types with patterns
        entity_patterns = {
            "Email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "Phone": r'\b(?:\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b',
            "Date": r'\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b',
            "Person": r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'
        }
        
        for entity_type, pattern in entity_patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append({
                    "text": match.group(),
                    "symbol": f"{entity_type}_{len(entities)}",
                    "position": (match.start(), match.end()),
                    "type": entity_type
                })
        
        return entities
    
    def _get_entity_type(self, symbol):
        """Determine entity type from symbolic representation"""
        # Map symbols to entity types
        type_prefixes = {
            "PersonalData": "Data",
            "DataSubject": "Actor",
            "DataController": "Actor",
            "DataProcessor": "Actor",
            "Consent": "State",
            "PHI": "Data",
            "CoveredEntity": "Actor",
            "Authorization": "State"
        }
        
        # Check if symbol starts with any of the prefixes
        for prefix, entity_type in type_prefixes.items():
            if symbol.startswith(prefix):
                return entity_type
        
        # Default type
        return "Unknown"
    
    def _extract_relationships(self, text, entities):
        """
        Extract relationships between entities
        
        Args:
            text: Input text
            entities: Previously extracted entities
            
        Returns:
            List of relationships between entities
        """
        relationships = []
        
        # Get relationship patterns
        relationship_patterns = {
            "processes": r'(\w+)\s+processes\s+(\w+)',
            "controls": r'(\w+)\s+controls\s+(\w+)',
            "sharesWith": r'(\w+)\s+shares\s+(\w+)\s+with\s+(\w+)',
            "givesConsentTo": r'(\w+)\s+gives\s+consent\s+to\s+(\w+)',
            "requires": r'(\w+)\s+requires\s+(\w+)',
            "authorizedBy": r'(\w+)\s+authorized\s+by\s+(\w+)'
        }
        
        # Check for relationships in text
        for rel_type, pattern in relationship_patterns.items():
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                # Find entities involved in relationship
                entity_names = match.groups()
                involved_entities = []
                
                for name in entity_names:
                    # Find matching entity
                    matching_entity = None
                    for entity in entities:
                        if name in entity["text"].lower():
                            matching_entity = entity
                            break
                    
                    involved_entities.append(matching_entity)
                
                # Only create relationship if all entities were found
                if all(involved_entities) and len(involved_entities) >= 2:
                    relationships.append({
                        "type": rel_type,
                        "source": involved_entities[0],
                        "target": involved_entities[1],
                        "additional": involved_entities[2:] if len(involved_entities) > 2 else []
                    })
        
        return relationships
    
    def _extract_actions(self, text, entities):
        """
        Extract actions mentioned in the text
        
        Args:
            text: Input text
            entities: Previously extracted entities
            
        Returns:
            List of actions with associated entities
        """
        actions = []
        
        # Action verbs to look for
        action_verbs = [
            "collect", "process", "share", "disclose", 
            "anonymize", "encrypt", "delete", "transfer"
        ]
        
        # Convert text to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Simple extraction based on verb proximity to entities
        for verb in action_verbs:
            # Find all occurrences of the verb
            start_pos = 0
            while True:
                pos = text_lower.find(verb, start_pos)
                if pos == -1:
                    break
                
                # Find nearby entities (within 50 characters)
                nearby_entities = []
                for entity in entities:
                    entity_pos = entity["position"][0]
                    if abs(entity_pos - pos) < 50:
                        nearby_entities.append(entity)
                
                # If entities found nearby, create action
                if nearby_entities:
                    # Try to determine subject and object
                    subject = None
                    object_entity = None
                    
                    for entity in nearby_entities:
                        if entity["type"] == "Actor":
                            subject = entity
                            break
                    
                    for entity in nearby_entities:
                        if entity["type"] == "Data" and entity != subject:
                            object_entity = entity
                            break
                    
                    actions.append({
                        "verb": verb,
                        "subject": subject,
                        "object": object_entity,
                        "position": pos,
                        "related_entities": [e for e in nearby_entities if e != subject and e != object_entity]
                    })
                
                # Move to next potential occurrence
                start_pos = pos + len(verb)
        
        return actions
    
    def _build_symbolic_representation(self, entities, relationships, actions):
        """
        Build formal symbolic representation from extracted information
        
        Args:
            entities: Extracted entities
            relationships: Extracted relationships
            actions: Extracted actions
            
        Returns:
            Formal symbolic representation for reasoning
        """
        # Convert to logical predicates
        predicates = []
        
        # Entity declarations
        for entity in entities:
            predicates.append(f"{entity['type']}({entity['symbol']})")
        
        # Relationships
        for rel in relationships:
            if rel['type'] in ['sharesWith'] and rel.get('additional'):
                # Ternary relationship
                predicates.append(f"{rel['type']}({rel['source']['symbol']}, {rel['target']['symbol']}, {rel['additional'][0]['symbol']})")
            else:
                # Binary relationship
                predicates.append(f"{rel['type']}({rel['source']['symbol']}, {rel['target']['symbol']})")
        
        # Actions
        for action in actions:
            if action['subject'] and action['object']:
                predicates.append(f"{action['verb']}({action['subject']['symbol']}, {action['object']['symbol']})")
            elif action['subject']:
                predicates.append(f"{action['verb']}({action['subject']['symbol']})")
            elif action['object']:
                predicates.append(f"{action['verb']}(Unknown, {action['object']['symbol']})")
        
        # Join all predicates with logical AND
        symbolic_representation = " ∧ ".join(predicates)
        
        return symbolic_representation

class SymbolicToNeuralTranslator:
    """Translator from symbolic representations to natural language."""
    
    def __init__(self, config: Dict[str, Any], language_model=None):
        """
        Initialize the symbolic-to-neural translator.
        
        Args:
            config: Configuration dictionary
            language_model: Optional language model
        """
        self.config = config
        self.language_model = language_model
        self.templates = self._load_templates()
    
    def translate_to_text(self, symbolic_repr: Dict[str, Any], mode: str = "standard") -> str:
        """
        Translate symbolic representation to natural language text.
        
        Args:
            symbolic_repr: Symbolic representation
            mode: Translation mode (standard, verbose, concise)
            
        Returns:
            Natural language text
        """
        # Extract components from symbolic representation
        statements = symbolic_repr.get("statements", [])
        concepts = symbolic_repr.get("concepts", [])
        intents = symbolic_repr.get("intents", [])
        
        # Determine appropriate template based on mode and content
        template_key = self._select_template(symbolic_repr, mode)
        template = self.templates.get(template_key, self.templates.get("default"))
        
        # Fill in template with content
        if "statements" in symbolic_repr and statements:
            # Convert statements to text
            statements_text = self._format_statements(statements)
            
            # Apply template
            text = template.format(
                statements=statements_text,
                concepts=", ".join(concepts),
                intents=", ".join(intents)
            )
        else:
            # Handle embedding-based representation
            concept_scores = symbolic_repr.get("embedding_concepts", {})
            top_concepts = sorted(concept_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            concepts_text = ", ".join(f"{concept} ({score:.2f})" for concept, score in top_concepts)
            
            text = f"This content relates to the following regulatory concepts: {concepts_text}."
        
        return text
    
    def translate_to_guidance(self, symbolic_result: Dict[str, Any], mode: str = "standard") -> Dict[str, Any]:
        """
        Translate symbolic reasoning result to neural guidance.
        
        Args:
            symbolic_result: Symbolic reasoning result
            mode: Translation mode
            
        Returns:
            Neural guidance for LLM control
        """
        # Extract key components for guidance
        guidance = {
            "allowed_concepts": [],
            "prohibited_concepts": [],
            "required_topics": [],
            "avoid_topics": [],
            "tone_guidance": {},
            "formality_level": 0.7,  # Default to slightly formal
            "explanation_required": False
        }
        
        # Check for reasoning results
        if "reasoning" in symbolic_result:
            reasoning = symbolic_result["reasoning"]
            
            # Extract allowed and prohibited concepts
            if "allowed_concepts" in reasoning:
                guidance["allowed_concepts"] = reasoning["allowed_concepts"]
                
            if "prohibited_concepts" in reasoning:
                guidance["prohibited_concepts"] = reasoning["prohibited_concepts"]
            
            # Set required explanations flag
            if reasoning.get("requires_explanation", False):
                guidance["explanation_required"] = True
        
        # Check for compliance results
        if "compliance" in symbolic_result:
            compliance = symbolic_result["compliance"]
            
            # Set formality based on compliance requirements
            if compliance.get("requires_formal_language", False):
                guidance["formality_level"] = 0.9
            
            # Set tone guidance
            if "tone_requirements" in compliance:
                guidance["tone_guidance"] = compliance["tone_requirements"]
        
        # Add prompt guidance
        prompt_guidance = self._generate_prompt_guidance(symbolic_result, guidance)
        guidance["prompt_guidance"] = prompt_guidance
        
        return guidance
    
    def translate_explanation(self, symbolic_explanation: Dict[str, Any], detail_level: str = "medium") -> str:
        """
        Translate symbolic explanation to human-readable text.
        
        Args:
            symbolic_explanation: Symbolic explanation
            detail_level: Detail level (brief, medium, detailed)
            
        Returns:
            Human-readable explanation
        """
        # Check for result field
        is_compliant = symbolic_explanation.get("result", False)
        
        # Get violations
        violations = symbolic_explanation.get("violations", [])
        
        # Select explanation template based on compliance result and detail level
        if is_compliant:
            template_key = f"compliant_{detail_level}"
        else:
            template_key = f"non_compliant_{detail_level}"
            
        template = self.explanation_templates.get(template_key, self.explanation_templates.get("default"))
        
        # Format violations text
        violations_text = ""
        if violations:
            violations_text = "\n".join([
                f"- {v.get('rule_id', 'Unknown rule')}: {v.get('severity', 'medium')} severity"
                for v in violations
            ])
        
        # Apply template
        explanation = template.format(
            is_compliant=is_compliant,
            violation_count=len(violations),
            violations=violations_text
        )
        
        return explanation
    
    def _load_templates(self) -> Dict[str, str]:
        """Load text generation templates."""
        # Templates for different modes and content types
        templates = {
            "default": "Based on the analysis, {statements}",
            
            # Standard mode templates
            "standard_statements": "{statements} {concepts}",
            "standard_concepts": "This content relates to {concepts}.",
            "standard_intents": "The intent appears to be {intents}.",
            
            # Verbose mode templates
            "verbose_statements": "The analysis identified the following statements: {statements}\n\nThis content covers these concepts: {concepts}",
            "verbose_concepts": "This content extensively covers the following regulatory concepts: {concepts}.",
            "verbose_intents": "The content exhibits these intents: {intents}.",
            
            # Concise mode templates
            "concise_statements": "{statements}",
            "concise_concepts": "Topics: {concepts}",
            "concise_intents": "Intent: {intents}"
        }
        
        # Explanation templates
        self.explanation_templates = {
            "default": "The content {'is compliant' if is_compliant else 'is not compliant'} with regulatory requirements.",
            
            # Compliant templates
            "compliant_brief": "The content is compliant with regulatory requirements.",
            "compliant_medium": "The content is compliant with all applicable regulatory requirements. No violations were detected.",
            "compliant_detailed": "The content has been verified and is fully compliant with all applicable regulatory requirements. The verification process detected no violations or concerns that would prevent use of this content.",
            
            # Non-compliant templates
            "non_compliant_brief": "The content is not compliant. {violation_count} violations found.",
            "non_compliant_medium": "The content is not compliant with regulatory requirements. {violation_count} violations were detected: {violations}",
            "non_compliant_detailed": "The content has been verified and found non-compliant with regulatory requirements. The verification process detected {violation_count} violations:\n\n{violations}\n\nThese violations must be addressed before the content can be used."
        }
        
        return templates
    
    def _select_template(self, symbolic_repr: Dict[str, Any], mode: str) -> str:
        """Select appropriate template based on content and mode."""
        has_statements = "statements" in symbolic_repr and symbolic_repr["statements"]
        has_concepts = "concepts" in symbolic_repr and symbolic_repr["concepts"]
        has_intents = "intents" in symbolic_repr and symbolic_repr["intents"]
        
        if mode not in ["standard", "verbose", "concise"]:
            mode = "standard"
        
        # Determine primary content type
        if has_statements:
            template_key = f"{mode}_statements"
        elif has_concepts:
            template_key = f"{mode}_concepts"
        elif has_intents:
            template_key = f"{mode}_intents"
        else:
            template_key = "default"

        return template_key
    
    def _format_statements(self, statements: List[Dict[str, str]]) -> str:
        """Format statements as natural language text."""
        formatted_statements = []
        
        for statement in statements:
            subject = statement.get("subject", "")
            predicate = statement.get("predicate", "is")
            obj = statement.get("object", "")
            
            # Simple formatting
            formatted = f"{subject} {predicate} {obj}"
            formatted_statements.append(formatted)
        
        return ". ".join(formatted_statements) + "." if formatted_statements else ""
    
    def _generate_prompt_guidance(self, symbolic_result: Dict[str, Any], guidance: Dict[str, Any]) -> str:
        """Generate natural language guidance for prompt engineering."""
        # Format guidance as a prompt instruction
        instructions = []
        
        # Add concept guidance
        if guidance["allowed_concepts"]:
            concepts_str = ", ".join(guidance["allowed_concepts"])
            instructions.append(f"Please focus on these concepts: {concepts_str}")
            
        if guidance["prohibited_concepts"]:
            concepts_str = ", ".join(guidance["prohibited_concepts"])
            instructions.append(f"Avoid discussing these concepts: {concepts_str}")
        
        # Add topic guidance
        if guidance.get("required_topics", []):
            topics_str = ", ".join(guidance["required_topics"])
            instructions.append(f"Ensure you address these topics: {topics_str}")
            
        if guidance.get("avoid_topics", []):
            topics_str = ", ".join(guidance["avoid_topics"])
            instructions.append(f"Do not discuss these topics: {topics_str}")
        
        # Add tone guidance
        if guidance.get("tone_guidance", {}):
            tone = guidance["tone_guidance"].get("preferred_tone", "neutral")
            instructions.append(f"Use a {tone} tone in your response")
        
        # Add formality guidance
        formality = guidance.get("formality_level", 0.5)
        if formality > 0.7:
            instructions.append("Use formal language")
        elif formality < 0.3:
            instructions.append("You can use casual, conversational language")
        
        # Add explanation requirement
        if guidance.get("explanation_required", False):
            instructions.append("Include explanations for any regulatory statements you make")
        
        # Combine instructions
        return "\n".join(instructions)


class ComplianceToSymbolicTranslator:
    """Translator from compliance verification results to symbolic explanations."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the compliance-to-symbolic translator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
    
    def translate(self, verification_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate compliance verification result to symbolic explanation.
        
        Args:
            verification_result: Compliance verification result
            
        Returns:
            Symbolic explanation
        """
        # Extract key components
        is_compliant = verification_result.get("is_compliant", False)
        compliance_score = verification_result.get("compliance_score", 0.0)
        violations = verification_result.get("violations", [])
        
        # Create symbolic explanation
        symbolic_explanation = {
            "result": is_compliant,
            "score": compliance_score,
            "violations": self._format_violations(violations),
            "conclusion": self._generate_conclusion(is_compliant, compliance_score, violations),
            "recommendations": self._generate_recommendations(violations)
        }
        
        # Add framework-specific information if available
        if "framework_details" in verification_result:
            symbolic_explanation["framework_details"] = verification_result["framework_details"]
        
        return symbolic_explanation
    
    def _format_violations(self, violations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format violations for symbolic explanation."""
        formatted_violations = []
        
        for violation in violations:
            formatted = {
                "rule_id": violation.get("rule_id", "unknown"),
                "severity": violation.get("severity", "medium"),
                "type": self._determine_violation_type(violation)
            }
            
            # Add description if available
            if "description" in violation:
                formatted["description"] = violation["description"]
                
            # Add locations if available
            if "locations" in violation:
                formatted["locations"] = violation["locations"]
                
            formatted_violations.append(formatted)
        
        return formatted_violations
    
    def _determine_violation_type(self, violation: Dict[str, Any]) -> str:
        """Determine violation type based on rule ID and metadata."""
        rule_id = violation.get("rule_id", "").lower()
        
        # Check for common types based on rule ID patterns
        if "pii" in rule_id or "personal" in rule_id:
            return "pii_violation"
        elif "phi" in rule_id or "health" in rule_id:
            return "phi_violation"
        elif "consent" in rule_id:
            return "consent_violation"
        elif "data_min" in rule_id:
            return "data_minimization_violation"
        elif "purpose" in rule_id:
            return "purpose_limitation_violation"
        elif "security" in rule_id or "safeguard" in rule_id:
            return "security_violation"
        else:
            return "general_violation"
    
    def _generate_conclusion(self, is_compliant: bool, score: float, violations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate conclusion for symbolic explanation."""
        if is_compliant:
            if score > 0.95:
                strength = "strong"
                confidence = "high"
            elif score > 0.8:
                strength = "adequate"
                confidence = "medium"
            else:
                strength = "marginal"
                confidence = "low"
                
            return {
                "compliance_status": "compliant",
                "strength": strength,
                "confidence": confidence
            }
        else:
            # Determine severity based on violations
            severities = [v.get("severity", "medium") for v in violations]
            if "critical" in severities:
                severity = "critical"
            elif "high" in severities:
                severity = "high"
            elif "medium" in severities:
                severity = "medium"
            else:
                severity = "low"
                
            return {
                "compliance_status": "non_compliant",
                "severity": severity,
                "violation_count": len(violations),
                "confidence": "high" if len(violations) > 0 else "medium"
            }
    
    def _generate_recommendations(self, violations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate recommendations based on violations."""
        if not violations:
            return []
            
        recommendations = []
        
        # Group violations by type
        violation_types = {}
        for violation in violations:
            v_type = self._determine_violation_type(violation)
            if v_type not in violation_types:
                violation_types[v_type] = []
            violation_types[v_type].append(violation)
        
        # Generate recommendations for each violation type
        for v_type, v_list in violation_types.items():
            recommendation = self._get_recommendation_for_type(v_type, v_list)
            if recommendation:
                recommendations.append(recommendation)
        
        return recommendations
    
    def _get_recommendation_for_type(self, violation_type: str, violations: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Get recommendation for a specific violation type."""
        # Templates for different violation types
        templates = {
            "pii_violation": "Remove or anonymize personal identifiable information",
            "phi_violation": "Ensure protected health information is properly secured",
            "consent_violation": "Obtain explicit consent before processing personal data",
            "data_minimization_violation": "Reduce data collection to only what is necessary",
            "purpose_limitation_violation": "Ensure data is processed only for specified purposes",
            "security_violation": "Implement appropriate security measures",
            "general_violation": "Review content for regulatory compliance issues"
        }
        
        # Get template
        template = templates.get(violation_type, templates["general_violation"])
        
        # Count violations of this type
        count = len(violations)
        
        # Get highest severity
        severities = [v.get("severity", "medium") for v in violations]
        if "critical" in severities:
            severity = "critical"
        elif "high" in severities:
            severity = "high"
        elif "medium" in severities:
            severity = "medium"
        else:
            severity = "low"
        
        return {
            "type": violation_type,
            "recommendation": template,
            "violation_count": count,
            "severity": severity,
            "priority": self._get_priority(severity)
        }
    
    def _get_priority(self, severity: str) -> int:
        """Convert severity to priority number (higher is more urgent)."""
        priorities = {
            "critical": 4,
            "high": 3,
            "medium": 2,
            "low": 1
        }
        return priorities.get(severity, 1)