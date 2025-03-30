import re
import json
import logging
from collections import defaultdict

class TemplateBasedNER:
    """Template-based entity extraction with minimal examples"""
    
    def __init__(self, config):
        self.config = config
        self.templates = config.get("entity_templates", {})
        self.entity_patterns = {}
        self.entity_examples = defaultdict(list)
        self.similarity_threshold = config.get("template_similarity_threshold", 0.7)
        
        # Initialize with default templates if available
        if not self.templates:
            self._initialize_default_templates()
        
        # Compile patterns from templates
        self._compile_patterns()
    
    def _initialize_default_templates(self):
        """Initialize default entity templates"""
        self.templates = {
            "PERSON": {
                "patterns": [
                    r"\b[A-Z][a-z]+ [A-Z][a-z]+\b",  # First Last
                    r"\bMr\.|Mrs\.|Ms\.|Dr\. [A-Z][a-z]+\b"  # Title Last
                ],
                "context_clues": ["said", "told", "according to", "spoke", "met with"],
                "examples": ["John Smith", "Dr. Johnson"]
            },
            "ORGANIZATION": {
                "patterns": [
                    r"\b[A-Z][a-zA-Z]+ (?:Inc\.|Corp\.|LLC|Company|Association)\b",
                    r"\b[A-Z][A-Za-z]+ [A-Z][A-Za-z]+ (?:Inc\.|Corp\.|LLC)\b"
                ],
                "context_clues": ["announced", "headquartered", "company", "organization"],
                "examples": ["Acme Corp.", "Global Systems LLC"]
            },
            "LOCATION": {
                "patterns": [
                    r"\b[A-Z][a-z]+, [A-Z]{2}\b",  # City, ST
                    r"\b[A-Z][a-z]+ [A-Z][a-z]+\b"  # Location name
                ],
                "context_clues": ["in", "at", "near", "from", "to", "located"],
                "examples": ["New York, NY", "Pacific Ocean"]
            },
            "DATE": {
                "patterns": [
                    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}(?:st|nd|rd|th)?, \d{4}\b",
                    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"
                ],
                "context_clues": ["on", "dated", "by", "before", "after", "since"],
                "examples": ["January 15th, 2023", "10/21/2022"]
            }
        }
    
    def _compile_patterns(self):
        """Compile regex patterns from templates"""
        for entity_type, template in self.templates.items():
            # Compile standard patterns
            patterns = [re.compile(p) for p in template.get("patterns", [])]
            
            # Store examples for similarity matching
            self.entity_examples[entity_type] = template.get("examples", [])
            
            # Store compiled patterns
            self.entity_patterns[entity_type] = patterns
    
    def add_entity_template(self, entity_type, patterns=None, context_clues=None, examples=None):
        """Add a new entity template or update existing one"""
        if entity_type not in self.templates:
            self.templates[entity_type] = {}
        
        if patterns:
            self.templates[entity_type]["patterns"] = patterns
        
        if context_clues:
            self.templates[entity_type]["context_clues"] = context_clues
            
        if examples:
            self.templates[entity_type]["examples"] = examples
            self.entity_examples[entity_type] = examples
            
        # Recompile patterns
        if patterns:
            self.entity_patterns[entity_type] = [re.compile(p) for p in patterns]
    
    def add_example(self, entity_type, example_text):
        """Add a new example for an entity type"""
        if entity_type not in self.templates:
            self.templates[entity_type] = {"examples": []}
            
        if "examples" not in self.templates[entity_type]:
            self.templates[entity_type]["examples"] = []
            
        self.templates[entity_type]["examples"].append(example_text)
        self.entity_examples[entity_type].append(example_text)
    
    def extract_entities(self, text):
        """Extract entities using templates"""
        entities = []
        entity_id = 0
        
        # First pass: regex pattern matching
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    entity_text = match.group(0)
                    start = match.start()
                    end = match.end()
                    
                    entities.append({
                        'id': f"t{entity_id}",
                        'text': entity_text,
                        'start': start,
                        'end': end,
                        'ner_type': entity_type,
                        'detection_method': 'template_pattern',
                        'confidence': 0.8
                    })
                    entity_id += 1
        
        # Second pass: context clue matching
        for entity_type, template in self.templates.items():
            context_clues = template.get("context_clues", [])
            if not context_clues:
                continue
                
            # Create pattern to find words between context clues and potential entities
            for clue in context_clues:
                # Pattern: [context clue] + up to 5 words + capitalized word/phrase
                pattern = rf"\b{re.escape(clue)}\b(?:\s+\w+){{0,5}}\s+([A-Z][a-zA-Z0-9]+(?: [A-Z][a-zA-Z0-9]+)*)\b"
                for match in re.finditer(pattern, text):
                    entity_text = match.group(1)
                    start = match.start(1)
                    end = match.end(1)
                    
                    # Check if this entity overlaps with any existing entities
                    overlaps = False
                    for entity in entities:
                        if (start <= entity['end'] and end >= entity['start']):
                            overlaps = True
                            break
                            
                    if not overlaps:
                        entities.append({
                            'id': f"t{entity_id}",
                            'text': entity_text,
                            'start': start,
                            'end': end,
                            'ner_type': entity_type,
                            'detection_method': 'template_context',
                            'confidence': 0.7
                        })
                        entity_id += 1
        
        # Third pass: example-based matching for unknown entities
        capitalized_pattern = r"\b([A-Z][a-zA-Z0-9]+(?: [A-Z][a-zA-Z0-9]+)*)\b"
        unknown_entities = []
        
        for match in re.finditer(capitalized_pattern, text):
            entity_text = match.group(1)
            start = match.start(1)
            end = match.end(1)
            
            # Check if this entity is already detected
            overlaps = False
            for entity in entities:
                if (start <= entity['end'] and end >= entity['start']):
                    overlaps = True
                    break
                    
            if not overlaps:
                unknown_entities.append({
                    'text': entity_text,
                    'start': start,
                    'end': end
                })
        
        # For each unknown entity, find closest example match
        for unknown in unknown_entities:
            best_type = None
            best_score = -1
            
            for entity_type, examples in self.entity_examples.items():
                for example in examples:
                    similarity = self._calculate_similarity(unknown['text'], example)
                    if similarity > best_score and similarity >= self.similarity_threshold:
                        best_score = similarity
                        best_type = entity_type
            
            if best_type:
                entities.append({
                    'id': f"t{entity_id}",
                    'text': unknown['text'],
                    'start': unknown['start'],
                    'end': unknown['end'],
                    'ner_type': best_type,
                    'detection_method': 'template_example',
                    'confidence': best_score
                })
                entity_id += 1
        
        return entities
    
    def _calculate_similarity(self, text1, text2):
        """Calculate simple string similarity"""
        # Use character-level Jaccard similarity
        set1 = set(text1.lower())
        set2 = set(text2.lower())
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            return 0
            
        return intersection / union
    
    def extract_entities_from_example(self, example_text, entity_examples):
        """Extract entities based on a specific set of examples"""
        # Create temporary examples dictionary
        temp_examples = defaultdict(list)
        for entity_type, examples in entity_examples.items():
            temp_examples[entity_type] = examples
            
        # Save original examples
        original_examples = self.entity_examples
        
        # Set temporary examples
        self.entity_examples = temp_examples
        
        # Extract entities
        entities = self.extract_entities(example_text)
        
        # Restore original examples
        self.entity_examples = original_examples
        
        return entities
    
    def learn_from_example(self, text, annotated_entities):
        """Learn new patterns and examples from annotated text"""
        for entity in annotated_entities:
            entity_type = entity.get('ner_type')
            entity_text = entity.get('text')
            
            if not entity_type or not entity_text:
                continue
                
            # Add as example
            self.add_example(entity_type, entity_text)
            
            # Try to generalize a pattern
            if len(entity_text) > 3:  # Only try for longer entities
                try:
                    # Create a generalized pattern
                    pattern = self._create_generalized_pattern(entity_text, text)
                    
                    # Add pattern if it doesn't exist yet
                    if entity_type in self.templates and "patterns" in self.templates[entity_type]:
                        if pattern not in self.templates[entity_type]["patterns"]:
                            self.templates[entity_type]["patterns"].append(pattern)
                            # Compile and add to patterns
                            self.entity_patterns[entity_type].append(re.compile(pattern))
                    else:
                        # Initialize patterns for this entity type
                        self.add_entity_template(entity_type, patterns=[pattern])
                except:
                    # Pattern creation failed, just use as example
                    pass
                    
            # Extract context clues
            start = entity.get('start', 0)
            end = entity.get('end', 0)
            if start > 0 and end < len(text):
                # Get context before entity (up to 3 words)
                before_context = text[max(0, start-30):start].strip().split()[-3:]
                
                # Get context after entity (up to 3 words)
                after_context = text[end:min(len(text), end+30)].strip().split()[:3]
                
                # Check for potential context clues
                potential_clues = []
                for word in before_context + after_context:
                    word = word.lower().strip(',.:;()"\'')
                    if word and len(word) > 2:
                        if entity_type in self.templates and "context_clues" in self.templates[entity_type]:
                            if word not in self.templates[entity_type]["context_clues"]:
                                potential_clues.append(word)
                        else:
                            potential_clues.append(word)
                
                # Add potential clues
                if potential_clues:
                    if entity_type in self.templates and "context_clues" in self.templates[entity_type]:
                        self.templates[entity_type]["context_clues"].extend(potential_clues)
                    else:
                        self.add_entity_template(entity_type, context_clues=potential_clues)
    
    def _create_generalized_pattern(self, entity_text, context):
        """Create a generalized regex pattern from an example entity"""
        # Start with exact text, escape regex special chars
        pattern = re.escape(entity_text)
        
        # Replace digits with \d+ pattern
        if re.search(r'\d', entity_text):
            pattern = re.sub(r'\\d+', r'\\d+', pattern)
            pattern = re.sub(r'\\d', r'\\d', pattern)
        
        # Allow variations in spaces
        pattern = pattern.replace('\\ ', r'\s+')
        
        # Allow variations in punctuation
        pattern = pattern.replace('\\,', r'[,]?')
        pattern = pattern.replace('\\.', r'[.]?')
        
        # Add word boundaries
        pattern = rf'\b{pattern}\b'
        
        # Test if pattern matches the original entity
        if not re.search(pattern, entity_text):
            raise ValueError("Generated pattern doesn't match original entity")
            
        return pattern