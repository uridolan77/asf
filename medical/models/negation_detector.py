"""
Negation Detection Module

This module provides utilities for detecting negation in medical text,
which is crucial for accurate contradiction detection.
"""

import logging
import re
from typing import Dict, List, Any, Tuple, Optional

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("negation-detector")

class NegationDetector:
    """
    Detect negated concepts in medical text.
    
    This class provides methods for identifying negated terms and concepts
    in medical text, which is crucial for accurate contradiction detection.
    It can use spaCy with negspaCy if available, or fall back to a rule-based
    approach using regular expressions.
    """
    
    def __init__(self, use_spacy: bool = True):
        """
        Initialize the negation detector.
        
        Args:
            use_spacy: Whether to use spaCy with negspaCy (if available)
        """
        self.use_spacy = use_spacy
        self.spacy_model = None
        
        if self.use_spacy:
            try:
                import spacy
                
                try:
                    self.spacy_model = spacy.load("en_core_sci_md")
                    
                    self.spacy_model.add_pipe(
                        "negex",
                        config={"ent_types": ["CONDITION", "TREATMENT", "TEST", "DISEASE", "CHEMICAL"]}
                    )
                    
                    logger.info("Initialized negspaCy for negation detection")
                except ImportError:
                    logger.warning("negspaCy not available. Falling back to rule-based approach.")
                    self.use_spacy = False
            except ImportError:
                logger.warning("spaCy not available. Falling back to rule-based approach.")
                self.use_spacy = False
        
        self.negation_triggers = [
            "no", "not", "none", "negative", "without", "absence of", "absent", 
            "deny", "denies", "denied", "doesn't", "does not", "don't", "do not",
            "didn't", "did not", "never", "cannot", "can't", "couldn't", "could not",
            "free of", "lack of", "lacking", "lacks", "neither", "nor", "non", "un",
            "rather than", "instead of", "ruled out", "excluded", "unlikely", "fails to",
            "failed to", "no evidence of", "no sign of", "no indication of", "no suggestion of"
        ]
        
        self.negation_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(trigger) for trigger in self.negation_triggers) + r')\b',
            re.IGNORECASE
        )
        
        self.window_size = 6
    
    def detect_negation_spacy(self, text: str) -> Dict[str, Any]:
        """
        Detect negation using spaCy and negspaCy.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with entities and their negation status
        """
        doc = self.spacy_model(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "is_negated": ent._.negex
            })
        
        return {
            "text": text,
            "entities": entities,
            "has_negation": any(ent["is_negated"] for ent in entities)
        }
    
    def detect_negation_rule_based(self, text: str) -> Dict[str, Any]:
        """
        Detect negation using rule-based approach.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with potential negated spans
        """
        negation_matches = list(self.negation_pattern.finditer(text))
        
        if not negation_matches:
            return {
                "text": text,
                "negated_spans": [],
                "has_negation": False
            }
        
        words = re.findall(r'\b\w+\b', text)
        
        negated_spans = []
        
        for match in negation_matches:
            trigger = match.group()
            trigger_start = match.start()
            trigger_end = match.end()
            
            trigger_word_index = None
            for i, word in enumerate(words):
                if word.lower() == trigger.lower():
                    trigger_word_index = i
                    break
            
            if trigger_word_index is not None:
                scope_start = trigger_word_index + 1
                scope_end = min(len(words), trigger_word_index + self.window_size + 1)
                
                negated_words = words[scope_start:scope_end]
                
                if negated_words:
                    negated_text = ' '.join(negated_words)
                    
                    approx_start = text.find(negated_words[0], trigger_end)
                    if approx_start != -1:
                        approx_end = approx_start + len(negated_text)
                        
                        negated_spans.append({
                            "text": negated_text,
                            "trigger": trigger,
                            "start": approx_start,
                            "end": approx_end
                        })
        
        return {
            "text": text,
            "negated_spans": negated_spans,
            "has_negation": len(negated_spans) > 0
        }
    
    def detect_negation(self, text: str) -> Dict[str, Any]:
        """
        Detect negation in text using the best available method.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with negation information
        """
        if self.use_spacy and self.spacy_model:
            return self.detect_negation_spacy(text)
        else:
            return self.detect_negation_rule_based(text)
    
    def compare_negation(self, text1: str, text2: str) -> Dict[str, Any]:
        """
        Compare negation between two texts to identify potential contradictions.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Dictionary with negation comparison results
        """
        negation1 = self.detect_negation(text1)
        negation2 = self.detect_negation(text2)
        
        contradictions = []
        
        if self.use_spacy and self.spacy_model:
            entities1 = {ent["text"].lower(): ent for ent in negation1["entities"]}
            entities2 = {ent["text"].lower(): ent for ent in negation2["entities"]}
            
            common_entities = set(entities1.keys()) & set(entities2.keys())
            
            for entity in common_entities:
                if entities1[entity]["is_negated"] != entities2[entity]["is_negated"]:
                    contradictions.append({
                        "entity": entity,
                        "negated_in_text1": entities1[entity]["is_negated"],
                        "negated_in_text2": entities2[entity]["is_negated"]
                    })
        else:
            negated_words1 = set()
            for span in negation1["negated_spans"]:
                negated_words1.update(re.findall(r'\b\w+\b', span["text"].lower()))
            
            negated_words2 = set()
            for span in negation2["negated_spans"]:
                negated_words2.update(re.findall(r'\b\w+\b', span["text"].lower()))
            
            words1 = set(re.findall(r'\b\w+\b', text1.lower()))
            words2 = set(re.findall(r'\b\w+\b', text2.lower()))
            
            common_words = words1 & words2
            
            for word in common_words:
                if word in negated_words1 and word not in negated_words2:
                    contradictions.append({
                        "word": word,
                        "negated_in_text1": True,
                        "negated_in_text2": False
                    })
                elif word not in negated_words1 and word in negated_words2:
                    contradictions.append({
                        "word": word,
                        "negated_in_text1": False,
                        "negated_in_text2": True
                    })
        
        return {
            "text1": text1,
            "text2": text2,
            "negation1": negation1,
            "negation2": negation2,
            "contradictions": contradictions,
            "has_negation_contradiction": len(contradictions) > 0
        }


class NegationAwareContradictionDetector:
    """
    Contradiction detector that is aware of negation.
    
    This class combines negation detection with other contradiction detection
    methods to improve accuracy.
    """
    
    def __init__(self, negation_detector: NegationDetector, biomedlm_scorer=None):
        """
        Initialize the negation-aware contradiction detector.
        
        Args:
            negation_detector: NegationDetector instance
            biomedlm_scorer: Optional BioMedLMScorer instance
        """
        self.negation_detector = negation_detector
        self.biomedlm_scorer = biomedlm_scorer
    
    def detect_contradiction(self, text1: str, text2: str) -> Dict[str, Any]:
        """
        Detect contradiction between two texts, considering negation.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Dictionary with contradiction detection results
        """
        negation_result = self.negation_detector.compare_negation(text1, text2)
        
        result = {
            "text1": text1,
            "text2": text2,
            "negation_analysis": negation_result,
            "has_contradiction": negation_result["has_negation_contradiction"],
            "contradiction_score": 0.0,
            "contradiction_type": "none"
        }
        
        if negation_result["has_negation_contradiction"]:
            result["contradiction_score"] = 0.9
            result["contradiction_type"] = "negation"
        
        if self.biomedlm_scorer:
            try:
                biomedlm_scores = self.biomedlm_scorer.get_detailed_scores(text1, text2)
                
                result["biomedlm_scores"] = biomedlm_scores
                
                if not result["has_contradiction"] and biomedlm_scores["contradiction_score"] > 0.7:
                    result["has_contradiction"] = True
                    result["contradiction_score"] = biomedlm_scores["contradiction_score"]
                    result["contradiction_type"] = "semantic"
                
                elif result["has_contradiction"]:
                    result["contradiction_score"] = max(
                        result["contradiction_score"],
                        biomedlm_scores["contradiction_score"]
                    )
                    result["contradiction_type"] = "combined"
            except Exception as e:
                logger.error(f"Error using BioMedLM for contradiction scoring: {e}")
        
        return result
