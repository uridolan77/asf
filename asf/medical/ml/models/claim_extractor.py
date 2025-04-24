"""
Claim Extraction Model for Medical Research Synthesizer.

This module implements a SciBERT-based model for automatic extraction of
scientific claims from medical abstracts and full texts. It identifies
assertion statements that can be used for contradiction detection.

Key features:
- Extract explicit claims/assertions from medical text
- Distinguish between factual claims and hedged statements
- Classify claim type (methodology, result, interpretation)
- Link claims to evidence within the text
"""

import os
import re
import json
import time
import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    pipeline,
    BatchEncoding
)

import spacy
from spacy.tokens import Doc, Span
try:
    import en_core_sci_md
    SCI_SPACY_AVAILABLE = True
except ImportError:
    SCI_SPACY_AVAILABLE = False

from asf.medical.core.logging_config import get_logger
from asf.medical.ml.models.model_registry import (
    ModelRegistry, ModelStatus, ModelMetrics, ModelFramework, get_model_registry
)

logger = get_logger(__name__)

class ClaimType(str, Enum):
    """Types of scientific claims."""
    FINDING = "finding"          # Empirical finding or result
    METHODOLOGY = "methodology"  # Methodological statement
    INTERPRETATION = "interpretation"  # Interpretation of findings
    BACKGROUND = "background"    # Background knowledge or context
    IMPLICATION = "implication"  # Implications of findings
    OTHER = "other"              # Other types of claims


@dataclass
class ExtractedClaim:
    """Representation of an extracted scientific claim."""
    text: str
    start_char: int
    end_char: int
    confidence: float
    claim_type: ClaimType = ClaimType.FINDING
    evidence_spans: List[Tuple[int, int]] = None
    hedge_level: float = 0.0  # 0.0 = absolute claim, 1.0 = highly hedged
    negated: bool = False
    related_entities: List[Dict[str, Any]] = None
    source_doc_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "text": self.text,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "confidence": self.confidence,
            "claim_type": self.claim_type.value,
            "hedge_level": self.hedge_level,
            "negated": self.negated,
        }
        
        if self.evidence_spans:
            result["evidence_spans"] = self.evidence_spans
        
        if self.related_entities:
            result["related_entities"] = self.related_entities
            
        if self.source_doc_id:
            result["source_doc_id"] = self.source_doc_id
            
        return result


class ClaimExtractor:
    """
    Extract scientific claims from medical abstracts and full texts.
    
    This class combines multiple techniques to extract claims:
    1. SciBERT-based sequence labeling for claim identification
    2. Rule-based post-processing for claim refinement
    3. Claim type classification using a separate classifier
    4. Entity and relation linking for context
    """
    
    def __init__(
        self,
        model_name: str = "allenai/scibert_scivocab_uncased",
        claim_classifier_name: str = "claim_classifier",
        device: str = None,
        batch_size: int = 4,
        cache_dir: Optional[str] = None,
        sci_spacy: bool = True,
        use_registry: bool = True
    ):
        """
        Initialize the claim extractor.
        
        Args:
            model_name: HuggingFace model name or path for token classification
            claim_classifier_name: Name of claim classifier model in registry
            device: Device to run models on (cuda, cpu, etc.)
            batch_size: Batch size for processing
            cache_dir: Directory to cache models
            sci_spacy: Whether to use ScispaCy for preprocessing
            use_registry: Whether to use model registry
        """
        self.model_name = model_name
        self.claim_classifier_name = claim_classifier_name
        self.batch_size = batch_size
        self.use_registry = use_registry
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Set cache directory
        if cache_dir is None:
            self.cache_dir = os.path.join(
                os.path.expanduser("~"), 
                ".cache", 
                "asf_medical", 
                "models"
            )
        else:
            self.cache_dir = cache_dir
            
        # Load ScispaCy for scientific text processing
        self.sci_spacy = sci_spacy and SCI_SPACY_AVAILABLE
        if self.sci_spacy:
            try:
                self.nlp = en_core_sci_md.load()
                logger.info("Loaded ScispaCy model for scientific text processing")
            except Exception as e:
                logger.warning(f"Failed to load ScispaCy: {e}. Using regular spaCy.")
                self.nlp = spacy.load("en_core_web_sm")
                self.sci_spacy = False
        else:
            logger.info("Using regular spaCy for text processing")
            self.nlp = spacy.load("en_core_web_sm")
        
        # Load NER model for claim extraction
        self._load_models()
        
        # Regex patterns for claim identification (as a backup)
        self.claim_patterns = [
            r"(?:we|they|authors|researchers|our analysis|our study|our results)?\s*(?:show|demonstrate|reveal|indicate|suggest|identify|find|conclude|report|observe|present)\s+(?:that|a|an)?\s+",
            r"(?:our|these|the)\s+(?:findings|results|data|observations)\s+(?:show|demonstrate|reveal|indicate|suggest|identify|confirm|support|prove)\s+(?:that|a|an)?\s+",
            r"(?:it|this|that)\s+(?:is|was)\s+(?:shown|demonstrated|revealed|indicated|suggested|found|observed|reported|concluded)\s+(?:that)?\s+",
            r"(?:we|they|authors|researchers)?\s*(?:conclude|reported|concluded|proposed|argued|hypothesized)\s+(?:that)?\s+"
        ]
        
        # Compiled regex patterns
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.claim_patterns]
        
        logger.info(f"ClaimExtractor initialized with model: {model_name}")
    
    def _load_models(self) -> None:
        """Load models for claim extraction."""
        try:
            # 1. Load token classification model for sequence labeling
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            # Try to load from registry first if enabled
            if self.use_registry:
                registry = get_model_registry()
                metadata = registry.get_production_model("claim_extractor")
                
                if metadata and metadata.path:
                    logger.info(f"Loading claim extractor model from registry: {metadata.path}")
                    self.model = AutoModelForTokenClassification.from_pretrained(
                        metadata.path,
                        cache_dir=self.cache_dir
                    ).to(self.device)
                else:
                    # Load base model and set for fine-tuning later
                    logger.info(f"Loading base model for claim extraction: {self.model_name}")
                    self.model = AutoModelForTokenClassification.from_pretrained(
                        self.model_name,
                        num_labels=3,  # B-CLAIM, I-CLAIM, O (Beginning, Inside, Outside)
                        cache_dir=self.cache_dir
                    ).to(self.device)
            else:
                # Just load the base model
                self.model = AutoModelForTokenClassification.from_pretrained(
                    self.model_name,
                    num_labels=3,  # B-CLAIM, I-CLAIM, O
                    cache_dir=self.cache_dir
                ).to(self.device)
            
            # 2. Create NER pipeline for claim extraction
            self.token_classifier = pipeline(
                "token-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                aggregation_strategy="simple",
                device=0 if self.device == "cuda" else -1
            )
            
            # 3. Load claim classifier model if available
            self.claim_classifier = None
            if self.use_registry:
                registry = get_model_registry()
                metadata = registry.get_production_model(self.claim_classifier_name)
                
                if metadata and metadata.path:
                    logger.info(f"Loading claim classifier from registry: {metadata.path}")
                    self.claim_classifier_tokenizer = AutoTokenizer.from_pretrained(
                        metadata.path,
                        cache_dir=self.cache_dir
                    )
                    self.claim_classifier = AutoModelForSequenceClassification.from_pretrained(
                        metadata.path,
                        cache_dir=self.cache_dir
                    ).to(self.device)
            
            logger.info("Successfully loaded claim extraction models")
        except Exception as e:
            logger.error(f"Failed to load claim extraction models: {e}")
            raise
    
    def _preprocess_text(self, text: str) -> Doc:
        """
        Preprocess text with spaCy.
        
        Args:
            text: Text to preprocess
            
        Returns:
            spaCy Doc object
        """
        return self.nlp(text)
    
    def _extract_claims_with_model(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract claims using the neural model.
        
        Args:
            text: Text to extract claims from
            
        Returns:
            List of extracted entity dictionaries
        """
        try:
            # Run token classification
            entities = self.token_classifier(text)
            
            # Group entities of the same type that are adjacent
            grouped_entities = []
            current_entity = None
            
            for entity in entities:
                # Only process CLAIM entities
                if entity["entity_group"] != "CLAIM" and entity["entity_group"] != "B-CLAIM" and entity["entity_group"] != "I-CLAIM":
                    continue
                
                if current_entity is None:
                    current_entity = entity
                elif entity["start"] == current_entity["end"] or entity["start"] - current_entity["end"] <= 1:
                    # Extend the current entity
                    current_entity["word"] = text[current_entity["start"]:entity["end"]]
                    current_entity["end"] = entity["end"]
                    current_entity["score"] = (current_entity["score"] + entity["score"]) / 2
                else:
                    # Save current entity and start a new one
                    grouped_entities.append(current_entity)
                    current_entity = entity
            
            # Add the last entity if it exists
            if current_entity is not None:
                grouped_entities.append(current_entity)
            
            return grouped_entities
        except Exception as e:
            logger.error(f"Error extracting claims with model: {e}")
            return []
    
    def _extract_claims_with_regex(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract claims using regex patterns as a backup.
        
        Args:
            text: Text to extract claims from
            
        Returns:
            List of extracted entity dictionaries
        """
        claims = []
        
        # Find potential claims
        for pattern in self.compiled_patterns:
            for match in pattern.finditer(text):
                start = match.start()
                
                # Find the end of the claim (next period, question mark, or exclamation mark)
                sentence_end = text.find(".", start)
                if sentence_end == -1:
                    sentence_end = len(text)
                
                # Adjust if there are quote marks
                quote_end = text.find('"', start)
                if quote_end != -1 and quote_end < sentence_end:
                    sentence_end = text.find('"', quote_end + 1)
                    if sentence_end == -1:
                        sentence_end = len(text)
                
                # Get the claim text
                claim_text = text[start:sentence_end].strip()
                
                # Add to list
                claims.append({
                    "entity_group": "CLAIM",
                    "score": 0.7,  # Lower confidence for regex-based
                    "word": claim_text,
                    "start": start,
                    "end": sentence_end
                })
        
        return claims
    
    def _classify_claim_type(self, claim_text: str) -> Tuple[ClaimType, float]:
        """
        Classify the type of claim.
        
        Args:
            claim_text: The text of the claim
            
        Returns:
            Tuple of (claim_type, confidence)
        """
        if self.claim_classifier is None:
            # Use rule-based classification
            if any(term in claim_text.lower() for term in ["method", "approach", "technique", "procedure", "protocol", "design"]):
                return ClaimType.METHODOLOGY, 0.8
            elif any(term in claim_text.lower() for term in ["suggest", "might", "may", "could", "potentially", "possibly", "interpret"]):
                return ClaimType.INTERPRETATION, 0.8
            elif any(term in claim_text.lower() for term in ["implication", "impact", "future", "should", "policy", "clinical practice"]):
                return ClaimType.IMPLICATION, 0.8
            elif any(term in claim_text.lower() for term in ["background", "previously", "historically", "literature", "known"]):
                return ClaimType.BACKGROUND, 0.8
            else:
                return ClaimType.FINDING, 0.7
        else:
            # Use the neural claim classifier
            try:
                inputs = self.claim_classifier_tokenizer(
                    claim_text, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=512
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.claim_classifier(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
                
                # Get the predicted class and confidence
                predicted_class_idx = torch.argmax(probs).item()
                confidence = probs[predicted_class_idx].item()
                
                # Map the index to ClaimType
                class_map = {
                    0: ClaimType.FINDING,
                    1: ClaimType.METHODOLOGY,
                    2: ClaimType.INTERPRETATION,
                    3: ClaimType.BACKGROUND,
                    4: ClaimType.IMPLICATION,
                    5: ClaimType.OTHER
                }
                
                return class_map.get(predicted_class_idx, ClaimType.FINDING), confidence
            except Exception as e:
                logger.error(f"Error classifying claim: {e}")
                return ClaimType.FINDING, 0.5
    
    def _check_hedge_level(self, claim_text: str) -> float:
        """
        Check the level of hedging in a claim.
        
        Args:
            claim_text: The text of the claim
            
        Returns:
            Hedge level from 0.0 (absolute) to 1.0 (highly hedged)
        """
        hedge_terms = {
            # Strong hedges
            "may": 0.7,
            "might": 0.8,
            "could": 0.7,
            "possibly": 0.8,
            "perhaps": 0.8,
            "potentially": 0.7,
            "suggest": 0.6,
            "indicate": 0.5,
            "seem": 0.6,
            
            # Moderate hedges
            "often": 0.4,
            "likely": 0.4,
            "generally": 0.4,
            "typically": 0.4,
            "usually": 0.3,
            
            # Weak hedges
            "somewhat": 0.3,
            "relatively": 0.2,
            "approximately": 0.2,
            "around": 0.2
        }
        
        # Check for hedge terms
        max_hedge = 0.0
        lower_claim = claim_text.lower()
        
        for term, value in hedge_terms.items():
            if f" {term} " in f" {lower_claim} ":
                max_hedge = max(max_hedge, value)
        
        return max_hedge
    
    def _check_negation(self, claim_text: str) -> bool:
        """
        Check if a claim is negated.
        
        Args:
            claim_text: The text of the claim
            
        Returns:
            True if claim is negated, False otherwise
        """
        negation_terms = [
            "not", "no", "never", "neither", "nor", "none", "doesn't", 
            "don't", "didn't", "haven't", "hasn't", "won't", "weren't",
            "wasn't", "isn't", "aren't", "cannot", "can't", "couldn't"
        ]
        
        lower_claim = claim_text.lower()
        
        # Check for simple negation
        for term in negation_terms:
            if f" {term} " in f" {lower_claim} ":
                return True
        
        return False
    
    def _extract_entities(self, doc: Doc, claim: ExtractedClaim) -> List[Dict[str, Any]]:
        """
        Extract entities related to the claim.
        
        Args:
            doc: spaCy Doc object
            claim: The extracted claim
            
        Returns:
            List of related entities
        """
        entities = []
        
        # Find entities within the claim span
        claim_span = doc.char_span(claim.start_char, claim.end_char)
        if claim_span is None:
            return entities
        
        # Extract entities
        for ent in doc.ents:
            # Check if entity is within or overlaps with claim
            if (ent.start_char >= claim.start_char and ent.end_char <= claim.end_char) or \
               (ent.start_char <= claim.start_char and ent.end_char >= claim.start_char) or \
               (ent.start_char <= claim.end_char and ent.end_char >= claim.end_char):
                entities.append({
                    "text": ent.text,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                    "label": ent.label_,
                    "within_claim": (ent.start_char >= claim.start_char and 
                                    ent.end_char <= claim.end_char)
                })
        
        return entities
    
    def extract_claims(self, text: str, doc_id: Optional[str] = None) -> List[ExtractedClaim]:
        """
        Extract claims from text.
        
        Args:
            text: Text to extract claims from
            doc_id: Optional document ID
            
        Returns:
            List of ExtractedClaim objects
        """
        if not text or not text.strip():
            return []
        
        # Preprocess text
        doc = self._preprocess_text(text)
        
        # Extract claims using model
        model_claims = self._extract_claims_with_model(text)
        
        # If model didn't find claims, try regex as backup
        if not model_claims:
            model_claims = self._extract_claims_with_regex(text)
        
        # Process extracted claims
        result = []
        for claim_info in model_claims:
            # Get claim text and position
            claim_text = claim_info["word"]
            start_pos = claim_info["start"]
            end_pos = claim_info["end"]
            confidence = claim_info["score"]
            
            # Classify claim type
            claim_type, type_confidence = self._classify_claim_type(claim_text)
            
            # Check hedge level
            hedge_level = self._check_hedge_level(claim_text)
            
            # Check negation
            negated = self._check_negation(claim_text)
            
            # Create claim object
            claim = ExtractedClaim(
                text=claim_text,
                start_char=start_pos,
                end_char=end_pos,
                confidence=confidence,
                claim_type=claim_type,
                hedge_level=hedge_level,
                negated=negated,
                evidence_spans=[],
                related_entities=[],
                source_doc_id=doc_id
            )
            
            # Extract related entities
            claim.related_entities = self._extract_entities(doc, claim)
            
            result.append(claim)
        
        return result
    
    def batch_extract_claims(
        self, 
        texts: List[str], 
        doc_ids: Optional[List[str]] = None
    ) -> Dict[str, List[ExtractedClaim]]:
        """
        Extract claims from a batch of texts.
        
        Args:
            texts: List of texts
            doc_ids: Optional list of document IDs
            
        Returns:
            Dictionary mapping document IDs to lists of ExtractedClaim objects
        """
        if doc_ids is None:
            doc_ids = [f"doc_{i}" for i in range(len(texts))]
        
        results = {}
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_ids = doc_ids[i:i + self.batch_size]
            
            for text, doc_id in zip(batch_texts, batch_ids):
                claims = self.extract_claims(text, doc_id)
                results[doc_id] = claims
        
        return results
    
    def extract_claims_from_abstract(
        self, 
        abstract: str, 
        doc_id: Optional[str] = None
    ) -> List[ExtractedClaim]:
        """
        Extract claims specifically from an abstract.
        
        Args:
            abstract: Abstract text
            doc_id: Optional document ID
            
        Returns:
            List of ExtractedClaim objects
        """
        # For abstracts we focus on the conclusions and results sections
        sentences = [sent.text for sent in self._preprocess_text(abstract).sents]
        
        # Process each sentence
        all_claims = []
        for i, sentence in enumerate(sentences):
            # Give more weight to final sentences (often contain conclusions)
            is_final = (i >= len(sentences) - 2)
            
            claims = self.extract_claims(sentence, doc_id)
            
            # Boost confidence for claims in final sentences
            if is_final:
                for claim in claims:
                    claim.confidence = min(0.95, claim.confidence * 1.2)
            
            all_claims.extend(claims)
        
        return all_claims
    
    def extract_claims_from_full_text(
        self, 
        full_text: str, 
        doc_id: Optional[str] = None,
        sections: Optional[Dict[str, str]] = None
    ) -> Dict[str, List[ExtractedClaim]]:
        """
        Extract claims from full text, optionally with section information.
        
        Args:
            full_text: Full text content
            doc_id: Optional document ID
            sections: Optional dictionary mapping section names to text
            
        Returns:
            Dictionary mapping section names to lists of ExtractedClaim objects
        """
        if sections is None:
            # Try to detect sections
            detected_sections = self._detect_sections(full_text)
            if detected_sections:
                sections = detected_sections
            else:
                sections = {"full_text": full_text}
        
        results = {}
        
        # Process each section
        for section_name, section_text in sections.items():
            # Give priority to key sections
            section_lower = section_name.lower()
            is_key_section = any(key in section_lower for key in 
                               ["conclusion", "result", "finding", "discussion"])
            
            claims = self.extract_claims(section_text, doc_id)
            
            # Boost confidence for claims in key sections
            if is_key_section:
                for claim in claims:
                    claim.confidence = min(0.95, claim.confidence * 1.15)
            
            results[section_name] = claims
        
        return results
    
    def _detect_sections(self, full_text: str) -> Optional[Dict[str, str]]:
        """
        Try to detect sections in full text.
        
        Args:
            full_text: Full text content
            
        Returns:
            Dictionary mapping section names to text, or None if no sections detected
        """
        section_patterns = [
            r"(?i)^\s*(abstract)\s*[:]*\s*$",
            r"(?i)^\s*(introduction|background)\s*[:]*\s*$",
            r"(?i)^\s*(methods?|methodology|materials and methods)\s*[:]*\s*$",
            r"(?i)^\s*(results?)\s*[:]*\s*$",
            r"(?i)^\s*(discussion)\s*[:]*\s*$",
            r"(?i)^\s*(conclusion)s?\s*[:]*\s*$",
            r"(?i)^\s*(references|bibliography)\s*[:]*\s*$",
        ]
        
        # Compile patterns
        compiled_patterns = [re.compile(p) for p in section_patterns]
        
        # Split text into lines
        lines = full_text.split('\n')
        
        # Find potential section headers
        sections = {}
        current_section = "preamble"
        current_content = []
        
        for line in lines:
            matched = False
            for i, pattern in enumerate(compiled_patterns):
                if pattern.match(line):
                    # Save previous section
                    if current_content:
                        sections[current_section] = '\n'.join(current_content)
                    
                    # Start new section
                    current_section = pattern.match(line).group(1)
                    current_content = []
                    matched = True
                    break
            
            if not matched:
                current_content.append(line)
        
        # Save the last section
        if current_content:
            sections[current_section] = '\n'.join(current_content)
        
        # If we only found the preamble, return None
        if len(sections) <= 1:
            return None
        
        return sections


def train_claim_extractor(
    training_data_path: str,
    output_dir: Optional[str] = None,
    base_model_name: str = "allenai/scibert_scivocab_uncased",
    num_train_epochs: int = 3,
    learning_rate: float = 2e-5,
    batch_size: int = 8,
    register_model: bool = True,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Train a claim extraction model.
    
    Args:
        training_data_path: Path to training data
        output_dir: Directory to save the model
        base_model_name: Base model to fine-tune
        num_train_epochs: Number of training epochs
        learning_rate: Learning rate for training
        batch_size: Batch size for training
        register_model: Whether to register model in model registry
        device: Device to train on
        
    Returns:
        Dictionary with training results
    """
    # Implementation would go here...
    # This is a placeholder for the fine-tuning function
    
    # Example structure:
    # 1. Load and preprocess training data
    # 2. Create dataset objects
    # 3. Load base model
    # 4. Set up training arguments
    # 5. Train the model
    # 6. Evaluate the model
    # 7. Save the model
    # 8. Register with model registry if requested
    
    return {"status": "Training not implemented yet"}
"""