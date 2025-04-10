"""
Entity Linking Detector for Medical Claims

This module provides entity linking capabilities for detecting contradictions
between related medical claims that are not directly compared.
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Any
import re
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("entity-linking-detector")

@dataclass
class MedicalEntity:
    """Medical entity extracted from text."""
    
    id: str
    name: str
    type: str
    start_pos: int
    end_pos: int
    metadata: Dict[str, Any] = field(default_factory=dict)

class EntityLinkingDetector:
    """
    Entity linking detector for medical claims.
    
    This class extracts medical entities from claims and links them to detect
    contradictions between related claims that are not directly compared.
    """
    
    def __init__(
        self, 
        biomedlm_scorer=None, 
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the entity linking detector.
        
        Args:
            biomedlm_scorer: BioMedLMScorer instance for contradiction detection
            device: Device to run the model on
            config: Configuration dictionary
        """
        self.biomedlm_scorer = biomedlm_scorer
        self.device = device
        self.config = config or {}
        
        # Initialize entity extractor
        self.entity_extractor = None
        self._initialize_entity_extractor()
        
        # Initialize entity linker
        self.entity_linker = None
        self._initialize_entity_linker()
        
        # Entity cache
        self.entity_cache = {}
    
    def _initialize_entity_extractor(self):
        """Initialize entity extractor."""
        try:
            # Try to use spaCy for entity extraction
            import spacy
            
            # Load spaCy model
            try:
                self.entity_extractor = spacy.load("en_core_sci_md")
                logger.info("Loaded spaCy model 'en_core_sci_md' for entity extraction")
            except OSError:
                # Fall back to standard model
                try:
                    self.entity_extractor = spacy.load("en_core_web_sm")
                    logger.info("Loaded spaCy model 'en_core_web_sm' for entity extraction")
                except OSError:
                    logger.warning("Failed to load spaCy model. Using regex-based entity extraction.")
                    self.entity_extractor = None
        except ImportError:
            logger.warning("spaCy not available. Using regex-based entity extraction.")
            self.entity_extractor = None
    
    def _initialize_entity_linker(self):
        """Initialize entity linker."""
        # Simple dictionary-based entity linker
        self.entity_linker = {
            # Disease synonyms
            "heart attack": ["myocardial infarction", "cardiac arrest", "coronary thrombosis"],
            "diabetes": ["diabetes mellitus", "type 2 diabetes", "t2dm"],
            "cancer": ["malignancy", "neoplasm", "tumor", "carcinoma"],
            "hypertension": ["high blood pressure", "elevated blood pressure"],
            "stroke": ["cerebrovascular accident", "cva", "brain attack"],
            
            # Treatment synonyms
            "aspirin": ["acetylsalicylic acid", "asa"],
            "statin": ["hmg-coa reductase inhibitor", "atorvastatin", "simvastatin", "rosuvastatin"],
            "antibiotic": ["antimicrobial", "antibacterial"],
            "surgery": ["operation", "surgical procedure", "surgical intervention"],
            "chemotherapy": ["chemo", "cytotoxic therapy", "anticancer drugs"],
            
            # Outcome synonyms
            "mortality": ["death rate", "fatality", "survival rate"],
            "efficacy": ["effectiveness", "efficiency", "potency"],
            "adverse effect": ["side effect", "adverse reaction", "adverse event", "complication"],
            "improvement": ["amelioration", "enhancement", "betterment"],
            "recurrence": ["relapse", "return", "reappearance"]
        }
    
    def extract_entities(self, text: str) -> List[MedicalEntity]:
        """
        Extract medical entities from text.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        # Check if we have cached entities for this text
        if text in self.entity_cache:
            return self.entity_cache[text]
        
        # Use spaCy if available
        if self.entity_extractor is not None:
            try:
                doc = self.entity_extractor(text)
                
                for ent in doc.ents:
                    entity = MedicalEntity(
                        id=f"ent_{len(entities)}",
                        name=ent.text,
                        type=ent.label_,
                        start_pos=ent.start_char,
                        end_pos=ent.end_char,
                        metadata={"confidence": 0.8}
                    )
                    entities.append(entity)
            except Exception as e:
                logger.error(f"Error extracting entities with spaCy: {e}")
        
        # Fall back to regex-based extraction if no entities found
        if not entities:
            # Extract medical terms using regex patterns
            patterns = {
                "DISEASE": [
                    r"\b(?:heart attack|myocardial infarction|cardiac arrest)\b",
                    r"\b(?:diabetes|diabetes mellitus|type 2 diabetes|t2dm)\b",
                    r"\b(?:cancer|malignancy|neoplasm|tumor|carcinoma)\b",
                    r"\b(?:hypertension|high blood pressure)\b",
                    r"\b(?:stroke|cerebrovascular accident|cva)\b",
                    r"\b(?:alzheimer's|dementia)\b",
                    r"\b(?:asthma|copd|respiratory disease)\b"
                ],
                "TREATMENT": [
                    r"\b(?:aspirin|acetylsalicylic acid|asa)\b",
                    r"\b(?:statin|hmg-coa reductase inhibitor|atorvastatin|simvastatin|rosuvastatin)\b",
                    r"\b(?:antibiotic|antimicrobial|antibacterial)\b",
                    r"\b(?:surgery|operation|surgical procedure)\b",
                    r"\b(?:chemotherapy|chemo|cytotoxic therapy)\b",
                    r"\b(?:radiation|radiotherapy)\b",
                    r"\b(?:vaccine|vaccination|immunization)\b"
                ],
                "OUTCOME": [
                    r"\b(?:mortality|death rate|fatality|survival rate)\b",
                    r"\b(?:efficacy|effectiveness|efficiency|potency)\b",
                    r"\b(?:adverse effect|side effect|adverse reaction|adverse event)\b",
                    r"\b(?:improvement|amelioration|enhancement)\b",
                    r"\b(?:recurrence|relapse|return)\b"
                ]
            }
            
            for entity_type, pattern_list in patterns.items():
                for pattern in pattern_list:
                    for match in re.finditer(pattern, text.lower()):
                        entity = MedicalEntity(
                            id=f"ent_{len(entities)}",
                            name=match.group(0),
                            type=entity_type,
                            start_pos=match.start(),
                            end_pos=match.end(),
                            metadata={"confidence": 0.6, "pattern": pattern}
                        )
                        entities.append(entity)
        
        # Cache entities
        self.entity_cache[text] = entities
        
        return entities
    
    def find_linked_entities(self, entity: MedicalEntity) -> List[str]:
        """
        Find linked entities for a given entity.
        
        Args:
            entity: Entity to find links for
            
        Returns:
            List of linked entity names
        """
        linked_entities = []
        
        # Check if entity name is in our linker
        entity_name = entity.name.lower()
        if entity_name in self.entity_linker:
            linked_entities.extend(self.entity_linker[entity_name])
        
        # Check if entity name is a value in our linker
        for key, values in self.entity_linker.items():
            if entity_name in values:
                linked_entities.append(key)
                linked_entities.extend([v for v in values if v != entity_name])
        
        return linked_entities
    
    def detect_contradiction(self, claim1: str, claim2: str) -> Dict[str, Any]:
        """
        Detect contradiction between claims using entity linking.
        
        Args:
            claim1: First medical claim
            claim2: Second medical claim
            
        Returns:
            Dictionary with contradiction detection results
        """
        result = {
            "text1": claim1,
            "text2": claim2,
            "has_contradiction": False,
            "contradiction_score": 0.0,
            "method": "entity_linking",
            "entity_analysis": {}
        }
        
        # Extract entities from claims
        entities1 = self.extract_entities(claim1)
        entities2 = self.extract_entities(claim2)
        
        # Add entities to result
        result["entity_analysis"]["entities1"] = [
            {"id": e.id, "name": e.name, "type": e.type} for e in entities1
        ]
        result["entity_analysis"]["entities2"] = [
            {"id": e.id, "name": e.name, "type": e.type} for e in entities2
        ]
        
        # Find linked entities
        linked_pairs = []
        for entity1 in entities1:
            linked_entities1 = self.find_linked_entities(entity1)
            
            for entity2 in entities2:
                # Check if entities are directly linked
                if entity2.name.lower() in linked_entities1:
                    linked_pairs.append((entity1, entity2))
                    continue
                
                # Check if entities are indirectly linked
                linked_entities2 = self.find_linked_entities(entity2)
                if any(e.lower() in linked_entities2 for e in linked_entities1):
                    linked_pairs.append((entity1, entity2))
        
        # Add linked pairs to result
        result["entity_analysis"]["linked_pairs"] = [
            {"entity1": e1.name, "entity2": e2.name} for e1, e2 in linked_pairs
        ]
        
        # If no linked entities found, return early
        if not linked_pairs:
            return result
        
        # Check for contradictions between linked entities
        contradictions = []
        for entity1, entity2 in linked_pairs:
            # Extract context around entities
            context1 = self._extract_context(claim1, entity1)
            context2 = self._extract_context(claim2, entity2)
            
            # Check for contradiction using BioMedLM
            if self.biomedlm_scorer is not None:
                try:
                    contradiction_result = self.biomedlm_scorer.detect_contradiction(context1, context2)
                    
                    if contradiction_result.get("has_contradiction", False):
                        contradictions.append({
                            "entity1": entity1.name,
                            "entity2": entity2.name,
                            "context1": context1,
                            "context2": context2,
                            "contradiction_score": contradiction_result.get("contradiction_score", 0.0)
                        })
                except Exception as e:
                    logger.error(f"Error detecting contradiction between contexts: {e}")
        
        # Add contradictions to result
        result["entity_analysis"]["contradictions"] = contradictions
        
        # Calculate overall contradiction score
        if contradictions:
            max_score = max(c["contradiction_score"] for c in contradictions)
            result["contradiction_score"] = max_score
            result["has_contradiction"] = max_score > 0.7
        
        return result
    
    def _extract_context(self, text: str, entity: MedicalEntity) -> str:
        """
        Extract context around an entity.
        
        Args:
            text: Full text
            entity: Entity to extract context for
            
        Returns:
            Context around the entity
        """
        # Get sentence containing the entity
        sentences = re.split(r'[.!?]', text)
        
        for sentence in sentences:
            if entity.name in sentence:
                return sentence.strip()
        
        # Fall back to window around entity
        start = max(0, entity.start_pos - 50)
        end = min(len(text), entity.end_pos + 50)
        
        return text[start:end].strip()
