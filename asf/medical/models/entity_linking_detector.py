"""
Entity Linking Detector for Medical Claims
This module provides entity linking capabilities for detecting contradictions
between related medical claims that are not directly compared.
"""
import logging
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass, field
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
        Initialize the entity linking detector.
        Args:
            biomedlm_scorer: BioMedLMScorer instance for contradiction detection
            device: Device to run the model on
            config: Configuration dictionary
        try:
            import spacy
            try:
                self.entity_extractor = spacy.load("en_core_sci_md")
                logger.info("Loaded spaCy model 'en_core_sci_md' for entity extraction")
            except OSError:
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
        """Initialize entity linker.
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
    """
        self.entity_linker = {
            "heart attack": ["myocardial infarction", "cardiac arrest", "coronary thrombosis"],
            "diabetes": ["diabetes mellitus", "type 2 diabetes", "t2dm"],
            "cancer": ["malignancy", "neoplasm", "tumor", "carcinoma"],
            "hypertension": ["high blood pressure", "elevated blood pressure"],
            "stroke": ["cerebrovascular accident", "cva", "brain attack"],
            "aspirin": ["acetylsalicylic acid", "asa"],
            "statin": ["hmg-coa reductase inhibitor", "atorvastatin", "simvastatin", "rosuvastatin"],
            "antibiotic": ["antimicrobial", "antibacterial"],
            "surgery": ["operation", "surgical procedure", "surgical intervention"],
            "chemotherapy": ["chemo", "cytotoxic therapy", "anticancer drugs"],
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
        Find linked entities for a given entity.
        Args:
            entity: Entity to find links for
        Returns:
            List of linked entity names
        Detect contradiction between claims using entity linking.
        Args:
            claim1: First medical claim
            claim2: Second medical claim
        Returns:
            Dictionary with contradiction detection results
        Extract context around an entity.
        Args:
            text: Full text
            entity: Entity to extract context for
        Returns:
            Context around the entity