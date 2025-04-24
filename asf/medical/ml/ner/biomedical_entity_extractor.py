"""
Biomedical Entity Extractor

This module provides functionality for extracting biomedical entities from medical texts
with UMLS (Unified Medical Language System) linking capabilities.
"""

import logging
from typing import List, Optional, Dict, Any
from ..preprocessing.document_structure import Entity, DocumentStructure

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BiomedicalEntityExtractor:
    """
    Extract biomedical entities with UMLS linking using SciSpacy.
    """
    
    # Entity label mapping from SciSpacy to human-readable types
    ENTITY_LABEL_MAP = {
        "CHEMICAL": "Chemical",
        "DISEASE": "Disease",
        "ANATOMICAL_STRUCTURE": "Anatomical Structure",
        "ORGANISM": "Organism",
        "GENE_OR_GENE_PRODUCT": "Gene/Protein", 
        "PROCEDURE": "Procedure",
        "DRUG": "Drug",
        "PHENOMENON": "Phenomenon",
    }
    
    def __init__(
        self,
        spacy_model: str = "en_core_sci_md",
        use_umls: bool = True
    ):
        """
        Initialize the biomedical entity extractor.
        
        Args:
            spacy_model: SciSpacy model to use
            use_umls: Whether to use UMLS entity linking
        """
        self.nlp = None
        self.linker = None
        
        try:
            # Load SciSpacy model
            import spacy
            logger.info(f"Loading SciSpacy model: {spacy_model}")
            self.nlp = spacy.load(spacy_model)
            
            # Add abbreviation detector to the pipeline if available
            try:
                from scispacy.abbreviation import AbbreviationDetector
            except ImportError:
                AbbreviationDetector = None
                logger.warning("Could not load AbbreviationDetector from scispacy")
            if AbbreviationDetector is not None:
                self.nlp.add_pipe("abbreviation_detector")
                logger.info("Added abbreviation detector to pipeline")
            
            # Add UMLS entity linker if requested
            if use_umls:
                try:
                    from scispacy.linking import EntityLinker
                    logger.info("Adding UMLS entity linker")
                    self.nlp.add_pipe(
                        "scispacy_linker", 
                        config={"resolve_abbreviations": True, "linker_name": "umls", "threshold": 0.8}
                    )
                    self.linker = self.nlp.get_pipe("scispacy_linker")
                    logger.info("UMLS entity linker added successfully")
                except (ImportError, ModuleNotFoundError):
                    logger.warning("Could not load EntityLinker from scispacy")
            
            logger.info("Biomedical entity extractor initialized successfully")
        except Exception as e:
            logger.error(f"Error loading SciSpacy model: {str(e)}")
            logger.info("Install required SciSpacy models with: pip install scispacy && pip install en_core_sci_md")
    
    def extract_entities(self, text: str) -> List[Entity]:
        """
        Extract biomedical entities from text with UMLS linking.
        
        Args:
            text: Input text
            
        Returns:
            List of Entity objects
        """
        if self.nlp is None:
            logger.error("NLP model not initialized")
            return []
            
        doc = self.nlp(text)
        entities = []
        
        # Process entities
        for ent in doc.ents:
            # Get readable entity type
            entity_type = self.ENTITY_LABEL_MAP.get(ent.label_, ent.label_)
            
            # Create entity object
            entity = Entity(
                text=ent.text,
                label=entity_type,
                start=ent.start_char,
                end=ent.end_char,
                confidence=1.0  # Default confidence
            )
            
            # Add UMLS information if available
            if self.linker and hasattr(ent._, 'kb_ents') and ent._.kb_ents:
                # Get top UMLS entity
                umls_ent = ent._.kb_ents[0]
                entity.cui = umls_ent[0]
                entity.confidence = umls_ent[1]
                entity.umls_entity = self.linker.kb.cui_to_entity.get(umls_ent[0])
            
            # Check for abbreviations
            if hasattr(ent._, 'is_abbreviated') and ent._.is_abbreviated:
                entity.abbreviation = ent._.long_form
            elif hasattr(ent._, 'is_long_form') and ent._.is_long_form:
                entity.abbreviation = ent._.short_form
            
            entities.append(entity)
        
        return entities
    
    def process_document(self, doc_structure: DocumentStructure) -> DocumentStructure:
        """
        Process a document structure to extract entities from all sections.
        
        Args:
            doc_structure: Document structure
            
        Returns:
            Updated document structure with entities
        """
        logger.info("Extracting biomedical entities...")
        
        # Process abstract
        if doc_structure.abstract:
            abstract_entities = self.extract_entities(doc_structure.abstract)
            for entity in abstract_entities:
                doc_structure.entities.append(entity)
        
        # Process sections
        for section in doc_structure.sections:
            section_entities = self.extract_entities(section.text)
            section.entities = section_entities
            
            # Add to document-level entities
            for entity in section_entities:
                doc_structure.entities.append(entity)
            
            # Process subsections recursively
            for subsection in section.subsections:
                subsection_entities = self.extract_entities(subsection.text)
                subsection.entities = subsection_entities
                
                # Add to document-level entities
                for entity in subsection_entities:
                    doc_structure.entities.append(entity)
        
        logger.info(f"Extracted {len(doc_structure.entities)} biomedical entities")
        return doc_structure
