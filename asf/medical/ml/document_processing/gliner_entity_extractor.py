"""
Biomedical Entity Extraction with GLiNER

This module provides functionality for extracting biomedical entities using GLiNER-biomed,
a generalist and lightweight model for named entity recognition that can handle
overlapping and complex biomedical entities.
"""

import logging
import torch
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field

# Local imports
from .document_structure import Entity, DocumentStructure

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class EntityType:
    """Represents a biomedical entity type with description."""
    name: str
    description: str
    umls_semantic_types: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)


class GLiNERBiomedExtractor:
    """
    Biomedical entity extractor using GLiNER-biomed.
    
    GLiNER-biomed is a generalist and lightweight model for biomedical named entity recognition
    that can handle overlapping and complex entities through a span-based approach.
    """
    
    # Standard biomedical entity types with descriptions
    ENTITY_TYPES = [
        EntityType(
            name="DISEASE",
            description="a disease or medical condition",
            umls_semantic_types=["T047", "T048", "T191"],
            examples=["Alzheimer's disease", "type 2 diabetes", "COVID-19"]
        ),
        EntityType(
            name="CHEMICAL",
            description="a chemical compound, drug, or pharmaceutical substance",
            umls_semantic_types=["T116", "T123", "T120", "T103"],
            examples=["aspirin", "glucose", "dopamine", "metformin"]
        ),
        EntityType(
            name="GENE",
            description="a gene or protein",
            umls_semantic_types=["T116", "T123", "T126"],
            examples=["BRCA1", "p53", "insulin receptor", "TNF-alpha"]
        ),
        EntityType(
            name="SPECIES",
            description="a biological species or organism",
            umls_semantic_types=["T007", "T204"],
            examples=["Homo sapiens", "E. coli", "mice", "Arabidopsis thaliana"]
        ),
        EntityType(
            name="CELL_TYPE",
            description="a type of cell",
            umls_semantic_types=["T025"],
            examples=["T cells", "neurons", "hepatocytes", "fibroblasts"]
        ),
        EntityType(
            name="CELL_LINE",
            description="a cell line used in research",
            umls_semantic_types=["T025"],
            examples=["HeLa", "MCF-7", "HEK293", "Jurkat"]
        ),
        EntityType(
            name="ANATOMICAL_STRUCTURE",
            description="an anatomical structure or body part",
            umls_semantic_types=["T017", "T022", "T023", "T024"],
            examples=["brain", "liver", "hippocampus", "left ventricle"]
        ),
        EntityType(
            name="BIOLOGICAL_PROCESS",
            description="a biological process or pathway",
            umls_semantic_types=["T038", "T067"],
            examples=["apoptosis", "glycolysis", "inflammation", "DNA replication"]
        ),
        EntityType(
            name="CLINICAL_FINDING",
            description="a clinical finding or symptom",
            umls_semantic_types=["T033", "T034", "T184"],
            examples=["fever", "hypertension", "rash", "elevated heart rate"]
        ),
        EntityType(
            name="PROCEDURE",
            description="a medical procedure or intervention",
            umls_semantic_types=["T060", "T061"],
            examples=["MRI", "surgery", "vaccination", "chemotherapy"]
        )
    ]
    
    def __init__(
        self,
        model_name: str = "ds4dh/GLiNER-biomed",
        device: Optional[str] = None,
        batch_size: int = 8,
        confidence_threshold: float = 0.5,
        use_umls_linking: bool = True
    ):
        """
        Initialize the GLiNER-biomed entity extractor.
        
        Args:
            model_name: GLiNER-biomed model name
            device: Device for PyTorch models
            batch_size: Batch size for processing
            confidence_threshold: Confidence threshold for entity extraction
            use_umls_linking: Whether to use UMLS linking for extracted entities
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        self.use_umls_linking = use_umls_linking
        
        # Initialize GLiNER model
        try:
            from gliner import GLiNER
            logger.info(f"Loading GLiNER-biomed model: {model_name}")
            self.model = GLiNER(model_name=model_name, device=self.device)
            logger.info("GLiNER-biomed model loaded successfully")
        except ImportError:
            logger.error("GLiNER package not installed. Install with: pip install gliner")
            self.model = None
        except Exception as e:
            logger.error(f"Error loading GLiNER-biomed model: {str(e)}")
            self.model = None
        
        # Initialize UMLS linker if requested
        self.umls_linker = None
        if use_umls_linking:
            try:
                import spacy
                from scispacy.linking import EntityLinker
                
                logger.info("Loading SciSpacy model for UMLS linking")
                self.nlp = spacy.load("en_core_sci_md")
                
                logger.info("Adding UMLS entity linker")
                self.nlp.add_pipe(
                    "scispacy_linker", 
                    config={"resolve_abbreviations": True, "linker_name": "umls", "threshold": 0.8}
                )
                
                # Get the UMLS linker component
                self.umls_linker = self.nlp.get_pipe("scispacy_linker")
                logger.info("UMLS linker initialized successfully")
            except ImportError:
                logger.warning("SciSpacy or UMLS linker not available. Install with: pip install scispacy")
                logger.warning("Also install the required model: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_md-0.5.1.tar.gz")
            except Exception as e:
                logger.warning(f"Error initializing UMLS linker: {str(e)}")
    
    def extract_entities(self, text: str) -> List[Entity]:
        """
        Extract biomedical entities from text using GLiNER-biomed.
        
        Args:
            text: Input text
            
        Returns:
            List of Entity objects
        """
        if self.model is None:
            logger.error("GLiNER-biomed model not initialized")
            return []
        
        entities = []
        
        try:
            # Prepare entity type descriptions for GLiNER
            entity_types = {et.name: et.description for et in self.ENTITY_TYPES}
            
            # Extract entities using GLiNER
            predictions = self.model.predict_entities(
                text=text,
                entity_types=entity_types,
                batch_size=self.batch_size
            )
            
            # Process predictions
            for pred in predictions:
                if pred["score"] >= self.confidence_threshold:
                    entity = Entity(
                        text=pred["entity"],
                        label=pred["type"],
                        start=pred["start"],
                        end=pred["end"],
                        confidence=pred["score"]
                    )
                    
                    # Add to entities list
                    entities.append(entity)
            
            # Link entities to UMLS if requested
            if self.use_umls_linking and self.umls_linker is not None:
                self._link_entities_to_umls(text, entities)
            
            # Sort entities by start position
            entities.sort(key=lambda e: e.start)
            
            return entities
        except Exception as e:
            logger.error(f"Error extracting entities with GLiNER-biomed: {str(e)}")
            return []
    
    def _link_entities_to_umls(self, text: str, entities: List[Entity]) -> None:
        """
        Link extracted entities to UMLS concepts.
        
        Args:
            text: Original text
            entities: List of extracted entities
        """
        if self.umls_linker is None:
            return
        
        try:
            # Process text with spaCy
            doc = self.nlp(text)
            
            # Create a mapping from entity spans to GLiNER entities
            span_to_entity = {}
            for entity in entities:
                span_key = (entity.start, entity.end)
                span_to_entity[span_key] = entity
            
            # Process each spaCy entity
            for ent in doc.ents:
                # Try to find a matching GLiNER entity
                span_key = (ent.start_char, ent.end_char)
                
                # Look for exact or overlapping spans
                matching_entity = None
                for (start, end), entity in span_to_entity.items():
                    # Check for exact match or significant overlap
                    if (start == ent.start_char and end == ent.end_char) or \
                       (start <= ent.start_char < end) or \
                       (start < ent.end_char <= end) or \
                       (ent.start_char <= start < ent.end_char):
                        matching_entity = entity
                        break
                
                if matching_entity and hasattr(ent._, 'kb_ents') and ent._.kb_ents:
                    # Get top UMLS entity
                    umls_ent = ent._.kb_ents[0]
                    matching_entity.cui = umls_ent[0]
                    matching_entity.umls_entity = self.umls_linker.kb.cui_to_entity.get(umls_ent[0])
                    
                    # Update confidence if UMLS linking confidence is higher
                    if umls_ent[1] > matching_entity.confidence:
                        matching_entity.confidence = umls_ent[1]
        except Exception as e:
            logger.warning(f"Error linking entities to UMLS: {str(e)}")
    
    def process_document(self, doc_structure: DocumentStructure) -> DocumentStructure:
        """
        Process a document structure to extract entities from all sections.
        
        Args:
            doc_structure: Document structure
            
        Returns:
            Updated document structure with entities
        """
        logger.info("Extracting biomedical entities with GLiNER-biomed...")
        
        # Track unique entities to avoid duplicates
        unique_entities: Set[Tuple[str, str, int, int]] = set()
        
        # Process abstract
        if doc_structure.abstract:
            abstract_entities = self.extract_entities(doc_structure.abstract)
            for entity in abstract_entities:
                entity_key = (entity.text, entity.label, entity.start, entity.end)
                if entity_key not in unique_entities:
                    doc_structure.entities.append(entity)
                    unique_entities.add(entity_key)
        
        # Process sections
        for section in doc_structure.sections:
            section_entities = self.extract_entities(section.text)
            section.entities = section_entities
            
            # Add to document-level entities
            for entity in section_entities:
                entity_key = (entity.text, entity.label, entity.start, entity.end)
                if entity_key not in unique_entities:
                    doc_structure.entities.append(entity)
                    unique_entities.add(entity_key)
            
            # Process subsections recursively
            for subsection in section.subsections:
                subsection_entities = self.extract_entities(subsection.text)
                subsection.entities = subsection_entities
                
                # Add to document-level entities
                for entity in subsection_entities:
                    entity_key = (entity.text, entity.label, entity.start, entity.end)
                    if entity_key not in unique_entities:
                        doc_structure.entities.append(entity)
                        unique_entities.add(entity_key)
        
        logger.info(f"Extracted {len(doc_structure.entities)} biomedical entities")
        return doc_structure
    
    def extract_entities_batch(self, texts: List[str]) -> List[List[Entity]]:
        """
        Extract entities from multiple texts in batch mode.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of entity lists, one for each input text
        """
        if self.model is None:
            logger.error("GLiNER-biomed model not initialized")
            return [[] for _ in texts]
        
        results = []
        
        try:
            # Prepare entity type descriptions for GLiNER
            entity_types = {et.name: et.description for et in self.ENTITY_TYPES}
            
            # Process texts in batches
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i+self.batch_size]
                
                # Extract entities for batch
                batch_predictions = self.model.predict_entities_batch(
                    texts=batch_texts,
                    entity_types=entity_types
                )
                
                # Process predictions for each text
                for text_idx, predictions in enumerate(batch_predictions):
                    text_entities = []
                    
                    for pred in predictions:
                        if pred["score"] >= self.confidence_threshold:
                            entity = Entity(
                                text=pred["entity"],
                                label=pred["type"],
                                start=pred["start"],
                                end=pred["end"],
                                confidence=pred["score"]
                            )
                            text_entities.append(entity)
                    
                    # Link entities to UMLS if requested
                    if self.use_umls_linking and self.umls_linker is not None:
                        self._link_entities_to_umls(batch_texts[text_idx], text_entities)
                    
                    # Sort entities by start position
                    text_entities.sort(key=lambda e: e.start)
                    results.append(text_entities)
            
            return results
        except Exception as e:
            logger.error(f"Error extracting entities in batch mode: {str(e)}")
            return [[] for _ in texts]
