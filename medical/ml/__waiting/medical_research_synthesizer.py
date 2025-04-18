import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    T5ForConditionalGeneration, 
    T5Tokenizer,
    AdamW,
    get_linear_schedule_with_warmup
)
from torch_geometric.nn import GCNConv, GATv2Conv, global_mean_pool
import spacy
import scispacy
from scispacy.linking import EntityLinker
from scispacy.abbreviation import AbbreviationDetector
import numpy as np
import re
import json
import logging
import fitz  # PyMuPDF
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union, Any, Set
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from itertools import combinations
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


#################################################
#          Document Structure Classes           #
#################################################

@dataclass
class Entity:
    """Biomedical entity with UMLS linking."""
    text: str
    label: str
    start: int
    end: int
    cui: Optional[str] = None  # UMLS Concept Unique Identifier
    umls_entity: Optional[Dict] = None
    confidence: float = 1.0
    abbreviation: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "text": self.text,
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "cui": self.cui,
            "confidence": self.confidence,
            "abbreviation": self.abbreviation,
            "umls_entity": self.umls_entity
        }

@dataclass
class SectionInfo:
    """Information about a section in a scientific document."""
    section_type: str
    heading: str
    text: str
    start_pos: int
    end_pos: int
    subsections: List['SectionInfo'] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)
    relations: List[Dict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DocumentStructure:
    """Structured representation of a scientific document."""
    title: str
    abstract: Optional[str] = None
    sections: List[SectionInfo] = field(default_factory=list)
    references: List[Dict[str, Any]] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)
    relations: List[Dict] = field(default_factory=list)
    knowledge_graph: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    summary: Optional[Dict[str, str]] = None

@dataclass
class RelationInstance:
    """A single relation instance between two entities."""
    head_entity: str
    tail_entity: str
    head_type: str
    tail_type: str
    relation_type: str
    confidence: float
    context: str
    sentence_id: Optional[int] = None
    head_cui: Optional[str] = None
    tail_cui: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "head_entity": self.head_entity,
            "tail_entity": self.tail_entity,
            "head_type": self.head_type,
            "tail_type": self.tail_type,
            "relation_type": self.relation_type,
            "confidence": self.confidence,
            "context": self.context,
            "sentence_id": self.sentence_id,
            "head_cui": self.head_cui,
            "tail_cui": self.tail_cui
        }

@dataclass
class ResearchSummary:
    """Research paper summary."""
    abstract: str
    key_findings: str
    methods: Optional[str] = None
    conclusions: Optional[str] = None
    limitations: Optional[str] = None
    clinical_implications: Optional[str] = None
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items() if v is not None}
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def to_markdown(self) -> str:
        """Convert to markdown format."""
        md = "# Research Summary\n\n"
        
        if self.abstract:
            md += "## Abstract\n\n" + self.abstract + "\n\n"
        
        if self.key_findings:
            md += "## Key Findings\n\n" + self.key_findings + "\n\n"
        
        if self.methods:
            md += "## Methods\n\n" + self.methods + "\n\n"
        
        if self.conclusions:
            md += "## Conclusions\n\n" + self.conclusions + "\n\n"
        
        if self.limitations:
            md += "## Limitations\n\n" + self.limitations + "\n\n"
        
        if self.clinical_implications:
            md += "## Clinical Implications\n\n" + self.clinical_implications + "\n\n"
        
        return md


#################################################
#       Document Processor with SciSpacy        #
#################################################

class BiomedicalDocumentProcessor:
    """
    Medical document processor using SciBERT and SciSpacy.
    
    Handles PDF and text inputs, section classification, and structure extraction.
    """
    
    def __init__(
        self,
        section_classifier_model: str = "allenai/scibert_scivocab_uncased",
        spacy_model: str = "en_core_sci_md",
        use_umls: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize the biomedical document processor.
        
        Args:
            section_classifier_model: SciBERT model for section classification
            spacy_model: SciSpacy model for NLP processing
            use_umls: Whether to use UMLS entity linking
            device: Device for PyTorch models
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load section classifier (SciBERT)
        logger.info(f"Loading section classifier model: {section_classifier_model}")
        self.section_tokenizer = AutoTokenizer.from_pretrained(section_classifier_model)
        self.section_model = AutoModelForSequenceClassification.from_pretrained(
            section_classifier_model,
            num_labels=len(self.SECTION_TYPES)
        )
        self.section_model.to(self.device)
        self.section_model.eval()
        
        # Load SciSpacy
        try:
            logger.info(f"Loading SciSpacy model: {spacy_model}")
            self.nlp = spacy.load(spacy_model)
            
            # Add abbreviation detector
            self.nlp.add_pipe("abbreviation_detector")
            
            # Add UMLS entity linker if requested
            if use_umls:
                logger.info("Adding UMLS entity linker")
                self.nlp.add_pipe(
                    "scispacy_linker", 
                    config={"resolve_abbreviations": True, "linker_name": "umls"}
                )
                self.linker = self.nlp.get_pipe("scispacy_linker")
            else:
                self.linker = None
            
        except OSError as e:
            logger.error(f"Error loading SciSpacy model: {str(e)}")
            logger.info("Install required SciSpacy models with: pip install scispacy && pip install en_core_sci_md")
            raise
        
        # Section patterns and types
        self.SECTION_TYPES = [
            "title", "abstract", "introduction", "background", "methods", "materials_and_methods",
            "results", "discussion", "conclusion", "references", "acknowledgments", "other"
        ]
        
        # Initialize section heading patterns
        self.section_heading_patterns = [
            r'^(?:\d+[\.\)]\s*)?([A-Z][A-Za-z\s]+)(?:\s*[\.:])?\s*$',  # 1. Introduction: or Introduction.
            r'^(?:\d+[\.\)]\s*)?([A-Z][A-Za-z\s]+)$',  # 1. Introduction or Introduction
            r'^(?:[I|V|X]+[\.\)]\s*)([A-Z][A-Za-z\s]+)(?:\s*[\.:])?\s*$'  # I. Introduction: or II. Methods.
        ]
        self.section_patterns = [re.compile(pattern) for pattern in self.section_heading_patterns]
    
    def extract_pdf_text(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text
        """
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for page in doc:
                text += page.get_text() + "\n\n"
            
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            return ""
    
    def classify_section(self, heading: str, text: str) -> str:
        """
        Classify a section based on its heading and text using SciBERT.
        
        Args:
            heading: Section heading
            text: Section text
            
        Returns:
            Section type
        """
        # First, try simple rule-based classification by heading
        heading_lower = heading.lower()
        for section_type in self.SECTION_TYPES:
            if section_type in heading_lower:
                return section_type
        
        # If no match, use SciBERT
        # Prepare input by combining heading and start of text
        input_text = f"{heading} {text[:200]}"  # Use heading and beginning of section
        
        inputs = self.section_tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.section_model(**inputs)
            predictions = outputs.logits.argmax(dim=-1).item()
        
        return self.SECTION_TYPES[predictions]
    
    def detect_section_headings(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Detect section headings in text.
        
        Args:
            text: Document text
            
        Returns:
            List of tuples (heading, start position, end position)
        """
        headings = []
        lines = text.split('\n')
        
        current_pos = 0
        for line in lines:
            line_stripped = line.strip()
            
            # Skip empty lines
            if not line_stripped:
                current_pos += len(line) + 1  # +1 for newline
                continue
            
            # Check if line matches a section heading pattern
            for pattern in self.section_patterns:
                match = pattern.match(line_stripped)
                if match:
                    heading = match.group(1).strip()
                    # Check if it looks like a section heading (not too long)
                    if len(heading.split()) <= 10:
                        start_pos = current_pos
                        end_pos = current_pos + len(line)
                        headings.append((heading, start_pos, end_pos))
                        break
            
            current_pos += len(line) + 1  # +1 for newline
        
        return headings
    
    def extract_structured_sections(self, text: str) -> List[SectionInfo]:
        """
        Extract structured sections from document text.
        
        Args:
            text: Document text
            
        Returns:
            List of SectionInfo objects
        """
        # Detect section headings
        headings = self.detect_section_headings(text)
        
        # If no headings detected, treat the entire text as a single section
        if not headings:
            return [SectionInfo(
                section_type="body",
                heading="Document Body",
                text=text,
                start_pos=0,
                end_pos=len(text),
                subsections=[],
                entities=[],
                relations=[],
                metadata={}
            )]
        
        # Create sections from headings
        sections = []
        for i, (heading, start_pos, end_pos) in enumerate(headings):
            # Get section text (from the end of this heading to the start of the next heading)
            if i < len(headings) - 1:
                section_start = end_pos
                section_end = headings[i + 1][1]
                section_text = text[section_start:section_end].strip()
            else:
                section_start = end_pos
                section_end = len(text)
                section_text = text[section_start:section_end].strip()
            
            # Classify section type
            section_type = self.classify_section(heading, section_text)
            
            # Create SectionInfo object
            section_info = SectionInfo(
                section_type=section_type,
                heading=heading,
                text=section_text,
                start_pos=start_pos,
                end_pos=section_end,
                subsections=[],
                entities=[],
                relations=[],
                metadata={"character_count": len(section_text), "word_count": len(section_text.split())}
            )
            
            sections.append(section_info)
        
        # Identify subsections based on hierarchical numbering
        self.identify_subsections(sections)
        
        return sections
    
    def identify_subsections(self, sections: List[SectionInfo]) -> None:
        """
        Identify subsections within main sections.
        
        Args:
            sections: List of section info objects
            
        Modifies the section objects in place to add subsections.
        """
        # This is a simple implementation that looks for hierarchical numbering or nested headings
        for i, section in enumerate(sections):
            # Skip processing the last section
            if i >= len(sections) - 1:
                break
            
            current_heading = section.heading
            next_heading = sections[i + 1].heading
            
            # Check if the next heading appears to be a subsection of the current one
            current_num_match = re.match(r'^(\d+)[.\)]', current_heading)
            next_num_match = re.match(r'^(\d+)\.(\d+)[.\)]', next_heading)
            
            # If the next heading has a numbering pattern that looks like a subsection
            if (current_num_match and next_num_match and 
                current_num_match.group(1) == next_num_match.group(1)):
                
                # Move the next section to be a subsection of the current one
                subsection = sections.pop(i + 1)
                section.subsections.append(subsection)
                
                # Adjust the end position of the parent section
                section.end_pos = max(section.end_pos, subsection.end_pos)
    
    def extract_title(self, text: str) -> str:
        """Extract title from document text."""
        lines = text.strip().split('\n')
        
        # Skip empty lines at the beginning
        title_candidate_lines = []
        for line in lines[:10]:  # Look at first 10 lines
            line = line.strip()
            if line:
                # Check if line looks like a title (first line, not too long)
                if len(title_candidate_lines) == 0 and len(line) < 200:
                    title_candidate_lines.append(line)
                # Add consecutive lines that may be part of a multi-line title
                elif title_candidate_lines and len(line) < 200 and not line.endswith('.'):
                    title_candidate_lines.append(line)
                else:
                    break
        
        # Join candidate lines to form title
        if title_candidate_lines:
            return ' '.join(title_candidate_lines)
        
        # Fallback: Use the first non-empty line
        for line in lines:
            if line.strip():
                return line.strip()
        
        return "Unknown Title"
    
    def extract_abstract(self, text: str, sections: List[SectionInfo]) -> Optional[str]:
        """
        Extract the abstract from the document.
        
        Args:
            text: Document text
            sections: Document sections
            
        Returns:
            Abstract text if found, None otherwise
        """
        # First, check if there's an abstract section
        for section in sections:
            if section.section_type == "abstract":
                return section.text
        
        # If no abstract section, try to find it at the beginning of the document
        abstract_patterns = [
            r'(?i)abstract[.\s:]*\n(.*?)(?:\n\n|\n(?:[A-Z][a-z]+\s*\n)|$)',
            r'(?i)summary[.\s:]*\n(.*?)(?:\n\n|\n(?:[A-Z][a-z]+\s*\n)|$)'
        ]
        
        for pattern in abstract_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        return None
    
    def extract_references(self, text: str, sections: List[SectionInfo]) -> List[Dict[str, Any]]:
        """
        Extract references from the document.
        
        Args:
            text: Document text
            sections: Document sections
            
        Returns:
            List of references
        """
        # First, check if there's a references section
        references_text = None
        for section in sections:
            if section.section_type == "references":
                references_text = section.text
                break
        
        # If no references section found, try to find it using patterns
        if not references_text:
            references_patterns = [
                r'(?i)references\s*\n(.*?)(?:\n\n\n|$)',
                r'(?i)bibliography\s*\n(.*?)(?:\n\n\n|$)'
            ]
            
            for pattern in references_patterns:
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    references_text = match.group(1).strip()
                    break
        
        # If still no references found, return empty list
        if not references_text:
            return []
        
        # Parse references
        references = []
        
        # Common reference patterns
        reference_patterns = [
            # Pattern for numbered references: [1] Author et al. Title. Journal, Year.
            r'\[([\d]+)\]\s*(.*?)(?=\[[\d]+\]|\Z)',
            # Pattern for numbered references: 1. Author et al. Title. Journal, Year.
            r'(?:^|\n)([\d]+)\.?\s+(.*?)(?=\n[\d]+\.|\Z)',
            # Pattern for author-year references: (Author et al., Year)
            r'(?:^|\n)([A-Z][a-z]+(?:\s+et\s+al\.|\s+and\s+[A-Z][a-z]+)?)\s*\((\d{4})\)(.*?)(?=\n[A-Z]|\Z)'
        ]
        
        # Try each pattern until we find references
        for pattern in reference_patterns:
            matches = re.finditer(pattern, references_text, re.DOTALL | re.MULTILINE)
            extracted_refs = []
            
            for match in matches:
                if len(match.groups()) >= 2:
                    if match.groups()[0].isdigit():  # Numbered reference
                        ref_num = match.group(1)
                        ref_text = match.group(2).strip()
                        extracted_refs.append({
                            "id": ref_num,
                            "text": ref_text
                        })
                    else:  # Author-year reference
                        author = match.group(1)
                        year = match.group(2)
                        rest = match.group(3).strip() if len(match.groups()) > 2 else ""
                        ref_text = f"{author} ({year}) {rest}"
                        extracted_refs.append({
                            "id": f"{author}_{year}",
                            "text": ref_text
                        })
            
            # If we found references with this pattern, use them
            if extracted_refs:
                references = extracted_refs
                break
        
        return references
    
    def process_document(self, text_or_path: str, is_pdf: bool = False) -> DocumentStructure:
        """
        Process a biomedical document to extract structured information.
        
        Args:
            text_or_path: Document text or path to PDF file
            is_pdf: Whether the input is a path to a PDF file
            
        Returns:
            DocumentStructure object
        """
        logger.info("Processing document...")
        
        # Extract text from PDF if needed
        if is_pdf:
            text = self.extract_pdf_text(text_or_path)
        else:
            text = text_or_path
        
        # Extract structured sections
        sections = self.extract_structured_sections(text)
        
        # Extract title
        title = self.extract_title(text)
        
        # Extract abstract
        abstract = self.extract_abstract(text, sections)
        
        # Extract references
        references = self.extract_references(text, sections)
        
        # Create document structure
        doc_structure = DocumentStructure(
            title=title,
            abstract=abstract,
            sections=sections,
            references=references,
            entities=[],
            relations=[],
            metadata={
                "character_count": len(text),
                "word_count": len(text.split()),
                "section_count": len(sections),
                "reference_count": len(references)
            }
        )
        
        logger.info(f"Processed document: '{title}' with {len(sections)} sections")
        return doc_structure


#################################################
#         Biomedical NER with UMLS Linking     #
#################################################

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
        try:
            # Load SciSpacy model
            logger.info(f"Loading SciSpacy model: {spacy_model}")
            self.nlp = spacy.load(spacy_model)
            
            # Add abbreviation detector to the pipeline
            self.nlp.add_pipe("abbreviation_detector")
            
            # Add UMLS entity linker if requested
            if use_umls:
                logger.info("Adding UMLS entity linker")
                self.nlp.add_pipe(
                    "scispacy_linker", 
                    config={"resolve_abbreviations": True, "linker_name": "umls", "threshold": 0.8}
                )
                self.linker = self.nlp.get_pipe("scispacy_linker")
            else:
                self.linker = None
            
            logger.info("Biomedical entity extractor initialized successfully")
        except OSError as e:
            logger.error(f"Error loading SciSpacy model: {str(e)}")
            logger.info("Install required SciSpacy models with: pip install scispacy && pip install en_core_sci_md")
            raise
    
    def extract_entities(self, text: str) -> List[Entity]:
        """
        Extract biomedical entities from text with UMLS linking.
        
        Args:
            text: Input text
            
        Returns:
            List of Entity objects
        """
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
            if self.linker and ent._.kb_ents:
                # Get top UMLS entity
                umls_ent = ent._.kb_ents[0]
                entity.cui = umls_ent[0]
                entity.confidence = umls_ent[1]
                entity.umls_entity = self.linker.kb.cui_to_entity.get(umls_ent[0])
            
            # Check for abbreviations
            if ent._.is_abbreviated:
                entity.abbreviation = ent._.long_form
            elif ent._.is_long_form:
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


#################################################
#        Medical Relation Extraction with GNN   #
#################################################

class BioGPTRelationExtractor(nn.Module):
    """
    Relation extraction model based on BioGPT with GNN for graph-based reasoning.
    
    This model extracts biomedical relations between entities using a combination
    of BioGPT for contextualized representations and a GNN for relation modeling.
    """
    
    def __init__(
        self,
        encoder_model: str = "microsoft/biogpt",
        gnn_hidden_dim: int = 256,
        num_relations: int = 8,
        dropout_prob: float = 0.1,
        device: Optional[str] = None
    ):
        """
        Initialize the relation extraction model.
        
        Args:
            encoder_model: BioGPT encoder model
            gnn_hidden_dim: Hidden dimension for GNN
            num_relations: Number of relation types
            dropout_prob: Dropout probability
            device: Device for model
        """
        super(BioGPTRelationExtractor, self).__init__()
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = encoder_model
        self.num_relations = num_relations
        
        # Load pre-trained model and tokenizer
        logger.info(f"Loading BioGPT encoder: {encoder_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_model)
        self.encoder = AutoModel.from_pretrained(encoder_model)
        
        # Graph neural network components
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout_prob)
        
        # GNN layers for relation extraction
        self.gnn_conv1 = GATv2Conv(hidden_size, gnn_hidden_dim, heads=4, dropout=dropout_prob)
        self.gnn_conv2 = GATv2Conv(gnn_hidden_dim * 4, gnn_hidden_dim, heads=1, dropout=dropout_prob)
        
        # Relation classification heads
        self.classifier = nn.Sequential(
            nn.Linear(gnn_hidden_dim * 2, gnn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(gnn_hidden_dim, num_relations)
        )
        
        self.to(self.device)
        logger.info(f"Initialized BioGPT+GNN relation extractor with {num_relations} relation types on {self.device}")
    
    def encode_text(self, text: str) -> torch.Tensor:
        """
        Encode text using BioGPT encoder.
        
        Args:
            text: Input text
            
        Returns:
            Encoded text tensor
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            # Use CLS token embedding
            return outputs.last_hidden_state[:, 0, :]
    
    def prepare_entity_pair_input(
        self,
        text: str,
        entity1: Entity,
        entity2: Entity
    ) -> str:
        """
        Prepare input text for relation extraction between two entities.
        
        Args:
            text: Source text
            entity1: First entity
            entity2: Second entity
            
        Returns:
            Text with entity markers
        """
        # Sort entities by position
        if entity1.start < entity2.start:
            first, second = entity1, entity2
        else:
            first, second = entity2, entity1
        
        # Insert entity markers
        marked_text = (
            text[:first.start] + 
            f"[E1] {first.text} [/E1]" + 
            text[first.end:second.start] + 
            f"[E2] {second.text} [/E2]" + 
            text[second.end:]
        )
        
        # Truncate to manageable length while preserving entities
        if len(marked_text) > 512:
            # Find positions of markers
            e1_start = marked_text.find("[E1]")
            e2_end = marked_text.find("[/E2]") + len("[/E2]")
            
            # Ensure we keep the context around entities
            context_len = min(250, (512 - (e2_end - e1_start)) // 2)
            context_start = max(0, e1_start - context_len)
            context_end = min(len(marked_text), e2_end + context_len)
            
            marked_text = marked_text[context_start:context_end]
        
        return marked_text
    
    def forward(self, node_features, edge_index):
        """
        Forward pass through the GNN.
        
        Args:
            node_features: Node features [num_nodes, hidden_size]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Node representations after GNN layers
        """
        # Apply GNN layers
        x = self.gnn_conv1(node_features, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        
        x = self.gnn_conv2(x, edge_index)
        x = F.relu(x)
        
        return x
    
    def extract_relation(
        self,
        text: str,
        entity1: Entity,
        entity2: Entity,
        relation_types: List[str]
    ) -> Tuple[str, float]:
        """
        Extract relation between two entities.
        
        Args:
            text: Context text
            entity1: First entity
            entity2: Second entity
            relation_types: List of relation type names
            
        Returns:
            Tuple of (relation_type, confidence)
        """
        # Prepare input
        marked_text = self.prepare_entity_pair_input(text, entity1, entity2)
        
        # Encode with BioGPT
        encoding = self.encode_text(marked_text)
        
        # For a simple pair, we can just use the encoded representation
        # directly with the classifier, without needing the full GNN
        logits = self.classifier(encoding)
        probs = F.softmax(logits, dim=1).squeeze(0)
        
        # Get prediction
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()
        
        # Return relation type and confidence
        if pred_idx == 0:
            # Assuming index 0 is "no_relation"
            return "no_relation", confidence
        else:
            return relation_types[pred_idx], confidence


class MedicalRelationExtractor:
    """
    Medical relation extractor for biomedical text with UMLS-linked entities.
    """
    
    # Medical relation types
    RELATION_TYPES = [
        "no_relation",
        "treats",
        "causes",
        "diagnoses",
        "prevents",
        "complicates",
        "predisposes",
        "associated_with"
    ]
    
    # Valid entity type pairs for each relation
    VALID_RELATION_PAIRS = {
        "treats": [
            ("Drug", "Disease"),
            ("Chemical", "Disease"),
            ("Procedure", "Disease")
        ],
        "causes": [
            ("Drug", "Disease"),
            ("Chemical", "Disease"),
            ("Disease", "Disease"),
            ("Organism", "Disease")
        ],
        "diagnoses": [
            ("Procedure", "Disease")
        ],
        "prevents": [
            ("Drug", "Disease"),
            ("Chemical", "Disease"),
            ("Procedure", "Disease")
        ],
        "complicates": [
            ("Disease", "Disease"),
            ("Procedure", "Disease")
        ],
        "predisposes": [
            ("Disease", "Disease"),
            ("Gene/Protein", "Disease")
        ],
        "associated_with": [
            # Any combination is valid
        ]
    }
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        use_pretrained: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize the medical relation extractor.
        
        Args:
            model_path: Path to saved model weights
            use_pretrained: Whether to use pretrained weights
            device: Device for model
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self.model = BioGPTRelationExtractor(
            encoder_model="microsoft/biogpt",
            num_relations=len(self.RELATION_TYPES),
            device=self.device
        )
        
        # Load pretrained weights if specified
        if not use_pretrained and model_path:
            logger.info(f"Loading model weights from {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.eval()
        logger.info(f"Initialized Medical Relation Extractor with {len(self.RELATION_TYPES)} relation types")
    
    def is_valid_pair(self, entity1: Entity, entity2: Entity, relation: str) -> bool:
        """
        Check if the entity pair is valid for the given relation type.
        
        Args:
            entity1: First entity
            entity2: Second entity
            relation: Relation type
            
        Returns:
            Whether the pair is valid for the relation
        """
        # Associated_with is valid for any pair
        if relation == "associated_with":
            return True
        
        # Check if the pair types are in valid combinations
        valid_pairs = self.VALID_RELATION_PAIRS.get(relation, [])
        return (entity1.label, entity2.label) in valid_pairs or (entity2.label, entity1.label) in valid_pairs
    
    def extract_relations_from_text(
        self,
        text: str,
        entities: List[Entity],
        threshold: float = 0.5
    ) -> List[RelationInstance]:
        """
        Extract relations between entities in text.
        
        Args:
            text: Input text
            entities: List of entities
            threshold: Confidence threshold
            
        Returns:
            List of extracted relations
        """
        relations = []
        
        # Consider all entity pairs
        for entity1, entity2 in combinations(entities, 2):
            # Skip self-relations
            if entity1.start == entity2.start:
                continue
            
            # Extract relation
            relation_type, confidence = self.model.extract_relation(
                text, entity1, entity2, self.RELATION_TYPES
            )
            
            # Skip no_relation or low confidence
            if relation_type == "no_relation" or confidence < threshold:
                continue
            
            # Skip invalid entity type combinations
            if not self.is_valid_pair(entity1, entity2, relation_type):
                continue
            
            # Create context snippet
            start_idx = max(0, min(entity1.start, entity2.start) - 50)
            end_idx = min(len(text), max(entity1.end, entity2.end) + 50)
            context = text[start_idx:end_idx]
            
            # Create relation instance
            relation = RelationInstance(
                head_entity=entity1.text,
                tail_entity=entity2.text,
                head_type=entity1.label,
                tail_type=entity2.label,
                relation_type=relation_type,
                confidence=confidence,
                context=context,
                head_cui=entity1.cui,
                tail_cui=entity2.cui
            )
            
            relations.append(relation)
        
        return relations
    
    def process_document(self, doc_structure: DocumentStructure) -> DocumentStructure:
        """
        Process a document structure to extract relations.
        
        Args:
            doc_structure: Document structure with entities
            
        Returns:
            Updated document structure with relations
        """
        logger.info("Extracting biomedical relations...")
        all_relations = []
        
        # Process abstract
        if doc_structure.abstract and doc_structure.entities:
            # Filter entities from abstract
            abstract_entities = [
                e for e in doc_structure.entities 
                if e.start >= 0 and e.end <= len(doc_structure.abstract)
            ]
            
            if abstract_entities:
                abstract_relations = self.extract_relations_from_text(
                    doc_structure.abstract, abstract_entities
                )
                all_relations.extend(abstract_relations)
        
        # Process each section
        for section in doc_structure.sections:
            if section.entities:
                section_relations = self.extract_relations_from_text(
                    section.text, section.entities
                )
                section.relations = [rel.to_dict() for rel in section_relations]
                all_relations.extend(section_relations)
            
            # Process subsections
            for subsection in section.subsections:
                if subsection.entities:
                    subsection_relations = self.extract_relations_from_text(
                        subsection.text, subsection.entities
                    )
                    subsection.relations = [rel.to_dict() for rel in subsection_relations]
                    all_relations.extend(subsection_relations)
        
        # Add all relations to document
        doc_structure.relations = [rel.to_dict() for rel in all_relations]
        
        # Build knowledge graph
        doc_structure.knowledge_graph = self.build_knowledge_graph(all_relations)
        
        logger.info(f"Extracted {len(all_relations)} biomedical relations")
        return doc_structure
    
    def build_knowledge_graph(self, relations: List[RelationInstance]) -> nx.DiGraph:
        """
        Build a knowledge graph from extracted relations.
        
        Args:
            relations: List of relations
            
        Returns:
            NetworkX directed graph
        """
        G = nx.DiGraph()
        
        # Add nodes and edges
        for rel in relations:
            # Add nodes with entity types as attributes
            head_id = rel.head_cui if rel.head_cui else rel.head_entity
            tail_id = rel.tail_cui if rel.tail_cui else rel.tail_entity
            
            G.add_node(head_id, 
                      entity_text=rel.head_entity, 
                      entity_type=rel.head_type, 
                      cui=rel.head_cui)
            G.add_node(tail_id, 
                      entity_text=rel.tail_entity, 
                      entity_type=rel.tail_type,
                      cui=rel.tail_cui)
            
            # Add edge with relation type and confidence
            G.add_edge(
                head_id,
                tail_id,
                relation=rel.relation_type,
                confidence=rel.confidence,
                context=rel.context
            )
        
        return G


#################################################
#        SciFive Research Summarization         #
#################################################

class SciFiveResearchSummarizer:
    """
    Research paper summarization using SciFive-Large.
    
    SciFive is a T5 model specifically trained on biomedical and scientific literature,
    making it ideal for generating concise and informative summaries of research papers.
    """
    
    def __init__(
        self,
        model_name: str = "razent/SciFive-large-Pubmed-paper_summary",
        device: Optional[str] = None,
        max_length: int = 512,
        min_length: int = 50
    ):
        """
        Initialize the SciFive summarizer.
        
        Args:
            model_name: SciFive model name
            device: Device for model
            max_length: Maximum output length
            min_length: Minimum output length
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.min_length = min_length
        
        # Load model and tokenizer
        try:
            logger.info(f"Loading SciFive model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"SciFive model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Error loading SciFive model: {str(e)}")
            raise
    
    def summarize(
        self,
        text: str,
        prompt: str = "summarize: ",
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        num_beams: int = 4,
        do_sample: bool = False,
        temperature: float = 1.0,
        repetition_penalty: float = 1.2
    ) -> str:
        """
        Generate an abstractive summary.
        
        Args:
            text: Text to summarize
            prompt: Prefix prompt
            max_length: Maximum output length
            min_length: Minimum output length
            num_beams: Number of beams for beam search
            do_sample: Whether to use sampling
            temperature: Sampling temperature
            repetition_penalty: Penalty for repetition
            
        Returns:
            Generated summary
        """
        # Use class defaults if not specified
        max_length = max_length or self.max_length
        min_length = min_length or self.min_length
        
        # Prepare input
        input_text = prompt + text
        
        # Truncate if needed (T5 has a limit)
        if len(input_text) > 1024:
            input_text = input_text[:1024]
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(self.device)
        
        # Generate summary
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=num_beams,
                    do_sample=do_sample,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    early_stopping=True
                )
            
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return summary
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return ""
    
    def summarize_section(
        self,
        section_text: str,
        section_type: str
    ) -> str:
        """
        Summarize a specific section based on its type.
        
        Args:
            section_text: Section text
            section_type: Section type
            
        Returns:
            Section summary
        """
        # Use different prompts based on section type
        if section_type in ["methods", "materials_and_methods"]:
            prompt = "summarize methods: "
            max_length = 200
        elif section_type == "results":
            prompt = "summarize key findings: "
            max_length = 250
        elif section_type in ["discussion", "conclusion"]:
            prompt = "summarize conclusions: "
            max_length = 200
        elif section_type == "abstract":
            prompt = "summarize concisely: "
            max_length = 150
        else:
            prompt = "summarize: "
            max_length = 200
        
        return self.summarize(
            text=section_text,
            prompt=prompt,
            max_length=max_length
        )
    
    def extract_clinical_implications(
        self,
        text: str,
        max_length: int = 200
    ) -> str:
        """
        Extract clinical implications from research text.
        
        Args:
            text: Text to analyze
            max_length: Maximum output length
            
        Returns:
            Clinical implications summary
        """
        prompt = "extract clinical implications: "
        return self.summarize(text=text, prompt=prompt, max_length=max_length)
    
    def process_document(self, doc_structure: DocumentStructure) -> DocumentStructure:
        """
        Generate summaries for a document structure.
        
        Args:
            doc_structure: Document structure
            
        Returns:
            Updated document structure with summaries
        """
        logger.info("Generating research summaries...")
        summaries = {}
        
        # Generate abstract summary
        if doc_structure.abstract:
            summaries["abstract"] = self.summarize_section(
                doc_structure.abstract, "abstract"
            )
        
        # Generate methods summary
        methods_sections = [s for s in doc_structure.sections 
                           if s.section_type in ["methods", "materials_and_methods"]]
        if methods_sections:
            methods_text = "\n".join([s.text for s in methods_sections])
            summaries["methods"] = self.summarize_section(
                methods_text, "methods"
            )
        
        # Generate findings summary (from results)
        results_sections = [s for s in doc_structure.sections 
                           if s.section_type == "results"]
        if results_sections:
            results_text = "\n".join([s.text for s in results_sections])
            summaries["key_findings"] = self.summarize_section(
                results_text, "results"
            )
        
        # Generate conclusions summary
        conclusion_sections = [s for s in doc_structure.sections 
                              if s.section_type in ["conclusion", "discussion"]]
        if conclusion_sections:
            conclusion_text = "\n".join([s.text for s in conclusion_sections])
            summaries["conclusions"] = self.summarize_section(
                conclusion_text, "conclusion"
            )
        
        # Generate clinical implications (if discussion exists)
        discussion_sections = [s for s in doc_structure.sections 
                              if s.section_type == "discussion"]
        if discussion_sections:
            discussion_text = "\n".join([s.text for s in discussion_sections])
            summaries["clinical_implications"] = self.extract_clinical_implications(
                discussion_text
            )
        
        # Extract limitations (looking for specific text)
        for section in doc_structure.sections:
            if section.section_type in ["discussion", "limitations"]:
                # Look for limitations paragraph
                limitations_pattern = r'(?:limitation|limitations)[^.]+\.'
                limitations = re.findall(limitations_pattern, section.text, re.IGNORECASE)
                if limitations:
                    limitations_text = " ".join(limitations)
                    summaries["limitations"] = limitations_text
                    break
        
        # Create research summary object
        research_summary = ResearchSummary(
            abstract=summaries.get("abstract", ""),
            key_findings=summaries.get("key_findings", ""),
            methods=summaries.get("methods", ""),
            conclusions=summaries.get("conclusions", ""),
            limitations=summaries.get("limitations", ""),
            clinical_implications=summaries.get("clinical_implications", "")
        )
        
        # Add to document structure
        doc_structure.summary = research_summary.to_dict()
        
        logger.info("Research summarization complete")
        return doc_structure


#################################################
#             Main Pipeline Class               #
#################################################

class MedicalResearchSynthesizer:
    """
    Complete pipeline for processing and synthesizing medical research papers.
    
    Integrates document processing, entity extraction, relation extraction, and
    research summarization components into a cohesive workflow.
    """
    
    def __init__(
        self,
        document_processor_args: Dict = None,
        entity_extractor_args: Dict = None,
        relation_extractor_args: Dict = None,
        summarizer_args: Dict = None,
        device: Optional[str] = None
    ):
        """
        Initialize the medical research synthesizer.
        
        Args:
            document_processor_args: Arguments for document processor
            entity_extractor_args: Arguments for entity extractor
            relation_extractor_args: Arguments for relation extractor
            summarizer_args: Arguments for summarizer
            device: Device for PyTorch models
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing Medical Research Synthesizer on {self.device}")
        
        # Initialize document processor
        doc_args = document_processor_args or {}
        self.document_processor = BiomedicalDocumentProcessor(
            device=self.device, **doc_args
        )
        
        # Initialize entity extractor
        entity_args = entity_extractor_args or {}
        self.entity_extractor = BiomedicalEntityExtractor(**entity_args)
        
        # Initialize relation extractor
        rel_args = relation_extractor_args or {}
        self.relation_extractor = MedicalRelationExtractor(
            device=self.device, **rel_args
        )
        
        # Initialize summarizer
        sum_args = summarizer_args or {}
        self.summarizer = SciFiveResearchSummarizer(
            device=self.device, **sum_args
        )
        
        logger.info("Medical Research Synthesizer initialized successfully")
    
    def process(self, text_or_path: str, is_pdf: bool = False) -> DocumentStructure:
        """
        Process a medical research paper end-to-end.
        
        Args:
            text_or_path: Text or path to PDF
            is_pdf: Whether the input is a PDF path
            
        Returns:
            Processed document structure
        """
        # Step 1: Document processing
        logger.info("Step 1: Document processing")
        doc_structure = self.document_processor.process_document(text_or_path, is_pdf)
        
        # Step 2: Entity extraction
        logger.info("Step 2: Biomedical entity extraction")
        doc_structure = self.entity_extractor.process_document(doc_structure)
        
        # Step 3: Relation extraction
        logger.info("Step 3: Relation extraction")
        doc_structure = self.relation_extractor.process_document(doc_structure)
        
        # Step 4: Research summarization
        logger.info("Step 4: Research summarization")
        doc_structure = self.summarizer.process_document(doc_structure)
        
        logger.info("Processing complete")
        return doc_structure
    
    def save_results(self, doc_structure: DocumentStructure, output_dir: str) -> None:
        """
        Save processing results to files.
        
        Args:
            doc_structure: Processed document structure
            output_dir: Output directory
        """
        import os
        import json
        import pandas as pd
        import networkx as nx
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save document structure as JSON
        with open(os.path.join(output_dir, "document_structure.json"), "w", encoding="utf-8") as f:
            # Convert to serializable format
            doc_dict = {
                "title": doc_structure.title,
                "abstract": doc_structure.abstract,
                "sections": [
                    {
                        "section_type": section.section_type,
                        "heading": section.heading,
                        "text": section.text[:500] + "..." if len(section.text) > 500 else section.text,
                        "entities_count": len(section.entities),
                        "relations_count": len(section.relations),
                        "subsections": [
                            {"section_type": ss.section_type, "heading": ss.heading}
                            for ss in section.subsections
                        ]
                    }
                    for section in doc_structure.sections
                ],
                "entities_count": len(doc_structure.entities),
                "relations_count": len(doc_structure.relations),
                "summary": doc_structure.summary
            }
            json.dump(doc_dict, f, indent=2)
        
        # Save entities as CSV
        entities_df = pd.DataFrame([e.to_dict() for e in doc_structure.entities])
        if not entities_df.empty:
            entities_df.to_csv(os.path.join(output_dir, "entities.csv"), index=False)
        
        # Save relations as CSV
        relations_df = pd.DataFrame(doc_structure.relations)
        if not relations_df.empty:
            relations_df.to_csv(os.path.join(output_dir, "relations.csv"), index=False)
        
        # Save summary as markdown
        if doc_structure.summary:
            research_summary = ResearchSummary(**doc_structure.summary)
            with open(os.path.join(output_dir, "summary.md"), "w", encoding="utf-8") as f:
                f.write(research_summary.to_markdown())
        
        # Export knowledge graph (if NetworkX is available)
        if doc_structure.knowledge_graph and isinstance(doc_structure.knowledge_graph, nx.Graph):
            # Export as GraphML
            nx.write_graphml(doc_structure.knowledge_graph, 
                            os.path.join(output_dir, "knowledge_graph.graphml"))
            
            # Generate a simple visualization
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(12, 10))
                pos = nx.spring_layout(doc_structure.knowledge_graph, seed=42)
                
                # Draw nodes
                nx.draw_networkx_nodes(doc_structure.knowledge_graph, pos, 
                                      node_size=300, alpha=0.8)
                
                # Draw edges
                nx.draw_networkx_edges(doc_structure.knowledge_graph, pos, 
                                      width=1.5, alpha=0.7)
                
                # Draw labels
                nx.draw_networkx_labels(doc_structure.knowledge_graph, pos, 
                                       font_size=10)
                
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "knowledge_graph.png"), dpi=300)
                plt.close()
            except Exception as e:
                logger.warning(f"Could not create graph visualization: {str(e)}")
        
        logger.info(f"Results saved to {output_dir}")


# Example usage
if __name__ == "__main__":
    # Initialize the synthesizer
    synthesizer = MedicalResearchSynthesizer()
    
    # Process a sample document
    sample_text = """
    Title: Effects of Exercise on Cardiovascular Health in Type 2 Diabetes
    
    Abstract
    Regular exercise has been shown to improve cardiovascular health, but its effects in patients with type 2 diabetes remain understudied. We conducted a randomized controlled trial involving 150 patients with type 2 diabetes who were assigned to either an exercise intervention (n=75) or standard care (n=75) for 12 weeks. The exercise group showed significant improvements in HbA1c levels (p<0.01), blood pressure (p<0.05), and lipid profiles (p<0.05) compared to the control group. Our findings demonstrate that regular exercise significantly improves cardiovascular health markers in patients with type 2 diabetes.
    
    Introduction
    Type 2 diabetes mellitus (T2DM) is a chronic metabolic disorder characterized by insulin resistance and hyperglycemia. Cardiovascular disease is the leading cause of morbidity and mortality in patients with T2DM, accounting for up to 80% of deaths in this population. Exercise has been recommended as a non-pharmacological intervention for improving glycemic control and cardiovascular health in patients with T2DM, but the mechanisms and optimal prescription remain unclear.
    
    Methods
    Study Design: This was a 12-week, single-center, randomized controlled trial. Participants were randomly assigned to either an exercise intervention group or a standard care control group.
    
    Participants: A total of 150 patients with T2DM were recruited from the diabetes clinic. Inclusion criteria were: age 40-70 years, diagnosed with T2DM for at least 1 year, and HbA1c between 7.0% and 10.0%. Exclusion criteria included uncontrolled hypertension, recent cardiovascular events, and physical limitations preventing exercise.
    
    Intervention: The exercise group participated in a structured program consisting of 150 minutes per week of moderate-intensity aerobic exercise (walking, cycling, or swimming) and resistance training twice weekly. The control group received standard care including dietary advice and medication management.
    
    Measurements: Assessments were performed at baseline and after 12 weeks. Primary outcomes included changes in HbA1c, blood pressure, and lipid profile. Secondary outcomes included weight, waist circumference, and quality of life.
    
    Results
    Baseline Characteristics: The exercise and control groups were well-matched at baseline with no significant differences in demographic or clinical characteristics.
    
    Primary Outcomes: After 12 weeks, the exercise group showed significant improvements compared to the control group:
    - HbA1c decreased by 0.8% in the exercise group versus 0.1% in the control group (p<0.01)
    - Systolic blood pressure decreased by 8.5 mmHg in the exercise group versus 1.2 mmHg in the control group (p<0.05)
    - LDL cholesterol decreased by 15.2 mg/dL in the exercise group versus 2.3 mg/dL in the control group (p<0.05)
    
    Secondary Outcomes: The exercise group also showed significant reductions in body weight (-3.2 kg vs -0.8 kg, p<0.05) and waist circumference (-4.1 cm vs -0.9 cm, p<0.05). Quality of life scores improved significantly in the exercise group (p<0.01).
    
    Discussion
    Our findings demonstrate that a 12-week structured exercise program leads to significant improvements in glycemic control and cardiovascular risk factors in patients with type 2 diabetes. The magnitude of HbA1c reduction in the exercise group (-0.8%) is clinically significant and comparable to what might be achieved with additional medication.
    
    The improvements in lipid profile and blood pressure suggest that regular exercise may reduce cardiovascular risk through multiple mechanisms. Previous studies have shown that exercise improves insulin sensitivity, endothelial function, and reduces systemic inflammation, all of which contribute to better cardiovascular health.
    
    Limitations of this study include the relatively short intervention period and the single-center design. Additionally, we did not assess the long-term adherence to exercise after the intervention period. Future studies should investigate the optimal exercise prescription and strategies to improve long-term adherence.
    
    Conclusion
    Regular exercise significantly improves glycemic control and cardiovascular risk factors in patients with type 2 diabetes. These findings support the recommendation for structured exercise programs as an essential component of diabetes management. Healthcare providers should encourage patients with type 2 diabetes to engage in regular physical activity and consider supervised exercise programs for those at high cardiovascular risk.
    
    References
    1. American Diabetes Association. Standards of Medical Care in Diabetes-2022. Diabetes Care. 2022;45(Suppl 1):S1-S264.
    2. Colberg SR, Sigal RJ, Yardley JE, et al. Physical Activity/Exercise and Diabetes: A Position Statement of the American Diabetes Association. Diabetes Care. 2016;39(11):2065-2079.
    3. Umpierre D, Ribeiro PA, Kramer CK, et al. Physical activity advice only or structured exercise training and association with HbA1c levels in type 2 diabetes: a systematic review and meta-analysis. JAMA. 2011;305(17):1790-1799.
    """
    
    result = synthesizer.process(sample_text)
    
    # Save results
    synthesizer.save_results(result, "output")
    
    # Print summary
    if result.summary:
        print("\nRESEARCH SUMMARY:")
        research_summary = ResearchSummary(**result.summary)
        print(research_summary.to_markdown())
    
    # Print entity and relation counts
    print(f"\nExtracted {len(result.entities)} entities and {len(result.relations)} relations")








import os
import torch
from medical_research_synthesizer import MedicalResearchSynthesizer

"""
Medical Research Synthesizer Usage Examples
------------------------------------------
This script demonstrates how to use the Medical Research Synthesizer 
for different use cases including processing PDFs, text, and 
extracting specific information from scientific papers.
"""

#--------------------------------------
# 1. Basic Setup and Configuration
#--------------------------------------

# Initialize with default settings (using all modules)
synthesizer = MedicalResearchSynthesizer()

# Initialize with custom configuration for specific components
custom_synthesizer = MedicalResearchSynthesizer(
    document_processor_args={
        "spacy_model": "en_core_sci_lg",  # Use larger SciSpacy model
        "use_umls": True  # Enable UMLS entity linking
    },
    entity_extractor_args={
        "use_umls": True  # Enable UMLS linking
    },
    relation_extractor_args={
        "use_pretrained": True  # Use pretrained weights
    },
    summarizer_args={
        "model_name": "razent/SciFive-large-Pubmed-paper_summary",
        "max_length": 512
    },
    device="cuda:0"  # Use specific GPU if available
)

#--------------------------------------
# 2. Processing Scientific Papers
#--------------------------------------

# Process a paper from text
def process_paper_text():
    with open("sample_paper.txt", "r") as f:
        paper_text = f.read()
    
    # Process the paper
    result = synthesizer.process(paper_text)
    
    # Save results to output directory
    synthesizer.save_results(result, "output/sample_paper")
    
    # Print summary
    print("\nPaper Summary:")
    if result.summary:
        print(f"Abstract: {result.summary['abstract']}")
        print(f"Key Findings: {result.summary['key_findings']}")
    
    return result

# Process a paper from PDF
def process_paper_pdf(pdf_path="papers/research_paper.pdf"):
    # Check if file exists
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file {pdf_path} not found")
        return None
    
    # Process the PDF
    result = synthesizer.process(pdf_path, is_pdf=True)
    
    # Save results to output directory
    output_dir = os.path.join("output", os.path.basename(pdf_path).replace(".pdf", ""))
    synthesizer.save_results(result, output_dir)
    
    return result

#--------------------------------------
# 3. Batch Processing Multiple Papers
#--------------------------------------

def batch_process_papers(pdf_dir="papers/"):
    results = []
    
    # Get all PDF files in directory
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return []
    
    print(f"Processing {len(pdf_files)} papers...")
    
    # Process each PDF
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        try:
            result = synthesizer.process(pdf_path, is_pdf=True)
            results.append(result)
            
            # Save individual results
            output_dir = os.path.join("output", pdf_file.replace(".pdf", ""))
            synthesizer.save_results(result, output_dir)
            
        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")
    
    return results

#--------------------------------------
# 4. Extracting Specific Information
#--------------------------------------

def extract_methods_summary(paper_text):
    """Extract and summarize research methods from a paper"""
    # Process paper
    result = synthesizer.process(paper_text)
    
    # Get methods sections
    methods_sections = [s for s in result.sections 
                       if s.section_type in ["methods", "materials_and_methods"]]
    
    if not methods_sections:
        return "No methods section found"
    
    # Generate methods summary
    methods_text = "\n".join([s.text for s in methods_sections])
    methods_summary = synthesizer.summarizer.summarize_section(methods_text, "methods")
    
    return methods_summary

def extract_clinical_implications(paper_text):
    """Extract clinical implications from a paper"""
    # Process paper
    result = synthesizer.process(paper_text)
    
    # Get discussion sections
    discussion_sections = [s for s in result.sections if s.section_type == "discussion"]
    
    if not discussion_sections:
        return "No discussion section found"
    
    # Generate clinical implications
    discussion_text = "\n".join([s.text for s in discussion_sections])
    clinical_implications = synthesizer.summarizer.extract_clinical_implications(
        discussion_text
    )
    
    return clinical_implications

def extract_disease_drug_relations(paper_text):
    """Extract disease-drug relations from a paper"""
    # Process paper
    result = synthesizer.process(paper_text)
    
    # Filter relations for disease-drug relationships
    disease_drug_relations = []
    for relation in result.relations:
        if relation["relation_type"] == "treats" and (
            (relation["head_type"] == "Drug" and relation["tail_type"] == "Disease") or
            (relation["head_type"] == "Disease" and relation["tail_type"] == "Drug")
        ):
            disease_drug_relations.append(relation)
    
    return disease_drug_relations

#--------------------------------------
# 5. Knowledge Graph Operations
#--------------------------------------

def explore_knowledge_graph(paper_text):
    """Demonstrate working with the knowledge graph"""
    # Process paper
    result = synthesizer.process(paper_text)
    
    if not result.knowledge_graph:
        return "No knowledge graph available"
    
    # Get network statistics
    import networkx as nx
    G = result.knowledge_graph
    
    stats = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "connected_components": nx.number_connected_components(G.to_undirected()),
        "degree_centrality": nx.degree_centrality(G),
        "betweenness_centrality": nx.betweenness_centrality(G)
    }
    
    # Find important entities (nodes with highest centrality)
    important_entities = sorted(
        stats["degree_centrality"].items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:5]
    
    # Find important relations (edges with highest confidence)
    important_relations = []
    for u, v, data in G.edges(data=True):
        important_relations.append((u, v, data["relation"], data["confidence"]))
    
    important_relations = sorted(important_relations, key=lambda x: x[3], reverse=True)[:5]
    
    return {
        "stats": stats,
        "important_entities": important_entities,
        "important_relations": important_relations
    }

#--------------------------------------
# 6. Comparison of Multiple Papers
#--------------------------------------

def compare_papers(paper_texts):
    """Compare multiple research papers"""
    results = []
    summaries = []
    
    # Process each paper
    for text in paper_texts:
        result = synthesizer.process(text)
        results.append(result)
        if result.summary:
            summaries.append(result.summary)
    
    if not summaries:
        return "No summaries available for comparison"
    
    # Compare methods
    methods_comparison = "Methods Comparison:\n"
    for i, summary in enumerate(summaries):
        if "methods" in summary:
            methods_comparison += f"Paper {i+1}: {summary['methods']}\n\n"
    
    # Compare findings
    findings_comparison = "Key Findings Comparison:\n"
    for i, summary in enumerate(summaries):
        if "key_findings" in summary:
            findings_comparison += f"Paper {i+1}: {summary['key_findings']}\n\n"
    
    # Compare conclusions
    conclusions_comparison = "Conclusions Comparison:\n"
    for i, summary in enumerate(summaries):
        if "conclusions" in summary:
            conclusions_comparison += f"Paper {i+1}: {summary['conclusions']}\n\n"
    
    return {
        "methods_comparison": methods_comparison,
        "findings_comparison": findings_comparison,
        "conclusions_comparison": conclusions_comparison
    }

#--------------------------------------
# Main Execution
#--------------------------------------

if __name__ == "__main__":
    # Check for CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Choose which examples to run
    sample_text = """
    Title: Effects of Exercise on Cardiovascular Health in Type 2 Diabetes
    
    Abstract
    Regular exercise has been shown to improve cardiovascular health, but its effects in patients with type 2 diabetes remain understudied. We conducted a randomized controlled trial involving 150 patients with type 2 diabetes who were assigned to either an exercise intervention (n=75) or standard care (n=75) for 12 weeks. The exercise group showed significant improvements in HbA1c levels (p<0.01), blood pressure (p<0.05), and lipid profiles (p<0.05) compared to the control group. Our findings demonstrate that regular exercise significantly improves cardiovascular health markers in patients with type 2 diabetes.
    
    [... rest of the paper text ...]
    """
    
    # Run a single example
    result = synthesizer.process(sample_text)
    synthesizer.save_results(result, "output/example")
    
    # Print entity and relation counts
    print(f"Extracted {len(result.entities)} entities and {len(result.relations)} relations")
    
    # Print summary if available
    if result.summary:
        from medical_research_synthesizer import ResearchSummary
        summary = ResearchSummary(**result.summary)
        print("\nRESEARCH SUMMARY:")
        print(summary.to_markdown())
    
    print("\nProcessing complete!")

from medical_research_synthesizer import MedicalResearchSynthesizer

# Initialize the system
synthesizer = MedicalResearchSynthesizer()

# Process a medical research paper (PDF or text)
result = synthesizer.process("paper.pdf", is_pdf=True)

# Access extracted information
print(f"Title: {result.title}")
print(f"Found {len(result.entities)} biomedical entities")
print(f"Extracted {len(result.relations)} relations")

# Get research summary
if result.summary:
    print(f"Abstract summary: {result.summary['abstract']}")
    print(f"Key findings: {result.summary['key_findings']}")
    print(f"Clinical implications: {result.summary['clinical_implications']}")

# Save all results
synthesizer.save_results(result, "output_folder")