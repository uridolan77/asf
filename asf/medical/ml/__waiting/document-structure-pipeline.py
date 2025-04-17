import os
import re
import json
import logging
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    pipeline
)
from peft import PeftModel
import spacy
from spacy.tokens import Doc, Span
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SectionInfo:
    """Information about a section in a scientific document."""
    section_type: str
    heading: str
    text: str
    start_pos: int
    end_pos: int
    subsections: List['SectionInfo'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DocumentStructure:
    """Structured representation of a scientific document."""
    title: str
    abstract: Optional[str] = None
    sections: List[SectionInfo] = field(default_factory=list)
    references: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class MedicalDocumentProcessor:
    """
    Pipeline for processing medical documents and extracting structured information.
    
    This class provides functionality for:
    1. Section detection and classification
    2. Information extraction from sections
    3. Reference parsing
    4. Document structuring
    """
    
    def __init__(
        self,
        section_classifier_model: str = "allenai/scibert_scivocab_uncased",
        embedding_model: str = "pritamdeka/S-PubMedBert-MS-MARCO",
        use_peft: bool = False,
        peft_model_path: Optional[str] = None,
        spacy_model: str = "en_core_sci_md",
        device: Optional[str] = None
    ):
        """
        Initialize the medical document processor.
        
        Args:
            section_classifier_model: Model for section classification
            embedding_model: Model for sentence embeddings
            use_peft: Whether to use a PEFT adapter
            peft_model_path: Path to PEFT adapter
            spacy_model: SpaCy model for NLP processing
            device: Device to use for models
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load section classifier
        self._load_section_classifier(section_classifier_model, use_peft, peft_model_path)
        
        # Load embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model, device=self.device)
        
        # Load SpaCy model
        try:
            logger.info(f"Loading SpaCy model: {spacy_model}")
            self.nlp = spacy.load(spacy_model)
            # Add custom components to the NLP pipeline if needed
            # self._add_custom_components()
        except OSError:
            logger.warning(f"SpaCy model {spacy_model} not found. Please download it using: python -m spacy download {spacy_model}")
            raise
        
        # Common section types in medical papers
        self.section_types = [
            "abstract", "introduction", "background", "methods", "materials and methods",
            "results", "discussion", "conclusion", "references", "acknowledgments",
            "figures", "tables", "supplementary", "appendix", "conflict of interest"
        ]
        
        # Regular expressions for section heading detection
        self.section_heading_patterns = [
            r'^(?:\d+[\.\)]\s*)?([A-Z][A-Za-z\s]+)(?:\s*[\.:])?\s*$',  # 1. Introduction: or Introduction.
            r'^(?:\d+[\.\)]\s*)?([A-Z][A-Za-z\s]+)$',  # 1. Introduction or Introduction
            r'^(?:[I|V|X]+[\.\)]\s*)([A-Z][A-Za-z\s]+)(?:\s*[\.:])?\s*$'  # I. Introduction: or II. Methods.
        ]
        self.section_patterns = [re.compile(pattern) for pattern in self.section_heading_patterns]
    
    def _load_section_classifier(self, model_name: str, use_peft: bool, peft_model_path: Optional[str]):
        """Load the section classifier model."""
        try:
            logger.info(f"Loading section classifier model: {model_name}")
            self.section_tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            if use_peft and peft_model_path:
                # Load the base model
                base_model = AutoModelForSequenceClassification.from_pretrained(model_name)
                # Load the PEFT adapter
                self.section_model = PeftModel.from_pretrained(base_model, peft_model_path)
                logger.info(f"Loaded PEFT adapter from {peft_model_path}")
            else:
                # Just load the regular model
                self.section_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            self.section_model.to(self.device)
            self.section_model.eval()
        except Exception as e:
            logger.error(f"Error loading section classifier: {str(e)}")
            raise
    
    def _detect_section_headings(self, text: str) -> List[Tuple[str, int, int]]:
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
            is_heading = False
            for pattern in self.section_patterns:
                match = pattern.match(line_stripped)
                if match:
                    heading = match.group(1).strip()
                    # Check if heading is a common section type
                    if heading.lower() in self.section_types or any(
                        section_type in heading.lower() for section_type in self.section_types
                    ):
                        start_pos = current_pos
                        end_pos = current_pos + len(line)
                        headings.append((heading, start_pos, end_pos))
                        is_heading = True
                        break
            
            current_pos += len(line) + 1  # +1 for newline
        
        return headings
    
    def _classify_section(self, heading: str, text: str) -> str:
        """
        Classify a section based on its heading and content.
        
        Args:
            heading: Section heading
            text: Section text
            
        Returns:
            Section type
        """
        # First, try to determine the section type by its heading
        heading_lower = heading.lower()
        
        # Check for exact matches in section types
        for section_type in self.section_types:
            if heading_lower == section_type or heading_lower.startswith(section_type):
                return section_type
        
        # If no match, use the embedding model to find the most similar section type
        heading_embedding = self.embedding_model.encode(heading_lower, convert_to_tensor=True)
        section_embeddings = self.embedding_model.encode(self.section_types, convert_to_tensor=True)
        
        # Calculate similarities
        similarities = cosine_similarity(
            heading_embedding.cpu().numpy().reshape(1, -1),
            section_embeddings.cpu().numpy()
        )[0]
        
        # Get most similar section type
        most_similar_idx = np.argmax(similarities)
        if similarities[most_similar_idx] > 0.5:  # Threshold for similarity
            return self.section_types[most_similar_idx]
        
        # If still no good match, use "other"
        return "other"
    
    def _extract_structured_sections(self, text: str) -> List[SectionInfo]:
        """
        Extract structured sections from document text.
        
        Args:
            text: Document text
            
        Returns:
            List of SectionInfo objects
        """
        # Detect section headings
        headings = self._detect_section_headings(text)
        
        # If no headings detected, treat the entire text as a single section
        if not headings:
            return [SectionInfo(
                section_type="body",
                heading="Document Body",
                text=text,
                start_pos=0,
                end_pos=len(text),
                subsections=[],
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
            section_type = self._classify_section(heading, section_text)
            
            # Create SectionInfo object
            section_info = SectionInfo(
                section_type=section_type,
                heading=heading,
                text=section_text,
                start_pos=start_pos,
                end_pos=section_end,
                subsections=[],
                metadata={"character_count": len(section_text), "word_count": len(section_text.split())}
            )
            
            sections.append(section_info)
        
        # Detect hierarchy - identify subsections
        self._identify_subsections(sections)
        
        return sections
    
    def _identify_subsections(self, sections: List[SectionInfo]) -> None:
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
    
    def _extract_abstract(self, text: str, sections: List[SectionInfo]) -> Optional[str]:
        """
        Extract the abstract from the document text or sections.
        
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
    
    def _extract_title(self, text: str) -> str:
        """
        Extract the title from document text.
        
        Args:
            text: Document text
            
        Returns:
            Document title
        """
        # Try to extract title from beginning of document
        lines = text.strip().split('\n')
        
        # Skip empty lines at the beginning
        title_candidate_lines = []
        