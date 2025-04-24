"""
Biomedical Document Processor

This module provides functionality for processing medical research papers,
extracting structured sections, and organizing document components.
"""

import os
import re
import logging
import torch
from typing import List, Dict, Tuple, Optional, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Try to import fitz (PyMuPDF) for PDF processing
try:
    import fitz
except ImportError:
    fitz = None

# Local imports
from .document_structure import DocumentStructure, SectionInfo

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BiomedicalDocumentProcessor:
    """
    Medical document processor for scientific papers.
    
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
        
        # Define section types
        self.SECTION_TYPES = [
            "title", "abstract", "introduction", "background", "methods", "materials_and_methods",
            "results", "discussion", "conclusion", "references", "acknowledgments", "other"
        ]
        
        # Try to load section classifier (SciBERT)
        try:
            logger.info(f"Loading section classifier model: {section_classifier_model}")
            self.section_tokenizer = AutoTokenizer.from_pretrained(section_classifier_model)
            self.section_model = AutoModelForSequenceClassification.from_pretrained(
                section_classifier_model,
                num_labels=len(self.SECTION_TYPES)
            )
            self.section_model.to(self.device)
            self.section_model.eval()
        except Exception as e:
            logger.warning(f"Could not load section classifier model: {e}")
            self.section_tokenizer = None
            self.section_model = None
        
        # Try to load SciSpacy
        self.nlp = None
        try:
            import spacy
            logger.info(f"Loading SciSpacy model: {spacy_model}")
            self.nlp = spacy.load(spacy_model)
            
            # Add abbreviation detector if available
            try:
                from scispacy.abbreviation import AbbreviationDetector
                self.nlp.add_pipe("abbreviation_detector")
                logger.info("Added abbreviation detector to pipeline")
            except (ImportError, ModuleNotFoundError):
                logger.warning("Could not load AbbreviationDetector from scispacy")
            
            # Add UMLS entity linker if requested
            if use_umls:
                try:
                    from scispacy.linking import EntityLinker
                    logger.info("Adding UMLS entity linker")
                    self.nlp.add_pipe(
                        "scispacy_linker", 
                        config={"resolve_abbreviations": True, "linker_name": "umls", "threshold": 0.8}
                    )
                except (ImportError, ModuleNotFoundError):
                    logger.warning("Could not load EntityLinker from scispacy")
        except Exception as e:
            logger.warning(f"Could not load SciSpacy model: {e}")
            logger.info("Install required SciSpacy models with: pip install scispacy && pip install en_core_sci_md")
        
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
        if fitz is None:
            logger.error("PyMuPDF (fitz) is not installed. Cannot extract text from PDF.")
            logger.info("Install with: pip install pymupdf")
            return ""
        
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for page in doc:
                text += page.get_text()
            
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            return ""
    
    def classify_section(self, heading: str, text: str) -> str:
        """
        Classify a section based on its heading and text.
        
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
        
        # If no match and the model is available, use SciBERT
        if self.section_model and self.section_tokenizer:
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
        
        # Fallback to "other" if model isn't available
        return "other"
    
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
                    headings.append((heading, current_pos, current_pos + len(line)))
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
                section_text = text[end_pos:headings[i+1][1]].strip()
                section_end = headings[i+1][1]
            else:
                section_text = text[end_pos:].strip()
                section_end = len(text)
            
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
                continue
            
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
                    references_text = match.group(1)
        
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
                    if pattern == reference_patterns[2]:  # Author-year pattern
                        ref = {
                            "author": match.group(1),
                            "year": match.group(2),
                            "text": match.group(0)
                        }
                    else:  # Numbered reference
                        ref = {
                            "number": match.group(1),
                            "text": match.group(2).strip()
                        }
                    extracted_refs.append(ref)
            
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