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

# Local imports
from .document_structure import DocumentStructure, SectionInfo
from .pdf_parser import PDFParser, ParsedPDF
from .section_classifier import IMRADSectionClassifier
from .reference_parser import ReferenceParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
        prefer_pdfminer: bool = False,
        use_enhanced_section_classifier: bool = True,
        use_advanced_reference_parser: bool = True,
        use_anystyle: bool = False,
        use_grobid: bool = False,
        device: Optional[str] = None
    ):
        """
        Initialize the biomedical document processor.

        Args:
            section_classifier_model: SciBERT model for section classification
            spacy_model: SciSpacy model for NLP processing
            use_umls: Whether to use UMLS entity linking
            prefer_pdfminer: Whether to prefer PDFMiner.six over PyMuPDF for PDF parsing
            use_enhanced_section_classifier: Whether to use the enhanced IMRAD section classifier
            device: Device for PyTorch models
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Initialize PDF parser
        self.pdf_parser = PDFParser(prefer_pdfminer=prefer_pdfminer)

        # Initialize reference parser
        self.reference_parser = ReferenceParser(
            use_anystyle=use_anystyle,
            use_grobid=use_grobid
        )
        self.use_advanced_reference_parser = use_advanced_reference_parser
        logger.info(f"PDF parser initialized with prefer_pdfminer={prefer_pdfminer}")

        # Initialize enhanced section classifier if requested
        self.use_enhanced_section_classifier = use_enhanced_section_classifier
        if use_enhanced_section_classifier:
            try:
                self.section_classifier = IMRADSectionClassifier(
                    model_name=section_classifier_model,
                    device=self.device
                )
                logger.info("Enhanced IMRAD section classifier initialized")
            except Exception as e:
                logger.warning(f"Could not initialize enhanced section classifier: {e}")
                self.use_enhanced_section_classifier = False

        # Define section types
        self.SECTION_TYPES = [
            "title", "abstract", "introduction", "background", "methods", "materials_and_methods",
            "results", "discussion", "conclusion", "references", "acknowledgments", "other"
        ]

        # Initialize section classifier
        try:
            logger.info(f"Loading section classifier model: {section_classifier_model}")
            self.tokenizer = AutoTokenizer.from_pretrained(section_classifier_model)
            self.section_classifier = AutoModelForSequenceClassification.from_pretrained(
                section_classifier_model,
                num_labels=len(self.SECTION_TYPES)
            ).to(self.device)
        except Exception as e:
            logger.warning(f"Could not load section classifier: {e}")
            self.tokenizer = None
            self.section_classifier = None

        # Initialize SpaCy model
        try:
            import spacy
            import scispacy

            logger.info(f"Loading SciSpacy model: {spacy_model}")
            self.nlp = spacy.load(spacy_model)

            # Add abbreviation detector
            from scispacy.abbreviation import AbbreviationDetector
            self.nlp.add_pipe("abbreviation_detector")

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
            self.nlp = None

    def extract_pdf_text(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file using the enhanced PDF parser.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Extracted text
        """
        parsed_pdf = self.pdf_parser.parse_pdf(pdf_path)

        if parsed_pdf is None:
            logger.error(f"Failed to parse PDF: {pdf_path}")
            return ""

        logger.info(f"Successfully parsed PDF using {parsed_pdf.parser_used} parser")
        return parsed_pdf.full_text

    def detect_section_headings(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Detect section headings in document text.

        Args:
            text: Document text

        Returns:
            List of tuples (heading, start_pos, end_pos)
        """
        # Common section heading patterns
        heading_patterns = [
            # Numbered sections (e.g., "1. Introduction")
            r'^\s*(\d+\.?\s+[A-Z][A-Za-z\s]+)$',

            # Uppercase headings (e.g., "INTRODUCTION")
            r'^\s*([A-Z]{2,}[A-Z\s]*)$',

            # Title case headings (e.g., "Materials and Methods")
            r'^\s*([A-Z][a-z]+(?:\s+(?:and|or|of|in|on|with|without|to|from|by|for|the|a|an)\s+|\s+)[A-Z][a-z]+(?:\s+[a-z]+)*)$'
        ]

        # Find all potential headings
        headings = []
        for line_match in re.finditer(r'(^|\n)(.+?)(\n|$)', text):
            line = line_match.group(2).strip()
            line_start = line_match.start(2)
            line_end = line_match.end(2)

            # Check if line matches any heading pattern
            for pattern in heading_patterns:
                if re.match(pattern, line, re.MULTILINE):
                    headings.append((line, line_start, line_end))
                    break

        return headings

    def classify_section(self, heading: str, text: str) -> str:
        """
        Classify section type based on heading and content.

        Args:
            heading: Section heading
            text: Section text

        Returns:
            Section type
        """
        # Use enhanced section classifier if available
        if self.use_enhanced_section_classifier and hasattr(self, 'section_classifier'):
            try:
                return self.section_classifier.classify_section(heading, text)
            except Exception as e:
                logger.warning(f"Error in enhanced section classification: {e}")

        # Fall back to simple rule-based classification
        heading_lower = heading.lower()

        # Check for common section types
        if any(x in heading_lower for x in ["abstract", "summary"]):
            return "abstract"
        elif any(x in heading_lower for x in ["introduction", "background"]):
            return "introduction"
        elif any(x in heading_lower for x in ["method", "materials", "procedure", "experimental"]):
            return "methods"
        elif "result" in heading_lower:
            return "results"
        elif "discussion" in heading_lower:
            return "discussion"
        elif any(x in heading_lower for x in ["conclusion", "summary", "findings"]):
            return "conclusion"
        elif any(x in heading_lower for x in ["reference", "bibliography", "literature"]):
            return "references"
        elif any(x in heading_lower for x in ["acknowledgment", "acknowledgement"]):
            return "acknowledgments"

        # Use model-based classification if available
        if self.tokenizer and self.section_classifier:
            try:
                # Prepare input
                inputs = self.tokenizer(
                    heading + ". " + text[:200],  # Use heading and start of text
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)

                # Get prediction
                with torch.no_grad():
                    outputs = self.section_classifier(**inputs)
                    prediction = torch.argmax(outputs.logits, dim=1).item()

                return self.SECTION_TYPES[prediction]
            except Exception as e:
                logger.warning(f"Error in model-based section classification: {e}")

        # Default to "other" if no match
        return "other"

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
        Identify subsections within main sections based on hierarchical numbering patterns,
        indentation, and semantic relationships.

        Args:
            sections: List of section info objects

        This function modifies the sections list in-place, reorganizing it into a hierarchical structure.
        """
        # Skip if there are no sections or only one section
        if len(sections) <= 1:
            return

        # First pass: Identify hierarchical numbering patterns
        i = 0
        while i < len(sections) - 1:
            current = sections[i]

            # Look ahead to find potential subsections
            j = i + 1
            subsections_added = False

            while j < len(sections):
                next_section = sections[j]

                # Check different numbering patterns
                # Pattern 1: Decimal numbering (1.1, 1.2, etc.)
                if self._is_decimal_subsection(current.heading, next_section.heading):
                    # Add as subsection and remove from main list
                    current.subsections.append(next_section)
                    sections.pop(j)
                    subsections_added = True
                    continue  # Don't increment j as we removed an element

                # Pattern 2: Outline numbering (I.A, I.B, etc.)
                elif self._is_outline_subsection(current.heading, next_section.heading):
                    current.subsections.append(next_section)
                    sections.pop(j)
                    subsections_added = True
                    continue

                # Pattern 3: Indentation or font size (detected from PDF layout if available)
                elif hasattr(next_section, 'metadata') and 'indentation' in next_section.metadata:
                    if next_section.metadata['indentation'] > current.metadata.get('indentation', 0):
                        current.subsections.append(next_section)
                        sections.pop(j)
                        subsections_added = True
                        continue

                # If we reach a section that's clearly not a subsection, break
                if self._is_same_level_or_higher(current.heading, next_section.heading):
                    break

                j += 1

            # If we added subsections, process them recursively
            if subsections_added and current.subsections:
                self.identify_subsections(current.subsections)

            i += 1

        # Second pass: Semantic grouping for sections without clear numbering
        self._group_related_sections(sections)

    def _is_decimal_subsection(self, parent_heading: str, child_heading: str) -> bool:
        """
        Check if child_heading is a decimal subsection of parent_heading.
        Examples: "1. Introduction" -> "1.1 Background"
        """
        parent_match = re.match(r'^(\d+)(?:[.)]\s*)', parent_heading)
        child_match = re.match(r'^(\d+)\.(\d+)(?:[.)]\s*)', child_heading)

        if parent_match and child_match:
            return parent_match.group(1) == child_match.group(1)
        return False

    def _is_outline_subsection(self, parent_heading: str, child_heading: str) -> bool:
        """
        Check if child_heading is an outline subsection of parent_heading.
        Examples: "I. Methods" -> "I.A. Study Design"
        """
        parent_match = re.match(r'^([IVX]+)(?:[.)]\s*)', parent_heading)
        child_match = re.match(r'^([IVX]+)\.([A-Z])(?:[.)]\s*)', child_heading)

        if parent_match and child_match:
            return parent_match.group(1) == child_match.group(1)
        return False

    def _is_same_level_or_higher(self, current_heading: str, next_heading: str) -> bool:
        """
        Check if next_heading is at the same level or higher than current_heading.
        """
        # Check decimal numbering
        current_decimal = re.match(r'^(\d+)(?:\.(\d+))?(?:[.)]\s*)', current_heading)
        next_decimal = re.match(r'^(\d+)(?:\.(\d+))?(?:[.)]\s*)', next_heading)

        if current_decimal and next_decimal:
            # If current is "1.1" and next is "1.2", they're same level
            if current_decimal.group(2) and next_decimal.group(2):
                return current_decimal.group(1) == next_decimal.group(1)
            # If current is "1" and next is "2", next is same level
            elif not current_decimal.group(2) and not next_decimal.group(2):
                return True
            # If current is "1.1" and next is "2", next is higher level
            elif current_decimal.group(2) and not next_decimal.group(2):
                return True
            return False

        # Check outline numbering
        current_outline = re.match(r'^([IVX]+)(?:\.([A-Z]))?(?:[.)]\s*)', current_heading)
        next_outline = re.match(r'^([IVX]+)(?:\.([A-Z]))?(?:[.)]\s*)', next_heading)

        if current_outline and next_outline:
            # Similar logic as above for outline numbering
            if current_outline.group(2) and next_outline.group(2):
                return current_outline.group(1) == next_outline.group(1)
            elif not current_outline.group(2) and not next_outline.group(2):
                return True
            elif current_outline.group(2) and not next_outline.group(2):
                return True
            return False

        # If numbering patterns don't match, use heuristics
        # Headings with similar length are likely at the same level
        return len(current_heading.split()) <= len(next_heading.split())

    def _group_related_sections(self, sections: List[SectionInfo]) -> None:
        """
        Group semantically related sections without clear hierarchical numbering.
        """
        # Common section groupings in scientific papers
        section_groups = {
            "methods": ["study design", "participants", "data collection", "statistical analysis"],
            "results": ["primary outcomes", "secondary outcomes", "adverse events"],
            "discussion": ["limitations", "implications", "future directions"]
        }

        i = 0
        while i < len(sections):
            current = sections[i]
            current_type = current.section_type.lower()

            # Check if this section is a potential parent section
            if current_type in section_groups:
                related_terms = section_groups[current_type]

                # Look ahead for related sections
                j = i + 1
                subsections_added = False

                while j < len(sections):
                    next_section = sections[j]
                    next_heading_lower = next_section.heading.lower()

                    # Check if next section is related to current section
                    is_related = False
                    for term in related_terms:
                        if term in next_heading_lower:
                            is_related = True
                            break

                    # If related and not a major section itself, add as subsection
                    if is_related and next_section.section_type.lower() not in section_groups:
                        current.subsections.append(next_section)
                        sections.pop(j)
                        subsections_added = True
                        continue

                    # If we hit another major section, stop looking
                    if next_section.section_type.lower() in section_groups:
                        break

                    j += 1

            i += 1

    def extract_title(self, text: str) -> str:
        """
        Extract the title from the document.

        Args:
            text: Document text

        Returns:
            Title text
        """
        # Try to find title at the beginning of the document
        lines = text.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]

        if non_empty_lines:
            # First non-empty line is often the title
            title_candidate = non_empty_lines[0]

            # Check if it's a reasonable title (not too long, not too short)
            if 3 <= len(title_candidate.split()) <= 30:
                return title_candidate

            # Try the first few lines
            for line in non_empty_lines[1:3]:
                if 3 <= len(line.split()) <= 30:
                    return line

        # Default title if none found
        return "Untitled Document"

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
        Extract references from the document using advanced reference parsing techniques.

        Args:
            text: Document text
            sections: Document sections

        Returns:
            List of reference dictionaries with detailed metadata
        """
        # Find references section
        ref_section = None
        for section in sections:
            if section.section_type == "references":
                ref_section = section
                break

        # If no dedicated references section found, try to find references at the end of the document
        if not ref_section:
            # Look for common reference section headings
            ref_headings = ["references", "bibliography", "literature cited"]
            ref_text = ""

            # Check the last 20% of the document for references
            last_part = text[int(len(text) * 0.8):]
            for heading in ref_headings:
                match = re.search(f"(?i)\\b{heading}\\b", last_part)
                if match:
                    ref_text = last_part[match.start():]
                    break

            if not ref_text:
                # If still no references found, use the last section as a fallback
                if sections:
                    ref_text = sections[-1].text
                else:
                    return []
        else:
            ref_text = ref_section.text

        # Use advanced reference parser if enabled
        if self.use_advanced_reference_parser:
            try:
                return self.reference_parser.extract_structured_references(ref_text)
            except Exception as e:
                logger.warning(f"Advanced reference parsing failed: {str(e)}. Falling back to basic parsing.")
                return self._basic_reference_extraction(ref_text)
        else:
            return self._basic_reference_extraction(ref_text)

    def _basic_reference_extraction(self, ref_text: str) -> List[Dict[str, Any]]:
        """
        Basic reference extraction using regex patterns (fallback method).

        Args:
            ref_text: Text containing references

        Returns:
            List of reference dictionaries
        """
        references = []

        # Extract individual references
        ref_patterns = [
            # Numbered references: [1] Author, Title...
            r'\[(\d+)\]\s+([^[]+?)(?=\[\d+\]|\Z)',

            # Author-year references: (Author et al., Year)
            r'(?:\n|\A)([A-Z][a-z]+(?:,?\s+et\s+al\.)?(?:,\s+[A-Z][a-z]+)*\s+\(\d{4}\)\..*?)(?=\n[A-Z]|\Z)',

            # Numbered list: 1. Author, Title...
            r'(?:\n|\A)(\d+\.\s+[^0-9]+?)(?=\n\d+\.|\Z)'
        ]

        for pattern in ref_patterns:
            matches = re.finditer(pattern, ref_text, re.DOTALL)
            if matches:
                for match in matches:
                    if len(match.groups()) >= 1:
                        ref_text = match.group(1).strip() if len(match.groups()) == 1 else match.group(2).strip()
                        ref_id = match.group(1) if len(match.groups()) > 1 else str(len(references) + 1)

                        # Parse reference components (basic)
                        ref_dict = {
                            "ref_id": f"ref_{ref_id}",
                            "raw_text": ref_text,
                            "authors": [],
                            "title": None,
                            "year": None,
                            "journal": None,
                            "volume": None,
                            "issue": None,
                            "pages": None,
                            "doi": None,
                            "url": None
                        }

                        # Try to extract year
                        year_match = re.search(r'\((\d{4})\)', ref_text)
                        if year_match:
                            ref_dict["year"] = int(year_match.group(1))

                        # Try to extract DOI
                        doi_match = re.search(r'(?:doi:|https?://doi.org/|DOI:?\s*)(10\.\d{4,}(?:\.\d+)*\/\S+)', ref_text)
                        if doi_match:
                            ref_dict["doi"] = doi_match.group(1)

                        references.append(ref_dict)

                # If we found references with one pattern, stop
                if references:
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

            # If PDF parsing failed, return empty document structure
            if not text:
                logger.error(f"Failed to extract text from PDF: {text_or_path}")
                return DocumentStructure(
                    title="Failed to Parse Document",
                    abstract=None,
                    sections=[],
                    references=[],
                    entities=[],
                    relations=[],
                    metadata={"error": "PDF parsing failed"}
                )
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
