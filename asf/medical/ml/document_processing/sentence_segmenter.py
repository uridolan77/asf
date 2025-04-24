"""
Sentence Segmentation Module

This module provides advanced sentence segmentation capabilities for scientific documents,
with special handling for biomedical text including abbreviations, citations, and
domain-specific sentence boundary patterns.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Sentence:
    """Represents a segmented sentence with metadata."""
    text: str
    start: int
    end: int
    section_id: Optional[str] = None
    paragraph_id: Optional[int] = None
    sentence_id: Optional[int] = None
    metadata: Dict[str, Any] = None


class SentenceSegmenter:
    """
    Advanced sentence segmenter for scientific documents.
    
    This class implements multiple strategies for sentence segmentation:
    1. Rule-based segmentation with scientific text awareness
    2. ML-based segmentation using spaCy or other NLP libraries
    3. Hybrid approach combining rules and ML
    """
    
    def __init__(
        self,
        use_spacy: bool = True,
        use_scibert: bool = False,
        spacy_model: str = "en_core_sci_md",
        respect_newlines: bool = False,
        handle_citations: bool = True
    ):
        """
        Initialize the sentence segmenter.
        
        Args:
            use_spacy: Whether to use spaCy for sentence segmentation
            use_scibert: Whether to use SciBERT for sentence segmentation
            spacy_model: spaCy model to use
            respect_newlines: Whether to treat newlines as sentence boundaries
            handle_citations: Whether to handle in-text citations specially
        """
        self.use_spacy = use_spacy
        self.use_scibert = use_scibert
        self.spacy_model = spacy_model
        self.respect_newlines = respect_newlines
        self.handle_citations = handle_citations
        
        # Common abbreviations in scientific text that shouldn't break sentences
        self.abbreviations = {
            "e.g.", "i.e.", "etc.", "vs.", "cf.", "Dr.", "Prof.", "Fig.", "Eq.",
            "et al.", "et. al.", "ca.", "approx.", "wrt.", "w.r.t.", "viz.",
            "al.", "No.", "no.", "pp.", "p.", "vol.", "Vol.", "ed.", "eds.",
            "Ch.", "ch.", "pg.", "pgs.", "Pg.", "Pgs."
        }
        
        # Compile abbreviation pattern
        abbr_pattern = "|".join(re.escape(abbr) for abbr in self.abbreviations)
        self.abbr_pattern = re.compile(f"({abbr_pattern})")
        
        # Initialize spaCy if requested
        self.nlp = None
        if use_spacy:
            try:
                import spacy
                logger.info(f"Loading spaCy model: {spacy_model}")
                self.nlp = spacy.load(spacy_model)
            except ImportError:
                logger.warning("spaCy not available. Install with: pip install spacy")
                self.use_spacy = False
            except Exception as e:
                logger.warning(f"Error loading spaCy model: {str(e)}")
                self.use_spacy = False
        
        # Initialize SciBERT if requested
        self.scibert_tokenizer = None
        self.scibert_model = None
        if use_scibert:
            try:
                from transformers import AutoTokenizer, AutoModelForTokenClassification
                import torch
                
                logger.info("Loading SciBERT for sentence segmentation")
                self.scibert_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
                self.scibert_model = AutoModelForTokenClassification.from_pretrained("allenai/scibert_scivocab_uncased")
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.scibert_model.to(self.device)
            except ImportError:
                logger.warning("Transformers not available. Install with: pip install transformers")
                self.use_scibert = False
            except Exception as e:
                logger.warning(f"Error loading SciBERT: {str(e)}")
                self.use_scibert = False
    
    def segment_text(self, text: str) -> List[Sentence]:
        """
        Segment text into sentences using the best available method.
        
        Args:
            text: Text to segment
            
        Returns:
            List of Sentence objects
        """
        if self.use_spacy and self.nlp:
            return self.segment_with_spacy(text)
        elif self.use_scibert and self.scibert_model and self.scibert_tokenizer:
            return self.segment_with_scibert(text)
        else:
            return self.segment_with_rules(text)
    
    def segment_with_rules(self, text: str) -> List[Sentence]:
        """
        Segment text into sentences using rule-based approach.
        
        Args:
            text: Text to segment
            
        Returns:
            List of Sentence objects
        """
        sentences = []
        
        # Replace abbreviations temporarily to avoid false sentence boundaries
        text_with_placeholders = text
        abbr_matches = list(self.abbr_pattern.finditer(text))
        for i, match in enumerate(reversed(abbr_matches)):
            placeholder = f"__ABBR{i}__"
            start, end = match.span()
            text_with_placeholders = text_with_placeholders[:start] + placeholder + text_with_placeholders[end:]
        
        # Handle citations if requested
        if self.handle_citations:
            # Replace citations temporarily
            citation_pattern = r'\([^)]*\d{4}[^)]*\)'  # Simple pattern for author-year citations
            citation_matches = list(re.finditer(citation_pattern, text_with_placeholders))
            for i, match in enumerate(reversed(citation_matches)):
                placeholder = f"__CITE{i}__"
                start, end = match.span()
                text_with_placeholders = text_with_placeholders[:start] + placeholder + text_with_placeholders[end:]
        
        # Split on sentence boundaries
        # Look for: period, question mark, or exclamation mark followed by space and uppercase letter
        boundaries = []
        for match in re.finditer(r'[.!?](?:\s+)(?=[A-Z])', text_with_placeholders):
            boundaries.append(match.end() - 1)  # End of the punctuation
        
        # If respecting newlines, add newlines as potential boundaries
        if self.respect_newlines:
            for match in re.finditer(r'\n\s*\n', text_with_placeholders):
                boundaries.append(match.start())
        
        # Sort boundaries
        boundaries.sort()
        
        # Create sentences
        start = 0
        for i, boundary in enumerate(boundaries):
            # Get the sentence text from the original text
            sentence_text = text[start:boundary + 1].strip()
            
            # Create Sentence object
            sentence = Sentence(
                text=sentence_text,
                start=start,
                end=boundary + 1,
                sentence_id=i
            )
            
            sentences.append(sentence)
            start = boundary + 1
        
        # Add the last sentence if there's text remaining
        if start < len(text):
            sentence_text = text[start:].strip()
            if sentence_text:
                sentence = Sentence(
                    text=sentence_text,
                    start=start,
                    end=len(text),
                    sentence_id=len(sentences)
                )
                sentences.append(sentence)
        
        return sentences
    
    def segment_with_spacy(self, text: str) -> List[Sentence]:
        """
        Segment text into sentences using spaCy.
        
        Args:
            text: Text to segment
            
        Returns:
            List of Sentence objects
        """
        sentences = []
        
        try:
            doc = self.nlp(text)
            
            for i, sent in enumerate(doc.sents):
                # Create Sentence object
                sentence = Sentence(
                    text=sent.text.strip(),
                    start=sent.start_char,
                    end=sent.end_char,
                    sentence_id=i
                )
                
                sentences.append(sentence)
            
            return sentences
        except Exception as e:
            logger.warning(f"Error in spaCy sentence segmentation: {str(e)}")
            return self.segment_with_rules(text)
    
    def segment_with_scibert(self, text: str) -> List[Sentence]:
        """
        Segment text into sentences using SciBERT.
        
        Args:
            text: Text to segment
            
        Returns:
            List of Sentence objects
        """
        # This is a placeholder for SciBERT-based segmentation
        # In a real implementation, you would fine-tune SciBERT for sentence boundary detection
        # and use it to predict sentence boundaries
        
        # For now, fall back to rule-based segmentation
        logger.warning("SciBERT segmentation not fully implemented, falling back to rules")
        return self.segment_with_rules(text)
    
    def segment_document(self, doc_structure) -> Dict[str, List[Sentence]]:
        """
        Segment a document structure into sentences.
        
        Args:
            doc_structure: DocumentStructure object
            
        Returns:
            Dictionary mapping section IDs to lists of sentences
        """
        segmented_document = {}
        
        # Segment abstract
        if doc_structure.abstract:
            abstract_sentences = self.segment_text(doc_structure.abstract)
            for sent in abstract_sentences:
                sent.section_id = "abstract"
            segmented_document["abstract"] = abstract_sentences
        
        # Segment each section
        for i, section in enumerate(doc_structure.sections):
            section_id = f"section_{i}"
            section_sentences = self.segment_text(section.text)
            
            for sent in section_sentences:
                sent.section_id = section_id
            
            segmented_document[section_id] = section_sentences
            
            # Segment subsections recursively
            for j, subsection in enumerate(section.subsections):
                subsection_id = f"{section_id}_subsection_{j}"
                subsection_sentences = self.segment_text(subsection.text)
                
                for sent in subsection_sentences:
                    sent.section_id = subsection_id
                
                segmented_document[subsection_id] = subsection_sentences
        
        return segmented_document
