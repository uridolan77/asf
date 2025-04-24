"""
PDF Parsing Module

This module provides robust PDF text extraction capabilities using multiple parsers.
It implements a dual-parser approach with PyMuPDF and PDFMiner.six to handle
complex scientific layouts and preserve document structure.
"""

import os
import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ParsedPage:
    """Represents a parsed page from a PDF document."""
    page_number: int
    text: str
    layout_info: Optional[Dict[str, Any]] = None
    

@dataclass
class ParsedPDF:
    """Represents a parsed PDF document."""
    pages: List[ParsedPage]
    metadata: Dict[str, Any]
    parser_used: str
    
    @property
    def full_text(self) -> str:
        """Get the full text of the document."""
        return "\n\n".join([page.text for page in self.pages])
    
    @property
    def page_count(self) -> int:
        """Get the number of pages in the document."""
        return len(self.pages)


class PDFParser:
    """
    PDF Parser that combines PyMuPDF and PDFMiner.six for robust text extraction.
    
    This class implements a dual-parser approach, selecting the appropriate parser
    based on document characteristics or using both and choosing the better result.
    """
    
    def __init__(self, prefer_pdfminer: bool = False):
        """
        Initialize the PDF parser.
        
        Args:
            prefer_pdfminer: Whether to prefer PDFMiner.six over PyMuPDF when both are available
        """
        self.prefer_pdfminer = prefer_pdfminer
        self.pymupdf_available = False
        self.pdfminer_available = False
        
        # Try to import PyMuPDF
        try:
            import fitz
            self.fitz = fitz
            self.pymupdf_available = True
            logger.info("PyMuPDF (fitz) is available")
        except ImportError:
            self.fitz = None
            logger.warning("PyMuPDF (fitz) is not available. Install with: pip install pymupdf")
        
        # Try to import PDFMiner.six
        try:
            from pdfminer.high_level import extract_pages, extract_text
            from pdfminer.layout import LAParams, LTTextContainer
            self.extract_pages = extract_pages
            self.extract_text = extract_text
            self.LAParams = LAParams
            self.LTTextContainer = LTTextContainer
            self.pdfminer_available = True
            logger.info("PDFMiner.six is available")
        except ImportError:
            self.extract_pages = None
            self.extract_text = None
            self.LAParams = None
            self.LTTextContainer = None
            logger.warning("PDFMiner.six is not available. Install with: pip install pdfminer.six")
    
    def parse_pdf(self, pdf_path: str) -> Optional[ParsedPDF]:
        """
        Parse a PDF file using the best available parser.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            ParsedPDF object or None if parsing failed
        """
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return None
        
        # Determine which parser to use
        if self.pymupdf_available and self.pdfminer_available:
            # Both parsers are available, use heuristics to choose
            if self.is_complex_layout(pdf_path):
                logger.info(f"Complex layout detected in {pdf_path}, using PDFMiner.six")
                return self.parse_with_pdfminer(pdf_path)
            elif self.prefer_pdfminer:
                logger.info(f"Using preferred parser PDFMiner.six for {pdf_path}")
                return self.parse_with_pdfminer(pdf_path)
            else:
                logger.info(f"Using PyMuPDF for {pdf_path}")
                return self.parse_with_pymupdf(pdf_path)
        elif self.pymupdf_available:
            logger.info(f"Only PyMuPDF is available, using it for {pdf_path}")
            return self.parse_with_pymupdf(pdf_path)
        elif self.pdfminer_available:
            logger.info(f"Only PDFMiner.six is available, using it for {pdf_path}")
            return self.parse_with_pdfminer(pdf_path)
        else:
            logger.error("No PDF parser is available. Install PyMuPDF or PDFMiner.six")
            return None
    
    def parse_with_pymupdf(self, pdf_path: str) -> Optional[ParsedPDF]:
        """
        Parse a PDF file using PyMuPDF (fitz).
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            ParsedPDF object or None if parsing failed
        """
        if not self.pymupdf_available:
            logger.error("PyMuPDF is not available")
            return None
        
        try:
            doc = self.fitz.open(pdf_path)
            pages = []
            
            for i, page in enumerate(doc):
                text = page.get_text()
                pages.append(ParsedPage(
                    page_number=i + 1,
                    text=text,
                    layout_info={"width": page.rect.width, "height": page.rect.height}
                ))
            
            metadata = {
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "keywords": doc.metadata.get("keywords", ""),
                "page_count": len(doc)
            }
            
            return ParsedPDF(
                pages=pages,
                metadata=metadata,
                parser_used="pymupdf"
            )
        except Exception as e:
            logger.error(f"Error parsing PDF with PyMuPDF: {str(e)}")
            return None
    
    def parse_with_pdfminer(self, pdf_path: str) -> Optional[ParsedPDF]:
        """
        Parse a PDF file using PDFMiner.six.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            ParsedPDF object or None if parsing failed
        """
        if not self.pdfminer_available:
            logger.error("PDFMiner.six is not available")
            return None
        
        try:
            # Extract text with layout analysis
            laparams = self.LAParams(
                line_margin=0.5,
                char_margin=2.0,
                word_margin=0.1,
                boxes_flow=0.5,
                detect_vertical=True
            )
            
            pages = []
            page_texts = []
            
            # Extract pages with layout
            for i, page_layout in enumerate(self.extract_pages(
                pdf_path, laparams=laparams
            )):
                page_text = ""
                layout_info = {
                    "width": page_layout.width,
                    "height": page_layout.height,
                    "elements": []
                }
                
                # Extract text from layout elements
                for element in page_layout:
                    if isinstance(element, self.LTTextContainer):
                        page_text += element.get_text() + "\n"
                        layout_info["elements"].append({
                            "type": element.__class__.__name__,
                            "x0": element.x0,
                            "y0": element.y0,
                            "x1": element.x1,
                            "y1": element.y1,
                            "text_length": len(element.get_text())
                        })
                
                pages.append(ParsedPage(
                    page_number=i + 1,
                    text=page_text.strip(),
                    layout_info=layout_info
                ))
                page_texts.append(page_text.strip())
            
            # Extract metadata (limited in PDFMiner)
            metadata = {
                "page_count": len(pages)
            }
            
            return ParsedPDF(
                pages=pages,
                metadata=metadata,
                parser_used="pdfminer"
            )
        except Exception as e:
            logger.error(f"Error parsing PDF with PDFMiner.six: {str(e)}")
            return None
    
    def is_complex_layout(self, pdf_path: str) -> bool:
        """
        Determine if a PDF has a complex layout that would benefit from PDFMiner.six.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            True if the PDF has a complex layout, False otherwise
        """
        if not self.pymupdf_available:
            # If PyMuPDF is not available, we can't check layout complexity
            return True
        
        try:
            doc = self.fitz.open(pdf_path)
            
            # Check a sample of pages (first, middle, last)
            pages_to_check = [0]
            if len(doc) > 2:
                pages_to_check.append(len(doc) // 2)
            if len(doc) > 1:
                pages_to_check.append(len(doc) - 1)
            
            complex_layout_indicators = 0
            
            for page_num in pages_to_check:
                page = doc[page_num]
                
                # Check for multiple columns
                blocks = page.get_text("blocks")
                if len(blocks) > 5:  # Arbitrary threshold for multiple blocks
                    # Check if blocks are arranged in columns
                    x_positions = [block[0] for block in blocks]
                    x_clusters = self._cluster_values(x_positions, threshold=50)
                    if len(x_clusters) >= 2:
                        complex_layout_indicators += 1
                
                # Check for tables
                tables = page.find_tables()
                if tables and len(tables.tables) > 0:
                    complex_layout_indicators += 1
                
                # Check for mathematical content
                text = page.get_text()
                if re.search(r'[∫∑∏√∂∇∆∞≈≠≤≥±]|[a-zA-Z]_\{[^}]+\}', text):
                    complex_layout_indicators += 1
            
            # If we have enough indicators of complexity, use PDFMiner
            return complex_layout_indicators >= 2
        except Exception as e:
            logger.warning(f"Error checking layout complexity: {str(e)}")
            return False
    
    def _cluster_values(self, values: List[float], threshold: float) -> List[List[float]]:
        """
        Cluster values that are close to each other.
        
        Args:
            values: List of values to cluster
            threshold: Maximum distance between values in the same cluster
            
        Returns:
            List of clusters, where each cluster is a list of values
        """
        if not values:
            return []
        
        sorted_values = sorted(values)
        clusters = [[sorted_values[0]]]
        
        for value in sorted_values[1:]:
            if value - clusters[-1][-1] <= threshold:
                clusters[-1].append(value)
            else:
                clusters.append([value])
        
        return clusters
    
    def compare_parser_outputs(self, pdf_path: str) -> Tuple[ParsedPDF, Dict[str, Any]]:
        """
        Compare outputs from both parsers and return the better one with comparison metrics.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple of (selected ParsedPDF, comparison metrics)
        """
        if not (self.pymupdf_available and self.pdfminer_available):
            logger.warning("Both parsers must be available for comparison")
            return self.parse_pdf(pdf_path), {"comparison": "not_available"}
        
        pymupdf_result = self.parse_with_pymupdf(pdf_path)
        pdfminer_result = self.parse_with_pdfminer(pdf_path)
        
        if not pymupdf_result:
            return pdfminer_result, {"selected": "pdfminer", "reason": "pymupdf_failed"}
        
        if not pdfminer_result:
            return pymupdf_result, {"selected": "pymupdf", "reason": "pdfminer_failed"}
        
        # Compare results
        comparison = {
            "pymupdf_char_count": len(pymupdf_result.full_text),
            "pdfminer_char_count": len(pdfminer_result.full_text),
            "pymupdf_page_count": pymupdf_result.page_count,
            "pdfminer_page_count": pdfminer_result.page_count,
        }
        
        # Calculate text similarity
        similarity = self._calculate_text_similarity(
            pymupdf_result.full_text, pdfminer_result.full_text
        )
        comparison["text_similarity"] = similarity
        
        # Check for structure preservation
        pymupdf_structure_score = self._evaluate_structure_preservation(pymupdf_result)
        pdfminer_structure_score = self._evaluate_structure_preservation(pdfminer_result)
        comparison["pymupdf_structure_score"] = pymupdf_structure_score
        comparison["pdfminer_structure_score"] = pdfminer_structure_score
        
        # Make decision based on comparison
        if pdfminer_structure_score > pymupdf_structure_score * 1.2:
            # PDFMiner is significantly better at preserving structure
            selected = pdfminer_result
            comparison["selected"] = "pdfminer"
            comparison["reason"] = "better_structure"
        elif pymupdf_result.page_count > pdfminer_result.page_count:
            # PyMuPDF found more pages
            selected = pymupdf_result
            comparison["selected"] = "pymupdf"
            comparison["reason"] = "more_pages"
        elif len(pymupdf_result.full_text) > len(pdfminer_result.full_text) * 1.2:
            # PyMuPDF extracted significantly more text
            selected = pymupdf_result
            comparison["selected"] = "pymupdf"
            comparison["reason"] = "more_text"
        elif len(pdfminer_result.full_text) > len(pymupdf_result.full_text) * 1.2:
            # PDFMiner extracted significantly more text
            selected = pdfminer_result
            comparison["selected"] = "pdfminer"
            comparison["reason"] = "more_text"
        else:
            # Default to preferred parser or PyMuPDF for speed
            selected = pdfminer_result if self.prefer_pdfminer else pymupdf_result
            comparison["selected"] = "pdfminer" if self.prefer_pdfminer else "pymupdf"
            comparison["reason"] = "default_preference"
        
        return selected, comparison
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two text strings.
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Similarity score between 0 and 1
        """
        # Simple character-based Jaccard similarity
        if not text1 or not text2:
            return 0.0
        
        # Convert to sets of words
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _evaluate_structure_preservation(self, parsed_pdf: ParsedPDF) -> float:
        """
        Evaluate how well the parser preserved document structure.
        
        Args:
            parsed_pdf: ParsedPDF object
            
        Returns:
            Structure preservation score between 0 and 1
        """
        # This is a simplified heuristic
        structure_indicators = 0
        max_indicators = 3
        
        # Check for paragraph breaks
        paragraph_breaks = 0
        for page in parsed_pdf.pages:
            paragraph_breaks += page.text.count('\n\n')
        
        if paragraph_breaks > 5:
            structure_indicators += 1
        
        # Check for section headings
        section_heading_pattern = r'\n[A-Z][A-Za-z\s]+\n'
        section_headings = 0
        for page in parsed_pdf.pages:
            section_headings += len(re.findall(section_heading_pattern, page.text))
        
        if section_headings > 3:
            structure_indicators += 1
        
        # Check for layout information
        has_layout_info = all(page.layout_info is not None for page in parsed_pdf.pages)
        if has_layout_info:
            structure_indicators += 1
        
        return structure_indicators / max_indicators
