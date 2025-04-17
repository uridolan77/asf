"""
Advanced Reference Parser Module

This module provides enhanced reference parsing capabilities for scientific documents.
It implements multiple parsing strategies including regex-based parsing, rule-based
extraction, and integration with specialized reference parsing libraries.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
import json
import os
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ParsedReference:
    """Represents a parsed bibliographic reference."""
    ref_id: str
    raw_text: str
    authors: List[str] = None
    title: Optional[str] = None
    journal: Optional[str] = None
    year: Optional[int] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "ref_id": self.ref_id,
            "raw_text": self.raw_text,
            "authors": self.authors,
            "title": self.title,
            "journal": self.journal,
            "year": self.year,
            "volume": self.volume,
            "issue": self.issue,
            "pages": self.pages,
            "doi": self.doi,
            "pmid": self.pmid,
            "url": self.url
        }


class ReferenceParser:
    """
    Advanced reference parser for scientific documents.
    
    This class implements multiple strategies for reference parsing:
    1. Regex-based parsing for common citation formats
    2. Rule-based extraction for structured references
    3. Integration with specialized reference parsing libraries (if available)
    """
    
    def __init__(
        self,
        use_anystyle: bool = False,
        use_grobid: bool = False,
        grobid_url: str = "http://localhost:8070",
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the reference parser.
        
        Args:
            use_anystyle: Whether to use Anystyle for reference parsing
            use_grobid: Whether to use GROBID for reference parsing
            grobid_url: URL of the GROBID service
            cache_dir: Directory to cache parsed references
        """
        self.use_anystyle = use_anystyle
        self.use_grobid = use_grobid
        self.grobid_url = grobid_url
        self.cache_dir = cache_dir
        
        # Initialize cache if specified
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        # Common patterns for DOIs, PMIDs, and URLs
        self.doi_pattern = re.compile(r'(?:doi:|https?://doi.org/|DOI:?\s*)(10\.\d{4,}(?:\.\d+)*\/\S+)')
        self.pmid_pattern = re.compile(r'PMID:?\s*(\d{1,8})')
        self.url_pattern = re.compile(r'(https?://[^\s\]]+)')
        
        # Initialize Anystyle if requested
        self.anystyle_available = False
        if use_anystyle:
            try:
                import anystyle
                self.anystyle = anystyle
                self.anystyle_available = True
                logger.info("Anystyle initialized for reference parsing")
            except ImportError:
                logger.warning("Anystyle requested but not available. Install with: pip install anystyle-cli")
        
        # Initialize GROBID client if requested
        self.grobid_available = False
        if use_grobid:
            try:
                import requests
                self.requests = requests
                
                # Test GROBID connection
                response = requests.get(f"{grobid_url}/api/isalive")
                if response.status_code == 200:
                    self.grobid_available = True
                    logger.info(f"GROBID service available at {grobid_url}")
                else:
                    logger.warning(f"GROBID service not responding at {grobid_url}")
            except ImportError:
                logger.warning("Requests library not available. Install with: pip install requests")
            except Exception as e:
                logger.warning(f"Error connecting to GROBID service: {str(e)}")
    
    def parse_references(self, text: str) -> List[ParsedReference]:
        """
        Parse references from text using the best available method.
        
        Args:
            text: Text containing references
            
        Returns:
            List of ParsedReference objects
        """
        # Try specialized parsers first if available
        if self.anystyle_available:
            try:
                refs = self.parse_with_anystyle(text)
                if refs:
                    return refs
            except Exception as e:
                logger.warning(f"Error parsing with Anystyle: {str(e)}")
        
        if self.grobid_available:
            try:
                refs = self.parse_with_grobid(text)
                if refs:
                    return refs
            except Exception as e:
                logger.warning(f"Error parsing with GROBID: {str(e)}")
        
        # Fall back to regex-based parsing
        return self.parse_with_regex(text)
    
    def parse_with_regex(self, text: str) -> List[ParsedReference]:
        """
        Parse references using regex patterns.
        
        Args:
            text: Text containing references
            
        Returns:
            List of ParsedReference objects
        """
        references = []
        
        # Split text into potential references
        # Look for common reference patterns like numbered references [1], [2], etc.
        ref_blocks = re.split(r'\n\s*(?:\[\d+\]|\d+\.)\s+', text)
        
        # If we couldn't split by numbers, try splitting by newlines
        if len(ref_blocks) <= 1:
            ref_blocks = [block.strip() for block in text.split('\n\n') if block.strip()]
        
        for i, block in enumerate(ref_blocks):
            if not block.strip():
                continue
                
            ref_id = f"ref_{i+1}"
            ref = ParsedReference(ref_id=ref_id, raw_text=block.strip())
            
            # Extract DOI if present
            doi_match = self.doi_pattern.search(block)
            if doi_match:
                ref.doi = doi_match.group(1)
            
            # Extract PMID if present
            pmid_match = self.pmid_pattern.search(block)
            if pmid_match:
                ref.pmid = pmid_match.group(1)
            
            # Extract URL if present
            url_match = self.url_pattern.search(block)
            if url_match:
                ref.url = url_match.group(1)
            
            # Extract year if present (4 digits in parentheses or after comma)
            year_match = re.search(r'(?:\(|\s)(\d{4})(?:\)|\s|,)', block)
            if year_match:
                try:
                    ref.year = int(year_match.group(1))
                except ValueError:
                    pass
            
            # Extract authors (simplified - assumes authors are at the beginning)
            author_match = re.match(r'^([^\.]+)\.', block)
            if author_match:
                authors_text = author_match.group(1).strip()
                # Split by commas and "and"
                authors = re.split(r',\s*(?:and\s+)?|\s+and\s+', authors_text)
                ref.authors = [author.strip() for author in authors if author.strip()]
            
            # Extract title (simplified - assumes title is after authors and before journal)
            if author_match:
                remaining = block[author_match.end():].strip()
                title_match = re.match(r'^([^\.]+)\.', remaining)
                if title_match:
                    ref.title = title_match.group(1).strip()
            
            # Extract journal and other details (simplified approach)
            journal_match = re.search(r'([A-Z][A-Za-z\s]+)\.?\s+(?:Vol\.?|Volume)?\s*(\d+)(?:\s*\((\d+)\))?(?:\s*:\s*(\d+(?:-\d+)?))?', block)
            if journal_match:
                ref.journal = journal_match.group(1).strip()
                ref.volume = journal_match.group(2)
                if journal_match.group(3):
                    ref.issue = journal_match.group(3)
                if journal_match.group(4):
                    ref.pages = journal_match.group(4)
            
            references.append(ref)
        
        return references
    
    def parse_with_anystyle(self, text: str) -> List[ParsedReference]:
        """
        Parse references using Anystyle.
        
        Args:
            text: Text containing references
            
        Returns:
            List of ParsedReference objects
        """
        if not self.anystyle_available:
            return []
        
        try:
            # Use Anystyle to parse references
            finder = self.anystyle.finder
            parser = self.anystyle.parser
            
            # Find reference strings
            ref_strings = finder.find(text)
            
            # Parse references
            parsed_refs = parser.parse(ref_strings)
            
            # Convert to our format
            references = []
            for i, ref_dict in enumerate(parsed_refs):
                ref_id = f"ref_{i+1}"
                
                # Get raw text
                raw_text = ref_dict.get('raw', '')
                
                # Create reference object
                ref = ParsedReference(
                    ref_id=ref_id,
                    raw_text=raw_text,
                    authors=[a for a in ref_dict.get('author', []) if a],
                    title=ref_dict.get('title', [None])[0],
                    journal=ref_dict.get('journal', [None])[0],
                    year=int(ref_dict.get('year', [0])[0]) if ref_dict.get('year') else None,
                    volume=ref_dict.get('volume', [None])[0],
                    issue=ref_dict.get('issue', [None])[0],
                    pages=ref_dict.get('pages', [None])[0],
                    doi=ref_dict.get('doi', [None])[0],
                    url=ref_dict.get('url', [None])[0]
                )
                
                references.append(ref)
            
            return references
        except Exception as e:
            logger.error(f"Error parsing with Anystyle: {str(e)}")
            return []
    
    def parse_with_grobid(self, text: str) -> List[ParsedReference]:
        """
        Parse references using GROBID.
        
        Args:
            text: Text containing references
            
        Returns:
            List of ParsedReference objects
        """
        if not self.grobid_available:
            return []
        
        try:
            # Create a temporary file with the text
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp:
                temp.write(text)
                temp_path = temp.name
            
            # Call GROBID API
            url = f"{self.grobid_url}/api/processReferences"
            files = {'input': open(temp_path, 'rb')}
            
            response = self.requests.post(url, files=files)
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            if response.status_code != 200:
                logger.warning(f"GROBID returned status code {response.status_code}")
                return []
            
            # Parse TEI XML response
            try:
                from lxml import etree
                root = etree.fromstring(response.content)
                
                # Extract references
                references = []
                for i, ref_elem in enumerate(root.xpath('//biblStruct')):
                    ref_id = f"ref_{i+1}"
                    
                    # Extract raw text
                    raw_text = ' '.join(ref_elem.xpath('.//text()'))
                    
                    # Extract authors
                    authors = []
                    for author in ref_elem.xpath('.//author'):
                        forename = ' '.join(author.xpath('./forename/text()'))
                        surname = ' '.join(author.xpath('./surname/text()'))
                        if surname:
                            if forename:
                                authors.append(f"{surname}, {forename}")
                            else:
                                authors.append(surname)
                    
                    # Extract other metadata
                    title = ' '.join(ref_elem.xpath('.//title[@level="a"]/text()'))
                    journal = ' '.join(ref_elem.xpath('.//title[@level="j"]/text()'))
                    
                    year_elem = ref_elem.xpath('.//date/@when')
                    year = int(year_elem[0][:4]) if year_elem else None
                    
                    volume = ' '.join(ref_elem.xpath('.//biblScope[@unit="volume"]/text()'))
                    issue = ' '.join(ref_elem.xpath('.//biblScope[@unit="issue"]/text()'))
                    pages = ' '.join(ref_elem.xpath('.//biblScope[@unit="page"]/text()'))
                    
                    # Extract identifiers
                    doi = None
                    for idno in ref_elem.xpath('.//idno'):
                        if idno.get('type') == 'DOI':
                            doi = idno.text
                    
                    # Create reference object
                    ref = ParsedReference(
                        ref_id=ref_id,
                        raw_text=raw_text,
                        authors=authors,
                        title=title if title else None,
                        journal=journal if journal else None,
                        year=year,
                        volume=volume if volume else None,
                        issue=issue if issue else None,
                        pages=pages if pages else None,
                        doi=doi
                    )
                    
                    references.append(ref)
                
                return references
            except ImportError:
                logger.warning("lxml not available. Install with: pip install lxml")
                return []
            except Exception as e:
                logger.error(f"Error parsing GROBID response: {str(e)}")
                return []
        except Exception as e:
            logger.error(f"Error calling GROBID service: {str(e)}")
            return []
    
    def extract_structured_references(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract structured references from text.
        
        Args:
            text: Text containing references
            
        Returns:
            List of structured reference dictionaries
        """
        parsed_refs = self.parse_references(text)
        return [ref.to_dict() for ref in parsed_refs]
    
    def extract_citations(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract in-text citations from document text.
        
        Args:
            text: Document text
            
        Returns:
            List of tuples (citation_text, ref_id)
        """
        citations = []
        
        # Look for common citation patterns
        # Pattern 1: [1], [2,3], [4-6]
        bracket_citations = re.finditer(r'\[(\d+(?:[-,]\d+)*)\]', text)
        for match in bracket_citations:
            citation_text = match.group(0)
            ref_ids = []
            
            # Parse the citation reference numbers
            ref_numbers = match.group(1)
            for part in re.split(r',', ref_numbers):
                if '-' in part:
                    # Handle ranges like [1-3]
                    start, end = map(int, part.split('-'))
                    ref_ids.extend([f"ref_{i}" for i in range(start, end + 1)])
                else:
                    # Handle single numbers
                    ref_ids.append(f"ref_{part}")
            
            citations.append((citation_text, ref_ids))
        
        # Pattern 2: (Author et al., 2020)
        author_year_citations = re.finditer(r'\(([A-Za-z]+(?:\s+et\s+al\.)?(?:,\s+\d{4})?)\)', text)
        for match in author_year_citations:
            citation_text = match.group(0)
            ref_text = match.group(1)
            
            # For author-year citations, we can't directly map to ref_id without additional processing
            # This would require matching against the parsed references
            citations.append((citation_text, [ref_text]))
        
        return citations
