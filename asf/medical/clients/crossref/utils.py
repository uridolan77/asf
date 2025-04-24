"""
Utility functions for CrossRef API client.
This module provides helper functions for processing CrossRef API data.
"""
import re
import json
import logging
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from .models import Author

logger = logging.getLogger(__name__)

def extract_year_from_date(date_data: Dict[str, Any]) -> Optional[int]:
    """
    Extract year from CrossRef date structure.
    
    Args:
        date_data: CrossRef date structure
    
    Returns:
        int: Year or None if not found
    """
    if not date_data:
        return None
    
    if 'date-parts' in date_data and len(date_data['date-parts']) > 0:
        parts = date_data['date-parts'][0]
        if len(parts) > 0:
            try:
                return int(parts[0])
            except (ValueError, TypeError):
                pass
    
    return None

def format_citation(work: Dict[str, Any], style: str = "apa") -> str:
    """
    Format a work as a citation in the specified style.
    
    Args:
        work: CrossRef work data
        style: Citation style (currently supports 'apa' and 'mla')
    
    Returns:
        str: Formatted citation
    """
    if not work:
        return ""
    
    # Extract common elements
    title = work.get('title', [''])[0] if 'title' in work and work['title'] else ""
    
    # Extract authors
    authors = []
    if 'author' in work and work['author']:
        for author in work['author']:
            if 'family' in author and 'given' in author:
                authors.append(f"{author['family']}, {author['given']}")
            elif 'family' in author:
                authors.append(author['family'])
    
    # Extract year
    year = None
    if 'issued' in work:
        year = extract_year_from_date(work['issued'])
    
    # Extract journal/container info
    container = ""
    if 'container-title' in work and work['container-title']:
        container = work['container-title'][0]
    
    # Extract volume, issue, pages
    volume = work.get('volume', '')
    issue = work.get('issue', '')
    page = work.get('page', '')
    
    # Extract DOI
    doi = work.get('DOI', '')
    
    # Format citation based on style
    if style.lower() == "apa":
        # Authors
        citation = ""
        if authors:
            if len(authors) == 1:
                citation += f"{authors[0]}. "
            elif len(authors) == 2:
                citation += f"{authors[0]} & {authors[1]}. "
            elif len(authors) > 2:
                citation += f"{authors[0]} et al. "
        
        # Year
        if year:
            citation += f"({year}). "
        
        # Title
        citation += f"{title}. "
        
        # Container (journal, book, etc.)
        if container:
            citation += f"{container}"
            
            # Volume and issue
            if volume:
                citation += f", {volume}"
                if issue:
                    citation += f"({issue})"
            
            # Pages
            if page:
                citation += f", {page}"
            
            citation += ". "
        
        # DOI
        if doi:
            citation += f"https://doi.org/{doi}"
        
        return citation
    
    elif style.lower() == "mla":
        # Authors
        citation = ""
        if authors:
            if len(authors) == 1:
                citation += f"{authors[0]}. "
            elif len(authors) == 2:
                name_parts = authors[0].split(", ")
                if len(name_parts) > 1:
                    citation += f"{name_parts[0]} and {name_parts[1]} {authors[1]}. "
                else:
                    citation += f"{authors[0]} and {authors[1]}. "
            elif len(authors) > 2:
                citation += f"{authors[0]}, et al. "
        
        # Title
        citation += f"\"{title}.\" "
        
        # Container (journal, book, etc.)
        if container:
            citation += f"{container}, "
            
            # Volume and issue
            if volume:
                citation += f"vol. {volume}"
                if issue:
                    citation += f", no. {issue}"
                citation += ", "
            
            # Year
            if year:
                citation += f"{year}, "
            
            # Pages
            if page:
                citation += f"pp. {page}"
            
            citation += ". "
        
        # DOI
        if doi:
            citation += f"DOI: {doi}."
        
        return citation
    
    else:
        return f"{', '.join(authors) if authors else 'Unknown'}, \"{title}\", {container}, {year if year else 'n.d.'}. DOI: {doi}"

def parse_author_string(author_string: str) -> Optional[Author]:
    """
    Parse an author string into an Author object.
    
    Args:
        author_string: String representation of an author (e.g., "Smith, John")
    
    Returns:
        Author: Author object or None if parsing failed
    """
    if not author_string:
        return None
    
    # Try "Last, First" format
    comma_parts = author_string.split(',', 1)
    if len(comma_parts) == 2:
        return Author(
            family=comma_parts[0].strip(),
            given=comma_parts[1].strip()
        )
    
    # Try "First Last" format
    space_parts = author_string.strip().split(' ')
    if len(space_parts) >= 2:
        return Author(
            given=' '.join(space_parts[:-1]),
            family=space_parts[-1]
        )
    
    # Just use the whole string as a family name
    return Author(family=author_string.strip())

def normalize_doi(doi: str) -> str:
    """
    Normalize a DOI string.
    
    Args:
        doi: DOI string
    
    Returns:
        str: Normalized DOI
    """
    if not doi:
        return ""
    
    # Remove any URL prefix
    doi = re.sub(r'^https?://doi\.org/', '', doi)
    doi = re.sub(r'^https?://dx\.doi\.org/', '', doi)
    doi = re.sub(r'^https?://', '', doi)
    doi = re.sub(r'^doi\.org/', '', doi)
    doi = re.sub(r'^dx\.doi\.org/', '', doi)
    
    # Trim whitespace
    doi = doi.strip()
    
    return doi

def extract_dois_from_text(text: str) -> List[str]:
    """
    Extract DOIs from text.
    
    Args:
        text: Text that may contain DOIs
    
    Returns:
        List[str]: List of extracted DOIs
    """
    if not text:
        return []
    
    # Regular expressions for DOI patterns
    doi_patterns = [
        r'10\.\d{4,9}/[-._;()/:A-Za-z0-9]+',  # Basic DOI pattern
        r'https?://doi\.org/10\.\d{4,9}/[-._;()/:A-Za-z0-9]+',  # DOI URL
        r'https?://dx\.doi\.org/10\.\d{4,9}/[-._;()/:A-Za-z0-9]+'  # dx.doi.org URL
    ]
    
    dois = []
    for pattern in doi_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            dois.append(normalize_doi(match))
    
    # Remove duplicates while preserving order
    unique_dois = []
    seen = set()
    for doi in dois:
        if doi not in seen:
            seen.add(doi)
            unique_dois.append(doi)
    
    return unique_dois

def convert_to_json(data: Any) -> str:
    """
    Convert data to a JSON string.
    
    Args:
        data: Data to convert
    
    Returns:
        str: JSON string representation
    """
    def json_serial(obj):
        """Helper method to serialize special types for JSON."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        return str(obj)
    
    return json.dumps(data, default=json_serial, indent=2)

def retry_function(func, max_retries=3, delay_seconds=1, backoff_factor=2):
    """
    Decorator to retry a function with exponential backoff.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retries
        delay_seconds: Initial delay in seconds
        backoff_factor: Factor to increase delay by on each retry
    
    Returns:
        Function result or raises the last exception
    """
    def wrapper(*args, **kwargs):
        retries = 0
        current_delay = delay_seconds
        
        while retries < max_retries:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                retries += 1
                if retries >= max_retries:
                    logger.error(f"Max retries ({max_retries}) exceeded for {func.__name__}: {str(e)}")
                    raise
                
                logger.warning(f"Retry {retries}/{max_retries} for {func.__name__} after error: {str(e)}")
                time.sleep(current_delay)
                current_delay *= backoff_factor
    
    return wrapper