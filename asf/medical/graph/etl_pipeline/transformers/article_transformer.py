"""
Article Data Transformer

This module transforms PubMed article data into a format suitable for loading
into a Neo4j graph database. It handles normalization, cleaning, and structuring
of article data.
"""

import logging
from typing import Dict, List, Any, Optional
import re
import pandas as pd
from datetime import datetime

logger = logging.getLogger("biomedical_etl.transformers.article")

class ArticleTransformer:
    """
    Transformer for PubMed article data.
    
    This class provides methods for transforming article data into a format
    suitable for loading into a Neo4j graph database. It handles normalization,
    cleaning, and structuring of article data.
    """
    
    def __init__(self):
        """Initialize the article transformer."""
        logger.info("Initialized Article Transformer")
    
    def transform(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Transform a list of articles.
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            List of transformed article dictionaries
        """
        logger.info(f"Transforming {len(articles)} articles")
        transformed_articles = []
        
        for article in articles:
            try:
                transformed = self._transform_article(article)
                transformed_articles.append(transformed)
            except Exception as e:
                logger.error(f"Error transforming article {article.get('pmid', 'unknown')}: {str(e)}")
        
        logger.info(f"Transformed {len(transformed_articles)} articles")
        return transformed_articles
    
    def _transform_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform a single article.
        
        Args:
            article: Article dictionary
            
        Returns:
            Transformed article dictionary
        """
        # Extract basic fields
        pmid = article.get("pmid", "")
        title = article.get("title", "")
        abstract = article.get("abstract", "")
        
        # Clean and normalize text fields
        title = self._clean_text(title)
        abstract = self._clean_text(abstract)
        
        # Format publication date
        publication_date = article.get("publication_date", "")
        formatted_date = self._format_date(publication_date)
        
        # Extract and clean journal information
        journal = article.get("journal", "")
        journal = self._clean_text(journal)
        
        # Process authors
        authors = article.get("authors", [])
        processed_authors = self._process_authors(authors)
        
        # Extract MeSH terms
        mesh_terms = article.get("mesh_terms", [])
        processed_mesh = self._process_mesh_terms(mesh_terms)
        
        # Extract keywords
        keywords = article.get("keywords", [])
        processed_keywords = self._process_keywords(keywords)
        
        # Create transformed article
        transformed = {
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "publication_date": formatted_date,
            "journal": journal,
            "authors": processed_authors,
            "mesh_terms": processed_mesh,
            "keywords": processed_keywords,
            "source": "pubmed"
        }
        
        # Extract sentences for embedding
        sentences = self._extract_sentences(title, abstract)
        transformed["sentences"] = sentences
        
        return transformed
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Normalize Unicode characters
        text = text.replace('\u2019', "'")  # Smart quotes
        text = text.replace('\u2018', "'")
        text = text.replace('\u201c', '"')
        text = text.replace('\u201d', '"')
        text = text.replace('\u2014', '-')  # Em dash
        text = text.replace('\u2013', '-')  # En dash
        
        return text.strip()
    
    def _format_date(self, date_str: str) -> str:
        """
        Format a date string.
        
        Args:
            date_str: Date string in various formats
            
        Returns:
            Formatted date string (YYYY-MM-DD)
        """
        if not date_str:
            return ""
        
        try:
            # Try parsing the date string
            # Handle common formats from PubMed
            
            # YYYY-MM-DD
            if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
                return date_str
            
            # YYYY/MM/DD
            elif re.match(r'^\d{4}/\d{2}/\d{2}$', date_str):
                return date_str.replace('/', '-')
            
            # YYYY
            elif re.match(r'^\d{4}$', date_str):
                return f"{date_str}-01-01"
            
            # YYYY Mon DD
            elif re.match(r'^\d{4} [A-Za-z]{3} \d{1,2}$', date_str):
                dt = datetime.strptime(date_str, '%Y %b %d')
                return dt.strftime('%Y-%m-%d')
            
            # Default to empty string if we can't parse the date
            return date_str
        
        except Exception:
            return date_str
    
    def _process_authors(self, authors: List[str]) -> List[Dict[str, str]]:
        """
        Process author names.
        
        Args:
            authors: List of author name strings
            
        Returns:
            List of processed author dictionaries
        """
        processed = []
        
        for author in authors:
            if not author:
                continue
            
            # Split author names into last and first
            parts = author.split(',', 1)
            
            if len(parts) > 1:
                last_name = parts[0].strip()
                first_name = parts[1].strip()
                
                processed.append({
                    "name": author,
                    "last_name": last_name,
                    "first_name": first_name
                })
            else:
                processed.append({
                    "name": author,
                    "last_name": author,
                    "first_name": ""
                })
        
        return processed
    
    def _process_mesh_terms(self, mesh_terms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process MeSH terms.
        
        Args:
            mesh_terms: List of MeSH term dictionaries
            
        Returns:
            List of processed MeSH term dictionaries
        """
        processed = []
        
        for term in mesh_terms:
            if not term:
                continue
            
            term_text = term.get("term", "")
            ui = term.get("ui", "")
            major = term.get("major", False)
            
            if term_text and ui:
                processed.append({
                    "term": term_text,
                    "ui": ui,
                    "major": major
                })
        
        return processed
    
    def _process_keywords(self, keywords: List[str]) -> List[str]:
        """
        Process keywords.
        
        Args:
            keywords: List of keyword strings
            
        Returns:
            List of processed keywords
        """
        processed = []
        
        for keyword in keywords:
            if not keyword:
                continue
            
            # Clean and normalize keyword
            clean_keyword = self._clean_text(keyword)
            if clean_keyword:
                processed.append(clean_keyword)
        
        return processed
    
    def _extract_sentences(self, title: str, abstract: str) -> List[Dict[str, Any]]:
        """
        Extract sentences from title and abstract.
        
        Args:
            title: Article title
            abstract: Article abstract
            
        Returns:
            List of sentence dictionaries
        """
        sentences = []
        
        # Add title as a sentence
        if title:
            sentences.append({
                "text": title,
                "section": "title",
                "position": 0
            })
        
        # Split abstract into sentences
        if abstract:
            # Simple sentence splitting with regular expressions
            # In a production system, we would use a more sophisticated NLP-based approach
            abstract_sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', abstract)
            
            for i, sentence in enumerate(abstract_sentences):
                clean_sentence = self._clean_text(sentence)
                if clean_sentence:
                    sentences.append({
                        "text": clean_sentence,
                        "section": "abstract",
                        "position": i
                    })
        
        return sentences