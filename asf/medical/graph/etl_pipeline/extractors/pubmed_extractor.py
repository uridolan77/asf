"""
PubMed Data Extractor

This module extracts article data from PubMed using the NCBI E-utilities API.
It supports searching by terms, date ranges, and other criteria.
"""

import os
import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import aiohttp
import pandas as pd
from pathlib import Path

logger = logging.getLogger("biomedical_etl.extractors.pubmed")

class PubmedExtractor:
    """
    Extractor for PubMed data using the NCBI E-utilities API.
    
    This class provides methods for searching and retrieving article data
    from PubMed. It supports batch processing, caching, and error handling.
    """
    
    def __init__(
        self,
        email: Optional[str] = None,
        api_key: Optional[str] = None,
        batch_size: int = 100,
        max_articles: int = 10000,
        cache_dir: str = "./cache/pubmed"
    ):
        """
        Initialize the PubMed extractor.
        
        Args:
            email: Email address for the NCBI API
            api_key: API key for the NCBI API (optional, increases rate limits)
            batch_size: Number of articles to fetch in each batch
            max_articles: Maximum number of articles to fetch in total
            cache_dir: Directory to store cache files
        """
        self.email = email
        self.api_key = api_key
        self.batch_size = batch_size
        self.max_articles = max_articles
        self.cache_dir = cache_dir
        
        # Base URLs for NCBI E-utilities
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info(f"Initialized PubMed extractor with batch_size={batch_size}, max_articles={max_articles}")
    
    async def extract(
        self,
        search_terms: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract article data from PubMed.
        
        Args:
            search_terms: List of search terms to use for querying PubMed
            start_date: Start date for search (format: YYYY-MM-DD)
            end_date: End date for search (format: YYYY-MM-DD)
            
        Returns:
            List of article dictionaries
        """
        logger.info(f"Extracting PubMed articles for {len(search_terms)} search terms")
        
        all_articles = []
        
        # Create a session for connection pooling
        async with aiohttp.ClientSession() as session:
            for term in search_terms:
                try:
                    # Generate cache key based on search parameters
                    cache_key = self._generate_cache_key(term, start_date, end_date)
                    cache_path = Path(self.cache_dir) / f"{cache_key}.json"
                    
                    # Check if we have cached results
                    if cache_path.exists():
                        logger.info(f"Loading cached results for '{term}'")
                        articles = self._load_from_cache(cache_path)
                    else:
                        logger.info(f"Searching PubMed for '{term}'")
                        articles = await self._search_and_fetch(session, term, start_date, end_date)
                        
                        # Cache the results
                        self._save_to_cache(articles, cache_path)
                    
                    logger.info(f"Found {len(articles)} articles for '{term}'")
                    all_articles.extend(articles)
                    
                except Exception as e:
                    logger.error(f"Error extracting articles for '{term}': {str(e)}")
        
        # Deduplicate articles based on PMID
        deduplicated_articles = self._deduplicate_articles(all_articles)
        
        logger.info(f"Extracted {len(deduplicated_articles)} unique PubMed articles")
        return deduplicated_articles
    
    async def _search_and_fetch(
        self,
        session: aiohttp.ClientSession,
        term: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search PubMed and fetch article details.
        
        Args:
            session: aiohttp ClientSession
            term: Search term
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            List of article dictionaries
        """
        # Step 1: Search PubMed to get PMIDs
        pmids = await self._search_pubmed(session, term, start_date, end_date)
        
        # Step 2: Fetch article details in batches
        articles = []
        for i in range(0, min(len(pmids), self.max_articles), self.batch_size):
            batch_pmids = pmids[i:i+self.batch_size]
            logger.info(f"Fetching details for {len(batch_pmids)} articles (batch {i//self.batch_size + 1})")
            
            batch_articles = await self._fetch_article_details(session, batch_pmids)
            articles.extend(batch_articles)
            
            # Add a small delay to respect NCBI rate limits
            await asyncio.sleep(0.34)  # ~3 requests per second
        
        return articles
    
    async def _search_pubmed(
        self,
        session: aiohttp.ClientSession,
        term: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[str]:
        """
        Search PubMed for articles matching the criteria.
        
        Args:
            session: aiohttp ClientSession
            term: Search term
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            List of PMIDs
        """
        # Build search query
        query = term
        
        # Add date range if provided
        if start_date and end_date:
            query += f" AND ({start_date}[PDAT]:{end_date}[PDAT])"
        elif start_date:
            query += f" AND {start_date}[PDAT]:3000[PDAT]"
        elif end_date:
            query += f" AND 1000[PDAT]:{end_date}[PDAT]"
        
        # Add parameters for the ESearch API
        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": self.max_articles,
            "usehistory": "y",
            "tool": "BiomedicalETL",
            "email": self.email
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
        
        # Make the request
        async with session.get(f"{self.base_url}/esearch.fcgi", params=params) as response:
            response.raise_for_status()
            data = await response.json()
        
        # Extract PMIDs from the response
        pmids = data.get("esearchresult", {}).get("idlist", [])
        return pmids
    
    async def _fetch_article_details(
        self,
        session: aiohttp.ClientSession,
        pmids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Fetch details for a batch of articles.
        
        Args:
            session: aiohttp ClientSession
            pmids: List of PMIDs to fetch
            
        Returns:
            List of article dictionaries
        """
        if not pmids:
            return []
        
        # Build parameters for the EFetch API
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "tool": "BiomedicalETL",
            "email": self.email
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
        
        # Make the request
        async with session.get(f"{self.base_url}/efetch.fcgi", params=params) as response:
            response.raise_for_status()
            xml_data = await response.text()
        
        # Parse the XML response
        articles = self._parse_pubmed_xml(xml_data)
        return articles
    
    def _parse_pubmed_xml(self, xml_data: str) -> List[Dict[str, Any]]:
        """
        Parse PubMed XML response into article dictionaries.
        
        Args:
            xml_data: XML response from PubMed
            
        Returns:
            List of article dictionaries
        """
        # In a real implementation, this would use a proper XML parser
        # For simplicity, we'll use a placeholder implementation that extracts basic information
        
        # Note: In a production system, use ElementTree, lxml, or a specialized PubMed XML parser
        
        from bs4 import BeautifulSoup
        
        articles = []
        soup = BeautifulSoup(xml_data, 'xml')
        
        # Find all PubmedArticle elements
        pubmed_articles = soup.find_all('PubmedArticle')
        
        for article_elem in pubmed_articles:
            try:
                # Extract basic metadata
                pmid_elem = article_elem.find('PMID')
                pmid = pmid_elem.text if pmid_elem else None
                
                article_data = {"pmid": pmid}
                
                # Title
                title_elem = article_elem.find('ArticleTitle')
                article_data["title"] = title_elem.text if title_elem else ""
                
                # Abstract
                abstract_texts = article_elem.find_all('AbstractText')
                abstract = " ".join([text.text for text in abstract_texts]) if abstract_texts else ""
                article_data["abstract"] = abstract
                
                # Journal
                journal_elem = article_elem.find('Journal')
                if journal_elem:
                    journal_title_elem = journal_elem.find('Title')
                    article_data["journal"] = journal_title_elem.text if journal_title_elem else ""
                
                # Publication date
                pub_date_elem = article_elem.find('PubDate')
                if pub_date_elem:
                    year_elem = pub_date_elem.find('Year')
                    month_elem = pub_date_elem.find('Month')
                    day_elem = pub_date_elem.find('Day')
                    
                    year = year_elem.text if year_elem else ""
                    month = month_elem.text if month_elem else "01"
                    day = day_elem.text if day_elem else "01"
                    
                    # Handle text month names
                    if month.isalpha():
                        month_map = {"Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04", "May": "05", "Jun": "06",
                                    "Jul": "07", "Aug": "08", "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12"}
                        month = month_map.get(month[:3], "01")
                    
                    article_data["publication_date"] = f"{year}-{month}-{day}"
                
                # Authors
                author_list = article_elem.find('AuthorList')
                authors = []
                
                if author_list:
                    author_elems = author_list.find_all('Author')
                    for author_elem in author_elems:
                        lastname_elem = author_elem.find('LastName')
                        forename_elem = author_elem.find('ForeName')
                        
                        if lastname_elem:
                            lastname = lastname_elem.text
                            forename = forename_elem.text if forename_elem else ""
                            
                            if lastname and forename:
                                authors.append(f"{lastname}, {forename}")
                            elif lastname:
                                authors.append(lastname)
                
                article_data["authors"] = authors
                
                # MeSH terms
                mesh_terms = []
                mesh_headings = article_elem.find('MeshHeadingList')
                
                if mesh_headings:
                    mesh_elem_list = mesh_headings.find_all('MeshHeading')
                    for mesh_elem in mesh_elem_list:
                        descriptor = mesh_elem.find('DescriptorName')
                        if descriptor:
                            term = descriptor.text
                            ui = descriptor.get('UI', '')
                            major = descriptor.get('MajorTopicYN', 'N') == 'Y'
                            
                            mesh_terms.append({
                                "term": term,
                                "ui": ui,
                                "major": major
                            })
                
                article_data["mesh_terms"] = mesh_terms
                
                # Keywords
                keyword_list = article_elem.find('KeywordList')
                keywords = []
                
                if keyword_list:
                    keyword_elems = keyword_list.find_all('Keyword')
                    for keyword_elem in keyword_elems:
                        keywords.append(keyword_elem.text)
                
                article_data["keywords"] = keywords
                
                articles.append(article_data)
                
            except Exception as e:
                logger.error(f"Error parsing article: {str(e)}")
        
        return articles
    
    def _generate_cache_key(self, term: str, start_date: Optional[str], end_date: Optional[str]) -> str:
        """
        Generate a cache key based on search parameters.
        
        Args:
            term: Search term
            start_date: Start date
            end_date: End date
            
        Returns:
            Cache key string
        """
        import hashlib
        
        # Create a string with all parameters
        key_str = f"term={term}"
        
        if start_date:
            key_str += f"&start_date={start_date}"
        
        if end_date:
            key_str += f"&end_date={end_date}"
        
        # Generate a hash of the parameter string
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _save_to_cache(self, articles: List[Dict[str, Any]], cache_path: Path) -> None:
        """
        Save articles to cache.
        
        Args:
            articles: List of article dictionaries
            cache_path: Path to save the cache file
        """
        with open(cache_path, 'w') as f:
            json.dump(articles, f)
    
    def _load_from_cache(self, cache_path: Path) -> List[Dict[str, Any]]:
        """
        Load articles from cache.
        
        Args:
            cache_path: Path to the cache file
            
        Returns:
            List of article dictionaries
        """
        with open(cache_path, 'r') as f:
            return json.load(f)
    
    def _deduplicate_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate articles based on PMID.
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            Deduplicated list of article dictionaries
        """
        seen_pmids = set()
        deduplicated = []
        
        for article in articles:
            pmid = article.get("pmid")
            if pmid and pmid not in seen_pmids:
                seen_pmids.add(pmid)
                deduplicated.append(article)
        
        return deduplicated