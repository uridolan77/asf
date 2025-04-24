import requests
from bs4 import BeautifulSoup
import json
import os
import time
import re
from urllib.parse import urljoin, urlparse
from collections import defaultdict
import logging
import random
from abc import ABC, abstractmethod
from typing import List, Dict, Set, Optional, Any, Tuple, Callable
from asf.medical.scripts.scraping.base import BaseScraper

# Example of creating a custom scraper for a specific website
class CustomDocScraper(BaseScraper):
    """
    Template for creating a custom scraper for any website.
    Override the necessary methods to adapt to the site structure.
    """
    
    def __init__(self, base_url, output_dir="scraped_docs"):
        super().__init__(
            base_url=base_url,
            output_dir=output_dir
        )
    
    def get_additional_directories(self) -> List[str]:
        """Define custom output directories"""
        return [
            "custom_directory1",
            "custom_directory2"
        ]
    
    def should_follow_url(self, url: str) -> bool:
        """Define custom URL filtering logic"""
        # Call parent method first
        if not super().should_follow_url(url):
            return False
        
        # Add custom logic for your site
        parsed_url = urlparse(url)
        
        # Example: only follow certain paths
        allowed_paths = ['/docs/', '/guide/', '/tutorial/']
        if not any(parsed_url.path.startswith(path) for path in allowed_paths):
            return False
            
        return True
    
    def extract_content(self, soup: BeautifulSoup, url: str) -> Optional[Dict[str, Any]]:
        """
        Extract content from a specific website.
        Customize selectors and structure for your target site.
        """
        # Extract title - adjust selector for your site
        title_elem = soup.find('h1') or soup.find('title')
        title = title_elem.text.strip() if title_elem else "Untitled"
        
        # Find main content - adjust selector for your site
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
        
        if not main_content:
            self.logger.warning(f"Could not find main content area for {url}")
            return None
        
        # Extract text content
        text_content = ""
        for elem in main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'ul', 'ol']):
            if elem.name.startswith('h'):
                level = int(elem.name[1])
                text_content += f"\n{'#' * level} {elem.text.strip()}\n\n"
            elif elem.name == 'p':
                text_content += elem.text.strip() + "\n\n"
            elif elem.name in ['ul', 'ol']:
                for li in elem.find_all('li'):
                    text_content += f"- {li.text.strip()}\n"
                text_content += "\n"
        
        # Extract any custom fields relevant to your site
        # Example: Extract metadata, dates, authors, etc.
        metadata = {}
        author_elem = soup.find('meta', attrs={'name': 'author'})
        if author_elem and 'content' in author_elem.attrs:
            metadata['author'] = author_elem['content']
        
        date_elem = soup.find('meta', attrs={'name': 'date'})
        if date_elem and 'content' in date_elem.attrs:
            metadata['date'] = date_elem['content']
        
        # Extract code samples
        code_samples = []
        for code_elem in main_content.find_all('pre'):
            code_text = code_elem.get_text()
            code_class = code_elem.get('class', [])
            language = next((cls.replace('language-', '') for cls in code_class if cls.startswith('language-')), 'plaintext')
            code_samples.append({
                "language": language,
                "code": code_text
            })
        
        return {
            "url": url,
            "title": title,
            "content": text_content,
            "metadata": metadata,
            "code_samples": code_samples
        }
