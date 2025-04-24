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

class GitHubDocScraper(BaseScraper):
    """
    Example scraper for GitHub documentation.
    Shows how to extend BaseScraper for a different site.
    """
    
    def __init__(self, output_dir: str = "github_docs"):
        super().__init__(
            base_url="https://docs.github.com",
            output_dir=output_dir
        )
    
    def get_additional_directories(self) -> List[str]:
        """Create additional directories for GitHub docs organization"""
        return [
            "by_category",
            "by_product"
        ]
    
    def should_follow_url(self, url: str) -> bool:
        """Filter URLs to only follow GitHub documentation links"""
        if not super().should_follow_url(url):
            return False
        
        parsed_url = urlparse(url)
        # Only follow URLs under docs.github.com
        if parsed_url.netloc != 'docs.github.com':
            return False
            
        # Skip certain paths like blog, etc.
        skip_paths = ['/blog/', '/changelog/', '/releases/']
        if any(parsed_url.path.startswith(path) for path in skip_paths):
            return False
            
        return True
    
    def extract_content(self, soup: BeautifulSoup, url: str) -> Optional[Dict[str, Any]]:
        """Extract content from a GitHub documentation page"""
        # Extract title
        title_elem = soup.find('h1')
        title = title_elem.text.strip() if title_elem else "Untitled"
        
        # Extract main content area (adjust selectors for GitHub docs)
        main_content = soup.find('div', class_='article-grid-body') or soup.find('main')
        
        if not main_content:
            self.logger.warning(f"Could not find main content area for {url}")
            return None
        
        # Extract category/product information (example)
        category = ""
        product = ""
        breadcrumbs = soup.find('nav', attrs={'aria-label': 'Breadcrumb'})
        if breadcrumbs:
            crumbs = breadcrumbs.find_all('a')
            if len(crumbs) > 0:
                product = crumbs[0].text.strip()
            if len(crumbs) > 1:
                category = crumbs[1].text.strip()
        
        # Extract content as Markdown
        content_md = ""
        for elem in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'pre', 'ul', 'ol', 'table']):
            if elem.name.startswith('h'):
                level = int(elem.name[1])
                content_md += f"\n{'#' * level} {elem.text.strip()}\n\n"
            elif elem.name == 'p':
                content_md += elem.text.strip() + "\n\n"
            elif elem.name == 'pre':
                code = elem.get_text()
                # Look for language class
                code_class = elem.get('class', [])
                language = next((cls.replace('language-', '') for cls in code_class if cls.startswith('language-')), '')
                content_md += f"\n```{language}\n{code}\n```\n\n"
            elif elem.name in ['ul', 'ol']:
                for li in elem.find_all('li'):
                    content_md += f"- {li.text.strip()}\n"
                content_md += "\n"
            elif elem.name == 'table':
                # Simple table handling
                content_md += "\n"
                for row in elem.find_all('tr'):
                    cells = []
                    for cell in row.find_all(['th', 'td']):
                        cells.append(cell.text.strip())
                    content_md += "| " + " | ".join(cells) + " |\n"
                    # Add separator after header row
                    if row.find('th'):
                        content_md += "| " + " | ".join(["---"] * len(cells)) + " |\n"
                content_md += "\n"
        
        return {
            "url": url,
            "title": title,
            "product": product,
            "category": category,
            "content": content_md
        }