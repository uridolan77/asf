"""
OpenAI documentation scraper module.
"""
import logging
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from collections import defaultdict
from typing import Optional, Dict, Any, List, Callable

# Import your BaseScraper class
from asf.medical.scripts.scraping.base import BaseScraper

class OpenAIDocScraper(BaseScraper):
    """
    Specialized scraper for OpenAI documentation.
    """
    
    def __init__(self, output_dir: str = "openai_docs"):
        super().__init__(
            base_url="https://platform.openai.com/docs",
            output_dir=output_dir
        )
        self.section_content = defaultdict(list)
    
    def get_additional_directories(self) -> List[str]:
        """Create additional directories for OpenAI docs organization"""
        return [
            "by_section",
            "by_api",
            "combined"
        ]
    
    def should_follow_url(self, url: str) -> bool:
        """Filter URLs to only follow OpenAI documentation links"""
        # First check with the parent method
        if not super().should_follow_url(url):
            return False
        
        # Only follow URLs under the docs path
        parsed_url = urlparse(url)
        return parsed_url.path.startswith('/docs')
    
    def get_section_path(self, url: str) -> str:
        """Extract the section path from a URL"""
        parsed_url = urlparse(url)
        path = parsed_url.path
        
        # Remove /docs/ prefix
        if path.startswith('/docs/'):
            path = path[6:]
        
        # Get the first directory level as the section
        parts = path.split('/')
        if parts and parts[0]:
            return parts[0]
        return "general"
    
    def extract_content(self, soup: BeautifulSoup, url: str) -> Optional[Dict[str, Any]]:
        """Extract content from an OpenAI documentation page"""
        # Extract title
        title_elem = soup.find('h1')
        title = title_elem.text.strip() if title_elem else "Untitled"
        
        # Extract main content area
        main_content = soup.find('main') or soup.find('div', class_='prose') or soup.find('article')
        
        if not main_content:
            self.logger.warning(f"Could not find main content area for {url}")
            return None
        
        # Extract headings and section content
        sections = []
        current_section = {"title": title, "content": "", "subsections": []}
        current_subsection = None
        
        for elem in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'pre', 'ul', 'ol', 'table']):
            if elem.name in ['h1', 'h2']:
                # New major section
                if current_section["title"] != elem.text.strip():
                    if current_section["content"] or current_section["subsections"]:
                        sections.append(current_section)
                    current_section = {"title": elem.text.strip(), "content": "", "subsections": []}
                    current_subsection = None
            elif elem.name == 'h3':
                # New subsection
                if current_subsection:
                    current_section["subsections"].append(current_subsection)
                current_subsection = {"title": elem.text.strip(), "content": ""}
            elif elem.name == 'h4' and current_subsection:
                # For h4 tags within a subsection, we can safely use a function
                def append_h4_to_subsection():
                    new_content = current_subsection["content"] + f"\n### {elem.text.strip()}\n"
                    current_subsection["content"] = new_content
                append_h4_to_subsection()
            elif current_subsection:
                # Add content to current subsection using a function
                def append_to_subsection(content_to_add):
                    new_content = current_subsection["content"] + content_to_add
                    current_subsection["content"] = new_content
                self._process_element(elem, append_to_subsection)
            else:
                # Add content directly to the section using a function
                def append_to_section(content_to_add):
                    new_content = current_section["content"] + content_to_add
                    current_section["content"] = new_content
                self._process_element(elem, append_to_section)
        
        # Add the last subsection if it exists
        if current_subsection:
            current_section["subsections"].append(current_subsection)
        
        # Add the last section if it's not empty
        if current_section["content"] or current_section["subsections"]:
            sections.append(current_section)
        
        # Extract code samples
        code_samples = []
        for code_elem in main_content.find_all('pre'):
            code_text = code_elem.get_text()
            # Try to detect language based on class or other attributes
            code_class = code_elem.get('class', [])
            language = next((cls.replace('language-', '') for cls in code_class if cls.startswith('language-')), 'plaintext')
            code_samples.append({"language": language, "code": code_text})
        
        # Store the section info for later aggregation
        section = self.get_section_path(url)
        self.section_content[section].append({
            "url": url,
            "title": title,
            "sections": sections,
            "code_samples": code_samples
        })
        
        return {
            "url": url,
            "title": title,
            "section": section,
            "sections": sections,
            "code_samples": code_samples,
        }
    
    def _process_element(self, elem, append_func):
        """Process a content element and append it using the provided function"""
        if elem.name == 'pre':
            code = elem.get_text()
            append_func(f"\n```\n{code}\n```\n")
        elif elem.name in ['ul', 'ol']:
            items = [f"- {li.get_text().strip()}" for li in elem.find_all('li')]
            append_func("\n" + "\n".join(items) + "\n")
        elif elem.name == 'table':
            # Simplified table extraction
            table_text = "| "
            headers = [th.get_text().strip() for th in elem.find_all('th')]
            if headers:
                table_text += " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n"
                
                for row in elem.find_all('tr')[1:]:  # Skip header row
                    cells = [td.get_text().strip() for td in row.find_all('td')]
                    if cells:
                        table_text += "| " + " | ".join(cells) + " |\n"
            else:
                # Handle tables without header rows
                for row in elem.find_all('tr'):
                    cells = [td.get_text().strip() for td in row.find_all('td')]
                    if cells:
                        table_text += "| " + " | ".join(cells) + " |\n"
            
            append_func("\n" + table_text + "\n")
        else:
            append_func("\n" + elem.get_text().strip())