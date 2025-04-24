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


class BaseScraper(ABC):
    """
    Generic base scraper class that can be extended for different websites.
    Provides core functionality while allowing customization through subclassing.
    """
    
    def __init__(
        self,
        base_url: str,
        output_dir: str = "scraped_data",
        delay_range: Tuple[float, float] = (1.0, 3.0),
        max_retries: int = 3,
        timeout: int = 30,
        user_agents: Optional[List[str]] = None,
        headers: Optional[Dict[str, str]] = None,
        cookies: Optional[Dict[str, str]] = None,
        proxies: Optional[Dict[str, str]] = None,
        verify_ssl: bool = True,
        log_level: int = logging.INFO,
    ):
        """
        Initialize the base scraper with configurable options.
        
        Args:
            base_url: The base URL of the website to scrape
            output_dir: Directory to save scraped data
            delay_range: Random delay range between requests (min, max) in seconds
            max_retries: Maximum number of retries for failed requests
            timeout: Request timeout in seconds
            user_agents: List of user agents to rotate through
            headers: Additional HTTP headers
            cookies: Cookies to include in requests
            proxies: Proxy configuration for requests
            verify_ssl: Whether to verify SSL certificates
            log_level: Logging level
        """
        self.base_url = base_url
        self.output_dir = output_dir
        self.delay_range = delay_range
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Default user agents if none provided
        if user_agents is None:
            user_agents = [
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0'
            ]
        self.user_agents = user_agents
        
        # Default headers if none provided
        if headers is None:
            headers = {
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Cache-Control': 'max-age=0',
            }
        self.headers = headers
        
        # Initialize session
        self.session = requests.Session()
        self.session.headers.update(headers)
        if cookies:
            self.session.cookies.update(cookies)
        self.session.verify = verify_ssl
        if proxies:
            self.session.proxies.update(proxies)
        
        # Set up tracking
        self.visited_urls: Set[str] = set()
        self.failed_urls: Dict[str, str] = {}  # URL to error message
        self.url_to_content: Dict[str, Dict] = {}
        
        # Create output directories
        self._setup_directories()
    
    def _setup_directories(self):
        """Create necessary output directories"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Default directories - can be extended in subclasses
        default_dirs = [
            "raw_html",
            "extracted",
            "json",
            "markdown",
            "logs"
        ]
        
        for directory in default_dirs:
            os.makedirs(os.path.join(self.output_dir, directory), exist_ok=True)
        
        # Create additional directories defined by subclasses
        for directory in self.get_additional_directories():
            os.makedirs(os.path.join(self.output_dir, directory), exist_ok=True)
    
    def get_additional_directories(self) -> List[str]:
        """
        Define additional directories to create.
        Override in subclasses to add custom directories.
        """
        return []
    
    def clean_filename(self, text: str) -> str:
        """Create a valid filename from text"""
        # Replace non-alphanumeric characters with underscores
        clean = re.sub(r'[^\w\-_]', '_', text)
        # Remove multiple consecutive underscores
        clean = re.sub(r'_+', '_', clean)
        # Strip leading/trailing underscores
        return clean.strip('_')
    
    def normalize_url(self, url: str) -> str:
        """
        Normalize URL to canonical form.
        Override in subclasses for site-specific URL normalization.
        """
        # Make URL absolute
        if not url.startswith(('http://', 'https://')):
            url = urljoin(self.base_url, url)
        
        # Remove fragments
        url = url.split('#')[0]
        
        # Remove trailing slash
        if url.endswith('/') and not url.endswith('://'):
            url = url[:-1]
        
        return url
    
    def should_follow_url(self, url: str) -> bool:
        """
        Determine if a URL should be followed.
        Override in subclasses for site-specific URL filtering.
        """
        parsed_url = urlparse(url)
        base_parsed = urlparse(self.base_url)
        
        # Check if URL is within the same domain
        if parsed_url.netloc != base_parsed.netloc:
            return False
        
        # Check if URL is part of the docs/content we want to scrape
        # This is a simplified check - override for more specific logic
        if not parsed_url.path.startswith(base_parsed.path):
            return False
        
        # Skip files that are not HTML
        file_extensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.zip', '.tar', '.gz', '.jpg', '.jpeg', '.png', '.gif']
        if any(parsed_url.path.endswith(ext) for ext in file_extensions):
            return False
        
        return True
    
    def extract_links(self, soup: BeautifulSoup, current_url: str) -> List[str]:
        """
        Extract links from the page.
        Override in subclasses for site-specific link extraction.
        """
        links = []
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            absolute_url = self.normalize_url(href)
            if self.should_follow_url(absolute_url):
                links.append(absolute_url)
        return links
    
    def get_request_headers(self) -> Dict[str, str]:
        """Get headers for the next request, including random user agent"""
        headers = self.headers.copy()
        headers['User-Agent'] = random.choice(self.user_agents)
        return headers
    
    def fetch_page(self, url: str) -> Tuple[Optional[requests.Response], Optional[str]]:
        """
        Fetch a page with retries and error handling.
        Returns (response, error_message) tuple.
        """
        retries = 0
        error_msg = None
        
        while retries < self.max_retries:
            try:
                # Update headers with random user agent
                self.session.headers.update(self.get_request_headers())
                
                # Make the request
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                
                # Success, return the response
                return response, None
            
            except requests.exceptions.HTTPError as e:
                error_msg = f"HTTP Error: {e}"
                self.logger.warning(f"HTTP Error {e.response.status_code} for {url}: {e}")
                # Don't retry for client errors (4xx)
                if 400 <= e.response.status_code < 500:
                    break
            
            except requests.exceptions.ConnectionError as e:
                error_msg = f"Connection Error: {e}"
                self.logger.warning(f"Connection Error for {url}: {e}")
            
            except requests.exceptions.Timeout as e:
                error_msg = f"Timeout Error: {e}"
                self.logger.warning(f"Timeout for {url}: {e}")
            
            except requests.exceptions.RequestException as e:
                error_msg = f"Request Error: {e}"
                self.logger.warning(f"Request Error for {url}: {e}")
            
            # Increase retry count and wait before retrying
            retries += 1
            sleep_time = 2 ** retries  # Exponential backoff
            self.logger.info(f"Retrying {url} in {sleep_time} seconds... (Attempt {retries+1}/{self.max_retries})")
            time.sleep(sleep_time)
        
        # All retries failed
        self.logger.error(f"Failed to fetch {url} after {self.max_retries} attempts")
        return None, error_msg
    
    def scrape_page(self, url: str) -> List[str]:
        """
        Scrape a single page and return new URLs to follow.
        Returns list of new URLs found on the page.
        """
        if url in self.visited_urls:
            return []
        
        self.visited_urls.add(url)
        normalized_url = self.normalize_url(url)
        
        self.logger.info(f"Scraping: {normalized_url}")
        
        # Fetch the page
        response, error = self.fetch_page(normalized_url)
        if error:
            self.failed_urls[normalized_url] = error
            return []
        
        # Random delay to avoid overloading the server
        delay = random.uniform(*self.delay_range)
        time.sleep(delay)
        
        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Save raw HTML
        self.save_raw_html(normalized_url, response.text)
        
        # Extract content
        content = self.extract_content(soup, normalized_url)
        if content:
            self.url_to_content[normalized_url] = content
            self.save_content(normalized_url, content)
        
        # Extract links to follow
        new_links = self.extract_links(soup, normalized_url)
        return new_links
    
    def save_raw_html(self, url: str, html: str):
        """Save the raw HTML of a page"""
        parsed_url = urlparse(url)
        filename = self.clean_filename(parsed_url.path)
        if not filename:
            filename = "index"
        
        filepath = os.path.join(self.output_dir, "raw_html", f"{filename}.html")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
    
    @abstractmethod
    def extract_content(self, soup: BeautifulSoup, url: str) -> Optional[Dict[str, Any]]:
        """
        Extract structured content from a page.
        Must be implemented by subclasses.
        
        Args:
            soup: BeautifulSoup object of the page
            url: URL of the page
        
        Returns:
            Dictionary of extracted content or None if extraction failed
        """
        pass
    
    def save_content(self, url: str, content: Dict[str, Any]):
        """
        Save the extracted content in various formats.
        Override in subclasses for custom saving logic.
        """
        parsed_url = urlparse(url)
        path_parts = parsed_url.path.strip('/').split('/')
        filename = self.clean_filename('_'.join(path_parts)) or "index"
        
        # Save as JSON
        json_path = os.path.join(self.output_dir, "json", f"{filename}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(content, f, indent=2, ensure_ascii=False)
        
        # Save as Markdown (basic implementation)
        md_path = os.path.join(self.output_dir, "markdown", f"{filename}.md")
        with open(md_path, 'w', encoding='utf-8') as f:
            if 'title' in content:
                f.write(f"# {content['title']}\n\n")
            f.write(f"Source: {url}\n\n")
            
            # Write other content (this is very basic and should be overridden)
            if 'content' in content and isinstance(content['content'], str):
                f.write(content['content'])
    
    def run(self, start_url: Optional[str] = None, max_pages: Optional[int] = None):
        """
        Main method to run the scraper starting from a URL.
        
        Args:
            start_url: URL to start scraping from (defaults to base_url)
            max_pages: Maximum number of pages to scrape (None for unlimited)
        """
        if start_url is None:
            start_url = self.base_url
        
        self.logger.info(f"Starting scrape from {start_url}")
        
        urls_to_scrape = [start_url]
        pages_scraped = 0
        
        while urls_to_scrape and (max_pages is None or pages_scraped < max_pages):
            current_url = urls_to_scrape.pop(0)
            
            # Skip if already visited
            if current_url in self.visited_urls:
                continue
            
            # Scrape the page and get new links
            new_links = self.scrape_page(current_url)
            pages_scraped += 1
            
            # Add new links to the queue
            for link in new_links:
                if link not in self.visited_urls and link not in urls_to_scrape:
                    urls_to_scrape.append(link)
            
            self.logger.info(f"Pages scraped: {pages_scraped}, Pages in queue: {len(urls_to_scrape)}")
        
        self.logger.info(f"Scraping completed. Scraped {len(self.visited_urls)} pages.")
        self.logger.info(f"Failed pages: {len(self.failed_urls)}")
        
        # Post-processing
        self.post_process()
    
    def post_process(self):
        """
        Perform post-processing after all pages are scraped.
        Override in subclasses for custom post-processing.
        """
        # Create a summary report
        self.create_summary_report()
    
    def create_summary_report(self):
        """Create a summary report of the scraping session"""
        report_path = os.path.join(self.output_dir, "scrape_report.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# Scrape Report\n\n")
            f.write(f"Base URL: {self.base_url}\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"## Statistics\n\n")
            f.write(f"- Pages scraped: {len(self.visited_urls)}\n")
            f.write(f"- Failed pages: {len(self.failed_urls)}\n\n")
            
            if self.failed_urls:
                f.write(f"## Failed URLs\n\n")
                for url, error in self.failed_urls.items():
                    f.write(f"- [{url}]({url}): {error}\n")
        
        self.logger.info(f"Created summary report at {report_path}")


class OpenAIDocScraper(BaseScraper):
    """
    Specialized scraper for OpenAI documentation.
    Extends the BaseScraper with OpenAI-specific extraction logic.
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
                # For h4 tags within a subsection, we can safely use a local function
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
    
    def save_content(self, url: str, content: Dict[str, Any]):
        """Save OpenAI documentation content in various formats"""
        # Call parent method to save individual files
        super().save_content(url, content)
        
        # Get section and create filename
        section = content.get('section', 'general')
        parsed_url = urlparse(url)
        path_parts = parsed_url.path.strip('/').split('/')
        filename = self.clean_filename('_'.join(path_parts)) or "index"
        
        # Save by page with more structured Markdown
        md_path = os.path.join(self.output_dir, "by_page", f"{filename}.md")
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(f"# {content['title']}\n\n")
            f.write(f"Source: {url}\n")
            f.write(f"Section: {section}\n\n")
            
            # Write sections
            for section_data in content['sections']:
                f.write(f"## {section_data['title']}\n\n")
                f.write(section_data['content'].strip() + "\n\n")
                
                for subsection in section_data['subsections']:
                    f.write(f"### {subsection['title']}\n\n")
                    f.write(subsection['content'].strip() + "\n\n")
            
            # Write code samples
            if content['code_samples']:
                f.write("## Code Samples\n\n")
                for i, sample in enumerate(content['code_samples']):
                    f.write(f"### Sample {i+1} ({sample['language']})\n\n")
                    f.write(f"```{sample['language']}\n{sample['code']}\n```\n\n")
    
    def post_process(self):
        """Post-process to generate section-based files and combined docs"""
        super().post_process()
        
        # Create section-based files
        self._create_section_files()
        
        # Create combined documentation file
        self._create_combined_file()
    
    def _create_section_files(self):
        """Create aggregated files for each section"""
        for section, pages in self.section_content.items():
            # Create JSON file
            json_path = os.path.join(self.output_dir, "by_section", f"{section}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(pages, f, indent=2, ensure_ascii=False)
            
            # Create Markdown file
            md_path = os.path.join(self.output_dir, "by_section", f"{section}.md")
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(f"# OpenAI {section.title()} Documentation\n\n")
                f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"## Table of Contents\n\n")
                
                # Create table of contents
                for page in pages:
                    page_title = page['title']
                    f.write(f"- [{page_title}](#{self.clean_filename(page_title.lower())})\n")
                
                f.write("\n---\n\n")
                
                # Write each page
                for page in pages:
                    page_title = page['title']
                    f.write(f"<a id='{self.clean_filename(page_title.lower())}'></a>\n")
                    f.write(f"# {page_title}\n\n")
                    f.write(f"Source: {page['url']}\n\n")
                    
                    # Write sections
                    for section_data in page['sections']:
                        f.write(f"## {section_data['title']}\n\n")
                        f.write(section_data['content'].strip() + "\n\n")
                        
                        for subsection in section_data['subsections']:
                            f.write(f"### {subsection['title']}\n\n")
                            f.write(subsection['content'].strip() + "\n\n")
                    
                    # Write code samples
                    if page['code_samples']:
                        f.write("## Code Samples\n\n")
                        for i, sample in enumerate(page['code_samples']):
                            f.write(f"### Sample {i+1} ({sample['language']})\n\n")
                            f.write(f"```{sample['language']}\n{sample['code']}\n```\n\n")
                    
                    f.write("\n---\n\n")
    
    def _create_combined_file(self):
        """Create a single combined documentation file"""
        combined_path = os.path.join(self.output_dir, "combined", "openai_complete_docs.md")
        
        with open(combined_path, 'w', encoding='utf-8') as f:
            f.write(f"# OpenAI Complete Documentation\n\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"## Table of Contents\n\n")
            
            # Create table of contents by section
            for section in sorted(self.section_content.keys()):
                f.write(f"- [{section.title()}](#{section})\n")
                for page in self.section_content[section]:
                    page_title = page['title']
                    f.write(f"  - [{page_title}](#{self.clean_filename(page_title.lower())})\n")
            
            f.write("\n---\n\n")
            
            # Write each section
            for section in sorted(self.section_content.keys()):
                f.write(f"<a id='{section}'></a>\n")
                f.write(f"# {section.title()}\n\n")
                
                # Write each page in the section
                for page in self.section_content[section]:
                    page_title = page['title']
                    f.write(f"<a id='{self.clean_filename(page_title.lower())}'></a>\n")
                    f.write(f"## {page_title}\n\n")
                    f.write(f"Source: {page['url']}\n\n")
                    
                    # Write sections
                    for section_data in page['sections']:
                        f.write(f"### {section_data['title']}\n\n")
                        f.write(section_data['content'].strip() + "\n\n")
                        
                        for subsection in section_data['subsections']:
                            f.write(f"#### {subsection['title']}\n\n")
                            f.write(subsection['content'].strip() + "\n\n")
                    
                    # Write code samples
                    if page['code_samples']:
                        f.write("### Code Samples\n\n")
                        for i, sample in enumerate(page['code_samples']):
                            f.write(f"#### Sample {i+1} ({sample['language']})\n\n")
                            f.write(f"```{sample['language']}\n{sample['code']}\n```\n\n")
                    
                    f.write("\n---\n\n")
        
        self.logger.info(f"Created combined documentation at {combined_path}")


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


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Web Documentation Scraper')
    parser.add_argument('--site', type=str, default='openai', 
                        choices=['openai', 'github', 'custom'],
                        help='Site to scrape (openai, github, or custom)')
    parser.add_argument('--url', type=str, 
                        help='Custom base URL for scraping (required for custom site)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for scraped content')
    parser.add_argument('--max-pages', type=int, default=None,
                        help='Maximum number of pages to scrape (None for unlimited)')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    
    args = parser.parse_args()
    
    # Set up logging level
    log_level = getattr(logging, args.log_level)
    
    # Initialize the appropriate scraper
    if args.site == 'openai':
        output_dir = args.output or 'openai_docs'
        scraper = OpenAIDocScraper(output_dir=output_dir)
        start_url = args.url or "https://platform.openai.com/docs"
    
    elif args.site == 'github':
        output_dir = args.output or 'github_docs'
        scraper = GitHubDocScraper(output_dir=output_dir)
        start_url = args.url or "https://docs.github.com"
    
    elif args.site == 'custom':
        if not args.url:
            parser.error("--url is required for custom site")
        output_dir = args.output or 'custom_docs'
        scraper = CustomDocScraper(base_url=args.url, output_dir=output_dir)
        start_url = args.url
    
    # Configure logger
    scraper.logger.setLevel(log_level)
    
    # Run the scraper
    print(f"Starting scraper for {args.site} site")
    print(f"Base URL: {start_url}")
    print(f"Output directory: {output_dir}")
    print(f"Maximum pages: {args.max_pages or 'Unlimited'}")
    
    try:
        scraper.run(start_url=start_url, max_pages=args.max_pages)
        print(f"Scraping completed. Results saved to {output_dir}")
    except KeyboardInterrupt:
        print("\nScraping interrupted by user.")
    except Exception as e:
        print(f"Error during scraping: {e}")
        import traceback
        traceback.print_exc()