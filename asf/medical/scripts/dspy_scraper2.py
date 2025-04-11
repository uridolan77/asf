"""
DSPy Documentation Scraper

A tool for extracting and organizing documentation from dspy.ai.
Features:
- Concurrent requests for faster scraping
- Content extraction with proper HTML parsing
- Markdown conversion for clean output
- Local caching to avoid redundant requests
- Command-line interface with customization options
- Sitemap parsing for comprehensive page discovery
- Robust retry mechanism for handling connection issues
- Optional proxy support to avoid IP blocking
"""

import os
import re
import json
import time
import argparse
import requests
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn

# Initialize Rich console for pretty output
console = Console()

class DSPyScraper:
    def __init__(self, base_url="https://dspy.ai/", max_workers=5, cache_dir=".dspy_cache", 
                 use_proxy=False, proxy_api_key=None, max_retries=3, output_dir="dspy_docs", 
                 starting_paths=None):
        """Initialize the DSPy documentation scraper.
        
        Args:
            base_url: Base URL of the DSPy documentation
            max_workers: Maximum number of concurrent workers
            cache_dir: Directory to store cached content
            use_proxy: Whether to use a proxy service (ScrapingBee) to avoid blocking
            proxy_api_key: API key for ScrapingBee if use_proxy is True
            max_retries: Maximum number of retry attempts for failed requests
            output_dir: Directory to save documentation
            starting_paths: List of paths to start crawling from (e.g., ["learn", "tutorials", "api"])
        """
        self.base_url = base_url
        self.max_workers = max_workers
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.use_proxy = use_proxy
        self.proxy_api_key = proxy_api_key
        self.max_retries = max_retries
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Default starting paths based on DSPy's actual structure
        self.starting_paths = starting_paths or ["learn", "tutorials", "api"]
        
        # Create directories for individual document types
        self.markdown_dir = self.output_dir / "markdown"
        self.markdown_dir.mkdir(exist_ok=True)
        self.json_dir = self.output_dir / "json"
        self.json_dir.mkdir(exist_ok=True)
        
        # Progress tracking file
        self.progress_file = self.output_dir / "progress.json"
        
        # Headers to mimic a browser request
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0",
        }
        
        # Load progress if it exists
        self.visited_urls = set()
        self.doc_pages = {}
        self._load_progress()
        
        # Session for connection pooling
        self.session = requests.Session()
        
        # Initialize proxy client if needed
        self.proxy_client = None
        if self.use_proxy and self.proxy_api_key:
            try:
                from scrapingbee import ScrapingBeeClient
                self.proxy_client = ScrapingBeeClient(api_key=self.proxy_api_key)
                console.print("[bold green]ScrapingBee proxy initialized successfully[/bold green]")
            except ImportError:
                console.print("[bold yellow]ScrapingBee not installed. Run 'pip install scrapingbee' to use proxy features.[/bold yellow]")
                self.use_proxy = False
        
    @retry(
        stop=stop_after_attempt(3),  # Stop after 3 attempts
        wait=wait_exponential(multiplier=1, min=4, max=10),  # Wait between 4-10 seconds, doubling each time
        retry=retry_if_exception_type(requests.RequestException)  # Only retry on request exceptions
    )
    def _fetch_with_retry(self, url):
        """Fetch a page with retry logic."""
        if self.use_proxy and self.proxy_client:
            # Use ScrapingBee proxy if enabled
            response = self.proxy_client.get(
                url,
                params={
                    'premium_proxy': True,
                    'country_code': 'us',
                    'block_resources': True,
                    'device': 'desktop',
                }
            )
        else:
            # Use regular session otherwise
            response = self.session.get(url, headers=self.headers, timeout=10)
            
        response.raise_for_status()
        return response
    
    def get_page(self, url):
        """Fetch a page and return its content."""
        cache_file = self.cache_dir / f"{self._url_to_filename(url)}.html"
        
        # Check if page is cached
        if cache_file.exists():
            return cache_file.read_text(encoding="utf-8")
        
        # Add small delay to be respectful to the server (only for direct requests)
        if not self.use_proxy:
            time.sleep(0.5)
        
        try:
            response = self._fetch_with_retry(url)
            
            # Cache the response
            cache_file.write_text(response.text, encoding="utf-8")
            return response.text
        except requests.RequestException as e:
            console.print(f"[bold red]Error fetching {url}: {str(e)}[/bold red]")
            return None
    
    def _url_to_filename(self, url):
        """Convert URL to a valid filename."""
        parsed = urlparse(url)
        path = parsed.path.strip("/").replace("/", "_")
        if not path:
            path = "index"
        return f"{parsed.netloc}_{path}"
    
    def is_documentation_page(self, url, html):
        """Check if a page is a documentation page."""
        if not html:
            return False
            
        soup = BeautifulSoup(html, "html.parser")
        
        # URL-based heuristics
        url_lower = url.lower()
        path = urlparse(url).path.lower()
        
        # Check for DSPy-specific paths that are likely to be documentation
        dspy_doc_paths = ["/learn/", "/tutorials/", "/api/", "/examples/", "/guide/"]
        is_doc_path = any(doc_path in path for doc_path in dspy_doc_paths)
        
        # Content-based heuristics
        has_headings = len(soup.select("h1, h2, h3")) > 0
        has_paragraphs = len(soup.select("p")) > 3  # Docs usually have several paragraphs
        has_code = len(soup.select("pre, code")) > 0  # Docs often include code examples
        
        # Check for content that suggests a documentation page (DSPy-specific checks)
        dspy_terms = ["dspy", "language model", "llm", "prompt", "module", "pipeline", "teleprompter"]
        text_lower = soup.get_text().lower()
        has_dspy_terms = any(term in text_lower for term in dspy_terms)
        
        # Combination of heuristics to determine if it's a documentation page
        is_doc = (
            # Either URL path suggests documentation
            is_doc_path or 
            # Or a combination of content features suggests documentation
            (has_headings and has_paragraphs and (has_code or has_dspy_terms))
        )
        
        return is_doc
    
    def extract_links(self, url, html):
        """Extract all links from a page that belong to the same domain."""
        if not html:
            return []
            
        soup = BeautifulSoup(html, "html.parser")
        base_domain = urlparse(self.base_url).netloc
        
        links = []
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            full_url = urljoin(url, href)
            
            # Only include links from the same domain
            if urlparse(full_url).netloc == base_domain:
                links.append(full_url)
                
        return list(set(links))
    
    def extract_content(self, url, html):
        """Extract and structure documentation content."""
        if not html:
            return None
            
        soup = BeautifulSoup(html, "html.parser")
        
        # Remove non-content elements
        for element in soup.select('script, style, nav, header, footer, .sidebar, .navigation'):
            element.decompose()
        
        # Find the main content area
        content_selectors = [
            "article", ".content", ".documentation", ".doc-content", 
            "main", "#content", "#main-content", ".markdown-body"
        ]
        
        content = None
        for selector in content_selectors:
            content = soup.select_one(selector)
            if content:
                break
        
        if not content:
            # Fallback to the body if no content area found
            content = soup.body
        
        # Extract the title
        title = None
        if soup.title:
            title = soup.title.text
        else:
            # Look for heading elements as fallback
            for heading in content.find_all(["h1"]):
                title = heading.get_text(strip=True)
                if title:
                    break
        
        if not title:
            # Use URL as last resort
            title = url.split("/")[-1].replace("-", " ").title()
        
        # Extract the pure text content first (for full text)
        full_text = content.get_text(separator='\n', strip=True)
        
        # Extract headings and their content
        structure = []
        headings = content.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
        
        # If no headings found, treat the entire content as one section
        if not headings:
            structure.append({
                "level": 1,
                "title": title,
                "content": full_text
            })
        else:
            for i, heading in enumerate(headings):
                heading_level = int(heading.name[1])
                heading_text = heading.get_text(strip=True)
                
                # Get content until next heading
                content_elements = []
                
                # Gather all elements until the next heading
                next_elements = []
                current = heading.next_sibling
                
                while current and not (hasattr(current, "name") and current.name and current.name.startswith("h")):
                    if hasattr(current, "name") and current.name:
                        next_elements.append(current)
                    current = current.next_sibling
                
                # Extract text from these elements
                for element in next_elements:
                    if element.name in ['pre', 'code']:
                        # Preserve code blocks with special formatting
                        code_text = element.get_text(strip=True)
                        if code_text:
                            content_elements.append(f"```\n{code_text}\n```")
                    elif element.name == 'p' or element.name in ['ul', 'ol', 'li', 'div']:
                        # Regular text elements
                        text = element.get_text(strip=True)
                        if text:
                            content_elements.append(text)
                
                section_content = "\n\n".join(content_elements)
                if not section_content and i < len(headings) - 1:
                    # If no content was found between headings, check if there's text directly
                    direct_text = heading.next_sibling
                    if direct_text and isinstance(direct_text, str) and direct_text.strip():
                        section_content = direct_text.strip()
                
                structure.append({
                    "level": heading_level,
                    "title": heading_text,
                    "content": section_content
                })
        
        return {
            "url": url,
            "title": title,
            "structure": structure,
            "full_text": full_text,  # Include the full text without HTML
            "last_updated": datetime.now().isoformat()
        }
    
    def parse_sitemap(self, sitemap_url=None):
        """Parse a sitemap to discover URLs."""
        if not sitemap_url:
            # Try common sitemap locations
            sitemap_url = urljoin(self.base_url, "sitemap.xml")
        
        console.print(f"[bold blue]Parsing sitemap: {sitemap_url}[/bold blue]")
        html = self.get_page(sitemap_url)
        
        if not html:
            console.print("[bold yellow]Could not fetch sitemap, falling back to link discovery[/bold yellow]")
            return []
        
        try:
            soup = BeautifulSoup(html, "xml")
            
            # Get URLs from sitemap
            urls = []
            
            # Process standard sitemaps
            for url_tag in soup.find_all("url"):
                loc = url_tag.find("loc")
                if loc and loc.text:
                    urls.append(loc.text)
            
            # Process sitemap indices (nested sitemaps)
            for sitemap_tag in soup.find_all("sitemap"):
                loc = sitemap_tag.find("loc")
                if loc and loc.text:
                    # Recursively parse nested sitemap
                    nested_urls = self.parse_sitemap(loc.text)
                    urls.extend(nested_urls)
            
            console.print(f"[bold green]Found {len(urls)} URLs in sitemap[/bold green]")
            return urls
        
        except Exception as e:
            console.print(f"[bold red]Error parsing sitemap: {str(e)}[/bold red]")
            return []
    
    def crawl(self, start_url=None, use_sitemap=True, save_interval=10):
        """Crawl the DSPy documentation site and extract content.
        
        Args:
            start_url: Starting URL for crawling
            use_sitemap: Whether to use sitemap.xml for URL discovery
            save_interval: How often to save progress (in number of pages processed)
        """
        if not start_url:
            start_url = urljoin(self.base_url, "/learn")
        
        # Initialize URLs to visit
        urls_to_visit = []
        
        # Try to use sitemap if requested
        if use_sitemap:
            sitemap_urls = self.parse_sitemap()
            # Filter URLs to only include those from the base domain
            base_domain = urlparse(self.base_url).netloc
            sitemap_urls = [url for url in sitemap_urls if urlparse(url).netloc == base_domain]
            urls_to_visit.extend(sitemap_urls)
        
        # Add the starting URL if not already included
        if start_url not in urls_to_visit and start_url not in self.visited_urls:
            urls_to_visit.append(start_url)
        
        # Remove URLs that have already been visited
        urls_to_visit = [url for url in urls_to_visit if url not in self.visited_urls]
        
        if not urls_to_visit:
            console.print("[bold yellow]No new URLs to visit. Scraping is complete or already processed.[/bold yellow]")
            return self.doc_pages
        
        processed_since_save = 0
        
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn()
        ) as progress:
            crawl_task = progress.add_task("[yellow]Crawling pages...", total=len(urls_to_visit))
            
            while urls_to_visit:
                # Process URLs in batches to control concurrency
                batch = urls_to_visit[:self.max_workers]
                urls_to_visit = urls_to_visit[self.max_workers:]
                
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = {executor.submit(self.get_page, url): url for url in batch if url not in self.visited_urls}
                    
                    for future in as_completed(futures):
                        url = futures[future]
                        self.visited_urls.add(url)
                        processed_since_save += 1
                        
                        try:
                            html = future.result()
                            if not html:
                                continue
                                
                            # Check if it's a documentation page
                            if self.is_documentation_page(url, html):
                                content = self.extract_content(url, html)
                                if content:
                                    # Save the doc page immediately
                                    self._save_doc_page(url, content)
                                    progress.update(crawl_task, description=f"[bold green]Found doc: {content['title']}")
                            
                            # Extract new links (even if using sitemap, to ensure thorough coverage)
                            new_links = self.extract_links(url, html)
                            new_links = [link for link in new_links if link not in self.visited_urls and link not in urls_to_visit]
                            urls_to_visit.extend(new_links)
                            
                            # Update progress
                            progress.update(crawl_task, completed=len(self.visited_urls), total=len(self.visited_urls) + len(urls_to_visit))
                            
                            # Save progress periodically
                            if processed_since_save >= save_interval:
                                self._save_progress()
                                processed_since_save = 0
                                progress.update(crawl_task, description=f"[bold blue]Progress saved: {len(self.doc_pages)} docs")
                        
                        except Exception as e:
                            console.print(f"[bold red]Error processing {url}: {str(e)}[/bold red]")
        
        # Final save
        self._save_progress()
        console.print(f"[bold green]Crawling complete! Found {len(self.doc_pages)} documentation pages.[/bold green]")
        return self.doc_pages
    
    def save_docs(self, output_dir="dspy_docs"):
        """Save the extracted documentation to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save metadata
        metadata = {
            "total_pages": len(self.doc_pages),
            "crawl_date": datetime.now().isoformat(),
            "base_url": self.base_url
        }
        
        with open(output_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        
        # Save individual pages
        for url, content in self.doc_pages.items():
            # Create a filename from the URL
            filename = self._url_to_filename(url)
            
            # Save as markdown
            md_content = self._convert_to_markdown(content)
            with open(output_path / f"{filename}.md", "w", encoding="utf-8") as f:
                f.write(md_content)
            
            # Also save raw JSON for further processing
            with open(output_path / f"{filename}.json", "w", encoding="utf-8") as f:
                json.dump(content, f, indent=2)
        
        console.print(f"[bold green]Documentation saved to {output_path}[/bold green]")

    def _load_progress(self):
        """Load progress from previous runs."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, "r", encoding="utf-8") as f:
                    progress_data = json.load(f)
                
                self.visited_urls = set(progress_data.get("visited_urls", []))
                
                # Load previously crawled doc pages
                for url, filename in progress_data.get("doc_pages", {}).items():
                    json_path = self.json_dir / f"{filename}.json"
                    if json_path.exists():
                        with open(json_path, "r", encoding="utf-8") as f:
                            self.doc_pages[url] = json.load(f)
                
                console.print(f"[bold green]Loaded progress: {len(self.visited_urls)} URLs visited, {len(self.doc_pages)} docs found[/bold green]")
            except Exception as e:
                console.print(f"[bold red]Error loading progress: {str(e)}[/bold red]")
                # Reset progress if loading fails
                self.visited_urls = set()
                self.doc_pages = {}
    
    def _save_progress(self):
        """Save current progress to allow resuming later."""
        # Create a mapping of URLs to filenames
        doc_pages_map = {url: self._url_to_filename(url) for url in self.doc_pages.keys()}
        
        progress_data = {
            "visited_urls": list(self.visited_urls),
            "doc_pages": doc_pages_map,
            "last_updated": datetime.now().isoformat()
        }
        
        with open(self.progress_file, "w", encoding="utf-8") as f:
            json.dump(progress_data, f, indent=2)
    
    def _save_doc_page(self, url, content):
        """Save a single documentation page as it's discovered."""
        filename = self._url_to_filename(url)
        
        # Save as markdown
        md_content = self._convert_to_markdown(content)
        with open(self.markdown_dir / f"{filename}.md", "w", encoding="utf-8") as f:
            f.write(md_content)
        
        # Save as JSON
        with open(self.json_dir / f"{filename}.json", "w", encoding="utf-8") as f:
            json.dump(content, f, indent=2)
        
        # Add to doc_pages dictionary
        self.doc_pages[url] = content#!/usr/bin/env python3    
    def _convert_to_markdown(self, content):
        """Convert the structured content to markdown."""
        md = f"# {content['title']}\n\n"
        md += f"Source: {content['url']}\n\n"
        md += f"Last Updated: {content['last_updated']}\n\n"
        
        for section in content["structure"]:
            level = section["level"]
            md += f"{'#' * level} {section['title']}\n\n"
            md += f"{section['content']}\n\n"
        
        return md
    
    def create_index(self, output_dir="dspy_docs"):
        """Create an index file of all documentation pages."""
        output_path = Path(output_dir)
        
        index_md = "# DSPy Documentation Index\n\n"
        index_md += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        index_md += f"Base URL: {self.base_url}\n\n"
        index_md += f"Total Pages: {len(self.doc_pages)}\n\n"
        
        # Group by top-level section if possible
        sections = {}
        
        for url, content in sorted(self.doc_pages.items(), key=lambda x: x[1]["title"]):
            path_parts = urlparse(url).path.strip("/").split("/")
            if len(path_parts) > 0:
                section = path_parts[0]
            else:
                section = "General"
            
            if section not in sections:
                sections[section] = []
            
            filename = self._url_to_filename(url)
            sections[section].append({
                "title": content["title"],
                "url": url,
                "filename": filename
            })
        
        # Write the sections
        for section, pages in sorted(sections.items()):
            index_md += f"## {section.title()}\n\n"
            
            for page in pages:
                index_md += f"- [{page['title']}]({page['filename']}.md) ([source]({page['url']}))\n"
            
            index_md += "\n"
        
        # Write the index file
        with open(output_path / "index.md", "w", encoding="utf-8") as f:
            f.write(index_md)
        
        console.print(f"[bold green]Index created at {output_path / 'index.md'}[/bold green]")

def main():
    parser = argparse.ArgumentParser(description="DSPy Documentation Scraper")
    parser.add_argument("--base-url", default="https://dspy.ai/", help="Base URL of the DSPy documentation")
    parser.add_argument("--start-url", help="Starting URL for crawling (overrides default starting paths)")
    parser.add_argument("--output-dir", default="dspy_docs", help="Directory to save documentation")
    parser.add_argument("--cache-dir", default=".dspy_cache", help="Directory to store cached content")
    parser.add_argument("--max-workers", type=int, default=5, help="Maximum number of concurrent workers")
    parser.add_argument("--use-sitemap", action="store_true", help="Try to use sitemap.xml for URL discovery")
    parser.add_argument("--starting-paths", nargs="+", default=["learn", "tutorials", "api"], 
                        help="Paths to start crawling from (default: learn tutorials api)")
    parser.add_argument("--use-proxy", action="store_true", help="Use proxy service to avoid IP blocking")
    parser.add_argument("--proxy-api-key", help="API key for ScrapingBee proxy service")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum number of retry attempts for failed requests")
    parser.add_argument("--clean-cache", action="store_true", help="Clear cache before starting")
    parser.add_argument("--save-interval", type=int, default=10, help="Save progress after processing this many pages")
    parser.add_argument("--resume", action="store_true", help="Resume from previous run")
    parser.add_argument("--reset", action="store_true", help="Reset progress and start fresh")
    
    args = parser.parse_args()
    
    console.print("[bold]DSPy Documentation Scraper[/bold]")
    console.print(f"Base URL: {args.base_url}")
    console.print(f"Output directory: {args.output_dir}")
    
    # Handle clean cache
    if args.clean_cache:
        import shutil
        cache_dir = Path(args.cache_dir)
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            console.print(f"[bold yellow]Cache directory {args.cache_dir} cleaned[/bold yellow]")
    
    # Handle reset
    if args.reset:
        import shutil
        progress_file = Path(args.output_dir) / "progress.json"
        if progress_file.exists():
            os.remove(progress_file)
            console.print(f"[bold yellow]Progress reset, starting fresh[/bold yellow]")
    
    scraper = DSPyScraper(
        base_url=args.base_url,
        max_workers=args.max_workers,
        cache_dir=args.cache_dir,
        use_proxy=args.use_proxy,
        proxy_api_key=args.proxy_api_key,
        max_retries=args.max_retries,
        output_dir=args.output_dir,
        starting_paths=args.starting_paths
    )
    
    console.print(f"[bold]Starting paths: {', '.join(args.starting_paths)}[/bold]")
    console.print(f"Using sitemap: {'Yes' if args.use_sitemap else 'No'}")
    console.print(f"Using proxy: {'Yes' if args.use_proxy else 'No'}")
    console.print(f"Resume mode: {'Yes' if args.resume else 'No'}")
    
    scraper.crawl(args.start_url, use_sitemap=args.use_sitemap, save_interval=args.save_interval)
    scraper.save_docs()
    scraper.create_index()
    
    console.print("[bold green]Scraping complete![/bold green]")
    console.print(f"[bold]Documentation saved to: {Path(args.output_dir).absolute()}[/bold]")
    console.print("[bold]To browse the documentation, open:[/bold]")
    console.print(f"[bold blue]{Path(args.output_dir).absolute() / 'index.md'}[/bold blue]")

if __name__ == "__main__":
    main()