#!/usr/bin/env python3
DSPy Documentation Scraper
A tool for extracting and organizing documentation from dspy.ai.
Features:
- Concurrent requests for faster scraping
- Content extraction with proper HTML parsing
- Markdown conversion for clean output
- Local caching to avoid redundant requests
- Command-line interface with customization options
import json
import time
import argparse
import requests
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
# Initialize Rich console for pretty output
console = Console()
class DSPyScraper:
    """
    DSPyScraper class.
    
    This class provides functionality for...
    """
    def __init__(self, base_url="https://dspy.ai/", max_workers=5, cache_dir=".dspy_cache"):
        """Initialize the DSPy documentation scraper.
        Args:
            base_url: Base URL of the DSPy documentation
            max_workers: Maximum number of concurrent workers
            cache_dir: Directory to store cached content
        """
        self.base_url = base_url
        self.max_workers = max_workers
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        # Headers to mimic a browser request
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0",
        }
        # Store visited URLs to avoid duplicates
        self.visited_urls = set()
        # Store documentation pages
        self.doc_pages = {}
        # Session for connection pooling
        self.session = requests.Session()
    def get_page(self, url):
        Fetch a page and return its content.
        
        Args:
            url: Description of url
        
        
        Returns:
            Description of return value
        parsed = urlparse(url)
        path = parsed.path.strip("/").replace("/", "_")
        if not path:
            path = "index"
        return f"{parsed.netloc}_{path}"
    def is_documentation_page(self, url, html):
        Check if a page is a documentation page.
        
        Args:
            url: Description of url
            html: Description of html
        
        
        Returns:
            Description of return value
        # This is a basic check - you might need to adjust based on actual site structure
        if not html:
            return False
        soup = BeautifulSoup(html, "html.parser")
        # Check if the page has common documentation elements
        has_sidebar = bool(soup.select(".sidebar, .docs-sidebar, .navigation, nav"))
        has_content = bool(soup.select("article, .content, .documentation, .doc-content"))
        has_headings = len(soup.select("h1, h2, h3")) > 0
        # Check if URL contains documentation-related paths
        doc_patterns = ["learn", "tutorials", "api", "guide", "tutorial", "reference", "api"]
        path_contains_doc = any(pattern in url.lower() for pattern in doc_patterns)
        return (has_sidebar and has_content) or (has_headings and path_contains_doc)
    def extract_links(self, url, html):
        Extract all links from a page that belong to the same domain.
        
        Args:
            url: Description of url
            html: Description of html
        
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
        Extract and structure documentation content.
        
        Args:
            url: Description of url
            html: Description of html
        
        if not html:
            return None
        soup = BeautifulSoup(html, "html.parser")
        # Find the main content area
        content_selectors = [
            "article", ".content", ".documentation", ".doc-content", 
            "main", "#content", "#main-content"
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
        title = soup.title.text if soup.title else url
        # Extract headings and their content
        structure = []
        headings = content.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
        for heading in headings:
            heading_level = int(heading.name[1])
            heading_text = heading.get_text(strip=True)
            # Get content until next heading
            content_elements = []
            current = heading.next_sibling
            while current and not (hasattr(current, "name") and current.name and current.name.startswith("h")):
                if hasattr(current, "name") and current.name:
                    # Handle code blocks specially
                    if current.name == "pre" or current.name == "code":
                        code_text = current.get_text()
                        content_elements.append(f"```python\n{code_text}\n```")
                    else:
                        text = current.get_text(strip=True)
                        if text:
                            content_elements.append(text)
                current = current.next_sibling
            structure.append({
                "level": heading_level,
                "title": heading_text,
                "content": "\n\n".join(content_elements)
            })
        return {
            "url": url,
            "title": title,
            "structure": structure,
            "last_updated": datetime.now().isoformat()
        }
    def crawl(self, start_url=None):
        Crawl the DSPy documentation site and extract content.
        
        Args:
            start_url: Description of start_url
        
        if not start_url:
            start_url = urljoin(self.base_url, "/learn")
        # Add the starting URL to the queue
        urls_to_visit = [start_url]
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn()
        ) as progress:
            crawl_task = progress.add_task("[yellow]Crawling pages...", total=None)
            while urls_to_visit:
                # Process URLs in batches to control concurrency
                batch = urls_to_visit[:self.max_workers]
                urls_to_visit = urls_to_visit[self.max_workers:]
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = {executor.submit(self.get_page, url): url for url in batch if url not in self.visited_urls}
                    for future in as_completed(futures):
                        url = futures[future]
                        self.visited_urls.add(url)
                        try:
                            html = future.result()
                            if not html:
                                continue
                            # Check if it's a documentation page
                            if self.is_documentation_page(url, html):
                                content = self.extract_content(url, html)
                                if content:
                                    self.doc_pages[url] = content
                                    progress.update(crawl_task, description=f"[bold green]Found doc: {content['title']}")
                            # Extract new links
                            new_links = self.extract_links(url, html)
                            new_links = [link for link in new_links if link not in self.visited_urls and link not in urls_to_visit]
                            urls_to_visit.extend(new_links)
                            progress.update(crawl_task, completed=len(self.visited_urls), total=len(self.visited_urls) + len(urls_to_visit))
                        except Exception as e:
                            console.print(f"[bold red]Error processing {url}: {str(e)}[/bold red]")
        console.print(f"[bold green]Crawling complete! Found {len(self.doc_pages)} documentation pages.[/bold green]")
        return self.doc_pages
    def save_docs(self, output_dir="dspy_docs"):
        Save the extracted documentation to files.
        
        Args:
            output_dir: Description of output_dir
        
        md = f"# {content['title']}\n\n"
        md += f"Source: {content['url']}\n\n"
        md += f"Last Updated: {content['last_updated']}\n\n"
        for section in content["structure"]:
            level = section["level"]
            md += f"{'#' * level} {section['title']}\n\n"
            md += f"{section['content']}\n\n"
        return md
    def create_index(self, output_dir="dspy_docs"):
        Create an index file of all documentation pages.
        
        Args:
            output_dir: Description of output_dir
        
    parser = argparse.ArgumentParser(description="DSPy Documentation Scraper")
    parser.add_argument("--base-url", default="https://dspy.ai/", help="Base URL of the DSPy documentation")
    parser.add_argument("--start-url", help="Starting URL for crawling (defaults to base-url/learn)")
    parser.add_argument("--output-dir", default="dspy_docs", help="Directory to save documentation")
    parser.add_argument("--cache-dir", default=".dspy_cache", help="Directory to store cached content")
    parser.add_argument("--max-workers", type=int, default=5, help="Maximum number of concurrent workers")
    args = parser.parse_args()
    console.print("[bold]DSPy Documentation Scraper[/bold]")
    console.print(f"Base URL: {args.base_url}")
    console.print(f"Output directory: {args.output_dir}")
    scraper = DSPyScraper(
        base_url=args.base_url,
        max_workers=args.max_workers,
        cache_dir=args.cache_dir
    )
    start_url = args.start_url or urljoin(args.base_url, "/learn")
    console.print(f"[bold]Starting crawl from: {start_url}[/bold]")
    scraper.crawl(start_url)
    scraper.save_docs(args.output_dir)
    scraper.create_index(args.output_dir)
    console.print("[bold green]Scraping complete![/bold green]")
if __name__ == "__main__":
    main()