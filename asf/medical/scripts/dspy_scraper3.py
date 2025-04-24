"""
Module description.

This module provides functionality for...
"""
def save_consolidated_files(self):
    Save the extracted documentation as consolidated files by section.
    This creates one big file per section (learn, tutorials, api, etc.)
    for both markdown and JSON formats, making it easier to upload to AI systems.
    
    Args:
    
    console.print("[bold blue]Saving consolidated files by section...[/bold blue]")
    # Group documents by section
    sections = {}
    for url, content in self.doc_pages.items():
        # Determine the section based on the URL path
        path_parts = urlparse(url).path.strip("/").split("/")
        if not path_parts:
            section = "other"
        else:
            section = path_parts[0].lower()
            # If it's not one of our known sections, put it in "other"
            if section not in self.starting_paths and section not in ["examples", "guide"]:
                section = "other"
        if section not in sections:
            sections[section] = []
        sections[section].append(content)
    # Create consolidated directory
    consolidated_dir = self.output_dir / "consolidated"
    consolidated_dir.mkdir(exist_ok=True)
    # Save markdown files
    for section, contents in sections.items():
        # Skip empty sections
        if not contents:
            continue
        # Sort by title for consistency
        contents.sort(key=lambda x: x.get('title', ''))
        # Create consolidated markdown
        md_content = f"# DSPy {section.title()} Documentation\n\n"
        md_content += f"Number of pages: {len(contents)}\n\n"
        md_content += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        md_content += "---\n\n"
        for i, content in enumerate(contents):
            md_content += f"# {content['title']}\n\n"
            md_content += f"Source: {content['url']}\n\n"
            # Add the full text if available
            if "full_text" in content:
                md_content += content["full_text"]
            else:
                # Add structured content if no full text
                for section in content["structure"]:
                    level = section["level"]
                    md_content += f"{'#' * (level + 1)} {section['title']}\n\n"
                    md_content += f"{section['content']}\n\n"
            # Add separator between documents
            if i < len(contents) - 1:
                md_content += "\n\n---\n\n"
        # Save the consolidated markdown
        md_path = consolidated_dir / f"dspy_{section}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        console.print(f"[bold green]Created consolidated markdown file: {md_path}[/bold green]")
        # Create consolidated JSON
        json_path = consolidated_dir / f"dspy_{section}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "section": section,
                "count": len(contents),
                "generated_at": datetime.now().isoformat(),
                "contents": contents
            }, f, indent=2)
        console.print(f"[bold green]Created consolidated JSON file: {json_path}[/bold green]")
        # Create a single massive file with everything
        all_content = []
        for contents in sections.values():
            all_content.extend(contents)
        # Sort by title
        all_content.sort(key=lambda x: x.get('title', ''))
        # Create all-in-one markdown
        md_content = f"# Complete DSPy Documentation\n\n"
        md_content += f"Number of pages: {len(all_content)}\n\n"
        md_content += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        md_content += "---\n\n"
        for i, content in enumerate(all_content):
            md_content += f"# {content['title']}\n\n"
            md_content += f"Source: {content['url']}\n\n"
            # Add the full text if available
            if "full_text" in content:
                md_content += content["full_text"]
            else:
                # Add structured content if no full text
                for section in content["structure"]:
                    level = section["level"]
                    md_content += f"{'#' * (level + 1)} {section['title']}\n\n"
                    md_content += f"{section['content']}\n\n"
            # Add separator between documents
            if i < len(all_content) - 1:
                md_content += "\n\n---\n\n"
        # Save the all-in-one markdown
        md_path = consolidated_dir / "dspy_complete.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        console.print(f"[bold green]Created complete markdown file: {md_path}[/bold green]")
        # Create all-in-one JSON
        json_path = consolidated_dir / "dspy_complete.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "count": len(all_content),
                "generated_at": datetime.now().isoformat(),
                "contents": all_content
            }, f, indent=2)
        console.print(f"[bold green]Created complete JSON file: {json_path}[/bold green]")
        # Create a plaintext version with just the full text for ML training
        txt_content = ""
        for content in all_content:
            if "full_text" in content:
                txt_content += content["full_text"] + "\n\n"
        txt_path = consolidated_dir / "dspy_training.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(txt_content)
        console.print(f"[bold green]Created training text file: {txt_path}[/bold green]")
    def _load_progress(self):
        """Load progress from previous runs."""
        if self.progress_file.exists():
            try:
                console.print(f"[bold blue]Loading progress from {self.progress_file}...[/bold blue]")
                with open(self.progress_file, "r", encoding="utf-8") as f:
                    progress_data = json.load(f)
                self.visited_urls = set(progress_data.get("visited_urls", []))
                # Load previously crawled doc pages
                doc_pages_map = progress_data.get("doc_pages", {})
                for url, filename in doc_pages_map.items():
                    json_path = self.json_dir / f"{filename}.json"
                    if json_path.exists():
                        try:
                            with open(json_path, "r", encoding="utf-8") as f:
                                self.doc_pages[url] = json.load(f)
                        except json.JSONDecodeError:
                            console.print(f"[bold yellow]Warning: Could not decode JSON in {json_path}[/bold yellow]")
                console.print(f"[bold green]Loaded progress: {len(self.visited_urls)} URLs visited, {len(self.doc_pages)} docs found[/bold green]")
            except Exception as e:
                console.print(f"[bold red]Error loading progress: {str(e)}[/bold red]")
                # Reset progress if loading fails
                self.visited_urls = set()
                self.doc_pages = {}
        else:
            console.print("[bold blue]No previous progress found. Starting fresh.[/bold blue]")
    def _save_progress(self):
        """Save current progress to allow resuming later."""
        try:
            # Create a mapping of URLs to filenames
            doc_pages_map = {url: self._url_to_filename(url) for url in self.doc_pages.keys()}
            progress_data = {
                "visited_urls": list(self.visited_urls),
                "doc_pages": doc_pages_map,
                "last_updated": datetime.now().isoformat()
            }
            with open(self.progress_file, "w", encoding="utf-8") as f:
                json.dump(progress_data, f, indent=2)
            return True
        except Exception as e:
            console.print(f"[bold red]Error saving progress: {str(e)}[/bold red]")
            return False
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
    """
    DSPyScraper class.
    
    This class provides functionality for...
    """
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
        # Store visited URLs to avoid duplicates
        self.visited_urls = set()
        # Store documentation pages
        self.doc_pages = {}
        # Load progress if it exists
        if self.progress_file.exists():
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
        
        if not start_url:
            start_url = urljoin(self.base_url, "/docs")
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
    parser.add_argument("--consolidated", action="store_true", help="Save consolidated files by section for AI training")
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
    console.print(f"Consolidated output: {'Yes' if args.consolidated else 'No'}")
    scraper.crawl(args.start_url, use_sitemap=args.use_sitemap, save_interval=args.save_interval)
    # Save in the requested format
    if args.consolidated:
        scraper.save_consolidated_files()
    else:
        scraper.save_docs()
    scraper.create_index()
    console.print("[bold green]Scraping complete![/bold green]")
    console.print(f"[bold]Documentation saved to: {Path(args.output_dir).absolute()}[/bold]")
    if args.consolidated:
        console.print("[bold]Consolidated files for AI training:[/bold]")
        consolidated_dir = Path(args.output_dir) / "consolidated"
        for file in consolidated_dir.glob("*.md"):
            console.print(f"[bold blue]{file}[/bold blue]")
    else:
        console.print("[bold]To browse the documentation, open:[/bold]")
        console.print(f"[bold blue]{Path(args.output_dir).absolute() / 'index.md'}[/bold blue]")
if __name__ == "__main__":
    main()