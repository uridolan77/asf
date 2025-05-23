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