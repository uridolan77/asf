#!/usr/bin/env python3
"""
UTS REST API Documentation Scraper
A tool for extracting and organizing documentation from the UMLS Terminology Services REST API.
Features:
- Concurrent requests for faster scraping
- Content extraction with proper HTML parsing
- Markdown conversion for clean output
- Local caching to avoid redundant requests
- Command-line interface with customization options
- Robust retry mechanism for handling connection issues
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

class UTSScraper:
    """
    UTSScraper class for extracting documentation from the UMLS Terminology Services REST API.
    
    This class provides functionality for crawling the UTS REST API documentation website,
    extracting content, and saving it in various formats for easier access and integration
    with AI systems.
    """
    
    def __init__(self, base_url="https://documentation.uts.nlm.nih.gov/rest/", 
                 max_workers=5, cache_dir=".uts_cache", max_retries=3, 
                 output_dir="uts_docs"):
        """Initialize the UTS documentation scraper.
        
        Args:
            base_url: Base URL of the UTS REST API documentation
            max_workers: Maximum number of concurrent workers
            cache_dir: Directory to store cached content
            max_retries: Maximum number of retry attempts for failed requests
            output_dir: Directory to save documentation
        """
        self.base_url = base_url
        self.max_workers = max_workers
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_retries = max_retries
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
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
    
    def _url_to_filename(self, url):
        """Convert a URL to a safe filename.
        
        Args:
            url: The URL to convert
            
        Returns:
            A filename-safe string based on the URL
        """
        parsed = urlparse(url)
        path = parsed.path.strip("/").replace("/", "_")
        if not path:
            path = "index"
        return f"{parsed.netloc}_{path}"
    
    @retry(
        stop=stop_after_attempt(3),  # Stop after 3 attempts
        wait=wait_exponential(multiplier=1, min=4, max=10),  # Wait between 4-10 seconds, doubling each time
        retry=retry_if_exception_type(requests.RequestException)  # Only retry on request exceptions
    )
    def _fetch_with_retry(self, url):
        """Fetch a page with retry logic.
        
        Args:
            url: The URL to fetch
            
        Returns:
            Response object from requests
        """
        response = self.session.get(url, headers=self.headers, timeout=10)
        response.raise_for_status()
        return response
    
    def get_page(self, url):
        """Fetch a page and return its content.
        
        Args:
            url: The URL to fetch
            
        Returns:
            HTML content of the page or None if it couldn't be fetched
        """
        # Check cache first
        cache_key = self._url_to_filename(url)
        cache_file = self.cache_dir / f"{cache_key}.html"
        
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    html = f.read()
                return html
            except Exception as e:
                console.print(f"[bold yellow]Cache read error for {url}: {str(e)}[/bold yellow]")
        
        # If not in cache, fetch it
        try:
            response = self._fetch_with_retry(url)
            html = response.text
            
            # Save to cache
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(html)
                
            return html
        except Exception as e:
            console.print(f"[bold red]Error fetching {url}: {str(e)}[/bold red]")
            return None
    
    def is_documentation_page(self, url, html):
        """Check if a page is a documentation page.
        
        Args:
            url: The URL of the page
            html: The HTML content of the page
            
        Returns:
            True if the page is a documentation page, False otherwise
        """
        if not html:
            return False
            
        soup = BeautifulSoup(html, "html.parser")
        
        # URL-based heuristics
        url_lower = url.lower()
        path = urlparse(url).path.lower()
        
        # Check for UTS-specific paths that are likely to be documentation
        uts_doc_paths = ["/rest/", "/api/", "/home.html", "/security.html", "/versions.html"]
        is_doc_path = any(doc_path in path for doc_path in uts_doc_paths)
        
        # Content-based heuristics
        has_headings = len(soup.select("h1, h2, h3")) > 0
        has_paragraphs = len(soup.select("p")) > 2  # Docs usually have several paragraphs
        has_code = len(soup.select("pre, code")) > 0  # Docs often include code examples
        
        # Check for content that suggests a documentation page (UTS-specific checks)
        uts_terms = ["umls", "terminology", "api", "rest", "authentication", "ticket", "json", "xml"]
        text_lower = soup.get_text().lower()
        has_uts_terms = any(term in text_lower for term in uts_terms)
        
        # Specific UTS REST API documentation structure checks
        has_main_content = soup.select_one("#main-content") is not None
        has_content_area = soup.select_one(".content-area") is not None
        
        # Combination of heuristics to determine if it's a documentation page
        is_doc = (
            # Either URL path suggests documentation
            is_doc_path or 
            # Or a combination of content features suggests documentation
            (has_headings and has_paragraphs and (has_code or has_uts_terms)) or
            # Or it has the UTS documentation structure
            (has_main_content or has_content_area)
        )
        
        return is_doc
    
    def extract_links(self, url, html):
        """Extract all links from a page that belong to the same domain.
        
        Args:
            url: The URL of the page
            html: The HTML content of the page
            
        Returns:
            List of links found on the page
        """
        if not html:
            return []
            
        soup = BeautifulSoup(html, "html.parser")
        base_domain = urlparse(self.base_url).netloc
        links = []
        
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            # Handle relative URLs
            full_url = urljoin(url, href)
            # Only include links from the same domain
            if urlparse(full_url).netloc == base_domain:
                links.append(full_url)
                
        return list(set(links))
    
    def extract_content(self, url, html):
        """Extract and structure documentation content.
        
        Args:
            url: The URL of the page
            html: The HTML content of the page
            
        Returns:
            Dictionary containing structured documentation content
        """
        if not html:
            return None
            
        soup = BeautifulSoup(html, "html.parser")
        
        # Extract title
        title_tag = soup.find("title")
        title = title_tag.text.strip() if title_tag else "Untitled"
        
        # Try to get a better title from the main heading
        h1_tag = soup.find("h1")
        if h1_tag:
            title = h1_tag.text.strip()
        
        # Extract main content
        main_content = soup.select_one("#main-content, .content-area, .main")
        if not main_content:
            # Fallback to body if no main content container is found
            main_content = soup.body
        
        # Extract structure with headings
        structure = []
        current_section = None
        current_content = []
        
        # Find all content elements
        for elem in main_content.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "pre", "ul", "ol", "table", "div"]):
            # Skip empty elements and non-content divs
            if not elem.text.strip() or (elem.name == "div" and not any(elem.find_all(["p", "pre", "ul", "ol", "table"]))):
                continue
                
            # If it's a heading, start a new section
            if elem.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                # Save the previous section if it exists
                if current_section and current_content:
                    structure.append({
                        "level": current_section["level"],
                        "title": current_section["title"],
                        "content": "\n\n".join(current_content)
                    })
                    current_content = []
                
                # Start a new section
                level = int(elem.name[1])
                current_section = {
                    "level": level,
                    "title": elem.text.strip()
                }
            else:
                # If we haven't encountered a heading yet, create a default section
                if not current_section:
                    current_section = {
                        "level": 1,
                        "title": "Introduction"
                    }
                
                # Add content to the current section
                if elem.name == "pre" or elem.find("code"):
                    # Format code blocks
                    code_text = elem.text.strip()
                    current_content.append(f"```\n{code_text}\n```")
                elif elem.name == "ul" or elem.name == "ol":
                    # Format lists
                    list_items = []
                    for li in elem.find_all("li"):
                        list_items.append(f"- {li.text.strip()}")
                    current_content.append("\n".join(list_items))
                elif elem.name == "table":
                    # Format tables (simplified)
                    table_text = "| " + " | ".join([th.text.strip() for th in elem.find_all("th")]) + " |\n"
                    table_text += "| " + " | ".join(["---" for _ in elem.find_all("th")]) + " |\n"
                    for tr in elem.find_all("tr")[1:]:  # Skip header row
                        table_text += "| " + " | ".join([td.text.strip() for td in tr.find_all("td")]) + " |\n"
                    current_content.append(table_text)
                else:
                    # Regular paragraph
                    current_content.append(elem.text.strip())
        
        # Add the last section if it exists
        if current_section and current_content:
            structure.append({
                "level": current_section["level"],
                "title": current_section["title"],
                "content": "\n\n".join(current_content)
            })
        
        # Combine all text for full-text search and embedding
        full_text = ""
        for section in structure:
            full_text += f"# {section['title']}\n\n{section['content']}\n\n"
        
        # Create the structured document
        doc = {
            "url": url,
            "title": title,
            "structure": structure,
            "full_text": full_text,
            "last_updated": datetime.now().isoformat()
        }
        
        return doc
    
    def crawl(self, start_url=None, save_interval=10):
        """Crawl the documentation site starting from the given URL.
        
        Args:
            start_url: URL to start crawling from (defaults to base_url)
            save_interval: How often to save progress (number of pages)
            
        Returns:
            Dictionary of documentation pages
        """
        if not start_url:
            start_url = urljoin(self.base_url, "home.html")
        
        # Initialize URLs to visit
        urls_to_visit = [start_url]
        
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
                            
                            # Extract new links
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
        self.doc_pages[url] = content
    
    def _convert_to_markdown(self, content):
        """Convert structured content to markdown format."""
        md = f"# {content['title']}\n\n"
        md += f"Source: {content['url']}\n\n"
        md += f"Last Updated: {content['last_updated']}\n\n"
        
        for section in content["structure"]:
            level = section["level"]
            md += f"{'#' * level} {section['title']}\n\n"
            md += f"{section['content']}\n\n"
            
        return md
    
    def save_docs(self):
        """Save the extracted documentation to files."""
        console.print("[bold blue]Saving documentation pages...[/bold blue]")
        
        # Create an index file
        index_content = "# UTS REST API Documentation\n\n"
        index_content += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        index_content += f"Total pages: {len(self.doc_pages)}\n\n"
        index_content += "## Documentation Pages\n\n"
        
        # Sort pages by title for the index
        sorted_pages = sorted(self.doc_pages.values(), key=lambda x: x.get("title", ""))
        
        for page in sorted_pages:
            filename = self._url_to_filename(page["url"])
            index_content += f"- [{page['title']}](markdown/{filename}.md) ([source]({page['url']}))\n"
        
        # Save the index file
        with open(self.output_dir / "index.md", "w", encoding="utf-8") as f:
            f.write(index_content)
            
        console.print(f"[bold green]Saved index file: {self.output_dir / 'index.md'}[/bold green]")
        console.print(f"[bold green]Documentation saved successfully![/bold green]")
    
    def save_consolidated_files(self):
        """Save the extracted documentation as consolidated files.
        This creates one big file for markdown and JSON formats,
        making it easier to upload to AI systems.
        """
        console.print("[bold blue]Saving consolidated files...[/bold blue]")
        
        # Create consolidated directory
        consolidated_dir = self.output_dir / "consolidated"
        consolidated_dir.mkdir(exist_ok=True)
        
        # Sort pages by title for consistency
        sorted_pages = sorted(self.doc_pages.values(), key=lambda x: x.get("title", ""))
        
        # Create consolidated markdown
        md_content = f"# UTS REST API Documentation\n\n"
        md_content += f"Number of pages: {len(sorted_pages)}\n\n"
        md_content += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        md_content += "---\n\n"
        
        for i, content in enumerate(sorted_pages):
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
            if i < len(sorted_pages) - 1:
                md_content += "\n\n---\n\n"
        
        # Save the consolidated markdown
        md_path = consolidated_dir / "uts_complete.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
            
        console.print(f"[bold green]Created consolidated markdown file: {md_path}[/bold green]")
        
        # Create consolidated JSON
        json_path = consolidated_dir / "uts_complete.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "count": len(sorted_pages),
                "generated_at": datetime.now().isoformat(),
                "contents": sorted_pages
            }, f, indent=2)
            
        console.print(f"[bold green]Created consolidated JSON file: {json_path}[/bold green]")
        
        # Create a plaintext version with just the full text for ML training
        txt_content = ""
        for content in sorted_pages:
            if "full_text" in content:
                txt_content += content["full_text"] + "\n\n"
                
        txt_path = consolidated_dir / "uts_training.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(txt_content)
            
        console.print(f"[bold green]Created training text file: {txt_path}[/bold green]")

def main():
    """Main function to run the scraper from command line."""
    parser = argparse.ArgumentParser(description="UTS REST API Documentation Scraper")
    parser.add_argument("--base-url", default="https://documentation.uts.nlm.nih.gov/rest/", 
                        help="Base URL of the UTS REST API documentation")
    parser.add_argument("--start-url", default="https://documentation.uts.nlm.nih.gov/rest/home.html",
                        help="Starting URL for crawling")
    parser.add_argument("--output-dir", default="uts_docs", 
                        help="Directory to save documentation")
    parser.add_argument("--cache-dir", default=".uts_cache", 
                        help="Directory to store cached content")
    parser.add_argument("--max-workers", type=int, default=5, 
                        help="Maximum number of concurrent workers")
    parser.add_argument("--max-retries", type=int, default=3, 
                        help="Maximum number of retry attempts for failed requests")
    parser.add_argument("--clean-cache", action="store_true", 
                        help="Clear cache before starting")
    parser.add_argument("--save-interval", type=int, default=10, 
                        help="Save progress after processing this many pages")
    parser.add_argument("--resume", action="store_true", 
                        help="Resume from previous run")
    parser.add_argument("--reset", action="store_true", 
                        help="Reset progress and start fresh")
    parser.add_argument("--consolidated", action="store_true", 
                        help="Save consolidated files for AI training")
    
    args = parser.parse_args()
    
    console.print("[bold]UTS REST API Documentation Scraper[/bold]")
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
    
    scraper = UTSScraper(
        base_url=args.base_url,
        max_workers=args.max_workers,
        cache_dir=args.cache_dir,
        max_retries=args.max_retries,
        output_dir=args.output_dir
    )
    
    console.print(f"Using start URL: {args.start_url}")
    console.print(f"Resume mode: {'Yes' if args.resume else 'No'}")
    console.print(f"Consolidated output: {'Yes' if args.consolidated else 'No'}")
    
    # Start crawling
    scraper.crawl(args.start_url, save_interval=args.save_interval)
    
    # Save in the requested format
    if args.consolidated:
        scraper.save_consolidated_files()
    
    # Always save individual docs and create index
    scraper.save_docs()
    
    console.print("[bold green]Scraping complete![/bold green]")
    console.print(f"[bold]Documentation saved to: {Path(args.output_dir).absolute()}[/bold]")
    
    if args.consolidated:
        console.print("[bold]Consolidated files for AI training:[/bold]")
        consolidated_dir = Path(args.output_dir) / "consolidated"
        for file in consolidated_dir.glob("*.*"):
            console.print(f"[bold blue]{file}[/bold blue]")
    else:
        console.print("[bold]To browse the documentation, open:[/bold]")
        console.print(f"[bold blue]{Path(args.output_dir).absolute() / 'index.md'}[/bold blue]")

if __name__ == "__main__":
    main()