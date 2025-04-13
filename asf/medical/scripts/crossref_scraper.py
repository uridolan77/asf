#!/usr/bin/env python3
"""
Crossref API Documentation Scraper

Directly extracts the OpenAPI specification from the Swagger UI JavaScript.
"""
import os
import json
import re
import argparse
import requests
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup
from rich.console import Console
from rich.progress import Progress
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Initialize Rich console for pretty output
console = Console()

class CrossrefDirectScraper:
    """
    Directly extracts the OpenAPI specification from the Swagger UI page's JavaScript.
    """
    
    def __init__(self, base_url="https://api.crossref.org/swagger-ui/index.html", 
                 output_dir="crossref_docs"):
        """Initialize the Crossref API documentation scraper."""
        self.base_url = base_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.consolidated_dir = self.output_dir / "consolidated"
        self.consolidated_dir.mkdir(exist_ok=True)
        
        # Headers to mimic a browser request
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        
        # Session for connection pooling
        self.session = requests.Session()
        
        # Store the extracted API data
        self.api_spec = None
        self.raw_html = None
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(requests.RequestException)
    )
    def _fetch_with_retry(self, url):
        """Fetch a URL with retry logic."""
        response = self.session.get(url, headers=self.headers, timeout=30)
        response.raise_for_status()
        return response
    
    def extract_spec_from_html(self, html):
        """
        Extract the OpenAPI specification JSON from the HTML content.
        
        This function looks for the spec in various ways:
        1. Try to find it in a <script> tag with spec data
        2. Look for window.ui = SwaggerUIBundle(...) with a spec
        3. Look for Swagger.init() calls with a spec
        """
        # Store the raw HTML for later inspection if needed
        self.raw_html = html
        
        # Try to find the spec in JSON format in a <script> tag
        soup = BeautifulSoup(html, "html.parser")
        
        # Method 1: Look for a script tag with id="swagger-data" or similar
        for script in soup.find_all("script", id=lambda x: x and ("swagger" in x.lower() or "spec" in x.lower())):
            try:
                return json.loads(script.string)
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Method 2: Look for spec data in any script tag
        spec_pattern = re.compile(r'spec:\s*({.+?}),\s*dom_id', re.DOTALL)
        for script in soup.find_all("script"):
            if script.string:
                match = spec_pattern.search(script.string)
                if match:
                    try:
                        # This is risky but sometimes necessary for JS objects
                        spec_json = match.group(1)
                        # Convert JS object to valid JSON
                        spec_json = re.sub(r'([{,])\s*(\w+):', r'\1"\2":', spec_json)
                        return json.loads(spec_json)
                    except (json.JSONDecodeError, TypeError):
                        pass
        
        # Method 3: Try to find SwaggerUIBundle configuration
        swagger_bundle_pattern = re.compile(r'SwaggerUIBundle\(\s*({.+?})\s*\)', re.DOTALL)
        for script in soup.find_all("script"):
            if script.string:
                match = swagger_bundle_pattern.search(script.string)
                if match:
                    try:
                        # Extract the configuration object
                        config_js = match.group(1)
                        # Look for the spec URL or spec object
                        url_match = re.search(r'url:\s*[\'"](.+?)[\'"]', config_js)
                        if url_match:
                            spec_url = url_match.group(1)
                            console.print(f"[bold blue]Found spec URL: {spec_url}[/bold blue]")
                            # Try to fetch the spec from the URL
                            try:
                                if not spec_url.startswith(('http://', 'https://')):
                                    # Handle relative URLs
                                    from urllib.parse import urljoin
                                    spec_url = urljoin(self.base_url, spec_url)
                                
                                spec_response = requests.get(spec_url, headers=self.headers, timeout=10)
                                return spec_response.json()
                            except Exception as e:
                                console.print(f"[bold yellow]Error fetching spec from URL: {str(e)}[/bold yellow]")
                    except Exception as e:
                        console.print(f"[bold yellow]Error parsing SwaggerUIBundle: {str(e)}[/bold yellow]")
        
        # Method 4: Last resort - look for any large JSON object in script tags
        json_pattern = re.compile(r'({[\s\S]*?"swagger"[\s\S]*?})')
        for script in soup.find_all("script"):
            if script.string:
                match = json_pattern.search(script.string)
                if match:
                    try:
                        potential_json = match.group(1)
                        # Try to fix common JavaScript syntax to make it valid JSON
                        cleaned_json = re.sub(r'([{,])\s*(\w+):', r'\1"\2":', potential_json)
                        spec = json.loads(cleaned_json)
                        if "swagger" in spec or "openapi" in spec:
                            return spec
                    except (json.JSONDecodeError, TypeError):
                        pass
        
        # Method 5: Parse OpenAPI URL from various script patterns
        url_patterns = [
            r'url:\s*["\'](.+?\.json)["\']',
            r'const\s+url\s*=\s*["\'](.+?\.json)["\']',
            r'["\'](.+?/swagger\.json)["\']',
            r'["\'](.+?/openapi\.json)["\']',
            r'["\'](.+?/api-docs)["\']'
        ]
        
        for pattern in url_patterns:
            for script in soup.find_all("script"):
                if script.string:
                    matches = re.findall(pattern, script.string)
                    for match in matches:
                        try:
                            if not match.startswith(('http://', 'https://')):
                                # Handle relative URLs
                                from urllib.parse import urljoin
                                match = urljoin(self.base_url, match)
                            
                            console.print(f"[bold blue]Trying to fetch spec from: {match}[/bold blue]")
                            spec_response = requests.get(match, headers=self.headers, timeout=10)
                            return spec_response.json()
                        except Exception as e:
                            console.print(f"[bold yellow]Error fetching spec from URL: {str(e)}[/bold yellow]")
        
        return None
    
    def direct_extract_from_swagger_ui(self):
        """
        Directly create an API specification by examining the Swagger UI structure.
        This is used as a fallback when we can't extract the spec from JavaScript.
        """
        try:
            if not self.raw_html:
                response = self._fetch_with_retry(self.base_url)
                self.raw_html = response.text
            
            console.print("[bold blue]Extracting API spec directly from Swagger UI structure...[/bold blue]")
            
            soup = BeautifulSoup(self.raw_html, "html.parser")
            
            api_spec = {
                "swagger": "2.0",
                "info": {
                    "title": "Crossref API",
                    "description": "API for accessing Crossref metadata",
                    "version": "1.0",
                },
                "tags": [],
                "paths": {},
                "definitions": {}
            }
            
            # Extract API title and info
            title_elem = soup.find("title")
            if title_elem:
                api_spec["info"]["title"] = title_elem.text.strip()
            
            # Find API description
            info_elem = soup.select_one(".information-container")
            if info_elem:
                desc_elem = info_elem.select_one(".description")
                if desc_elem:
                    api_spec["info"]["description"] = desc_elem.text.strip()
            
            # Find version info
            version_elem = soup.select_one(".version")
            if version_elem:
                api_spec["info"]["version"] = version_elem.text.strip()
            
            # Extract tags and endpoints from the Swagger UI
            tag_sections = soup.select(".opblock-tag-section")
            
            for tag_section in tag_sections:
                tag_elem = tag_section.select_one(".opblock-tag")
                if not tag_elem:
                    continue
                
                tag_name = tag_elem.text.strip()
                # Remove count if present (e.g., "Tag (3)")
                tag_name = re.sub(r'\s*\(\d+\)\s*$', '', tag_name)
                
                # Add the tag
                api_spec["tags"].append({
                    "name": tag_name,
                    "description": ""
                })
                
                # Find all operations for this tag
                operations = tag_section.select(".opblock")
                
                for op in operations:
                    # Get HTTP method
                    method_class = None
                    for cls in op.get("class", []):
                        if "opblock-" in cls and "opblock-tag" not in cls:
                            method_class = cls
                            break
                    
                    if not method_class:
                        continue
                    
                    method = method_class.replace("opblock-", "").lower()
                    
                    # Get the path
                    path_elem = op.select_one(".opblock-summary-path") or op.select_one(".opblock-summary-path span")
                    if not path_elem:
                        continue
                    
                    path = path_elem.text.strip()
                    
                    # Get summary
                    summary_elem = op.select_one(".opblock-summary-description")
                    summary = summary_elem.text.strip() if summary_elem else ""
                    
                    # Initialize the path if it doesn't exist
                    if path not in api_spec["paths"]:
                        api_spec["paths"][path] = {}
                    
                    # Add the operation
                    api_spec["paths"][path][method] = {
                        "tags": [tag_name],
                        "summary": summary,
                        "description": "",
                        "operationId": f"{method}_{path.replace('/', '_').replace('{', '').replace('}', '')}",
                        "parameters": [],
                        "responses": {
                            "200": {
                                "description": "Success"
                            }
                        }
                    }
                    
                    # Try to get detailed information by looking at parameters
                    param_rows = op.select(".parameters tr")
                    for row in param_rows:
                        cols = row.find_all("td")
                        if len(cols) >= 2:
                            param_name = cols[0].text.strip()
                            # Skip header row
                            if param_name in ["Name", "Parameter"]:
                                continue
                            
                            param_type = cols[1].text.strip() if len(cols) > 1 else ""
                            param_desc = cols[2].text.strip() if len(cols) > 2 else ""
                            
                            param_in = "query"
                            if "{" + param_name + "}" in path:
                                param_in = "path"
                            
                            api_spec["paths"][path][method]["parameters"].append({
                                "name": param_name,
                                "in": param_in,
                                "description": param_desc,
                                "required": param_in == "path",
                                "type": param_type.lower() or "string"
                            })
            
            # Try to extract models/schemas
            models = soup.select(".models .model-container")
            for model in models:
                model_name_elem = model.select_one(".model-title")
                if not model_name_elem:
                    continue
                
                model_name = model_name_elem.text.strip()
                # Clean up model name
                model_name = model_name.replace("«", "").replace("»", "").split(":", 1)[0].strip()
                
                # Initialize model definition
                api_spec["definitions"][model_name] = {
                    "type": "object",
                    "properties": {}
                }
                
                # Extract properties
                prop_rows = model.select(".model-properties tr")
                for row in prop_rows:
                    cols = row.find_all("td")
                    if len(cols) >= 2:
                        prop_name = cols[0].text.strip()
                        # Skip header row
                        if prop_name in ["Name", "Property"]:
                            continue
                        
                        prop_type = cols[1].text.strip() if len(cols) > 1 else "string"
                        
                        api_spec["definitions"][model_name]["properties"][prop_name] = {
                            "type": prop_type.lower() or "string",
                            "description": cols[2].text.strip() if len(cols) > 2 else ""
                        }
            
            console.print(f"[bold green]Extracted API spec with {len(api_spec['paths'])} paths and {len(api_spec['definitions'])} models[/bold green]")
            
            return api_spec
            
        except Exception as e:
            console.print(f"[bold red]Error during direct extraction: {str(e)}[/bold red]")
            import traceback
            console.print(f"[red]{traceback.format_exc()}[/red]")
            return None
    
    def create_api_docs_from_spec(self, spec):
        """Convert raw OpenAPI spec to our structured API docs format."""
        api_docs = {
            "info": {},
            "tags": {},
            "paths": {},
            "models": {},
        }
        
        # Extract API info
        if "info" in spec:
            api_docs["info"] = spec["info"]
        
        # Process tags
        if "tags" in spec:
            for tag in spec["tags"]:
                tag_name = tag["name"]
                api_docs["tags"][tag_name] = {
                    "name": tag_name,
                    "description": tag.get("description", ""),
                    "endpoints": [],
                }
        
        # Process paths/endpoints
        if "paths" in spec:
            for path, path_methods in spec["paths"].items():
                for method, operation in path_methods.items():
                    if method in ["get", "post", "put", "delete", "patch", "options", "head"]:
                        endpoint = {
                            "path": path,
                            "method": method.upper(),
                            "summary": operation.get("summary", ""),
                            "description": operation.get("description", ""),
                            "operationId": operation.get("operationId", ""),
                            "parameters": [],
                            "responses": {},
                            "tags": operation.get("tags", []),
                        }
                        
                        # Process parameters
                        if "parameters" in operation:
                            for param in operation["parameters"]:
                                param_type = param.get("type", "")
                                if not param_type and "schema" in param:
                                    param_type = param["schema"].get("type", "")
                                
                                endpoint["parameters"].append({
                                    "name": param.get("name", ""),
                                    "in": param.get("in", ""),
                                    "description": param.get("description", ""),
                                    "required": param.get("required", False),
                                    "type": param_type,
                                })
                        
                        # Process responses
                        if "responses" in operation:
                            for status, response in operation["responses"].items():
                                endpoint["responses"][status] = {
                                    "description": response.get("description", ""),
                                    "content": {},
                                }
                        
                        # Add endpoint to main paths list
                        endpoint_id = f"{method.upper()} {path}"
                        api_docs["paths"][endpoint_id] = endpoint
                        
                        # Add endpoint to tag groups
                        for tag in operation.get("tags", []):
                            if tag not in api_docs["tags"]:
                                api_docs["tags"][tag] = {
                                    "name": tag,
                                    "description": "",
                                    "endpoints": [],
                                }
                            
                            api_docs["tags"][tag]["endpoints"].append(endpoint_id)
        
        # Process models/definitions/schemas
        schemas_section = None
        if "definitions" in spec:
            schemas_section = spec["definitions"]
        elif "components" in spec and "schemas" in spec["components"]:
            schemas_section = spec["components"]["schemas"]
        
        if schemas_section:
            for model_name, model_schema in schemas_section.items():
                api_docs["models"][model_name] = {
                    "name": model_name,
                    "type": model_schema.get("type", "object"),
                    "properties": model_schema.get("properties", {}),
                    "required": model_schema.get("required", []),
                    "description": model_schema.get("description", ""),
                }
        
        return api_docs
    
    def scrape(self):
        """Execute the full scraping process."""
        try:
            console.print("[bold]Starting Crossref API Documentation Scraping[/bold]")
            
            # Fetch the Swagger UI page
            response = self._fetch_with_retry(self.base_url)
            html = response.text
            
            # Try to extract the OpenAPI spec from the HTML
            console.print("[bold blue]Attempting to extract API specification from HTML...[/bold blue]")
            spec = self.extract_spec_from_html(html)
            
            # If we couldn't extract the spec, try direct extraction
            if not spec:
                console.print("[bold yellow]Could not extract spec from HTML, trying direct UI parsing...[/bold yellow]")
                spec = self.direct_extract_from_swagger_ui()
            
            if not spec:
                console.print("[bold red]Failed to extract API specification![/bold red]")
                return False
            
            # Save the raw spec
            with open(self.output_dir / "swagger_spec.json", "w", encoding="utf-8") as f:
                json.dump(spec, f, indent=2)
                
            console.print(f"[bold green]Saved raw API specification to {self.output_dir / 'swagger_spec.json'}[/bold green]")
            
            # Convert the spec to our structured API docs format
            self.api_docs = self.create_api_docs_from_spec(spec)
            
            # Generate AI training files
            self.generate_ai_training_files(chunk_size=8000)
            
            console.print("[bold green]Scraping completed successfully![/bold green]")
            console.print(f"[bold]Documentation saved to: {self.output_dir.absolute()}[/bold]")
            
            return True
        except Exception as e:
            console.print(f"[bold red]Error during scraping: {str(e)}[/bold red]")
            import traceback
            console.print(f"[red]{traceback.format_exc()}[/red]")
            return False
    
    def generate_ai_training_files(self, chunk_size=None):
        """Generate consolidated files for AI training."""
        if not self.api_docs:
            console.print("[bold red]No API documentation to generate files from![/bold red]")
            return
        
        console.print("[bold blue]Generating AI training files...[/bold blue]")
        
        # Create text version
        txt_content = f"# {self.api_docs['info'].get('title', 'Crossref API')} Documentation\n\n"
        
        if 'version' in self.api_docs['info']:
            txt_content += f"Version: {self.api_docs['info']['version']}\n\n"
            
        if 'description' in self.api_docs['info']:
            txt_content += f"{self.api_docs['info']['description']}\n\n"
        
        # Add endpoints by tag
        for tag_name, tag in self.api_docs["tags"].items():
            txt_content += f"# {tag['name']}\n\n"
            
            if tag['description']:
                txt_content += f"{tag['description']}\n\n"
            
            for endpoint_id in tag["endpoints"]:
                endpoint = self.api_docs["paths"][endpoint_id]
                txt_content += f"## {endpoint['method']} {endpoint['path']}\n\n"
                
                if endpoint['summary']:
                    txt_content += f"Summary: {endpoint['summary']}\n\n"
                    
                if endpoint['description']:
                    txt_content += f"Description: {endpoint['description']}\n\n"
                
                # Add parameters
                if endpoint.get("parameters"):
                    txt_content += "Parameters:\n\n"
                    for param in endpoint["parameters"]:
                        required = "Required" if param.get('required', False) else "Optional"
                        txt_content += f"- {param.get('name', '')} ({param.get('in', '')}, {required}): {param.get('description', '')}\n"
                    txt_content += "\n"
                
                # Add responses
                if endpoint.get("responses"):
                    txt_content += "Responses:\n\n"
                    for status, response in endpoint["responses"].items():
                        txt_content += f"- Status {status}: {response.get('description', '')}\n"
                    txt_content += "\n"
        
        # Add models
        if self.api_docs["models"]:
            txt_content += "# Data Models\n\n"
            
            for model_name, model in self.api_docs["models"].items():
                txt_content += f"## {model_name}\n\n"
                
                if model.get('description'):
                    txt_content += f"{model['description']}\n\n"
                
                if model.get("properties"):
                    txt_content += "Properties:\n\n"
                    for prop_name, prop in model["properties"].items():
                        is_required = "Required" if prop_name in model.get("required", []) else "Optional"
                        
                        prop_type = "string"
                        if isinstance(prop, dict):
                            prop_type = prop.get("type", "string")
                            description = prop.get("description", "")
                        else:
                            description = ""
                            
                        txt_content += f"- {prop_name} ({prop_type}, {is_required}): {description}\n"
                    txt_content += "\n"
        
        # Save the text file
        txt_path = self.consolidated_dir / "crossref_api_training.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(txt_content)
        
        console.print(f"[bold green]Created training text file: {txt_path}[/bold green]")
        
        # Create JSONL (one JSON per line)
        jsonl_path = self.consolidated_dir / "crossref_api.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            # Add API info
            f.write(json.dumps({"type": "api_info", "content": self.api_docs["info"]}) + "\n")
            
            # Add endpoints
            for endpoint_id, endpoint in self.api_docs["paths"].items():
                f.write(json.dumps({
                    "type": "endpoint",
                    "path": endpoint["path"],
                    "method": endpoint["method"],
                    "summary": endpoint["summary"],
                    "description": endpoint["description"],
                    "tags": endpoint["tags"],
                    "parameters": endpoint["parameters"],
                    "responses": endpoint.get("responses", {})
                }) + "\n")
            
            # Add models
            for model_name, model in self.api_docs["models"].items():
                f.write(json.dumps({
                    "type": "model",
                    "name": model_name,
                    "description": model.get("description", ""),
                    "properties": model.get("properties", {}),
                    "required": model.get("required", [])
                }) + "\n")
        
        console.print(f"[bold green]Created JSONL file: {jsonl_path}[/bold green]")
        
        # Create a simplified JSON
        simple_docs = {
            "info": self.api_docs["info"],
            "endpoints": [],
            "models": []
        }
        
        # Add simplified endpoints
        for endpoint_id, endpoint in self.api_docs["paths"].items():
            simple_endpoint = {
                "path": endpoint["path"],
                "method": endpoint["method"],
                "summary": endpoint["summary"],
                "description": endpoint["description"],
                "tags": endpoint["tags"],
                "parameters": endpoint["parameters"],
            }
            
            # Add response summaries
            simple_endpoint["responses"] = {}
            for status, response in endpoint.get("responses", {}).items():
                simple_endpoint["responses"][status] = response.get("description", "")
            
            simple_docs["endpoints"].append(simple_endpoint)
        
        # Add simplified models
        for model_name, model in self.api_docs["models"].items():
            simple_model = {
                "name": model_name,
                "type": model.get("type", "object"),
                "description": model.get("description", ""),
                "properties": []
            }
            
            # Add property summaries
            for prop_name, prop in model.get("properties", {}).items():
                if isinstance(prop, dict):
                    simple_model["properties"].append({
                        "name": prop_name,
                        "type": prop.get("type", ""),
                        "description": prop.get("description", ""),
                        "required": prop_name in model.get("required", [])
                    })
            
            simple_docs["models"].append(simple_model)
        
        # Save the simplified JSON
        json_path = self.consolidated_dir / "crossref_api_simplified.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(simple_docs, f, indent=2)
        
        console.print(f"[bold green]Created simplified JSON file: {json_path}[/bold green]")
        
        # Create a Markdown version
        md_content = f"# {self.api_docs['info'].get('title', 'Crossref API')} Documentation\n\n"
        
        if 'version' in self.api_docs['info']:
            md_content += f"Version: {self.api_docs['info']['version']}\n\n"
            
        if 'description' in self.api_docs['info']:
            md_content += f"{self.api_docs['info']['description']}\n\n"
        
        # Create table of contents
        md_content += "## Table of Contents\n\n"
        md_content += "- [API Overview](#api-overview)\n"
        md_content += "- [Endpoints](#endpoints)\n"
        
        for tag_name in sorted(self.api_docs["tags"].keys()):
            safe_tag = tag_name.lower().replace(' ', '-')
            md_content += f"  - [{tag_name}](#{safe_tag})\n"
        
        if self.api_docs["models"]:
            md_content += "- [Models](#models)\n"
        
        md_content += "\n## API Overview\n\n"
        
        if 'description' in self.api_docs['info']:
            md_content += f"{self.api_docs['info']['description']}\n\n"
        
        # Add endpoints by tag
        md_content += "## Endpoints\n\n"
        
        for tag_name, tag in sorted(self.api_docs["tags"].items()):
            safe_tag = tag_name.lower().replace(' ', '-')
            md_content += f"### {tag_name}\n\n"
            
            if tag['description']:
                md_content += f"{tag['description']}\n\n"
            
            for endpoint_id in tag["endpoints"]:
                endpoint = self.api_docs["paths"][endpoint_id]
                safe_endpoint = f"{endpoint['method'].lower()}-{endpoint['path'].replace('/', '-').replace('{', '').replace('}', '')}"
                
                md_content += f"#### {endpoint['method']} {endpoint['path']}\n\n"
                
                if endpoint['summary']:
                    md_content += f"**Summary:** {endpoint['summary']}\n\n"
                    
                if endpoint['description']:
                    md_content += f"**Description:** {endpoint['description']}\n\n"
                
                # Add parameters
                if endpoint.get("parameters"):
                    md_content += "**Parameters:**\n\n"
                    md_content += "| Name | Located in | Description | Required | Type |\n"
                    md_content += "| ---- | ---------- | ----------- | -------- | ---- |\n"
                    
                    for param in endpoint["parameters"]:
                        required = "Yes" if param.get('required', False) else "No"
                        md_content += f"| {param.get('name', '')} | {param.get('in', '')} | {param.get('description', '')} | {required} | {param.get('type', '')} |\n"
                    
                    md_content += "\n"
                
                # Add responses
                if endpoint.get("responses"):
                    md_content += "**Responses:**\n\n"
                    for status, response in endpoint["responses"].items():
                        md_content += f"**Status {status}**: {response.get('description', '')}\n\n"
                
                md_content += "---\n\n"
        
        # Add models
        if self.api_docs["models"]:
            md_content += "## Models\n\n"
            
            for model_name, model in sorted(self.api_docs["models"].items()):
                md_content += f"### {model_name}\n\n"
                
                if model.get('description'):
                    md_content += f"{model['description']}\n\n"
                
                if model.get("properties"):
                    md_content += "**Properties:**\n\n"
                    md_content += "| Name | Type | Description | Required |\n"
                    md_content += "| ---- | ---- | ----------- | -------- |\n"
                    
                    for prop_name, prop in model["properties"].items():
                        if isinstance(prop, dict):
                            prop_type = prop.get("type", "")
                            description = prop.get("description", "")
                        else:
                            prop_type = "string"
                            description = ""
                            
                        is_required = "Yes" if prop_name in model.get("required", []) else "No"
                        md_content += f"| {prop_name} | {prop_type} | {description} | {is_required} |\n"
                    
                    md_content += "\n"
                
                md_content += "---\n\n"
        
        # Save the markdown file
        md_path = self.consolidated_dir / "crossref_api_complete.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        
        console.print(f"[bold green]Created markdown documentation file: {md_path}[/bold green]")
        
        # Create chunked files if requested
        if chunk_size:
            self.create_chunked_files(txt_content, chunk_size)
        
        # Create metadata file
        meta_path = self.consolidated_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "source": "Crossref API Documentation",
                "base_url": self.base_url,
                "api_version": self.api_docs["info"].get("version", ""),
                "title": self.api_docs["info"].get("title", "Crossref API"),
                "generated_at": datetime.now().isoformat(),
                "endpoint_count": len(self.api_docs["paths"]),
                "model_count": len(self.api_docs["models"]),
                "tag_count": len(self.api_docs["tags"]),
                "file_formats": ["markdown", "json", "txt", "jsonl"] + (["chunks"] if chunk_size else [])
            }, f, indent=2)
        
        console.print(f"[bold green]Created metadata file: {meta_path}[/bold green]")
    
    def create_chunked_files(self, content, chunk_size):
        """Create chunked text files for large content."""
        chunks = []
        
        # Get API info for each chunk header
        api_info = f"# {self.api_docs['info'].get('title', 'Crossref API')} Documentation\n\n"
        
        if 'version' in self.api_docs['info']:
            api_info += f"Version: {self.api_docs['info']['version']}\n\n"
        
        # Split content by sections (## headings)
        sections = re.split(r'(?=## )', content)
        
        # First section is the intro
        intro = sections[0]
        current_chunk = intro
        
        # Process remaining sections
        for section in sections[1:]:
            # If adding this section would exceed chunk size, start a new chunk
            if len(current_chunk) + len(section) > chunk_size:
                chunks.append(current_chunk)
                current_chunk = api_info + section
            else:
                current_chunk += section
        
        # Add the last chunk if not empty
        if current_chunk and current_chunk != api_info:
            chunks.append(current_chunk)
        
        # Save each chunk to a separate file
        chunks_dir = self.consolidated_dir / "chunks"
        chunks_dir.mkdir(exist_ok=True)
        
        for i, chunk in enumerate(chunks):
            chunk_path = chunks_dir / f"crossref_api_chunk_{i+1}.txt"
            with open(chunk_path, "w", encoding="utf-8") as f:
                f.write(chunk)
        
        console.print(f"[bold green]Created {len(chunks)} chunked files in {chunks_dir}[/bold green]")

def main():
    """Main function to run the scraper from command line."""
    parser = argparse.ArgumentParser(description="Crossref API Documentation Scraper")
    parser.add_argument("--base-url", default="https://api.crossref.org/swagger-ui/index.html", 
                        help="Base URL of the Crossref API Swagger UI")
    parser.add_argument("--output-dir", default="crossref_docs", 
                        help="Directory to save documentation")
    parser.add_argument("--chunk-size", type=int, default=8000,
                        help="Chunk size in characters for text files (for AI context windows)")
    
    args = parser.parse_args()
    
    console.print("[bold]Crossref API Documentation Scraper[/bold]")
    console.print(f"Base URL: {args.base_url}")
    console.print(f"Output directory: {args.output_dir}")
    
    scraper = CrossrefDirectScraper(
        base_url=args.base_url,
        output_dir=args.output_dir
    )
    
    # Run the scraper
    success = scraper.scrape()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())