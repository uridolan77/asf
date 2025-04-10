"""
Migration helper for updating client code to use the unified Medical Research Synthesizer API client.

This script helps migrate existing client code to use the new unified client library.
"""

import os
import re
import argparse
import logging
from typing import List, Dict, Any, Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Patterns to match in client code
PATTERNS = [
    # Direct requests to the API
    {
        "pattern": r"requests\.(?P<method>get|post|put|delete)\s*\(\s*[\"'](?P<url>https?://[^/\"']+/v\d+/[^\"']+)[\"']",
        "replacement": "client.{method}(\"{endpoint}\"",
        "import_add": "from asf.medical.client.api_client import MedicalResearchSynthesizerClient",
        "description": "Direct requests to the API"
    },
    # Authentication
    {
        "pattern": r"requests\.post\s*\(\s*[\"'](?P<url>https?://[^/\"']+/v\d+/auth/token)[\"']\s*,\s*data\s*=\s*\{[\"']username[\"']\s*:\s*(?P<email>[^,]+)\s*,\s*[\"']password[\"']\s*:\s*(?P<password>[^}]+)\}",
        "replacement": "client.login({email}, {password})",
        "import_add": "from asf.medical.client.api_client import MedicalResearchSynthesizerClient",
        "description": "Authentication"
    },
    # Search
    {
        "pattern": r"requests\.post\s*\(\s*[\"'](?P<url>https?://[^/\"']+/v\d+/search)[\"']\s*,\s*json\s*=\s*\{[\"']query[\"']\s*:\s*(?P<query>[^,]+)(?:\s*,\s*[\"']max_results[\"']\s*:\s*(?P<max_results>[^}]+))?\}",
        "replacement": "client.search({query}{max_results_param})",
        "import_add": "from asf.medical.client.api_client import MedicalResearchSynthesizerClient",
        "description": "Search"
    },
    # Contradiction analysis
    {
        "pattern": r"requests\.post\s*\(\s*[\"'](?P<url>https?://[^/\"']+/v\d+/analysis/contradictions)[\"']\s*,\s*json\s*=\s*\{[\"']query[\"']\s*:\s*(?P<query>[^,]+)(?:\s*,\s*[\"']max_results[\"']\s*:\s*(?P<max_results>[^,}]+))?(?:\s*,\s*[\"']threshold[\"']\s*:\s*(?P<threshold>[^,}]+))?(?:\s*,\s*[\"']use_biomedlm[\"']\s*:\s*(?P<use_biomedlm>[^,}]+))?(?:\s*,\s*[\"']use_tsmixer[\"']\s*:\s*(?P<use_tsmixer>[^,}]+))?(?:\s*,\s*[\"']use_lorentz[\"']\s*:\s*(?P<use_lorentz>[^,}]+))?\}",
        "replacement": "client.analyze_contradictions({query}{max_results_param}{threshold_param}{use_biomedlm_param}{use_tsmixer_param}{use_lorentz_param})",
        "import_add": "from asf.medical.client.api_client import MedicalResearchSynthesizerClient",
        "description": "Contradiction analysis"
    },
    # Knowledge base creation
    {
        "pattern": r"requests\.post\s*\(\s*[\"'](?P<url>https?://[^/\"']+/v\d+/knowledge-base)[\"']\s*,\s*json\s*=\s*\{[\"']name[\"']\s*:\s*(?P<name>[^,]+)\s*,\s*[\"']query[\"']\s*:\s*(?P<query>[^,]+)(?:\s*,\s*[\"']update_schedule[\"']\s*:\s*(?P<update_schedule>[^}]+))?\}",
        "replacement": "client.create_knowledge_base({name}, {query}{update_schedule_param})",
        "import_add": "from asf.medical.client.api_client import MedicalResearchSynthesizerClient",
        "description": "Knowledge base creation"
    },
    # Knowledge base listing
    {
        "pattern": r"requests\.get\s*\(\s*[\"'](?P<url>https?://[^/\"']+/v\d+/knowledge-base)[\"']",
        "replacement": "client.list_knowledge_bases()",
        "import_add": "from asf.medical.client.api_client import MedicalResearchSynthesizerClient",
        "description": "Knowledge base listing"
    },
    # Knowledge base retrieval
    {
        "pattern": r"requests\.get\s*\(\s*[\"'](?P<url>https?://[^/\"']+/v\d+/knowledge-base)/(?P<kb_id>[^\"']+)[\"']",
        "replacement": "client.get_knowledge_base({kb_id})",
        "import_add": "from asf.medical.client.api_client import MedicalResearchSynthesizerClient",
        "description": "Knowledge base retrieval"
    },
    # Knowledge base update
    {
        "pattern": r"requests\.post\s*\(\s*[\"'](?P<url>https?://[^/\"']+/v\d+/knowledge-base)/(?P<kb_id>[^/\"']+)/update[\"']",
        "replacement": "client.update_knowledge_base({kb_id})",
        "import_add": "from asf.medical.client.api_client import MedicalResearchSynthesizerClient",
        "description": "Knowledge base update"
    },
    # Knowledge base deletion
    {
        "pattern": r"requests\.delete\s*\(\s*[\"'](?P<url>https?://[^/\"']+/v\d+/knowledge-base)/(?P<kb_id>[^\"']+)[\"']",
        "replacement": "client.delete_knowledge_base({kb_id})",
        "import_add": "from asf.medical.client.api_client import MedicalResearchSynthesizerClient",
        "description": "Knowledge base deletion"
    },
    # Export
    {
        "pattern": r"requests\.post\s*\(\s*[\"'](?P<url>https?://[^/\"']+/v\d+/export)/(?P<format>[^\"']+)[\"']\s*,\s*json\s*=\s*\{(?:[\"']result_id[\"']\s*:\s*(?P<result_id>[^,}]+)|[\"']query[\"']\s*:\s*(?P<query>[^,}]+)(?:\s*,\s*[\"']max_results[\"']\s*:\s*(?P<max_results>[^}]+))?)\}",
        "replacement": "client.export_results(\"{format}\"{result_id_param}{query_param}{max_results_param})",
        "import_add": "from asf.medical.client.api_client import MedicalResearchSynthesizerClient",
        "description": "Export"
    },
    # Screening
    {
        "pattern": r"requests\.post\s*\(\s*[\"'](?P<url>https?://[^/\"']+/v\d+/screening/prisma)[\"']\s*,\s*json\s*=\s*\{[\"']query[\"']\s*:\s*(?P<query>[^,]+)(?:\s*,\s*[\"']max_results[\"']\s*:\s*(?P<max_results>[^,}]+))?(?:\s*,\s*[\"']stage[\"']\s*:\s*(?P<stage>[^,}]+))?(?:\s*,\s*[\"']criteria[\"']\s*:\s*(?P<criteria>[^}]+))?\}",
        "replacement": "client.screen_articles({query}{max_results_param}{stage_param}{criteria_param})",
        "import_add": "from asf.medical.client.api_client import MedicalResearchSynthesizerClient",
        "description": "Screening"
    },
    # Bias assessment
    {
        "pattern": r"requests\.post\s*\(\s*[\"'](?P<url>https?://[^/\"']+/v\d+/screening/bias-assessment)[\"']\s*,\s*json\s*=\s*\{[\"']query[\"']\s*:\s*(?P<query>[^,]+)(?:\s*,\s*[\"']max_results[\"']\s*:\s*(?P<max_results>[^,}]+))?(?:\s*,\s*[\"']domains[\"']\s*:\s*(?P<domains>[^}]+))?\}",
        "replacement": "client.assess_bias({query}{max_results_param}{domains_param})",
        "import_add": "from asf.medical.client.api_client import MedicalResearchSynthesizerClient",
        "description": "Bias assessment"
    },
    # Response handling
    {
        "pattern": r"(?P<response>\w+)\.json\(\)\[[\"'](?P<field>data|results|access_token|token_type|role|expires_in)[\"']\]",
        "replacement": "{response}.data.get(\"{field}\")",
        "description": "Response handling"
    },
    # Headers with token
    {
        "pattern": r"headers\s*=\s*\{[\"']Authorization[\"']\s*:\s*f[\"']Bearer\s+\{(?P<token>\w+)\}[\"']\}",
        "replacement": "# Token is handled automatically by the client",
        "description": "Headers with token"
    },
    # Requests import
    {
        "pattern": r"import\s+requests",
        "replacement": "import asyncio\nfrom asf.medical.client.api_client import MedicalResearchSynthesizerClient",
        "description": "Requests import"
    }
]

def find_patterns(file_path: str) -> List[Dict[str, Any]]:
    """
    Find patterns in a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        List of matches
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    matches = []
    for pattern_info in PATTERNS:
        pattern = pattern_info["pattern"]
        for match in re.finditer(pattern, content):
            matches.append({
                "pattern_info": pattern_info,
                "match": match,
                "start": match.start(),
                "end": match.end(),
                "file_path": file_path
            })
    
    # Sort matches by position
    matches.sort(key=lambda m: m["start"])
    
    return matches

def process_match(match: Dict[str, Any]) -> str:
    """
    Process a match.
    
    Args:
        match: Match information
        
    Returns:
        Replacement string
    """
    pattern_info = match["pattern_info"]
    match_obj = match["match"]
    replacement = pattern_info["replacement"]
    
    # Extract named groups
    groups = match_obj.groupdict()
    
    # Process special cases
    if "method" in groups and "url" in groups:
        method = groups["method"]
        url = groups["url"]
        endpoint = url.split("/v1/")[-1] if "/v1/" in url else url.split("/v2/")[-1] if "/v2/" in url else ""
        replacement = replacement.format(method=method, endpoint=endpoint)
    
    elif "max_results" in groups:
        max_results_param = f", max_results={groups['max_results']}" if groups["max_results"] else ""
        replacement = replacement.format(
            query=groups.get("query", ""),
            max_results_param=max_results_param
        )
    
    elif "threshold" in groups:
        max_results_param = f", max_results={groups['max_results']}" if groups.get("max_results") else ""
        threshold_param = f", threshold={groups['threshold']}" if groups.get("threshold") else ""
        use_biomedlm_param = f", use_biomedlm={groups['use_biomedlm']}" if groups.get("use_biomedlm") else ""
        use_tsmixer_param = f", use_tsmixer={groups['use_tsmixer']}" if groups.get("use_tsmixer") else ""
        use_lorentz_param = f", use_lorentz={groups['use_lorentz']}" if groups.get("use_lorentz") else ""
        replacement = replacement.format(
            query=groups.get("query", ""),
            max_results_param=max_results_param,
            threshold_param=threshold_param,
            use_biomedlm_param=use_biomedlm_param,
            use_tsmixer_param=use_tsmixer_param,
            use_lorentz_param=use_lorentz_param
        )
    
    elif "update_schedule" in groups:
        update_schedule_param = f", update_schedule={groups['update_schedule']}" if groups.get("update_schedule") else ""
        replacement = replacement.format(
            name=groups.get("name", ""),
            query=groups.get("query", ""),
            update_schedule_param=update_schedule_param
        )
    
    elif "kb_id" in groups:
        replacement = replacement.format(kb_id=groups["kb_id"])
    
    elif "format" in groups:
        result_id_param = f", result_id={groups['result_id']}" if groups.get("result_id") else ""
        query_param = f", query={groups['query']}" if groups.get("query") else ""
        max_results_param = f", max_results={groups['max_results']}" if groups.get("max_results") else ""
        replacement = replacement.format(
            format=groups["format"],
            result_id_param=result_id_param,
            query_param=query_param,
            max_results_param=max_results_param
        )
    
    elif "stage" in groups:
        max_results_param = f", max_results={groups['max_results']}" if groups.get("max_results") else ""
        stage_param = f", stage={groups['stage']}" if groups.get("stage") else ""
        criteria_param = f", criteria={groups['criteria']}" if groups.get("criteria") else ""
        replacement = replacement.format(
            query=groups.get("query", ""),
            max_results_param=max_results_param,
            stage_param=stage_param,
            criteria_param=criteria_param
        )
    
    elif "domains" in groups:
        max_results_param = f", max_results={groups['max_results']}" if groups.get("max_results") else ""
        domains_param = f", domains={groups['domains']}" if groups.get("domains") else ""
        replacement = replacement.format(
            query=groups.get("query", ""),
            max_results_param=max_results_param,
            domains_param=domains_param
        )
    
    elif "response" in groups and "field" in groups:
        replacement = replacement.format(
            response=groups["response"],
            field=groups["field"]
        )
    
    elif "token" in groups:
        replacement = replacement.format(token=groups["token"])
    
    elif "email" in groups and "password" in groups:
        replacement = replacement.format(
            email=groups["email"],
            password=groups["password"]
        )
    
    return replacement

def migrate_file(file_path: str, apply: bool = False) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Migrate a file.
    
    Args:
        file_path: Path to the file
        apply: Whether to apply the changes
        
    Returns:
        Tuple of (new content, matches)
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    matches = find_patterns(file_path)
    if not matches:
        return content, []
    
    # Process matches in reverse order to avoid changing positions
    matches.sort(key=lambda m: m["start"], reverse=True)
    
    new_content = content
    for match in matches:
        replacement = process_match(match)
        new_content = new_content[:match["start"]] + replacement + new_content[match["end"]:]
    
    # Add imports
    imports_to_add = set()
    for match in matches:
        if "import_add" in match["pattern_info"]:
            imports_to_add.add(match["pattern_info"]["import_add"])
    
    if imports_to_add:
        import_section = "\n".join(imports_to_add) + "\n\n"
        if "import " in new_content:
            # Find the last import statement
            import_matches = list(re.finditer(r"^import\s+.*$|^from\s+.*\s+import\s+.*$", new_content, re.MULTILINE))
            if import_matches:
                last_import = import_matches[-1]
                new_content = new_content[:last_import.end()] + "\n" + import_section + new_content[last_import.end():]
            else:
                new_content = import_section + new_content
        else:
            new_content = import_section + new_content
    
    # Add async wrapper if needed
    if "await client." in new_content and "async def" not in new_content:
        # Wrap the main code in an async function
        main_function = """
async def main():
    # Create client
    client = MedicalResearchSynthesizerClient(
        base_url=os.getenv("API_BASE_URL", "http://localhost:8000"),
        api_version=os.getenv("API_VERSION", "v1")
    )
    
    try:
        # Your code here
"""
        
        # Indent the existing code
        lines = new_content.split("\n")
        indented_lines = []
        for line in lines:
            if line.strip() and not line.startswith("import ") and not line.startswith("from "):
                indented_lines.append("        " + line)
            else:
                indented_lines.append(line)
        
        # Add the closing part of the async function
        main_function_end = """
    finally:
        # Close client
        await client.close()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
"""
        
        # Combine everything
        new_content = "\n".join(indented_lines) + main_function_end
    
    if apply:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
    
    return new_content, matches

def migrate_directory(directory: str, apply: bool = False) -> Dict[str, List[Dict[str, Any]]]:
    """
    Migrate all Python files in a directory.
    
    Args:
        directory: Directory to migrate
        apply: Whether to apply the changes
        
    Returns:
        Dictionary of file paths to matches
    """
    results = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                _, matches = migrate_file(file_path, apply)
                if matches:
                    results[file_path] = matches
    
    return results

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Migrate client code to use the unified API client")
    parser.add_argument("--file", help="File to migrate")
    parser.add_argument("--directory", help="Directory to migrate")
    parser.add_argument("--apply", action="store_true", help="Apply the changes")
    args = parser.parse_args()
    
    if args.file:
        new_content, matches = migrate_file(args.file, args.apply)
        if matches:
            logger.info(f"Found {len(matches)} matches in {args.file}")
            for match in matches:
                logger.info(f"  {match['pattern_info']['description']}: {match['match'].group(0)}")
            
            if args.apply:
                logger.info(f"Applied changes to {args.file}")
            else:
                logger.info("Run with --apply to apply the changes")
        else:
            logger.info(f"No matches found in {args.file}")
    
    elif args.directory:
        results = migrate_directory(args.directory, args.apply)
        if results:
            logger.info(f"Found matches in {len(results)} files")
            for file_path, matches in results.items():
                logger.info(f"  {file_path}: {len(matches)} matches")
            
            if args.apply:
                logger.info(f"Applied changes to {len(results)} files")
            else:
                logger.info("Run with --apply to apply the changes")
        else:
            logger.info(f"No matches found in {args.directory}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
