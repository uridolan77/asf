Standardize Caching Script for the ASF Medical Research Synthesizer Codebase.
This script standardizes the caching implementations across the codebase,
consolidating on the EnhancedCacheManager as the single caching solution.
Usage:
    python -m asf.medical.scripts.standardize_caching [--dry-run]
import os
import re
import logging
import argparse
from typing import List, Tuple, Any
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("StandardizeCaching")
CACHE_PATTERNS = {
    r"from asf\.medical\.core\.cache import (LRUCache|cache_manager|cached)":
    "from asf.medical.core.enhanced_cache import enhanced_cache_manager as cache_manager, enhanced_cached as cached",
    r"cache\s*=\s*LRUCache\(":
    "cache = enhanced_cache_manager",
    r"await cache_manager\.(get|set|delete|clear|invalidate_pattern)\(":
    r"await enhanced_cache_manager.\1(",
    r"@cached\(":
    "@enhanced_cached(",
    r"from asf\.layer4_environmental_coupling\.components\.distributed_cache import DistributedCouplingCache":
    "from asf.medical.core.enhanced_cache import enhanced_cache_manager",
    r"class Cache:[\s\S]*?def __init__\(self, ttl: int = 60\):[\s\S]*?def get\(self, key: str\)[\s\S]*?def set\(self, key: str, value: Any\)[\s\S]*?def clear\(self\)":
    "# Using enhanced_cache_manager instead of custom Cache implementation"
}
EXCLUDE_DIRS = [
    "__pycache__",
    ".git",
    ".github",
    ".vscode",
    "venv",
    "env",
    "node_modules",
    "dist",
    "build",
    "htmlcov",
]
EXCLUDE_FILES = [
    ".gitignore",
    ".env",
    "README.md",
    "LICENSE",
    "requirements.txt",
    "setup.py",
    "pyproject.toml",
    "enhanced_cache.py",  # Don't modify the enhanced cache implementation itself
]
def find_python_files(directory: str) -> List[str]:
    Find all Python files in the given directory and its subdirectories.
    
    Args:
        directory: Description of directory
    
    
    Returns:
        Description of return value
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        original_content = content
        content, changed = standardize_caching(content)
        if changed and not dry_run:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"Standardized caching in {file_path}")
        elif changed:
            logger.info(f"Would standardize caching in {file_path} (dry run)")
        return changed
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return False
def main():
    """Main function to run the script.
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
    """
    parser = argparse.ArgumentParser(description="Standardize caching implementations in the ASF Medical Research Synthesizer codebase")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually modify files, just show what would be done")
    parser.add_argument("--directory", default=None, help="Directory to process (default: asf/medical)")
    args = parser.parse_args()
    if args.directory:
        directory = args.directory
    else:
        directory = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "medical")
    logger.info(f"Scanning directory: {directory}")
    python_files = find_python_files(directory)
    logger.info(f"Found {len(python_files)} Python files")
    standardized_files = 0
    for file in python_files:
        if process_file(file, args.dry_run):
            standardized_files += 1
    logger.info(f"Standardized caching in {standardized_files} files")
    if args.dry_run:
        logger.info("This was a dry run. No files were actually modified.")
if __name__ == "__main__":
    main()