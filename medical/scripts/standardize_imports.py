"""Standardize Imports in ASF Medical Codebase.

This script identifies and fixes inconsistent import patterns in the codebase,
implementing a standardized approach with proper import organization.
"""
import os
import re
import sys
import logging
from typing import List, Dict, Any
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("StandardizeImports")
# Directories to exclude from processing
EXCLUDE_DIRS = [
    "__pycache__",
    ".git",
    ".vscode",
    "venv",
    "env",
    "node_modules",
    "dist",
    "build"
]
# Files to exclude from processing
EXCLUDE_FILES = [
    "__init__.py",
    "conftest.py",
    "standardize_imports.py",
    "deep_cleanup_phase2.py",
    "fix_unused_imports.py",
    "fix_docstrings.py",
    "standardize_error_handling.py",
    "standardize_service_naming.py"
]
# Import patterns to standardize
IMPORT_PATTERNS = {
    # Old import pattern: new import pattern
    r"from asf\.medical\.ml\.services\.enhanced_contradiction_classifier import ContradictionClassifierService":
        "from asf.medical.ml.services.contradiction_classifier_service import ContradictionClassifierService",
    r"from asf\.medical\.ml\.services\.contradiction_service import ContradictionService":
        "from asf.medical.ml.services.unified_contradiction_service import ContradictionService",
    r"from asf\.medical\.ml\.services\.enhanced_contradiction_service import ContradictionService":
        "from asf.medical.ml.services.unified_contradiction_service import ContradictionService",
    r"from asf\.medical\.ml\.services\.contradiction_service_new import ContradictionService":
        "from asf.medical.ml.services.unified_contradiction_service import ContradictionService"
}
def find_python_files(directory: str) -> List[str]:
    """Find all Python files in the given directory and its subdirectories.

    Args:
        directory: Directory to search for Python files

    Returns:
        List of Python file paths
    """
    python_files = []
    for root, dirs, files in os.walk(directory):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for file in files:
            if file.endswith(".py") and file not in EXCLUDE_FILES:
                python_files.append(os.path.join(root, file))
    return python_files
def find_import_issues(content: str) -> Dict[str, List[Dict[str, Any]]]:
    """Find import issues in the content.

    Args:
        content: Content to search for import issues

    Returns:
        Dictionary of import issues found
    """
    issues = {}
    for old_pattern, new_import in IMPORT_PATTERNS.items():
        matches = []
        for match in re.finditer(old_pattern, content):
            line_num = content[:match.start()].count('\n') + 1
            matches.append({
                "line": line_num,
                "match": match.group(0),
                "start": match.start(),
                "end": match.end()
            })
        if matches:
            issues[old_pattern] = matches
    return issues
def fix_imports(content: str, issues: Dict[str, List[Dict[str, Any]]]) -> str:
    """Fix import issues in the content.

    Args:
        content: Content to fix import issues in
        issues: Dictionary of import issues to fix

    Returns:
        Updated content with fixed imports
    """
    if not issues:
        return content
    # Sort all matches by position in reverse order to avoid changing positions
    all_matches = []
    for old_pattern, matches in issues.items():
        for match in matches:
            all_matches.append({
                "old_pattern": old_pattern,
                "new_import": IMPORT_PATTERNS[old_pattern],
                "start": match["start"],
                "end": match["end"],
                "match": match["match"]
            })
    all_matches.sort(key=lambda x: x["start"], reverse=True)
    # Replace matches
    content_chars = list(content)
    for match in all_matches:
        content_chars[match["start"]:match["end"]] = match["new_import"]
    return "".join(content_chars)
def process_file(file_path: str, fix: bool = False) -> Dict[str, Any]:
    """Process a single file to find and optionally fix import issues.

    Args:
        file_path: Path to the file to process
        fix: Whether to fix the issues found

    Returns:
        Dictionary with issues found and whether they were fixed
    """
    results = {
        "issues": {},
        "fixed": False
    }
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        # Find issues
        issues = find_import_issues(content)
        results["issues"] = issues
        # Fix issues if requested
        if fix and issues:
            updated_content = fix_imports(content, issues)
            # Write updated content if changes were made
            if updated_content != content:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(updated_content)
                results["fixed"] = True
        return results
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return results
def main():
    """Main function to standardize imports in the codebase."""

    if len(sys.argv) < 2:
        print("Usage: python standardize_imports.py <directory> [--fix]")
        sys.exit(1)
    directory = sys.argv[1]
    fix = "--fix" in sys.argv
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a directory")
        sys.exit(1)
    logger.info(f"Starting standardize imports for {directory}")
    logger.info(f"Fix mode: {fix}")
    python_files = find_python_files(directory)
    logger.info(f"Found {len(python_files)} Python files")
    total_issues = 0
    fixed_files = 0
    for file_path in python_files:
        logger.info(f"Processing {file_path}")
        results = process_file(file_path, fix)
        # Count issues
        for old_pattern, matches in results["issues"].items():
            total_issues += len(matches)
            for match in matches:
                logger.info(f"{file_path}:{match['line']} - {old_pattern} -> {IMPORT_PATTERNS[old_pattern]}")
        if results["fixed"]:
            fixed_files += 1
            logger.info(f"Fixed imports in {file_path}")
    # Report summary
    logger.info("=" * 50)
    logger.info("Summary:")
    logger.info(f"Total Python files processed: {len(python_files)}")
    logger.info(f"Total import issues: {total_issues}")
    if fix:
        logger.info(f"Fixed files: {fixed_files}")
    logger.info("=" * 50)
if __name__ == "__main__":
    main()