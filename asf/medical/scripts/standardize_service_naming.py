Standardize Service Naming in ASF Medical Codebase.
This script identifies and fixes inconsistent service naming patterns in the codebase,
implementing a standardized approach with proper naming conventions.
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
logger = logging.getLogger("StandardizeServiceNaming")
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
    "standardize_service_naming.py",
    "deep_cleanup_phase2.py",
    "fix_unused_imports.py",
    "fix_docstrings.py",
    "standardize_error_handling.py"
]
# Service naming patterns to standardize
SERVICE_NAMING_PATTERNS = {
    # Old name pattern: new name pattern
    r"EnhancedUnifiedUnifiedContradictionService": "ContradictionService",
    r"UnifiedUnifiedUnifiedContradictionService": "ContradictionService",
    r"UnifiedUnifiedContradictionService": "ContradictionService",
    r"UnifiedContradictionService": "ContradictionService",
    r"EnhancedContradictionClassifier": "ContradictionClassifierService",
    r"BiasAssessmentEngine": "BiasAssessmentService",
    r"PRISMAScreeningEngine": "PRISMAScreeningService",
    r"TemporalAnalysisEngine": "TemporalService",
    r"ExplanationGeneratorEngine": "ExplanationGeneratorService",
    r"ContradictionResolutionEngine": "ContradictionResolutionService",
    r"MedicalContradictionResolutionEngine": "MedicalContradictionResolutionService"
}
def find_python_files(directory: str) -> List[str]:
    Find all Python files in the given directory and its subdirectories.
    
    Args:
        directory: Description of directory
    
    
    Returns:
        Description of return value
    python_files = []
    for root, dirs, files in os.walk(directory):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for file in files:
            if file.endswith(".py") and file not in EXCLUDE_FILES:
                python_files.append(os.path.join(root, file))
    return python_files
def find_service_naming_issues(content: str) -> Dict[str, List[Dict[str, Any]]]:
    Find service naming issues in the content.
    
    Args:
        content: Description of content
    
    
    Returns:
        Description of return value
    issues = {}
    for old_pattern, new_name in SERVICE_NAMING_PATTERNS.items():
        matches = []
        for match in re.finditer(r'\b' + old_pattern + r'\b', content):
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
def fix_service_naming(content: str, issues: Dict[str, List[Dict[str, Any]]]) -> str:
    Fix service naming issues in the content.
    
    Args:
        content: Description of content
        issues: Description of issues
    
    
    Returns:
        Description of return value
    if not issues:
        return content
    # Sort all matches by position in reverse order to avoid changing positions
    all_matches = []
    for old_pattern, matches in issues.items():
        for match in matches:
            all_matches.append({
                "old_pattern": old_pattern,
                "new_name": SERVICE_NAMING_PATTERNS[old_pattern],
                "start": match["start"],
                "end": match["end"],
                "match": match["match"]
            })
    all_matches.sort(key=lambda x: x["start"], reverse=True)
    # Replace matches
    content_chars = list(content)
    for match in all_matches:
        content_chars[match["start"]:match["end"]] = match["new_name"]
    return "".join(content_chars)
def process_file(file_path: str, fix: bool = False) -> Dict[str, Any]:
    Process a single file to find and optionally fix service naming issues.
    
    Args:
        file_path: Description of file_path
        fix: Description of fix
    
    
    Returns:
        Description of return value
    results = {
        "issues": {},
        "fixed": False
    }
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        # Find issues
        issues = find_service_naming_issues(content)
        results["issues"] = issues
        # Fix issues if requested
        if fix and issues:
            updated_content = fix_service_naming(content, issues)
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
    Main function.
    if len(sys.argv) < 2:
        print("Usage: python standardize_service_naming.py <directory> [--fix]")
        sys.exit(1)
    directory = sys.argv[1]
    fix = "--fix" in sys.argv
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a directory")
        sys.exit(1)
    logger.info(f"Starting standardize service naming for {directory}")
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
                logger.info(f"{file_path}:{match['line']} - {old_pattern} -> {SERVICE_NAMING_PATTERNS[old_pattern]}")
        if results["fixed"]:
            fixed_files += 1
            logger.info(f"Fixed service naming in {file_path}")
    # Report summary
    logger.info("=" * 50)
    logger.info("Summary:")
    logger.info(f"Total Python files processed: {len(python_files)}")
    logger.info(f"Total service naming issues: {total_issues}")
    if fix:
        logger.info(f"Fixed files: {fixed_files}")
    logger.info("=" * 50)
if __name__ == "__main__":
    main()