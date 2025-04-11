"""
Deep Cleanup Phase 2 for ASF Medical Codebase.

This script performs the second phase of deep cleanup for the ASF Medical codebase,
focusing on:
1. Identifying and fixing remaining naming inconsistencies
2. Removing unused imports
3. Fixing inconsistent error handling
4. Improving docstrings
"""

import os
import re
import sys
import logging
from typing import List, Dict, Tuple, Set, Optional, Any
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("DeepCleanupPhase2")

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
    "deep_cleanup.py",
    "deep_cleanup_phase2.py"
]

# Patterns for finding issues
UNUSED_IMPORT_PATTERN = r"^from\s+[\w.]+\s+import\s+[\w,\s]+$"
INCONSISTENT_ERROR_PATTERN = r"except\s+Exception\s+as\s+e:"
INCOMPLETE_DOCSTRING_PATTERN = r'"""[^"]*"""'
OLD_SERVICE_REFERENCES = {
    r"EnhancedContradictionClassifier": "ContradictionClassifierService",
    r"EnhancedUnifiedUnifiedContradictionService": "ContradictionService",
    r"UnifiedUnifiedUnifiedContradictionService": "ContradictionService",
    r"UnifiedUnifiedContradictionService": "ContradictionService",
    r"UnifiedContradictionService": "ContradictionService"
}

# Replacements for fixing issues
ERROR_HANDLING_REPLACEMENT = (
    r"except Exception as e:\s*\n\s*logger\.error\(f\"([^\"]+):\s*{str\(e\)}\"\)\s*\n\s*raise",
    r"except Exception as e:\n    logger.error(f\"\1: {str(e)}\")\n    raise DatabaseError(f\"\1: {str(e)}\")"
)

def find_python_files(directory: str) -> List[str]:
    """Find all Python files in the given directory and its subdirectories."""
    python_files = []
    for root, dirs, files in os.walk(directory):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        
        for file in files:
            if file.endswith(".py") and file not in EXCLUDE_FILES:
                python_files.append(os.path.join(root, file))
    return python_files

def find_old_service_references(content: str) -> List[Tuple[str, str, int, int]]:
    """Find references to old service names in the content."""
    references = []
    lines = content.split("\n")
    
    for i, line in enumerate(lines):
        for old_name, new_name in OLD_SERVICE_REFERENCES.items():
            if re.search(r"\b" + old_name + r"\b", line):
                references.append((old_name, new_name, i + 1, line))
    
    return references

def find_unused_imports(content: str) -> List[Tuple[int, str]]:
    """Find unused imports in the content."""
    # This is a simplified approach and might have false positives/negatives
    unused_imports = []
    lines = content.split("\n")
    
    import_lines = []
    for i, line in enumerate(lines):
        if line.startswith("from ") or line.startswith("import "):
            import_lines.append((i, line))
    
    for i, line in import_lines:
        # Extract imported names
        if "import" in line:
            if "from" in line:
                # from module import name1, name2
                match = re.match(r"from\s+[\w.]+\s+import\s+([\w,\s]+)", line)
                if match:
                    imported_names = [name.strip() for name in match.group(1).split(",")]
            else:
                # import module
                match = re.match(r"import\s+([\w,\s]+)", line)
                if match:
                    imported_names = [name.strip() for name in match.group(1).split(",")]
        
            # Check if each imported name is used in the rest of the file
            for name in imported_names:
                if name == "*":
                    continue  # Skip wildcard imports
                
                # Check if the name is used in the rest of the file
                name_pattern = r"\b" + name + r"\b"
                used = False
                for j, other_line in enumerate(lines):
                    if j != i and re.search(name_pattern, other_line):
                        used = True
                        break
                
                if not used:
                    unused_imports.append((i + 1, name))
    
    return unused_imports

def find_inconsistent_error_handling(content: str) -> List[Tuple[int, str]]:
    """Find inconsistent error handling in the content."""
    inconsistent_errors = []
    lines = content.split("\n")
    
    for i, line in enumerate(lines):
        if re.search(INCONSISTENT_ERROR_PATTERN, line):
            # Check if the next line logs the error
            if i + 1 < len(lines) and "logger.error" in lines[i + 1]:
                # Check if the next line after that raises the error
                if i + 2 < len(lines) and "raise" in lines[i + 2]:
                    # Check if it's raising a custom error
                    if "raise " in lines[i + 2] and not re.search(r"raise\s+\w+Error", lines[i + 2]):
                        inconsistent_errors.append((i + 1, lines[i:i+3]))
            else:
                inconsistent_errors.append((i + 1, line))
    
    return inconsistent_errors

def find_incomplete_docstrings(content: str) -> List[Tuple[int, str]]:
    """Find incomplete docstrings in the content."""
    incomplete_docstrings = []
    lines = content.split("\n")
    
    for i, line in enumerate(lines):
        if '"""' in line and '"""' in line[line.index('"""') + 3:]:
            # Single line docstring
            if not re.search(r'""".+"""', line):
                incomplete_docstrings.append((i + 1, line))
    
    # Find multi-line docstrings
    docstring_start = None
    for i, line in enumerate(lines):
        if '"""' in line and not re.search(r'""".+"""', line):
            if docstring_start is None:
                docstring_start = i
            else:
                # End of docstring
                docstring = "\n".join(lines[docstring_start:i+1])
                
                # Check if docstring is incomplete
                if not re.search(r"Args:|Parameters:|Returns:|Raises:", docstring):
                    incomplete_docstrings.append((docstring_start + 1, docstring))
                
                docstring_start = None
    
    return incomplete_docstrings

def process_file(file_path: str, fix: bool = False) -> Dict[str, Any]:
    """Process a single file to find and optionally fix issues."""
    results = {
        "old_service_references": [],
        "unused_imports": [],
        "inconsistent_error_handling": [],
        "incomplete_docstrings": [],
        "fixed": False
    }
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Find issues
        results["old_service_references"] = find_old_service_references(content)
        results["unused_imports"] = find_unused_imports(content)
        results["inconsistent_error_handling"] = find_inconsistent_error_handling(content)
        results["incomplete_docstrings"] = find_incomplete_docstrings(content)
        
        # Fix issues if requested
        if fix:
            updated_content = content
            
            # Fix old service references
            for old_name, new_name, _, _ in results["old_service_references"]:
                updated_content = re.sub(r"\b" + old_name + r"\b", new_name, updated_content)
            
            # Fix inconsistent error handling
            old_pattern, new_pattern = ERROR_HANDLING_REPLACEMENT
            updated_content = re.sub(old_pattern, new_pattern, updated_content)
            
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
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python deep_cleanup_phase2.py <directory> [--fix]")
        sys.exit(1)
    
    directory = sys.argv[1]
    fix = "--fix" in sys.argv
    
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a directory")
        sys.exit(1)
    
    logger.info(f"Starting deep cleanup phase 2 for {directory}")
    logger.info(f"Fix mode: {fix}")
    
    python_files = find_python_files(directory)
    logger.info(f"Found {len(python_files)} Python files")
    
    total_issues = {
        "old_service_references": 0,
        "unused_imports": 0,
        "inconsistent_error_handling": 0,
        "incomplete_docstrings": 0,
        "fixed_files": 0
    }
    
    for file_path in python_files:
        logger.info(f"Processing {file_path}")
        results = process_file(file_path, fix)
        
        # Count issues
        total_issues["old_service_references"] += len(results["old_service_references"])
        total_issues["unused_imports"] += len(results["unused_imports"])
        total_issues["inconsistent_error_handling"] += len(results["inconsistent_error_handling"])
        total_issues["incomplete_docstrings"] += len(results["incomplete_docstrings"])
        
        if results["fixed"]:
            total_issues["fixed_files"] += 1
            logger.info(f"Fixed issues in {file_path}")
        
        # Report issues
        for old_name, new_name, line_num, line in results["old_service_references"]:
            logger.info(f"{file_path}:{line_num} - Old service reference: {old_name} -> {new_name}")
        
        for line_num, name in results["unused_imports"]:
            logger.info(f"{file_path}:{line_num} - Unused import: {name}")
        
        for line_num, lines in results["inconsistent_error_handling"]:
            logger.info(f"{file_path}:{line_num} - Inconsistent error handling")
        
        for line_num, docstring in results["incomplete_docstrings"]:
            logger.info(f"{file_path}:{line_num} - Incomplete docstring")
    
    # Report summary
    logger.info("=" * 50)
    logger.info("Summary:")
    logger.info(f"Total Python files processed: {len(python_files)}")
    logger.info(f"Old service references: {total_issues['old_service_references']}")
    logger.info(f"Unused imports: {total_issues['unused_imports']}")
    logger.info(f"Inconsistent error handling: {total_issues['inconsistent_error_handling']}")
    logger.info(f"Incomplete docstrings: {total_issues['incomplete_docstrings']}")
    
    if fix:
        logger.info(f"Fixed files: {total_issues['fixed_files']}")
    
    logger.info("=" * 50)

if __name__ == "__main__":
    main()
