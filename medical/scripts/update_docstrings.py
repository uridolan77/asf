"""
Update Docstrings Script for ASF Medical Research Synthesizer.

This script scans the codebase for incomplete docstrings (containing TODO placeholders)
and updates them with proper documentation based on the defined standards. It identifies
docstrings that need improvement in modules, classes, and functions, and replaces them
with more complete versions that include proper sections for Args, Returns, and Raises.

Usage:
    python -m asf.medical.scripts.update_docstrings [--dry-run] [--directory DIR]

Options:
    --dry-run       Don't actually update files, just show what would be done
    --directory DIR  Directory to process (default: asf/medical)

The script excludes certain directories and files from processing to avoid modifying
generated code, third-party libraries, and configuration files.
"""
import os
import re
import ast
import argparse
import logging
from typing import List, Dict, Any
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("UpdateDocstrings")
# Patterns for finding incomplete docstrings
TODO_PATTERNS = [
    r"# TODO: Add parameter descriptions",
    r"# TODO: Add return description",
    r"# TODO: Add docstring",
    r"# TODO: Update docstring",
]
# Files to exclude from processing
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
]
def find_python_files(directory: str) -> List[str]:
    """
    Find all Python files in the given directory and its subdirectories.

    This function recursively walks through the specified directory and collects
    paths to all Python files (.py extension) that are not in excluded directories
    and are not excluded files.

    Args:
        directory: The directory to search in.

    Returns:
        List of paths to Python files that should be processed.
    """
    python_files = []
    for root, dirs, files in os.walk(directory):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for file in files:
            if file.endswith(".py") and file not in EXCLUDE_FILES:
                python_files.append(os.path.join(root, file))
    return python_files
def extract_docstring_info(file_path: str) -> List[Dict[str, Any]]:
    """
    Extract information about docstrings in a Python file.

    This function parses a Python file into an AST and extracts information about
    docstrings that need to be updated. It identifies module, class, and function
    docstrings that contain TODO patterns indicating they are incomplete.

    Args:
        file_path: Path to the Python file to analyze.

    Returns:
        List of dictionaries containing docstring information, including:
        - type: 'module', 'class', or 'function'
        - name: Name of the module, class, or function
        - docstring: Current docstring content
        - lineno: Starting line number of the docstring
        - end_lineno: Ending line number of the docstring
        - needs_update: Boolean indicating if the docstring needs updating
        - args: List of argument names (for functions only)
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    try:
        tree = ast.parse(content)
    except SyntaxError:
        logger.error(f"Syntax error in {file_path}")
        return []
    docstring_info = []
    # Extract module docstring
    if (len(tree.body) > 0 and
            isinstance(tree.body[0], ast.Expr) and
            (isinstance(tree.body[0].value, ast.Constant) or
             isinstance(tree.body[0].value, ast.Str))):
        # Handle both ast.Str (Python < 3.8) and ast.Constant (Python >= 3.8)
        if hasattr(tree.body[0].value, 's'):
            docstring = tree.body[0].value.s  # ast.Str
        else:
            docstring = tree.body[0].value.value  # ast.Constant
        if any(re.search(pattern, docstring) for pattern in TODO_PATTERNS):
            docstring_info.append({
                "type": "module",
                "name": os.path.basename(file_path),
                "docstring": docstring,
                "lineno": tree.body[0].lineno,
                "end_lineno": tree.body[0].end_lineno,
                "needs_update": True,
            })
    # Extract class and function docstrings
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            docstring = ast.get_docstring(node)
            if docstring and any(re.search(pattern, docstring) for pattern in TODO_PATTERNS):
                docstring_info.append({
                    "type": "class" if isinstance(node, ast.ClassDef) else "function",
                    "name": node.name,
                    "docstring": docstring,
                    "lineno": node.lineno,
                    "end_lineno": node.end_lineno,
                    "needs_update": True,
                    "args": [arg.arg for arg in node.args.args] if hasattr(node, "args") else [],
                })
    return docstring_info
def update_docstring(file_path: str, docstring_info: Dict[str, Any]) -> bool:
    """
    Update a docstring in a file.

    This function reads a Python file, locates the docstring specified in the
    docstring_info dictionary, generates a new docstring, and replaces the old
    one with the new one. It then writes the updated content back to the file.

    Args:
        file_path: Path to the Python file to update.
        docstring_info: Dictionary containing docstring information, including
            type, name, lineno, and other details needed to locate and update
            the docstring.

    Returns:
        True if the docstring was successfully updated, False otherwise.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # Find the docstring lines
    docstring_lines = []
    in_docstring = False
    docstring_start = None
    docstring_end = None
    for i, line in enumerate(lines):
        if i + 1 >= docstring_info["lineno"]:
            if not in_docstring and '"""' in line:
                in_docstring = True
                docstring_start = i
            elif in_docstring and '"""' in line:
                in_docstring = False
                docstring_end = i
                break
            if in_docstring:
                docstring_lines.append(line)
    if docstring_start is None or docstring_end is None:
        logger.warning(f"Could not find docstring for {docstring_info['name']} in {file_path}")
        return False
    # Generate new docstring
    new_docstring = generate_docstring(docstring_info)
    # Replace old docstring with new one
    lines[docstring_start:docstring_end + 1] = new_docstring.splitlines(True)
    # Write updated file
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return True
def generate_docstring(docstring_info: Dict[str, Any]) -> str:
    """
    Generate a new docstring based on the docstring information.

    This function creates a new docstring for a module, class, or function based
    on the information provided in the docstring_info dictionary. It generates
    appropriate sections (Args, Returns, Raises) based on the type of entity and
    available information.

    Args:
        docstring_info: Dictionary containing docstring information, including
            type ('module', 'class', or 'function'), name, and for functions,
            a list of argument names.

    Returns:
        A string containing the generated docstring, properly formatted with
        appropriate indentation and triple quotes.
    """
    if docstring_info["type"] == "module":
        # Generate module docstring
        return f'"""\n{docstring_info["name"]}\n\nThis module provides functionality for working with {docstring_info["name"]}.\n"""\n'
    elif docstring_info["type"] == "class":
        # Generate class docstring
        return f'    """\n    {docstring_info["name"]} class.\n    \n    This class provides functionality for working with {docstring_info["name"]}.\n    """\n'
    elif docstring_info["type"] == "function":
        # Generate function docstring
        args_section = ""
        for arg in docstring_info["args"]:
            if arg != "self" and arg != "cls":
                args_section += f"        {arg}: Description of {arg}.\n"
        if not args_section:
            args_section = "        This function takes no arguments.\n"
        return f'    """\n    {docstring_info["name"]} function.\n    \n    Args:\n{args_section}\n    Returns:\n        Description of return value.\n    \n    Raises:\n        Exception: Description of when this exception is raised.\n    """\n'
    return docstring_info["docstring"]
def process_file(file_path: str, dry_run: bool = False) -> int:
    """
    Process a file and update incomplete docstrings.

    This function analyzes a Python file to identify docstrings that need to be
    updated, and either updates them or reports what would be updated (in dry run
    mode). It handles exceptions gracefully to ensure the script continues running
    even if one file has issues.

    Args:
        file_path: Path to the Python file to process.
        dry_run: If True, don't actually update the file, just report what would
            be updated.

    Returns:
        Number of docstrings that were updated or would be updated (in dry run mode).
    """
    try:
        docstring_info_list = extract_docstring_info(file_path)
        if not docstring_info_list:
            return 0
        updated_count = 0
        for docstring_info in docstring_info_list:
            if docstring_info["needs_update"]:
                if dry_run:
                    logger.info(f"Would update {docstring_info['type']} docstring for {docstring_info['name']} in {file_path}")
                    updated_count += 1
                else:
                    if update_docstring(file_path, docstring_info):
                        logger.info(f"Updated {docstring_info['type']} docstring for {docstring_info['name']} in {file_path}")
                        updated_count += 1
        return updated_count
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return 0
def main():
    """
    Main function to run the docstring update script.

    This function parses command-line arguments, finds Python files in the specified
    directory, processes each file to update incomplete docstrings, and reports a
    summary of the results.

    Command-line arguments:
        --dry-run: Don't actually update files, just show what would be done
        --directory: Directory to process (default: asf/medical)

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Update incomplete docstrings in the ASF Medical Research Synthesizer codebase")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually update files, just show what would be done")
    parser.add_argument("--directory", default=None, help="Directory to process (default: asf/medical)")
    args = parser.parse_args()
    if args.directory:
        directory = args.directory
    else:
        directory = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "medical")
    logger.info(f"Scanning directory: {directory}")
    python_files = find_python_files(directory)
    logger.info(f"Found {len(python_files)} Python files")
    updated_files = 0
    updated_docstrings = 0
    for file in python_files:
        file_updated_count = process_file(file, args.dry_run)
        if file_updated_count > 0:
            updated_files += 1
            updated_docstrings += file_updated_count
    logger.info(f"Updated {updated_docstrings} docstrings in {updated_files} files")
    if args.dry_run:
        logger.info("This was a dry run. No files were actually modified.")
if __name__ == "__main__":
    main()