"""
Check docstrings in Python files.

This script checks Python files for missing or incomplete docstrings.
It uses the AST module to parse Python files and check for docstrings
in modules, classes, and functions.

Usage:
    python -m asf.medical.scripts.check_docstrings [directory]
"""

import ast
import os
import sys
import argparse
import logging
from typing import List, Tuple, Dict, Any, Optional, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("check-docstrings")

# Directories to exclude
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

# Files to exclude
EXCLUDE_FILES = [
    "__init__.py",
    "conftest.py",
]

# Patterns for finding incomplete docstrings
TODO_PATTERNS = [
    "# TODO:",
    "# FIXME:",
    "# NOTE:",
    "# XXX:",
]


class DocstringVisitor(ast.NodeVisitor):
    """AST visitor to check for docstrings."""

    def __init__(self):
        """Initialize the visitor.

        This visitor tracks missing and incomplete docstrings in the AST.
        """
        self.missing_docstrings = []
        self.incomplete_docstrings = []

    def visit_Module(self, node):
        """Visit a module node.

        Args:
            node: The module node to visit
        """
        # Check for module docstring
        docstring = ast.get_docstring(node)
        if not docstring:
            self.missing_docstrings.append(("module", None, 1))
        elif self._is_incomplete_docstring(docstring):
            self.incomplete_docstrings.append(("module", None, 1, docstring))

        # Visit all nodes in the module
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        """Visit a class definition node.

        Args:
            node: The class definition node to visit
        """
        # Check for class docstring
        docstring = ast.get_docstring(node)
        if not docstring:
            self.missing_docstrings.append(("class", node.name, node.lineno))
        elif self._is_incomplete_docstring(docstring):
            self.incomplete_docstrings.append(("class", node.name, node.lineno, docstring))

        # Visit all nodes in the class
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        """Visit a function definition node.

        Args:
            node: The function definition node to visit
        """
        # Skip special methods
        if node.name.startswith("__") and node.name.endswith("__"):
            self.generic_visit(node)
            return

        # Check for function docstring
        docstring = ast.get_docstring(node)
        if not docstring:
            self.missing_docstrings.append(("function", node.name, node.lineno))
        elif self._is_incomplete_docstring(docstring, node):
            self.incomplete_docstrings.append(("function", node.name, node.lineno, docstring))

        # Visit all nodes in the function
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        """Visit an async function definition node.

        Args:
            node: The async function definition node to visit
        """
        # Skip special methods
        if node.name.startswith("__") and node.name.endswith("__"):
            self.generic_visit(node)
            return

        # Check for function docstring
        docstring = ast.get_docstring(node)
        if not docstring:
            self.missing_docstrings.append(("async function", node.name, node.lineno))
        elif self._is_incomplete_docstring(docstring, node):
            self.incomplete_docstrings.append(("async function", node.name, node.lineno, docstring))

        # Visit all nodes in the function
        self.generic_visit(node)

    def _is_incomplete_docstring(self, docstring: str, node: Optional[ast.AST] = None) -> bool:
        """Check if a docstring is incomplete.

        Args:
            docstring: The docstring to check
            node: The AST node (optional)

        Returns:
            True if the docstring is incomplete, False otherwise
        """
        # Check for TODO patterns
        for pattern in TODO_PATTERNS:
            if pattern in docstring:
                return True

        # Check for empty docstring
        if not docstring.strip():
            return True

        # Check for function parameters
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Get parameter names
            param_names = [arg.arg for arg in node.args.args if arg.arg != "self" and arg.arg != "cls"]

            # Check if docstring mentions all parameters
            if param_names and "Args:" not in docstring and "Parameters:" not in docstring:
                return True

            # Check for return value
            if "return" in ast.unparse(node).lower() and "Returns:" not in docstring:
                return True

        return False


def check_file(file_path: str) -> Tuple[List[Tuple], List[Tuple]]:
    """Check a file for docstrings.

    Args:
        file_path: Path to the file to check

    Returns:
        Tuple of missing docstrings and incomplete docstrings
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse the file
        tree = ast.parse(content, filename=file_path)

        # Visit the AST
        visitor = DocstringVisitor()
        visitor.visit(tree)

        return visitor.missing_docstrings, visitor.incomplete_docstrings
    except SyntaxError as e:
        logger.error(f"Syntax error in {file_path}: {e}")
        return [], []
    except Exception as e:
        logger.error(f"Error checking {file_path}: {e}")
        return [], []


def check_directory(
    directory: str,
    exclude_dirs: Optional[List[str]] = None,
    exclude_files: Optional[List[str]] = None
) -> Dict[str, Dict[str, List[Tuple]]]:
    """Check a directory for docstrings.

    Args:
        directory: Directory to check
        exclude_dirs: Directories to exclude
        exclude_files: Files to exclude

    Returns:
        Dictionary of files with missing or incomplete docstrings
    """
    exclude_dirs = exclude_dirs or EXCLUDE_DIRS
    exclude_files = exclude_files or EXCLUDE_FILES

    results = {}

    for root, dirs, files in os.walk(directory):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        for file in files:
            if file.endswith(".py") and file not in exclude_files:
                file_path = os.path.join(root, file)
                missing, incomplete = check_file(file_path)

                if missing or incomplete:
                    results[file_path] = {
                        "missing": missing,
                        "incomplete": incomplete
                    }

    return results


def print_report(results: Dict[str, Dict[str, List[Tuple]]]) -> None:
    """Print a report of missing and incomplete docstrings.

    Args:
        results: Dictionary of files with missing or incomplete docstrings
    """
    if not results:
        logger.info("No missing or incomplete docstrings found.")
        return

    total_missing = sum(len(file_results["missing"]) for file_results in results.values())
    total_incomplete = sum(len(file_results["incomplete"]) for file_results in results.values())

    logger.info(f"Found {total_missing} missing and {total_incomplete} incomplete docstrings in {len(results)} files.")

    for file_path, file_results in results.items():
        if file_results["missing"]:
            logger.info(f"\n{file_path} - Missing docstrings:")
            for item_type, name, lineno in file_results["missing"]:
                if name:
                    logger.info(f"  Line {lineno}: {item_type} '{name}'")
                else:
                    logger.info(f"  Line {lineno}: {item_type}")

        if file_results["incomplete"]:
            logger.info(f"\n{file_path} - Incomplete docstrings:")
            for item_type, name, lineno, docstring in file_results["incomplete"]:
                if name:
                    logger.info(f"  Line {lineno}: {item_type} '{name}'")
                else:
                    logger.info(f"  Line {lineno}: {item_type}")

                # Print the first line of the docstring
                first_line = docstring.strip().split("\n")[0]
                logger.info(f"    {first_line}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Check Python files for missing or incomplete docstrings")
    parser.add_argument("directory", nargs="?", default=".", help="Directory to check")
    parser.add_argument("--exclude-dirs", "-d", nargs="*", help="Directories to exclude")
    parser.add_argument("--exclude-files", "-f", nargs="*", help="Files to exclude")
    parser.add_argument("--output", "-o", help="Output file path")

    args = parser.parse_args()

    logger.info(f"Checking {args.directory}...")

    results = check_directory(
        args.directory,
        exclude_dirs=args.exclude_dirs,
        exclude_files=args.exclude_files
    )

    print_report(results)

    if args.output:
        import json
        with open(args.output, "w", encoding="utf-8") as f:
            # Convert tuples to lists for JSON serialization
            json_results = {}
            for file_path, file_results in results.items():
                json_results[file_path] = {
                    "missing": [list(item) for item in file_results["missing"]],
                    "incomplete": [list(item) for item in file_results["incomplete"]]
                }
            json.dump(json_results, f, indent=2)
        logger.info(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()