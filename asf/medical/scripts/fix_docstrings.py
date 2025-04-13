"""
Fix Incomplete Docstrings in ASF Medical Codebase.

This script analyzes Python files in the ASF Medical codebase to identify missing
or incomplete docstrings and optionally fixes them by adding or improving docstrings.
It supports module, class, and function docstrings, and can generate appropriate
docstring content based on the context (function arguments, return values, etc.).

Usage:
    python -m asf.medical.scripts.fix_docstrings <directory> [--fix]

Arguments:
    directory: Path to the directory to process
    --fix: Optional flag to actually fix the docstrings (otherwise just reports issues)

The script excludes certain directories and files from processing, and focuses on
public functions and methods (skipping those that start with an underscore).
"""
import os
import sys
import logging
import ast
from typing import List, Dict
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("FixDocstrings")
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
    "fix_docstrings.py",
    "deep_cleanup_phase2.py",
    "fix_unused_imports.py"
]
def find_python_files(directory: str) -> List[str]:
    """Find all Python files in the given directory and its subdirectories.

    This function recursively walks through the specified directory and collects
    paths to all Python files (.py extension) that are not in excluded directories
    and are not excluded files.

    Args:
        directory: Path to the directory to search

    Returns:
        List of paths to Python files that should be processed
    """
    python_files = []
    for root, dirs, files in os.walk(directory):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for file in files:
            if file.endswith(".py") and file not in EXCLUDE_FILES:
                python_files.append(os.path.join(root, file))
    return python_files
class DocstringVisitor(ast.NodeVisitor):
    """AST visitor to find nodes with missing or incomplete docstrings.

    This class extends ast.NodeVisitor to traverse the abstract syntax tree of a
    Python file and identify modules, classes, and functions that have missing or
    incomplete docstrings. It maintains separate lists for missing and incomplete
    docstrings, which can be used to generate reports or fix the issues.
    """
    def __init__(self):
        self.missing_docstrings = []
        self.incomplete_docstrings = []
    def visit_Module(self, node):
        """Visit a module node."""
        # Check if module has a docstring
        if not ast.get_docstring(node):
            self.missing_docstrings.append((node, "module"))
        elif not self._is_complete_docstring(ast.get_docstring(node)):
            self.incomplete_docstrings.append((node, "module"))
        self.generic_visit(node)
    def visit_ClassDef(self, node):
        """Visit a class definition node."""
        # Check if class has a docstring
        if not ast.get_docstring(node):
            self.missing_docstrings.append((node, "class"))
        elif not self._is_complete_docstring(ast.get_docstring(node)):
            self.incomplete_docstrings.append((node, "class"))
        self.generic_visit(node)
    def visit_FunctionDef(self, node):
        """Visit a function definition node."""
        # Skip private methods and functions
        if node.name.startswith("_") and node.name != "__init__":
            self.generic_visit(node)
            return
        # Check if function has a docstring
        if not ast.get_docstring(node):
            self.missing_docstrings.append((node, "function"))
        elif not self._is_complete_docstring(ast.get_docstring(node)):
            self.incomplete_docstrings.append((node, "function"))
        self.generic_visit(node)
    def _is_complete_docstring(self, docstring: str) -> bool:
        """Check if a docstring is complete.

        A docstring is considered complete if it has at least one of the following sections:
        - Args or Parameters: Describing the function/method parameters
        - Returns: Describing the return value
        - Raises: Describing exceptions that might be raised

        Args:
            docstring: The docstring to check

        Returns:
            True if the docstring is considered complete, False otherwise
        """
        if not docstring:
            return False
        # Check if docstring has sections
        has_args = "Args:" in docstring or "Parameters:" in docstring
        has_returns = "Returns:" in docstring
        has_raises = "Raises:" in docstring
        # A docstring is considered complete if it has at least one section
        return has_args or has_returns or has_raises
def generate_docstring(node: ast.AST, node_type: str) -> str:
    """Generate a docstring for a node.

    This function generates an appropriate docstring for a module, class, or function
    based on the node type and available information (such as function arguments and
    return values). The generated docstring includes placeholders that should be
    filled in by the developer.

    Args:
        node: The AST node for which to generate a docstring
        node_type: The type of the node ('module', 'class', or 'function')

    Returns:
        A string containing the generated docstring
    """
    if node_type == "module":
        return '"""\nModule description.\n\nThis module provides functionality for...\n"""'
    elif node_type == "class":
        class_name = node.name if hasattr(node, "name") else "Unknown"
        return f'"""\n{class_name} class.\n\nThis class provides functionality for...\n"""'
    elif node_type == "function":
        func_name = node.name if hasattr(node, "name") else "Unknown"
        # Generate Args section if function has arguments
        args_section = ""
        if hasattr(node, "args") and node.args.args:
            args_section = "\nArgs:\n"
            for arg in node.args.args:
                if arg.arg != "self":
                    args_section += f"    {arg.arg}: Description of {arg.arg}\n"
        # Generate Returns section
        returns_section = ""
        if hasattr(node, "returns") and node.returns:
            returns_section = "\nReturns:\n    Description of return value\n"
        else:
            # Check if function name suggests it returns something
            if func_name.startswith(("get_", "find_", "calculate_", "compute_", "is_", "has_")):
                returns_section = "\nReturns:\n    Description of return value\n"
        return f'"""\n{func_name} function.\n\nThis function provides functionality for...{args_section}{returns_section}"""'
    return '"""Docstring."""'
def improve_docstring(node: ast.AST, node_type: str, existing_docstring: str) -> str:
    """Improve an existing docstring.

    This function enhances an existing docstring by adding missing sections such as
    Args and Returns. It preserves the existing content while adding the new sections
    in appropriate locations. For functions, it analyzes the arguments and function
    name to determine what sections should be added.

    Args:
        node: The AST node containing the docstring
        node_type: The type of the node ('module', 'class', or 'function')
        existing_docstring: The current docstring content

    Returns:
        An improved version of the docstring with added sections as needed
    """
    if not existing_docstring:
        return generate_docstring(node, node_type)
    # Split docstring into lines
    lines = existing_docstring.split("\n")
    # Check if docstring has Args section
    has_args = any("Args:" in line for line in lines)
    has_parameters = any("Parameters:" in line for line in lines)
    # Check if docstring has Returns section
    has_returns = any("Returns:" in line for line in lines)
    # Check if docstring has Raises section
    # This variable is currently not used but kept for future enhancements
    _ = any("Raises:" in line for line in lines)
    # Improve docstring based on node type
    if node_type == "function":
        # Add Args section if missing and function has arguments
        if not has_args and not has_parameters and hasattr(node, "args") and node.args.args:
            args_section = "\nArgs:\n"
            for arg in node.args.args:
                if arg.arg != "self":
                    args_section += f"    {arg.arg}: Description of {arg.arg}\n"
            # Add Args section before Returns section if it exists
            if has_returns:
                returns_index = next((i for i, line in enumerate(lines) if "Returns:" in line), -1)
                if returns_index > 0:
                    lines.insert(returns_index, args_section)
                else:
                    lines.append(args_section)
            else:
                lines.append(args_section)
        # Add Returns section if missing and function likely returns something
        if not has_returns:
            func_name = node.name if hasattr(node, "name") else "Unknown"
            if hasattr(node, "returns") and node.returns:
                lines.append("\nReturns:\n    Description of return value")
            elif func_name.startswith(("get_", "find_", "calculate_", "compute_", "is_", "has_")):
                lines.append("\nReturns:\n    Description of return value")
    return "\n".join(lines)
def fix_docstrings(file_path: str, fix: bool = False) -> Dict[str, int]:
    """Fix docstrings in a Python file.

    This function analyzes a Python file to identify missing or incomplete docstrings
    and optionally fixes them. It parses the file into an AST, uses DocstringVisitor
    to identify issues, and then modifies the file content to add or improve docstrings.

    Args:
        file_path: Path to the Python file to process
        fix: If True, modify the file to fix docstring issues; if False, just report issues

    Returns:
        A dictionary with the following keys:
        - 'missing_docstrings': Number of missing docstrings found
        - 'incomplete_docstrings': Number of incomplete docstrings found
        - 'fixed': Boolean indicating whether the file was modified
    """
    results = {
        "missing_docstrings": 0,
        "incomplete_docstrings": 0,
        "fixed": False
    }
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        # Parse the file
        tree = ast.parse(content)
        # Find missing and incomplete docstrings
        visitor = DocstringVisitor()
        visitor.visit(tree)
        results["missing_docstrings"] = len(visitor.missing_docstrings)
        results["incomplete_docstrings"] = len(visitor.incomplete_docstrings)
        # Fix docstrings if requested
        if fix and (visitor.missing_docstrings or visitor.incomplete_docstrings):
            # Sort nodes by line number in reverse order to avoid changing positions
            all_nodes = [(node, node_type) for node, node_type in visitor.missing_docstrings + visitor.incomplete_docstrings]
            all_nodes.sort(key=lambda x: x[0].lineno if hasattr(x[0], "lineno") else 0, reverse=True)
            # Fix docstrings
            lines = content.split("\n")
            for node, node_type in all_nodes:
                if node_type == "module":
                    # Add module docstring at the beginning of the file
                    if not ast.get_docstring(node):
                        lines.insert(0, generate_docstring(node, node_type))
                    else:
                        # Improve existing module docstring
                        existing_docstring = ast.get_docstring(node)
                        improved_docstring = improve_docstring(node, node_type, existing_docstring)
                        # Replace existing docstring
                        docstring_start = 0
                        for i, line in enumerate(lines):
                            if '"""' in line:
                                docstring_start = i
                                break
                        docstring_end = docstring_start
                        for i in range(docstring_start + 1, len(lines)):
                            if '"""' in lines[i]:
                                docstring_end = i
                                break
                        # Replace docstring
                        lines = lines[:docstring_start] + [improved_docstring] + lines[docstring_end + 1:]
                elif hasattr(node, "lineno"):
                    # Add docstring after the class or function definition
                    lineno = node.lineno - 1  # Convert to 0-based index
                    # Find the line with the colon
                    colon_line = lineno
                    while colon_line < len(lines) and ":" not in lines[colon_line]:
                        colon_line += 1
                    if colon_line < len(lines):
                        # Find the indentation level
                        indent = len(lines[colon_line]) - len(lines[colon_line].lstrip())
                        indent_str = " " * indent
                        # Generate or improve docstring
                        if not ast.get_docstring(node):
                            # Add new docstring
                            docstring = generate_docstring(node, node_type)
                            docstring_lines = [indent_str + "    " + line for line in docstring.split("\n")]
                            lines.insert(colon_line + 1, "\n".join(docstring_lines))
                        else:
                            # Improve existing docstring
                            existing_docstring = ast.get_docstring(node)
                            improved_docstring = improve_docstring(node, node_type, existing_docstring)
                            # Find the existing docstring
                            docstring_start = colon_line + 1
                            while docstring_start < len(lines) and '"""' not in lines[docstring_start]:
                                docstring_start += 1
                            if docstring_start < len(lines):
                                docstring_end = docstring_start
                                for i in range(docstring_start + 1, len(lines)):
                                    if '"""' in lines[i]:
                                        docstring_end = i
                                        break
                                # Replace docstring
                                improved_docstring_lines = [indent_str + "    " + line for line in improved_docstring.split("\n")]
                                lines = lines[:docstring_start] + improved_docstring_lines + lines[docstring_end + 1:]
            # Write updated content
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            results["fixed"] = True
        return results
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return results
def main():
    """Main entry point for the docstring fixing script.

    This function parses command-line arguments, finds Python files in the specified
    directory, processes each file to identify and optionally fix docstring issues,
    and reports a summary of the results.

    Command-line arguments:
        <directory>: Directory to process
        --fix: Optional flag to actually fix the docstrings

    Returns:
        None, but exits with a non-zero code if there's an error
    """
    if len(sys.argv) < 2:
        print("Usage: python fix_docstrings.py <directory> [--fix]")
        sys.exit(1)
    directory = sys.argv[1]
    fix = "--fix" in sys.argv
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a directory")
        sys.exit(1)
    logger.info(f"Starting fix docstrings for {directory}")
    logger.info(f"Fix mode: {fix}")
    python_files = find_python_files(directory)
    logger.info(f"Found {len(python_files)} Python files")
    total_missing_docstrings = 0
    total_incomplete_docstrings = 0
    fixed_files = 0
    for file_path in python_files:
        logger.info(f"Processing {file_path}")
        results = fix_docstrings(file_path, fix)
        # Count issues
        total_missing_docstrings += results["missing_docstrings"]
        total_incomplete_docstrings += results["incomplete_docstrings"]
        if results["fixed"]:
            fixed_files += 1
            logger.info(f"Fixed docstrings in {file_path}")
    # Report summary
    logger.info("=" * 50)
    logger.info("Summary:")
    logger.info(f"Total Python files processed: {len(python_files)}")
    logger.info(f"Total missing docstrings: {total_missing_docstrings}")
    logger.info(f"Total incomplete docstrings: {total_incomplete_docstrings}")
    if fix:
        logger.info(f"Fixed files: {fixed_files}")
    logger.info("=" * 50)
if __name__ == "__main__":
    main()