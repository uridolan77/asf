#!/usr/bin/env python
"""
Script to check for detailed and consistent docstrings in the codebase.

This script scans the codebase for functions, methods, classes, and modules that are
missing docstrings or have incomplete docstrings. It checks for:
- Missing module docstrings
- Missing class docstrings
- Missing function/method docstrings
- Missing parameter descriptions in docstrings
- Missing return value descriptions in docstrings
- Inconsistent docstring styles (Google, NumPy, reStructuredText)

Usage:
    python -m asf.medical.scripts.check_docstrings
"""

import os
import ast
import sys
import re
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("check-docstrings")


class DocstringVisitor(ast.NodeVisitor):
    """AST visitor to check for docstrings."""
    
    def __init__(self):
        """Initialize the visitor."""
        self.missing_docstrings = []
        self.incomplete_docstrings = []
        self.current_file = None
    
    def set_file(self, file_path: str) -> None:
        """Set the current file being processed."""
        self.current_file = file_path
    
    def visit_Module(self, node: ast.Module) -> None:
        """Visit a module."""
        # Check module docstring
        if not ast.get_docstring(node):
            self.missing_docstrings.append({
                "file": self.current_file,
                "line": 1,
                "type": "module",
                "name": self.current_file,
                "message": f"Module is missing a docstring"
            })
        
        self.generic_visit(node)
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit a class definition."""
        # Check class docstring
        docstring = ast.get_docstring(node)
        if not docstring:
            self.missing_docstrings.append({
                "file": self.current_file,
                "line": node.lineno,
                "type": "class",
                "name": node.name,
                "message": f"Class '{node.name}' is missing a docstring"
            })
        else:
            # Check docstring completeness
            self._check_class_docstring_completeness(node, docstring)
        
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit a function definition."""
        # Skip special methods
        if node.name.startswith("__") and node.name.endswith("__") and node.name != "__init__":
            self.generic_visit(node)
            return
        
        # Check function docstring
        docstring = ast.get_docstring(node)
        if not docstring:
            self.missing_docstrings.append({
                "file": self.current_file,
                "line": node.lineno,
                "type": "function",
                "name": node.name,
                "message": f"Function '{node.name}' is missing a docstring"
            })
        else:
            # Check docstring completeness
            self._check_function_docstring_completeness(node, docstring)
        
        self.generic_visit(node)
    
    def _check_class_docstring_completeness(self, node: ast.ClassDef, docstring: str) -> None:
        """
        Check if a class docstring is complete.
        
        Args:
            node: Class node
            docstring: Class docstring
        """
        # Check if docstring is just a placeholder
        if len(docstring.strip().split("\n")) <= 1 and len(docstring.strip()) < 20:
            self.incomplete_docstrings.append({
                "file": self.current_file,
                "line": node.lineno,
                "type": "class",
                "name": node.name,
                "message": f"Class '{node.name}' has a placeholder docstring"
            })
    
    def _check_function_docstring_completeness(self, node: ast.FunctionDef, docstring: str) -> None:
        """
        Check if a function docstring is complete.
        
        Args:
            node: Function node
            docstring: Function docstring
        """
        # Check if docstring is just a placeholder
        if len(docstring.strip().split("\n")) <= 1 and len(docstring.strip()) < 20:
            self.incomplete_docstrings.append({
                "file": self.current_file,
                "line": node.lineno,
                "type": "function",
                "name": node.name,
                "message": f"Function '{node.name}' has a placeholder docstring"
            })
            return
        
        # Get parameter names
        param_names = [arg.arg for arg in node.args.args if arg.arg != "self" and arg.arg != "cls"]
        if node.args.vararg:
            param_names.append(f"*{node.args.vararg.arg}")
        if node.args.kwarg:
            param_names.append(f"**{node.args.kwarg.arg}")
        
        # Check docstring style
        if re.search(r"Args:", docstring):
            # Google style
            self._check_google_style_docstring(node, docstring, param_names)
        elif re.search(r"Parameters", docstring):
            # NumPy style
            self._check_numpy_style_docstring(node, docstring, param_names)
        elif re.search(r":param", docstring):
            # reStructuredText style
            self._check_rest_style_docstring(node, docstring, param_names)
        else:
            # Unknown style
            self.incomplete_docstrings.append({
                "file": self.current_file,
                "line": node.lineno,
                "type": "function",
                "name": node.name,
                "message": f"Function '{node.name}' has a docstring with unknown style"
            })
    
    def _check_google_style_docstring(self, node: ast.FunctionDef, docstring: str, param_names: List[str]) -> None:
        """
        Check if a Google style docstring is complete.
        
        Args:
            node: Function node
            docstring: Function docstring
            param_names: Parameter names
        """
        # Check for Args section
        if param_names and "Args:" not in docstring:
            self.incomplete_docstrings.append({
                "file": self.current_file,
                "line": node.lineno,
                "type": "function",
                "name": node.name,
                "message": f"Function '{node.name}' is missing Args section in docstring"
            })
            return
        
        # Check for Returns section
        if node.returns and "Returns:" not in docstring:
            self.incomplete_docstrings.append({
                "file": self.current_file,
                "line": node.lineno,
                "type": "function",
                "name": node.name,
                "message": f"Function '{node.name}' is missing Returns section in docstring"
            })
        
        # Check for parameter descriptions
        if param_names:
            args_section = docstring.split("Args:")[1].split("\n\n")[0]
            for param in param_names:
                if param not in args_section:
                    self.incomplete_docstrings.append({
                        "file": self.current_file,
                        "line": node.lineno,
                        "type": "function",
                        "name": node.name,
                        "message": f"Function '{node.name}' is missing description for parameter '{param}'"
                    })
    
    def _check_numpy_style_docstring(self, node: ast.FunctionDef, docstring: str, param_names: List[str]) -> None:
        """
        Check if a NumPy style docstring is complete.
        
        Args:
            node: Function node
            docstring: Function docstring
            param_names: Parameter names
        """
        # Check for Parameters section
        if param_names and "Parameters" not in docstring:
            self.incomplete_docstrings.append({
                "file": self.current_file,
                "line": node.lineno,
                "type": "function",
                "name": node.name,
                "message": f"Function '{node.name}' is missing Parameters section in docstring"
            })
            return
        
        # Check for Returns section
        if node.returns and "Returns" not in docstring:
            self.incomplete_docstrings.append({
                "file": self.current_file,
                "line": node.lineno,
                "type": "function",
                "name": node.name,
                "message": f"Function '{node.name}' is missing Returns section in docstring"
            })
        
        # Check for parameter descriptions
        if param_names:
            params_section = docstring.split("Parameters")[1].split("\n\n")[0]
            for param in param_names:
                if param not in params_section:
                    self.incomplete_docstrings.append({
                        "file": self.current_file,
                        "line": node.lineno,
                        "type": "function",
                        "name": node.name,
                        "message": f"Function '{node.name}' is missing description for parameter '{param}'"
                    })
    
    def _check_rest_style_docstring(self, node: ast.FunctionDef, docstring: str, param_names: List[str]) -> None:
        """
        Check if a reStructuredText style docstring is complete.
        
        Args:
            node: Function node
            docstring: Function docstring
            param_names: Parameter names
        """
        # Check for parameter descriptions
        for param in param_names:
            if f":param {param}:" not in docstring and f":param: {param}" not in docstring:
                self.incomplete_docstrings.append({
                    "file": self.current_file,
                    "line": node.lineno,
                    "type": "function",
                    "name": node.name,
                    "message": f"Function '{node.name}' is missing description for parameter '{param}'"
                })
        
        # Check for return description
        if node.returns and ":return:" not in docstring and ":returns:" not in docstring:
            self.incomplete_docstrings.append({
                "file": self.current_file,
                "line": node.lineno,
                "type": "function",
                "name": node.name,
                "message": f"Function '{node.name}' is missing return description in docstring"
            })


def check_file(file_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Check a file for docstrings.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Tuple of missing docstrings and incomplete docstrings
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        tree = ast.parse(content)
        visitor = DocstringVisitor()
        visitor.set_file(file_path)
        visitor.visit(tree)
        
        return visitor.missing_docstrings, visitor.incomplete_docstrings
    except SyntaxError as e:
        logger.warning(f"Syntax error in {file_path}: {e}")
        return [], []
    except Exception as e:
        logger.error(f"Error checking {file_path}: {e}")
        return [], []


def check_directory(directory: str, exclude_dirs: List[str] = None, exclude_files: List[str] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Check a directory for docstrings.
    
    Args:
        directory: Directory to check
        exclude_dirs: Directories to exclude
        exclude_files: Files to exclude
        
    Returns:
        Tuple of missing docstrings and incomplete docstrings
    """
    exclude_dirs = exclude_dirs or [
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
    exclude_files = exclude_files or [
        ".gitignore",
        ".env",
        "README.md",
        "LICENSE",
        "requirements.txt",
        "setup.py",
        "pyproject.toml",
    ]
    
    missing_docstrings = []
    incomplete_docstrings = []
    
    for root, dirs, files in os.walk(directory):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            # Skip non-Python files
            if not file.endswith(".py"):
                continue
            
            # Skip excluded files
            if file in exclude_files:
                continue
            
            file_path = os.path.join(root, file)
            missing, incomplete = check_file(file_path)
            missing_docstrings.extend(missing)
            incomplete_docstrings.extend(incomplete)
    
    return missing_docstrings, incomplete_docstrings


def print_report(missing_docstrings: List[Dict[str, Any]], incomplete_docstrings: List[Dict[str, Any]]) -> None:
    """
    Print a report of missing and incomplete docstrings.
    
    Args:
        missing_docstrings: List of missing docstrings
        incomplete_docstrings: List of incomplete docstrings
    """
    # Group by file
    missing_by_file = {}
    for docstring in missing_docstrings:
        if docstring["file"] not in missing_by_file:
            missing_by_file[docstring["file"]] = []
        missing_by_file[docstring["file"]].append(docstring)
    
    incomplete_by_file = {}
    for docstring in incomplete_docstrings:
        if docstring["file"] not in incomplete_by_file:
            incomplete_by_file[docstring["file"]] = []
        incomplete_by_file[docstring["file"]].append(docstring)
    
    # Print report
    print("\n=== Docstring Check Report ===\n")
    print(f"Total files with missing docstrings: {len(missing_by_file)}")
    print(f"Total missing docstrings: {len(missing_docstrings)}")
    print(f"Total files with incomplete docstrings: {len(incomplete_by_file)}")
    print(f"Total incomplete docstrings: {len(incomplete_docstrings)}")
    
    # Print by type
    missing_by_type = {}
    for docstring in missing_docstrings:
        if docstring["type"] not in missing_by_type:
            missing_by_type[docstring["type"]] = 0
        missing_by_type[docstring["type"]] += 1
    
    print("\nMissing docstrings by type:")
    for docstring_type, count in missing_by_type.items():
        print(f"  {docstring_type}: {count}")
    
    incomplete_by_type = {}
    for docstring in incomplete_docstrings:
        if docstring["type"] not in incomplete_by_type:
            incomplete_by_type[docstring["type"]] = 0
        incomplete_by_type[docstring["type"]] += 1
    
    print("\nIncomplete docstrings by type:")
    for docstring_type, count in incomplete_by_type.items():
        print(f"  {docstring_type}: {count}")
    
    # Print top files with missing docstrings
    print("\nTop files with missing docstrings:")
    for file, docstrings in sorted(missing_by_file.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
        print(f"  {file}: {len(docstrings)}")
    
    # Print top files with incomplete docstrings
    print("\nTop files with incomplete docstrings:")
    for file, docstrings in sorted(incomplete_by_file.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
        print(f"  {file}: {len(docstrings)}")
    
    # Print details of missing docstrings
    print("\nMissing docstrings details:")
    for file, docstrings in sorted(missing_by_file.items()):
        print(f"\n{file}:")
        for docstring in sorted(docstrings, key=lambda x: x["line"]):
            print(f"  Line {docstring['line']}: {docstring['message']}")
    
    # Print details of incomplete docstrings
    print("\nIncomplete docstrings details:")
    for file, docstrings in sorted(incomplete_by_file.items()):
        print(f"\n{file}:")
        for docstring in sorted(docstrings, key=lambda x: x["line"]):
            print(f"  Line {docstring['line']}: {docstring['message']}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Check for detailed and consistent docstrings in the codebase")
    parser.add_argument("--directory", "-d", default=".", help="Directory to check")
    parser.add_argument("--exclude-dirs", "-e", nargs="*", help="Directories to exclude")
    parser.add_argument("--exclude-files", "-f", nargs="*", help="Files to exclude")
    parser.add_argument("--output", "-o", help="Output file path")
    args = parser.parse_args()
    
    logger.info(f"Checking {args.directory}...")
    missing_docstrings, incomplete_docstrings = check_directory(
        args.directory,
        exclude_dirs=args.exclude_dirs,
        exclude_files=args.exclude_files,
    )
    
    print_report(missing_docstrings, incomplete_docstrings)
    
    if args.output:
        import json
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({
                "missing_docstrings": missing_docstrings,
                "incomplete_docstrings": incomplete_docstrings,
            }, f, indent=2)
        logger.info(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
