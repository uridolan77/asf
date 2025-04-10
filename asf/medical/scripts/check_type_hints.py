#!/usr/bin/env python
"""
Script to check for comprehensive type hinting in the codebase.

This script scans the codebase for functions and methods that are missing type hints
for parameters and return values. It also checks for variables that are missing type hints.

Usage:
    python -m asf.medical.scripts.check_type_hints
"""

import os
import ast
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("check-type-hints")


class TypeHintVisitor(ast.NodeVisitor):
    """AST visitor to check for type hints."""
    
    def __init__(self):
        """Initialize the visitor."""
        self.missing_type_hints = []
        self.current_file = None
    
    def set_file(self, file_path: str) -> None:
        """Set the current file being processed."""
        self.current_file = file_path
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit a function definition."""
        # Check return type annotation
        if node.returns is None:
            self.missing_type_hints.append({
                "file": self.current_file,
                "line": node.lineno,
                "type": "function_return",
                "name": node.name,
                "message": f"Function '{node.name}' is missing return type annotation"
            })
        
        # Check parameter type annotations
        for arg in node.args.args:
            if arg.annotation is None and arg.arg != "self" and arg.arg != "cls":
                self.missing_type_hints.append({
                    "file": self.current_file,
                    "line": node.lineno,
                    "type": "function_param",
                    "name": f"{node.name}.{arg.arg}",
                    "message": f"Parameter '{arg.arg}' of function '{node.name}' is missing type annotation"
                })
        
        # Check *args and **kwargs
        if node.args.vararg and node.args.vararg.annotation is None:
            self.missing_type_hints.append({
                "file": self.current_file,
                "line": node.lineno,
                "type": "function_param",
                "name": f"{node.name}.*args",
                "message": f"*args parameter of function '{node.name}' is missing type annotation"
            })
        
        if node.args.kwarg and node.args.kwarg.annotation is None:
            self.missing_type_hints.append({
                "file": self.current_file,
                "line": node.lineno,
                "type": "function_param",
                "name": f"{node.name}.**kwargs",
                "message": f"**kwargs parameter of function '{node.name}' is missing type annotation"
            })
        
        # Visit function body
        self.generic_visit(node)
    
    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Visit an annotated assignment."""
        # This is a variable with a type annotation, so we don't need to report it
        self.generic_visit(node)
    
    def visit_Assign(self, node: ast.Assign) -> None:
        """Visit an assignment."""
        # Check if this is a class or module level variable assignment
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            # Skip private variables
            if not node.targets[0].id.startswith("_"):
                # Skip constants (all uppercase)
                if not node.targets[0].id.isupper():
                    # Check if this is a simple assignment (not in a function)
                    parent = self._get_parent(node)
                    if isinstance(parent, ast.Module) or isinstance(parent, ast.ClassDef):
                        self.missing_type_hints.append({
                            "file": self.current_file,
                            "line": node.lineno,
                            "type": "variable",
                            "name": node.targets[0].id,
                            "message": f"Variable '{node.targets[0].id}' is missing type annotation"
                        })
        
        self.generic_visit(node)
    
    def _get_parent(self, node: ast.AST) -> Optional[ast.AST]:
        """Get the parent node of a node."""
        for parent in ast.walk(ast.Module(body=[node])):
            for child in ast.iter_child_nodes(parent):
                if child == node:
                    return parent
        return None


def check_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Check a file for type hints.
    
    Args:
        file_path: Path to the file
        
    Returns:
        List of missing type hints
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        tree = ast.parse(content)
        visitor = TypeHintVisitor()
        visitor.set_file(file_path)
        visitor.visit(tree)
        
        return visitor.missing_type_hints
    except SyntaxError as e:
        logger.warning(f"Syntax error in {file_path}: {e}")
        return []
    except Exception as e:
        logger.error(f"Error checking {file_path}: {e}")
        return []


def check_directory(directory: str, exclude_dirs: List[str] = None, exclude_files: List[str] = None) -> List[Dict[str, Any]]:
    """
    Check a directory for type hints.
    
    Args:
        directory: Directory to check
        exclude_dirs: Directories to exclude
        exclude_files: Files to exclude
        
    Returns:
        List of missing type hints
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
    
    missing_type_hints = []
    
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
            missing_type_hints.extend(check_file(file_path))
    
    return missing_type_hints


def print_report(missing_type_hints: List[Dict[str, Any]]) -> None:
    """
    Print a report of missing type hints.
    
    Args:
        missing_type_hints: List of missing type hints
    """
    # Group by file
    by_file = {}
    for hint in missing_type_hints:
        if hint["file"] not in by_file:
            by_file[hint["file"]] = []
        by_file[hint["file"]].append(hint)
    
    # Print report
    print("\n=== Type Hint Check Report ===\n")
    print(f"Total files with missing type hints: {len(by_file)}")
    print(f"Total missing type hints: {len(missing_type_hints)}")
    
    # Print by type
    by_type = {}
    for hint in missing_type_hints:
        if hint["type"] not in by_type:
            by_type[hint["type"]] = 0
        by_type[hint["type"]] += 1
    
    print("\nMissing type hints by type:")
    for hint_type, count in by_type.items():
        print(f"  {hint_type}: {count}")
    
    # Print top files
    print("\nTop files with missing type hints:")
    for file, hints in sorted(by_file.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
        print(f"  {file}: {len(hints)}")
    
    # Print details
    print("\nDetails:")
    for file, hints in sorted(by_file.items()):
        print(f"\n{file}:")
        for hint in sorted(hints, key=lambda x: x["line"]):
            print(f"  Line {hint['line']}: {hint['message']}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Check for comprehensive type hinting in the codebase")
    parser.add_argument("--directory", "-d", default=".", help="Directory to check")
    parser.add_argument("--exclude-dirs", "-e", nargs="*", help="Directories to exclude")
    parser.add_argument("--exclude-files", "-f", nargs="*", help="Files to exclude")
    parser.add_argument("--output", "-o", help="Output file path")
    args = parser.parse_args()
    
    logger.info(f"Checking {args.directory}...")
    missing_type_hints = check_directory(
        args.directory,
        exclude_dirs=args.exclude_dirs,
        exclude_files=args.exclude_files,
    )
    
    print_report(missing_type_hints)
    
    if args.output:
        import json
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(missing_type_hints, f, indent=2)
        logger.info(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
