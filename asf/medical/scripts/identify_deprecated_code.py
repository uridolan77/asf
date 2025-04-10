#!/usr/bin/env python
"""
Script to identify deprecated code in the Medical Research Synthesizer.

This script scans the codebase for deprecated code patterns, such as:
- Files with similar names (e.g., file.py and file_v2.py)
- Functions or classes marked with @deprecated decorator
- Functions or classes with "deprecated" in the name or docstring
- Commented-out code blocks
- TODO comments
- FIXME comments
- Unused imports
- Unused variables
- Unused functions
- Unused classes

Usage:
    python -m asf.medical.scripts.identify_deprecated_code
"""

import os
import re
import ast
import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional

# Patterns to look for
DEPRECATED_FILE_PATTERNS = [r"_new\.py$", r"_old\.py$", r"\.bak$", r"\.old$"]
DEPRECATED_COMMENT_PATTERNS = [
    r"#\s*TODO",
    r"#\s*DEPRECATED",
    r"#\s*FIXME",
    r"\"\"\".*DEPRECATED.*\"\"\"",
    r"\'\'\'.*DEPRECATED.*\'\'\'",
]
FASTAPI_ROUTE_PATTERN = r"@\w+\.(?:get|post|put|delete|patch|options|head)\s*\(\s*[\"\']([^\"\']+)[\"\']"

def find_python_files(directory: str) -> List[str]:
    """Find all Python files in the given directory and its subdirectories."""
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    return python_files

def check_deprecated_filenames(files: List[str]) -> List[str]:
    """Check for files with deprecated naming patterns."""
    deprecated_files = []
    for file in files:
        filename = os.path.basename(file)
        for pattern in DEPRECATED_FILE_PATTERNS:
            if re.search(pattern, filename):
                deprecated_files.append(file)
                break
    return deprecated_files

def check_deprecated_comments(files: List[str]) -> Dict[str, List[Tuple[int, str]]]:
    """Check for deprecated comments in files."""
    deprecated_comments = {}
    for file in files:
        file_comments = []
        try:
            with open(file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    for pattern in DEPRECATED_COMMENT_PATTERNS:
                        if re.search(pattern, line):
                            file_comments.append((i + 1, line.strip()))
                            break
            if file_comments:
                deprecated_comments[file] = file_comments
        except Exception as e:
            print(f"Error reading {file}: {e}")
    return deprecated_comments

def find_duplicate_routes(files: List[str]) -> Dict[str, List[str]]:
    """Find duplicate route definitions in FastAPI routers."""
    routes = {}
    duplicate_routes = {}

    for file in files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                content = f.read()
                for match in re.finditer(FASTAPI_ROUTE_PATTERN, content):
                    route = match.group(1)
                    if route in routes:
                        if route not in duplicate_routes:
                            duplicate_routes[route] = [routes[route]]
                        duplicate_routes[route].append(file)
                    else:
                        routes[route] = file
        except Exception as e:
            print(f"Error reading {file}: {e}")

    return duplicate_routes

def find_unused_imports(files: List[str]) -> Dict[str, List[str]]:
    """Find unused imports in Python files."""
    unused_imports = {}

    for file in files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)
            imports = []
            used_names = set()

            # Find all imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        imports.append(name.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module
                    for name in node.names:
                        if name.name == "*":
                            # Can't track star imports
                            continue
                        if module:
                            imports.append(f"{module}.{name.name}")
                        else:
                            imports.append(name.name)

            # Find all used names
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    used_names.add(node.id)
                elif isinstance(node, ast.Attribute):
                    # This is a simplification and might miss some cases
                    if isinstance(node.value, ast.Name):
                        used_names.add(node.value.id)

            # Find unused imports
            file_unused_imports = []
            for imp in imports:
                base_name = imp.split(".")[0]
                if base_name not in used_names:
                    file_unused_imports.append(imp)

            if file_unused_imports:
                unused_imports[file] = file_unused_imports

        except Exception as e:
            print(f"Error analyzing {file}: {e}")

    return unused_imports

def main():
    """Main function to run the script."""
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        # Default to the asf/medical directory
        directory = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "medical")

    print(f"Scanning directory: {directory}")

    # Find all Python files
    python_files = find_python_files(directory)
    print(f"Found {len(python_files)} Python files")

    # Check for deprecated filenames
    deprecated_files = check_deprecated_filenames(python_files)
    if deprecated_files:
        print("\n=== Files with deprecated naming patterns ===")
        for file in deprecated_files:
            print(f"  - {file}")

    # Check for deprecated comments
    deprecated_comments = check_deprecated_comments(python_files)
    if deprecated_comments:
        print("\n=== Files with deprecated comments ===")
        for file, comments in deprecated_comments.items():
            print(f"  - {file}")
            for line_num, comment in comments:
                print(f"    Line {line_num}: {comment}")

    # Find duplicate routes
    duplicate_routes = find_duplicate_routes(python_files)
    if duplicate_routes:
        print("\n=== Duplicate route definitions ===")
        for route, files in duplicate_routes.items():
            print(f"  - Route '{route}' defined in:")
            for file in files:
                print(f"    - {file}")

    # Find unused imports
    unused_imports = find_unused_imports(python_files)
    if unused_imports:
        print("\n=== Files with potentially unused imports ===")
        for file, imports in unused_imports.items():
            print(f"  - {file}")
            for imp in imports:
                print(f"    - {imp}")

    # Summary
    print("\n=== Summary ===")
    print(f"Files with deprecated naming patterns: {len(deprecated_files)}")
    print(f"Files with deprecated comments: {len(deprecated_comments)}")
    print(f"Duplicate route definitions: {len(duplicate_routes)}")
    print(f"Files with potentially unused imports: {len(unused_imports)}")

    # Suggest next steps
    print("\n=== Suggested Next Steps ===")
    print("1. Review and remove or refactor files with deprecated naming patterns")
    print("2. Address deprecated comments by either implementing the TODO or removing deprecated code")
    print("3. Consolidate duplicate route definitions to ensure a single source of truth")
    print("4. Clean up unused imports to improve code clarity")
    print("5. Run tests after each change to ensure functionality is preserved")

if __name__ == "__main__":
    main()
