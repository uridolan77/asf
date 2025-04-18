import os
import re
import ast
from typing import List, Dict, Tuple
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
    logger.error(f\"Error reading {file}: {str(e)}\")
    raise DatabaseError(f\"Error reading {file}: {str(e)}\")
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
    logger.error(f\"Error reading {file}: {str(e)}\")
    raise DatabaseError(f\"Error reading {file}: {str(e)}\")
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
    logger.error(f\"Error analyzing {file}: {str(e)}\")
    raise DatabaseError(f\"Error analyzing {file}: {str(e)}\")
    return unused_imports
def main():
    """Main function to run the script.
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description