"""
Fix Unused Imports in ASF Medical Codebase.

This script removes unused imports from Python files in the ASF Medical codebase.
"""

import os
import re
import sys
import logging
from typing import List, Dict, Set, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("FixUnusedImports")

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
    "fix_unused_imports.py",
    "deep_cleanup_phase2.py"
]

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

def find_unused_imports(content: str) -> Dict[int, List[str]]:
    """Find unused imports in the content."""
    # This is a simplified approach and might have false positives/negatives
    unused_imports = {}
    lines = content.split("\n")
    
    import_lines = []
    for i, line in enumerate(lines):
        if line.startswith("from ") or line.startswith("import "):
            import_lines.append((i, line))
    
    for i, line in import_lines:
        # Extract imported names
        if "import" in line:
            imported_names = []
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
            unused_names = []
            for name in imported_names:
                if name == "*":
                    continue  # Skip wildcard imports
                
                # Handle "as" imports
                if " as " in name:
                    name = name.split(" as ")[1].strip()
                
                # Check if the name is used in the rest of the file
                name_pattern = r"\b" + name + r"\b"
                used = False
                for j, other_line in enumerate(lines):
                    if j != i and re.search(name_pattern, other_line):
                        used = True
                        break
                
                if not used:
                    unused_names.append(name)
            
            if unused_names:
                unused_imports[i] = unused_names
    
    return unused_imports

def remove_unused_imports(content: str, unused_imports: Dict[int, List[str]]) -> str:
    """Remove unused imports from the content."""
    lines = content.split("\n")
    
    # Process lines in reverse order to avoid index issues
    for line_num in sorted(unused_imports.keys(), reverse=True):
        line = lines[line_num]
        unused_names = unused_imports[line_num]
        
        if "from" in line:
            # from module import name1, name2
            match = re.match(r"from\s+([\w.]+)\s+import\s+([\w,\s]+)", line)
            if match:
                module = match.group(1)
                imported_names = [name.strip() for name in match.group(2).split(",")]
                
                # Remove unused names
                remaining_names = [name for name in imported_names if name not in unused_names]
                
                if remaining_names:
                    # Update the line with remaining names
                    new_line = f"from {module} import {', '.join(remaining_names)}"
                    lines[line_num] = new_line
                else:
                    # Remove the entire line if all imports are unused
                    lines[line_num] = ""
        else:
            # import module
            match = re.match(r"import\s+([\w,\s]+)", line)
            if match:
                imported_names = [name.strip() for name in match.group(1).split(",")]
                
                # Remove unused names
                remaining_names = [name for name in imported_names if name not in unused_names]
                
                if remaining_names:
                    # Update the line with remaining names
                    new_line = f"import {', '.join(remaining_names)}"
                    lines[line_num] = new_line
                else:
                    # Remove the entire line if all imports are unused
                    lines[line_num] = ""
    
    # Remove empty lines
    return "\n".join(line for line in lines if line.strip())

def process_file(file_path: str, fix: bool = False) -> Dict[str, int]:
    """Process a single file to find and optionally fix unused imports."""
    results = {
        "unused_imports": 0,
        "fixed": False
    }
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Find unused imports
        unused_imports = find_unused_imports(content)
        results["unused_imports"] = sum(len(names) for names in unused_imports.values())
        
        # Fix unused imports if requested
        if fix and unused_imports:
            updated_content = remove_unused_imports(content, unused_imports)
            
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
        print("Usage: python fix_unused_imports.py <directory> [--fix]")
        sys.exit(1)
    
    directory = sys.argv[1]
    fix = "--fix" in sys.argv
    
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a directory")
        sys.exit(1)
    
    logger.info(f"Starting fix unused imports for {directory}")
    logger.info(f"Fix mode: {fix}")
    
    python_files = find_python_files(directory)
    logger.info(f"Found {len(python_files)} Python files")
    
    total_unused_imports = 0
    fixed_files = 0
    
    for file_path in python_files:
        logger.info(f"Processing {file_path}")
        results = process_file(file_path, fix)
        
        # Count issues
        total_unused_imports += results["unused_imports"]
        
        if results["fixed"]:
            fixed_files += 1
            logger.info(f"Fixed unused imports in {file_path}")
    
    # Report summary
    logger.info("=" * 50)
    logger.info("Summary:")
    logger.info(f"Total Python files processed: {len(python_files)}")
    logger.info(f"Total unused imports: {total_unused_imports}")
    
    if fix:
        logger.info(f"Fixed files: {fixed_files}")
    
    logger.info("=" * 50)

if __name__ == "__main__":
    main()
