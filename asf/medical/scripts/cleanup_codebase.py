import os
import re
import argparse
import logging
from typing import List, Tuple
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("CodebaseCleanup")
FILES_TO_REMOVE = [
    # r"asf[/\\]medical[/\\]api[/\\]export_utils\.py",  # Keeping this file as it contains consolidated export functionality
    r"asf[/\\]medical[/\\]ml[/\\]services[/\\]contradiction_service\.py",
    r"asf[/\\]medical[/\\]ml[/\\]services[/\\]enhanced_contradiction_service\.py",
    r"asf[/\\]medical[/\\]api[/\\]routers[/\\]contradiction\.py",
    r"asf[/\\]medical[/\\]api[/\\]routers[/\\]enhanced_contradiction\.py",
    r"asf[/\\]medical[/\\]storage[/\\]repositories[/\\]base_repository\.py",
    r"asf[/\\]__examples[/\\].*\.py",
    r"asf[/\\]__tests[/\\].*\.py",
]
IMPORT_REPLACEMENTS = {
    # Removing the incorrect export utils replacements
    # r"from asf\.medical\.api\.export_utils import": "from asf.medical.api.export_utils_consolidated import",
    # r"import asf\.medical\.api\.export_utils": "import asf.medical.api.export_utils_consolidated_consolidated_consolidated_consolidated",
    r"from asf\.medical\.ml\.services\.contradiction_service import ContradictionService": "from asf.medical.ml.services.unified_contradiction_service import ContradictionService",
    r"from asf\.medical\.ml\.services\.enhanced_contradiction_service import ContradictionService": "from asf.medical.ml.services.unified_contradiction_service import ContradictionService",
    r"from asf\.medical\.api\.routers\.contradiction import router as contradiction_router": "from asf.medical.api.routers.unified_contradiction import router as contradiction_router",
    r"from asf\.medical\.api\.routers\.enhanced_contradiction import router as enhanced_contradiction_router": "from asf.medical.api.routers.unified_contradiction import router as contradiction_router",
    r"from asf\.medical\.storage\.repositories\.base_repository import BaseRepository": "from asf.medical.storage.repositories.enhanced_base_repository import EnhancedBaseRepository",
}
CLASS_REPLACEMENTS = {
    r"class \w+\(BaseRepository\[": "class \\g<0>".replace("BaseRepository", "EnhancedBaseRepository"),
    r"ContradictionService": "ContradictionService",
    r"ContradictionService": "ContradictionService",
}
DB_USAGE_FIXES = {
    r"await \w+\.get_async\(db,": "await \\g<0>".replace("db", "db=db_session"),
    r"await \w+\.create_async\(db,": "await \\g<0>".replace("db", "db=db_session"),
    r"await \w+\.update_async\(db,": "await \\g<0>".replace("db", "db=db_session"),
    r"await \w+\.delete_async\(db,": "await \\g<0>".replace("db", "db=db_session"),
    r"await \w+\.list_async\(db": "await \\g<0>".replace("db", "db=db_session"),
    r"await \w+\.get_by_\w+_async\(db,": "await \\g<0>".replace("db", "db=db_session"),
}
def find_python_files(directory: str) -> List[str]:
    """Find all Python files in the given directory and its subdirectories.

    This function recursively walks through the specified directory and collects
    paths to all files with a .py extension.

    Args:
        directory: Path to the directory to search

    Returns:
        List of paths to Python files
    """
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    return python_files
def should_remove_file(file_path: str) -> bool:
    """Check if a file should be removed based on predefined patterns.

    This function checks if the given file path matches any of the patterns
    in the FILES_TO_REMOVE list, which identifies files that should be removed
    during the cleanup process.

    Args:
        file_path: Path to the file to check

    Returns:
        True if the file should be removed, False otherwise
    """
    for pattern in FILES_TO_REMOVE:
        if re.search(pattern, file_path):
            return True
    return False

def fix_imports(content: str) -> Tuple[str, bool]:
    """Fix import statements in a file.

    This function applies the import replacements defined in IMPORT_REPLACEMENTS
    to the content of a file. It replaces import statements that match the
    patterns with their updated versions.

    Args:
        content: Content of the file to process

    Returns:
        Tuple containing the updated content and a boolean indicating if changes were made
    """
    updated_content = content
    changed = False
    for old_import, new_import in IMPORT_REPLACEMENTS.items():
        if re.search(old_import, updated_content):
            updated_content = re.sub(old_import, new_import, updated_content)
            changed = True
    return updated_content, changed
def fix_classes(content: str) -> Tuple[str, bool]:
    """Fix class definitions in a file.

    This function applies the class replacements defined in CLASS_REPLACEMENTS
    to the content of a file. It replaces class definitions that match the
    patterns with their updated versions.

    Args:
        content: Content of the file to process

    Returns:
        Tuple containing the updated content and a boolean indicating if changes were made
    """
    updated_content = content
    changed = False
    for old_class, new_class in CLASS_REPLACEMENTS.items():
        if re.search(old_class, updated_content):
            updated_content = re.sub(old_class, new_class, updated_content)
            changed = True
    return updated_content, changed
def fix_db_usage(content: str) -> Tuple[str, bool]:
    """Fix database usage in a file.

    This function applies the database usage fixes defined in DB_USAGE_FIXES
    to the content of a file. It standardizes database session parameter names
    and usage patterns across the codebase.

    Args:
        content: Content of the file to process

    Returns:
        Tuple containing the updated content and a boolean indicating if changes were made
    """
    updated_content = content
    changed = False
    for old_pattern, new_pattern in DB_USAGE_FIXES.items():
        if re.search(old_pattern, updated_content):
            updated_content = re.sub(old_pattern, new_pattern, updated_content)
            changed = True
    return updated_content, changed
def remove_commented_code(content: str) -> Tuple[str, bool]:
    """Remove commented-out code blocks from a file.

    This function identifies and removes commented-out code blocks while preserving
    docstrings and important comments (TODOs, FIXMEs, etc.). It handles both
    triple-quoted blocks and line comments.

    Args:
        content: Content of the file to process

    Returns:
        Tuple containing the updated content and a boolean indicating if changes were made
    """
    lines = content.split('\n')
    new_lines = []
    # Track if we're inside a comment block
    # This variable is used for clarity but not currently accessed
    changed = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if not new_lines and not line.strip():
            i += 1
            continue
        if re.match(r'\s*"""', line) or re.match(r"\s*'''", line):
            if (i == 0 or
                (i > 0 and re.match(r'\s*def\s+', lines[i-1])) or
                (i > 0 and re.match(r'\s*class\s+', lines[i-1])) or
                (i > 0 and not lines[i-1].strip())):
                new_lines.append(line)
                i += 1
                while i < len(lines) and not (re.search(r'"""', lines[i]) or re.search(r"'''", lines[i])):
                    new_lines.append(lines[i])
                    i += 1
                if i < len(lines):
                    new_lines.append(lines[i])
                    i += 1
            else:
                changed = True
                i += 1
                while i < len(lines) and not (re.search(r'"""', lines[i]) or re.search(r"'''", lines[i])):
                    i += 1
                if i < len(lines):
                    i += 1
        elif re.match(r'\s*#', line):
            if re.search(r'#\s*(TODO|FIXME|NOTE|WARNING|IMPORTANT)', line, re.IGNORECASE):
                new_lines.append(line)
            else:
                changed = True
            i += 1
        elif re.match(r'\s*# ', line):
            j = i
            commented_block = []
            while j < len(lines) and re.match(r'\s*# ', lines[j]):
                commented_block.append(lines[j])
                j += 1
            if len(commented_block) <= 2:
                for comment_line in commented_block:
                    new_lines.append(comment_line)
            else:
                changed = True
            i = j
        else:
            new_lines.append(line)
            i += 1
    return '\n'.join(new_lines), changed
def process_file(file_path: str, dry_run: bool = False) -> bool:
    """Process a Python file by applying all cleanup operations.

    This function reads a Python file, applies all cleanup operations (fixing imports,
    classes, database usage, and removing commented code), and writes the updated
    content back to the file if changes were made and dry_run is False.

    Args:
        file_path: Path to the Python file to process
        dry_run: If True, don't actually modify the file

    Returns:
        True if changes were detected, False otherwise
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        updated_content, imports_changed = fix_imports(content)
        updated_content, classes_changed = fix_classes(updated_content)
        updated_content, db_usage_changed = fix_db_usage(updated_content)
        updated_content, comments_changed = remove_commented_code(updated_content)
        changed = imports_changed or classes_changed or db_usage_changed or comments_changed
        if changed:
            logger.info(f"Changes detected in {file_path}")
            if not dry_run:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(updated_content)
                logger.info(f"Updated {file_path}")
            else:
                logger.info(f"Would update {file_path} (dry run)")
        return changed
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return False
def main():
    """Main entry point for the codebase cleanup script.

    This function parses command-line arguments, finds all Python files in the
    specified directory, processes each file to apply cleanup operations, and
    removes files that should be removed. It reports statistics on the number
    of files processed, changed, and removed.

    Command-line arguments:
        --dry-run, -n: Perform a dry run (no changes)
        --directory, -d: Directory to process (default: asf)

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Clean up the Medical Research Synthesizer codebase")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Perform a dry run (no changes)")
    parser.add_argument("--directory", "-d", default="asf", help="Directory to process")
    args = parser.parse_args()
    python_files = find_python_files(args.directory)
    logger.info(f"Found {len(python_files)} Python files")
    changed_files = 0
    for file_path in python_files:
        if process_file(file_path, args.dry_run):
            changed_files += 1
    removed_files = 0
    for file_path in python_files:
        if should_remove_file(file_path):
            logger.info(f"File to remove: {file_path}")
            if not args.dry_run:
                try:
                    os.remove(file_path)
                    logger.info(f"Removed {file_path}")
                    removed_files += 1
                except Exception as e:
                    logger.error(f"Error removing {file_path}: {str(e)}")
            else:
                logger.info(f"Would remove {file_path} (dry run)")
                removed_files += 1
    logger.info(f"Processed {len(python_files)} files, {changed_files} files changed, {removed_files} files removed")
    if args.dry_run:
        logger.info("Dry run completed. No files were modified.")
    else:
        logger.info("Cleanup completed successfully.")
if __name__ == "__main__":
    main()