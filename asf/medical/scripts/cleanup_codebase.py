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
    r"asf[/\\]medical[/\\]api[/\\]export_utils\.py",
    r"asf[/\\]medical[/\\]ml[/\\]services[/\\]contradiction_service\.py",
    r"asf[/\\]medical[/\\]ml[/\\]services[/\\]enhanced_contradiction_service\.py",
    r"asf[/\\]medical[/\\]api[/\\]routers[/\\]contradiction\.py",
    r"asf[/\\]medical[/\\]api[/\\]routers[/\\]enhanced_contradiction\.py",
    r"asf[/\\]medical[/\\]storage[/\\]repositories[/\\]base_repository\.py",
    r"asf[/\\]__examples[/\\].*\.py",
    r"asf[/\\]__tests[/\\].*\.py",
]
IMPORT_REPLACEMENTS = {
    r"from asf\.medical\.api\.export_utils import": "from asf.medical.api.export_utils_consolidated import",
    r"import asf\.medical\.api\.export_utils": "import asf.medical.api.export_utils_consolidated_consolidated_consolidated_consolidated",
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
    """Find all Python files in the given directory and its subdirectories."""
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    return python_files
def should_remove_file(file_path: str) -> bool:
    """Check if a file should be removed.
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
    updated_content = content
    changed = False
    for old_import, new_import in IMPORT_REPLACEMENTS.items():
        if re.search(old_import, updated_content):
            updated_content = re.sub(old_import, new_import, updated_content)
            changed = True
    return updated_content, changed
def fix_classes(content: str) -> Tuple[str, bool]:
    """Fix class definitions in a file."""
    updated_content = content
    changed = False
    for old_class, new_class in CLASS_REPLACEMENTS.items():
        if re.search(old_class, updated_content):
            updated_content = re.sub(old_class, new_class, updated_content)
            changed = True
    return updated_content, changed
def fix_db_usage(content: str) -> Tuple[str, bool]:
    """Fix database usage in a file."""
    updated_content = content
    changed = False
    for old_pattern, new_pattern in DB_USAGE_FIXES.items():
        if re.search(old_pattern, updated_content):
            updated_content = re.sub(old_pattern, new_pattern, updated_content)
            changed = True
    return updated_content, changed
def remove_commented_code(content: str) -> Tuple[str, bool]:
    """Remove commented-out code blocks."""
    lines = content.split('\n')
    new_lines = []
    in_comment_block = False
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
    """Process a Python file.
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
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
    """Main function.
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
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