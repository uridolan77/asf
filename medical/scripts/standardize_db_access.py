Standardize Database Access Script for the ASF Medical Research Synthesizer Codebase.
This script standardizes the database access patterns across the codebase,
ensuring consistent async database operations, error handling, and transaction management.
Usage:
    python -m asf.medical.scripts.standardize_db_access [--dry-run]
import os
import re
import logging
import argparse
from typing import List, Tuple
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("StandardizeDBAccess")
DB_PATTERNS = {
    r"(\w+)_repository\.(\w+)\(db": r"\1_repository.\2(db",
    r"def get_by_(\w+)\(self, db: Session,": r"async def get_by_\1_async(self, db: AsyncSession,",
    r"def create_(\w+)\(self, db: Session,": r"async def create_\1_async(self, db: AsyncSession,",
    r"def update_(\w+)\(self, db: Session,": r"async def update_\1_async(self, db: AsyncSession,",
    r"def delete_(\w+)\(self, db: Session,": r"async def delete_\1_async(self, db: AsyncSession,",
    r"db\.commit\(\)\s+db\.refresh\((\w+)\)": r"await db.commit()\n        await db.refresh(\1)",
    r"db\.rollback\(\)": r"await await await await db.rollback()",
    r"except SQLAlchemyError as e:\s+logger\.error\(f\"([^\"]+):\s*{str\(e\)}\"\)\s+raise":
    r"except SQLAlchemyError as e:\n            await await await await db.rollback()\n            logger.error(f\"\1: {str(e)}\")\n            raise DatabaseError(f\"\1: {str(e)}\")",
}
IMPORT_PATTERNS = {
    r"from sqlalchemy\.orm import Session":
    "from sqlalchemy.orm import Session\nfrom sqlalchemy.ext.asyncio import AsyncSession",
    r"from sqlalchemy import (\w+)":
    r"from sqlalchemy import \1\nfrom asf.medical.core.exceptions import DatabaseError",
}
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
    "database.py",  # Don't modify the database module itself
]
def find_python_files(directory: str) -> List[str]:
    Find all Python files in the given directory and its subdirectories.
    
    Args:
        directory: Description of directory
    
    
    Returns:
        Description of return value
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        original_content = content
        content, changed = standardize_db_access(content)
        if changed and not dry_run:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"Standardized database access in {file_path}")
        elif changed:
            logger.info(f"Would standardize database access in {file_path} (dry run)")
        return changed
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return False
def main():
    """Main function to run the script.
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
    """
    parser = argparse.ArgumentParser(description="Standardize database access patterns in the ASF Medical Research Synthesizer codebase")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually modify files, just show what would be done")
    parser.add_argument("--directory", default=None, help="Directory to process (default: asf/medical)")
    args = parser.parse_args()
    if args.directory:
        directory = args.directory
    else:
        directory = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "medical")
    logger.info(f"Scanning directory: {directory}")
    python_files = find_python_files(directory)
    logger.info(f"Found {len(python_files)} Python files")
    standardized_files = 0
    for file in python_files:
        if process_file(file, args.dry_run):
            standardized_files += 1
    logger.info(f"Standardized database access in {standardized_files} files")
    if args.dry_run:
        logger.info("This was a dry run. No files were actually modified.")
if __name__ == "__main__":
    main()