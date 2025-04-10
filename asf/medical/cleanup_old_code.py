"""
Script to clean up old code in the Medical Research Synthesizer codebase.

This script removes old code that has been replaced by unified versions.
"""

import os
import shutil
import argparse
import logging
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Files to remove
FILES_TO_REMOVE = [
    # Old router files
    "asf/medical/api/routers/contradiction.py",
    "asf/medical/api/routers/screening.py",
    "asf/medical/api/routers/export.py",
    "asf/medical/api/routers/analysis.py",
    "asf/medical/api/routers/knowledge_base.py",
    
    # Old authentication implementations
    "asf/medical/api/auth.py",
    "asf/medical/api/auth_v2.py",
    "asf/medical/api/auth_service.py",
    
    # Old main files
    "asf/medical/api/main.py",
    "asf/medical/api/main_new.py",
    "asf/medical/api/main.py.new",
    
    # Old model files
    "asf/medical/api/models/auth.py",
]

# Files to rename
FILES_TO_RENAME = {
    "asf/medical/api/main_unified.py": "asf/medical/api/main.py",
    "asf/medical/api/routers/auth_unified.py": "asf/medical/api/routers/auth.py",
    "asf/medical/api/routers/search_unified.py": "asf/medical/api/routers/search.py",
    "asf/medical/api/routers/contradiction_unified.py": "asf/medical/api/routers/contradiction.py",
    "asf/medical/api/routers/screening_unified.py": "asf/medical/api/routers/screening.py",
    "asf/medical/api/routers/export_unified.py": "asf/medical/api/routers/export.py",
    "asf/medical/api/routers/analysis_unified.py": "asf/medical/api/routers/analysis.py",
    "asf/medical/api/routers/knowledge_base_unified.py": "asf/medical/api/routers/knowledge_base.py",
    "asf/medical/api/auth_unified.py": "asf/medical/api/auth.py",
}

def remove_files(files: List[str], dry_run: bool = False) -> List[str]:
    """
    Remove files from the filesystem.
    
    Args:
        files: List of files to remove
        dry_run: Whether to perform a dry run
        
    Returns:
        List of files that were removed
    """
    removed_files = []
    
    for file_path in files:
        if os.path.exists(file_path):
            if dry_run:
                logger.info(f"Would remove: {file_path}")
            else:
                try:
                    os.remove(file_path)
                    logger.info(f"Removed: {file_path}")
                    removed_files.append(file_path)
                except Exception as e:
                    logger.error(f"Error removing {file_path}: {str(e)}")
        else:
            logger.warning(f"File not found: {file_path}")
    
    return removed_files

def rename_files(files: Dict[str, str], dry_run: bool = False) -> List[Dict[str, str]]:
    """
    Rename files in the filesystem.
    
    Args:
        files: Dictionary of source -> destination file paths
        dry_run: Whether to perform a dry run
        
    Returns:
        List of files that were renamed
    """
    renamed_files = []
    
    for source, destination in files.items():
        if os.path.exists(source):
            if os.path.exists(destination):
                if dry_run:
                    logger.info(f"Would remove existing destination: {destination}")
                else:
                    try:
                        os.remove(destination)
                        logger.info(f"Removed existing destination: {destination}")
                    except Exception as e:
                        logger.error(f"Error removing existing destination {destination}: {str(e)}")
                        continue
            
            if dry_run:
                logger.info(f"Would rename: {source} -> {destination}")
            else:
                try:
                    os.rename(source, destination)
                    logger.info(f"Renamed: {source} -> {destination}")
                    renamed_files.append({"source": source, "destination": destination})
                except Exception as e:
                    logger.error(f"Error renaming {source} to {destination}: {str(e)}")
        else:
            logger.warning(f"Source file not found: {source}")
    
    return renamed_files

def update_imports(directory: str, dry_run: bool = False) -> int:
    """
    Update imports in Python files.
    
    Args:
        directory: Directory to search for Python files
        dry_run: Whether to perform a dry run
        
    Returns:
        Number of files updated
    """
    updated_files = 0
    
    # Import replacements
    replacements = {
        "from asf.medical.api.auth import": "from asf.medical.api.auth import",
        "from asf.medical.api.routers.auth import": "from asf.medical.api.routers.auth import",
        "from asf.medical.api.routers.search import": "from asf.medical.api.routers.search import",
        "from asf.medical.api.routers.contradiction import": "from asf.medical.api.routers.contradiction import",
        "from asf.medical.api.routers.screening import": "from asf.medical.api.routers.screening import",
        "from asf.medical.api.routers.export import": "from asf.medical.api.routers.export import",
        "from asf.medical.api.routers.analysis import": "from asf.medical.api.routers.analysis import",
        "from asf.medical.api.routers.knowledge_base import": "from asf.medical.api.routers.knowledge_base import",
        "import asf.medical.api.auth": "import asf.medical.api.auth",
        "import asf.medical.api.routers.auth": "import asf.medical.api.routers.auth",
        "import asf.medical.api.routers.search": "import asf.medical.api.routers.search",
        "import asf.medical.api.routers.contradiction": "import asf.medical.api.routers.contradiction",
        "import asf.medical.api.routers.screening": "import asf.medical.api.routers.screening",
        "import asf.medical.api.routers.export": "import asf.medical.api.routers.export",
        "import asf.medical.api.routers.analysis": "import asf.medical.api.routers.analysis",
        "import asf.medical.api.routers.knowledge_base": "import asf.medical.api.routers.knowledge_base",
    }
    
    # Walk through the directory
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                
                # Read the file
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Check if any replacements are needed
                new_content = content
                for old, new in replacements.items():
                    if old in new_content:
                        new_content = new_content.replace(old, new)
                
                # Update the file if changes were made
                if new_content != content:
                    if dry_run:
                        logger.info(f"Would update imports in: {file_path}")
                    else:
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(new_content)
                        logger.info(f"Updated imports in: {file_path}")
                        updated_files += 1
    
    return updated_files

def update_init_file(dry_run: bool = False) -> bool:
    """
    Update the __init__.py file in the routers directory.
    
    Args:
        dry_run: Whether to perform a dry run
        
    Returns:
        Whether the file was updated
    """
    init_file = "asf/medical/api/routers/__init__.py"
    
    if not os.path.exists(init_file):
        logger.warning(f"Init file not found: {init_file}")
        return False
    
    # New content for the init file
    new_content = '''"""
Router modules for the Medical Research Synthesizer API.
"""

from . import auth
from . import search
from . import contradiction
from . import screening
from . import export
from . import analysis
from . import knowledge_base
'''
    
    if dry_run:
        logger.info(f"Would update: {init_file}")
        return True
    
    try:
        with open(init_file, "w", encoding="utf-8") as f:
            f.write(new_content)
        logger.info(f"Updated: {init_file}")
        return True
    except Exception as e:
        logger.error(f"Error updating {init_file}: {str(e)}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Clean up old code in the Medical Research Synthesizer codebase")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run without making changes")
    args = parser.parse_args()
    
    logger.info("Starting cleanup...")
    
    # Remove old files
    logger.info("Removing old files...")
    removed_files = remove_files(FILES_TO_REMOVE, args.dry_run)
    logger.info(f"Removed {len(removed_files)} files")
    
    # Rename unified files
    logger.info("Renaming unified files...")
    renamed_files = rename_files(FILES_TO_RENAME, args.dry_run)
    logger.info(f"Renamed {len(renamed_files)} files")
    
    # Update imports
    logger.info("Updating imports...")
    updated_files = update_imports("asf/medical", args.dry_run)
    logger.info(f"Updated imports in {updated_files} files")
    
    # Update __init__.py file
    logger.info("Updating __init__.py file...")
    update_init_file(args.dry_run)
    
    logger.info("Cleanup complete!")

if __name__ == "__main__":
    main()
