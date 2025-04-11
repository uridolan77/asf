"""
Migration Script for Unified Architecture.

This script helps migrate from the old architecture to the new unified architecture
by updating imports, replacing deprecated components, and cleaning up redundant code.
"""

import os
import re
import argparse
import logging
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define migration mappings
IMPORT_MAPPINGS = {
    # Contradiction service mappings
    "from asf.medical.ml.services.contradiction_service import ContradictionService": "from asf.medical.ml.services.unified_contradiction_service import UnifiedContradictionService",
    "from asf.medical.ml.services.enhanced_contradiction_service import EnhancedContradictionService": "from asf.medical.ml.services.unified_contradiction_service import UnifiedContradictionService",
    
    # Repository mappings
    "from asf.medical.storage.repositories.base_repository import BaseRepository": "from asf.medical.storage.repositories.enhanced_base_repository import EnhancedBaseRepository",
    
    # Task storage mappings
    "from asf.medical.tasks.export_tasks import task_results": "from asf.medical.core.unified_task_storage import unified_task_storage",
    
    # Router mappings
    "from asf.medical.api.routers.contradiction import router as contradiction_router": "from asf.medical.api.routers.unified_contradiction import router as contradiction_router",
    "from asf.medical.api.routers.enhanced_contradiction import router as enhanced_contradiction_router": "from asf.medical.api.routers.unified_contradiction import router as contradiction_router",
}

CLASS_MAPPINGS = {
    "ContradictionService": "UnifiedContradictionService",
    "EnhancedContradictionService": "UnifiedContradictionService",
    "BaseRepository": "EnhancedBaseRepository",
}

FUNCTION_MAPPINGS = {
    "task_results\\[task_id\\]": "unified_task_storage.get_task_result_sync(task_id)",
    "task_results\\[task_id\\] = ": "unified_task_storage.set_task_result_sync(task_id, ",
}

def find_python_files(directory: str) -> List[str]:
    """
    Find all Python files in a directory and its subdirectories.
    
    Args:
        directory: Directory to search
        
    Returns:
        List of Python file paths
    """
    python_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    
    return python_files

def update_imports(content: str) -> Tuple[str, bool]:
    """
    Update imports in a file.
    
    Args:
        content: File content
        
    Returns:
        Tuple of updated content and whether any changes were made
    """
    updated_content = content
    changed = False
    
    for old_import, new_import in IMPORT_MAPPINGS.items():
        if old_import in updated_content:
            updated_content = updated_content.replace(old_import, new_import)
            changed = True
    
    return updated_content, changed

def update_classes(content: str) -> Tuple[str, bool]:
    """
    Update class names in a file.
    
    Args:
        content: File content
        
    Returns:
        Tuple of updated content and whether any changes were made
    """
    updated_content = content
    changed = False
    
    for old_class, new_class in CLASS_MAPPINGS.items():
        # Match class inheritance and instantiation
        pattern = r"(?<!\w)(" + old_class + r")(?!\w)"
        if re.search(pattern, updated_content):
            updated_content = re.sub(pattern, new_class, updated_content)
            changed = True
    
    return updated_content, changed

def update_functions(content: str) -> Tuple[str, bool]:
    """
    Update function calls in a file.
    
    Args:
        content: File content
        
    Returns:
        Tuple of updated content and whether any changes were made
    """
    updated_content = content
    changed = False
    
    for old_function, new_function in FUNCTION_MAPPINGS.items():
        # Match function calls
        pattern = old_function
        if re.search(pattern, updated_content):
            updated_content = re.sub(pattern, new_function, updated_content)
            changed = True
    
    return updated_content, changed

def remove_duplicate_routers(content: str) -> Tuple[str, bool]:
    """
    Remove duplicate router imports and inclusions.
    
    Args:
        content: File content
        
    Returns:
        Tuple of updated content and whether any changes were made
    """
    # Check if both contradiction_router and enhanced_contradiction_router are imported
    if "contradiction_router" in content and "enhanced_contradiction_router" in content:
        # Remove enhanced_contradiction_router import
        pattern = r"from asf\.medical\.api\.routers\.enhanced_contradiction import router as enhanced_contradiction_router\n"
        updated_content = re.sub(pattern, "", content)
        
        # Remove enhanced_contradiction_router inclusion in app.include_router
        pattern = r"app\.include_router\(enhanced_contradiction_router\)\n"
        updated_content = re.sub(pattern, "", updated_content)
        
        return updated_content, updated_content != content
    
    return content, False

def process_file(file_path: str, dry_run: bool = False) -> bool:
    """
    Process a Python file.
    
    Args:
        file_path: Path to the Python file
        dry_run: Whether to perform a dry run
        
    Returns:
        Whether any changes were made
    """
    try:
        # Read file content
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Update imports
        updated_content, imports_changed = update_imports(content)
        
        # Update classes
        updated_content, classes_changed = update_classes(updated_content)
        
        # Update functions
        updated_content, functions_changed = update_functions(updated_content)
        
        # Remove duplicate routers
        updated_content, routers_changed = remove_duplicate_routers(updated_content)
        
        # Check if any changes were made
        changed = imports_changed or classes_changed or functions_changed or routers_changed
        
        if changed:
            logger.info(f"Changes detected in {file_path}")
            
            if not dry_run:
                # Write updated content
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
    """Main function."""
    parser = argparse.ArgumentParser(description="Migrate to unified architecture")
    parser.add_argument("--directory", "-d", type=str, default="asf/medical", help="Directory to process")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Perform a dry run")
    args = parser.parse_args()
    
    # Find Python files
    python_files = find_python_files(args.directory)
    logger.info(f"Found {len(python_files)} Python files")
    
    # Process files
    changed_files = 0
    for file_path in python_files:
        if process_file(file_path, args.dry_run):
            changed_files += 1
    
    logger.info(f"Processed {len(python_files)} files, {changed_files} files changed")
    
    if args.dry_run:
        logger.info("Dry run completed. No files were modified.")
    else:
        logger.info("Migration completed successfully.")

if __name__ == "__main__":
    main()
