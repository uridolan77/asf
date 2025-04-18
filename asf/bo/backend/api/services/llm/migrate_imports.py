#!/usr/bin/env python3
"""
Migration script for updating import statements to use the new LLM Gateway structure.

This script recursively searches through Python files in a directory and
updates import statements according to the mapping defined below.
"""

import os
import re
import argparse
import logging
from typing import Dict, List, Tuple
import ast
import astunparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Define import mappings - OLD_IMPORT: NEW_IMPORT
IMPORT_MAPPING = {
    "asf.medical.llm_gateway.core.client": "asf.medical.llm_gateway.client",
    "asf.medical.llm_gateway.core.models": "asf.medical.llm_gateway.models",
    "asf.medical.llm_gateway.core.factory.ProviderFactory": "asf.medical.llm_gateway.transport.factory.TransportFactory",
    "asf.medical.llm_gateway.core.provider": "asf.medical.llm_gateway.providers.base",
    "asf.medical.llm_gateway.core.provider.Provider": "asf.medical.llm_gateway.providers.base.LLMProvider",
    "asf.medical.llm_gateway.core.transport": "asf.medical.llm_gateway.transport.base",
    "asf.medical.llm_gateway.mcp.observability": "asf.medical.llm_gateway.observability",
    "asf.medical.llm_gateway.mcp.resilience": "asf.medical.llm_gateway.resilience",
    "asf.medical.llm_gateway.mcp.observability.metrics": "asf.medical.llm_gateway.observability.metrics",
    "asf.medical.llm_gateway.mcp.resilience.circuit_breaker": "asf.medical.llm_gateway.resilience.circuit_breaker",
}

# Define class mappings - OLD_CLASS: NEW_CLASS
CLASS_MAPPING = {
    "InterventionContext": "ConversationContext",
    "MCPRole": "MessageRole",
    "ProviderConfig": "LLMProviderConfig",
    "Transport": "Transport",
}


class ImportUpdater(ast.NodeVisitor):
    """AST visitor to update import statements."""

    def __init__(self):
        super().__init__()
        self.modified = False
        self.replacements = []

    def visit_Import(self, node):
        """Process simple 'import x' statements."""
        for alias in node.names:
            if alias.name in IMPORT_MAPPING:
                self.modified = True
                self.replacements.append((node, f"import {IMPORT_MAPPING[alias.name]}"))
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Process 'from x import y' statements."""
        module = node.module
        if module in IMPORT_MAPPING:
            self.modified = True
            new_module = IMPORT_MAPPING[module]
            imports = [alias.name for alias in node.names]
            
            # Map class names if needed
            mapped_imports = [CLASS_MAPPING.get(name, name) for name in imports]
            
            # Create new import statement
            import_list = ", ".join(mapped_imports)
            self.replacements.append(
                (node, f"from {new_module} import {import_list}")
            )
        
        # Look for specific module.class mappings
        elif module:
            for old_import, new_import in IMPORT_MAPPING.items():
                if old_import.endswith(module):
                    # Match found for a module
                    self.modified = True
                    # Extract the base package from the old import
                    parts = old_import.split('.')
                    base_package = '.'.join(parts[:-1]) if len(parts) > 1 else parts[0]
                    new_module = new_import
                    
                    imports = [alias.name for alias in node.names]
                    mapped_imports = [CLASS_MAPPING.get(name, name) for name in imports]
                    import_list = ", ".join(mapped_imports)
                    
                    self.replacements.append(
                        (node, f"from {new_module} import {import_list}")
                    )
                    break
        
        self.generic_visit(node)


def update_imports_in_file(file_path: str) -> Tuple[bool, List[Tuple[str, str]]]:
    """
    Update import statements in a file.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        Tuple of (was_modified, list_of_changes)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
        updater = ImportUpdater()
        updater.visit(tree)
        
        changes = []
        
        if updater.modified:
            # Insert modified nodes
            for node, replacement in updater.replacements:
                changes.append((astunparse.unparse(node).strip(), replacement))
            
            # Write back the modified content
            modified_content = content
            for original, replacement in changes:
                modified_content = modified_content.replace(original, replacement)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            
            return True, changes
        
        return False, []
    
    except SyntaxError as e:
        logger.error(f"Syntax error in file {file_path}: {e}")
        return False, []


def update_imports_in_directory(directory: str, recursive: bool = True) -> Dict[str, List[Tuple[str, str]]]:
    """
    Update import statements in all Python files in a directory.
    
    Args:
        directory: Directory to process
        recursive: Whether to recursively process subdirectories
        
    Returns:
        Dictionary of file paths to lists of changes made
    """
    changes_by_file = {}
    
    if recursive:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    modified, changes = update_imports_in_file(file_path)
                    if modified:
                        changes_by_file[file_path] = changes
                        logger.info(f"Updated imports in {file_path}")
    else:
        for file in os.listdir(directory):
            if file.endswith('.py'):
                file_path = os.path.join(directory, file)
                modified, changes = update_imports_in_file(file_path)
                if modified:
                    changes_by_file[file_path] = changes
                    logger.info(f"Updated imports in {file_path}")
    
    return changes_by_file


def print_summary(changes_by_file: Dict[str, List[Tuple[str, str]]]):
    """Print a summary of changes made."""
    if not changes_by_file:
        logger.info("No changes were made.")
        return
    
    logger.info(f"Updated {len(changes_by_file)} files:")
    
    for file_path, changes in changes_by_file.items():
        logger.info(f"\n{file_path}:")
        for original, replacement in changes:
            logger.info(f"  - {original} -> {replacement}")


def main():
    parser = argparse.ArgumentParser(description="Update LLM Gateway import statements")
    parser.add_argument("directory", help="Directory to process")
    parser.add_argument("--no-recursive", action="store_true", help="Don't process subdirectories")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without modifying files")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        logger.error(f"Directory not found: {args.directory}")
        return 1
    
    if args.dry_run:
        logger.info("Running in dry-run mode - no files will be modified")
    
    changes_by_file = update_imports_in_directory(args.directory, not args.no_recursive)
    print_summary(changes_by_file)
    
    logger.info(f"Migration complete. {len(changes_by_file)} files updated.")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())