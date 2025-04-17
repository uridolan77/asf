#!/usr/bin/env python
# create_symlinks.py

"""
Script to create symbolic links to make imports work.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_symlink(source, target):
    """Create a symbolic link."""
    try:
        if os.path.exists(target):
            logger.info(f"Symlink {target} already exists")
            return
        
        os.symlink(source, target, target_is_directory=True)
        logger.info(f"Created symlink {target} -> {source}")
    except Exception as e:
        logger.error(f"Error creating symlink {target} -> {source}: {e}")

def main():
    """Main function."""
    # Get the backend directory
    backend_dir = Path(__file__).parent.absolute()
    logger.info(f"Backend directory: {backend_dir}")
    
    # Create symlinks for each module
    modules = [
        "db",
        "services",
        "utils",
        "auth",
        "models",
        "schemas",
    ]
    
    for module in modules:
        source = os.path.join(backend_dir, module)
        target = os.path.join(backend_dir, "api", "routers", "config", module)
        create_symlink(source, target)
    
    logger.info("Symlinks created successfully")

if __name__ == "__main__":
    main()
