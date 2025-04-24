#!/usr/bin/env python
# run_add_users_providers.py

"""
Script to run the migration for the users-providers association table.
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

# Add project root to Python path if needed
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    logger.info(f"Added {project_root} to Python path")

# Import the migration script
from add_users_providers import main

if __name__ == "__main__":
    logger.info("Running migration to add users_providers association table...")
    main()
