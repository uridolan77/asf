"""
This module configures the Python path to include necessary directories.
Import this at the beginning of your application to ensure all imports work correctly.
"""
import os
import sys
from pathlib import Path

# Add root directory to Python path to make absolute imports work
root_dir = Path(__file__).resolve().parent.parent.parent  # Go up to the asf directory
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
    print(f"Added {root_dir} to Python path")

# Also add the parent of root_dir (which is c:\code\asf) to make absolute imports like 'from asf' work
parent_dir = root_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
    print(f"Added {parent_dir} to Python path")