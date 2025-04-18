"""
Debug runner for the BO backend.
This script runs the backend with enhanced debugging enabled.
"""

import os
import sys
import uvicorn

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)  # Insert at the beginning of sys.path
    print(f"Added {project_root} to Python path")

# Also add the parent directory of the project root
parent_dir = os.path.dirname(project_root)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    print(f"Added {parent_dir} to Python path")

# Import debug configuration
from debug_config import setup_debug_logging

# Set up debug logging
setup_debug_logging()

# Import the main application
from main import app

if __name__ == "__main__":
    print("Starting backend in DEBUG mode...")
    print("Debug logs will be displayed in the console")

    # Run the application with uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug"
    )
