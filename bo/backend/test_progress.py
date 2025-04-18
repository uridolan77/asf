"""
Test script for progress module.
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from asf.medical.llm_gateway.progress import (
    ProgressTracker,
    ProgressState,
    ProgressStep,
    ProgressError,
    ProgressRegistry,
    get_progress_registry,
    track_progress,
    track_llm_progress,
    ProgressDetails,
    OperationType,
    ProgressUpdate
)

def main():
    """Main function."""
    print("Testing progress module imports...")
    print(f"ProgressTracker: {ProgressTracker}")
    print(f"ProgressState: {ProgressState}")
    print(f"ProgressStep: {ProgressStep}")
    print(f"ProgressError: {ProgressError}")
    print(f"ProgressRegistry: {ProgressRegistry}")
    print(f"get_progress_registry: {get_progress_registry}")
    print(f"track_progress: {track_progress}")
    print(f"track_llm_progress: {track_llm_progress}")
    print(f"ProgressDetails: {ProgressDetails}")
    print(f"OperationType: {OperationType}")
    print(f"ProgressUpdate: {ProgressUpdate}")
    print("Progress module imports successful!")

if __name__ == "__main__":
    main()
