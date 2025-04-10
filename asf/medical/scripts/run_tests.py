"""
Script to run tests for the Medical Research Synthesizer.

This script runs pytest on the tests directory.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

def main():
    """Run tests."""
    parser = argparse.ArgumentParser(description="Run tests for the Medical Research Synthesizer")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration", action="store_true", help="Run only integration tests")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    # Set up test command
    cmd = ["pytest"]
    
    # Add verbosity
    if args.verbose:
        cmd.append("-v")
    
    # Add coverage
    if args.coverage:
        cmd.append("--cov=asf.medical")
        cmd.append("--cov-report=term")
        cmd.append("--cov-report=html")
    
    # Add test selection
    if args.unit:
        cmd.append("tests/unit")
    elif args.integration:
        cmd.append("tests/integration")
    else:
        cmd.append("tests")
    
    # Run tests
    print(f"Running command: {' '.join(cmd)}")
    os.system(" ".join(cmd))

if __name__ == "__main__":
    main()
