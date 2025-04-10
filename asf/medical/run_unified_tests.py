"""
Script to run tests for the unified Medical Research Synthesizer API.

This script runs the tests for the unified API.
"""

import os
import sys
import subprocess
import argparse

def run_tests(test_type=None, verbose=False, coverage=False):
    """
    Run tests for the unified Medical Research Synthesizer API.
    
    Args:
        test_type: Type of tests to run (unit, integration, performance, or None for all)
        verbose: Whether to show verbose output
        coverage: Whether to generate coverage report
    """
    # Change to the medical directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Build command
    cmd = ["pytest"]
    
    # Add test type
    if test_type == "unit":
        cmd.append("tests/unit/test_auth_unified.py")
    elif test_type == "integration":
        cmd.append("tests/integration/test_api_unified.py")
    elif test_type == "auth":
        cmd.append("tests/unit/test_auth_unified.py")
        cmd.append("tests/integration/test_api_unified.py::TestAuthAPI")
    
    # Add verbose flag
    if verbose:
        cmd.append("-v")
    
    # Add coverage flags
    if coverage:
        cmd.append("--cov=asf.medical.api")
        cmd.append("--cov-report=term")
        cmd.append("--cov-report=html")
    
    # Run tests
    cmd_str = " ".join(cmd)
    print(f"Running command: {cmd_str}")
    
    try:
        # Use subprocess.run with check=True to raise an exception if the command fails
        subprocess.run(cmd, check=True)
        print("Tests completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Tests failed with exit code {e.returncode}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tests for the unified Medical Research Synthesizer API")
    parser.add_argument("--type", choices=["unit", "integration", "performance", "auth"], help="Type of tests to run")
    parser.add_argument("--verbose", action="store_true", help="Show verbose output")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    
    args = parser.parse_args()
    
    run_tests(args.type, args.verbose, args.coverage)
