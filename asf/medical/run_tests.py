"""
Script to run tests for the Medical Research Synthesizer.

This script runs the tests for the Medical Research Synthesizer.
"""

import os
import sys
import subprocess
import argparse

def run_tests(test_type=None, verbose=False, coverage=False):
    """
    Run tests for the Medical Research Synthesizer.
    
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
        cmd.append("-m unit")
    elif test_type == "integration":
        cmd.append("-m integration")
    elif test_type == "performance":
        cmd.append("-m performance")
    
    # Add verbose flag
    if verbose:
        cmd.append("-v")
    
    # Add coverage flags
    if coverage:
        cmd.append("--cov=asf.medical")
        cmd.append("--cov-report=term")
        cmd.append("--cov-report=html")
    
    # Run tests
    cmd_str = " ".join(cmd)
    print(f"Running command: {cmd_str}")
    subprocess.run(cmd_str, shell=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tests for the Medical Research Synthesizer")
    parser.add_argument("--type", choices=["unit", "integration", "performance"], help="Type of tests to run")
    parser.add_argument("--verbose", action="store_true", help="Show verbose output")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    
    args = parser.parse_args()
    
    run_tests(args.type, args.verbose, args.coverage)
