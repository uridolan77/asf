"""
Script to check for vulnerabilities in dependencies.

This script uses the safety package to check for known vulnerabilities in the
dependencies listed in requirements.txt.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

def main():
    """Check for vulnerabilities in dependencies."""
    parser = argparse.ArgumentParser(description="Check for vulnerabilities in dependencies")
    parser.add_argument("--requirements", "-r", default="requirements.txt", help="Path to requirements.txt file")
    parser.add_argument("--output", "-o", default="safety-report.txt", help="Path to output file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--json", "-j", action="store_true", help="Output in JSON format")
    args = parser.parse_args()
    
    # Get the path to the requirements.txt file
    requirements_path = Path(args.requirements)
    if not requirements_path.is_absolute():
        requirements_path = Path(__file__).parent.parent / args.requirements
    
    # Check if the requirements.txt file exists
    if not requirements_path.exists():
        print(f"Error: Requirements file not found: {requirements_path}")
        sys.exit(1)
    
    # Get the path to the output file
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).parent.parent / args.output
    
    # Build the command
    cmd = ["safety", "check", "-r", str(requirements_path)]
    
    # Add output format
    if args.json:
        cmd.extend(["--json"])
    
    # Add output file
    cmd.extend(["--output", str(output_path)])
    
    # Run the command
    print(f"Running command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Print the output
        if args.verbose:
            print(result.stdout)
        
        # Check if there are vulnerabilities
        if result.returncode != 0:
            print(f"Vulnerabilities found! See {output_path} for details.")
            return 1
        else:
            print("No vulnerabilities found.")
            return 0
    except Exception as e:
        print(f"Error running safety check: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
