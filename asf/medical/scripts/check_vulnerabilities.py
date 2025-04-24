"""
Script to check for vulnerabilities in dependencies.
This script uses the safety package to check for known vulnerabilities in the
dependencies listed in requirements.txt.
"""
import sys
import subprocess
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
def main():
    """Check for vulnerabilities in dependencies.
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
    """
    parser = argparse.ArgumentParser(description="Check for vulnerabilities in dependencies")
    parser.add_argument("--requirements", "-r", default="requirements.txt", help="Path to requirements.txt file")
    parser.add_argument("--output", "-o", default="safety-report.txt", help="Path to output file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--json", "-j", action="store_true", help="Output in JSON format")
    args = parser.parse_args()
    requirements_path = Path(args.requirements)
    if not requirements_path.is_absolute():
        requirements_path = Path(__file__).parent.parent / args.requirements
    if not requirements_path.exists():
        print(f"Error: Requirements file not found: {requirements_path}")
        sys.exit(1)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).parent.parent / args.output
    cmd = ["safety", "check", "-r", str(requirements_path)]
    if args.json:
        cmd.extend(["--json"])
    cmd.extend(["--output", str(output_path)])
    print(f"Running command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if args.verbose:
            print(result.stdout)
        if result.returncode != 0:
            print(f"Vulnerabilities found! See {output_path} for details.")
            return 1
        else:
            print("No vulnerabilities found.")
            return 0
    except Exception as e:
    logger.error(f\"Error running safety check: {str(e)}\")
    raise DatabaseError(f\"Error running safety check: {str(e)}\")
        return 1
if __name__ == "__main__":
    sys.exit(main())