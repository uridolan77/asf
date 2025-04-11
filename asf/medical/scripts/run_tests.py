"""
Script to run tests for the Medical Research Synthesizer.

This script runs pytest on the tests directory.
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

def main():
    """Run tests.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
    parser = argparse.ArgumentParser(description="Run tests for the Medical Research Synthesizer")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration", action="store_true", help="Run only integration tests")
    parser.add_argument("--performance", action="store_true", help="Run only performance tests")
    parser.add_argument("--security", action="store_true", help="Run only security tests")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--xvs", action="store_true", help="Show extra verbose output")
    parser.add_argument("--junit-xml", action="store_true", help="Generate JUnit XML report")
    parser.add_argument("--html-report", action="store_true", help="Generate HTML report")
    parser.add_argument("--markers", action="store_true", help="Show available markers")
    parser.add_argument("--collect-only", action="store_true", help="Only collect tests, don't run them")
    parser.add_argument("--filter", type=str, help="Filter tests by name")
    parser.add_argument("--mark", type=str, help="Run tests with specific marker")
    args = parser.parse_args()

    cmd = ["pytest"]

    if args.verbose:
        cmd.append("-v")
    elif args.xvs:
        cmd.append("-vv")

    if args.coverage:
        cmd.append("--cov=asf.medical")
        cmd.append("--cov-report=term")
        cmd.append("--cov-report=html")

    if args.junit_xml:
        cmd.append("--junitxml=test-results.xml")

    if args.html_report:
        cmd.append("--html=test-report.html")
        cmd.append("--self-contained-html")

    if args.markers:
        cmd.append("--markers")
        os.system(" ".join(cmd))
        return

    if args.collect_only:
        cmd.append("--collect-only")

    if args.filter:
        cmd.append(f"-k {args.filter}")

    if args.mark:
        cmd.append(f"-m {args.mark}")

    if args.unit:
        cmd.append("tests/unit")
    elif args.integration:
        cmd.append("tests/integration")
    elif args.performance:
        cmd.append("tests/performance")
    elif args.security:
        cmd.append("tests/security")
    else:
        cmd.append("tests")

    print(f"Running command: {' '.join(cmd)}")
    result = os.system(" ".join(cmd))

    sys.exit(result >> 8)

if __name__ == "__main__":
    main()
