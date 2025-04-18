"""
Test Runner Script for the Medical Research Synthesizer.

This script provides a convenient command-line interface for running tests using pytest.
It supports various options for running specific test types (unit, integration, performance,
security), generating reports (coverage, JUnit XML, HTML), and filtering tests.

Usage:
    python -m asf.medical.scripts.run_tests [options]

Options:
    --unit           Run only unit tests
    --integration    Run only integration tests
    --performance    Run only performance tests
    --security       Run only security tests
    --coverage       Generate coverage report
    --verbose, -v    Verbose output
    --xvs            Show extra verbose output
    --junit-xml      Generate JUnit XML report
    --html-report    Generate HTML report
    --markers        Show available markers
    --collect-only   Only collect tests, don't run them
    --filter TEXT    Filter tests by name
    --mark TEXT      Run tests with specific marker
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

def main():
    """Run tests using pytest with the specified options.

    This function parses command-line arguments and constructs a pytest command
    with the appropriate options. It supports:
    - Running specific test types (unit, integration, performance, security)
    - Generating reports (coverage, JUnit XML, HTML)
    - Controlling verbosity
    - Filtering tests by name or marker
    - Listing available markers
    - Collecting tests without running them

    The function executes the pytest command using os.system and propagates
    the exit code to the caller.

    Returns:
        None, but exits the process with pytest's exit code
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
